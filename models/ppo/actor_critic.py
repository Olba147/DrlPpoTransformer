from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.distributions import Normal


def _mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.GELU())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


@dataclass
class ActionDistribution:
    mean: torch.Tensor
    log_std: torch.Tensor
    std: torch.Tensor


class TanhGaussianPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        action_dim: int = 1,
        log_std_init: float = -0.5,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        state_dependent_std: bool = False,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.state_dependent_std = state_dependent_std

        self.trunk = _mlp(input_dim, hidden_dims, hidden_dims[-1])
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        if state_dependent_std:
            self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        else:
            self.log_std_param = nn.Parameter(torch.ones(action_dim) * log_std_init)

    def _distribution(self, state: torch.Tensor) -> ActionDistribution:
        features = self.trunk(state)
        mean = self.mean_head(features)
        if self.state_dependent_std:
            log_std = self.log_std_head(features)
        else:
            log_std = self.log_std_param.expand_as(mean)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return ActionDistribution(mean=mean, log_std=log_std, std=std)

    def _tanh_squash(self, raw_action: torch.Tensor) -> torch.Tensor:
        return torch.tanh(raw_action)

    def _log_prob(self, dist: ActionDistribution, raw_action: torch.Tensor) -> torch.Tensor:
        normal = Normal(dist.mean, dist.std)
        log_prob = normal.log_prob(raw_action).sum(dim=-1)
        squashed = self._tanh_squash(raw_action)
        correction = torch.log(1 - squashed.pow(2) + 1e-6).sum(dim=-1)
        return log_prob - correction

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._distribution(state)
        raw_action = Normal(dist.mean, dist.std).rsample()
        action = self._tanh_squash(raw_action)
        log_prob = self._log_prob(dist, raw_action)
        entropy = Normal(dist.mean, dist.std).entropy().sum(dim=-1)
        return action, log_prob, entropy

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self._distribution(state)
        raw_action = torch.atanh(action.clamp(-0.999999, 0.999999))
        log_prob = self._log_prob(dist, raw_action)
        entropy = Normal(dist.mean, dist.std).entropy().sum(dim=-1)
        return log_prob, entropy

    def mean_action(self, state: torch.Tensor) -> torch.Tensor:
        dist = self._distribution(state)
        return self._tanh_squash(dist.mean)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sample(state)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        self.net = _mlp(input_dim, hidden_dims, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
