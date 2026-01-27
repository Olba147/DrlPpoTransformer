from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn


@dataclass
class RolloutBatch:
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    last_value: torch.Tensor
    turnovers: torch.Tensor


class PPOTrainer:
    def __init__(
        self,
        env,
        policy: nn.Module,
        value_fn: nn.Module,
        jepa_encoder: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: Optional[torch.device] = None,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        gamma: float = 0.99,
        lam: float = 0.95,
        target_kl: Optional[float] = None,
    ) -> None:
        self.env = env
        self.policy = policy
        self.value_fn = value_fn
        self.jepa_encoder = jepa_encoder
        self.optimizer = optimizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.target_kl = target_kl

        self.policy.to(self.device)
        self.value_fn.to(self.device)
        self.jepa_encoder.to(self.device)

    def _to_tensor(self, value: torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        return torch.tensor(value, device=self.device, dtype=torch.float32)

    def _unpack_env_output(
        self, output
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(output, dict):
            x_context = output.get("x_context")
            t_context = output.get("t_context")
            reward = output.get("reward")
            done = output.get("done")
            info = output.get("info", {})
        else:
            info = {}
            if len(output) == 4:
                x_context, t_context, reward, done = output
            elif len(output) == 5:
                x_context, t_context, reward, done, info = output
            else:
                raise ValueError("env output must be (x_context, t_context, reward, done[, info])")
        turnover = info.get("turnover", 0.0) if isinstance(info, dict) else 0.0
        return (
            self._to_tensor(x_context),
            self._to_tensor(t_context),
            self._to_tensor(reward),
            self._to_tensor(done),
            self._to_tensor(turnover),
        )

    def _extract_state(
        self, x_context: torch.Tensor, t_context: torch.Tensor, w_prev, wealth_feats
    ) -> torch.Tensor:
        with torch.no_grad():
            z_t = self.jepa_encoder(x_context, t_context)
        w_prev = self._to_tensor(w_prev)
        wealth_feats = self._to_tensor(wealth_feats)
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)
        if w_prev.dim() == 1:
            w_prev = w_prev.unsqueeze(0)
        if wealth_feats.dim() == 1:
            wealth_feats = wealth_feats.unsqueeze(0)
        return torch.cat([z_t, w_prev, wealth_feats], dim=-1)

    def collect_rollout(self, env, policy, value_fn, rollout_len: int) -> RolloutBatch:
        states: List[torch.Tensor] = []
        actions: List[torch.Tensor] = []
        log_probs: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        rewards: List[torch.Tensor] = []
        dones: List[torch.Tensor] = []
        turnovers: List[torch.Tensor] = []

        reset_out = env.reset()
        if isinstance(reset_out, dict):
            x_context = reset_out.get("x_context")
            t_context = reset_out.get("t_context")
            w_prev = reset_out.get("w_prev", 0.0)
            wealth_feats = reset_out.get("wealth_feats", 0.0)
        else:
            x_context, t_context, *extras = reset_out
            w_prev = extras[0] if len(extras) > 0 else 0.0
            wealth_feats = extras[1] if len(extras) > 1 else 0.0

        x_context = self._to_tensor(x_context)
        t_context = self._to_tensor(t_context)
        state = self._extract_state(x_context, t_context, w_prev, wealth_feats)

        for _ in range(rollout_len):
            with torch.no_grad():
                dist = policy(state)
                if isinstance(dist, tuple):
                    action, log_prob = dist[:2]
                    entropy = dist[2] if len(dist) > 2 else torch.tensor(0.0, device=self.device)
                else:
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    entropy = dist.entropy()
                value = value_fn(state).squeeze(-1)

            step_out = env.step(action.detach().cpu().numpy())
            x_context, t_context, reward, done, turnover = self._unpack_env_output(step_out)
            if isinstance(step_out, dict):
                w_prev = step_out.get("w_prev", w_prev)
                wealth_feats = step_out.get("wealth_feats", wealth_feats)
            elif len(step_out) >= 6:
                w_prev = step_out[4]
                wealth_feats = step_out[5]

            states.append(state.squeeze(0))
            actions.append(action.squeeze(0))
            log_probs.append(log_prob.squeeze(0))
            values.append(value.squeeze(0))
            rewards.append(reward.squeeze(0))
            dones.append(done.squeeze(0))
            turnovers.append(turnover.squeeze(0))

            if done.item():
                reset_out = env.reset()
                if isinstance(reset_out, dict):
                    x_context = reset_out.get("x_context")
                    t_context = reset_out.get("t_context")
                    w_prev = reset_out.get("w_prev", 0.0)
                    wealth_feats = reset_out.get("wealth_feats", 0.0)
                else:
                    x_context, t_context, *extras = reset_out
                    w_prev = extras[0] if len(extras) > 0 else 0.0
                    wealth_feats = extras[1] if len(extras) > 1 else 0.0
                x_context = self._to_tensor(x_context)
                t_context = self._to_tensor(t_context)
            state = self._extract_state(x_context, t_context, w_prev, wealth_feats)

        with torch.no_grad():
            last_value = value_fn(state).squeeze(-1)

        return RolloutBatch(
            states=torch.stack(states),
            actions=torch.stack(actions),
            log_probs=torch.stack(log_probs),
            values=torch.stack(values),
            rewards=torch.stack(rewards),
            dones=torch.stack(dones),
            last_value=last_value,
            turnovers=torch.stack(turnovers),
        )

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
        lam: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(1, device=rewards.device)
        if values.shape[0] == rewards.shape[0] + 1:
            values_ext = values
            values_current = values[:-1]
        else:
            values_ext = torch.cat([values, values.new_tensor([0.0])])
            values_current = values
        for t in reversed(range(rewards.shape[0])):
            next_value = values_ext[t + 1]
            not_done = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * not_done - values_current[t]
            gae = delta + gamma * lam * not_done * gae
            advantages[t] = gae
        returns = advantages + values_current
        return advantages, returns

    def update_policy(
        self,
        rollout: RolloutBatch,
        epochs: int,
        minibatch_size: int,
    ) -> Dict[str, float]:
        rewards = rollout.rewards
        values = torch.cat([rollout.values, rollout.last_value.unsqueeze(0)])
        dones = rollout.dones
        advantages, returns = self.compute_gae(rewards, values, dones, self.gamma, self.lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = rewards.shape[0]
        indices = torch.arange(batch_size, device=self.device)

        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []
        kls: List[float] = []
        clip_fracs: List[float] = []

        for _ in range(epochs):
            perm = indices[torch.randperm(batch_size)]
            for start in range(0, batch_size, minibatch_size):
                mb_idx = perm[start : start + minibatch_size]

                mb_states = rollout.states[mb_idx]
                mb_actions = rollout.actions[mb_idx]
                mb_log_probs = rollout.log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                dist = self.policy(mb_states)
                if isinstance(dist, tuple):
                    new_actions, new_log_probs = dist[:2]
                    entropy = dist[2] if len(dist) > 2 else torch.tensor(0.0, device=self.device)
                    if new_actions.shape != mb_actions.shape:
                        new_actions = mb_actions
                else:
                    new_log_probs = dist.log_prob(mb_actions)
                    entropy = dist.entropy()

                new_values = self.value_fn(mb_states).squeeze(-1)

                ratio = (new_log_probs - mb_log_probs).exp()
                unclipped = ratio * mb_advantages
                clipped = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * mb_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = 0.5 * (mb_returns - new_values).pow(2).mean()
                entropy_loss = entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_fn.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                approx_kl = (mb_log_probs - new_log_probs).mean()
                clip_frac = ((ratio - 1.0).abs() > self.clip_coef).float().mean()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_loss.item())
                kls.append(approx_kl.item())
                clip_fracs.append(clip_frac.item())

                if self.target_kl is not None and approx_kl > self.target_kl:
                    break

        metrics = {
            "policy_loss": float(torch.tensor(policy_losses).mean()),
            "value_loss": float(torch.tensor(value_losses).mean()),
            "entropy": float(torch.tensor(entropies).mean()),
            "kl": float(torch.tensor(kls).mean()),
            "clip_fraction": float(torch.tensor(clip_fracs).mean()),
            "mean_reward": float(rewards.mean().item()),
            "mean_turnover": float(rollout.turnovers.mean().item()),
        }
        return metrics
