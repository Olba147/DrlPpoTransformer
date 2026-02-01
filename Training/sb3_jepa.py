from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from Training.callbacks import make_patches


class JEPAFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        jepa_encoder: nn.Module,
        embedding_dim: int,
        patch_len: int,
        patch_stride: int,
    ) -> None:
        w_prev_dim = observation_space["w_prev"].shape[0] if "w_prev" in observation_space.spaces else 0
        wealth_dim = observation_space["wealth_feats"].shape[0] if "wealth_feats" in observation_space.spaces else 0
        features_dim = embedding_dim + w_prev_dim + wealth_dim
        super().__init__(observation_space, features_dim=features_dim)

        self.jepa_encoder = jepa_encoder
        self.embedding_dim = embedding_dim
        self.patch_len = patch_len
        self.patch_stride = patch_stride

        for param in self.jepa_encoder.parameters():
            param.requires_grad = False
        self.jepa_encoder.eval()

    def _ensure_patched(self, x_context: torch.Tensor, t_context: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x_context.dim() == 2:
            x_context = x_context.unsqueeze(0)
        if t_context.dim() == 2:
            t_context = t_context.unsqueeze(0)
        x_patched = make_patches(x_context, self.patch_len, self.patch_stride)
        t_patched = make_patches(t_context, self.patch_len, self.patch_stride)
        return {"x_context": x_patched, "t_context": t_patched}

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        x_context = observations["x_context"]
        t_context = observations["t_context"]
        patched = self._ensure_patched(x_context, t_context)

        with torch.no_grad():
            z_t = self.jepa_encoder(patched["x_context"], patched["t_context"])

        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)

        w_prev = observations.get("w_prev", None)
        wealth_feats = observations.get("wealth_feats", None)

        if w_prev is None:
            w_prev = torch.zeros((z_t.size(0), 0), device=z_t.device)
        if wealth_feats is None:
            wealth_feats = torch.zeros((z_t.size(0), 0), device=z_t.device)

        if w_prev.dim() == 1:
            w_prev = w_prev.unsqueeze(0)
        if wealth_feats.dim() == 1:
            wealth_feats = wealth_feats.unsqueeze(0)

        return torch.cat([z_t, w_prev, wealth_feats], dim=-1)
