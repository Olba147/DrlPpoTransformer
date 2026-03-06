from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from torch.nn import functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import explained_variance

from Training.callbacks import make_patches
from models.jepa.jepa import JEPA


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.query = nn.Parameter(th.empty(1, 1, d_model))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: th.Tensor) -> th.Tensor:
        batch_size = tokens.size(0)
        query = self.query.expand(batch_size, -1, -1)
        pooled, _ = self.attn(query=query, key=tokens, value=tokens, need_weights=False)
        return self.norm(pooled.squeeze(1))


class JEPAAuxFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        jepa_model: JEPA,
        embedding_dim: int,
        patch_len: int,
        patch_stride: int,
        attn_pool_heads: int = 4,
    ) -> None:
        w_prev_dim = observation_space["w_prev"].shape[0] if "w_prev" in observation_space.spaces else 0
        wealth_dim = observation_space["wealth_feats"].shape[0] if "wealth_feats" in observation_space.spaces else 0
        asset_dim = 0
        if "asset_id" in observation_space.spaces:
            asset_space = observation_space["asset_id"]
            if isinstance(asset_space, spaces.Discrete):
                asset_dim = int(asset_space.n)
            else:
                asset_dim = int(np.prod(asset_space.shape))
        features_dim = embedding_dim + w_prev_dim + wealth_dim + asset_dim
        super().__init__(observation_space, features_dim=features_dim)

        self.jepa_model = jepa_model
        self.embedding_dim = embedding_dim
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.asset_dim = asset_dim
        self.attn_pool = AttentionPooling(embedding_dim, num_heads=attn_pool_heads)

    def _ensure_batched(self, x: th.Tensor) -> th.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(0)
        return x

    def _get_asset_id(self, observations: Dict[str, th.Tensor]) -> Optional[th.Tensor]:
        asset_id = observations.get("asset_id")
        if asset_id is None:
            return None
        # SB3 preprocesses Discrete obs to one-hot (float). Convert back to indices.
        if asset_id.dim() >= 2 and asset_id.shape[-1] > 1:
            asset_id = th.argmax(asset_id, dim=-1)
        if asset_id.dim() == 0:
            asset_id = asset_id.unsqueeze(0)
        if asset_id.dim() == 2 and asset_id.shape[1] == 1:
            asset_id = asset_id.squeeze(1)
        return asset_id.long()

    def _patch(self, x: th.Tensor) -> th.Tensor:
        return make_patches(x, self.patch_len, self.patch_stride)

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        x_context = self._ensure_batched(observations["x_context"])
        t_context = self._ensure_batched(observations["t_context"])
        asset_id = self._get_asset_id(observations)

        # Features for PPO: use full context embedding
        x_full = self._ensure_batched(observations["x_context"])
        t_full = self._ensure_batched(observations["t_context"])
        x_full_p = self._patch(x_full)
        t_full_p = self._patch(t_full)

        tokens = self.jepa_model.context_enc(
            x_full_p,
            t_full_p,
            asset_id=asset_id,
            return_tokens=True,
        )
        z_t = self.attn_pool(tokens)

        w_prev = observations.get("w_prev", None)
        wealth_feats = observations.get("wealth_feats", None)
        if w_prev is None:
            w_prev = th.zeros((z_t.size(0), 0), device=z_t.device)
        if wealth_feats is None:
            wealth_feats = th.zeros((z_t.size(0), 0), device=z_t.device)
        if w_prev.dim() == 1:
            w_prev = w_prev.unsqueeze(0)
        if wealth_feats.dim() == 1:
            wealth_feats = wealth_feats.unsqueeze(0)

        asset_raw = observations.get("asset_id", None)
        if asset_raw is None or self.asset_dim == 0:
            asset_feat = th.zeros((z_t.size(0), 0), device=z_t.device)
        else:
            asset_feat = asset_raw
            if asset_feat.dim() == 1:
                asset_feat = asset_feat.view(1, self.asset_dim)
            elif asset_feat.dim() == 2:
                pass
            elif asset_feat.dim() == 3 and asset_feat.shape[1] == 1 and asset_feat.shape[2] == self.asset_dim:
                asset_feat = asset_feat.squeeze(1)
            else:
                raise ValueError(f"Unexpected asset_id shape {tuple(asset_feat.shape)} for asset_dim {self.asset_dim}")

        return th.cat([z_t, w_prev, wealth_feats, asset_feat], dim=-1)


class PPOWithJEPA(PPO):
    def __init__(
        self,
        policy: str | type[ActorCriticPolicy],
        env,
        learning_rate: float | Schedule = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float | Schedule = 0.2,
        clip_range_vf: None | float | Schedule = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class=None,
        rollout_buffer_kwargs=None,
        target_kl: float | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
        update_jepa: bool = False,
        optimizer_name: str = "adam",
        optimizer_kwargs: dict[str, Any] | None = None,
        policy_learning_rate: float | None = None,
    ):
        if update_jepa:
            raise ValueError(
                "update_jepa=true is not supported in this branch. "
                "JEPA auxiliary PPO updates are disabled."
            )

        # Set custom optimizer fields before SB3 init, because SB3 may call
        # _setup_model() inside super().__init__ when _init_setup_model=True.
        self.optimizer_name = str(optimizer_name).lower()
        self.optimizer_kwargs_custom = dict(optimizer_kwargs or {})
        self.policy_learning_rate = None if policy_learning_rate is None else float(policy_learning_rate)

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

        self.update_jepa = False

    def _setup_model(self) -> None:
        super()._setup_model()
        self.configure_optimizer()

    def _resolve_optimizer_class(self):
        optimizers = {
            "adam": th.optim.Adam,
            "adamw": th.optim.AdamW,
            "sgd": th.optim.SGD,
            "rmsprop": th.optim.RMSprop,
        }
        if self.optimizer_name not in optimizers:
            raise ValueError(f"Unsupported optimizer_name: {self.optimizer_name}")
        return optimizers[self.optimizer_name]

    def configure_optimizer(self) -> None:
        optimizer_class = self._resolve_optimizer_class()
        optimizer_kwargs = dict(self.optimizer_kwargs_custom)
        optimizer_kwargs.pop("lr", None)
        if self.optimizer_name == "adam" and "eps" not in optimizer_kwargs:
            # Match SB3 default ActorCriticPolicy Adam epsilon.
            optimizer_kwargs["eps"] = 1e-5

        scheduled_lr = float(self.lr_schedule(1.0))
        policy_lr = self.policy_learning_rate if self.policy_learning_rate is not None else scheduled_lr
        fx = getattr(self.policy, "features_extractor", None)
        jepa_model = getattr(fx, "jepa_model", None)
        if jepa_model is not None:
            trainable_jepa = [p for p in jepa_model.parameters() if p.requires_grad]
            if trainable_jepa:
                raise ValueError(
                    "Found trainable JEPA parameters in PPO optimizer setup. "
                    "Freeze JEPA (requires_grad=False) before PPO training."
                )

        policy_params = [p for p in self.policy.parameters() if p.requires_grad]
        if not policy_params:
            raise ValueError("No trainable parameters found for optimizer configuration.")

        self.policy.optimizer = optimizer_class(
            [{"params": policy_params, "lr": policy_lr, "group_name": "policy"}],
            **optimizer_kwargs,
        )

    def _update_group_learning_rates(self) -> None:
        scheduled_lr = float(self.lr_schedule(self._current_progress_remaining))
        policy_lr = self.policy_learning_rate if self.policy_learning_rate is not None else scheduled_lr

        for group in self.policy.optimizer.param_groups:
            group["lr"] = policy_lr

        self.logger.record("train/lr_policy", float(policy_lr))
        self.logger.record("train/learning_rate", float(policy_lr))

    def train(self) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Keep policy optimizer groups on their configured rates.
        self._update_group_learning_rates()
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        weighted_entropy_losses = []
        weighted_value_losses = []
        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )

                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                weighted_value_losses.append((self.vf_coef * value_loss).item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())
                weighted_entropy_losses.append((self.ent_coef * entropy_loss).item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        # EMA update for JEPA target encoder
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        reward_std = float(np.std(self.rollout_buffer.rewards)) if self.rollout_buffer.rewards.size else 0.0

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        if weighted_entropy_losses:
            self.logger.record("train/weighted_entropy_loss", np.mean(weighted_entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        if weighted_value_losses:
            self.logger.record("train/weighted_value_loss", np.mean(weighted_value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/reward_std", reward_std)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
