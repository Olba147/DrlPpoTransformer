from __future__ import annotations

from typing import Dict, Optional

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


class JEPAAuxFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        jepa_model: JEPA,
        embedding_dim: int,
        patch_len: int,
        patch_stride: int,
        use_obs_targets: bool = False,
        target_ratio: float | None = 0.5,
        target_len: int | None = None,
    ) -> None:
        w_prev_dim = observation_space["w_prev"].shape[0] if "w_prev" in observation_space.spaces else 0
        wealth_dim = observation_space["wealth_feats"].shape[0] if "wealth_feats" in observation_space.spaces else 0
        features_dim = embedding_dim + w_prev_dim + wealth_dim
        super().__init__(observation_space, features_dim=features_dim)

        self.jepa_model = jepa_model
        self.embedding_dim = embedding_dim
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.use_obs_targets = use_obs_targets
        self.target_ratio = target_ratio
        self.target_len = target_len

        # Ensure target encoder/projection are not updated by SGD
        for param in self.jepa_model.target_enc.parameters():
            param.requires_grad = False
        for param in self.jepa_model.proj_target.parameters():
            param.requires_grad = False

        self.last_jepa_loss: Optional[th.Tensor] = None
        self.last_pred_std: Optional[float] = None
        self.last_tgt_std: Optional[float] = None

    def _ensure_batched(self, x: th.Tensor) -> th.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(0)
        return x

    def _split_context(self, x: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        seq_len = x.shape[1]
        if self.target_len is not None:
            tgt_len = min(max(1, self.target_len), seq_len - 1)
            split_idx = seq_len - tgt_len
        else:
            ratio = self.target_ratio if self.target_ratio is not None else 0.5
            split_idx = max(1, int(seq_len * ratio))
            split_idx = min(seq_len - 1, split_idx)
        return x[:, :split_idx], x[:, split_idx:]

    def _patch(self, x: th.Tensor) -> th.Tensor:
        return make_patches(x, self.patch_len, self.patch_stride)

    def compute_jepa_aux(
        self,
        observations: Dict[str, th.Tensor],
        actions: Optional[th.Tensor] = None,
    ) -> tuple[Optional[th.Tensor], Optional[float], Optional[float]]:
        x_context = self._ensure_batched(observations["x_context"])
        t_context = self._ensure_batched(observations["t_context"])

        # Build JEPA targets
        if self.use_obs_targets and "x_target" in observations and "t_target" in observations:
            x_target = self._ensure_batched(observations["x_target"])
            t_target = self._ensure_batched(observations["t_target"])
        else:
            x_ctx_split, x_target = self._split_context(x_context)
            t_ctx_split, t_target = self._split_context(t_context)
            x_context = x_ctx_split
            t_context = t_ctx_split

        if x_context.shape[1] < self.patch_len or x_target.shape[1] < self.patch_len:
            self.last_jepa_loss = None
            self.last_pred_std = None
            self.last_tgt_std = None
            return None, None, None

        x_ctx_p = self._patch(x_context)
        t_ctx_p = self._patch(t_context)
        x_tgt_p = self._patch(x_target)
        t_tgt_p = self._patch(t_target)

        if actions is not None and actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        p_c, z_t = self.jepa_model(x_ctx_p, t_ctx_p, x_tgt_p, t_tgt_p, action=actions)
        loss = F.mse_loss(p_c, z_t)
        pred_std = float(p_c.std(dim=-1).mean().detach().cpu().item())
        tgt_std = float(z_t.std(dim=-1).mean().detach().cpu().item())
        self.last_jepa_loss = loss
        self.last_pred_std = pred_std
        self.last_tgt_std = tgt_std
        return loss, pred_std, tgt_std

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        x_context = self._ensure_batched(observations["x_context"])
        t_context = self._ensure_batched(observations["t_context"])

        # Features for PPO: use full context embedding
        x_full = self._ensure_batched(observations["x_context"])
        t_full = self._ensure_batched(observations["t_context"])
        x_full_p = self._patch(x_full)
        t_full_p = self._patch(t_full)

        z_t = self.jepa_model.context_enc(x_full_p, t_full_p)
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)

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

        return th.cat([z_t, w_prev, wealth_feats], dim=-1)


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
        update_jepa: bool = True,
        jepa_coef: float = 1.0,
    ):
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

        self.update_jepa = update_jepa
        self.jepa_coef = jepa_coef

    def train(self) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        jepa_losses = []
        jepa_pred_stds = []
        jepa_tgt_stds = []
        weighted_entropy_losses = []
        weighted_value_losses = []
        weighted_jepa_losses = []
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

                jepa_loss = None
                if self.update_jepa and hasattr(self.policy, "features_extractor"):
                    fx = self.policy.features_extractor
                    if hasattr(fx, "compute_jepa_aux"):
                        jepa_loss, pred_std, tgt_std = fx.compute_jepa_aux(
                            rollout_data.observations,
                            actions=rollout_data.actions,
                        )
                        if jepa_loss is not None:
                            loss = loss + self.jepa_coef * jepa_loss
                            jepa_losses.append(jepa_loss.item())
                            weighted_jepa_losses.append((self.jepa_coef * jepa_loss).item())
                        if pred_std is not None:
                            jepa_pred_stds.append(pred_std)
                        if tgt_std is not None:
                            jepa_tgt_stds.append(tgt_std)

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
        if self.update_jepa and hasattr(self.policy, "features_extractor"):
            fx = self.policy.features_extractor
            if hasattr(fx, "jepa_model"):
                fx.jepa_model.ema_update(self._n_updates)

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

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
        if jepa_losses:
            self.logger.record("train/jepa_loss", np.mean(jepa_losses))
        if weighted_jepa_losses:
            self.logger.record("train/weighted_jepa_loss", np.mean(weighted_jepa_losses))
        if jepa_pred_stds:
            self.logger.record("train/jepa_pred_std", np.mean(jepa_pred_stds))
        if jepa_tgt_stds:
            self.logger.record("train/jepa_tgt_std", np.mean(jepa_tgt_stds))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
