from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable
from collections import deque
import os
import csv
import math
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import time
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import numpy as np

# ----------------------------
# Minimal callback primitives
# ----------------------------

class Callback:
    "Base class. Override the hooks you need."
    order: int = 0       # lower runs earlier

    def set_learner(self, learn):
        self.learn = learn

    # fit-level
    def before_fit(self): pass
    def after_fit(self): pass

    # epoch-level
    def before_epoch(self): pass
    def after_epoch(self): pass

    # mode switches
    def before_train(self): pass
    def after_train(self): pass
    def before_validate(self): pass
    def after_validate(self): pass

    # batch-level
    def before_batch(self, batch: Dict[str, torch.Tensor]): pass
    def after_batch(self, batch: Dict[str, torch.Tensor]): pass

    # per-step
    def before_forward(self, batch: Dict[str, torch.Tensor]): pass
    def after_pred(self, pred: Any, batch: Dict[str, torch.Tensor]): pass
    def after_loss(self, loss: torch.Tensor, batch: Dict[str, torch.Tensor]): pass
    def after_backward(self): pass
    def after_step(self): pass


class CallbackList:
    def __init__(self, cbs: List[Callback]):
        self.cbs = sorted(cbs, key=lambda cb: cb.order)

    def set_learner(self, learn):
        for cb in self.cbs:
            cb.set_learner(learn)

    def _call(self, name: str, *args, **kwargs):
        for cb in self.cbs:
            getattr(cb, name)(*args, **kwargs)


# ----------------------------
# Utility
# ----------------------------

def make_patches(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """
    x: [B, L, C] or [L, C]
    returns: [B, N, patch_len*C]
    """
    if x.ndim == 2:
        x = x.unsqueeze(0)
    B, L, C = x.shape
    if patch_len > L:
        raise ValueError("patch_len must be <= sequence length.")
    n = 1 + (L - patch_len) // stride
    patches = x.unfold(dimension=1, size=patch_len, step=stride)  # [B, N, P, C]
    return patches.contiguous().view(B, n, patch_len * C)


# ----------------------------
# Built-in callbacks
# ----------------------------

class PatchingCallback(Callback):
    """
    Creates train-time patches from batch["x_context"] / ["x_target"] and writes them back.
    - By default, replaces x_context/x_target with patched tensors (set replace=False to keep originals).
    - Adds keys "..._patched" if replace=False.
    """
    order = -10  # run early

    def __init__(
        self,
        patch_len: int,
        stride: int,
        context_key: str = "x_context",
        target_key: str = "x_target",
        replace: bool = True,
        do_on_train: bool = True,
        do_on_val: bool = True,
    ):
        self.P = patch_len
        self.S = stride
        self.ctx_k = context_key
        self.tgt_k = target_key
        self.replace = replace
        self.do_on_train = do_on_train
        self.do_on_val = do_on_val

    def before_forward(self, batch: Dict[str, torch.Tensor]):
        is_train = self.learn.training
        if (is_train and not self.do_on_train) or ((not is_train) and not self.do_on_val):
            return

        # context is required; target optional
        if self.ctx_k not in batch:
            return

        x_ctx = batch[self.ctx_k]  # [B, L, C]
        x_ctx_p = make_patches(x_ctx, self.P, self.S)  # [B, N, P*C]

        # Patch target if present
        x_tgt_p = None
        if self.tgt_k in batch:
            x_tgt = batch[self.tgt_k]
            if x_tgt.ndim >= 2 and x_tgt.shape[1] >= self.P:
                x_tgt_p = make_patches(x_tgt, self.P, self.S)  # [B, N_t, P*C]

        if self.replace:
            batch[self.ctx_k] = x_ctx_p
            if x_tgt_p is not None:
                batch[self.tgt_k] = x_tgt_p
        else:
            batch[self.ctx_k + "_patched"] = x_ctx_p
            if x_tgt_p is not None:
                batch[self.tgt_k + "_patched"] = x_tgt_p


class StatsPrinter(Callback):
    """
    Prints running stats every log_every steps.
    """
    order = 50

    def __init__(self, log_every: int = 100):
        self.log_every = log_every
        self.start_time = time.time()

    def before_fit(self):
        self.step = 0

    def after_loss(self, loss: torch.Tensor, batch: Dict[str, torch.Tensor]):
        self.step += 1
        if self.learn.training and (self.step % self.log_every == 0):
            lr = self.learn.opt.param_groups[0]["lr"] if self.learn.opt else float("nan")
            ema_decay = self.learn.last_ema_decay
            ema_str = f" ema_decay={ema_decay:.6f}" if ema_decay is not None else ""
            l1 = self.learn.last_l1_loss
            var = self.learn.last_var_loss
            l1_str = f" l1={l1:.5f}" if l1 is not None else ""
            var_str = f" var={var:.5f}" if var is not None else ""
            print(f"[train] time={time.time()-self.start_time:.2f} step={self.step} loss={loss.item():.5f}{l1_str}{var_str} lr={lr:.2e}{ema_str} cosine sim.={self.learn.cosine_similarity:.3f} std context={self.learn.std_ctx:.3f} std target={self.learn.std_tgt:.3f}")

    def after_epoch(self):
        print(f"Epoch {self.learn.epoch+1}/{self.learn.n_epochs} "
              f"train_loss={self.learn.epoch_train_loss:.5f} "
              f"train_l1={self.learn.epoch_l1_loss:.5f} "
              f"train_var_loss={self.learn.epoch_var_loss:.5f} "
              f"val_loss={self.learn.epoch_val_loss:.5f}"
              )


class CSVLogger(Callback):
    """
    Saves epoch-level metrics to CSV.
    """
    order = 60

    def __init__(self, path: str = "logs/train_log.csv"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._wrote_header = False

    def after_epoch(self):
        row = {
            "epoch": self.learn.epoch,
            "train_loss": self.learn.epoch_train_loss,
            "val_loss": self.learn.epoch_val_loss,
            "train_cosine_similarity": self.learn.epoch_cosine_similarity,
            "train_std_ctx": self.learn.epoch_std_ctx,
            "train_std_tgt": self.learn.epoch_std_tgt,
        }
        write_header = not os.path.exists(self.path) or (not self._wrote_header)
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
                self._wrote_header = True
            w.writerow(row)


class CheckpointCallback(Callback):
    """
    Saves best model by monitor metric (default val_loss) and optional periodic snapshots.
    """
    order = 70

    def __init__(
        self,
        dirpath: str = "checkpoints",
        monitor: str = "val_loss",
        mode: str = "min",           # 'min' or 'max'
        every_n_epochs: Optional[int] = None,
        filename_best: str = "best.pt",
        dont_save_for_epochs: Optional[int] = 0
    ):
        os.makedirs(dirpath, exist_ok=True)
        self.dir = dirpath
        self.monitor = monitor
        self.mode = mode
        self.every_n = every_n_epochs
        self.filename_best = filename_best
        self.best = math.inf if mode == "min" else -math.inf
        self.dont_save_for_epochs = dont_save_for_epochs

    def _is_better(self, current: float) -> bool:
        return current < self.best if self.mode == "min" else current > self.best

    def _extra_checkpoint_data(self) -> Dict[str, Any]:
        extra: Dict[str, Any] = {}
        train_dl = getattr(self.learn, "train_dl", None)
        dataset = getattr(train_dl, "dataset", None)
        if dataset is None:
            return extra
        asset_universe = getattr(dataset, "asset_universe", None)
        if asset_universe:
            extra["asset_universe"] = list(asset_universe)
        return extra

    def after_epoch(self):
        metric = getattr(self.learn, f"epoch_{self.monitor}", None)
        if metric is None:
            # default: try attribute matching e.g. epoch_val_loss
            if self.monitor == "val_loss":
                metric = self.learn.epoch_val_loss
            else:
                return

        # Save best
        if self._is_better(metric) and self.learn.epoch + 1 > self.dont_save_for_epochs:
            self.best = metric
            path = os.path.join(self.dir, self.filename_best)
            payload = {
                "model": self.learn.model.state_dict(),
                "epoch": self.learn.epoch,
                "monitor": metric,
            }
            payload.update(self._extra_checkpoint_data())
            torch.save(payload, path)
            print(f"[ckpt] Saved best to {path} ({self.monitor}={metric:.5f})")

        # Periodic
        if self.every_n and ((self.learn.epoch + 1) % self.every_n == 0):
            path = os.path.join(self.dir, f"epoch{self.learn.epoch+1}.pt")
            payload = {
                "model": self.learn.model.state_dict(),
                "epoch": self.learn.epoch,
            }
            payload.update(self._extra_checkpoint_data())
            torch.save(payload, path)
            print(f"[ckpt] Saved periodic to {path}")


class CustomTensorboardCallback(BaseCallback):
    def __init__(self, episode_window: int = 100, trade_eps: float = 1e-8, verbose=0):
        super().__init__(verbose=verbose)
        self.episode_window = int(episode_window)
        self.trade_eps = float(trade_eps)
        self._episode_rewards: list[float] = []
        self._episode_trades: list[int] = []
        self._recent_episode_rewards: deque[float] = deque(maxlen=self.episode_window)
        self._recent_episode_trades: deque[float] = deque(maxlen=self.episode_window)

    def _on_training_start(self) -> None:
        n_envs = int(getattr(self.training_env, "num_envs", 1))
        self._episode_rewards = [0.0 for _ in range(n_envs)]
        self._episode_trades = [0 for _ in range(n_envs)]
        self._recent_episode_rewards = deque(maxlen=self.episode_window)
        self._recent_episode_trades = deque(maxlen=self.episode_window)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not infos:
            return True

        turnovers = []
        positions = []
        wealths = []
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        ended_episode_rewards = []
        ended_episode_trades = []
        turnover_by_env = [None for _ in range(len(infos))]

        for idx, info in enumerate(infos):
            if not info:
                continue
            if "turnover" in info:
                turnovers.append(info["turnover"])
                turnover_by_env[idx] = float(info["turnover"])
            if "position" in info:
                positions.append(abs(info["position"]))
            if "wealth" in info:
                wealths.append(info["wealth"])

        if rewards is not None:
            rewards_arr = np.asarray(rewards, dtype=np.float64).reshape(-1)
            self.logger.record("custom/reward_mean", float(np.mean(rewards_arr)))

            if dones is not None and len(self._episode_rewards) == len(rewards_arr):
                dones_arr = np.asarray(dones, dtype=bool).reshape(-1)
                for i, step_reward in enumerate(rewards_arr):
                    self._episode_rewards[i] += float(step_reward)
                    turnover_i = turnover_by_env[i]
                    if turnover_i is not None and turnover_i > self.trade_eps:
                        self._episode_trades[i] += 1
                    if dones_arr[i]:
                        episode_reward = self._episode_rewards[i]
                        episode_trades = self._episode_trades[i]
                        ended_episode_rewards.append(episode_reward)
                        ended_episode_trades.append(episode_trades)
                        self._recent_episode_rewards.append(episode_reward)
                        self._recent_episode_trades.append(episode_trades)
                        self._episode_rewards[i] = 0.0
                        self._episode_trades[i] = 0

        if turnovers:
            self.logger.record("custom/turnover_mean", float(np.mean(turnovers)))
        if positions:
            self.logger.record("custom/position_abs_mean", float(np.mean(positions)))
        if wealths:
            self.logger.record("custom/wealth_mean", float(np.mean(wealths)))
        if ended_episode_rewards:
            self.logger.record("custom/episode_reward_mean", float(np.mean(ended_episode_rewards)))
        if ended_episode_trades:
            self.logger.record("custom/episode_trades_mean", float(np.mean(ended_episode_trades)))
        return True


class RewardEvalCallback(EvalCallback):
    def __init__(self, *args, trade_eps: float = 1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.trade_eps = float(trade_eps)
        self._eval_trade_counts_by_env: dict[int, int] = {}
        self._last_eval_episode_trades: list[int] = []

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        super()._log_success_callback(locals_, globals_)
        info = locals_.get("info")
        done = bool(locals_.get("done", False))
        env_idx = int(locals_.get("i", 0))

        if env_idx not in self._eval_trade_counts_by_env:
            self._eval_trade_counts_by_env[env_idx] = 0

        if isinstance(info, dict):
            turnover = info.get("turnover")
            if turnover is not None and float(turnover) > self.trade_eps:
                self._eval_trade_counts_by_env[env_idx] += 1

        if done:
            self._last_eval_episode_trades.append(self._eval_trade_counts_by_env[env_idx])
            self._eval_trade_counts_by_env[env_idx] = 0

    def _on_step(self) -> bool:
        eval_now = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if eval_now:
            self._eval_trade_counts_by_env = {}
            self._last_eval_episode_trades = []
        continue_training = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0 and self.evaluations_results:
            eval_rewards = np.asarray(self.evaluations_results[-1], dtype=np.float64)
            if eval_rewards.size:
                self.logger.record("custom/eval_episode_reward_mean", float(np.mean(eval_rewards)))
            if self._last_eval_episode_trades:
                self.logger.record(
                    "custom/eval_episode_trades_mean",
                    float(np.mean(self._last_eval_episode_trades)),
                )
            if self.evaluations_length:
                eval_lengths = np.asarray(self.evaluations_length[-1], dtype=np.float64)
                if eval_rewards.size and eval_lengths.size:
                    mean_len = float(np.mean(eval_lengths))
                    if mean_len > 0:
                        self.logger.record(
                            "custom/eval_reward_mean",
                            float(np.mean(eval_rewards) / mean_len),
                        )
        return continue_training


class EntropyScheduleCallback(BaseCallback):
    def __init__(self, total_timesteps: int, warmup_fraction: float, ent_coef_start: float, ent_coef_end: float, verbose=0):
        super().__init__(verbose=verbose)
        self.total_timesteps = max(1, int(total_timesteps))
        self.warmup_fraction = float(warmup_fraction)
        self.ent_coef_start = float(ent_coef_start)
        self.ent_coef_end = float(ent_coef_end)

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps
        if progress < self.warmup_fraction:
            ent_coef = self.ent_coef_start
        else:
            ent_coef = self.ent_coef_end

        self.model.ent_coef = ent_coef
        self.logger.record("custom/ent_coef", float(ent_coef))
        return True


class TransactionCostScheduleCallback(BaseCallback):
    def __init__(
        self,
        total_timesteps: int,
        cost_start: float,
        cost_end: float,
        cost_steps: int,
        cost_warmup_timesteps: int = 0,
        eval_env=None,
        verbose=0,
    ):
        super().__init__(verbose=verbose)
        self.total_timesteps = max(1, int(total_timesteps))
        self.cost_start = float(cost_start)
        self.cost_end = float(cost_end)
        self.cost_steps = max(1, int(cost_steps))
        self.cost_warmup_timesteps = max(0, int(cost_warmup_timesteps))
        self.eval_env = eval_env
        self._last_cost: float | None = None
        self._last_level: int | None = None

    def _scheduled_cost_and_level(self) -> tuple[float, int]:
        # Hold at start cost during warmup.
        if self.num_timesteps < self.cost_warmup_timesteps:
            return self.cost_start, 0

        # After warmup, increase in discrete steps until reaching end cost.
        if self.cost_start == self.cost_end:
            return self.cost_end, 0

        n_increments = self.cost_steps
        post_warmup_total = max(1, self.total_timesteps - self.cost_warmup_timesteps)
        post_warmup_steps = max(0, self.num_timesteps - self.cost_warmup_timesteps)
        progress = min(1.0, max(0.0, post_warmup_steps / post_warmup_total))
        increment_idx = min(int(progress * n_increments), n_increments)
        alpha = increment_idx / n_increments
        cost = self.cost_start + alpha * (self.cost_end - self.cost_start)
        return cost, increment_idx

    def _set_cost(self, env, cost: float) -> None:
        if env is None:
            return
        # GymTradingEnv stores the trading env in .env; update all vectorized workers.
        env.env_method("set_transaction_cost", float(cost))

    def _on_training_start(self) -> None:
        initial_cost, initial_level = self._scheduled_cost_and_level()
        self._set_cost(self.training_env, initial_cost)
        self._set_cost(self.eval_env, initial_cost)
        self._last_cost = initial_cost
        self._last_level = initial_level
        self.logger.record("custom/transaction_cost", float(initial_cost))
        self.logger.record("custom/transaction_cost_level", int(initial_level))
        self.logger.record("custom/transaction_cost_levels_total", int(self.cost_steps))
        self.logger.record("custom/transaction_cost_warmup_timesteps", int(self.cost_warmup_timesteps))

    def _on_step(self) -> bool:
        cost, level = self._scheduled_cost_and_level()
        if self._last_cost is None or abs(cost - self._last_cost) > 1e-12:
            self._set_cost(self.training_env, cost)
            self._set_cost(self.eval_env, cost)
            self._last_cost = cost
            self._last_level = level
        self.logger.record("custom/transaction_cost", float(cost))
        self.logger.record("custom/transaction_cost_level", int(level))
        return True
