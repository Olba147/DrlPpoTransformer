from __future__ import annotations

from typing import Callable, Dict, Optional

import torch

from Training.callbacks import Callback, CallbackList


class PPOTrainer:
    def __init__(
        self,
        policy: torch.nn.Module,
        value: torch.nn.Module,
        cbs: Optional[list[Callback]] = None,
        device: Optional[torch.device] = None,
    ):
        self.policy = policy
        self.value = value
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.value.to(self.device)
        self.cbs = CallbackList(cbs or [])
        self.cbs.set_learner(self)

        self.update = 0
        self.n_updates: Optional[int] = None
        self.last_rollout_metrics: Dict[str, float] = {}
        self.last_update_metrics: Dict[str, float] = {}

    def _cb(self, name: str, *args, **kwargs):
        self.cbs._call(name, *args, **kwargs)

    def train(
        self,
        n_updates: int,
        rollout_fn: Callable[[], Dict[str, float]],
        update_fn: Callable[[Dict[str, float]], Dict[str, float]],
    ):
        self.n_updates = n_updates
        self._cb("before_fit")
        for update_idx in range(n_updates):
            self.update = update_idx
            self._cb("before_update", update_idx)

            rollout_metrics = rollout_fn()
            self.last_rollout_metrics = rollout_metrics or {}
            self._cb("after_rollout", update_idx, self.last_rollout_metrics)

            update_metrics = update_fn(self.last_rollout_metrics)
            self.last_update_metrics = update_metrics or {}
            self._cb("after_update", update_idx, self.last_update_metrics)

        self._cb("after_fit")
