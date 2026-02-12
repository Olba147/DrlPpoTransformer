from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Iterable
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np

from Training.callbacks import Callback, CallbackList

# ----------------------------
# Learner
# ----------------------------


class Learner:

    def __init__(
        self,
        model: torch.nn.Module,
        train_dl: DataLoader,
        val_dl: DataLoader,
        opt: Any,                                # callable(params, lr) -> optimizer
        loss_func: Any = torch.nn.MSELoss(reduction="mean"),         # callable(pred, batch) -> loss tensor
        cbs: Optional[List[Callback]] = None,
        device: Optional[torch.device] = None,
        grad_clip: Optional[float] = None,
        amp: bool = False,                  # fp16 autocast
        start_epoch: int = 0,
        warmup_epochs: int = 20,
        global_step: int = 0,
        var_loss: bool = False,
        var_loss_gamma: float = 1.0,
        var_loss_weight: float = 1.0
    ):
    
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.opt = opt
        self.cbs = cbs
        self.device = device
        self.grad_clip = grad_clip
        self.amp = amp
        
        # ema update
        self.global_step = global_step
        self.warmup_steps = len(self.train_dl) * warmup_epochs  # 20 epochs worth of steps
        self.ema_arr = torch.linspace(self.model.ema_tau_min, self.model.ema_tau_max, self.warmup_steps)

        # variance loss
        self.var_loss = var_loss
        self.var_loss_gamma = var_loss_gamma
        self.var_loss_weight = var_loss_weight

        # propertios
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.cbs = CallbackList(self.cbs or [])
        self.cbs.set_learner(self)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        # training state
        self.training = True
        self.start_epoch = start_epoch
        self.epoch = start_epoch
        self.n_epochs = None
        self.epoch_train_loss: float = float("nan")
        self.epoch_val_loss: float = float("nan")
        
        # additional tracking
        self.cosine_similarity = None
        self.std_ctx = None
        self.std_tgt = None
        self.epoch_cosine_similarity = float("nan")
        self.epoch_std_ctx = float("nan")
        self.epoch_std_tgt = float("nan")
        self.last_ema_decay: Optional[float] = None
        self.last_mse_loss: Optional[float] = None
        self.last_cos_loss: Optional[float] = None
        self.last_var_loss: Optional[float] = None
        self.epoch_mse_loss: float = float("nan")
        self.epoch_cos_loss: float = float("nan")
        self.epoch_var_loss: float = float("nan")


    # ---- helpers to call callbacks ----
    def _cb(self, name: str, *args, **kwargs):
        self.cbs._call(name, *args, **kwargs)

    def _get_ema_tau(self):    
        """Cosine schedule for EMA tau"""
        if self.global_step < self.warmup_steps:
            return self.ema_arr[self.global_step]
        else:
            return self.model.ema_tau_max
    
    @staticmethod
    def variance_penalty(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
        """
        VICReg-style variance hinge penalty.
        Penalizes dimensions whose batch std < gamma.

        z: [B, D]
        """
        if z.dim() != 2:
            z = z.view(z.size(0), -1)
        std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)  # [D]
        return F.relu(gamma - std).mean()

    # ---- main API ----
    def fit(self, n_epochs: int):
        self.n_epochs = n_epochs
        self._cb("before_fit")
        for ep in range(n_epochs):
            self.epoch = self.start_epoch + ep
            self._one_epoch(train=True)
            if self.val_dl is not None:
                self._one_epoch(train=False)
            self._cb("after_epoch")
        self._cb("after_fit")

    # ---- one epoch ----
    def _one_epoch(self, train: bool):
        self.training = train
        dl = self.train_dl if train else self.val_dl
        if dl is None: return

        self.model.train(train)
        self._cb("before_train" if train else "before_validate")

        total_loss = 0.0
        mse_sum = 0.0
        cos_sum_loss = 0.0
        var_sum = 0.0
        n_steps = 0
        cos_sum = 0.0
        std_ctx_sum = 0.0
        std_tgt_sum = 0.0

        for batch in dl:
            # move batch to device
            batch = {key: value.to(self.device) for key, value in batch.items()}

            self._cb("before_batch", batch)
            self._cb("before_forward", batch)

            x_context = batch["x_context"]
            t_context = batch["t_context"]
            x_target = batch["x_target"]
            t_target = batch["t_target"]
            asset_id = batch.get("asset_id", None)

            with torch.cuda.amp.autocast(enabled=self.amp):
                # Expect model to accept the whole dict or at least the features under known keys.
                # If your model signature is different, adapt here.
                pred = self.model(x_context, t_context, x_target, t_target, asset_id=asset_id)
                self._cb("after_pred", pred, batch)
                mse_loss = F.mse_loss(pred[0], pred[1])
                cos_loss = 1.0 - F.cosine_similarity(pred[0], pred[1], dim=1).mean()
                
                if self.var_loss:
                    var_loss = self.variance_penalty(pred[0], gamma=self.var_loss_gamma)
                    var_loss = self.var_loss_weight * var_loss
                    self.last_var_loss = float(var_loss.detach().item())
                else:
                    var_loss = mse_loss.new_tensor(0.0)

                loss = mse_loss + cos_loss + var_loss

                self.last_mse_loss = float(mse_loss.detach().item())
                self.last_cos_loss = float(cos_loss.detach().item())
                self._cb("after_loss", loss, batch)

            if train:
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self._cb("after_backward")
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()

                # jepa ema update
                if hasattr(self.model, "ema_update"):
                    t = self._get_ema_tau()
                    self.model.ema_update(t)
                    self.last_ema_decay = t
                    self.global_step += 1


                # calculate cosine similarity for each batch for tracking
                with torch.no_grad():
                    self.cosine_similarity = F.cosine_similarity(pred[0], pred[1], dim=1).mean().item()
                    self.std_ctx = pred[0].std(dim=0).mean().item()
                    self.std_tgt = pred[1].std(dim=0).mean().item()
                    cos_sum += self.cosine_similarity
                    std_ctx_sum += self.std_ctx
                    std_tgt_sum += self.std_tgt

                self._cb("after_step")

            total_loss += loss.detach().item()
            mse_sum += mse_loss.detach().item()
            cos_sum_loss += cos_loss.detach().item()
            var_sum += var_loss.detach().item()
            n_steps += 1
            self._cb("after_batch", batch)

        avg_loss = total_loss / max(1, n_steps)
        if train:
            self.epoch_train_loss = avg_loss
            denom = max(1, n_steps)
            self.epoch_cosine_similarity = cos_sum / denom
            self.epoch_std_ctx = std_ctx_sum / denom
            self.epoch_std_tgt = std_tgt_sum / denom
            self.epoch_mse_loss = mse_sum / denom
            self.epoch_cos_loss = cos_sum_loss / denom
            self.epoch_var_loss = var_sum / denom
        else:
            self.epoch_val_loss = avg_loss

        self._cb("after_train" if train else "after_validate")
