from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Iterable
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

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
        start_epoch: int = 0
    ):
    
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.opt = opt
        self.loss_func = loss_func
        self.cbs = cbs
        self.device = device
        self.grad_clip = grad_clip
        self.amp = amp
        
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


    # ---- helpers to call callbacks ----
    def _cb(self, name: str, *args, **kwargs):
        self.cbs._call(name, *args, **kwargs)

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

            with torch.cuda.amp.autocast(enabled=self.amp):
                # Expect model to accept the whole dict or at least the features under known keys.
                # If your model signature is different, adapt here.
                pred = self.model(x_context, t_context, x_target, t_target)
                self._cb("after_pred", pred, batch)
                loss = self.loss_func(pred[0], pred[1])
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
                    self.model.ema_update(epoch=self.epoch)


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
            n_steps += 1
            self._cb("after_batch", batch)

        avg_loss = total_loss / max(1, n_steps)
        if train:
            self.epoch_train_loss = avg_loss
            denom = max(1, n_steps)
            self.epoch_cosine_similarity = cos_sum / denom
            self.epoch_std_ctx = std_ctx_sum / denom
            self.epoch_std_tgt = std_tgt_sum / denom
        else:
            self.epoch_val_loss = avg_loss

        self._cb("after_train" if train else "after_validate")
