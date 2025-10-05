from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable
import os
import csv
import math
import torch
from torch.utils.data import DataLoader

# ----------------------------
# Minimal callback primitives
# ----------------------------

class Callback:
    "Base class. Override the hooks you need."
    order: int = 0       # lower runs earlier

    def set_learner(self, learn: "Learner"):
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

    def set_learner(self, learn: "Learner"):
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

    def before_fit(self):
        self.step = 0

    def after_loss(self, loss: torch.Tensor, batch: Dict[str, torch.Tensor]):
        self.step += 1
        if self.learn.training and (self.step % self.log_every == 0):
            lr = self.learn.opt.param_groups[0]["lr"] if self.learn.opt else float("nan")
            print(f"[train] step={self.step} loss={loss.item():.5f} lr={lr:.2e}")

    def after_epoch(self):
        print(f"Epoch {self.learn.epoch+1}/{self.learn.n_epochs} "
              f"train_loss={self.learn.epoch_train_loss:.5f} "
              f"val_loss={self.learn.epoch_val_loss:.5f}")


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
    ):
        os.makedirs(dirpath, exist_ok=True)
        self.dir = dirpath
        self.monitor = monitor
        self.mode = mode
        self.every_n = every_n_epochs
        self.filename_best = filename_best
        self.best = math.inf if mode == "min" else -math.inf

    def _is_better(self, current: float) -> bool:
        return current < self.best if self.mode == "min" else current > self.best

    def after_epoch(self):
        metric = getattr(self.learn, f"epoch_{self.monitor}", None)
        if metric is None:
            # default: try attribute matching e.g. epoch_val_loss
            if self.monitor == "val_loss":
                metric = self.learn.epoch_val_loss
            else:
                return

        # Save best
        if self._is_better(metric):
            self.best = metric
            path = os.path.join(self.dir, self.filename_best)
            torch.save({"model": self.learn.model.state_dict(),
                        "epoch": self.learn.epoch,
                        "monitor": metric}, path)
            print(f"[ckpt] Saved best to {path} ({self.monitor}={metric:.5f})")

        # Periodic
        if self.every_n and ((self.learn.epoch + 1) % self.every_n == 0):
            path = os.path.join(self.dir, f"epoch{self.learn.epoch+1}.pt")
            torch.save({"model": self.learn.model.state_dict(),
                        "epoch": self.learn.epoch}, path)
            print(f"[ckpt] Saved periodic to {path}")


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
        self.epoch = 0
        self.n_epochs = None
        self.epoch_train_loss: float = float("nan")
        self.epoch_val_loss: float = float("nan")

    # ---- helpers to call callbacks ----
    def _cb(self, name: str, *args, **kwargs):
        self.cbs._call(name, *args, **kwargs)

    # ---- main API ----
    def fit(self, n_epochs: int):
        self.n_epochs = n_epochs
        self._cb("before_fit")
        for ep in range(n_epochs):
            self.epoch = ep
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
                self._cb("after_step")

            total_loss += loss.detach().item()
            n_steps += 1
            self._cb("after_batch", batch)

        avg_loss = total_loss / max(1, n_steps)
        if train:
            self.epoch_train_loss = avg_loss
        else:
            self.epoch_val_loss = avg_loss

        self._cb("after_train" if train else "after_validate")