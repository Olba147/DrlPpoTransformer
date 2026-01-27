from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable
import os
import csv
import math
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import time

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
            print(f"[train] time={time.time()-self.start_time:.2f} step={self.step} loss={loss.item():.5f} lr={lr:.2e} cosine sim.={self.learn.cosine_similarity:.3f} std context={self.learn.std_ctx:.3f} std target={self.learn.std_tgt:.3f}")

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

