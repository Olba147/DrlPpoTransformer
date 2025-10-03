from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------- Config ----------------

@dataclass
class SeqConfig:
    csv_path: str
    seq_len: int = 512
    pred_len: int = 96
    patch_len: int = 16
    patch_stride: int = 16
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    standardize: bool = True   # z-score per feature on TRAIN slice
    use_patching: bool = True  # default; can be overridden by create_dataloaders(..., use_patch=...)

# --------------- Small helpers ------------

def _split_indices(T: int, tr: float, vr: float) -> Tuple[slice, slice, slice]:
    n_tr = int(T * tr)
    n_vr = int(T * vr)
    return slice(0, n_tr), slice(n_tr, n_tr + n_vr), slice(n_tr + n_vr, T)

def _fit_stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(0)
    std = x.std(0, unbiased=False).clamp_min(1e-6)
    return mean, std

def _build_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X:  [T, 6] = [log_open, log_high, log_low, log_ret_close, log1p(volume), log1p(quote_asset_volume)]
      W:  [T]    = weekday index (Mon=0..Sun=6)  -- int64
      TC: [T, 2] = [weekday, minute_of_day]
    """
    # price transforms
    o = np.log(df["open"].astype(float).to_numpy())
    h = np.log(df["high"].astype(float).to_numpy())
    l = np.log(df["low"].astype(float).to_numpy())
    c = np.log(df["close"].astype(float).to_numpy())
    log_ret_c = np.diff(c, prepend=np.nan)

    # volumes
    v = np.log1p(np.clip(df["volume"].astype(float).to_numpy(), 0.0, None))
    qv = np.log1p(np.clip(df["quote_asset_volume"].astype(float).to_numpy(), 0.0, None))

    # time features from ms
    ts = pd.to_datetime(df["close_time"].astype("int64"), unit="ms", utc=True)
    weekday = ts.dt.dayofweek.to_numpy()
    hour = ts.dt.hour.to_numpy()
    minute = ts.dt.minute.to_numpy()
    minute_of_day = minute + hour * 60

    time_cont = np.stack([weekday, minute_of_day], axis=1)

    # align after diff (drop first row across all)
    mask = ~np.isnan(log_ret_c)
    X = np.stack([o, h, l, log_ret_c, v, qv], axis=1)[mask].astype(np.float32)
    W = weekday[mask]
    TC = time_cont[mask].astype(np.float32)

    return X, W, TC

def _make_patch_starts(window_len: int, patch_len: int, stride: int) -> torch.Tensor:
    if patch_len > window_len:
        raise ValueError("patch_len must be <= window_len when patching.")
    starts = torch.arange(0, window_len - patch_len + 1, stride)
    if starts.numel() == 0:
        raise ValueError("Invalid patch_len/patch_stride for window_len.")
    return starts

def _num_patches(window_len: int, patch_len: int, stride: int) -> int:
    return 1 + (window_len - patch_len) // stride

# --------------- Dataset ------------------

class JointSeqDataset(Dataset):
    """
    Always returns the same keys; shapes depend on patching mode.

    If use_patch == False:
      X               : [seq_len, 6]
      X_tgt           : [pred_len, 6]
      time_cont       : [seq_len, 2]          (weekday, minute_of_day)
      time_cont_tgt   : [pred_len, 2]
      stats (optional)

    If use_patch == True:
      X               : [N_ctx, patch_len*6]
      X_tgt           : [N_tgt, patch_len*6]
      time_cont       : [N_ctx, patch_len*2]  (weekday, minute_of_day flattened per step)
      time_cont_tgt   : [N_tgt, patch_len*2]
      stats (optional)
    """

    def __init__(
        self,
        cfg: SeqConfig,
        split: str,
        stats: Optional[Dict[str, torch.Tensor]] = None,
        use_patch: Optional[bool] = None,
    ):
        usecols = ["open", "high", "low", "close", "volume", "quote_asset_volume", "close_time"]
        df = pd.read_csv(cfg.csv_path, usecols=usecols)
        X_np, W_np, TC_np = _build_arrays(df)

        X = torch.from_numpy(X_np)
        W = torch.from_numpy(W_np)
        TC = torch.from_numpy(TC_np)

        tr_sl, va_sl, te_sl = _split_indices(len(X), cfg.train_ratio, cfg.val_ratio)
        sl = {"train": tr_sl, "val": va_sl, "test": te_sl}[split]
        self.X = X[sl]
        self.W = W[sl]          # kept for completeness (not returned)
        self.TC = TC[sl]
        self.cfg = cfg

        # determine patching mode (override allowed)
        self.use_patch = cfg.use_patching if use_patch is None else bool(use_patch)

        # standardize (train stats only)
        if cfg.standardize:
            if split == "train":
                mean, std = _fit_stats(self.X)
                self.stats = {
                    "mean": mean,
                    "std": std,
                    "target_mean": mean[3].detach(),
                    "target_std": std[3].detach(),
                }
            else:
                if stats is None:
                    raise ValueError("Provide train stats for val/test when standardize=True.")
                self.stats = stats
            self.X = (self.X - self.stats["mean"]) / self.stats["std"]
        else:
            self.stats = None

        Lx, Ly = cfg.seq_len, cfg.pred_len
        self.max_start = self.X.size(0) - (Lx + Ly)
        if self.max_start <= 0:
            raise ValueError("Not enough timesteps for seq_len/pred_len.")

        # Precompute patch starts for context/target if needed
        if self.use_patch:
            self._starts_ctx = _make_patch_starts(cfg.seq_len, cfg.patch_len, cfg.patch_stride)
            if cfg.pred_len < cfg.patch_len:
                raise ValueError("pred_len must be >= patch_len to build target patches.")
            self._starts_tgt = _make_patch_starts(cfg.pred_len, cfg.patch_len, cfg.patch_stride)
        else:
            self._starts_ctx = None
            self._starts_tgt = None

    def __len__(self) -> int:
        return self.max_start

    def __getitem__(self, i: int):
        cfg = self.cfg
        # context and target contiguous blocks
        X_ctx = self.X[i : i + cfg.seq_len]                                # [Lx, 6]
        TC_ctx = self.TC[i : i + cfg.seq_len]                               # [Lx, 2]
        X_tgt = self.X[i + cfg.seq_len : i + cfg.seq_len + cfg.pred_len]    # [Ly, 6]
        TC_tgt = self.TC[i + cfg.seq_len : i + cfg.seq_len + cfg.pred_len]  # [Ly, 2]

        if not self.use_patch:
            item = {
                "X": X_ctx,                     # [Lx, 6]
                "X_tgt": X_tgt,                 # [Ly, 6]
                "time_cont": TC_ctx,            # [Lx, 2]
                "time_cont_tgt": TC_tgt,        # [Ly, 2]
            }
        else:
            P = cfg.patch_len
            # context patches
            Xp_ctx = torch.stack([X_ctx[s:s+P].reshape(-1) for s in self._starts_ctx.tolist()], dim=0)   # [N_ctx, P*6]
            TCp_ctx = torch.stack([TC_ctx[s:s+P].reshape(-1) for s in self._starts_ctx.tolist()], dim=0) # [N_ctx, P*2]
            # target patches
            Xp_tgt = torch.stack([X_tgt[s:s+P].reshape(-1) for s in self._starts_tgt.tolist()], dim=0)   # [N_tgt, P*6]
            TCp_tgt = torch.stack([TC_tgt[s:s+P].reshape(-1) for s in self._starts_tgt.tolist()], dim=0) # [N_tgt, P*2]

            item = {
                "X": Xp_ctx,                    # [N_ctx, P*6]
                "X_tgt": Xp_tgt,                # [N_tgt, P*6]
                "time_cont": TCp_ctx,           # [N_ctx, P*2]
                "time_cont_tgt": TCp_tgt,       # [N_tgt, P*2]
            }

        if self.stats is not None:
            item["stats"] = self.stats

        return item

# --------------- Dataloaders ---------------

def create_dataloaders(cfg: SeqConfig, batch_size: int = 64, use_patch: Optional[bool] = None):
    """
    Set use_patch=True/False to force a mode (overrides cfg.use_patching).
    Returns train/val/test loaders and stats (if standardized).
    """
    train_ds = JointSeqDataset(cfg, "train", stats=None, use_patch=use_patch)
    stats = train_ds.stats if cfg.standardize else None
    val_ds = JointSeqDataset(cfg, "val", stats=stats, use_patch=use_patch)
    test_ds = JointSeqDataset(cfg, "test", stats=stats, use_patch=use_patch)

    def mk(ds: JointSeqDataset, sh: bool) -> DataLoader:
        return DataLoader(ds, batch_size=batch_size, shuffle=sh, num_workers=0, drop_last=False)

    return mk(train_ds, True), mk(val_ds, False), mk(test_ds, False), stats