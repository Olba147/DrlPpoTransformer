from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

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

# --------------- Small helpers ------------

def _split_indices(T: int, tr: float, vr: float) -> Tuple[slice, slice, slice]:
    n_tr = int(T * tr); n_vr = int(T * vr)
    return slice(0, n_tr), slice(n_tr, n_tr + n_vr), slice(n_tr + n_vr, T)

def _fit_stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(0); std = x.std(0, unbiased=False).clamp_min(1e-6)
    return mean, std

def _build_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X:  [T, 6] = [log_open, log_high, log_low, log_ret_close, log1p(volume), log1p(quote_asset_volume)]
      W:  [T]    = weekday index (Mon=0..Sun=6)  -- int64
      TC: [T, 2] = time-of-day cyclical [sin, cos]
    """
    # --- price transforms ---
    o = np.log(df["open"].astype(float).to_numpy())
    h = np.log(df["high"].astype(float).to_numpy())
    l = np.log(df["low"].astype(float).to_numpy())
    c = np.log(df["close"].astype(float).to_numpy())
    log_ret_c = np.diff(c, prepend=np.nan)

    # --- volumes ---
    v  = np.log1p(np.clip(df["volume"].astype(float).to_numpy(), 0.0, None))
    qv = np.log1p(np.clip(df["quote_asset_volume"].astype(float).to_numpy(), 0.0, None))

    # --- time features from ms ---
    ts = pd.to_datetime(df["close_time"].astype("int64"), unit="ms", utc=True)
    # get weekday index (Mon=0..Sun=6)
    weekday = ts.dt.dayofweek.to_numpy()
    hour = ts.dt.hour.to_numpy()
    minute = ts.dt.minute.to_numpy()
    time_cont = np.stack([weekday, hour, minute], axis=1)              # [T,2]

    # align after diff (drop first row across all)
    mask = ~np.isnan(log_ret_c)
    X = np.stack([o, h, l, log_ret_c, v, qv], axis=1)[mask].astype(np.float32)
    W = weekday[mask]                   # int64
    TC = time_cont[mask].astype(np.float32)

    return X, W, TC  # [T-1,6], [T-1], [T-1,2]

# --------------- Dataset ------------------

class PatchSequenceDataset(Dataset):
    """
    Yields:
      X_patch : [N_patches, patch_len * 6]   # tokens for patch embed
      y       : [pred_len]                   # future log-returns of close (possibly standardized)
      time_idx: [seq_len] (int64)            # weekday indices for embedding
      time_cont: [seq_len, 2] (float32)      # time-of-day [sin, cos]
      stats   : {mean,std,target_mean,target_std} (on train only)
    """
    def __init__(self, cfg: SeqConfig, split: str, stats: Optional[Dict[str, torch.Tensor]] = None):
        usecols = ["open","high","low","close","volume","quote_asset_volume","close_time"]
        df = pd.read_csv(cfg.csv_path, usecols=usecols)
        X_np, W_np, TC_np = _build_arrays(df)

        X = torch.from_numpy(X_np)                 # [T,6]
        W = torch.from_numpy(W_np)                 # [T]
        TC = torch.from_numpy(TC_np)               # [T,2]

        # splits (contiguous)
        tr_sl, va_sl, te_sl = _split_indices(len(X), cfg.train_ratio, cfg.val_ratio)
        sl = {"train": tr_sl, "val": va_sl, "test": te_sl}[split]
        self.X = X[sl]; self.W = W[sl]; self.TC = TC[sl]
        self.cfg = cfg

        # standardize features (not time tensors)
        if cfg.standardize:
            if split == "train":
                mean, std = _fit_stats(self.X)
                self.stats = {
                    "mean": mean, "std": std,
                    "target_mean": mean[3].detach(),   # channel 3 = log_ret_close
                    "target_std":  std[3].detach(),
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

        # patch geometry
        self.n_patches = 1 + (cfg.seq_len - cfg.patch_len) // cfg.patch_stride
        if self.n_patches <= 0:
            raise ValueError("Invalid patch_len/stride for seq_len.")

    def __len__(self) -> int:
        return self.max_start

    def __getitem__(self, i: int):
        Lx, Ly, pl, ps = self.cfg.seq_len, self.cfg.pred_len, self.cfg.patch_len, self.cfg.patch_stride
        X = self.X[i:i+Lx]                    # [Lx,6]
        y = self.X[i+Lx:i+Lx+Ly, 3]           # [Ly]  (future log-ret-close)
        W = self.W[i:i+Lx].to(torch.long)     # weekday indices
        TC = self.TC[i:i+Lx]                  # [Lx,2] time-of-day sin/cos

        # patchify X along time (flatten each patch)
        starts = torch.arange(0, Lx - pl + 1, ps)
        X_patch = torch.stack([X[s:s+pl].reshape(-1) for s in starts.tolist()], dim=0)  # [N_patches, pl*6]

        item = {"X_patch": X_patch, "y": y, "time_idx": W, "time_cont": TC}
        if self.stats is not None:
            item["stats"] = self.stats
        return item

# --------------- Dataloaders ---------------

def create_patch_dataloaders(cfg: SeqConfig, batch_size: int = 64):
    train_ds = PatchSequenceDataset(cfg, "train", stats=None)
    stats = train_ds.stats if cfg.standardize else None
    val_ds  = PatchSequenceDataset(cfg, "val",  stats=stats)
    test_ds = PatchSequenceDataset(cfg, "test", stats=stats)

    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh, num_workers=0, drop_last=False)
    return mk(train_ds, True), mk(val_ds, False), mk(test_ds, False), stats