# finance_dataset.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# -----------------------------
# Utilities
# -----------------------------
def to_datetime(series):
    # expects ms epoch or ISO string; adapt if needed
    try:
        # Binance close_time is usually in ms
        return pd.to_datetime(series, unit="ms", utc=True)
    except Exception:
        return pd.to_datetime(series, utc=True)

def true_range(high, low, close):
    prev_close = close.shift(1)
    a = (high - low).abs()
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)

def make_time_features(dt_index: pd.DatetimeIndex) -> np.ndarray:
    """
    Simple calendar features (kept for parity but not returned).
    weekday, minute_of_day
    """
    df = pd.DataFrame({"date": dt_index})
    df["weekday"] = df["date"].dt.weekday
    hour = df["date"].dt.hour
    minute = df["date"].dt.minute
    df["minute_of_day"]  = (hour * 60 + minute)
    return df[["weekday", "minute_of_day"]]

def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Lookahead-safe rolling z-score.
    Uses mean/std computed on [t-window, t-1] via shift(1).
    """
    m = series.rolling(window, min_periods=window).mean().shift(1)
    s = series.rolling(window, min_periods=window).std().shift(1)
    return (series - m) / s

def rolling_zscore_df(df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for c in df.columns:
        out[c] = rolling_zscore(df[c], window)
    return out

# -----------------------------
# Finance-aware Dataset (train/val/test)
# -----------------------------
class Dataset_Finance(Dataset):
    """
    Finance-friendly dataset:
    - Reads CSV with columns: open,high,low,close,volume,close_time
    - Builds finance features
    - Applies rolling z-score (leakage-safe)
    - Returns dicts: x_context, x_target, ohlcv_context, ohlcv_target
    """
    def __init__(
        self,
        root_path: str,
        data_path: str,
        start_date: str = None,
        split: str = "train",            # 'train' | 'val' | 'test'
        size=None,                       # [seq_len, pred_len]
        use_time_features: bool = True, # kept for parity (not returned)
        time_col_name: str = "close_time",
        train_split: float = 0.7,
        test_split: float = 0.15,
        rolling_window: int = 252        # tune to your bar frequency
    ):
        assert split in ["train", "val", "test"]
        self.set_type = {"train": 0, "val": 1, "test": 2}[split]

        if size is None:
            self.seq_len   = 96
            self.pred_len  = 24
        else:
            self.seq_len, self.pred_len = size

        self.root_path = root_path
        self.data_path = data_path
        self.use_time_features = use_time_features
        self.time_col_name = time_col_name
        self.train_split = train_split
        self.test_split  = test_split
        self.rolling_window = rolling_window
        self.start_date = start_date

        self._read_and_build()

    # ---- core IO ----
    def _read_and_build(self):
        path = os.path.join(self.root_path, self.data_path)
        df = pd.read_csv(path)

        if self.start_date is not None:
            # filter by start date
            # read the start date and convert to milliseconds
            start_date = pd.to_datetime(self.start_date).timestamp() * 1000
            df = df[df["close_time"] >= start_date]

        # Ensure columns exist
        req = {"open","high","low","close","volume",self.time_col_name}
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Sort by time just in case
        dt = to_datetime(df[self.time_col_name])
        df = df.assign(_dt=dt).sort_values("_dt").reset_index(drop=True)

        # make time features
        if self.use_time_features:
            time_features = make_time_features(df["_dt"])

        # Keep raw OHLCV before any diff/rolling drops; we'll realign later
        ohlcv_cols = ["open","high","low","close","volume"]
        raw_ohlcv = df[ohlcv_cols].copy()

        # -----------------------------
        # Finance transforms / features
        # -----------------------------
        # Log prices
        for col in ["open","high","low","close"]:
            df[f"log_{col}"] = np.log(df[col].astype(float))

        # Log returns (1-step) — primary signal inputs
        df["ret_close"] = df["log_close"].diff()
        df["ret_open"]  = df["log_open"].diff()
        df["ret_high"]  = df["log_high"].diff()
        df["ret_low"]   = df["log_low"].diff()

        # Ranges & volatility proxies
        df["hl_range"]  = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
        df["tr"]        = true_range(df["high"], df["low"], df["close"]) / df["close"].replace(0, np.nan)
        df["vol20"]     = df["ret_close"].rolling(20, min_periods=20).std()

        # Volume transforms
        df["log_vol"]   = np.log(df["volume"].astype(float) + 1.0)
        df["vol_z20"]   = (df["volume"] - df["volume"].rolling(20, min_periods=20).mean()) / \
                           df["volume"].rolling(20, min_periods=20).std()

        # Feature set for model
        feat_cols = [
            "ret_close","ret_open","ret_high","ret_low",
            "hl_range","tr","vol20","log_vol","vol_z20"
        ]

        # We need to drop NaNs created by diffs/rolls before normalization
        base = df[feat_cols].copy()
        base = base.replace([np.inf, -np.inf], np.nan)
        base_valid = base.dropna()
        valid_index = base_valid.index

        # Align raw OHLCV and dates to valid index
        # raw_ohlcv = raw_ohlcv.loc[valid_index].reset_index(drop=True)
        # dates = df.loc[valid_index, "_dt"].reset_index(drop=True)

        # -----------------------------
        # Rolling z-score normalization (no leakage)
        # -----------------------------
        Z = rolling_zscore_df(base.loc[valid_index], self.rolling_window)
        # drop initial NaNs while window warms up
        Z = Z.replace([np.inf, -np.inf], np.nan).dropna()
        final_index = Z.index

        X = Z.to_numpy(dtype=np.float32)
        raw_ohlcv = raw_ohlcv.loc[final_index].reset_index(drop=True).to_numpy(dtype=np.float32)
        dates = time_features.loc[final_index].reset_index(drop=True).to_numpy(dtype=np.float32)
        

        # -----------------------------
        # Time-based split (like PatchTST)
        # -----------------------------
        n = len(X)
        num_train = int(n * self.train_split)
        num_test  = int(n * self.test_split)
        num_val   = n - num_train - num_test

        # borders for slices
        b1s = [0, num_train - self.seq_len, n - num_test - self.seq_len]
        b2s = [num_train, num_train + num_val, n]
        start = b1s[self.set_type]
        end   = b2s[self.set_type]

        # Store slices
        self.data_x = X[start:end]
        self.ohlcv  = raw_ohlcv[start:end]
        self.dates  = dates[start:end]

    def __len__(self):
        # last index that can form seq_len + pred_len
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end   = s_begin + self.seq_len

        #future-only target due to encoding only
        r_begin = s_end
        r_end   = s_end + self.pred_len

        # normalized features
        x_context = self.data_x[s_begin:s_end]
        x_target  = self.data_x[r_begin:r_end]        # shape: [pred_len, F]

        # raw OHLCV aligned
        ohlcv_context = self.ohlcv[s_begin:s_end]
        ohlcv_target  = self.ohlcv[r_begin:r_end]     # shape: [pred_len, 5]

        # date features
        dates_context = self.dates[s_begin:s_end]
        dates_target  = self.dates[r_begin:r_end]     # shape: [pred_len, 2]

        return {
            "x_context": torch.from_numpy(x_context).float(),
            "x_target": torch.from_numpy(x_target).float(),
            "t_context": torch.from_numpy(dates_context).float(),
            "t_target": torch.from_numpy(dates_target).float(),
            "ohlcv_context": torch.from_numpy(ohlcv_context).float(),
            "ohlcv_target": torch.from_numpy(ohlcv_target).float(),
        }

# -----------------------------
# Inference-only Dataset (like Dataset_Pred)
# -----------------------------
class Dataset_Finance_Pred(Dataset):
    """
    Builds the *last* seq_len window for prediction plus label scaffold (label_len only).
    Returns dicts: x_context, x_target (scaffold), ohlcv_context, ohlcv_target (scaffold)
    """
    def __init__(
        self,
        root_path: str,
        data_path: str,
        size=None,
        time_col_name: str = "close_time",
        rolling_window: int = 252,
        train_split: float = 0.7,  # used for choosing the "past" portion when warming up
        test_split: float = 0.15,
    ):
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 24
        else:
            self.seq_len, self.label_len, self.pred_len = size

        self.root_path = root_path
        self.data_path = data_path
        self.time_col_name = time_col_name
        self.rolling_window = rolling_window
        self.train_split = train_split
        self.test_split  = test_split

        self._read_and_build()

    def _read_and_build(self):
        path = os.path.join(self.root_path, self.data_path)
        df = pd.read_csv(path)
        dt = to_datetime(df[self.time_col_name])
        df = df.assign(_dt=dt).sort_values("_dt").reset_index(drop=True)

        # Keep raw OHLCV
        ohlcv_cols = ["open","high","low","close","volume"]
        raw_ohlcv_all = df[ohlcv_cols].copy()

        # Features (same as training)
        for col in ["open","high","low","close"]:
            df[f"log_{col}"] = np.log(df[col].astype(float))
        df["ret_close"] = df["log_close"].diff()
        df["ret_open"]  = df["log_open"].diff()
        df["ret_high"]  = df["log_high"].diff()
        df["ret_low"]   = df["log_low"].diff()
        df["hl_range"]  = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
        df["tr"]        = true_range(df["high"], df["low"], df["close"]) / df["close"].replace(0, np.nan)
        df["vol20"]     = df["ret_close"].rolling(20, min_periods=20).std()
        df["log_vol"]   = np.log(df["volume"].astype(float) + 1.0)
        df["vol_z20"]   = (df["volume"] - df["volume"].rolling(20, min_periods=20).mean()) / \
                           df["volume"].rolling(20, min_periods=20).std()

        feat_cols = [
            "ret_close","ret_open","ret_high","ret_low",
            "hl_range","tr","vol20","log_vol","vol_z20"
        ]

        base = df[feat_cols].copy().replace([np.inf, -np.inf], np.nan).dropna()
        valid_index = base.index

        # Align raw OHLCV and dates
        raw_ohlcv_all = raw_ohlcv_all.loc[valid_index]
        dates = df.loc[valid_index, "_dt"]

        # Rolling z-score
        Z = rolling_zscore_df(base, self.rolling_window)
        Z = Z.replace([np.inf, -np.inf], np.nan).dropna()
        final_index = Z.index

        X = Z.to_numpy(dtype=np.float32)
        raw_ohlcv = raw_ohlcv_all.loc[final_index].reset_index(drop=True).to_numpy(dtype=np.float32)
        dates = dates.loc[final_index].reset_index(drop=True)

        # Keep only the last span needed to form one prediction sample
        needed = self.seq_len + self.label_len  # scaffold only; no future available
        if len(X) < needed:
            raise ValueError("Not enough data to build the prediction sample with the requested sizes.")

        start = len(X) - needed
        self.data_x = X[start:]
        self.ohlcv  = raw_ohlcv[start:]
        self.dates  = dates[start:]

    def __len__(self):
        # single rolling sample at the end
        return len(self.data_x) - self.seq_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end   = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end   = r_begin + self.label_len  # scaffold only (no future available here)

        x_context = self.data_x[s_begin:s_end]
        x_target  = self.data_x[r_begin:r_end]        # scaffold

        ohlcv_context = self.ohlcv[s_begin:s_end]
        ohlcv_target  = self.ohlcv[r_begin:r_end]     # scaffold

        return {
            "x_context": torch.from_numpy(x_context).float(),
            "x_target": torch.from_numpy(x_target).float(),
            "ohlcv_context": torch.from_numpy(ohlcv_context).float(),
            "ohlcv_target": torch.from_numpy(ohlcv_target).float(),
        }