# multi_asset_dataset.py
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
        return pd.to_datetime(series, unit="ms", utc=True)
    except Exception:
        return pd.to_datetime(series, utc=True)


def true_range(high, low, close):
    prev_close = close.shift(1)
    a = (high - low).abs()
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def make_time_features(dt_index: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame({"date": dt_index})
    df["weekday"] = df["date"].dt.weekday
    hour = df["date"].dt.hour
    minute = df["date"].dt.minute
    df["minute_of_day"] = (hour * 60 + minute)
    return df[["weekday", "minute_of_day"]]


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    m = series.rolling(window, min_periods=window).mean().shift(1)
    s = series.rolling(window, min_periods=window).std().shift(1)
    return (series - m) / s


def rolling_zscore_df(df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for c in df.columns:
        out[c] = rolling_zscore(df[c], window)
    return out


def filter_regular_us_hours(dt: pd.Series) -> pd.Series:
    """
    Return boolean mask for regular U.S. trading hours (09:30-16:00 ET, weekdays).
    dt should be timezone-aware (UTC).
    """
    dt_et = dt.dt.tz_convert("America/New_York")
    minutes = dt_et.dt.hour * 60 + dt_et.dt.minute
    is_weekday = dt_et.dt.weekday < 5
    in_rth = (minutes >= 570) & (minutes <= 960)
    return is_weekday & in_rth


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = df.set_index("_dt")
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = df.resample(rule).agg(agg).dropna()
    return out.reset_index()


# -----------------------------
# Multi-asset Finance Dataset
# -----------------------------
class Dataset_Finance_MultiAsset(Dataset):
    """
    Multi-asset dataset for Polygon parquet files.
    Keeps per-asset preprocessing and z-score history isolated.
    Returns the same keys as Dataset_Finance.
    """
    def __init__(
        self,
        root_path: str,
        data_path: str,
        start_date: str = None,
        split: str = "train",             # 'train' | 'val' | 'test'
        size=None,                        # [seq_len, pred_len]
        use_time_features: bool = True,   # kept for parity (returned)
        train_split: float = 0.7,
        test_split: float = 0.15,
        rolling_window: int = 252,
        tickers: list = None,
        asset_universe: list | None = None,
        file_suffix: str = ".parquet",
        regular_hours_only: bool = True,
        timeframe: str = "1min",
        log_splits: bool = True,
    ):
        assert split in ["train", "val", "test"]
        self.set_type = {"train": 0, "val": 1, "test": 2}[split]

        if size is None:
            self.seq_len = 96
            self.pred_len = 24
        else:
            self.seq_len, self.pred_len = size

        self.root_path = root_path
        self.data_path = data_path
        self.use_time_features = use_time_features
        self.train_split = train_split
        self.test_split = test_split
        self.rolling_window = rolling_window
        self.start_date = start_date
        self.tickers = tickers
        self.asset_universe = list(asset_universe) if asset_universe is not None else None
        self.file_suffix = file_suffix
        self.regular_hours_only = regular_hours_only
        self.timeframe = timeframe
        self.log_splits = log_splits

        self.asset_ids = []
        self.asset_id_to_idx = {}
        self.data_x = {}
        self.ohlcv = {}
        self.dates = {}
        self.index_registry = []
        self.split_info = {}
        self.num_asset_ids = 0

        self._read_and_build()
        if self.asset_universe is None:
            self.asset_universe = list(self.asset_ids)
        else:
            seen = set()
            deduped = []
            for asset_id in self.asset_universe:
                if asset_id not in seen:
                    deduped.append(asset_id)
                    seen.add(asset_id)
            self.asset_universe = deduped
            universe_set = set(self.asset_universe)
            unknown_assets = [asset_id for asset_id in self.asset_ids if asset_id not in universe_set]
            if unknown_assets:
                raise ValueError(
                    f"Loaded assets not present in asset_universe: {unknown_assets}"
                )

        self.asset_id_to_idx = {asset_id: idx for idx, asset_id in enumerate(self.asset_universe)}
        self.num_asset_ids = len(self.asset_universe)

    def _load_asset(self, path: str = None, df: pd.DataFrame = None):
        if df is None:
            df = pd.read_parquet(path)
        time_col = "timestamp"

        if self.start_date is not None:
            start_dt = pd.to_datetime(self.start_date, utc=True)
            dt = to_datetime(df[time_col])
            df = df.loc[dt >= start_dt].copy()

        dt = to_datetime(df[time_col])
        if self.regular_hours_only:
            mask = filter_regular_us_hours(dt)
            df = df.loc[mask].copy()
            dt = dt.loc[mask]
        df = df.assign(_dt=dt).sort_values("_dt", ascending=True).reset_index(drop=True)

        if self.timeframe != "1min":
            df = resample_ohlcv(df, self.timeframe)

        if self.use_time_features:
            time_features = make_time_features(df["_dt"])
        else:
            time_features = pd.DataFrame(index=df.index)

        ohlcv_cols = ["open", "high", "low", "close", "volume", "_dt"]
        
        raw_ohlcv = df[ohlcv_cols].copy()

        for col in ["open", "high", "low", "close"]:
            df[f"log_{col}"] = np.log(df[col].astype(float))

        df["ret_close"] = df["log_close"].diff()
        df["ret_open"] = df["log_open"].diff()
        df["ret_high"] = df["log_high"].diff()
        df["ret_low"] = df["log_low"].diff()

        df["hl_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
        df["tr"] = true_range(df["high"], df["low"], df["close"]) / df["close"].replace(0, np.nan)
        df["vol20"] = df["ret_close"].rolling(20, min_periods=20).std()

        df["log_vol"] = np.log(df["volume"].astype(float) + 1.0)
        df["vol_z20"] = (df["volume"] - df["volume"].rolling(20, min_periods=20).mean()) / \
                        df["volume"].rolling(20, min_periods=20).std()

        feat_cols = [
            "ret_close", "ret_open", "ret_high", "ret_low",
            "hl_range", "tr", "vol20", "log_vol", "vol_z20"
        ]

        base = df[feat_cols].copy().replace([np.inf, -np.inf], np.nan)
        base_valid = base.dropna()
        valid_index = base_valid.index

        Z = rolling_zscore_df(base.loc[valid_index], self.rolling_window)
        Z = Z.replace([np.inf, -np.inf], np.nan).dropna()
        final_index = Z.index

        X = Z.to_numpy(dtype=np.float32)
        raw_ohlcv = raw_ohlcv.loc[final_index].reset_index(drop=True).to_numpy(dtype=np.float32)
        dt_final = df.loc[final_index, "_dt"].reset_index(drop=True)

        if self.use_time_features:
            dates = time_features.loc[final_index].reset_index(drop=True).to_numpy(dtype=np.float32)
        else:
            dates = np.empty((len(X), 0), dtype=np.float32)

        

        return X, raw_ohlcv, dates, dt_final

    def _iter_asset_files(self):
        path = os.path.join(self.root_path, self.data_path)
        if self.tickers:
            for t in self.tickers:
                yield t, os.path.join(path, f"{t}{self.file_suffix}")
        else:
            for fname in sorted(os.listdir(path)):
                if fname.endswith(self.file_suffix):
                    asset_id = os.path.splitext(fname)[0]
                    yield asset_id, os.path.join(path, fname)

    def _read_and_build(self):
        asset_cache = []
        global_dates = []

        for asset_id, fpath in self._iter_asset_files():
            if not os.path.exists(fpath):
                continue
            df = pd.read_parquet(fpath)
            X, raw_ohlcv, dates, dt_final = self._load_asset(df=df)
            if len(X) < (self.seq_len + self.pred_len + 1):
                continue
            asset_cache.append((asset_id, X, raw_ohlcv, dates, dt_final))
            global_dates.append(dt_final)

        if not asset_cache:
            return

        global_dates = pd.Index(pd.concat(global_dates).unique()).sort_values()
        n_global = len(global_dates)
        if n_global == 0:
            return

        num_train = int(n_global * self.train_split)
        num_test = int(n_global * self.test_split)
        num_val = n_global - num_train - num_test

        train_end = global_dates[max(num_train - 1, 0)]
        val_end = global_dates[max(num_train + num_val - 1, 0)]

        self.split_info = {
            "global_total_rows": n_global,
            "train_end": train_end,
            "val_end": val_end,
            "num_train_dates": num_train,
            "num_val_dates": num_val,
            "num_test_dates": num_test,
        }
        if self.log_splits:
            print(
                "[Dataset_Finance_MultiAsset] Global date splits:",
                f"train_end={train_end}",
                f"val_end={val_end}",
                f"n_dates={n_global}",
                f"n_train={num_train}",
                f"n_val={num_val}",
                f"n_test={num_test}",
            )

        for asset_id, X, raw_ohlcv, dates, dt_final in asset_cache:
            if self.set_type == 0:
                mask = dt_final <= train_end
            elif self.set_type == 1:
                mask = (dt_final > train_end) & (dt_final <= val_end)
            else:
                mask = dt_final > val_end

            mask = mask.to_numpy()
            X_split = X[mask]
            ohlcv_split = raw_ohlcv[mask]
            dates_split = dates[mask]

            if len(X_split) < (self.seq_len + self.pred_len):
                continue

            self.asset_ids.append(asset_id)
            self.data_x[asset_id] = X_split
            self.ohlcv[asset_id] = ohlcv_split
            self.dates[asset_id] = dates_split

            max_start = len(X_split) - self.seq_len - self.pred_len + 1
            for i in range(max_start):
                self.index_registry.append((asset_id, i))

    def __len__(self):
        return len(self.index_registry)

    def __getitem__(self, index):
        asset_id, i = self.index_registry[index]
        asset_idx = self.asset_id_to_idx.get(asset_id, -1)

        s_begin = i
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = s_end + self.pred_len

        x_context = self.data_x[asset_id][s_begin:s_end]
        x_target = self.data_x[asset_id][r_begin:r_end]

        ohlcv_context = self.ohlcv[asset_id][s_begin:s_end]
        ohlcv_target = self.ohlcv[asset_id][r_begin:r_end]

        dates_context = self.dates[asset_id][s_begin:s_end]
        dates_target = self.dates[asset_id][r_begin:r_end]

        return {
            "x_context": torch.from_numpy(x_context).float(),
            "x_target": torch.from_numpy(x_target).float(),
            "t_context": torch.from_numpy(dates_context).float(),
            "t_target": torch.from_numpy(dates_target).float(),
            "ohlcv_context": torch.from_numpy(ohlcv_context).float(),
            "ohlcv_target": torch.from_numpy(ohlcv_target).float(),
            "asset_id": torch.tensor(asset_idx, dtype=torch.long),
        }


