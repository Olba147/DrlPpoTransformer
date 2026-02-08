import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from Datasets.multi_asset_dataset import Dataset_Finance_MultiAsset
from Training.sb3_jepa_ppo import JEPAAuxFeatureExtractor, PPOWithJEPA
from models.jepa.jepa import JEPA
from models.time_series.patchTransformer import PatchTSTEncoder
from train_jepa_initial import (
    PATCH_LEN,
    PATCH_STRIDE,
    JEPA_D_MODEL,
    JEPA_N_FEATURES,
    JEPA_N_TIME_FEATURES,
    JEPA_NHEAD,
    JEPA_NUM_LAYERS,
    JEPA_DIM_FF,
    JEPA_DROPOUT,
    JEPA_POOLING,
    JEPA_PRED_LEN,
    EMA_START,
    EMA_END,
    EMA_EPOCHS,
)

MODEL_NAME = "jepa_ppo3_emptry_start"
PPO_CHECKPOINT_PATH = f"checkpoints/{MODEL_NAME}/ppo_5920000_steps"
JEPA_CHECKPOINT_PATH = f"checkpoints/{MODEL_NAME}/jepa_step_370000.pt"
TICKER_LIST_PATH = "logs/selected_tickers.txt"

# ------------------------
# Dataset + Env settings
# ------------------------
EPISODE_LENGTH_STEPS = 2048
TRANSACTION_COST = 1e-5
ALLOW_SHORT = True
INCLUDE_WEALTH = False
ASSET_EMBED_DIM = 16

dataset_kwargs = {
    "root_path": r"Data/polygon",
    "data_path": r"data_raw_1m",
    "start_date": None,
    "split": "test",
    "size": [1024, 48],
    "use_time_features": True,
    "rolling_window": 252,
    "train_split": 0.7,
    "test_split": 0.15,
    "regular_hours_only": True,
    "timeframe": "5min",
}

def load_tickers(path: str) -> list | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]
    return tickers or None


@dataclass
class EvalConfig:
    annual_trading_days: int = 252
    regular_hours_only: bool = True
    timeframe: str = "5min"
    flat_threshold: float = 1e-3


def _timeframe_to_minutes(timeframe: str) -> int:
    tf = timeframe.strip().lower()
    if tf.endswith("min"):
        return int(tf[:-3])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def annualization_factor(cfg: EvalConfig) -> float:
    minutes_per_day = 390 if cfg.regular_hours_only else 24 * 60
    minutes = _timeframe_to_minutes(cfg.timeframe)
    bars_per_day = max(1, minutes_per_day // minutes)
    return bars_per_day * cfg.annual_trading_days


def build_jepa_model(device: str, num_assets: int) -> JEPA:
    jepa_context_encoder = PatchTSTEncoder(
        patch_len=PATCH_LEN,
        d_model=JEPA_D_MODEL,
        n_features=JEPA_N_FEATURES,
        n_time_features=JEPA_N_TIME_FEATURES,
        nhead=JEPA_NHEAD,
        num_layers=JEPA_NUM_LAYERS,
        dim_ff=JEPA_DIM_FF,
        dropout=JEPA_DROPOUT,
        add_cls=True,
        pooling=JEPA_POOLING,
        pred_len=JEPA_PRED_LEN,
        num_assets=num_assets,
    )

    jepa_target_encoder = PatchTSTEncoder(
        patch_len=PATCH_LEN,
        d_model=JEPA_D_MODEL,
        n_features=JEPA_N_FEATURES,
        n_time_features=JEPA_N_TIME_FEATURES,
        nhead=JEPA_NHEAD,
        num_layers=JEPA_NUM_LAYERS,
        dim_ff=JEPA_DIM_FF,
        dropout=JEPA_DROPOUT,
        add_cls=True,
        pooling=JEPA_POOLING,
        pred_len=JEPA_PRED_LEN,
        num_assets=num_assets,
    )

    jepa_model = JEPA(
        jepa_context_encoder,
        jepa_target_encoder,
        d_model=JEPA_D_MODEL,
        ema_start=EMA_START,
        ema_end=EMA_END,
        n_epochs=EMA_EPOCHS,
    )

    if os.path.exists(JEPA_CHECKPOINT_PATH):
        print(f"Loading JEPA weights from {JEPA_CHECKPOINT_PATH}")
        checkpoint = torch.load(JEPA_CHECKPOINT_PATH, map_location="cpu")
        missing, unexpected = jepa_model.load_state_dict(checkpoint["model"], strict=False)
        if missing:
            print(f"Missing keys in checkpoint: {missing}")
        if unexpected:
            print(f"Unexpected keys in checkpoint: {unexpected}")
    else:
        print("No JEPA checkpoint found, using randomly initialized encoder.")

    for param in jepa_model.parameters():
        param.requires_grad = False
    jepa_model.eval()
    return jepa_model.to(device)


def load_ppo_model(model_path: str, device: str, policy_kwargs: Dict) -> PPOWithJEPA:
    try:
        return PPOWithJEPA.load(model_path, device=device)
    except Exception as exc:
        print(f"Primary PPO load failed ({exc}); retrying with custom policy_kwargs.")
        return PPOWithJEPA.load(model_path, device=device, custom_objects={"policy_kwargs": policy_kwargs})


def compute_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return float(np.min(drawdown))


def safe_sharpe(mean: float, std: float, ann_factor: float) -> float:
    if std <= 0 or np.isnan(std):
        return float("nan")
    return float(mean / std * np.sqrt(ann_factor))


def eval_asset(
    model: PPOWithJEPA,
    dataset: Dataset_Finance_MultiAsset,
    asset_id: str,
    cfg: EvalConfig,
) -> Dict[str, float]:
    asset_idx = dataset.asset_id_to_idx.get(asset_id, -1)
    X = dataset.data_x[asset_id]
    dates = dataset.dates[asset_id]
    ohlcv = dataset.ohlcv[asset_id]

    seq_len = dataset.seq_len
    pred_len = dataset.pred_len
    n_steps = len(X) - seq_len - pred_len
    if n_steps <= 0:
        return {}

    w_prev = 0
    wealth = 1.0
    rewards = []
    asset_returns = []
    positions = []
    turnovers = []
    equity = []

    for cursor in range(n_steps):
        x_context = X[cursor : cursor + seq_len].astype(np.float32)
        t_context = dates[cursor : cursor + seq_len].astype(np.float32)
        x_target = X[cursor + seq_len : cursor + seq_len + pred_len].astype(np.float32)
        t_target = dates[cursor + seq_len : cursor + seq_len + pred_len].astype(np.float32)

        obs = {
            "x_context": x_context,
            "t_context": t_context,
            "x_target": x_target,
            "t_target": t_target,
            "asset_id": np.int64(asset_idx),
            "w_prev": np.array([w_prev], dtype=np.float32),
        }
        if INCLUDE_WEALTH:
            obs["wealth_feats"] = np.array([np.log(wealth)], dtype=np.float32)

        action, _ = model.predict(obs, deterministic=True)
        w_t = float(np.clip(np.array(action).reshape(-1)[0], -1.0, 1.0))
        if not ALLOW_SHORT:
            w_t = max(0.0, w_t)

        close_t = float(ohlcv[cursor + seq_len - 1][3])
        close_tp1 = float(ohlcv[cursor + seq_len][3])
        r_tp1 = float(np.log(close_tp1 / close_t))

        turnover = abs(w_t - w_prev)
        reward = w_t * r_tp1 - TRANSACTION_COST * turnover
        wealth *= float(np.exp(reward))

        rewards.append(reward)
        asset_returns.append(r_tp1)
        positions.append(w_t)
        turnovers.append(turnover)
        equity.append(wealth)

        w_prev = w_t

    rewards = np.asarray(rewards, dtype=np.float64)
    asset_returns = np.asarray(asset_returns, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    turnovers = np.asarray(turnovers, dtype=np.float64)
    equity = np.asarray(equity, dtype=np.float64)

    ann_factor = annualization_factor(cfg)
    mean_reward = float(np.mean(rewards)) if rewards.size else float("nan")
    std_reward = float(np.std(rewards, ddof=1)) if rewards.size > 1 else float("nan")

    total_log_return = float(np.sum(rewards)) if rewards.size else float("nan")
    total_return = float(np.exp(total_log_return) - 1.0) if rewards.size else float("nan")
    annualized_return = float(np.exp(mean_reward * ann_factor) - 1.0) if rewards.size else float("nan")
    annualized_vol = float(std_reward * np.sqrt(ann_factor)) if rewards.size > 1 else float("nan")
    sharpe = safe_sharpe(mean_reward, std_reward, ann_factor)

    downside = rewards[rewards < 0]
    downside_std = float(np.std(downside, ddof=1)) if downside.size > 1 else float("nan")
    sortino = safe_sharpe(mean_reward, downside_std, ann_factor)

    max_drawdown = compute_drawdown(equity)
    calmar = float(annualized_return / abs(max_drawdown)) if max_drawdown < 0 else float("nan")

    win_rate = float(np.mean(rewards > 0)) if rewards.size else float("nan")
    avg_turnover = float(np.mean(turnovers)) if turnovers.size else float("nan")
    total_turnover = float(np.sum(turnovers)) if turnovers.size else float("nan")
    avg_position = float(np.mean(positions)) if positions.size else float("nan")
    pos_std = float(np.std(positions, ddof=1)) if positions.size > 1 else float("nan")
    abs_pos = float(np.mean(np.abs(positions))) if positions.size else float("nan")

    flat_mask = np.abs(positions) <= cfg.flat_threshold
    long_mask = positions > cfg.flat_threshold
    short_mask = positions < -cfg.flat_threshold
    flat_frac = float(np.mean(flat_mask)) if positions.size else float("nan")
    long_frac = float(np.mean(long_mask)) if positions.size else float("nan")
    short_frac = float(np.mean(short_mask)) if positions.size else float("nan")

    trade_count = int(np.sum(np.abs(np.diff(positions)) > cfg.flat_threshold)) if positions.size > 1 else 0

    bh_mean = float(np.mean(asset_returns)) if asset_returns.size else float("nan")
    bh_std = float(np.std(asset_returns, ddof=1)) if asset_returns.size > 1 else float("nan")
    bh_total_return = float(np.exp(np.sum(asset_returns)) - 1.0) if asset_returns.size else float("nan")
    bh_annualized_return = float(np.exp(bh_mean * ann_factor) - 1.0) if asset_returns.size else float("nan")
    bh_annualized_vol = float(bh_std * np.sqrt(ann_factor)) if asset_returns.size > 1 else float("nan")
    bh_sharpe = safe_sharpe(bh_mean, bh_std, ann_factor)

    return {
        "asset_id": asset_id,
        "steps": int(n_steps),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "avg_reward": mean_reward,
        "reward_volatility": std_reward,
        "win_rate": win_rate,
        "avg_turnover": avg_turnover,
        "total_turnover": total_turnover,
        "avg_position": avg_position,
        "position_std": pos_std,
        "avg_abs_position": abs_pos,
        "long_frac": long_frac,
        "short_frac": short_frac,
        "flat_frac": flat_frac,
        "trade_count": trade_count,
        "bh_total_return": bh_total_return,
        "bh_annualized_return": bh_annualized_return,
        "bh_annualized_volatility": bh_annualized_vol,
        "bh_sharpe": bh_sharpe,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading test dataset...")
    tickers = load_tickers(TICKER_LIST_PATH)
    if tickers:
        print(f"Using tickers from {TICKER_LIST_PATH}: {tickers}")
        dataset_kwargs["tickers"] = tickers
    test_dataset = Dataset_Finance_MultiAsset(**dataset_kwargs)
    if not test_dataset.asset_ids:
        raise RuntimeError("No assets found in the test dataset.")

    print("Loading JEPA model...")
    jepa_model = build_jepa_model(device, num_assets=len(test_dataset.asset_ids))
    policy_kwargs = dict(
        features_extractor_class=JEPAAuxFeatureExtractor,
        features_extractor_kwargs=dict(
            jepa_model=jepa_model,
            embedding_dim=JEPA_D_MODEL,
            patch_len=PATCH_LEN,
            patch_stride=PATCH_STRIDE,
            use_obs_targets=True,
            target_len=test_dataset.pred_len,
            num_assets=len(test_dataset.asset_ids),
            asset_embed_dim=ASSET_EMBED_DIM,
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    print(f"Loading PPO model from {PPO_CHECKPOINT_PATH}...")
    model = load_ppo_model(PPO_CHECKPOINT_PATH, device=device, policy_kwargs=policy_kwargs)
    model.policy.eval()

    cfg = EvalConfig(
        annual_trading_days=252,
        regular_hours_only=dataset_kwargs.get("regular_hours_only", True),
        timeframe=dataset_kwargs.get("timeframe", "5min"),
    )

    print(f"Evaluating {len(test_dataset.asset_ids)} assets...")
    results: List[Dict[str, float]] = []
    for idx, asset_id in enumerate(test_dataset.asset_ids, start=1):
        print(f"[{idx}/{len(test_dataset.asset_ids)}] Evaluating {asset_id}...")
        metrics = eval_asset(model, test_dataset, asset_id, cfg)
        if metrics:
            results.append(metrics)

    if not results:
        raise RuntimeError("No evaluation results produced.")

    df = pd.DataFrame(results).sort_values("asset_id")
    os.makedirs("logs", exist_ok=True)
    out_path = f"logs/{MODEL_NAME}_test_metrics.csv"
    df.to_csv(out_path, index=False)

    summary = df.drop(columns=["asset_id"]).agg(["mean", "median"])
    summary_path = f"logs/{MODEL_NAME}_test_summary.csv"
    summary.to_csv(summary_path)

    print(f"Saved per-asset metrics to {out_path}")
    print(f"Saved summary stats to {summary_path}")


if __name__ == "__main__":
    main()
