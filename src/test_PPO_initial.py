import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from config.config_utils import load_json_config
from Datasets.multi_asset_dataset import Dataset_Finance_MultiAsset
from Training.sb3_jepa_ppo import JEPAAuxFeatureExtractor, PPOWithJEPA
from models.jepa.jepa import JEPA
from models.time_series.patchTransformer import PatchTSTEncoder

DEFAULT_PPO_CONFIG_PATH = "configs/ppo_jepa_train.json"


def _build_dataset_kwargs(cfg: dict, split: str) -> dict:
    dataset_cfg = cfg["dataset"]
    return {
        "root_path": dataset_cfg["root_path"],
        "data_path": dataset_cfg["data_path"],
        "start_date": dataset_cfg.get("start_date"),
        "split": split,
        "size": [dataset_cfg["context_len"], dataset_cfg["target_len"]],
        "use_time_features": dataset_cfg.get("use_time_features", True),
        "rolling_window": dataset_cfg["rolling_window"],
        "train_split": dataset_cfg["train_split"],
        "test_split": dataset_cfg["test_split"],
        "regular_hours_only": dataset_cfg.get("regular_hours_only", True),
        "timeframe": dataset_cfg.get("timeframe", "15min"),
        "train_start_date": dataset_cfg.get("train_start_date"),
        "train_end_date": dataset_cfg.get("train_end_date"),
        "val_end_date": dataset_cfg.get("val_end_date"),
        "test_end_date": dataset_cfg.get("test_end_date"),
    }


def load_tickers(path: str | None) -> list | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]
    return tickers or None


def load_asset_universe_from_checkpoint(path: str | None) -> list | None:
    if not path or not os.path.exists(path):
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        return None
    asset_universe = checkpoint.get("asset_universe")
    return list(asset_universe) if asset_universe else None


@dataclass
class EvalConfig:
    annual_trading_days: int = 252
    regular_hours_only: bool = True
    timeframe: str = "15min"
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


def action_to_weight(action, action_mode: str, allow_short: bool) -> float:
    if action_mode == "discrete_3":
        discrete_actions = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        idx = int(np.asarray(action).reshape(-1)[0])
        idx = int(np.clip(idx, 0, len(discrete_actions) - 1))
        w_t = float(discrete_actions[idx])
    else:
        w_t = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
    if not allow_short:
        w_t = max(0.0, w_t)
    return w_t


def build_jepa_model(device: str, num_assets: int, jepa_cfg: dict, checkpoint_path: str) -> JEPA:
    jepa_context_encoder = PatchTSTEncoder(
        patch_len=jepa_cfg["patch_len"],
        d_model=jepa_cfg["d_model"],
        n_features=jepa_cfg["n_features"],
        n_time_features=jepa_cfg["n_time_features"],
        nhead=jepa_cfg["nhead"],
        num_layers=jepa_cfg["num_layers"],
        dim_ff=jepa_cfg["dim_ff"],
        dropout=jepa_cfg["dropout"],
        add_cls=jepa_cfg.get("add_cls", True),
        pooling=jepa_cfg["pooling"],
        pred_len=jepa_cfg["pred_len"],
        num_assets=num_assets if jepa_cfg.get("use_asset_embeddings", False) else None,
    )

    jepa_target_encoder = PatchTSTEncoder(
        patch_len=jepa_cfg["patch_len"],
        d_model=jepa_cfg["d_model"],
        n_features=jepa_cfg["n_features"],
        n_time_features=jepa_cfg["n_time_features"],
        nhead=jepa_cfg["nhead"],
        num_layers=jepa_cfg["num_layers"],
        dim_ff=jepa_cfg["dim_ff"],
        dropout=jepa_cfg["dropout"],
        add_cls=jepa_cfg.get("add_cls", True),
        pooling=jepa_cfg["pooling"],
        pred_len=jepa_cfg["pred_len"],
        num_assets=num_assets if jepa_cfg.get("use_asset_embeddings", False) else None,
    )

    jepa_model = JEPA(
        jepa_context_encoder,
        jepa_target_encoder,
        d_model=jepa_cfg["d_model"],
        ema_tau_min=jepa_cfg["ema_tau_min"],
        ema_tau_max=jepa_cfg["ema_tau_max"],
    )

    if os.path.exists(checkpoint_path):
        print(f"Loading JEPA weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        missing, unexpected = jepa_model.load_state_dict(checkpoint["model"], strict=False)
        if missing:
            print(f"Missing keys in checkpoint: {missing}")
        if unexpected:
            print(f"Unexpected keys in checkpoint: {unexpected}")
    else:
        raise FileNotFoundError(f"JEPA checkpoint not found: {checkpoint_path}")

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
    include_wealth: bool,
    include_asset_id: bool,
    obs_space_keys: set[str],
    transaction_cost: float,
    allow_short: bool,
    action_mode: str,
) -> Tuple[Dict[str, float], List[Dict[str, float | str | int]]]:
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
    action_rows: List[Dict[str, float | str | int]] = []

    for cursor in range(n_steps):
        x_context = X[cursor : cursor + seq_len].astype(np.float32)
        t_context = dates[cursor : cursor + seq_len].astype(np.float32)
        x_target = X[cursor + seq_len : cursor + seq_len + pred_len].astype(np.float32)
        t_target = dates[cursor + seq_len : cursor + seq_len + pred_len].astype(np.float32)

        obs = {}
        if "x_context" in obs_space_keys:
            obs["x_context"] = x_context
        if "t_context" in obs_space_keys:
            obs["t_context"] = t_context
        if "x_target" in obs_space_keys:
            obs["x_target"] = x_target
        if "t_target" in obs_space_keys:
            obs["t_target"] = t_target
        if "w_prev" in obs_space_keys:
            obs["w_prev"] = np.array([w_prev], dtype=np.float32)
        if include_asset_id and "asset_id" in obs_space_keys:
            obs["asset_id"] = np.int64(asset_idx)
        if include_wealth and "wealth_feats" in obs_space_keys:
            obs["wealth_feats"] = np.array([np.log(wealth)], dtype=np.float32)

        action, _ = model.predict(obs, deterministic=True)
        raw_action = float(np.asarray(action).reshape(-1)[0])
        w_t = action_to_weight(action, action_mode=action_mode, allow_short=allow_short)

        close_t = float(ohlcv[cursor + seq_len - 1][3])
        close_tp1 = float(ohlcv[cursor + seq_len][3])
        r_tp1 = float(np.log(close_tp1 / close_t))

        turnover = abs(w_t - w_prev)
        reward = w_t * r_tp1 - transaction_cost * turnover
        wealth *= float(np.exp(reward))

        rewards.append(reward)
        asset_returns.append(r_tp1)
        positions.append(w_t)
        turnovers.append(turnover)
        equity.append(wealth)
        action_rows.append(
            {
                "asset_id": asset_id,
                "step": int(cursor),
                "raw_action": raw_action,
                "position": float(w_t),
                "prev_position": float(w_prev),
                "turnover": float(turnover),
                "reward": float(reward),
                "wealth": float(wealth),
                "asset_log_return": float(r_tp1),
            }
        )

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

    return (
        {
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
        },
        action_rows,
    )


def main(config_path: str | None = None):
    cfg = load_json_config(config_path, DEFAULT_PPO_CONFIG_PATH, __file__)

    model_name = cfg["model_name"]
    paths_cfg = cfg["paths"]
    dataset_cfg = cfg["dataset"]
    env_cfg = cfg["env"]
    jepa_cfg = cfg["jepa_model"]
    test_cfg = cfg.get("test", {})

    checkpoint_root = paths_cfg.get("checkpoint_root", "checkpoints")
    log_root = paths_cfg.get("log_root", "logs")
    split = test_cfg.get("split", "test")

    ppo_checkpoint_path = test_cfg.get("ppo_checkpoint_path") or os.path.join(
        checkpoint_root, model_name, "best_model.zip"
    )
    jepa_checkpoint_path = (
        test_cfg.get("jepa_checkpoint_path")
        or paths_cfg.get("jepa_checkpoint_path")
        or os.path.join(paths_cfg["jepa_checkpoint_dir"], "best.pt")
    )

    print("Loading test dataset...")
    dataset_kwargs = _build_dataset_kwargs(cfg, split=split)

    asset_universe = load_asset_universe_from_checkpoint(jepa_checkpoint_path)
    if not asset_universe:
        asset_universe = load_tickers(paths_cfg.get("asset_universe_path"))
    if asset_universe:
        dataset_kwargs["asset_universe"] = asset_universe

    tickers = dataset_cfg.get("tickers")
    if not tickers:
        tickers = load_tickers(test_cfg.get("ticker_list_path") or paths_cfg.get("ticker_list_path"))
    if tickers:
        print(f"Using tickers: {tickers}")
        dataset_kwargs["tickers"] = tickers

    test_dataset = Dataset_Finance_MultiAsset(**dataset_kwargs)
    if not test_dataset.asset_ids:
        raise RuntimeError("No assets found in the test dataset.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading JEPA model...")
    num_asset_ids = int(getattr(test_dataset, "num_asset_ids", len(test_dataset.asset_ids)))
    jepa_model = build_jepa_model(device, num_assets=num_asset_ids, jepa_cfg=jepa_cfg, checkpoint_path=jepa_checkpoint_path)

    policy_kwargs = dict(
        features_extractor_class=JEPAAuxFeatureExtractor,
        features_extractor_kwargs=dict(
            jepa_model=jepa_model,
            embedding_dim=jepa_cfg["d_model"],
            patch_len=jepa_cfg["patch_len"],
            patch_stride=jepa_cfg["patch_stride"],
            use_obs_targets=True,
            target_len=test_dataset.pred_len,
            jepa_loss_type=cfg.get("ppo", {}).get("jepa_loss_type", "mse"),
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    print(f"Loading PPO model from {ppo_checkpoint_path}...")
    model = load_ppo_model(ppo_checkpoint_path, device=device, policy_kwargs=policy_kwargs)
    model.policy.eval()

    eval_cfg = EvalConfig(
        annual_trading_days=int(test_cfg.get("annual_trading_days", 252)),
        regular_hours_only=dataset_kwargs.get("regular_hours_only", True),
        timeframe=dataset_kwargs.get("timeframe", "15min"),
        flat_threshold=float(test_cfg.get("flat_threshold", 1e-3)),
    )

    transaction_cost = float(env_cfg.get("transaction_cost_start", env_cfg.get("transaction_cost", 0.0)))
    allow_short = bool(env_cfg.get("allow_short", True))
    action_mode = str(env_cfg.get("action_mode", "continuous"))
    include_wealth = bool(env_cfg.get("include_wealth", True))
    include_asset_id = bool(env_cfg.get("include_asset_id", True))
    obs_space_keys = set(model.policy.observation_space.spaces.keys())

    print(f"Evaluating {len(test_dataset.asset_ids)} assets on split='{split}'...")
    results: List[Dict[str, float]] = []
    action_logs: List[Dict[str, float | str | int]] = []
    for idx, asset_id in enumerate(test_dataset.asset_ids, start=1):
        print(f"[{idx}/{len(test_dataset.asset_ids)}] Evaluating {asset_id}...")
        metrics, asset_actions = eval_asset(
            model,
            test_dataset,
            asset_id,
            eval_cfg,
            include_wealth=include_wealth,
            include_asset_id=include_asset_id,
            obs_space_keys=obs_space_keys,
            transaction_cost=transaction_cost,
            allow_short=allow_short,
            action_mode=action_mode,
        )
        if metrics:
            results.append(metrics)
            action_logs.extend(asset_actions)

    if not results:
        raise RuntimeError("No evaluation results produced.")

    df = pd.DataFrame(results).sort_values("asset_id")
    os.makedirs(log_root, exist_ok=True)
    output_prefix = test_cfg.get("output_prefix", model_name)
    out_path = os.path.join(log_root, f"{output_prefix}_{split}_metrics.csv")
    df.to_csv(out_path, index=False)

    summary = df.drop(columns=["asset_id"]).agg(["mean", "median"])
    summary_path = os.path.join(log_root, f"{output_prefix}_{split}_summary.csv")
    summary.to_csv(summary_path)

    actions_path = os.path.join(log_root, f"{output_prefix}_{split}_actions.csv")
    pd.DataFrame(action_logs).to_csv(actions_path, index=False)

    print(f"Saved per-asset metrics to {out_path}")
    print(f"Saved summary stats to {summary_path}")
    print(f"Saved per-step actions to {actions_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO+JEPA model on a dataset split")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(config_path=args.config)
