import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

from config.config_utils import load_json_config, resolve_config_path
from train_jepa_initial import main as train_jepa_main
from train_PPO_initial import main as train_ppo_main

DEFAULT_CONFIG_PATH = "configs/single_asset_run.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_ticker_list(path: str | None) -> list[str]:
    if not path:
        return []
    ticker_path = resolve_config_path(path, "", __file__)
    if not ticker_path.exists():
        return []
    tickers: list[str] = []
    seen: set[str] = set()
    with ticker_path.open("r", encoding="utf-8") as f:
        for line in f:
            ticker = line.strip()
            if not ticker or ticker in seen:
                continue
            tickers.append(ticker)
            seen.add(ticker)
    return tickers


def _discover_assets_from_dataset(dataset_cfg: dict[str, Any]) -> list[str]:
    project_root = Path(__file__).resolve().parents[1]
    root = Path(dataset_cfg["root_path"])
    if not root.is_absolute():
        root = project_root / root
    data_dir = root / dataset_cfg["data_path"]
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    assets = sorted({p.stem for p in data_dir.glob("*.parquet")})
    if not assets:
        raise ValueError(f"No parquet files found under: {data_dir}")
    return assets


def _resolve_assets(cfg: dict[str, Any], ppo_cfg: dict[str, Any]) -> list[str]:
    assets_cfg = cfg.get("assets", {})

    explicit = assets_cfg.get("tickers")
    if explicit:
        return sorted({str(t).strip() for t in explicit if str(t).strip()})

    from_file = _load_ticker_list(assets_cfg.get("ticker_list_path"))
    if from_file:
        return from_file

    if assets_cfg.get("discover_from_dataset", True):
        return _discover_assets_from_dataset(ppo_cfg["dataset"])

    raise ValueError(
        "No assets resolved. Provide assets.tickers, assets.ticker_list_path, "
        "or set assets.discover_from_dataset=true."
    )


def main(config_path: str | None = None, stage: str = "all") -> None:
    cfg = load_json_config(config_path, DEFAULT_CONFIG_PATH, __file__)

    base_jepa_cfg = load_json_config(cfg["jepa_config"], "", __file__)
    base_ppo_cfg = load_json_config(cfg["ppo_config"], "", __file__)
    ppo_feature_mode = str(base_ppo_cfg.get("ppo", {}).get("feature_mode", "jepa")).strip().lower()
    if ppo_feature_mode not in {"jepa", "basic"}:
        raise ValueError("ppo.feature_mode must be one of: jepa, basic")
    stage = str(stage).lower()
    if stage not in {"all", "jepa", "ppo"}:
        raise ValueError("stage must be one of: all, jepa, ppo")

    out_cfg = cfg.get("output", {})
    run_name = str(out_cfg.get("run_name", "single_asset"))
    generated_cfg_dir = resolve_config_path(
        out_cfg.get("generated_config_dir"),
        f"configs/generated/{run_name}",
        __file__,
    )
    checkpoint_root = Path(out_cfg.get("checkpoint_root", "checkpoints/single_asset"))
    log_root = Path(out_cfg.get("log_root", "logs/single_asset"))
    run_checkpoint_root = checkpoint_root / run_name
    run_log_root = log_root / run_name
    run_checkpoint_root.mkdir(parents=True, exist_ok=True)
    run_log_root.mkdir(parents=True, exist_ok=True)

    jepa_model_name = f"{run_name}_jepa"
    jepa_cfg = copy.deepcopy(base_jepa_cfg)
    jepa_cfg["model_name"] = jepa_model_name
    jepa_cfg.setdefault("paths", {})
    jepa_cfg["paths"]["checkpoint_root"] = str(run_checkpoint_root)
    jepa_cfg["paths"]["log_root"] = str(run_log_root)
    jepa_cfg_path_out = Path(generated_cfg_dir) / "jepa_pretrain.json"
    _write_json(jepa_cfg_path_out, jepa_cfg)

    jepa_ckpt_override = cfg.get("shared", {}).get("jepa_checkpoint_path")
    jepa_checkpoint = (
        Path(jepa_ckpt_override)
        if jepa_ckpt_override
        else run_checkpoint_root / jepa_model_name / "best.pt"
    )

    if stage in {"all", "jepa"}:
        if ppo_feature_mode == "basic" and stage == "all":
            print("[single-asset] Skipping JEPA pretraining (ppo.feature_mode=basic).")
        else:
            print(f"[single-asset] Pretraining JEPA once: {jepa_cfg_path_out}")
            train_jepa_main(str(jepa_cfg_path_out))

    if stage in {"all", "ppo"}:
        if ppo_feature_mode == "jepa" and not jepa_checkpoint.exists():
            raise FileNotFoundError(
                f"Shared JEPA checkpoint not found: {jepa_checkpoint}. "
                "Run stage=all/jepa first or set shared.jepa_checkpoint_path."
            )

        assets = _resolve_assets(cfg, base_ppo_cfg)
        if not assets:
            raise ValueError("Asset list is empty.")
        print(f"[single-asset] Training PPO models for {len(assets)} assets")

        skip_existing = bool(cfg.get("training", {}).get("skip_existing", False))
        ppo_cfg_dir = Path(generated_cfg_dir) / "ppo"
        manifest_rows: list[dict[str, Any]] = []

        for idx, asset in enumerate(assets):
            asset_tag = str(asset).upper()
            model_name = f"{run_name}_ppo_{asset_tag}"
            print(f"[single-asset] [{idx+1}/{len(assets)}] {asset_tag}")

            ppo_cfg = copy.deepcopy(base_ppo_cfg)
            ppo_cfg["model_name"] = model_name
            ppo_cfg.setdefault("paths", {})
            ppo_cfg["paths"]["checkpoint_root"] = str(run_checkpoint_root)
            ppo_cfg["paths"]["log_root"] = str(run_log_root)
            if ppo_feature_mode == "jepa":
                ppo_cfg["paths"]["jepa_checkpoint_path"] = str(jepa_checkpoint)
            else:
                ppo_cfg["paths"]["jepa_checkpoint_path"] = None
                ppo_cfg["paths"]["jepa_checkpoint_dir"] = None
            if "ticker_list_path" in ppo_cfg["paths"]:
                ppo_cfg["paths"]["ticker_list_path"] = None
            ppo_cfg.setdefault("dataset", {})
            ppo_cfg["dataset"]["tickers"] = [asset_tag]

            asset_cfg_path = ppo_cfg_dir / f"{asset_tag}_ppo.json"
            _write_json(asset_cfg_path, ppo_cfg)

            best_model_path = run_checkpoint_root / model_name / "best_model.zip"
            status = "trained"
            if skip_existing and best_model_path.exists():
                status = "skipped_existing"
                print(f"[single-asset] Skipping {asset_tag}; existing {best_model_path}")
            else:
                train_ppo_main(str(asset_cfg_path))

            manifest_rows.append(
                {
                    "asset": asset_tag,
                    "model_name": model_name,
                    "config_path": str(asset_cfg_path),
                    "best_model_path": str(best_model_path),
                    "status": status,
                }
            )

        manifest_json = run_log_root / f"{run_name}_asset_runs.json"
        manifest_csv = run_log_root / f"{run_name}_asset_runs.csv"
        _write_json(manifest_json, {"run_name": run_name, "rows": manifest_rows})
        _write_csv(manifest_csv, manifest_rows)
        print(f"[single-asset] Manifest written: {manifest_json}")
        print(f"[single-asset] Manifest written: {manifest_csv}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-asset PPO training with one shared pretrained JEPA encoder"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to single-asset run config")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "jepa", "ppo"],
        help="Run full pipeline, JEPA only, or PPO-only per-asset sweep",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(config_path=args.config, stage=args.stage)
