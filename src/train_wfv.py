import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd

from config.config_utils import load_json_config, resolve_config_path
from test_PPO_initial import main as test_ppo_main
from train_jepa_initial import main as train_jepa_main
from train_PPO_initial import main as train_ppo_main

DEFAULT_CONFIG_PATH = "configs/wfv_run.json"


def _load_json(path: str, caller_file: str) -> dict[str, Any]:
    return load_json_config(path, "", caller_file)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _set_window_dates(cfg: dict[str, Any], train_start, train_end, val_end, test_end) -> None:
    ds = cfg["dataset"]
    ds["train_start_date"] = pd.Timestamp(train_start).isoformat()
    ds["train_end_date"] = pd.Timestamp(train_end).isoformat()
    ds["val_end_date"] = pd.Timestamp(val_end).isoformat()
    ds["test_end_date"] = pd.Timestamp(test_end).isoformat()


def _normalize_explicit_windows(raw_windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    required = ("train_start_date", "train_end_date", "val_end_date", "test_end_date")
    for idx, w in enumerate(raw_windows):
        missing = [k for k in required if k not in w]
        if missing:
            raise ValueError(f"windows[{idx}] missing required keys: {missing}")
        ws = pd.to_datetime(w["train_start_date"], utc=True)
        we = pd.to_datetime(w["train_end_date"], utc=True)
        ve = pd.to_datetime(w["val_end_date"], utc=True)
        te = pd.to_datetime(w["test_end_date"], utc=True)
        if not (ws <= we < ve < te):
            raise ValueError(
                f"windows[{idx}] must satisfy "
                "train_start_date <= train_end_date < val_end_date < test_end_date."
            )
        windows.append(
            {
                "train_start": ws,
                "train_end": we,
                "val_end": ve,
                "test_end": te,
                "tag": str(w.get("tag", f"w{idx:03d}")),
            }
        )
    return windows


def _window_tag(window_def: dict[str, Any], idx: int) -> str:
    return str(window_def.get("tag", f"w{idx:03d}"))


def _upsert_window_manifest(manifest_path: Path, run_name: str, window_entry: dict[str, Any]) -> None:
    payload: dict[str, Any] = {"run_name": run_name, "windows": []}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("run_name") != run_name:
            payload = {"run_name": run_name, "windows": []}

    windows = payload.get("windows", [])
    windows = [w for w in windows if w.get("window") != window_entry.get("window")]
    windows.append(window_entry)
    windows.sort(key=lambda w: str(w.get("window", "")))
    payload["windows"] = windows

    _write_json(manifest_path, payload)
    pd.DataFrame(windows).to_csv(manifest_path.with_suffix(".csv"), index=False)


def main(
    config_path: str | None = None,
    window_index: int | None = None,
    stage: str = "all",
) -> None:
    cfg = load_json_config(config_path, DEFAULT_CONFIG_PATH, __file__)

    jepa_cfg_path = cfg["jepa_config"]
    ppo_cfg_path = cfg["ppo_config"]
    test_cfg_path = cfg.get("test_config", ppo_cfg_path)

    base_jepa_cfg = _load_json(jepa_cfg_path, __file__)
    base_ppo_cfg = _load_json(ppo_cfg_path, __file__)
    base_test_cfg = _load_json(test_cfg_path, __file__)

    wf_cfg = cfg["walk_forward"]
    out_cfg = cfg.get("output", {})

    run_name = out_cfg.get("run_name", "wfv_run")
    generated_cfg_dir = resolve_config_path(
        out_cfg.get("generated_config_dir"),
        f"configs/generated/{run_name}",
        __file__,
    )

    checkpoint_root = out_cfg.get("checkpoint_root", "checkpoints/wfv")
    log_root = out_cfg.get("log_root", "logs/wfv")
    run_checkpoint_root = str(Path(checkpoint_root) / run_name)
    run_log_root = str(Path(log_root) / run_name)
    Path(run_checkpoint_root).mkdir(parents=True, exist_ok=True)
    Path(run_log_root).mkdir(parents=True, exist_ok=True)

    freeze_jepa_for_ppo = bool(wf_cfg.get("freeze_jepa_for_ppo", True))
    warm_start_jepa = bool(wf_cfg.get("warm_start_jepa", True))
    warm_start_ppo = bool(wf_cfg.get("warm_start_ppo", True))
    first_window_jepa_epochs = wf_cfg.get("first_window_jepa_epochs")
    warm_start_jepa_epochs = wf_cfg.get("warm_start_jepa_epochs")
    warm_start_ppo_timesteps = wf_cfg.get("warm_start_ppo_timesteps")
    explicit_windows_cfg = wf_cfg.get("windows")
    if not explicit_windows_cfg:
        raise ValueError("walk_forward.windows is required and must contain at least one window.")
    explicit_windows = _normalize_explicit_windows(explicit_windows_cfg)
    if window_index is None:
        raise ValueError("window_index is required. Pass --window <index>.")
    if window_index < 0 or window_index >= len(explicit_windows):
        raise ValueError(
            f"window_index out of range: {window_index}. "
            f"Valid range is 0..{len(explicit_windows)-1}."
        )

    window_def = explicit_windows[window_index]
    train_start = window_def["train_start"]
    train_end = window_def["train_end"]
    val_end = window_def["val_end"]
    test_end = window_def["test_end"]
    window_tag = _window_tag(window_def, window_index)
    jepa_model_name = f"{run_name}_jepa_{window_tag}"
    ppo_model_name = f"{run_name}_ppo_{window_tag}"

    prev_jepa_ckpt: str | None = None
    prev_ppo_ckpt: str | None = None
    if window_index > 0:
        prev_def = explicit_windows[window_index - 1]
        prev_tag = _window_tag(prev_def, window_index - 1)
        prev_jepa_model_name = f"{run_name}_jepa_{prev_tag}"
        prev_ppo_model_name = f"{run_name}_ppo_{prev_tag}"
        if warm_start_jepa:
            candidate = Path(run_checkpoint_root) / prev_jepa_model_name / "best.pt"
            if not candidate.exists():
                raise FileNotFoundError(f"Previous JEPA checkpoint missing: {candidate}")
            prev_jepa_ckpt = str(candidate)
        if warm_start_ppo:
            candidate = Path(run_checkpoint_root) / prev_ppo_model_name / "best_model.zip"
            if not candidate.exists():
                raise FileNotFoundError(f"Previous PPO checkpoint missing: {candidate}")
            prev_ppo_ckpt = str(candidate)

    print(
        f"[WFV] {window_tag}: "
        f"train={pd.Timestamp(train_start)}..{pd.Timestamp(train_end)} "
        f"val_end={pd.Timestamp(val_end)} test_end={pd.Timestamp(test_end)}"
    )
    stage = str(stage).lower()
    valid_stages = {"all", "jepa", "ppo", "test"}
    if stage not in valid_stages:
        raise ValueError(f"Invalid stage '{stage}'. Use one of: {sorted(valid_stages)}")

    jepa_cfg = copy.deepcopy(base_jepa_cfg)
    jepa_cfg["model_name"] = jepa_model_name
    jepa_cfg["paths"]["checkpoint_root"] = run_checkpoint_root
    jepa_cfg["paths"]["log_root"] = run_log_root
    _set_window_dates(jepa_cfg, train_start, train_end, val_end, test_end)
    is_warm_start_window = bool(prev_jepa_ckpt and warm_start_jepa)
    jepa_cfg["resume"] = {"path": prev_jepa_ckpt if is_warm_start_window else None}
    if window_index == 0 and first_window_jepa_epochs is not None:
        jepa_cfg["training"]["epochs"] = int(first_window_jepa_epochs)
    elif is_warm_start_window and warm_start_jepa_epochs is not None:
        jepa_cfg["training"]["epochs"] = int(warm_start_jepa_epochs)
    print(
        f"[WFV] JEPA epochs ({window_tag}) = {jepa_cfg['training']['epochs']} "
        f"(warm_start={is_warm_start_window})"
    )

    ppo_cfg = copy.deepcopy(base_ppo_cfg)
    ppo_cfg["model_name"] = ppo_model_name
    ppo_cfg["paths"]["checkpoint_root"] = run_checkpoint_root
    ppo_cfg["paths"]["log_root"] = run_log_root
    _set_window_dates(ppo_cfg, train_start, train_end, val_end, test_end)
    is_warm_start_ppo_window = bool(prev_ppo_ckpt and warm_start_ppo)
    if freeze_jepa_for_ppo:
        ppo_cfg.setdefault("ppo", {})["update_jepa"] = False
    ppo_cfg["resume"] = {
        "path": prev_ppo_ckpt if is_warm_start_ppo_window else None,
        "auto_resume": False,
    }
    if is_warm_start_ppo_window and warm_start_ppo_timesteps is not None:
        effective_ppo_steps = max(1, int(warm_start_ppo_timesteps))
        ppo_cfg["ppo"]["total_timesteps"] = effective_ppo_steps
        print(
            f"[WFV] PPO warm-start timesteps ({window_tag}) = {effective_ppo_steps} "
            f"(warm_start_ppo_timesteps={effective_ppo_steps})"
        )

    jepa_cfg_path_out = Path(generated_cfg_dir) / f"{window_tag}_jepa.json"
    ppo_cfg_path_out = Path(generated_cfg_dir) / f"{window_tag}_ppo.json"
    test_cfg_path_out = Path(generated_cfg_dir) / f"{window_tag}_test.json"

    _write_json(jepa_cfg_path_out, jepa_cfg)
    window_jepa_dir = Path(run_checkpoint_root) / jepa_model_name
    current_jepa_ckpt = str(window_jepa_dir / "best.pt")
    if stage in {"all", "jepa"}:
        print(f"[WFV] Training JEPA ({window_tag})")
        train_jepa_main(str(jepa_cfg_path_out))
    if not Path(current_jepa_ckpt).exists():
        raise FileNotFoundError(
            f"Missing JEPA best checkpoint for {window_tag}: {current_jepa_ckpt}. "
            "Run JEPA stage first."
        )

    ppo_cfg["paths"]["jepa_checkpoint_path"] = current_jepa_ckpt
    _write_json(ppo_cfg_path_out, ppo_cfg)

    window_ppo_dir = Path(run_checkpoint_root) / ppo_model_name
    if stage in {"all", "ppo"}:
        print(f"[WFV] Training PPO ({window_tag})")
        train_ppo_main(str(ppo_cfg_path_out))

    current_ppo_ckpt = str(window_ppo_dir / "best_model.zip")
    if stage in {"all", "test"}:
        if not Path(current_ppo_ckpt).exists():
            raise FileNotFoundError(
                f"Missing PPO best checkpoint for {window_tag}: {current_ppo_ckpt}. "
                "Run PPO stage first."
            )

        test_cfg_local = copy.deepcopy(base_test_cfg)
        test_cfg_local["model_name"] = ppo_model_name
        test_cfg_local["paths"]["checkpoint_root"] = run_checkpoint_root
        test_cfg_local["paths"]["log_root"] = run_log_root
        test_cfg_local["paths"]["jepa_checkpoint_path"] = current_jepa_ckpt
        _set_window_dates(test_cfg_local, train_start, train_end, val_end, test_end)
        test_cfg_local["test"] = {
            "split": "test",
            "ppo_checkpoint_path": current_ppo_ckpt,
            "jepa_checkpoint_path": current_jepa_ckpt,
            "output_prefix": ppo_model_name,
        }
        _write_json(test_cfg_path_out, test_cfg_local)

        print(f"[WFV] Testing PPO ({window_tag})")
        test_ppo_main(str(test_cfg_path_out))

        window_entry = {
            "window": window_tag,
            "window_index": int(window_index),
            "train_start": pd.Timestamp(train_start).isoformat(),
            "train_end": pd.Timestamp(train_end).isoformat(),
            "val_end": pd.Timestamp(val_end).isoformat(),
            "test_end": pd.Timestamp(test_end).isoformat(),
            "jepa_checkpoint": current_jepa_ckpt,
            "ppo_checkpoint": current_ppo_ckpt,
        }
        manifest_path = Path(run_log_root) / f"{run_name}_windows.json"
        _upsert_window_manifest(manifest_path=manifest_path, run_name=run_name, window_entry=window_entry)
        print(f"[WFV] Completed {window_tag} (index={window_index}). Manifest: {manifest_path}")
    else:
        print(f"[WFV] Completed stage '{stage}' for {window_tag} (index={window_index}).")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward JEPA+PPO training/testing")
    parser.add_argument("--config", type=str, default=None, help="Path to WFV json config")
    parser.add_argument("--window", type=int, required=True, help="Window index to run")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "jepa", "ppo", "test"],
        help="Which stage to run for the selected window",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(config_path=args.config, window_index=args.window, stage=args.stage)
