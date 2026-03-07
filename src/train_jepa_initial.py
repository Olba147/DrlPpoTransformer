import argparse
import copy
import os

import torch
import torch.nn as nn

from config.config_utils import load_json_config
from Datasets.dataloaders import DataLoaders
from Datasets.multi_asset_dataset import Dataset_Finance_MultiAsset
from models.jepa.jepa import JEPA
from models.time_series.patchTransformer import PatchTSTEncoder
from Training.callbacks import CSVLogger, CheckpointCallback, PatchingCallback, StatsPrinter
from Training.engine import Learner


def _build_dataset_kwargs(cfg: dict) -> dict:
    dataset_cfg = cfg["dataset"]
    return {
        "root_path": dataset_cfg["root_path"],
        "data_path": dataset_cfg["data_path"],
        "start_date": dataset_cfg.get("start_date"),
        "split": dataset_cfg.get("split", "train"),
        "size": dataset_cfg["context_len"],
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


def _parse_horizon_blocks(horizon_blocks: dict[str, list[int]]) -> tuple[dict[str, slice], int]:
    if not horizon_blocks:
        raise ValueError("jepa_model.horizon_blocks must be a non-empty dict.")

    parsed: list[tuple[str, int, int]] = []
    for name, bounds in horizon_blocks.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"horizon_blocks[{name}] must be [start, end], got {bounds}.")
        start, end = int(bounds[0]), int(bounds[1])
        if start < 1 or end < start:
            raise ValueError(
                f"horizon_blocks[{name}] must satisfy 1 <= start <= end, got [{start}, {end}]."
            )
        parsed.append((name, start, end))

    parsed.sort(key=lambda x: x[1])
    if parsed[0][1] != 1:
        raise ValueError("horizon blocks must start at 1 and be contiguous.")

    prev_end = 0
    slices: dict[str, slice] = {}
    for name, start, end in parsed:
        if start != prev_end + 1:
            raise ValueError(
                "horizon blocks must be contiguous without gaps/overlaps; "
                f"expected start {prev_end + 1}, got {start} for block '{name}'."
            )
        slices[name] = slice(start - 1, end)
        prev_end = end
    return slices, prev_end


class MultiHorizonWeightedLoss(nn.Module):
    def __init__(
        self,
        base_loss_type: str,
        horizon_slices: dict[str, slice],
        near_weight: float = 1.0,
        med_weight: float = 1.0,
        far_weight: float = 0.5,
    ):
        super().__init__()
        loss_type = str(base_loss_type).lower()
        if loss_type == "mse":
            self.base_loss = torch.nn.MSELoss()
        elif loss_type in {"smoothl1", "smooth_l1"}:
            self.base_loss = torch.nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {base_loss_type}")
        self.horizon_slices = dict(horizon_slices)
        required = {"near", "med", "far"}
        missing = required.difference(self.horizon_slices.keys())
        if missing:
            raise ValueError(
                "MultiHorizonWeightedLoss requires near/med/far horizon blocks. "
                f"Missing: {sorted(missing)}"
            )
        self.near_weight = float(near_weight)
        self.med_weight = float(med_weight)
        self.far_weight = float(far_weight)
        self.last_components: dict[str, float] = {}

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        near_slice = self.horizon_slices["near"]
        med_slice = self.horizon_slices["med"]
        far_slice = self.horizon_slices["far"]

        loss_near = self.base_loss(pred[:, near_slice, :], target[:, near_slice, :])
        loss_med = self.base_loss(pred[:, med_slice, :], target[:, med_slice, :])
        pred_far_pooled = pred[:, far_slice, :].mean(dim=1)
        tgt_far_pooled = target[:, far_slice, :].mean(dim=1)
        loss_far = self.base_loss(pred_far_pooled, tgt_far_pooled)

        total = (
            self.near_weight * loss_near
            + self.med_weight * loss_med
            + self.far_weight * loss_far
        )
        self.last_components = {
            "loss_near": float(loss_near.detach().item()),
            "loss_med": float(loss_med.detach().item()),
            "loss_far_pooled": float(loss_far.detach().item()),
            "loss_near_weighted": float((self.near_weight * loss_near).detach().item()),
            "loss_med_weighted": float((self.med_weight * loss_med).detach().item()),
            "loss_far_weighted": float((self.far_weight * loss_far).detach().item()),
            "loss_total": float(total.detach().item()),
        }
        return total


class JEPAPretrainModel(nn.Module):
    def __init__(self, jepa_model: JEPA):
        super().__init__()
        self.jepa_model = jepa_model
        self.ema_tau_min = jepa_model.ema_tau_min
        self.ema_tau_max = jepa_model.ema_tau_max

    def forward(self, X_ctx, T_ctx, asset_id=None):
        return self.jepa_model(X_ctx, T_ctx, asset_id=asset_id)

    @torch.no_grad()
    def ema_update(self, decay):
        self.jepa_model.ema_update(decay)

    def state_dict(self, *args, **kwargs):
        return self.jepa_model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.jepa_model.load_state_dict(*args, **kwargs)


def main(config_path: str):
    cfg = load_json_config(config_path, "", __file__)

    model_name = cfg["model_name"]
    paths_cfg = cfg["paths"]
    resume_cfg = cfg.get("resume", {})
    train_cfg = cfg["training"]
    model_cfg = cfg["jepa_model"]
    loss_cfg = cfg["loss"]

    patch_len = int(model_cfg["patch_len"])
    patch_stride = int(model_cfg["patch_stride"])
    context_len_steps = int(model_cfg.get("context_len_steps", cfg["dataset"]["context_len"]))
    if context_len_steps < patch_len:
        raise ValueError(
            f"context_len_steps must be >= patch_len ({patch_len}), got {context_len_steps}."
        )
    if (context_len_steps - patch_len) % patch_stride != 0:
        raise ValueError(
            f"(context_len_steps - patch_len) must be divisible by patch_stride. "
            f"Got context_len_steps={context_len_steps}, patch_len={patch_len}, patch_stride={patch_stride}."
        )

    horizon_blocks = model_cfg.get(
        "horizon_blocks",
        {
            "near": [1, 1],
            "med": [2, 5],
            "far": [6, 17],
        },
    )
    horizon_slices, horizon_tokens = _parse_horizon_blocks(horizon_blocks)
    context_tokens = 1 + (context_len_steps - patch_len) // patch_stride
    total_tokens = context_tokens + horizon_tokens
    total_steps = (total_tokens - 1) * patch_stride + patch_len

    run_dataset_kwargs = _build_dataset_kwargs(cfg)
    run_dataset_kwargs["size"] = total_steps
    print(
        "JEPA multi-horizon setup: "
        f"context_steps={context_len_steps}, context_tokens={context_tokens}, "
        f"horizon_tokens={horizon_tokens}, sample_steps={total_steps}"
    )

    print("Loading dataset...")
    dataloaders = DataLoaders(
        datasetCLS=Dataset_Finance_MultiAsset,
        dataset_kwargs=run_dataset_kwargs,
        batch_size_train=train_cfg["batch_size_train"],
        batch_size_eval=train_cfg["batch_size_eval"],
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg.get("pin_memory", True),
        drop_last_train=train_cfg.get("drop_last_train", False),
        drop_last_eval=train_cfg.get("drop_last_eval", False),
        persistent_workers=train_cfg.get("persistent_workers", True),
        prefetch_factor=train_cfg.get("prefetch_factor", 2),
    )
    print(
        f"Dataset loaded {len(dataloaders.train_loader())} train batches and "
        f"{len(dataloaders.val_loader())} val batches"
    )

    train_loader = dataloaders.train_loader()
    val_loader = dataloaders.val_loader()
    num_assets = len(dataloaders.train.dataset.asset_ids)

    # number of assets, and whether to use asset embeddings
    encoder_num_assets = num_assets if model_cfg.get("use_asset_embeddings", True) else None
    print(f"Asset embeddings: {model_cfg.get('use_asset_embeddings', True)}, {encoder_num_assets} assets")

    jepa_context_encoder = PatchTSTEncoder(
        patch_len=model_cfg["patch_len"],
        d_model=model_cfg["d_model"],
        n_features=model_cfg["n_features"],
        n_time_features=model_cfg["n_time_features"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
        dim_ff=model_cfg["dim_ff"],
        dropout=model_cfg["dropout"],
        num_assets=encoder_num_assets,
    )

    # Copy architecture + current weights into an independent target encoder.
    jepa_target_encoder = copy.deepcopy(jepa_context_encoder)

    jepa_model = JEPA(
        jepa_context_encoder,
        jepa_target_encoder,
        d_model=model_cfg["d_model"],
        ema_tau_min=model_cfg["ema_tau_min"],
        ema_tau_max=model_cfg["ema_tau_max"],
        nhead=model_cfg["nhead"],
        dim_ff=model_cfg["dim_ff"],
        dropout=model_cfg["dropout"],
        predictor_num_layers=model_cfg.get("predictor_num_layers", 2),
        context_tokens=context_tokens,
        horizon_blocks=horizon_blocks,
    )

    checkpoint_dir = os.path.join(paths_cfg.get("checkpoint_root", "checkpoints"), model_name)
    checkpoint_path = resume_cfg.get("path") or os.path.join(checkpoint_dir, "best.pt")
    if resume_cfg.get("path"):
        print(f"Using explicit JEPA resume path: {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            missing, unexpected = jepa_model.load_state_dict(checkpoint["model"], strict=False)
            if missing:
                print(f"Missing keys in checkpoint: {missing}")
            if unexpected:
                print(f"Unexpected keys in checkpoint: {unexpected}")
            epoch = int(checkpoint.get("epoch", -1)) + 1
            global_step = int(checkpoint.get("global_step", 0))
            monitor = checkpoint.get("monitor")
            print(
                f"Loaded model weights from {checkpoint_path} "
                f"(resume start_epoch={epoch}, global_step={global_step}, monitor={monitor})"
            )
        except Exception:
            print(f"Failed to load model weights from {checkpoint_path}")
            epoch = 0
            global_step = 0
    else:
        print("There is no best model, starting training from zero")
        epoch = 0
        global_step = 0

    loss_fn = MultiHorizonWeightedLoss(
        loss_cfg["loss_type"],
        horizon_slices,
        near_weight=1.0,
        med_weight=1.0,
        far_weight=1.0,
    )

    opt = torch.optim.AdamW(
        list(jepa_model.context_enc.parameters())
        + list(jepa_model.predictor.parameters())
        + list(jepa_model.predictor_norm.parameters())
        + [jepa_model.target_queries],
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    log_root = paths_cfg.get("log_root", "logs")
    checkpoint_root = paths_cfg.get("checkpoint_root", "checkpoints")
    cbs = [
        PatchingCallback(
            patch_len=model_cfg["patch_len"],
            stride=model_cfg["patch_stride"],
            context_key="x_context",
            target_key="__unused__",
            replace=True,
            do_on_train=True,
            do_on_val=True,
        ),
        PatchingCallback(
            patch_len=model_cfg["patch_len"],
            stride=model_cfg["patch_stride"],
            context_key="t_context",
            target_key="__unused__",
            replace=True,
            do_on_train=True,
            do_on_val=True,
        ),
        StatsPrinter(log_every=train_cfg["log_every"]),
        CSVLogger(path=f"{log_root}/{model_name}_train_log.csv"),
        CheckpointCallback(
            dirpath=f"{checkpoint_root}/{model_name}",
            monitor="val_loss",
            mode="min",
            filename_best="best.pt",
            dont_save_for_epochs=train_cfg.get("dont_save_for_epochs", 3),
            save_last=True,
            filename_last="last.pt",
        ),
    ]

    learn = Learner(
        model=JEPAPretrainModel(jepa_model),
        train_dl=train_loader,
        val_dl=val_loader,
        loss_func=loss_fn,
        opt=opt,
        cbs=cbs,
        amp=train_cfg.get("amp", True),
        grad_clip=train_cfg.get("grad_clip", 1.0),
        start_epoch=epoch,
        global_step=global_step,
        warmup_epochs=train_cfg["warmup_epochs"],
    )

    learn.fit(n_epochs=train_cfg["epochs"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train JEPA pretraining model")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    args = parser.parse_args()
    main(config_path=args.config)
