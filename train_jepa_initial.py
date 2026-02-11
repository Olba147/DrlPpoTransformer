# train.py
import os
import torch

from Datasets.dataloaders import DataLoaders
from Datasets.multi_asset_dataset import Dataset_Finance_MultiAsset
from models.jepa.jepa import JEPA  # your model that accepts a batch dict
from models.time_series.patchTransformer import PatchTSTEncoder
from Training.engine import Learner
from Training.callbacks import PatchingCallback, StatsPrinter, CSVLogger, CheckpointCallback
from Training.helpers import variance_loss

MODEL_NAME = "jepa_initial3"

# ------------------------
# Hyperparameters (edit here)
# ------------------------
DATASET_CONTEXT_LEN = 1024
DATASET_TARGET_LEN = 48

TRAIN_EPOCHS = 100
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 256
LEARNING_RATE = 3e-4
PATCH_LEN = 8
PATCH_STRIDE = 8
LOG_EVERY = 500
CHECKPOINT_EVERY_EPOCHS = 5
AMP = True
GRAD_CLIP = 1.0

JEPA_D_MODEL = 192
JEPA_N_FEATURES = 9
JEPA_N_TIME_FEATURES = 2
JEPA_NHEAD = 3
JEPA_NUM_LAYERS = 3
JEPA_DIM_FF = 768
JEPA_DROPOUT = 0.1
JEPA_ADD_CLS = True
JEPA_POOLING = "cls"   # "cls" | "mean"
JEPA_PRED_LEN = 48

EMA_START = 0.996
EMA_END   = 0.99

dataset_kwargs = {
    "root_path": r"Data/polygon",
    "data_path": r"data_raw_1m",
    "start_date": None,
    "split": "train",
    "size": [DATASET_CONTEXT_LEN, DATASET_TARGET_LEN],  # label_len ignored by your __getitem__
    "use_time_features": True,
    "rolling_window": 252,
    "train_split": 0.7,
    "test_split": 0.15,
    "regular_hours_only": True,
    "timeframe": "15min",
    "tickers": ["AAPL", "AMZN", "QQQ"],
}


def main():
    print("Loading dataset...")
    dataloaders = DataLoaders(
        datasetCLS=Dataset_Finance_MultiAsset,
        dataset_kwargs=dataset_kwargs,
        batch_size_train=TRAIN_BATCH_SIZE,
        batch_size_eval=EVAL_BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
        drop_last_train=False,
        drop_last_eval=False,
        persistent_workers=True,
        prefetch_factor=2,
    )
    print(
        f"Dataset loaded {len(dataloaders.train_loader())} train batches and "
        f"{len(dataloaders.val_loader())} val batches"
    )

    train_loader = dataloaders.train_loader()
    val_loader = dataloaders.val_loader()
    num_assets = len(dataloaders.train.dataset.asset_ids)

    # 2) Model
    jepa_context_encoder = PatchTSTEncoder(
        patch_len=PATCH_LEN,
        d_model=JEPA_D_MODEL,  # transformer hidden dim
        n_features=JEPA_N_FEATURES,
        n_time_features=JEPA_N_TIME_FEATURES,
        nhead=JEPA_NHEAD,  # num of attention heads
        num_layers=JEPA_NUM_LAYERS,  # num of transformer blocks
        dim_ff=JEPA_DIM_FF,  # FFN hidden dim
        dropout=JEPA_DROPOUT,
        add_cls=JEPA_ADD_CLS,
        pooling=JEPA_POOLING,  # "cls" | "mean"
        pred_len=JEPA_PRED_LEN,
        num_assets=num_assets,
    )

    jepa_target_encoder = PatchTSTEncoder(
        patch_len=PATCH_LEN,
        d_model=JEPA_D_MODEL,  # transformer hidden dim
        n_features=JEPA_N_FEATURES,
        n_time_features=JEPA_N_TIME_FEATURES,
        nhead=JEPA_NHEAD,  # num of attention heads
        num_layers=JEPA_NUM_LAYERS,  # num of transformer blocks
        dim_ff=JEPA_DIM_FF,  # FFN hidden dim
        dropout=JEPA_DROPOUT,
        add_cls=JEPA_ADD_CLS,
        pooling=JEPA_POOLING,  # "cls" | "mean"
        pred_len=JEPA_PRED_LEN,
        num_assets=num_assets,
    )

    jepa_model = JEPA(
        jepa_context_encoder,
        jepa_target_encoder,
        d_model=JEPA_D_MODEL,
        ema_start=EMA_START,
        ema_end=EMA_END
    )

    # load model weights from checkpoints/best.pt if exists
    checkpoint_dir = os.path.join("checkpoints", MODEL_NAME)
    checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
    if os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            missing, unexpected = jepa_model.load_state_dict(checkpoint["model"], strict=False)
            if missing:
                print(f"Missing keys in checkpoint: {missing}")
            if unexpected:
                print(f"Unexpected keys in checkpoint: {unexpected}")
            epoch = checkpoint["epoch"]
            monitor = checkpoint["monitor"]
            print(f"Loaded model weights from {checkpoint_path} at epoch {epoch} with monitor {monitor}")
        except Exception:
            print(f"Failed to load model weights from {checkpoint_path}")
            epoch = 0
    else:
        print("There is no best model, starting training from zero")
        epoch = 0

    # should accept the batch dict and use keys like "x_context" (patched by callback)

    # 3) Loss function
    # def loss_fn(yhat, y):
    #     return torch.nn.MSELoss()(yhat, y) + variance_loss(yhat)
    loss_fn = torch.nn.MSELoss()

    # 4) Optimizer
    opt = torch.optim.Adam(
        list(jepa_model.context_enc.parameters())
        # + list(jepa_model.proj_online.parameters())
        + list(jepa_model.predictor.parameters()),
        lr=LEARNING_RATE,
    )

    # 5) Callbacks
    cbs = [
        # Create train-time patches of x_context (+optional x_target) just-in-time
        PatchingCallback(
            patch_len=PATCH_LEN,
            stride=PATCH_STRIDE,
            context_key="x_context",
            target_key="x_target",
            replace=True,
            do_on_train=True,
            do_on_val=True,
        ),
        PatchingCallback(
            patch_len=PATCH_LEN,
            stride=PATCH_STRIDE,
            context_key="t_context",
            target_key="t_target",
            replace=True,
            do_on_train=True,
            do_on_val=True,
        ),
        StatsPrinter(log_every=LOG_EVERY),
        CSVLogger(path=f"logs/{MODEL_NAME}_train_log.csv"),
        CheckpointCallback(
            dirpath=f"checkpoints/{MODEL_NAME}",
            monitor="val_loss",
            mode="min",
            every_n_epochs=CHECKPOINT_EVERY_EPOCHS,
            filename_best="best.pt",
        ),
    ]

    # 6) Learner and fit
    learn = Learner(
        model=jepa_model,
        train_dl=train_loader,
        val_dl=val_loader,
        loss_func=loss_fn,
        opt=opt,
        cbs=cbs,
        amp=AMP,
        grad_clip=GRAD_CLIP,
        start_epoch=epoch,
    )

    learn.fit(n_epochs=TRAIN_EPOCHS)


if __name__ == "__main__":
    main()
