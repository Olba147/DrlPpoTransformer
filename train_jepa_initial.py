# train.py
import torch
from Datasets.dataloaders import DataLoaders
from Datasets.datasets import Dataset_Finance
from models.jepa.jepa import JEPA  # your model that accepts a batch dict
from models.time_series.patchTransformer import PatchTSTEncoder
from Training.engine import Learner, PatchingCallback, StatsPrinter, CSVLogger, CheckpointCallback
from Training.helpers import variance_loss
import os

MODEL_NAME = "jepa_initial"

dataset_kwargs = {
    "root_path": r"Data\binance",
    "data_path": r"BTCUSDT_5m_1504137600000_1759190400000_klines.csv",
    "start_date": "2020-12-31",
    "size": [1024, 96],  # label_len ignored by your __getitem__
    "rolling_window": 252,
    "use_time_features": True,
    "time_col_name": "close_time",
    "train_split": 0.7,
    "test_split": 0.15,
}

dataloaders = DataLoaders(
    datasetCLS=Dataset_Finance,
    dataset_kwargs=dataset_kwargs,
    batch_size_train=64,
    batch_size_eval=256,
    num_workers=0,
    pin_memory=False,
    drop_last_train=False,
    drop_last_eval=False,
    persistent_workers=False,
    prefetch_factor=None,
)

train_loader = dataloaders.train_loader()
val_loader = dataloaders.val_loader()

# 2) Model
jepa_context_encoder = PatchTSTEncoder(
    patch_len=16,
    d_model= 256,          # transformer hidden dim
    n_features = 9,
    n_time_features = 2,
    nhead= 4,              # num of attention heads   
    num_layers = 3,         # num of transformer blocks
    dim_ff = 512,           # FFN hidden dim
    dropout = 0.1,
    add_cls = True,
    pooling = "mean",        # "cls" | "mean"
    pred_len = 96

)

jepa_target_encoder = PatchTSTEncoder(
    patch_len=16,
    d_model= 256,          # transformer hidden dim
    n_features = 9,
    n_time_features = 2,
    nhead= 4,              # num of attention heads   
    num_layers = 3,         # num of transformer blocks
    dim_ff = 512,           # FFN hidden dim
    dropout = 0.1,
    add_cls = True,
    pooling = "mean",        # "cls" | "mean"
    pred_len = 96

)
    
jepa_model = JEPA(
    jepa_context_encoder,
    jepa_target_encoder,
    d_model = 256,
    ema_start = 0.99,
    ema_end = 0.999,
    n_epochs = 20
)

# load model weights from checkpoints/best.pt if exists
if "best.pt" in os.listdir(os.path.join("checkpoints", MODEL_NAME)):
    print(f"Loading model weights from checkpoints/{MODEL_NAME}/best.pt")
    try:
        checkpoint = checkpoint = torch.load(f"checkpoints/{MODEL_NAME}/best.pt", map_location="cpu")
        jepa_model.load_state_dict(checkpoint["model"])
        epoch = checkpoint["epoch"]
        monitor = checkpoint["monitor"]
        print(f"Loaded model weights from checkpoints/{MODEL_NAME}/best.pt at epoch", epoch, "with monitor", monitor)
    except Exception as e:
        print(f"Failed to load model weights from checkpoints/{MODEL_NAME}/best.pt:")



# should accept the batch dict and use keys like "x_context" (patched by callback)

# 3) Loss function
# def loss_fn(yhat, y):
#     return torch.nn.MSELoss()(yhat, y) + variance_loss(yhat)
loss_fn = torch.nn.MSELoss()

# 4) Optimizer
opt = torch.optim.Adam(
    list(jepa_model.context_enc.parameters()) +
    list(jepa_model.proj_online.parameters()) +
    list(jepa_model.predictor.parameters()),
    lr=3e-4,
)

# 5) Callbacks
cbs = [
    # Create train-time patches of x_context (+optional x_target) just-in-time
    PatchingCallback(patch_len=16, stride=16, context_key="x_context",
                     target_key="x_target", replace=True, do_on_train=True, do_on_val=True),
    PatchingCallback(patch_len=16, stride=16, context_key="t_context",
                     target_key="t_target", replace=True, do_on_train=True, do_on_val=True),
    StatsPrinter(log_every=100),
    CSVLogger(path=f"logs/{MODEL_NAME}_train_log.csv"),
    CheckpointCallback(dirpath=f"checkpoints/{MODEL_NAME}", monitor="val_loss", mode="min",
                       every_n_epochs=5, filename_best="best.pt"),
]

# 6) Learner and fit
learn = Learner(model=jepa_model, train_dl=train_loader, val_dl=val_loader,
                loss_func=loss_fn, opt=opt, cbs=cbs, amp=True, grad_clip=1.0, start_epoch=epoch)

learn.fit(n_epochs=20)