import os
import torch

from Datasets.multi_asset_dataset import Dataset_Finance_MultiAsset
from Training.ppo_env import TradingEnv
from Training.ppo_engine import PPOTrainer, PPOTrainingEngine
from Training.callbacks import PPOCSVLogger, PPOStatsPrinter, PPOCheckpoint
from models.jepa.jepa import JEPA
from models.ppo import TanhGaussianPolicy, ValueNetwork
from models.time_series.patchTransformer import PatchTSTEncoder

MODEL_NAME = "ppo_initial"
JEPA_CHECKPOINT_DIR = "checkpoints/jepa_initial"

dataset_kwargs = {
    "root_path": r"Data\polygon",
    "data_path": r"data_raw_1m",
    "start_date": None,
    "split": "train",
    "size": [1024, 96],
    "use_time_features": True,
    "rolling_window": 252,
    "train_split": 0.7,
    "test_split": 0.15,
    "regular_hours_only": True,
    "timeframe": "5min",
}

print("Loading datasets...")
train_dataset = Dataset_Finance_MultiAsset(**dataset_kwargs)
val_dataset = Dataset_Finance_MultiAsset(**{**dataset_kwargs, "split": "val"})

print("Building environments...")
train_env = TradingEnv(train_dataset, episode_len=256, transaction_cost=1e-3, allow_short=True)
eval_env = TradingEnv(val_dataset, episode_len=256, transaction_cost=1e-3, allow_short=True)

print("Loading JEPA encoder...")
jepa_context_encoder = PatchTSTEncoder(
    patch_len=16,
    d_model=256,
    n_features=9,
    n_time_features=2,
    nhead=4,
    num_layers=3,
    dim_ff=512,
    dropout=0.1,
    add_cls=True,
    pooling="mean",
    pred_len=96,
)

jepa_target_encoder = PatchTSTEncoder(
    patch_len=16,
    d_model=256,
    n_features=9,
    n_time_features=2,
    nhead=4,
    num_layers=3,
    dim_ff=512,
    dropout=0.1,
    add_cls=True,
    pooling="mean",
    pred_len=96,
)

jepa_model = JEPA(
    jepa_context_encoder,
    jepa_target_encoder,
    d_model=256,
    ema_start=0.99,
    ema_end=0.999,
    n_epochs=20,
)

checkpoint_path = os.path.join(JEPA_CHECKPOINT_DIR, "best.pt")
if os.path.exists(checkpoint_path):
    print(f"Loading JEPA weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    jepa_model.load_state_dict(checkpoint["model"])
else:
    print("No JEPA checkpoint found, using randomly initialized encoder.")

jepa_encoder = jepa_model.context_enc
for param in jepa_encoder.parameters():
    param.requires_grad = False
jepa_encoder.eval()

state_dim = 256 + 1 + 1
policy = TanhGaussianPolicy(input_dim=state_dim, hidden_dims=(256, 256), action_dim=1)
value_fn = ValueNetwork(input_dim=state_dim, hidden_dims=(256, 256))

optimizer = torch.optim.Adam(
    list(policy.parameters()) + list(value_fn.parameters()),
    lr=3e-4,
)

trainer = PPOTrainer(
    env=train_env,
    policy=policy,
    value_fn=value_fn,
    jepa_encoder=jepa_encoder,
    optimizer=optimizer,
    clip_coef=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    gamma=0.99,
    lam=0.95,
    target_kl=0.015,
)

callbacks = [
    PPOStatsPrinter(log_every=1),
    PPOCSVLogger(path=f"logs/{MODEL_NAME}_train_log.csv"),
    PPOCheckpoint(dirpath=f"checkpoints/{MODEL_NAME}", every_n_updates=10),
]

engine = PPOTrainingEngine(trainer, callbacks=callbacks)
engine.train(
    num_updates=50,
    rollout_len=512,
    update_epochs=4,
    minibatch_size=128,
    eval_env=eval_env,
    eval_episodes=2,
    eval_every=10,
)
