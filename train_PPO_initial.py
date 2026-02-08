import os
import numpy as np
import torch
import random

# import jepa params
from train_jepa_initial import DATASET_CONTEXT_LEN, DATASET_TARGET_LEN
from train_jepa_initial import PATCH_LEN, PATCH_STRIDE, JEPA_D_MODEL, JEPA_N_FEATURES, JEPA_N_TIME_FEATURES, JEPA_NHEAD, JEPA_NUM_LAYERS
from train_jepa_initial import JEPA_DIM_FF, JEPA_DROPOUT, JEPA_POOLING, JEPA_PRED_LEN
from train_jepa_initial import EMA_START, EMA_END, EMA_EPOCHS

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from Datasets.multi_asset_dataset import Dataset_Finance_MultiAsset
from Training.ppo_env import GymTradingEnv
from Training.sb3_jepa_ppo import JEPAAuxFeatureExtractor, PPOWithJEPA
from models.jepa.jepa import JEPA
from models.time_series.patchTransformer import PatchTSTEncoder
from Training.callbacks import CustomTensorboardCallback, EntropyScheduleCallback

MODEL_NAME = "jepa_ppo3_emptry_start"
JEPA_CHECKPOINT_DIR = "checkpoints/jepa_initial3"

# ------------------------
# Hyperparameters (edit here)
# ------------------------
EPISODE_LENGTH_STEPS = 2048
ROLLOUT_LENGTH_STEPS = 2048
TOTAL_TIMESTEPS = 6_000_000
N_ENVS = 16

LEARNING_RATE = 5e-5
PPO_EPOCHS = 1
BATCH_SIZE = 512
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2

ENT_COEF_START = 0.02
ENT_COEF_END = 0.01
ENT_WARMUP_FRACTION = 0.3

VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TARGET_KL = 0.02
UPDATE_JEPA = True
JEPA_LOSS_COEF = 0.1

EVAL_EPISODES = 2
EVAL_EPISODE_LEN = 512
EVAL_EVERY_STEPS = 4*1024
CHECKPOINT_EVERY_STEPS = 10_000

TRANSACTION_COST = 0 # only for initial training to support exploration
INCLUDE_WEALTH = False
NUM_RANDOM_TICKERS = 5
RANDOM_TICKER_SEED = None  # set int for reproducibility
TICKER_LIST_PATH = "logs/selected_tickers.txt"

dataset_kwargs = {
    "root_path": r"Data/polygon",
    "data_path": r"data_raw_1m",
    "start_date": None,
    "split": "train",
    "size": [1024, 48],
    "use_time_features": True,
    "rolling_window": 252,
    "train_split": 0.7,
    "test_split": 0.15,
    "regular_hours_only": True,
    "timeframe": "5min",
}

def pick_random_tickers(root_path: str, data_path: str, num: int, seed: int | None = None) -> list:
    path = os.path.join(root_path, data_path)
    if seed is not None:
        random.seed(seed)
    tickers = []
    if os.path.isdir(path):
        for fname in os.listdir(path):
            if fname.endswith(".parquet"):
                tickers.append(os.path.splitext(fname)[0])
    if len(tickers) < num:
        raise RuntimeError(f"Not enough tickers to sample {num} from {path}. Found {len(tickers)}.")
    return random.sample(tickers, num)

def save_tickers(tickers: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in tickers:
            f.write(f"{t}\n")

def make_env(dataset, episode_len):
    return lambda: GymTradingEnv(
        dataset,
        episode_len=episode_len,
        transaction_cost=TRANSACTION_COST,
        allow_short=True,
        include_wealth=INCLUDE_WEALTH,
    )

def main():
    print("Loading datasets...")
    tickers = pick_random_tickers(
        root_path=dataset_kwargs["root_path"],
        data_path=dataset_kwargs["data_path"],
        num=NUM_RANDOM_TICKERS,
        seed=RANDOM_TICKER_SEED,
    )
    print(f"Selected tickers: {tickers}")
    dataset_kwargs["tickers"] = tickers
    save_tickers(tickers, TICKER_LIST_PATH)
    train_dataset = Dataset_Finance_MultiAsset(**dataset_kwargs)
    val_dataset = Dataset_Finance_MultiAsset(**{**dataset_kwargs, "split": "val"})
    num_assets = len(train_dataset.asset_ids)

    print("Building environments...")
    train_env = SubprocVecEnv([make_env(train_dataset, EPISODE_LENGTH_STEPS) for _ in range(N_ENVS)])
    eval_env = DummyVecEnv([make_env(val_dataset, EPISODE_LENGTH_STEPS)])

    print("Loading JEPA encoder...")
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
        action_dim=1,
    )

    checkpoint_path = os.path.join(JEPA_CHECKPOINT_DIR, "best.pt")
    if os.path.exists(checkpoint_path):
        print(f"Loading JEPA weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        missing, unexpected = jepa_model.load_state_dict(checkpoint["model"], strict=False)
        if missing:
            print(f"Missing keys in checkpoint: {missing}")
        if unexpected:
            print(f"Unexpected keys in checkpoint: {unexpected}")
    else:
        print("No JEPA checkpoint found, using randomly initialized encoder.")

    if not UPDATE_JEPA:
        for param in jepa_model.parameters():
            param.requires_grad = False
        jepa_model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_kwargs = dict(
        features_extractor_class=JEPAAuxFeatureExtractor,
        features_extractor_kwargs=dict(
            jepa_model=jepa_model,
            embedding_dim=JEPA_D_MODEL,
            patch_len=PATCH_LEN,
            patch_stride=PATCH_STRIDE,
            use_obs_targets=True,
            target_len=DATASET_TARGET_LEN,
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    model = PPOWithJEPA(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=LEARNING_RATE,
        n_steps=ROLLOUT_LENGTH_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=PPO_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF_START,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        target_kl=TARGET_KL,
        update_jepa=UPDATE_JEPA,
        jepa_coef=JEPA_LOSS_COEF,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=1,
        tensorboard_log="logs",
    )

    class JEPACheckpoint(BaseCallback):
        def __init__(self, jepa_model: JEPA, save_dir: str, every_n_steps: int):
            super().__init__()
            self.jepa_model = jepa_model
            self.save_dir = save_dir
            self.every_n_steps = every_n_steps

        def _on_step(self) -> bool:
            if self.n_calls % self.every_n_steps != 0:
                return True
            os.makedirs(self.save_dir, exist_ok=True)
            path = os.path.join(self.save_dir, f"jepa_step_{self.n_calls}.pt")
            torch.save({"model": self.jepa_model.state_dict(), "step": self.n_calls}, path)
            return True

    class JEPABestSync(BaseCallback):
        def __init__(self, jepa_model: JEPA, ppo_best_path: str, save_dir: str):
            super().__init__()
            self.jepa_model = jepa_model
            self.ppo_best_path = ppo_best_path
            self.save_dir = save_dir
            self._last_mtime = None

        def _on_step(self) -> bool:
            if not os.path.exists(self.ppo_best_path):
                return True
            mtime = os.path.getmtime(self.ppo_best_path)
            if self._last_mtime is None or mtime > self._last_mtime:
                os.makedirs(self.save_dir, exist_ok=True)
                path = os.path.join(self.save_dir, "best_jepa_ppo.pt")
                torch.save({"model": self.jepa_model.state_dict(), "step": self.n_calls}, path)
                self._last_mtime = mtime
            return True

    callbacks = [
        CustomTensorboardCallback(),
        EntropyScheduleCallback(
            total_timesteps=TOTAL_TIMESTEPS,
            warmup_fraction=ENT_WARMUP_FRACTION,
            ent_coef_start=ENT_COEF_START,
            ent_coef_end=ENT_COEF_END,
        ),
        CheckpointCallback(
            save_freq=CHECKPOINT_EVERY_STEPS,
            save_path=f"checkpoints/{MODEL_NAME}",
            name_prefix="ppo",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=f"checkpoints/{MODEL_NAME}",
            log_path=f"logs/{MODEL_NAME}_eval",
            eval_freq=EVAL_EVERY_STEPS,
            n_eval_episodes=EVAL_EPISODES,
            deterministic=True,
        ),
        JEPACheckpoint(
            jepa_model=jepa_model,
            save_dir=f"checkpoints/{MODEL_NAME}",
            every_n_steps=CHECKPOINT_EVERY_STEPS,
        ),
        JEPABestSync(
            jepa_model=jepa_model,
            ppo_best_path=f"checkpoints/{MODEL_NAME}/best_model.zip",
            save_dir=f"checkpoints/{MODEL_NAME}",
        ),
    ]

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)


if __name__ == "__main__":
    main()
