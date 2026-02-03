import os
import numpy as np
import torch

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

MODEL_NAME = "ppo_initial2"
JEPA_CHECKPOINT_DIR = "checkpoints/jepa_initial2"

# ------------------------
# Hyperparameters (edit here)
# ------------------------
EPISODE_LENGTH_STEPS = 2340
ROLLOUT_LENGTH_STEPS = 1024
TOTAL_TIMESTEPS = 6_000_000
N_ENVS = 4

LEARNING_RATE = 5e-5
PPO_EPOCHS = 4
BATCH_SIZE = 512
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.15

ENT_COEF_START = 0.05
ENT_COEF_END = 0.01
ENT_WARMUP_FRACTION = 0.2

VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TARGET_KL = 0.02
UPDATE_JEPA = True
JEPA_LOSS_COEF = 1.0

EVAL_EPISODES = 2
EVAL_EVERY_STEPS = 10_000
CHECKPOINT_EVERY_STEPS = 50_000

TRANSACTION_COST = 1e-5 # only for initial training to support exploration
INCLUDE_WEALTH = False

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
    train_dataset = Dataset_Finance_MultiAsset(**dataset_kwargs)
    val_dataset = Dataset_Finance_MultiAsset(**{**dataset_kwargs, "split": "val"})

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
    )

    jepa_model = JEPA(
        jepa_context_encoder,
        jepa_target_encoder,
        d_model=JEPA_D_MODEL,
        ema_start=EMA_END,
        ema_end=EMA_END,
        n_epochs=EMA_EPOCHS,
    )

    checkpoint_path = os.path.join(JEPA_CHECKPOINT_DIR, "best.pt")
    if os.path.exists(checkpoint_path):
        print(f"Loading JEPA weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        jepa_model.load_state_dict(checkpoint["model"])
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
    ]

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)


if __name__ == "__main__":
    main()
