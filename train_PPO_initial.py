import os
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from Datasets.multi_asset_dataset import Dataset_Finance_MultiAsset
from Training.ppo_env import GymTradingEnv
from Training.sb3_jepa import JEPAFeatureExtractor
from models.jepa.jepa import JEPA
from models.time_series.patchTransformer import PatchTSTEncoder

MODEL_NAME = "ppo_initial"
JEPA_CHECKPOINT_DIR = "checkpoints/jepa_initial"

# ------------------------
# Hyperparameters (edit here)
# ------------------------
EPISODE_LENGTH_STEPS = 2340
ROLLOUT_LENGTH_STEPS = 2048
TOTAL_TIMESTEPS = 5_000_000
N_ENVS = 4

LEARNING_RATE = 3e-4
PPO_EPOCHS = 8
BATCH_SIZE = 1024
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TARGET_KL = 0.015

EVAL_EPISODES = 10
EVAL_EVERY_STEPS = 10_000
CHECKPOINT_EVERY_STEPS = 50_000

dataset_kwargs = {
    "root_path": r"Data/polygon",
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

def make_env(dataset, episode_len):
    return lambda: GymTradingEnv(
        dataset,
        episode_len=episode_len,
        transaction_cost=1e-3,
        allow_short=True,
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_kwargs = dict(
        features_extractor_class=JEPAFeatureExtractor,
        features_extractor_kwargs=dict(
            jepa_encoder=jepa_encoder,
            embedding_dim=256,
            patch_len=16,
            patch_stride=16,
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=LEARNING_RATE,
        n_steps=ROLLOUT_LENGTH_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=PPO_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        target_kl=TARGET_KL,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=1,
        tensorboard_log="logs",
    )

    callbacks = [
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
