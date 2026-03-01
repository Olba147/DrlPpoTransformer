import argparse
import os

import numpy as np
import torch
import copy

from config.config_utils import load_json_config
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from Datasets.multi_asset_dataset import Dataset_Finance_MultiAsset
from Training.callbacks import (
    CustomTensorboardCallback,
    EntropyScheduleCallback,
    LastModelCallback,
    RewardEvalCallback,
    TransactionCostScheduleCallback,
)
from Training.ppo_env import GymTradingEnv
from Training.sb3_jepa_ppo import JEPAAuxFeatureExtractor, PPOWithJEPA
from models.jepa.jepa import JEPA
from models.time_series.patchTransformer import PatchTSTEncoder

DEFAULT_CONFIG_PATH = "configs/ppo_jepa_train.json"


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
        "tickers": dataset_cfg.get("tickers"),
        "regular_hours_only": dataset_cfg.get("regular_hours_only", True),
        "timeframe": dataset_cfg.get("timeframe", "15min"),
        "train_start_date": dataset_cfg.get("train_start_date"),
        "train_end_date": dataset_cfg.get("train_end_date"),
        "val_end_date": dataset_cfg.get("val_end_date"),
        "test_end_date": dataset_cfg.get("test_end_date"),
    }


def _load_tickers(path: str) -> list | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]
    return tickers or None


def _load_asset_universe_from_path(path: str | None) -> list | None:
    if not path:
        return None
    return _load_tickers(path)


def make_env(
    dataset,
    episode_len,
    transaction_cost: float,
    reward_scale: float,
    include_wealth: bool,
    include_asset_id: bool,
    allow_short: bool,
    action_mode: str,
    fixed_asset_id: str | None = None,
    full_episode: bool = False,
):
    return lambda: GymTradingEnv(
        dataset,
        episode_len=episode_len,
        transaction_cost=transaction_cost,
        reward_scale=reward_scale,
        allow_short=allow_short,
        action_mode=action_mode,
        include_wealth=include_wealth,
        include_asset_id=include_asset_id,
        fixed_asset_id=fixed_asset_id,
        full_episode=full_episode,
    )


def get_latest_checkpoint(dir_path: str) -> str | None:
    if not os.path.isdir(dir_path):
        return None
    explicit_candidates = [
        os.path.join(dir_path, "last_model.zip"),
        os.path.join(dir_path, "best_model.zip"),
    ]
    for path in explicit_candidates:
        if os.path.exists(path):
            return path

    ckpts = []
    for fname in os.listdir(dir_path):
        if fname.startswith("ppo_") and fname.endswith("_steps.zip"):
            ckpts.append(os.path.join(dir_path, fname))
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: os.path.getmtime(p))
    return ckpts[-1]


def main(config_path: str | None = None):
    cfg = load_json_config(config_path, DEFAULT_CONFIG_PATH, __file__)

    model_name = cfg["model_name"]
    paths_cfg = cfg["paths"]
    resume_cfg = cfg["resume"]
    dataset_cfg = cfg["dataset"]
    env_cfg = cfg["env"]
    ppo_cfg = cfg["ppo"]
    eval_cfg = cfg["evaluation"]
    jepa_cfg = cfg["jepa_model"]

    checkpoint_root = paths_cfg.get("checkpoint_root", "checkpoints")
    log_root = paths_cfg.get("log_root", "logs")
    jepa_checkpoint_dir = paths_cfg.get("jepa_checkpoint_dir")
    ppo_checkpoint_dir = os.path.join(checkpoint_root, model_name)
    jepa_checkpoint_path = paths_cfg.get("jepa_checkpoint_path")
    if not jepa_checkpoint_path:
        if not jepa_checkpoint_dir:
            raise ValueError("Set either paths.jepa_checkpoint_path or paths.jepa_checkpoint_dir in config.")
        jepa_checkpoint_path = os.path.join(jepa_checkpoint_dir, "best.pt")

    run_dataset_kwargs = _build_dataset_kwargs(cfg)

    # Optional subset of assets for PPO can be provided directly or via ticker list file.
    if not run_dataset_kwargs.get("tickers"):
        tickers = _load_tickers(paths_cfg.get("ticker_list_path", ""))
        if tickers:
            run_dataset_kwargs["tickers"] = tickers

    asset_universe = None
    jepa_checkpoint = None
    if os.path.exists(jepa_checkpoint_path):
        jepa_checkpoint = torch.load(jepa_checkpoint_path, map_location="cpu")
        asset_universe = jepa_checkpoint.get("asset_universe")
    if not asset_universe:
        asset_universe = _load_asset_universe_from_path(paths_cfg.get("asset_universe_path"))
    if asset_universe:
        run_dataset_kwargs["asset_universe"] = asset_universe

    print("Loading datasets...")
    if run_dataset_kwargs.get("tickers"):
        print(f"Using subset tickers for PPO: {run_dataset_kwargs['tickers']}")
    else:
        print("Using all available tickers for PPO.")
    train_dataset = Dataset_Finance_MultiAsset(**run_dataset_kwargs)
    val_dataset = Dataset_Finance_MultiAsset(**{**run_dataset_kwargs, "split": "val"})
    num_assets = int(getattr(train_dataset, "num_asset_ids", len(train_dataset.asset_ids)))
    print(
        f"Loaded {len(train_dataset.asset_ids)} PPO assets with "
        f"{num_assets} embedding IDs in the shared universe."
    )

    print("Building environments...")
    action_mode = env_cfg.get("action_mode", "continuous")
    include_asset_id = env_cfg.get("include_asset_id", True)
    transaction_cost_start = env_cfg.get("transaction_cost_start", env_cfg.get("transaction_cost", 0.0))
    transaction_cost_end = env_cfg.get("transaction_cost_end", transaction_cost_start)
    transaction_cost_steps = env_cfg.get("transaction_cost_steps", 1)
    transaction_cost_schedule_timesteps = env_cfg.get("transaction_cost_schedule_timesteps")
    transaction_cost_warmup = env_cfg.get("transaction_cost_warmup", 0)
    print(
        "Transaction cost schedule: "
        f"start={transaction_cost_start}, end={transaction_cost_end}, "
        f"steps={transaction_cost_steps}, schedule_timesteps={transaction_cost_schedule_timesteps}, "
        f"warmup={transaction_cost_warmup}"
    )
    eval_episode_len = int(eval_cfg.get("episode_len", env_cfg["episode_length_steps"]))
    print(f"Action mode: {action_mode}, include_asset_id: {include_asset_id}")
    train_env = SubprocVecEnv(
        [
            make_env(
                train_dataset,
                env_cfg["episode_length_steps"],
                transaction_cost_start,
                env_cfg["reward_scale"],
                env_cfg["include_wealth"],
                include_asset_id,
                env_cfg.get("allow_short", True),
                action_mode,
                full_episode=False,
            )
            for _ in range(env_cfg["n_envs"])
        ]
    )
    eval_env = SubprocVecEnv(
        [
            make_env(
                val_dataset,
                eval_episode_len,
                transaction_cost_start,
                env_cfg["reward_scale"],
                env_cfg["include_wealth"],
                include_asset_id,
                env_cfg.get("allow_short", True),
                action_mode,
                fixed_asset_id=asset_id,
                full_episode=True,
            )
            for asset_id in val_dataset.asset_ids
        ]
    )
    eval_n_episodes = len(val_dataset.asset_ids)
    print(f"Evaluation setup: {eval_n_episodes} envs, one fixed asset per env, one episode per asset.")
    # number of assets, and whether to use asset embeddings
    use_asset_embeddings = jepa_cfg.get("use_asset_embeddings", True) and include_asset_id
    encoder_num_assets = num_assets if use_asset_embeddings else None
    print(f"Asset embeddings: {use_asset_embeddings}, {encoder_num_assets} assets")

    print("Loading JEPA encoder...")
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
        num_assets=encoder_num_assets,
    )

    jepa_target_encoder = copy.deepcopy(jepa_context_encoder)

    jepa_model = JEPA(
        jepa_context_encoder,
        jepa_target_encoder,
        d_model=jepa_cfg["d_model"],
        ema_tau_min=jepa_cfg["ema_tau_min"],
        ema_tau_max=jepa_cfg["ema_tau_max"],
        nhead=jepa_cfg["nhead"],
        dim_ff=jepa_cfg["dim_ff"],
        dropout=jepa_cfg["dropout"],
        predictor_num_layers=jepa_cfg.get("predictor_num_layers", 2),
        mask_ratio=jepa_cfg.get("mask_ratio", 0.5),
    )

    checkpoint_path = jepa_checkpoint_path
    if os.path.exists(checkpoint_path):
        print(f"Loading JEPA weights from {checkpoint_path}")
        checkpoint = jepa_checkpoint if jepa_checkpoint is not None else torch.load(checkpoint_path, map_location="cpu")
        missing, unexpected = jepa_model.load_state_dict(checkpoint["model"], strict=False)
        if missing:
            print(f"Missing keys in checkpoint: {missing}")
        if unexpected:
            print(f"Unexpected keys in checkpoint: {unexpected}")
    else:
        print("No JEPA checkpoint found, using randomly initialized encoder.")

    if not ppo_cfg.get("update_jepa", True):
        for param in jepa_model.parameters():
            param.requires_grad = False
        jepa_model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    jepa_loss_type = ppo_cfg.get("jepa_loss_type", "mse")
    optimizer_name = ppo_cfg.get("optimizer", "adam")
    optimizer_kwargs = ppo_cfg.get("optimizer_kwargs")
    policy_learning_rate = ppo_cfg.get("policy_learning_rate")
    jepa_learning_rate = ppo_cfg.get("jepa_learning_rate")
    print(
        "Optimizer setup: "
        f"name={optimizer_name}, policy_lr={policy_learning_rate or ppo_cfg['learning_rate']}, "
        f"jepa_lr={jepa_learning_rate if jepa_learning_rate is not None else 'same_as_policy'}"
    )
    policy_kwargs = dict(
        features_extractor_class=JEPAAuxFeatureExtractor,
        features_extractor_kwargs=dict(
            jepa_model=jepa_model,
            embedding_dim=jepa_cfg["d_model"],
            patch_len=jepa_cfg["patch_len"],
            patch_stride=jepa_cfg["patch_stride"],
            jepa_loss_type=jepa_loss_type,
        ),
        net_arch=dict(pi=[256, 256], vf=[512, 512])
    )

    resume_path = resume_cfg.get("path")
    if resume_path is None and resume_cfg.get("auto_resume", False):
        resume_path = get_latest_checkpoint(ppo_checkpoint_dir)
        if resume_path:
            print(f"Auto-resume from latest PPO checkpoint: {resume_path}")

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming PPO from {resume_path}")
        model = PPOWithJEPA.load(
            resume_path,
            env=train_env,
            device=device,
            custom_objects={"policy_kwargs": policy_kwargs},
        )
        model.tensorboard_log = log_root
        model.update_jepa = ppo_cfg.get("update_jepa", True)
        model.jepa_coef = ppo_cfg.get("jepa_loss_coef", 0.01)
        model.optimizer_name = str(optimizer_name).lower()
        model.optimizer_kwargs_custom = dict(optimizer_kwargs or {})
        model.policy_learning_rate = (
            None if policy_learning_rate is None else float(policy_learning_rate)
        )
        model.jepa_learning_rate = (
            None if jepa_learning_rate is None else float(jepa_learning_rate)
        )
        model.configure_optimizer()
    else:
        model = PPOWithJEPA(
            policy="MultiInputPolicy",
            env=train_env,
            learning_rate=ppo_cfg["learning_rate"],
            n_steps=ppo_cfg["rollout_length_steps"],
            batch_size=ppo_cfg["batch_size"],
            n_epochs=ppo_cfg["n_epochs"],
            gamma=ppo_cfg["gamma"],
            gae_lambda=ppo_cfg["gae_lambda"],
            clip_range=ppo_cfg["clip_range"],
            ent_coef=ppo_cfg["ent_coef_start"],
            vf_coef=ppo_cfg["vf_coef"],
            max_grad_norm=ppo_cfg["max_grad_norm"],
            target_kl=ppo_cfg["target_kl"],
            update_jepa=ppo_cfg.get("update_jepa", True),
            jepa_coef=ppo_cfg.get("jepa_loss_coef", 0.01),
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=1,
            tensorboard_log=log_root,
            optimizer_name=optimizer_name,
            optimizer_kwargs=optimizer_kwargs,
            policy_learning_rate=policy_learning_rate,
            jepa_learning_rate=jepa_learning_rate,
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
            total_timesteps=ppo_cfg["total_timesteps"],
            warmup_fraction=ppo_cfg["ent_warmup_fraction"],
            ent_coef_start=ppo_cfg["ent_coef_start"],
            ent_coef_end=ppo_cfg["ent_coef_end"],
            ent_decay_steps=ppo_cfg.get("ent_decay_steps"),
        ),
        TransactionCostScheduleCallback(
            total_timesteps=ppo_cfg["total_timesteps"],
            cost_start=transaction_cost_start,
            cost_end=transaction_cost_end,
            cost_steps=transaction_cost_steps,
            cost_schedule_timesteps=transaction_cost_schedule_timesteps,
            cost_warmup_timesteps=transaction_cost_warmup,
            eval_env=eval_env,
        ),
        RewardEvalCallback(
            eval_env,
            best_model_save_path=f"{checkpoint_root}/{model_name}",
            log_path=f"{log_root}/{model_name}_eval",
            eval_freq=eval_cfg["every_steps"],
            n_eval_episodes=eval_n_episodes,
            deterministic=True,
            moving_average_window=eval_cfg.get("moving_average_window", 20),
        ),
        LastModelCallback(
            save_path=f"{checkpoint_root}/{model_name}/last_model.zip",
            verbose=1,
        ),
        JEPACheckpoint(
            jepa_model=jepa_model,
            save_dir=f"{checkpoint_root}/{model_name}",
            every_n_steps=eval_cfg["checkpoint_every_steps"],
        ),
        JEPABestSync(
            jepa_model=jepa_model,
            ppo_best_path=f"{checkpoint_root}/{model_name}/best_model.zip",
            save_dir=f"{checkpoint_root}/{model_name}",
        ),
    ]

    model.learn(
        total_timesteps=ppo_cfg["total_timesteps"],
        callback=callbacks,
        reset_num_timesteps=False,
        tb_log_name=model_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO with JEPA auxiliary objective")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    args = parser.parse_args()
    main(config_path=args.config)
