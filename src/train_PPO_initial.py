import argparse
import os

import numpy as np
import torch
import copy

from config.config_utils import load_json_config
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
from Training.sb3_jepa_ppo import JEPAAuxFeatureExtractor, PPOWithJEPA, PatchTSTAuxFeatureExtractor
from models.jepa.jepa import JEPA
from models.time_series.patchTransformer import PatchTSTEncoder

DEFAULT_CONFIG_PATH = "configs/ppo_single_asset_base.json"


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
    jepa_cfg = cfg.get("jepa_model", {})
    feature_mode = str(ppo_cfg.get("feature_mode", "jepa")).strip().lower()
    if feature_mode not in {"jepa", "patch"}:
        raise ValueError("ppo.feature_mode must be either 'jepa' or 'patch'.")
    print(f"Feature mode: {feature_mode}")

    if ppo_cfg.get("update_jepa", False):
        raise ValueError(
            "PPO JEPA auxiliary updates are disabled in this branch. "
            "Set ppo.update_jepa to false."
        )

    checkpoint_root = paths_cfg.get("checkpoint_root", "checkpoints")
    log_root = paths_cfg.get("log_root", "logs")
    jepa_checkpoint_dir = paths_cfg.get("jepa_checkpoint_dir")
    ppo_checkpoint_dir = os.path.join(checkpoint_root, model_name)
    jepa_checkpoint_path = paths_cfg.get("jepa_checkpoint_path")
    if feature_mode == "jepa":
        if not jepa_checkpoint_path:
            if not jepa_checkpoint_dir:
                raise ValueError("Set either paths.jepa_checkpoint_path or paths.jepa_checkpoint_dir in config.")
            jepa_checkpoint_path = os.path.join(jepa_checkpoint_dir, "best.pt")
    elif feature_mode == "patch":
        jepa_checkpoint_path = None

    run_dataset_kwargs = _build_dataset_kwargs(cfg)

    # Optional subset of assets for PPO can be provided directly or via ticker list file.
    if not run_dataset_kwargs.get("tickers"):
        tickers = _load_tickers(paths_cfg.get("ticker_list_path", ""))
        if tickers:
            run_dataset_kwargs["tickers"] = tickers

    asset_universe = None
    jepa_checkpoint = None
    if feature_mode == "jepa" and jepa_checkpoint_path and os.path.exists(jepa_checkpoint_path):
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
    eval_n_envs = int(env_cfg["n_envs"])
    eval_n_episodes = int(eval_cfg.get("episodes", eval_n_envs))
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
            )
            for _ in range(eval_n_envs)
        ]
    )
    print(
        "Evaluation setup: "
        f"{eval_n_envs} envs, "
        f"episode_len={eval_episode_len}, "
        f"n_eval_episodes={eval_n_episodes}."
    )
    jepa_model = None
    if feature_mode == "jepa":
        # number of assets, and whether to use asset embeddings
        use_asset_embeddings = jepa_cfg.get("use_asset_embeddings", True) and include_asset_id
        encoder_num_assets = num_assets if use_asset_embeddings else None
        print(f"Asset embeddings: {use_asset_embeddings}, {encoder_num_assets} assets")

        print("Loading JEPA encoder...")
        patch_len = int(jepa_cfg["patch_len"])
        patch_stride = int(jepa_cfg["patch_stride"])
        context_len_steps = int(jepa_cfg.get("context_len_steps", dataset_cfg["context_len"]))
        if context_len_steps < patch_len:
            raise ValueError(
                f"jepa_model.context_len_steps must be >= patch_len ({patch_len}), got {context_len_steps}."
            )
        if (context_len_steps - patch_len) % patch_stride != 0:
            raise ValueError(
                f"(context_len_steps - patch_len) must be divisible by patch_stride. "
                f"Got context_len_steps={context_len_steps}, patch_len={patch_len}, patch_stride={patch_stride}."
            )
        context_tokens = 1 + (context_len_steps - patch_len) // patch_stride
        horizon_blocks = jepa_cfg.get(
            "horizon_blocks",
            {
                "near": [1, 1],
                "med": [2, 5],
                "far": [6, 17],
            },
        )

        jepa_context_encoder = PatchTSTEncoder(
            patch_len=jepa_cfg["patch_len"],
            d_model=jepa_cfg["d_model"],
            n_features=jepa_cfg["n_features"],
            n_time_features=jepa_cfg["n_time_features"],
            nhead=jepa_cfg["nhead"],
            num_layers=jepa_cfg["num_layers"],
            dim_ff=jepa_cfg["dim_ff"],
            dropout=jepa_cfg["dropout"],
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
            context_tokens=context_tokens,
            horizon_blocks=horizon_blocks,
        )

        checkpoint_path = str(jepa_checkpoint_path)
        if os.path.exists(checkpoint_path):
            print(f"Loading JEPA weights from {checkpoint_path}")
            checkpoint = (
                jepa_checkpoint
                if jepa_checkpoint is not None
                else torch.load(checkpoint_path, map_location="cpu")
            )
            missing, unexpected = jepa_model.load_state_dict(checkpoint["model"], strict=False)
            if missing:
                print(f"Missing keys in checkpoint: {missing}")
            if unexpected:
                print(f"Unexpected keys in checkpoint: {unexpected}")
        else:
            print("No JEPA checkpoint found, using randomly initialized encoder.")

        for param in jepa_model.parameters():
            param.requires_grad = False
        jepa_model.eval()
    else:
        print("JEPA loading skipped (ppo.feature_mode=patch).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer_name = ppo_cfg.get("optimizer", "adam")
    optimizer_kwargs = ppo_cfg.get("optimizer_kwargs")
    policy_learning_rate = ppo_cfg.get("policy_learning_rate")
    pi_arch = ppo_cfg.get("net_arch_pi", [128, 128])
    vf_arch = ppo_cfg.get("net_arch_vf", [256, 256])
    share_features_extractor = bool(ppo_cfg.get("share_features_extractor", False))
    print(
        "Optimizer setup: "
        f"name={optimizer_name}, policy_lr={policy_learning_rate or ppo_cfg['learning_rate']}, "
        f"feature_mode={feature_mode}"
    )
    policy_kwargs = dict(
        share_features_extractor=share_features_extractor,
        net_arch=dict(pi=pi_arch, vf=vf_arch),
    )
    if feature_mode == "jepa":
        policy_kwargs.update(
            features_extractor_class=JEPAAuxFeatureExtractor,
            features_extractor_kwargs=dict(
                jepa_model=jepa_model,
                embedding_dim=jepa_cfg["d_model"],
                patch_len=jepa_cfg["patch_len"],
                patch_stride=jepa_cfg["patch_stride"],
                attn_pool_heads=jepa_cfg.get("attn_pool_heads", 4),
            ),
        )
    else:
        patch_cfg = ppo_cfg.get("patch_encoder", {})
        if not patch_cfg:
            raise ValueError(
                "ppo.patch_encoder config is required when ppo.feature_mode='patch'."
            )
        obs_n_features = int(train_dataset.data_x[train_dataset.asset_ids[0]].shape[-1])
        obs_n_time_features = int(train_dataset.dates[train_dataset.asset_ids[0]].shape[-1])
        use_asset_embeddings = bool(patch_cfg.get("use_asset_embeddings", False) and include_asset_id)
        policy_kwargs.update(
            features_extractor_class=PatchTSTAuxFeatureExtractor,
            features_extractor_kwargs=dict(
                embedding_dim=int(patch_cfg.get("d_model", 64)),
                patch_len=int(patch_cfg.get("patch_len", 8)),
                patch_stride=int(patch_cfg.get("patch_stride", 8)),
                n_features=int(patch_cfg.get("n_features", obs_n_features)),
                n_time_features=int(patch_cfg.get("n_time_features", obs_n_time_features)),
                nhead=int(patch_cfg.get("nhead", 4)),
                num_layers=int(patch_cfg.get("num_layers", 4)),
                dim_ff=int(patch_cfg.get("dim_ff", 256)),
                dropout=float(patch_cfg.get("dropout", 0.1)),
                attn_pool_heads=int(patch_cfg.get("attn_pool_heads", 4)),
                use_asset_embeddings=use_asset_embeddings,
                num_assets=(num_assets if use_asset_embeddings else None),
            ),
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
        model.update_jepa = False
        model.optimizer_name = str(optimizer_name).lower()
        model.optimizer_kwargs_custom = dict(optimizer_kwargs or {})
        model.policy_learning_rate = (
            None if policy_learning_rate is None else float(policy_learning_rate)
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
            update_jepa=False,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=1,
            tensorboard_log=log_root,
            optimizer_name=optimizer_name,
            optimizer_kwargs=optimizer_kwargs,
            policy_learning_rate=policy_learning_rate,
        )

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
            every_n_steps=eval_cfg["checkpoint_every_steps"],
            verbose=1,
        ),
    ]

    reset_num_timesteps_cfg = ppo_cfg.get("reset_num_timesteps")
    if reset_num_timesteps_cfg is None:
        # Default to non-cumulative timesteps unless user explicitly opts in.
        reset_num_timesteps = True
    else:
        reset_num_timesteps = bool(reset_num_timesteps_cfg)

    print(
        "PPO learn setup: "
        f"total_timesteps={int(ppo_cfg['total_timesteps'])}, "
        f"reset_num_timesteps={reset_num_timesteps}, "
        f"resume_path={resume_path if resume_path else 'None'}"
    )

    model.learn(
        total_timesteps=int(ppo_cfg["total_timesteps"]),
        callback=callbacks,
        reset_num_timesteps=reset_num_timesteps,
        tb_log_name=model_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO (JEPA features or trainable patch-transformer features)")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    args = parser.parse_args()
    main(config_path=args.config)
