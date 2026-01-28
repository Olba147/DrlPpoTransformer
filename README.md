# DrlPpoTransformer

This repository combines a JEPA-style self-supervised encoder with a patch-based Transformer for financial time-series data, and then uses that encoder inside a PPO trading agent. The project includes dataset utilities for Polygon parquet files and Binance CSV data, JEPA pretraining scripts, and PPO training with a custom trading environment.

## Dataset and Dataloader

### Multi-asset Polygon dataset

`Datasets/multi_asset_dataset.py` defines `Dataset_Finance_MultiAsset`, which loads per-asset parquet files and produces aligned train/val/test splits across all assets by global timestamp. The dataset:

- Converts timestamps to UTC, optionally filters regular U.S. trading hours (09:30–16:00 ET), and resamples OHLCV to a configurable timeframe.
- Computes log prices, log returns, volatility and range features (`ret_*`, `hl_range`, `tr`, `vol20`, `log_vol`, `vol_z20`), then applies rolling z-score normalization per asset.
- Tracks raw OHLCV and time features (weekday, minute-of-day) alongside the normalized features.
- Returns a context/target window for each asset with keys such as `x_context`, `x_target`, `t_context`, `t_target`, and the corresponding raw OHLCV.

Train/val/test splits are determined by global dates to keep asset alignment consistent across assets. The dataset’s `seq_len` and `pred_len` define context and target windows. The inference-only variant `Dataset_Finance_MultiAsset_Pred` returns the most recent `seq_len` window per asset for prediction.

### Dataloader wrapper

`Datasets/dataloaders.py` provides a `DataLoaders` convenience wrapper that instantiates train/val/test datasets and exposes ready-to-use PyTorch dataloaders. The wrapper takes a `dataset_kwargs` dict (minus `split`) and builds each loader with separate batch sizes for training and evaluation.

### Binance CSV sequence dataset

`DataLoaders/binance_dataloader.py` defines `JointSeqDataset` and `create_dataloaders()` for single-asset Binance CSV data. It builds arrays of log prices, log returns, log volumes, and time features, splits by train/val/test ratios, and optionally standardizes features on the training split. It can return either raw sequences (`[seq_len, features]`) or patchified sequences (`[num_patches, patch_len * features]`), matching the patch-based JEPA encoder.

## PatchTransformer (PatchTSTEncoder)

`models/time_series/patchTransformer.py` implements a PatchTST-style Transformer encoder. It:

- Accepts patchified price features and patchified time features.
- Encodes time using sin/cos for weekday (period 7) and minute-of-day (period 1440), then projects price and time patches into a shared `d_model` space.
- Adds positional encoding and passes tokens through Transformer encoder layers.
- Outputs a pooled embedding (CLS token or mean pooling), which is used as the JEPA representation and later as the PPO state embedding.

## JEPA Model

`models/jepa/jepa.py` defines the JEPA model with:

- A **context encoder** and **target encoder** (both PatchTST encoders).
- Online and target projection heads, where the target projection is updated by EMA.
- A small predictor network that maps the context projection toward the target projection.

The forward pass returns the predictor output and the target projection; the training loss typically minimizes the MSE between them.

## JEPA Pretraining

JEPA pretraining scripts live in:

- `train_jepa_initial.py`: trains on the multi-asset Polygon dataset.
- `train_jepa_reg.py`: trains on Binance CSV data (older script variant).

Common pretraining flow:

1. Build datasets/dataloaders for train/val splits.
2. Instantiate PatchTST encoders for context and target, then wrap them in `JEPA`.
3. Train with `Training/engine.py`’s `Learner`, using `Training/callbacks.py` to patchify input sequences on-the-fly, log metrics, and checkpoint models.
4. The `Learner` computes MSE loss between context predictions and target projections and performs EMA updates on the target encoder.

The primary checkpoint for downstream PPO is stored under `checkpoints/jepa_initial/best.pt` if `train_jepa_initial.py` is used.

## PPO Model

`models/ppo/actor_critic.py` provides the PPO policy and value networks:

- `TanhGaussianPolicy` outputs a squashed Gaussian action distribution (with optional state-dependent log standard deviation) and exposes `sample`, `log_prob`, and `mean_action` methods.
- `ValueNetwork` is an MLP that outputs a scalar value estimate.

These models are used with a state vector that concatenates the JEPA embedding with additional trading features (previous position and optional wealth feature).

## PPO Environment

`Training/ppo_env.py` defines `TradingEnv`, a custom environment that:

- Samples a random asset and starting index per episode.
- Produces observations containing `x_context`, `t_context`, previous position (`w_prev`), and optional log-wealth features.
- Computes rewards based on portfolio return from the next-step close price and applies a transaction cost proportional to position turnover.

The environment uses the `Dataset_Finance_MultiAsset` data (normalized features and raw OHLCV) to extract close prices and time-aligned windows.

## PPO Training

PPO training is orchestrated by:

- `Training/ppo_engine.py`: `PPOTrainer` collects rollouts, computes GAE advantages, and updates policy/value networks using PPO clipping; `PPOTrainingEngine` handles update loops and optional evaluation.
- `Training/callbacks.py`: `PPOStatsPrinter`, `PPOCSVLogger`, and `PPOCheckpoint` provide logging and checkpointing.
- `train_PPO_initial.py`: sets up the datasets/environments, loads the pretrained JEPA encoder (frozen), and runs PPO training.

High-level PPO workflow:

1. Load datasets and construct training/evaluation environments.
2. Load the pretrained JEPA checkpoint and freeze the context encoder.
3. Build the policy/value networks with `state_dim = JEPA_dim + w_prev + wealth_feat`.
4. Collect rollouts from the trading environment, update policy/value networks, and log metrics/checkpoints.

## Quickstart scripts

```bash
# JEPA pretraining (Polygon multi-asset)
python train_jepa_initial.py

# PPO training (requires JEPA checkpoint)
python train_PPO_initial.py
```

Make sure the dataset paths in the training scripts match your local data layout (e.g., `Data/polygon/...`).
