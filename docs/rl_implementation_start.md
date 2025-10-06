# RL Integration Kickoff Plan

This document outlines the initial milestones for connecting the existing JEPA + PatchTST
representation learner to a reinforcement-learning trading agent. The focus is on staging the
work so that each component can be trained and validated independently before wiring the full
RL loop.

## 1. Formalize the Latent Encoder API
- **Goal**: provide a single module that owns the PatchTST encoder and JEPA wrapper.
- **Why**: both self-supervised pretraining and the RL agent should call the same interface to
  obtain latent states.
- **Key steps**:
  1. Wrap the `PatchTSTEncoder` so that it exposes a `forward(X_patch, time_patch) -> latent`
     method with normalized output.
  2. Instantiate `JEPA` with tied context and target encoders (the same `PatchTSTEncoder`
     copy) and ensure `ema_update` is called after each optimization step.
  3. Save and load checkpoints that include encoder weights plus normalization statistics from
     the data loader (`stats['mean']`, `stats['std']`).

Relevant code: `models/time_series/patchTransformer.py`, `models/jepa/jepa.py`, and
`DataLoaders/binance_dataloader.py`.

## 2. Offline JEPA Pretraining Pipeline
- **Goal**: create a script that pretrains the encoder using JEPA on historical Binance data.
- **Key steps**:
  1. Use `JointSeqDataset` with `use_patch=True` so samples match the encoder expectations.
  2. Implement training loop: sample batches, run JEPA forward pass, compute cosine-similarity
     or BYOL-style regression loss, backprop only through the context branch, call
     `ema_update` on the target encoder.
  3. Periodically evaluate embedding quality (e.g., next-return regression) to confirm learning.
  4. Export pretrained encoder weights for downstream RL.

## 3. Latent-State Trading Environment
- **Goal**: define an RL `Env` that steps through time-series windows and emits latent states.
- **Key steps**:
  1. Implement a `gym.Env` (or `pettingzoo` style) that internally uses `JointSeqDataset` to
     retrieve sequential windows.
  2. On `reset`, draw a starting index, load the normalized price window, and encode the context
     patches using the **frozen** pretrained encoder to produce the observation.
  3. Define actions (e.g., {-1, 0, +1} position) and reward as realized log-return minus trading
     costs over the next step (using the target slice from the dataset).
  4. Maintain portfolio state (position, cash) and expose auxiliary info (raw prices) for
     diagnostics.

## 4. RL Policy & Training Loop
- **Goal**: train a policy/value network that consumes JEPA latents.
- **Key steps**:
  1. Start with PPO (already partially implemented in the repository name) using a small MLP
     policy whose input dimension matches the latent size (`d_model`).
  2. Implement replay buffer / rollout storage that records latent observations, actions,
     rewards, and done flags.
  3. Ensure gradients do **not** flow into the encoder during RL (encoder stays frozen). Later
     experiments can add fine-tuning.
  4. Track metrics: episode return, Sharpe ratio, max drawdown, percentage of profitable trades.

## 5. End-to-End Evaluation Harness
- **Goal**: validate the combined system.
- **Key steps**:
  1. Run rollouts on validation segments, compute trading KPIs, and compare against baselines
     (buy-and-hold, moving-average crossover).
  2. Implement logging/visualization: latent trajectories, policy logits, reward decomposition.
  3. Add unit tests/mocks for encoder interface and environment stepping to guard future
     refactors.

Following this plan will bootstrap the RL implementation while reusing the existing JEPA and
PatchTST components effectively.
