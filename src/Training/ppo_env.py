from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class EnvState:
    asset_id: str
    cursor: int
    step: int
    w_prev: float
    wealth: float


class TradingEnv:
    def __init__(
        self,
        dataset,
        episode_len: int = 256,
        transaction_cost: float = 1e-3,
        reward_scale: float = 1.0,
        allow_short: bool = True,
        action_mode: str = "continuous",
        include_wealth: bool = True,
        include_asset_id: bool = True,
        fixed_asset_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.episode_len = episode_len
        self.transaction_cost = transaction_cost
        self.reward_scale = reward_scale
        self.allow_short = allow_short
        self.action_mode = action_mode
        self.include_wealth = include_wealth
        self.include_asset_id = include_asset_id
        self.seq_len = dataset.seq_len
        self.pred_len = dataset.pred_len
        self.rng = np.random.default_rng(seed)
        self._discrete_actions = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

        if not getattr(dataset, "asset_ids", []):
            raise ValueError("Dataset must contain asset_ids to build episodes.")
        if self.action_mode not in {"continuous", "discrete_3"}:
            raise ValueError(f"Unknown action_mode: {self.action_mode}")
        if fixed_asset_id is not None and fixed_asset_id not in dataset.asset_ids:
            raise ValueError(f"Unknown fixed_asset_id: {fixed_asset_id}")
        self.fixed_asset_id = fixed_asset_id

        self.state: Optional[EnvState] = None

    def _sample_start(self) -> Tuple[str, int]:
        asset_id = self.fixed_asset_id or self.rng.choice(self.dataset.asset_ids)
        data_len = len(self.dataset.data_x[asset_id])
        max_start = data_len - self.seq_len - self.pred_len - self.episode_len
        if max_start <= 0:
            max_start = max(0, data_len - self.seq_len - 2)
        start = int(self.rng.integers(0, max_start + 1))
        return asset_id, start

    def _get_close(self, asset_id: str, idx: int) -> float:
        return float(self.dataset.ohlcv[asset_id][idx][3])

    def _observe(self, asset_id: str, cursor: int, w_prev: float, wealth: float) -> Dict:
        asset_idx = self.dataset.asset_id_to_idx.get(asset_id, -1)
        x_context = self.dataset.data_x[asset_id][cursor : cursor + self.seq_len]
        t_context = self.dataset.dates[asset_id][cursor : cursor + self.seq_len]
        x_target = self.dataset.data_x[asset_id][
            cursor + self.seq_len : cursor + self.seq_len + self.dataset.pred_len
        ]
        t_target = self.dataset.dates[asset_id][
            cursor + self.seq_len : cursor + self.seq_len + self.dataset.pred_len
        ]
        obs = {
            "x_context": x_context.astype(np.float32),
            "t_context": t_context.astype(np.float32),
            "x_target": x_target.astype(np.float32),
            "t_target": t_target.astype(np.float32),
            "w_prev": np.array([w_prev], dtype=np.float32),
        }
        if self.include_asset_id:
            obs["asset_id"] = np.int64(asset_idx)
        if self.include_wealth:
            obs["wealth_feats"] = np.array([np.log(wealth)], dtype=np.float32)
        return obs

    def reset(self) -> Dict:
        asset_id, start = self._sample_start()
        self.state = EnvState(asset_id=asset_id, cursor=start, step=0, w_prev=0.0, wealth=1.0)
        return self._observe(asset_id, start, self.state.w_prev, self.state.wealth)

    def _action_to_weight(self, action) -> float:
        if self.action_mode == "discrete_3":
            action_idx = int(np.asarray(action).reshape(-1)[0])
            action_idx = int(np.clip(action_idx, 0, len(self._discrete_actions) - 1))
            w_t = float(self._discrete_actions[action_idx])
        else:
            w_t = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        if not self.allow_short:
            w_t = max(0.0, w_t)
        return w_t

    def step(self, action: np.ndarray) -> Dict:
        if self.state is None:
            raise RuntimeError("Environment must be reset before stepping.")

        # Action is interpreted either as direct weight or as index over [-1, 0, 1].
        w_t = self._action_to_weight(action)

        asset_id = self.state.asset_id
        cursor = self.state.cursor
        close_t = self._get_close(asset_id, cursor + self.seq_len - 1)
        close_tp1 = self._get_close(asset_id, cursor + self.seq_len)
        r_tp1 = float(np.log(close_tp1 / close_t))

        turnover = abs(w_t - self.state.w_prev)
        cost = min(self.transaction_cost * turnover, 0.99)
        reward = w_t * r_tp1 + float(np.log1p(-cost))
        self.state.wealth *= float(np.exp(reward))
        scaled_reward = reward * self.reward_scale

        self.state.cursor += 1
        self.state.step += 1
        self.state.w_prev = w_t

        done = (
            self.state.step >= self.episode_len
            or (self.state.cursor + self.seq_len + self.pred_len) >= len(self.dataset.data_x[asset_id])
        )

        obs = self._observe(asset_id, self.state.cursor, self.state.w_prev, self.state.wealth)
        obs["reward"] = np.array(scaled_reward, dtype=np.float32)
        obs["done"] = np.array(done, dtype=np.float32)
        obs["info"] = {
            "position": w_t,
            "turnover": turnover,
            "return": r_tp1,
            "wealth": self.state.wealth,
        }
        return obs


class GymTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset,
        episode_len: int = 256,
        transaction_cost: float = 1e-3,
        reward_scale: float = 1.0,
        allow_short: bool = True,
        action_mode: str = "continuous",
        include_wealth: bool = True,
        include_asset_id: bool = True,
        fixed_asset_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.env = TradingEnv(
            dataset=dataset,
            episode_len=episode_len,
            transaction_cost=transaction_cost,
            reward_scale=reward_scale,
            allow_short=allow_short,
            action_mode=action_mode,
            include_wealth=include_wealth,
            include_asset_id=include_asset_id,
            fixed_asset_id=fixed_asset_id,
            seed=seed,
        )

        seq_len = dataset.seq_len
        pred_len = dataset.pred_len
        n_features = dataset.data_x[dataset.asset_ids[0]].shape[-1]
        n_time_features = dataset.dates[dataset.asset_ids[0]].shape[-1]
        obs_spaces = {
            "x_context": spaces.Box(
                low=-np.inf, high=np.inf, shape=(seq_len, n_features), dtype=np.float32
            ),
            "t_context": spaces.Box(
                low=-np.inf, high=np.inf, shape=(seq_len, n_time_features), dtype=np.float32
            ),
            "x_target": spaces.Box(
                low=-np.inf, high=np.inf, shape=(pred_len, n_features), dtype=np.float32
            ),
            "t_target": spaces.Box(
                low=-np.inf, high=np.inf, shape=(pred_len, n_time_features), dtype=np.float32
            ),
            "w_prev": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        }
        if include_asset_id:
            n_assets = int(getattr(dataset, "num_asset_ids", len(dataset.asset_ids)))
            obs_spaces["asset_id"] = spaces.Discrete(n_assets)
        if include_wealth:
            obs_spaces["wealth_feats"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            )

        self.observation_space = spaces.Dict(obs_spaces)
        if action_mode == "discrete_3":
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.env.rng = np.random.default_rng(seed)
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        obs = self.env.step(action)
        reward = float(obs.pop("reward"))
        done = bool(obs.pop("done"))
        info = obs.pop("info", {})
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info

    def set_transaction_cost(self, transaction_cost: float):
        self.env.transaction_cost = float(transaction_cost)
