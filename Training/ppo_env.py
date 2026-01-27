from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


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
        allow_short: bool = True,
        include_wealth: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.episode_len = episode_len
        self.transaction_cost = transaction_cost
        self.allow_short = allow_short
        self.include_wealth = include_wealth
        self.seq_len = dataset.seq_len
        self.rng = np.random.default_rng(seed)

        if not getattr(dataset, "asset_ids", []):
            raise ValueError("Dataset must contain asset_ids to build episodes.")

        self.state: Optional[EnvState] = None

    def _sample_start(self) -> Tuple[str, int]:
        asset_id = self.rng.choice(self.dataset.asset_ids)
        data_len = len(self.dataset.data_x[asset_id])
        max_start = data_len - self.seq_len - self.episode_len - 1
        if max_start <= 0:
            max_start = max(0, data_len - self.seq_len - 2)
        start = int(self.rng.integers(0, max_start + 1))
        return asset_id, start

    def _get_close(self, asset_id: str, idx: int) -> float:
        return float(self.dataset.ohlcv[asset_id][idx][3])

    def _observe(self, asset_id: str, cursor: int, w_prev: float, wealth: float) -> Dict:
        x_context = self.dataset.data_x[asset_id][cursor : cursor + self.seq_len]
        t_context = self.dataset.dates[asset_id][cursor : cursor + self.seq_len]
        wealth_feats = np.array([np.log(wealth)], dtype=np.float32) if self.include_wealth else np.array([], dtype=np.float32)
        return {
            "x_context": x_context.astype(np.float32),
            "t_context": t_context.astype(np.float32),
            "w_prev": np.array([w_prev], dtype=np.float32),
            "wealth_feats": wealth_feats,
        }

    def reset(self) -> Dict:
        asset_id, start = self._sample_start()
        self.state = EnvState(asset_id=asset_id, cursor=start, step=0, w_prev=0.0, wealth=1.0)
        return self._observe(asset_id, start, self.state.w_prev, self.state.wealth)

    def step(self, action: np.ndarray) -> Dict:
        if self.state is None:
            raise RuntimeError("Environment must be reset before stepping.")

        w_t = float(np.clip(action, -1.0, 1.0))
        if not self.allow_short:
            w_t = max(0.0, w_t)

        asset_id = self.state.asset_id
        cursor = self.state.cursor
        close_t = self._get_close(asset_id, cursor + self.seq_len - 1)
        close_tp1 = self._get_close(asset_id, cursor + self.seq_len)
        r_tp1 = close_tp1 / close_t - 1.0

        turnover = abs(w_t - self.state.w_prev)
        reward = w_t * r_tp1 - self.transaction_cost * turnover
        self.state.wealth *= 1.0 + reward

        self.state.cursor += 1
        self.state.step += 1
        self.state.w_prev = w_t

        done = (
            self.state.step >= self.episode_len
            or (self.state.cursor + self.seq_len + 1) >= len(self.dataset.data_x[asset_id])
        )

        obs = self._observe(asset_id, self.state.cursor, self.state.w_prev, self.state.wealth)
        obs["reward"] = np.array(reward, dtype=np.float32)
        obs["done"] = np.array(done, dtype=np.float32)
        obs["info"] = {
            "turnover": turnover,
            "return": r_tp1,
            "wealth": self.state.wealth,
        }
        return obs
