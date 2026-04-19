"""
env.py — Custom Gymnasium environment for bit-sequence matching.

Observation (fixed): shape `(2)` — `[target at current step, step/n]`.
The input dimension does not depend on n, so one Q-network can serve all lengths.

Action: Discrete(2) — choose 0 or 1 for the next position.

Reward (every step, for the bit just placed):
    Shaped (default) →  (1/n) if this bit matches target[position], else 0.
    Sparse (--no_shaping) → 1.0 if this bit matches, else 0.0.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class BitSequenceEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n: int = 8,
        reward_shaping: bool = True,
        render_mode=None,
    ):
        super().__init__()
        assert 1 <= n <= 50, "Sequence length n must be in [1, 50]."
        self.n = n
        self.reward_shaping = reward_shaping
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.target: np.ndarray = np.zeros(n, dtype=np.float32)
        self.generated: np.ndarray = np.zeros(n, dtype=np.float32)
        self.step_count: int = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.target = self.np_random.integers(0, 2, self.n).astype(np.float32)
        self.generated = np.zeros(self.n, dtype=np.float32)
        self.step_count = 0
        return self._obs(), {}

    def step(self, action: int):
        if self.step_count >= self.n:
            raise RuntimeError("Episode already finished — call reset() first.")

        idx = self.step_count
        self.generated[idx] = float(action)
        self.step_count += 1

        bit_correct = float(self.generated[idx]) == float(self.target[idx])
        if self.reward_shaping:
            reward = (1.0 / self.n) if bit_correct else 0.0
        else:
            reward = 1.0 if bit_correct else 0.0

        terminated = self.step_count == self.n
        truncated = False

        if terminated:
            success = bool(np.array_equal(self.generated, self.target))
            info = {"success": success}
        else:
            info = {"success": False}

        return self._obs(), reward, terminated, truncated, info

    def render(self):
        t = self.target.astype(int).tolist()
        g = self.generated[: self.step_count].astype(int).tolist()
        print(f"  Target   : {t}")
        print(f"  Generated: {g}  (step {self.step_count}/{self.n})")

    def _obs(self) -> np.ndarray:
        if self.step_count >= self.n:
            return np.array([0.0, 1.0], dtype=np.float32)
        return np.array(
            [
                float(self.target[self.step_count]),
                self.step_count / self.n,
            ],
            dtype=np.float32,
        )
