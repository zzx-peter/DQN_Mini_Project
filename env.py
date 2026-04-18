"""
env.py — Custom Gymnasium environment for bit-sequence matching.

The agent generates a binary sequence of length n, one token per timestep.
A random target sequence is drawn at the start of each episode.
The episode ends after exactly n steps; reward is issued only at the final step.

Observation (shape = 2n+1):
    [target_0, …, target_{n-1},   # the target bit-string (n floats ∈ {0,1})
     gen_0,    …, gen_{n-1},      # bits generated so far (0-padded)
     step / n]                    # normalised progress scalar ∈ [0, 1]

Action: Discrete(2) — choose 0 or 1 for the next position.

Reward (at terminal step only):
    Shaped  → fraction of positions that match  (∈ [0, 1])
    Sparse  → 1.0 if fully correct, else 0.0
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class BitSequenceEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, n: int = 8, reward_shaping: bool = True, render_mode=None):
        super().__init__()
        assert 1 <= n <= 50, "Sequence length n must be in [1, 50]."
        self.n = n
        self.reward_shaping = reward_shaping
        self.render_mode = render_mode

        # Two actions: append a 0 or a 1
        self.action_space = spaces.Discrete(2)

        # Observation: target (n) + generated (n) + normalised step (1)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2 * n + 1,), dtype=np.float32
        )

        # Internal state (initialised properly in reset)
        self.target: np.ndarray = np.zeros(n, dtype=np.float32)
        self.generated: np.ndarray = np.zeros(n, dtype=np.float32)
        self.step_count: int = 0

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # seeds self.np_random
        self.target = self.np_random.integers(0, 2, self.n).astype(np.float32)
        self.generated = np.zeros(self.n, dtype=np.float32)
        self.step_count = 0
        return self._obs(), {}

    # ------------------------------------------------------------------
    def step(self, action: int):
        if self.step_count >= self.n:
            raise RuntimeError("Episode already finished — call reset() first.")

        self.generated[self.step_count] = float(action)
        self.step_count += 1

        terminated = self.step_count == self.n
        truncated = False

        if terminated:
            success = bool(np.array_equal(self.generated, self.target))
            if self.reward_shaping:
                # Partial credit: each correct bit contributes 1/n
                reward = float(np.sum(self.generated == self.target)) / self.n
            else:
                reward = 1.0 if success else 0.0
            info = {"success": success}
        else:
            reward = 0.0
            info = {"success": False}

        return self._obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self):
        t = self.target.astype(int).tolist()
        g = self.generated[: self.step_count].astype(int).tolist()
        print(f"  Target   : {t}")
        print(f"  Generated: {g}  (step {self.step_count}/{self.n})")

    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        return np.concatenate(
            [self.target, self.generated, [self.step_count / self.n]]
        ).astype(np.float32)
