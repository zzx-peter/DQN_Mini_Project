"""
agent.py — Experience Replay buffer and DQN agent.

DQN components implemented here:
  1. ReplayBuffer  — fixed-capacity circular buffer for (s, a, r, s', done) tuples.
  2. DQNAgent      — wraps two QNetworks (online + target), optimizer, and replay
                     buffer; exposes select_action(), push(), update(), save(), load().

Epsilon is a public attribute; the training loop is responsible for scheduling it
(see train.py), keeping decay logic separate from the agent internals.
"""

import random
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from network import QNetwork


# ---------------------------------------------------------------------------
class ReplayBuffer:
    """
    Circular replay buffer.  Stores transitions as numpy arrays for efficiency.
    """

    def __init__(self, capacity: int = 30_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append(
            (
                np.array(state, dtype=np.float32),
                int(action),
                float(reward),
                np.array(next_state, dtype=np.float32),
                float(done),
            )
        )

    def sample(self, batch_size: int):
        """Return a random mini-batch as numpy arrays."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            states.astype(np.float32),
            actions.astype(np.int64),
            rewards.astype(np.float32),
            next_states.astype(np.float32),
            dones.astype(np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
class DQNAgent:
    """
    Standard DQN agent:
      - Online Q-network updated every step.
      - Target network hard-synced every `target_update_freq` gradient steps.
      - Huber loss (SmoothL1) for training stability.
      - Gradient clipping to prevent exploding gradients.

    Epsilon scheduling is left to the caller; set agent.epsilon directly.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 2,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 30_000,
        batch_size: int = 128,
        target_update_freq: int = 200,
        epsilon: float = 1.0,
        epsilon_end: float = 0.05,
        hidden_sizes: Sequence[int] = (256, 256),
        device: str = "cpu",
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon          # current exploration rate (set externally)
        self.epsilon_end = epsilon_end  # minimum exploration rate
        self.device = torch.device(device)
        self._update_count = 0

        # Online and target networks
        self.q_net = QNetwork(obs_dim, n_actions, hidden_sizes).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """
        Epsilon-greedy action selection.
        Pass greedy=True at evaluation time to act purely greedily.
        """
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.q_net(t).argmax(dim=1).item())

    # ------------------------------------------------------------------
    def push(self, state, action, reward, next_state, done) -> None:
        """Store a transition in the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    def update(self) -> float | None:
        """
        One gradient update step.
        Returns the scalar loss, or None if the buffer has too few samples.
        """
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Q(s, a) for the chosen actions
        q_pred = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Bellman target: r + γ · max_a' Q_target(s', a') · (1 − done)
        with torch.no_grad():
            q_next = self.target_net(next_states_t).max(dim=1)[0]
            q_target = rewards_t + self.gamma * q_next * (1.0 - dones_t)

        loss = self.loss_fn(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to avoid large updates
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Periodically sync the target network
        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save network weights, optimizer state, and epsilon to a checkpoint."""
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "update_count": self._update_count,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load a checkpoint saved by save()."""
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_end)
        self._update_count = ckpt.get("update_count", 0)
