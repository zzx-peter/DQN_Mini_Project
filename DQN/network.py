"""
network.py — Q-Network definition for DQN.

A fully-connected MLP that maps an observation vector to one Q-value per action.

Architecture:
    Input  → Linear(obs_dim, h0) → ReLU
           → Linear(h0, h1)      → ReLU
           → …
           → Linear(h_{k-1}, n_actions)   (no activation on output)

Default hidden sizes: (256, 256) — works well for n up to ~20.
Increase to (512, 512) for larger n if needed.
"""

import torch
import torch.nn as nn
from typing import Sequence


class QNetwork(nn.Module):
    """
    Multi-layer perceptron Q-network.

    Args:
        obs_dim     : Dimension of the flattened observation vector.
        n_actions   : Number of discrete actions (2 for bit generation).
        hidden_sizes: Sizes of hidden layers, e.g. (256, 256).
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 2,
        hidden_sizes: Sequence[int] = (256, 256),
    ):
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Observation batch of shape (batch, obs_dim).
        Returns:
            Q-values of shape (batch, n_actions).
        """
        return self.net(x)
