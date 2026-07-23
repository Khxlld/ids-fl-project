"""The shared MLP model and reproducibility helpers.

The same architecture is used by the centralized baseline and by every
federated client, which is what makes FedAvg parameter averaging valid.
"""

from __future__ import annotations

import random
from typing import List

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    """Fully connected binary classifier producing a single logit.

    Layout (defaults): Input -> 128 -> ReLU -> Dropout(0.20)
                              -> 64  -> ReLU -> Dropout(0.10)
                              -> 32  -> ReLU
                              -> 1 (logit).

    The final layer outputs a raw logit; use torch.sigmoid() to get a
    probability. Loss is BCEWithLogitsLoss, which applies the sigmoid
    internally in a numerically stable way.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = (128, 64, 32),
        dropout_rates: List[float] = (0.20, 0.10),
    ) -> None:
        super().__init__()
        hidden_layers = list(hidden_layers)
        dropout_rates = list(dropout_rates)

        layers: List[nn.Module] = []
        prev = input_dim
        for i, width in enumerate(hidden_layers):
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            # Apply a dropout after a hidden layer only if one is defined for it.
            if i < len(dropout_rates):
                layers.append(nn.Dropout(dropout_rates[i]))
            prev = width
        layers.append(nn.Linear(prev, 1))  # single logit output
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(input_dim: int, cfg) -> MLP:
    """Construct the MLP from a Config object."""
    return MLP(
        input_dim=input_dim,
        hidden_layers=cfg.hidden_layers,
        dropout_rates=cfg.dropout_rates,
    )
