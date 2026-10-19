"""Shared encoder and class-specific normalized projection heads for CLOSR.

Adapted from the architecture described by Wilkie et al. and their Apache-2.0
reference implementation at revision 79da4d1f2a6f4e5ecf92c4588195aaf163df40f8.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class CLOSRModel(nn.Module):
    """MLP encoder followed by one linear unit-sphere head per known class."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        projection_dim: int,
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or projection_dim <= 0 or num_classes <= 1:
            raise ValueError("input_dim/projection_dim must be positive and num_classes must exceed one")
        widths = list(hidden_dims)
        if not widths or any(width <= 0 for width in widths):
            raise ValueError("hidden_dims must contain positive widths")

        layers: list[nn.Module] = []
        previous = input_dim
        for width in widths:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous = width
        self.encoder = nn.Sequential(*layers)
        self.projection_heads = nn.ModuleList(
            nn.Linear(previous, projection_dim) for _ in range(num_classes)
        )
        self.input_dim = input_dim
        self.hidden_dims = widths
        self.projection_dim = projection_dim
        self.num_classes = num_classes
        self.dropout = float(dropout)

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.forward_features(features)
        projected = torch.stack([head(encoded) for head in self.projection_heads], dim=1)
        return F.normalize(projected, p=2, dim=-1)

    def configuration(self) -> dict[str, object]:
        return {
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "projection_dim": self.projection_dim,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
        }
