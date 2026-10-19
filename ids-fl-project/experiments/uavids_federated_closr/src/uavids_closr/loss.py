"""Class-specific CLAD objective used by CLOSR.

The default path numerically matches ``losses/closr_loss.py`` in the official
Apache-2.0 implementation at revision 79da4d1f... .  Positive and negative
terms are handled independently: an unavailable term is a differentiable zero,
while the other term remains trainable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class HeadLossDiagnostic:
    class_index: int
    target_samples: int
    negative_samples: int
    positive_pairs: int
    negative_pairs: int
    positive_active_pairs: int
    negative_active_pairs: int
    positive_available: bool
    negative_available: bool
    active: bool


class CLOSRLoss(nn.Module):
    """One-vs-rest CLAD loss across the class-specific CLOSR heads."""

    def __init__(
        self,
        num_classes: int,
        margin: float = 1.0,
        squared: bool = True,
        alpha: float = 0.5,
        normalization: str = "official_active_pairs",
        head_reduction: str = "mean_all",
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if normalization not in {"official_active_pairs", "paper_all_pairs"}:
            raise ValueError(f"unsupported normalization: {normalization}")
        if head_reduction not in {"mean_all", "mean_active", "sum"}:
            raise ValueError(f"unsupported head reduction: {head_reduction}")
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        self.num_classes = int(num_classes)
        self.margin = float(margin)
        self.squared = bool(squared)
        self.alpha = float(alpha)
        self.normalization = normalization
        self.head_reduction = head_reduction
        self.epsilon = float(epsilon)
        self.last_diagnostics: tuple[HeadLossDiagnostic, ...] = ()

    def _mean_term(self, values: torch.Tensor, *, active_only: bool) -> torch.Tensor:
        if values.numel() == 0:
            return values.sum() * 0.0
        transformed = values.square() if self.squared else values
        if active_only:
            # Upstream explicitly casts the active-pair count to float32,
            # including when embeddings use float64 in parity tests.
            count = torch.count_nonzero(values > self.epsilon).float()
        else:
            count = values.new_tensor(float(values.numel()))
        return transformed.sum() / (count + self.epsilon)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [batch, classes, projection_dim]")
        if embeddings.shape[1] != self.num_classes:
            raise ValueError("embedding head count does not match num_classes")
        labels = labels.to(dtype=torch.long)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        by_head = embeddings.transpose(0, 1)
        similarities = torch.bmm(by_head, by_head.transpose(1, 2))
        distances = (1.0 - similarities) / 2.0
        batch_size = labels.numel()
        diagonal = torch.eye(batch_size, dtype=torch.bool, device=labels.device)
        equal_labels = labels[:, None].eq(labels[None, :])

        losses: list[torch.Tensor] = []
        diagnostics: list[HeadLossDiagnostic] = []
        active_losses: list[torch.Tensor] = []
        for class_index in range(self.num_classes):
            target = labels.eq(class_index)
            target_samples = int(target.sum().item())
            negative_samples = int((~target).sum().item())
            # Upstream includes the target-class diagonal in its mask.  Its
            # zero distances do not enter the active-pair denominator, but
            # retaining it here gives exact reference behavior (including
            # floating-point diagonal residuals).  The literal paper-pair
            # option excludes self-pairs.
            similar_mask = equal_labels & target[:, None]
            if self.normalization == "paper_all_pairs":
                similar_mask = similar_mask & ~diagonal
            negative_mask = target[:, None] & (~target)[None, :]
            positive_available = target_samples >= 2
            negative_available = target_samples >= 1 and negative_samples >= 1
            active = positive_available or negative_available
            zero = distances[class_index].sum() * 0.0
            positive_loss = zero
            negative_loss = zero
            positive_active_pairs = 0
            negative_active_pairs = 0
            if positive_available:
                positive_values = distances[class_index][similar_mask]
                positive_active_pairs = int(
                    torch.count_nonzero(positive_values > self.epsilon).item()
                )
                active_only = self.normalization == "official_active_pairs"
                positive_loss = self._mean_term(positive_values, active_only=active_only)
            if negative_available:
                negative_hinge = F.relu(self.margin - distances[class_index][negative_mask])
                negative_active_pairs = int(
                    torch.count_nonzero(negative_hinge > self.epsilon).item()
                )
                active_only = self.normalization == "official_active_pairs"
                negative_loss = self._mean_term(negative_hinge, active_only=active_only)
            if active:
                head_loss = self.alpha * positive_loss + (1.0 - self.alpha) * negative_loss
                active_losses.append(head_loss)
            else:
                head_loss = zero
            losses.append(head_loss)
            diagnostics.append(
                HeadLossDiagnostic(
                    class_index=class_index,
                    target_samples=target_samples,
                    negative_samples=negative_samples,
                    positive_pairs=int(similar_mask.sum().item()),
                    negative_pairs=int(negative_mask.sum().item()),
                    positive_active_pairs=positive_active_pairs,
                    negative_active_pairs=negative_active_pairs,
                    positive_available=positive_available,
                    negative_available=negative_available,
                    active=active,
                )
            )

        self.last_diagnostics = tuple(diagnostics)
        stacked = torch.stack(losses)
        if self.head_reduction == "sum":
            return stacked.sum()
        if self.head_reduction == "mean_active":
            return torch.stack(active_losses).mean() if active_losses else stacked.sum() * 0.0
        return stacked.mean()
