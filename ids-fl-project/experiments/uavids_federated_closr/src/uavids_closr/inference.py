"""Center computation, CLOSR scoring, calibration, and evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


@dataclass(frozen=True)
class CenterStatistics:
    sums: torch.Tensor
    counts: torch.Tensor


@torch.no_grad()
def local_center_statistics(
    model: torch.nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    device: torch.device,
    batch_size: int = 1024,
) -> CenterStatistics:
    model.to(device).eval()
    projection_dim = int(model.projection_dim)
    sums = torch.zeros((num_classes, projection_dim), dtype=torch.float64)
    counts = torch.zeros(num_classes, dtype=torch.int64)
    for start in range(0, len(labels), batch_size):
        end = min(start + batch_size, len(labels))
        x = torch.from_numpy(
            np.array(features[start:end], dtype=np.float32, order="C", copy=True)
        ).to(device)
        y = torch.from_numpy(np.array(labels[start:end], dtype=np.int64, order="C", copy=True))
        embeddings = model(x).cpu().to(dtype=torch.float64)
        for class_index in range(num_classes):
            mask = y.eq(class_index)
            if mask.any():
                sums[class_index] += embeddings[mask, class_index, :].sum(dim=0)
                counts[class_index] += int(mask.sum().item())
    return CenterStatistics(sums=sums, counts=counts)


def aggregate_center_statistics(statistics: Iterable[CenterStatistics]) -> CenterStatistics:
    items = list(statistics)
    if not items:
        raise ValueError("at least one center statistic is required")
    sums = torch.stack([item.sums for item in items]).sum(dim=0)
    counts = torch.stack([item.counts for item in items]).sum(dim=0)
    return CenterStatistics(sums=sums, counts=counts)


def centers_from_statistics(statistics: CenterStatistics) -> torch.Tensor:
    if torch.any(statistics.counts <= 0):
        missing = torch.flatnonzero(statistics.counts <= 0).tolist()
        raise ValueError(f"cannot compute globally missing centers: {missing}")
    norms = torch.linalg.vector_norm(statistics.sums, dim=-1, keepdim=True)
    if torch.any(norms <= 1e-12) or not torch.isfinite(norms).all():
        raise ValueError("center sum is zero or non-finite")
    return (statistics.sums / norms).to(dtype=torch.float32)


@torch.no_grad()
def embeddings_for(
    model: torch.nn.Module,
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> torch.Tensor:
    model.to(device).eval()
    chunks: list[torch.Tensor] = []
    for start in range(0, len(features), batch_size):
        values = torch.from_numpy(
            np.array(
                features[start : start + batch_size], dtype=np.float32, order="C", copy=True
            )
        ).to(device)
        chunks.append(model(values).cpu())
    return torch.cat(chunks, dim=0) if chunks else torch.empty((0, model.num_classes, model.projection_dim))


def similarities_and_unknown_scores(embeddings: torch.Tensor, centers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if embeddings.ndim != 3 or centers.ndim != 2:
        raise ValueError("expected embeddings [B,C,D] and centers [C,D]")
    normalized_centers = torch.nn.functional.normalize(centers, p=2, dim=-1)
    normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    similarities = torch.sum(normalized_embeddings * normalized_centers.unsqueeze(0), dim=-1)
    probabilities = torch.softmax(similarities, dim=-1)
    unknown_scores = -torch.sum(probabilities * similarities.square(), dim=-1)
    return similarities, unknown_scores


@torch.no_grad()
def score_features(
    model: torch.nn.Module,
    centers: torch.Tensor,
    features: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = embeddings_for(model, features, device)
    similarities, unknown_scores = similarities_and_unknown_scores(embeddings, centers)
    return similarities.numpy(), unknown_scores.numpy()


def calibrate_known_threshold(scores: np.ndarray, known_rejection_rate_target: float) -> tuple[float, float]:
    if not 0 < known_rejection_rate_target < 1:
        raise ValueError("known_rejection_rate_target must lie in (0, 1)")
    if len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("finite known calibration scores are required")
    # The deployed rule is strictly ``score > threshold``.  Choose an observed
    # order statistic that never rejects more than floor(target * n) known
    # calibration rows.  Ties at the threshold can make the achieved rate
    # lower than the requested target; they are never split arbitrarily.
    maximum_rejections = int(np.floor(known_rejection_rate_target * len(scores)))
    if maximum_rejections == 0:
        threshold = float(np.max(scores))
    else:
        threshold = float(np.sort(scores)[len(scores) - maximum_rejections - 1])
    observed = float(np.mean(scores > threshold))
    return threshold, observed


def calibration_metrics(
    similarities: np.ndarray,
    unknown_scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    class_names: list[str],
    benign_index: int = 0,
) -> dict[str, object]:
    """Known-only rejection and benign alert diagnostics at a fixed threshold."""
    if len(labels) == 0 or len(similarities) != len(labels) or len(unknown_scores) != len(labels):
        raise ValueError("aligned non-empty calibration arrays are required")
    rejected = unknown_scores > threshold
    closed_set_predictions = np.argmax(similarities, axis=1)
    final_predictions = closed_set_predictions.astype(np.int64, copy=True)
    final_predictions[rejected] = -1
    per_class: dict[str, dict[str, float | int]] = {}
    for class_index, class_name in enumerate(class_names):
        mask = labels == class_index
        per_class[class_name] = {
            "rows": int(mask.sum()),
            "known_rejection_rate": float(np.mean(rejected[mask])) if mask.any() else float("nan"),
        }
    benign_mask = labels == benign_index
    benign_rejection = float(np.mean(rejected[benign_mask])) if benign_mask.any() else float("nan")
    benign_closed_set_attack = (
        float(np.mean((~rejected[benign_mask]) & (closed_set_predictions[benign_mask] != benign_index)))
        if benign_mask.any()
        else float("nan")
    )
    benign_total_alert = (
        float(np.mean(final_predictions[benign_mask] != benign_index))
        if benign_mask.any()
        else float("nan")
    )
    return {
        "known_rejection_rate_overall": float(np.mean(rejected)),
        "known_rejections": int(rejected.sum()),
        "threshold_tie_rows": int(np.sum(unknown_scores == threshold)),
        "per_class": per_class,
        "benign_rejection_rate": benign_rejection,
        "benign_closed_set_attack_prediction_rate": benign_closed_set_attack,
        "benign_total_alert_fpr": benign_total_alert,
    }


def known_open_set_diagnostics(
    similarities: np.ndarray,
    unknown_scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    class_names: list[str],
    benign_index: int = 0,
) -> dict[str, object]:
    """Detailed known-only diagnostics after applying a calibrated threshold."""
    if len(labels) == 0 or len(similarities) != len(labels) or len(unknown_scores) != len(labels):
        raise ValueError("aligned non-empty known arrays are required")
    if set(np.unique(labels)) - set(range(len(class_names))):
        raise ValueError("known_open_set_diagnostics received a non-known label")
    closed_predictions = np.argmax(similarities, axis=1).astype(np.int64)
    rejected = unknown_scores > threshold
    predictions = closed_predictions.copy()
    predictions[rejected] = -1
    matrix_labels = list(range(len(class_names))) + [-1]
    matrix = confusion_matrix(labels, predictions, labels=matrix_labels)
    per_class: dict[str, dict[str, float | int]] = {}
    for class_index, class_name in enumerate(class_names):
        mask = labels == class_index
        per_class[class_name] = {
            "rows": int(mask.sum()),
            "closed_set_recall": float(np.mean(closed_predictions[mask] == class_index)),
            "recall_after_rejection": float(np.mean(predictions[mask] == class_index)),
            "rejection_rate": float(np.mean(rejected[mask])),
        }
    benign_mask = labels == benign_index
    return {
        "rows": int(len(labels)),
        "threshold": float(threshold),
        "prediction_columns": class_names + ["UNKNOWN"],
        "true_rows": class_names + ["UNKNOWN"],
        "confusion_matrix": matrix.tolist(),
        "closed_set_accuracy": float(accuracy_score(labels, closed_predictions)),
        "closed_set_macro_f1": float(
            f1_score(labels, closed_predictions, labels=list(range(len(class_names))), average="macro", zero_division=0)
        ),
        "macro_f1_after_rejection": float(
            f1_score(labels, predictions, labels=list(range(len(class_names))), average="macro", zero_division=0)
        ),
        "known_rejection_rate": float(np.mean(rejected)),
        "per_class": per_class,
        "benign_rejection_rate": float(np.mean(rejected[benign_mask])),
        "benign_total_alert_fpr": float(np.mean(predictions[benign_mask] != benign_index)),
    }


def known_class_metrics(similarities: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    predictions = np.argmax(similarities, axis=1)
    return {
        "rows": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def open_set_metrics(
    similarities: np.ndarray,
    unknown_scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    num_known_classes: int,
    benign_index: int = 0,
    class_names: list[str] | None = None,
) -> dict[str, object]:
    """Evaluate labels encoded as known indices and -1 for globally unknown."""
    known_mask = labels >= 0
    unknown_mask = labels == -1
    closed_set_predictions = np.argmax(similarities, axis=1).astype(np.int64)
    rejected = unknown_scores > threshold
    predictions = closed_set_predictions.copy()
    predictions[rejected] = -1
    known_macro_f1 = f1_score(
        labels[known_mask],
        predictions[known_mask],
        labels=list(range(num_known_classes)),
        average="macro",
        zero_division=0,
    )
    benign_mask = labels == benign_index
    benign_alert_fpr = float(np.mean(predictions[benign_mask] != benign_index)) if benign_mask.any() else float("nan")
    benign_rejection = float(np.mean(rejected[benign_mask])) if benign_mask.any() else float("nan")
    benign_closed_set_attack = (
        float(np.mean((~rejected[benign_mask]) & (closed_set_predictions[benign_mask] != benign_index)))
        if benign_mask.any()
        else float("nan")
    )
    unknown_recall = float(np.mean(predictions[unknown_mask] == -1)) if unknown_mask.any() else float("nan")
    binary_unknown = unknown_mask.astype(np.int64)
    auroc = float(roc_auc_score(binary_unknown, unknown_scores)) if len(np.unique(binary_unknown)) == 2 else float("nan")
    names = class_names or [str(class_index) for class_index in range(num_known_classes)]
    if len(names) != num_known_classes:
        raise ValueError("class_names must match num_known_classes")
    per_class_rejection: dict[str, dict[str, float | int]] = {}
    for class_index, class_name in enumerate(names):
        mask = labels == class_index
        per_class_rejection[class_name] = {
            "rows": int(mask.sum()),
            "known_rejection_rate": float(np.mean(rejected[mask])) if mask.any() else float("nan"),
        }
    return {
        "rows": int(len(labels)),
        "known_rows": int(known_mask.sum()),
        "unknown_rows": int(unknown_mask.sum()),
        "known_class_macro_f1": float(known_macro_f1),
        "known_rejection_rate_overall": float(np.mean(rejected[known_mask])) if known_mask.any() else float("nan"),
        "per_class": per_class_rejection,
        "unknown_recall": unknown_recall,
        "benign_rejection_rate": benign_rejection,
        "benign_closed_set_attack_prediction_rate": benign_closed_set_attack,
        "benign_total_alert_fpr": benign_alert_fpr,
        "known_vs_unknown_auroc": auroc,
    }
