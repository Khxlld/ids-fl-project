"""Deterministic binary MLP training, metrics, and explicit FedAvg helpers."""

from __future__ import annotations

import copy
import math
import random
import time
from collections import OrderedDict
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset


def set_deterministic(seed: int, num_threads: int = 1) -> None:
    """Configure deterministic CPU-oriented execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


class BinaryMLP(nn.Module):
    """Fully connected binary classifier returning one raw logit."""

    def __init__(self, input_dim: int, hidden_layers: Iterable[int], dropout: Iterable[float]):
        super().__init__()
        widths = list(hidden_layers)
        dropouts = list(dropout)
        layers: list[nn.Module] = []
        previous = input_dim
        for index, width in enumerate(widths):
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            rate = dropouts[index] if index < len(dropouts) else 0.0
            if rate > 0:
                layers.append(nn.Dropout(rate))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def clone_state_dict(model_or_state) -> OrderedDict[str, torch.Tensor]:
    state = model_or_state.state_dict() if hasattr(model_or_state, "state_dict") else model_or_state
    return OrderedDict((name, value.detach().cpu().clone()) for name, value in state.items())


def _loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).view(-1, 1),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


def _criterion(pos_weight: float | None, device: torch.device) -> nn.Module:
    if pos_weight is None:
        return nn.BCEWithLogitsLoss()
    weight = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=weight)


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    model.to(device).eval()
    values = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    output: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        logits = model(values[start : start + batch_size].to(device))
        output.append(torch.sigmoid(logits).cpu().numpy().ravel())
    return np.concatenate(output) if output else np.empty(0, dtype=np.float32)


def metric_bundle(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def division(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    both_classes = len(np.unique(y_true)) == 2
    return {
        "rows": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "attack_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "attack_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "attack_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "normal_precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "normal_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "normal_f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if both_classes else float("nan"),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": division(fp, fp + tn),
        "fnr": division(fn, fn + tp),
    }


def _selection_key(metrics: dict) -> tuple[float, float, float, float]:
    return (
        float(metrics["macro_f1"]),
        float(metrics["attack_recall"]),
        -float(metrics["fpr"]),
        -float(metrics["log_loss"]),
    )


def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    minimum: float,
    maximum: float,
    step: float,
) -> tuple[float, dict]:
    thresholds = np.round(np.arange(minimum, maximum + step / 2, step), 10)
    best_threshold = 0.5
    best_metrics = metric_bundle(y_true, y_prob, best_threshold)
    best_key = (
        best_metrics["macro_f1"],
        best_metrics["attack_recall"],
        -best_metrics["fpr"],
        -abs(best_threshold - 0.5),
    )
    for threshold in thresholds:
        metrics = metric_bundle(y_true, y_prob, float(threshold))
        key = (
            metrics["macro_f1"],
            metrics["attack_recall"],
            -metrics["fpr"],
            -abs(float(threshold) - 0.5),
        )
        if key > best_key:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_key = key
    return best_threshold, best_metrics


def evaluate_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    threshold: float,
) -> tuple[dict, np.ndarray]:
    probabilities = predict_proba(model, X, device)
    return metric_bundle(y, probabilities, threshold), probabilities


def train_with_early_stopping(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    maximum_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    pos_weight: float | None,
    seed: int,
    checkpoint_threshold: float = 0.5,
) -> tuple[nn.Module, list[dict], dict]:
    loader = _loader(X_train, y_train, batch_size, True, seed)
    criterion = _criterion(pos_weight, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.to(device)
    best_state = clone_state_dict(model)
    best_metrics: dict | None = None
    best_epoch = 0
    stale = 0
    history: list[dict] = []
    started = time.perf_counter()

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(features)
            seen += len(features)

        validation_probabilities = predict_proba(model, X_validation, device)
        validation_metrics = metric_bundle(y_validation, validation_probabilities, checkpoint_threshold)
        history.append({
            "epoch": epoch,
            "train_weighted_loss": loss_sum / max(seen, 1),
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        })
        if best_metrics is None or _selection_key(validation_metrics) > _selection_key(best_metrics):
            best_state = clone_state_dict(model)
            best_metrics = validation_metrics
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    assert best_metrics is not None
    metadata = {
        "best_epoch": best_epoch,
        "epochs_executed": len(history),
        "training_seconds": time.perf_counter() - started,
        "checkpoint_metrics_at_0_5": best_metrics,
    }
    return model, history, metadata


def train_local_epochs(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    pos_weight: float | None,
    seed: int,
) -> float:
    loader = _loader(X, y, batch_size, True, seed)
    criterion = _criterion(pos_weight, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.to(device)
    final_loss = 0.0
    for _ in range(epochs):
        model.train()
        loss_sum = 0.0
        seen = 0
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(features)
            seen += len(features)
        final_loss = loss_sum / max(seen, 1)
    return final_loss


def fedavg(states: list[OrderedDict[str, torch.Tensor]], sample_counts: list[int]) -> OrderedDict[str, torch.Tensor]:
    assert states and len(states) == len(sample_counts)
    assert all(count > 0 for count in sample_counts)
    total = float(sum(sample_counts))
    keys = list(states[0])
    assert all(list(state) == keys for state in states)
    averaged: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in keys:
        accumulator = torch.zeros_like(states[0][key], dtype=torch.float64)
        for state, count in zip(states, sample_counts):
            accumulator += state[key].to(dtype=torch.float64) * (count / total)
        averaged[key] = accumulator.to(dtype=states[0][key].dtype)
    return averaged
