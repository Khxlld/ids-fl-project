"""PyTorch training / evaluation loops and Flower parameter helpers."""

from __future__ import annotations

import copy
import time
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .evaluation import compute_metrics


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """Wrap arrays into a DataLoader of float32 tensors (target shape [N, 1])."""
    ds = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).view(-1, 1),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_epochs(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> float:
    """Train `model` in place for `epochs`; return the last-epoch mean loss."""
    model.to(device).train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    last_loss = 0.0
    for _ in range(epochs):
        running, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        last_loss = running / max(n, 1)
    return last_loss


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    """Return attack probabilities (sigmoid of logits) for every row of X."""
    model.to(device).eval()
    probs: List[np.ndarray] = []
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    for i in range(0, len(Xt), batch_size):
        xb = Xt[i : i + batch_size].to(device)
        p = torch.sigmoid(model(xb)).cpu().numpy().ravel()
        probs.append(p)
    return np.concatenate(probs) if probs else np.array([], dtype=np.float32)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    threshold: float,
) -> Dict[str, float]:
    """Compute BCE loss + full metric bundle on (X, y)."""
    model.to(device).eval()
    criterion = nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(device)
    yt = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).view(-1, 1).to(device)
    logits = model(Xt)
    loss = criterion(logits, yt).item()
    probs = torch.sigmoid(logits).cpu().numpy().ravel()
    return compute_metrics(y, probs, threshold=threshold, loss=loss)


def train_centralized(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg,
    device: torch.device,
    epochs: int | None = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, List[float]], Dict[str, float], float]:
    """Train centrally with early stopping on validation F1.

    Returns (best_model, history, best_val_metrics, train_time_seconds).
    The returned model has the best-F1 weights loaded.
    """
    epochs = epochs or cfg.centralized_epochs
    train_loader = make_loader(X_train, y_train, cfg.batch_size, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    model.to(device)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_val_metrics: Dict[str, float] = {}
    epochs_without_improve = 0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = running / max(n, 1)

        train_metrics = evaluate(model, X_train, y_train, device, cfg.decision_threshold)
        val_metrics = evaluate(model, X_val, y_val, device, cfg.decision_threshold)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])

        improved = val_metrics["f1"] > best_f1
        if improved:
            best_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_val_metrics = val_metrics
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if verbose:
            print(
                f"  epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_f1={val_metrics['f1']:.4f}"
                + ("  <-- best" if improved else "")
            )

        if epochs_without_improve >= cfg.early_stopping_patience:
            if verbose:
                print(f"  early stopping at epoch {epoch} (no val-F1 gain for "
                      f"{cfg.early_stopping_patience} epochs)")
            break

    train_time = time.time() - t0
    model.load_state_dict(best_state)
    return model, history, best_val_metrics, train_time


# --------------------------------------------------------------------------- #
# Flower parameter <-> numpy helpers (state_dict order is stable across clients)
# --------------------------------------------------------------------------- #
def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Model weights as a list of NumPy arrays (Flower's parameter format)."""
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Load a list of NumPy arrays back into the model's state_dict."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
