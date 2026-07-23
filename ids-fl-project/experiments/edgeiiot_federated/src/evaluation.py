"""Metrics and plotting for the binary intrusion-detection task.

All metric helpers use explicit zero-division handling so degenerate batches
never crash a run, and they report the full confusion-matrix breakdown that an
IDS evaluation needs (false-positive / false-negative rates in particular).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

CLASS_LABELS = ["Normal (0)", "Attack (1)"]


def bce_loss_from_prob(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-7) -> float:
    """Binary cross-entropy computed from probabilities (for reporting)."""
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    loss: float | None = None,
) -> Dict[str, float]:
    """Full metric bundle for binary IDS. Probabilities in, metrics out."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix forced to the 2x2 [[TN, FP], [FN, TP]] layout.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    def safe_div(a: float, b: float) -> float:
        return float(a / b) if b else 0.0

    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)  # a.k.a. detection rate / TPR
    f1 = safe_div(2 * precision * recall, precision + recall)
    fpr = safe_div(fp, fp + tn)  # false alarm rate
    fnr = safe_div(fn, fn + tp)  # missed-attack rate

    # ROC-AUC / PR-AUC need both classes present; guard against single-class y.
    if len(np.unique(y_true)) < 2:
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "fpr": fpr,
        "fnr": fnr,
    }
    if loss is not None:
        metrics["loss"] = float(loss)
    return metrics


def format_metrics(m: Dict[str, float]) -> str:
    """One-line human-readable summary of the key metrics."""
    return (
        f"acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
        f"rec={m['recall']:.4f} f1={m['f1']:.4f} "
        f"roc_auc={m['roc_auc']:.4f} pr_auc={m['pr_auc']:.4f}"
    )


# --------------------------------------------------------------------------- #
# Plotting helpers (imported lazily so `import evaluation` is cheap).
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib.pyplot as plt

    return plt


def plot_training_curves(history: Dict[str, List[float]], title_prefix: str = "Centralized"):
    plt = _plt()
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="validation")
    axes[0].set_title(f"{title_prefix} loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], marker="o", label="train")
    axes[1].plot(epochs, history["val_acc"], marker="o", label="validation")
    axes[1].set_title(f"{title_prefix} accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    fig.tight_layout()
    return fig


def plot_federated_rounds(round_df, metrics: Sequence[str] = ("val_loss", "val_acc", "val_f1")):
    plt = _plt()
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        ax.plot(round_df["round"], round_df[metric], marker="o")
        ax.set_title(f"Global {metric} by round")
        ax.set_xlabel("federated round")
        ax.set_ylabel(metric)
    fig.tight_layout()
    return fig


def plot_confusion(m: Dict[str, float], title: str = "Confusion matrix"):
    plt = _plt()
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred Normal", "Pred Attack"])
    ax.set_yticks([0, 1], labels=["True Normal", "True Attack"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_roc(y_true, y_prob, title: str = "ROC curve"):
    plt = _plt()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_pr(y_true, y_prob, title: str = "Precision-Recall curve"):
    plt = _plt()
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_client_distribution(dist_df, client_col="client", normal_col="normal", attack_col="attack"):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(dist_df[client_col], dist_df[normal_col], label="Normal")
    ax.bar(dist_df[client_col], dist_df[attack_col], bottom=dist_df[normal_col], label="Attack")
    ax.set_ylabel("rows")
    ax.set_title("IID client class distribution")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_comparison(centralized: Dict[str, float], federated: Dict[str, float],
                    keys: Sequence[str] = ("accuracy", "precision", "recall", "f1", "roc_auc")):
    plt = _plt()
    x = np.arange(len(keys))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, [centralized[k] for k in keys], w, label="Centralized")
    ax.bar(x + w / 2, [federated[k] for k in keys], w, label="Federated")
    ax.set_xticks(x, labels=keys, rotation=20)
    ax.set_ylim(0, 1.02)
    ax.set_title("Centralized vs Federated")
    ax.legend()
    for i, k in enumerate(keys):
        ax.text(i - w / 2, centralized[k] + 0.01, f"{centralized[k]:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, federated[k] + 0.01, f"{federated[k]:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    return fig
