"""Reusable modeling components for the UAVIDS federated IDS experiment."""

from .modeling import (
    BinaryMLP,
    clone_state_dict,
    evaluate_model,
    fedavg,
    metric_bundle,
    predict_proba,
    select_threshold,
    set_deterministic,
    train_local_epochs,
    train_with_early_stopping,
)

__all__ = [
    "BinaryMLP",
    "clone_state_dict",
    "evaluate_model",
    "fedavg",
    "metric_bundle",
    "predict_proba",
    "select_threshold",
    "set_deterministic",
    "train_local_epochs",
    "train_with_early_stopping",
]
