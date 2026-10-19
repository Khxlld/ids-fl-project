"""Portable CLOSR model bundle save/reload helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .federated import clone_state_dict
from .model import CLOSRModel


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_model_bundle(
    path: Path,
    model: CLOSRModel,
    centers: torch.Tensor,
    threshold: float,
    class_names: list[str],
    feature_names: list[str],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_config": model.configuration(),
            "model_state": clone_state_dict(model),
            "centers": centers.detach().cpu().clone(),
            "threshold": float(threshold),
            "class_names": list(class_names),
            "feature_names": list(feature_names),
            "metadata": json_ready(metadata),
        },
        path,
    )


def load_model_bundle(path: Path) -> tuple[CLOSRModel, torch.Tensor, float, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported CLOSR bundle version")
    model = CLOSRModel(**payload["model_config"])
    model.load_state_dict(payload["model_state"])
    model.eval()
    metadata = {
        "class_names": payload["class_names"],
        "feature_names": payload["feature_names"],
        "metadata": payload["metadata"],
    }
    return model, payload["centers"], float(payload["threshold"]), metadata
