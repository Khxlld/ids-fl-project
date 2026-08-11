"""Generate a presentation-safe 24-flow stream with the frozen Phase 3 model.

This is an offline evidence-generation helper, not a runtime GUI dependency. It
reads the locked validation partition and frozen artifacts, performs inference
only, and writes prediction responses without feature values or source labels.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from gui_integration.backend import FrozenBinaryIDS  # noqa: E402
from uavids_fl import predict_proba  # noqa: E402


OUTPUT = ROOT / "data" / "inference_stream.json"
NORMAL_COUNT = 16
ATTACK_COUNT = 8


def spread_indices(pool: np.ndarray, probabilities: np.ndarray, count: int) -> list[int]:
    """Choose distinct, probability-spread rows without consulting labels."""

    ordered = pool[np.argsort(probabilities[pool])]
    positions = np.linspace(0, len(ordered) - 1, count + 2, dtype=int)[1:-1]
    selected = [int(ordered[position]) for position in positions]
    if len(set(selected)) != count:
        raise RuntimeError("probability-spread selection did not produce distinct rows")
    return selected


def interleave(normal: list[int], attack: list[int]) -> list[int]:
    """Create a readable N,N,A rhythm while preserving distinct records."""

    ordered: list[int] = []
    normal_cursor = 0
    for attack_index in attack:
        ordered.extend(normal[normal_cursor : normal_cursor + 2])
        normal_cursor += 2
        ordered.append(attack_index)
    ordered.extend(normal[normal_cursor:])
    return ordered


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    ids = FrozenBinaryIDS(EXPERIMENT_ROOT)
    validation_path = EXPERIMENT_ROOT / "partitions" / "phase2" / "validation.csv"
    validation = pd.read_csv(validation_path)
    feature_values = validation[list(ids.features)].to_numpy(dtype=float)
    finite = np.isfinite(feature_values).all(axis=1)
    candidates = validation.loc[finite, list(ids.features)].reset_index(drop=True)
    transformed = np.asarray(ids.preprocessor.transform(candidates), dtype=np.float32)
    probabilities = predict_proba(ids.model, transformed, torch.device("cpu"))

    normal_pool = np.flatnonzero(probabilities < ids.threshold)
    attack_pool = np.flatnonzero(probabilities >= ids.threshold)
    if len(normal_pool) < NORMAL_COUNT or len(attack_pool) < ATTACK_COUNT:
        raise RuntimeError("validation partition does not contain enough predicted examples")

    normal = spread_indices(normal_pool, probabilities, NORMAL_COUNT)
    attack = spread_indices(attack_pool, probabilities, ATTACK_COUNT)
    selected = interleave(normal, attack)
    generated_utc = utc_now()
    predictions = []
    for sequence, row_index in enumerate(selected, start=1):
        request = {
            "record_id": f"sponsor-flow-{sequence:03d}",
            "source": "recorded-validation-sample",
            "features": {name: float(candidates.iloc[row_index][name]) for name in ids.features},
        }
        prediction = ids.predict(request)
        predictions.append(
            {
                **prediction,
                "prediction_id": f"recorded-stream-{sequence:03d}",
                "replayed": True,
                "stream_sequence": sequence,
            }
        )

    payload = {
        "schema_version": "uavids-recorded-inference-stream-v1",
        "generated_utc": generated_utc,
        "generator": "generate_recorded_stream.py",
        "model_id": ids.model_id,
        "decision_threshold": ids.threshold,
        "event_count": len(predictions),
        "distinct_record_count": len({item["record_id"] for item in predictions}),
        "normal_count": sum(item["label"] == "Normal" for item in predictions),
        "attack_count": sum(item["label"] == "Attack" for item in predictions),
        "source_note": (
            "Twenty-four distinct finite-feature rows selected deterministically across the frozen model's "
            "Normal and Attack probability ranges from the locked Phase 2 validation partition. The stream "
            "is curated for presentation and is not a prevalence estimate. Feature values and labels are not stored."
        ),
        "predictions": predictions,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "written": str(OUTPUT),
                "events": len(predictions),
                "distinct_records": payload["distinct_record_count"],
                "normal": payload["normal_count"],
                "attack": payload["attack_count"],
                "minimum_probability": min(item["attack_probability"] for item in predictions),
                "maximum_probability": max(item["attack_probability"] for item in predictions),
            }
        )
    )


if __name__ == "__main__":
    main()
