"""Verify frozen Phase 3 artifacts without reopening the final-test partition."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from uavids_fl import metric_bundle  # noqa: E402


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def safe_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def main() -> None:
    lock_path = ROOT / "config" / "phase3_locked_config.json"
    record_path = ROOT / "results_phase3" / "locked_test_evaluation_record.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    record = json.loads(record_path.read_text(encoding="utf-8"))

    assert digest(lock_path) == record["lock_config_sha256"]
    assert record["lock_unchanged_after_evaluation"] is True
    assert record["model_or_threshold_revised_after_test"] is False
    assert record["preprocessor_refit"] is False

    for artifact in lock["model_artifacts"].values():
        assert digest(ROOT / artifact["path"]) == artifact["sha256"]
    preprocessor = lock["preprocessor"]
    assert digest(ROOT / preprocessor["path"]) == preprocessor["sha256"]
    for artifact in record["result_artifacts"].values():
        assert digest(ROOT / artifact["path"]) == artifact["sha256"]

    metrics_path = ROOT / record["result_artifacts"]["model_metrics"]["path"]
    predictions_path = ROOT / record["result_artifacts"]["predictions"]["path"]
    saved_metrics = pd.read_csv(metrics_path).set_index("model")
    predictions = pd.read_csv(predictions_path)
    y_true = predictions["binary_label"].to_numpy(dtype=int)
    assert len(predictions) == record["test_partition"]["rows"]
    assert predictions["partition_row_index"].tolist() == list(range(len(predictions)))

    checked_metrics = [
        "accuracy", "balanced_accuracy", "macro_f1", "attack_precision",
        "attack_recall", "attack_f1", "normal_precision", "normal_recall",
        "normal_f1", "roc_auc", "pr_auc", "log_loss", "fpr", "fnr",
    ]
    checked_counts = ["rows", "tn", "fp", "fn", "tp"]
    for model_name, threshold in lock["model_thresholds"].items():
        prefix = safe_name(model_name)
        probability = predictions[f"{prefix}_probability"].to_numpy(dtype=float)
        expected_prediction = (probability >= float(threshold)).astype(int)
        np.testing.assert_array_equal(
            expected_prediction,
            predictions[f"{prefix}_prediction"].to_numpy(dtype=int),
        )
        recomputed = metric_bundle(y_true, probability, float(threshold))
        saved = saved_metrics.loc[model_name]
        for metric in checked_metrics:
            # Saved probabilities use nine significant digits; tolerate only that
            # serialization loss while requiring predictions and counts exactly.
            np.testing.assert_allclose(recomputed[metric], saved[metric], rtol=0, atol=1e-7)
        for metric in checked_counts:
            assert int(recomputed[metric]) == int(saved[metric])
        np.testing.assert_allclose(float(saved["threshold"]), float(threshold), rtol=0, atol=1e-12)

    required_plots = [
        "candidate_validation_macro_f1.png",
        "centralized_validation_history.png",
        "federated_validation_history.png",
        "local_only_validation_macro_f1.png",
        "validation_model_comparison.png",
        "locked_test_confusion_centralized.png",
        "locked_test_confusion_federated.png",
        "locked_test_macro_f1_comparison.png",
    ]
    for name in required_plots:
        path = ROOT / "results_phase3" / "plots" / name
        assert path.is_file() and path.stat().st_size > 0

    print(
        "Phase 3 verification passed: lock, 8 checkpoints/preprocessor artifacts, "
        "3 result hashes, 7 saved prediction/metric pairs, and 8 plots."
    )


if __name__ == "__main__":
    main()
