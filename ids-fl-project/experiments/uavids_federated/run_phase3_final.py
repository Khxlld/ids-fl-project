"""Evaluate locked Phase 3 checkpoints on the final test exactly after policy lock."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from uavids_fl import BinaryMLP, metric_bundle, predict_proba, set_deterministic  # noqa: E402


LOCK_PATH = ROOT / "config" / "phase3_locked_config.json"
PHASE2_MANIFEST_PATH = ROOT / "results_phase2" / "partition_manifest.json"
RESULTS = ROOT / "results_phase3"
PLOTS = RESULTS / "plots"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def load_model(record: dict, expected_sha256: str) -> tuple[BinaryMLP, dict]:
    path = ROOT / record["path"]
    assert digest(path) == expected_sha256
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = BinaryMLP(
        int(checkpoint["input_dim"]),
        checkpoint["hidden_layers"],
        checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model, checkpoint


def plot_confusion(metrics: dict, title: str, path: Path) -> None:
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, axis = plt.subplots(figsize=(4.8, 4.2))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], labels=["Pred Normal", "Pred Attack"])
    axis.set_yticks([0, 1], labels=["True Normal", "True Attack"])
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    axis.set_title(title)
    fig.colorbar(image, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    assert LOCK_PATH.is_file(), "Run validation-only development before final evaluation"
    lock_hash_before = digest(LOCK_PATH)
    locked = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    phase2 = json.loads(PHASE2_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert locked["locked_after_validation"] is True
    assert locked["test_partition_loaded_during_development"] is False
    assert locked["final_test"]["status"] == "locked_not_evaluated"
    assert digest(PHASE2_MANIFEST_PATH) == locked["phase2_manifest_sha256"]

    test_path = ROOT / locked["final_test"]["path"]
    assert digest(test_path) == locked["final_test"]["expected_sha256"]
    test_frame = pd.read_csv(test_path)
    features = list(locked["features"])
    assert test_frame.columns.tolist() == features + ["original_label", "binary_label"]
    y_test = test_frame["binary_label"].to_numpy(dtype=np.int64)
    assert set(y_test) == {0, 1} and len(y_test) == 53949

    preprocessor_path = ROOT / locked["preprocessor"]["path"]
    assert digest(preprocessor_path) == locked["preprocessor"]["sha256"]
    preprocessor = joblib.load(preprocessor_path)
    X_test = np.ascontiguousarray(preprocessor.transform(test_frame[features]), dtype=np.float32)
    assert X_test.shape == (53949, 15) and np.isfinite(X_test).all()

    set_deterministic(int(locked["seed"]), 1)
    device = torch.device(locked["reference_device"])
    model_names = ["centralized", "federated_fedavg"] + [
        f"local/uav-client-{index}" for index in range(1, 6)
    ]
    metrics_rows: list[dict] = []
    probabilities: dict[str, np.ndarray] = {}
    predictions = pd.DataFrame({
        "partition_row_index": np.arange(len(test_frame), dtype=np.int64),
        "original_label": test_frame["original_label"],
        "binary_label": y_test,
    })

    for model_name in model_names:
        record = locked["model_artifacts"][model_name]
        model, checkpoint = load_model(record, record["sha256"])
        threshold = float(locked["model_thresholds"][model_name])
        assert abs(float(checkpoint["threshold"]) - threshold) < 1e-12
        probability = predict_proba(model, X_test, device)
        metrics = metric_bundle(y_test, probability, threshold)
        metrics_rows.append({"model": model_name, **metrics})
        probabilities[model_name] = probability
        column_prefix = safe_name(model_name)
        predictions[f"{column_prefix}_probability"] = probability
        predictions[f"{column_prefix}_prediction"] = (probability >= threshold).astype("int8")

    test_metrics = pd.DataFrame(metrics_rows)
    test_metrics_path = RESULTS / "locked_test_model_metrics.csv"
    test_metrics.to_csv(test_metrics_path, index=False)
    predictions_path = RESULTS / "locked_test_predictions.csv"
    predictions.to_csv(predictions_path, index=False, float_format="%.9g", lineterminator="\n")

    subclass_rows = []
    for model_name in model_names:
        threshold = float(locked["model_thresholds"][model_name])
        predicted = probabilities[model_name] >= threshold
        for original_label, indices in test_frame.groupby("original_label", sort=False).groups.items():
            index_array = np.asarray(list(indices), dtype=int)
            predicted_attack_rate = float(predicted[index_array].mean())
            subclass_rows.append({
                "model": model_name,
                "original_label": original_label,
                "rows": int(len(index_array)),
                "predicted_attack_rate": predicted_attack_rate,
                "interpretation": (
                    "false_positive_rate_for_normal"
                    if original_label == "Normal Traffic"
                    else "attack_detection_recall"
                ),
            })
    subclass_detection = pd.DataFrame(subclass_rows)
    subclass_detection.to_csv(RESULTS / "locked_test_original_class_detection.csv", index=False)

    centralized = test_metrics.loc[test_metrics["model"].eq("centralized")].iloc[0]
    federated = test_metrics.loc[test_metrics["model"].eq("federated_fedavg")].iloc[0]
    local_metrics = test_metrics[test_metrics["model"].str.startswith("local/")].copy()
    local_mean_macro = float(local_metrics["macro_f1"].mean())
    local_worst = local_metrics.loc[local_metrics["macro_f1"].idxmin()]
    local_best = local_metrics.loc[local_metrics["macro_f1"].idxmax()]

    if federated["macro_f1"] > local_mean_macro and federated["macro_f1"] < centralized["macro_f1"]:
        federation_assessment = "mixed: FedAvg improves on the mean local-only model but trails centralized pooling"
    elif federated["macro_f1"] >= max(local_mean_macro, centralized["macro_f1"]):
        federation_assessment = "helped: FedAvg matches or exceeds both centralized and mean local-only performance"
    elif federated["macro_f1"] <= min(local_mean_macro, centralized["macro_f1"]):
        federation_assessment = "hurt: FedAvg trails both centralized and mean local-only performance"
    else:
        federation_assessment = "mixed: comparison depends on the baseline and metric"

    plot_confusion(
        centralized.to_dict(),
        "Centralized locked-test confusion matrix",
        PLOTS / "locked_test_confusion_centralized.png",
    )
    plot_confusion(
        federated.to_dict(),
        "FedAvg locked-test confusion matrix",
        PLOTS / "locked_test_confusion_federated.png",
    )
    fig, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(test_metrics["model"], test_metrics["macro_f1"])
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("macro_f1")
    axis.set_title("Locked-test macro-F1 comparison")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS / "locked_test_macro_f1_comparison.png", dpi=170)
    plt.close(fig)

    record = {
        "phase": 3,
        "evaluation_stage": "final_locked_test",
        "lock_config_path": str(LOCK_PATH.relative_to(ROOT)),
        "lock_config_sha256": lock_hash_before,
        "lock_unchanged_after_evaluation": digest(LOCK_PATH) == lock_hash_before,
        "test_partition": {
            "path": str(test_path.relative_to(ROOT)),
            "sha256": digest(test_path),
            "rows": int(len(test_frame)),
        },
        "preprocessor_refit": False,
        "model_or_threshold_revised_after_test": False,
        "metrics": {row["model"]: {key: value for key, value in row.items() if key != "model"} for row in metrics_rows},
        "local_only_summary": {
            "mean_macro_f1": local_mean_macro,
            "best_model": local_best["model"],
            "best_macro_f1": float(local_best["macro_f1"]),
            "worst_model": local_worst["model"],
            "worst_macro_f1": float(local_worst["macro_f1"]),
        },
        "federation_assessment": federation_assessment,
        "result_artifacts": {
            "model_metrics": {"path": str(test_metrics_path.relative_to(ROOT)), "sha256": digest(test_metrics_path)},
            "predictions": {"path": str(predictions_path.relative_to(ROOT)), "sha256": digest(predictions_path)},
            "original_class_detection": {
                "path": "results_phase3/locked_test_original_class_detection.csv",
                "sha256": digest(RESULTS / "locked_test_original_class_detection.csv"),
            },
        },
    }
    record_path = RESULTS / "locked_test_evaluation_record.json"
    write_json(record_path, record)

    def metric_row(row) -> str:
        return (
            f"| {row['model']} | {row['threshold']:.2f} | {row['accuracy']:.4f} | "
            f"{row['macro_f1']:.4f} | {row['attack_precision']:.4f} | {row['attack_recall']:.4f} | "
            f"{row['fpr']:.4f} | {int(row['tn'])} | {int(row['fp'])} | {int(row['fn'])} | {int(row['tp'])} |"
        )

    table_rows = "\n".join(metric_row(row) for _, row in test_metrics.iterrows())
    selected = locked["selected_candidate"]
    architecture = " → ".join(["15", *[str(width) for width in selected["hidden_layers"]], "1"])
    final_summary = f"""# Phase 3 — binary model comparison and locked-test results

The policy was locked from validation before `test.csv` was opened. The locked configuration hash remained unchanged during final evaluation; preprocessing was not refitted and no model, threshold, or training setting was revised from test performance.

## Selected preprocessing and MLP

- Training-only pooled median imputation and `StandardScaler` over the 15 approved Phase 2 features.
- MLP: **{architecture}**, ReLU activations and dropout `{selected['dropout']}`.
- Loss: `{selected['loss_policy']}`; optimizer: AdamW at learning rate {locked['optimizer']['learning_rate']} and weight decay {locked['optimizer']['weight_decay']}.
- FedAvg: all five logical clients, {locked['federated_training']['local_epochs']} local epochs per round, {locked['federated_training']['rounds']} rounds, sample-count weighting. The validation-selected global checkpoint is recorded in the federated checkpoint.

## Final locked-test comparison

| Model | Threshold | Accuracy | Macro-F1 | Attack precision | Attack recall | FPR | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table_rows}

Local-only mean macro-F1 is **{local_mean_macro:.4f}**. Best local model: **{local_best['model']} ({local_best['macro_f1']:.4f})**. Worst local model: **{local_worst['model']} ({local_worst['macro_f1']:.4f})**.

Federation assessment: **{federation_assessment}**.

## Interpretation and limitations

- Centralized training measures the benefit of pooled training rows under the shared train-only preprocessor.
- Local-only results show how strongly generalization depends on one logical source's data distribution.
- FedAvg measures distributed optimization over the five fixed non-IID logical sources, not privacy protection or a networked deployment.
- Pooled preprocessing centrally accessed all training-client feature values; it is not a private federated preprocessing protocol.
- The held-out test is source-address-disjoint and signature-disjoint, but not scenario-, run-, temporal-, independent-network-, or verified-physical-UAV-disjoint.
- Original-class detection rates are saved separately to expose attack-family weaknesses hidden by the binary aggregate.

## Docker handoff

Carry forward the locked feature order, fitted preprocessor, MLP architecture, global checkpoint, decision threshold, binary label mapping, and exact client partition/checksum manifest. Docker clients must load only their own Phase 2 CSV, while the control center owns global validation/evaluation and weighted FedAvg. The deployment must continue to describe clients as logical sources and must not claim that pooled preprocessing is privacy-preserving.
"""
    (RESULTS / "PHASE3_FINAL_SUMMARY.md").write_text(final_summary, encoding="utf-8")

    assert digest(LOCK_PATH) == lock_hash_before
    print(test_metrics[["model", "threshold", "accuracy", "macro_f1", "attack_precision", "attack_recall", "fpr", "tn", "fp", "fn", "tp"]].to_string(index=False))
    print(f"\nFederation assessment: {federation_assessment}")
    print("Confirmed: locked config unchanged; no preprocessing refit or post-test revision.")


if __name__ == "__main__":
    main()
