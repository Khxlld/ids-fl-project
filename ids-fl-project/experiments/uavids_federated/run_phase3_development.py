"""Validation-only Phase 3 development.

This program intentionally never opens the Phase 2 final-test CSV. It fits the
pooled-training preprocessor, compares MLP candidates on validation, trains the
local/centralized/FedAvg models, and writes a locked configuration for the
separate final-test program.
"""

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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from uavids_fl import (  # noqa: E402
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


CONFIG_PATH = ROOT / "config" / "phase3_development_config.json"
PHASE2_MANIFEST_PATH = ROOT / "results_phase2" / "partition_manifest.json"
ARTIFACTS = ROOT / "artifacts_phase3"
RESULTS = ROOT / "results_phase3"
PLOTS = RESULTS / "plots"
LOCK_PATH = ROOT / "config" / "phase3_locked_config.json"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def score(metrics: dict) -> tuple[float, float, float, float]:
    return (
        float(metrics["macro_f1"]),
        float(metrics["attack_recall"]),
        -float(metrics["fpr"]),
        -float(metrics["log_loss"]),
    )


def pos_weight_for(policy: str, pooled_y: np.ndarray) -> float | None:
    if policy == "unweighted":
        return None
    if policy == "global_training_pos_weight":
        normal = int(np.sum(pooled_y == 0))
        attack = int(np.sum(pooled_y == 1))
        assert normal and attack
        return float(normal / attack)
    raise ValueError(f"Unknown loss policy: {policy}")


def make_model(input_dim: int, candidate: dict, seed: int, threads: int) -> BinaryMLP:
    set_deterministic(seed, threads)
    return BinaryMLP(input_dim, candidate["hidden_layers"], candidate["dropout"])


def checkpoint_payload(model: BinaryMLP, candidate: dict, threshold: float, validation_metrics: dict, extra: dict) -> dict:
    return {
        "state_dict": clone_state_dict(model),
        "input_dim": next(model.parameters()).shape[1],
        "hidden_layers": list(candidate["hidden_layers"]),
        "dropout": list(candidate["dropout"]),
        "loss_policy": candidate["loss_policy"],
        "threshold": float(threshold),
        "validation_metrics": validation_metrics,
        **extra,
    }


def plot_line(frame: pd.DataFrame, x: str, ys: list[str], title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, len(ys), figsize=(5.2 * len(ys), 4))
    if len(ys) == 1:
        axes = [axes]
    for axis, column in zip(axes, ys):
        axis.plot(frame[x], frame[column], marker="o", markersize=3)
        axis.set_xlabel(x)
        axis.set_ylabel(column)
        axis.set_title(column.replace("_", " ").title())
        axis.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_bars(frame: pd.DataFrame, label_column: str, metric: str, title: str, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(frame[label_column], frame[metric])
    axis.set_ylim(0, 1.02)
    axis.set_ylabel(metric)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    phase2 = json.loads(PHASE2_MANIFEST_PATH.read_text(encoding="utf-8"))
    features = list(phase2["approved_model_features"])
    seed = int(config["seed"])
    threads = int(config["torch_num_threads"])
    device = torch.device(config["reference_device"])
    assert device.type == "cpu", "The reference development run is deliberately deterministic and CPU-compatible"

    train_items = phase2["training_clients"]
    partition_records = phase2["artifact_checksums"]
    loaded_paths: list[str] = []
    client_frames: dict[str, pd.DataFrame] = {}
    for item in train_items:
        client_id = item["client_id"]
        record = partition_records[f"partition::train/{client_id}"]
        path = ROOT / record["path"]
        assert digest(path) == record["sha256"]
        frame = pd.read_csv(path)
        assert frame.columns.tolist() == features + ["original_label", "binary_label"]
        client_frames[client_id] = frame
        loaded_paths.append(str(path.relative_to(ROOT)))

    validation_record = partition_records["partition::validation"]
    validation_path = ROOT / validation_record["path"]
    assert digest(validation_path) == validation_record["sha256"]
    validation_frame = pd.read_csv(validation_path)
    assert validation_frame.columns.tolist() == features + ["original_label", "binary_label"]
    loaded_paths.append(str(validation_path.relative_to(ROOT)))
    assert not any("test.csv" in path.lower() for path in loaded_paths)

    pooled_training = pd.concat(client_frames.values(), ignore_index=True)
    X_train_frame = pooled_training[features]
    y_train = pooled_training["binary_label"].to_numpy(dtype=np.int64)
    X_validation_frame = validation_frame[features]
    y_validation = validation_frame["binary_label"].to_numpy(dtype=np.int64)
    assert len(pooled_training) == 6148
    assert len(validation_frame) == 53786
    assert set(y_train) == {0, 1} and set(y_validation) == {0, 1}

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                features,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    X_train = np.ascontiguousarray(preprocessor.fit_transform(X_train_frame), dtype=np.float32)
    X_validation = np.ascontiguousarray(preprocessor.transform(X_validation_frame), dtype=np.float32)
    assert X_train.shape == (6148, 15) and X_validation.shape == (53786, 15)
    assert np.isfinite(X_train).all() and np.isfinite(X_validation).all()
    client_arrays = {
        client_id: (
            np.ascontiguousarray(preprocessor.transform(frame[features]), dtype=np.float32),
            frame["binary_label"].to_numpy(dtype=np.int64),
        )
        for client_id, frame in client_frames.items()
    }

    preprocessor_path = ARTIFACTS / "training_only_preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    write_json(ARTIFACTS / "feature_list.json", features)
    fitted_numeric = preprocessor.named_transformers_["numeric"]
    preprocessing_metadata = {
        "fit_scope": "pooled five Phase 2 training clients only",
        "fit_rows": int(len(X_train)),
        "validation_rows_transformed_only": int(len(X_validation)),
        "test_partition_loaded": False,
        "features": features,
        "imputer": "median",
        "imputer_statistics": fitted_numeric.named_steps["imputer"].statistics_.tolist(),
        "scaler": "StandardScaler",
        "scaler_mean": fitted_numeric.named_steps["scaler"].mean_.tolist(),
        "scaler_scale": fitted_numeric.named_steps["scaler"].scale_.tolist(),
        "output_dtype": "float32",
        "privacy_limitation": config["preprocessing"]["privacy_limitation"],
    }
    write_json(ARTIFACTS / "preprocessing_metadata.json", preprocessing_metadata)

    optimizer = config["optimizer"]
    central_cfg = config["centralized_training"]
    threshold_cfg = config["threshold_selection"]
    candidate_records: list[dict] = []
    candidate_histories: list[pd.DataFrame] = []

    for candidate in config["candidate_models"]:
        model = make_model(X_train.shape[1], candidate, seed, threads)
        positive_weight = pos_weight_for(candidate["loss_policy"], y_train)
        model, history, training_metadata = train_with_early_stopping(
            model,
            X_train,
            y_train,
            X_validation,
            y_validation,
            device=device,
            batch_size=config["batch_size"],
            maximum_epochs=central_cfg["maximum_epochs"],
            patience=central_cfg["early_stopping_patience"],
            learning_rate=optimizer["learning_rate"],
            weight_decay=optimizer["weight_decay"],
            pos_weight=positive_weight,
            seed=seed,
            checkpoint_threshold=config["checkpoint_selection"]["validation_threshold"],
        )
        probabilities = predict_proba(model, X_validation, device)
        threshold, validation_metrics = select_threshold(
            y_validation,
            probabilities,
            threshold_cfg["minimum"],
            threshold_cfg["maximum"],
            threshold_cfg["step"],
        )
        candidate_records.append({
            "candidate_id": candidate["candidate_id"],
            "hidden_layers": json.dumps(candidate["hidden_layers"]),
            "dropout": json.dumps(candidate["dropout"]),
            "loss_policy": candidate["loss_policy"],
            "positive_class_weight": positive_weight if positive_weight is not None else 1.0,
            "selected_threshold": threshold,
            "best_epoch": training_metadata["best_epoch"],
            "epochs_executed": training_metadata["epochs_executed"],
            **validation_metrics,
        })
        history_frame = pd.DataFrame(history)
        history_frame.insert(0, "candidate_id", candidate["candidate_id"])
        candidate_histories.append(history_frame)

    candidate_results = pd.DataFrame(candidate_records)
    selected_index = max(candidate_results.index, key=lambda index: score(candidate_results.loc[index].to_dict()))
    selected_record = candidate_results.loc[selected_index].to_dict()
    selected_candidate = next(
        candidate for candidate in config["candidate_models"]
        if candidate["candidate_id"] == selected_record["candidate_id"]
    )
    selected_pos_weight = pos_weight_for(selected_candidate["loss_policy"], y_train)
    candidate_results["selected"] = candidate_results.index == selected_index
    candidate_results.to_csv(RESULTS / "candidate_validation_comparison.csv", index=False)
    pd.concat(candidate_histories, ignore_index=True).to_csv(RESULTS / "candidate_training_histories.csv", index=False)
    plot_bars(
        candidate_results,
        "candidate_id",
        "macro_f1",
        "Validation macro-F1 for controlled candidates",
        PLOTS / "candidate_validation_macro_f1.png",
    )

    # One shared initialization makes the selected centralized/local/FedAvg runs comparable.
    initial_model = make_model(X_train.shape[1], selected_candidate, seed, threads)
    initial_state = clone_state_dict(initial_model)

    centralized = make_model(X_train.shape[1], selected_candidate, seed, threads)
    centralized.load_state_dict(initial_state)
    centralized, central_history, central_metadata = train_with_early_stopping(
        centralized,
        X_train,
        y_train,
        X_validation,
        y_validation,
        device=device,
        batch_size=config["batch_size"],
        maximum_epochs=central_cfg["maximum_epochs"],
        patience=central_cfg["early_stopping_patience"],
        learning_rate=optimizer["learning_rate"],
        weight_decay=optimizer["weight_decay"],
        pos_weight=selected_pos_weight,
        seed=seed,
        checkpoint_threshold=config["checkpoint_selection"]["validation_threshold"],
    )
    central_prob = predict_proba(centralized, X_validation, device)
    central_threshold, central_validation = select_threshold(
        y_validation, central_prob, threshold_cfg["minimum"], threshold_cfg["maximum"], threshold_cfg["step"]
    )
    assert abs(central_validation["macro_f1"] - float(selected_record["macro_f1"])) < 1e-12
    central_history_frame = pd.DataFrame(central_history)
    central_history_frame.to_csv(RESULTS / "centralized_training_history.csv", index=False)
    plot_line(
        central_history_frame,
        "epoch",
        ["validation_macro_f1", "validation_log_loss"],
        "Centralized validation history",
        PLOTS / "centralized_validation_history.png",
    )

    central_path = ARTIFACTS / "centralized_model.pt"
    deterministic_central_metadata = {
        key: value for key, value in central_metadata.items() if key != "training_seconds"
    }
    torch.save(
        checkpoint_payload(
            centralized,
            selected_candidate,
            central_threshold,
            central_validation,
            {"training_mode": "centralized", **deterministic_central_metadata},
        ),
        central_path,
    )

    local_models: dict[str, BinaryMLP] = {}
    local_thresholds: dict[str, float] = {}
    local_validation_rows: list[dict] = []
    local_history_rows: list[dict] = []
    local_paths: dict[str, Path] = {}
    local_cfg = config["local_only_training"]
    for client_index, item in enumerate(train_items):
        client_id = item["client_id"]
        X_client, y_client = client_arrays[client_id]
        model = make_model(X_train.shape[1], selected_candidate, seed, threads)
        model.load_state_dict(initial_state)
        model, history, metadata = train_with_early_stopping(
            model,
            X_client,
            y_client,
            X_validation,
            y_validation,
            device=device,
            batch_size=config["batch_size"],
            maximum_epochs=local_cfg["maximum_epochs"],
            patience=local_cfg["early_stopping_patience"],
            learning_rate=optimizer["learning_rate"],
            weight_decay=optimizer["weight_decay"],
            pos_weight=selected_pos_weight,
            seed=seed + client_index + 1,
            checkpoint_threshold=config["checkpoint_selection"]["validation_threshold"],
        )
        probabilities = predict_proba(model, X_validation, device)
        threshold, validation_metrics = select_threshold(
            y_validation, probabilities, threshold_cfg["minimum"], threshold_cfg["maximum"], threshold_cfg["step"]
        )
        local_models[client_id] = model
        local_thresholds[client_id] = threshold
        local_validation_rows.append({
            "model": f"local/{client_id}",
            "client_id": client_id,
            "training_rows": len(y_client),
            "best_epoch": metadata["best_epoch"],
            "epochs_executed": metadata["epochs_executed"],
            **validation_metrics,
        })
        for row in history:
            local_history_rows.append({"client_id": client_id, **row})
        path = ARTIFACTS / f"local_{client_id.replace('-', '_')}_model.pt"
        deterministic_local_metadata = {
            key: value for key, value in metadata.items() if key != "training_seconds"
        }
        torch.save(
            checkpoint_payload(
                model,
                selected_candidate,
                threshold,
                validation_metrics,
                {"training_mode": "local_only", "client_id": client_id, **deterministic_local_metadata},
            ),
            path,
        )
        local_paths[client_id] = path

    local_validation = pd.DataFrame(local_validation_rows)
    local_validation.to_csv(RESULTS / "local_only_validation_metrics.csv", index=False)
    pd.DataFrame(local_history_rows).to_csv(RESULTS / "local_only_training_history.csv", index=False)
    plot_bars(
        local_validation,
        "client_id",
        "macro_f1",
        "Local-only models on common validation data",
        PLOTS / "local_only_validation_macro_f1.png",
    )

    # Explicit sequential FedAvg. All five clients participate and sample counts weight aggregation.
    federated_cfg = config["federated_training"]
    global_state = clone_state_dict(initial_state)
    federated_history_rows: list[dict] = []
    federated_client_rows: list[dict] = []
    best_federated_state = clone_state_dict(global_state)
    best_federated_metrics: dict | None = None
    best_round = 0
    for server_round in range(0, federated_cfg["rounds"] + 1):
        global_model = make_model(X_train.shape[1], selected_candidate, seed, threads)
        global_model.load_state_dict(global_state)
        round_probabilities = predict_proba(global_model, X_validation, device)
        round_metrics = metric_bundle(
            y_validation, round_probabilities, config["checkpoint_selection"]["validation_threshold"]
        )
        federated_history_rows.append({"round": server_round, **round_metrics})
        if server_round > 0 and (best_federated_metrics is None or score(round_metrics) > score(best_federated_metrics)):
            best_federated_metrics = round_metrics
            best_federated_state = clone_state_dict(global_state)
            best_round = server_round
        if server_round == federated_cfg["rounds"]:
            break

        local_states = []
        sample_counts = []
        for client_index, item in enumerate(train_items):
            client_id = item["client_id"]
            X_client, y_client = client_arrays[client_id]
            local_model = make_model(X_train.shape[1], selected_candidate, seed, threads)
            local_model.load_state_dict(global_state)
            local_loss = train_local_epochs(
                local_model,
                X_client,
                y_client,
                device=device,
                batch_size=config["batch_size"],
                epochs=federated_cfg["local_epochs"],
                learning_rate=optimizer["learning_rate"],
                weight_decay=optimizer["weight_decay"],
                pos_weight=selected_pos_weight,
                seed=seed + (server_round + 1) * 100 + client_index,
            )
            local_states.append(clone_state_dict(local_model))
            sample_counts.append(len(y_client))
            federated_client_rows.append({
                "round": server_round + 1,
                "client_id": client_id,
                "samples": len(y_client),
                "local_epochs": federated_cfg["local_epochs"],
                "final_local_weighted_loss": local_loss,
            })
        global_state = fedavg(local_states, sample_counts)

    assert best_federated_metrics is not None and best_round > 0
    federated_model = make_model(X_train.shape[1], selected_candidate, seed, threads)
    federated_model.load_state_dict(best_federated_state)
    federated_prob = predict_proba(federated_model, X_validation, device)
    federated_threshold, federated_validation = select_threshold(
        y_validation, federated_prob, threshold_cfg["minimum"], threshold_cfg["maximum"], threshold_cfg["step"]
    )
    federated_history = pd.DataFrame(federated_history_rows)
    federated_history.to_csv(RESULTS / "federated_round_history.csv", index=False)
    pd.DataFrame(federated_client_rows).to_csv(RESULTS / "federated_client_history.csv", index=False)
    plot_line(
        federated_history,
        "round",
        ["macro_f1", "log_loss"],
        "FedAvg validation history",
        PLOTS / "federated_validation_history.png",
    )

    federated_path = ARTIFACTS / "federated_global_model.pt"
    torch.save(
        checkpoint_payload(
            federated_model,
            selected_candidate,
            federated_threshold,
            federated_validation,
            {
                "training_mode": "federated_fedavg",
                "best_round": best_round,
                "rounds_executed": federated_cfg["rounds"],
                "local_epochs": federated_cfg["local_epochs"],
                "aggregation_weight": federated_cfg["aggregation_weight"],
            },
        ),
        federated_path,
    )

    validation_rows = [
        {"model": "centralized", **central_validation},
        {"model": "federated_fedavg", **federated_validation},
        *[{key: value for key, value in row.items() if key != "client_id"} for row in local_validation_rows],
    ]
    validation_comparison = pd.DataFrame(validation_rows)
    validation_comparison.to_csv(RESULTS / "validation_model_comparison.csv", index=False)
    plot_bars(
        validation_comparison,
        "model",
        "macro_f1",
        "Validation comparison after model-specific threshold locking",
        PLOTS / "validation_model_comparison.png",
    )

    diagnostic_rows: list[dict] = []
    for item in train_items:
        client_id = item["client_id"]
        X_client, y_client = client_arrays[client_id]
        for model_name, model, threshold in [
            ("centralized", centralized, central_threshold),
            ("federated_fedavg", federated_model, federated_threshold),
            (f"local/{client_id}", local_models[client_id], local_thresholds[client_id]),
        ]:
            metrics, _ = evaluate_model(model, X_client, y_client, device, threshold)
            diagnostic_rows.append({
                "client_id": client_id,
                "model": model_name,
                "scope": "training_client_in_sample_diagnostic",
                **metrics,
            })
    client_diagnostics = pd.DataFrame(diagnostic_rows)
    client_diagnostics.to_csv(RESULTS / "training_client_diagnostics.csv", index=False)

    model_artifacts = {
        "centralized": central_path,
        "federated_fedavg": federated_path,
        **{f"local/{client_id}": path for client_id, path in local_paths.items()},
    }
    locked_config = {
        "phase": 3,
        "design_version": config["design_version"],
        "locked_after_validation": True,
        "test_partition_loaded_during_development": False,
        "development_loaded_paths": loaded_paths,
        "phase2_manifest_sha256": digest(PHASE2_MANIFEST_PATH),
        "development_config_sha256": digest(CONFIG_PATH),
        "preprocessor": {
            "path": str(preprocessor_path.relative_to(ROOT)),
            "sha256": digest(preprocessor_path),
            "fit_scope": preprocessing_metadata["fit_scope"],
            "fit_rows": preprocessing_metadata["fit_rows"],
        },
        "features": features,
        "selected_candidate": selected_candidate,
        "selected_positive_class_weight": selected_pos_weight if selected_pos_weight is not None else 1.0,
        "optimizer": optimizer,
        "batch_size": config["batch_size"],
        "seed": seed,
        "reference_device": str(device),
        "checkpoint_selection": config["checkpoint_selection"],
        "threshold_selection": config["threshold_selection"],
        "centralized_training": central_cfg,
        "local_only_training": local_cfg,
        "federated_training": federated_cfg,
        "model_thresholds": {
            "centralized": central_threshold,
            "federated_fedavg": federated_threshold,
            **{f"local/{key}": value for key, value in local_thresholds.items()},
        },
        "validation_metrics": {
            row["model"]: {key: value for key, value in row.items() if key != "model"}
            for row in validation_rows
        },
        "model_artifacts": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
            for name, path in model_artifacts.items()
        },
        "final_test": {
            "path": partition_records["partition::test"]["path"],
            "expected_sha256": partition_records["partition::test"]["sha256"],
            "status": "locked_not_evaluated",
        },
        "privacy_limitation": config["preprocessing"]["privacy_limitation"],
    }
    write_json(LOCK_PATH, locked_config)

    local_mean_macro = float(local_validation["macro_f1"].mean())
    local_worst_row = local_validation.loc[local_validation["macro_f1"].idxmin()]
    development_manifest = {
        "test_partition_loaded": False,
        "selected_candidate_id": selected_candidate["candidate_id"],
        "selected_loss_policy": selected_candidate["loss_policy"],
        "positive_class_weight": selected_pos_weight if selected_pos_weight is not None else 1.0,
        "centralized_best_epoch": central_metadata["best_epoch"],
        "federated_best_round": best_round,
        "local_validation_mean_macro_f1": local_mean_macro,
        "local_validation_worst_model": local_worst_row["model"],
        "local_validation_worst_macro_f1": float(local_worst_row["macro_f1"]),
        "locked_config_path": str(LOCK_PATH.relative_to(ROOT)),
        "locked_config_sha256": digest(LOCK_PATH),
        "artifacts": {
            "preprocessor": digest(preprocessor_path),
            **{name: digest(path) for name, path in model_artifacts.items()},
        },
    }
    write_json(RESULTS / "development_manifest.json", development_manifest)

    summary = f"""# Phase 3 development and validation summary

The final-test CSV was not opened during this run. All decisions below were made from the five Phase 2 training clients and the validation partition.

## Locked policy

- Preprocessing: median imputation plus `StandardScaler`, fitted once on all **{len(X_train):,} pooled training-client rows** and only transformed on validation.
- Selected candidate: **{selected_candidate['candidate_id']}**, hidden layers `{selected_candidate['hidden_layers']}`, dropout `{selected_candidate['dropout']}`.
- Loss: **{selected_candidate['loss_policy']}** with attack `pos_weight={selected_pos_weight if selected_pos_weight is not None else 1.0:.6f}`. This weight was calculated from pooled training labels only.
- Optimizer: AdamW, learning rate {optimizer['learning_rate']}, weight decay {optimizer['weight_decay']}, batch size {config['batch_size']}.
- Checkpoints: highest validation macro-F1 at threshold 0.5, then validation attack recall, lower FPR, and lower log loss.
- Final decision thresholds: selected separately on validation macro-F1 and now locked.
- FedAvg: all five clients each round, {federated_cfg['local_epochs']} local epochs, {federated_cfg['rounds']} rounds, aggregation weighted by retained client sample count.

Pooled preprocessing is not privacy-preserving: raw training features were centrally available to fit global medians and scaling statistics. This is acceptable only as a research-prototype baseline and must be replaced or explicitly disclosed in a distributed deployment.

## Validation results

| Model | Threshold | Accuracy | Macro-F1 | Attack precision | Attack recall | FPR | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Centralized | {central_threshold:.2f} | {central_validation['accuracy']:.4f} | {central_validation['macro_f1']:.4f} | {central_validation['attack_precision']:.4f} | {central_validation['attack_recall']:.4f} | {central_validation['fpr']:.4f} | {central_validation['tn']} | {central_validation['fp']} | {central_validation['fn']} | {central_validation['tp']} |
| Federated FedAvg | {federated_threshold:.2f} | {federated_validation['accuracy']:.4f} | {federated_validation['macro_f1']:.4f} | {federated_validation['attack_precision']:.4f} | {federated_validation['attack_recall']:.4f} | {federated_validation['fpr']:.4f} | {federated_validation['tn']} | {federated_validation['fp']} | {federated_validation['fn']} | {federated_validation['tp']} |

The five local-only validation macro-F1 scores have mean **{local_mean_macro:.4f}**; the worst is **{local_worst_row['model']} at {local_worst_row['macro_f1']:.4f}**. Full local and candidate tables are saved as CSV.

The centralized checkpoint was epoch **{central_metadata['best_epoch']}**. The selected federated checkpoint was round **{best_round}**.

## Test lock

`phase3_locked_config.json` records all feature, architecture, loss, optimizer, round, threshold, checkpoint, and artifact hashes. Phase 3 final evaluation must consume that file without revising it.
"""
    (RESULTS / "DEVELOPMENT_VALIDATION_SUMMARY.md").write_text(summary, encoding="utf-8")

    print(candidate_results[["candidate_id", "loss_policy", "selected_threshold", "macro_f1", "attack_recall", "fpr", "selected"]].to_string(index=False))
    print("\nValidation comparison:")
    print(validation_comparison[["model", "threshold", "accuracy", "macro_f1", "attack_precision", "attack_recall", "fpr"]].to_string(index=False))
    print(f"\nLocked selected candidate: {selected_candidate['candidate_id']}")
    print(f"Centralized best epoch: {central_metadata['best_epoch']} | FedAvg best round: {best_round}")
    print("Confirmed: test.csv was not loaded; final policy is locked.")


if __name__ == "__main__":
    main()
