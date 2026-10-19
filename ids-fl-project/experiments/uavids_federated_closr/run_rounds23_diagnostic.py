"""Extend the RNG-neutral known-only FedAvg diagnostic through rounds 2 and 3."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from run_round1_diagnostic import (
    COSINE_NEAR_ZERO_NORM,
    GRADIENT_ZERO_NORM,
    MODES,
    ROOT,
    SEEDS,
    _centers,
    _rng_equal,
    _rng_snapshot,
    _state_comparison,
    _state_sha256,
    _update_diagnostics,
    instrumented_train_epoch,
)
from uavids_closr.artifacts import write_json
from uavids_closr.data import ClientDataset, make_iid_control_clients, prepare_data, resolve_config
from uavids_closr.federated import clone_state_dict, sample_weighted_fedavg
from uavids_closr.inference import (
    calibrate_known_threshold,
    known_open_set_diagnostics,
    score_features,
)
from uavids_closr.model import CLOSRModel
from uavids_closr.training import (
    _state_group_deltas,
    make_criterion,
    set_deterministic,
    train_epoch,
)


MAX_ROUND = 3


@dataclass
class RoundSnapshot:
    round_index: int
    round_start_state: OrderedDict[str, torch.Tensor]
    model: CLOSRModel
    centers: torch.Tensor
    similarities: np.ndarray
    scores: np.ndarray
    global_state: OrderedDict[str, torch.Tensor]
    local_states: list[OrderedDict[str, torch.Tensor]]
    client_records: list[dict[str, Any]]
    gradient_rows: list[dict[str, Any]]


@dataclass
class Trajectory:
    round0_model: CLOSRModel
    round0_centers: torch.Tensor
    round0_similarities: np.ndarray
    round0_scores: np.ndarray
    rounds: list[RoundSnapshot]
    rng_state: dict[str, Any]


def _replay_trajectory(
    model_config: dict[str, Any],
    initial_state: OrderedDict[str, torch.Tensor],
    clients: dict[str, ClientDataset],
    prepared,
    config: dict[str, Any],
    instrument: bool,
) -> Trajectory:
    """Mirror the first three rounds of ``fit_federated`` exactly."""
    training = config["training"]
    device = torch.device(training["device"])
    global_state = clone_state_dict(initial_state)
    initial_model = CLOSRModel(**model_config)
    initial_model.load_state_dict(initial_state)
    round0_centers = _centers(initial_model, clients, device)
    round0_similarities, round0_scores = score_features(
        initial_model, round0_centers, prepared.selection_features, device
    )
    _ = f1_score(prepared.selection_labels, np.argmax(round0_similarities, axis=1), average="macro")

    snapshots = []
    ordered_clients = list(clients.values())
    for server_round in range(1, MAX_ROUND + 1):
        round_start_state = clone_state_dict(global_state)
        local_states = []
        sample_counts = []
        client_records = []
        gradient_rows = []
        for client_index, client in enumerate(ordered_clients):
            local_model = CLOSRModel(**model_config)
            local_model.load_state_dict(global_state)
            criterion = make_criterion(config["loss"], local_model.num_classes)
            optimizer = torch.optim.AdamW(
                local_model.parameters(),
                lr=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
            )
            local_history = []
            for local_epoch in range(1, int(training["local_epochs"]) + 1):
                epoch_seed = (
                    int(training["seed"])
                    + server_round * 1000
                    + client_index * 100
                    + local_epoch
                )
                if instrument:
                    epoch_record, observed_gradients = instrumented_train_epoch(
                        local_model,
                        client.features,
                        client.labels,
                        optimizer,
                        criterion,
                        device,
                        int(training["batch_size"]),
                        epoch_seed,
                        bool(training["balanced_sampling"]),
                        list(prepared.class_names),
                    )
                    for row in observed_gradients:
                        gradient_rows.append(
                            {
                                "client_id": client.client_id,
                                "local_epoch": local_epoch,
                                **row,
                            }
                        )
                else:
                    epoch_record = train_epoch(
                        local_model,
                        client.features,
                        client.labels,
                        optimizer,
                        criterion,
                        device,
                        int(training["batch_size"]),
                        epoch_seed,
                        bool(training["balanced_sampling"]),
                    )
                local_history.append(epoch_record)
            local_states.append(clone_state_dict(local_model))
            sample_counts.append(len(client.labels))
            client_records.append(
                {
                    "client_id": client.client_id,
                    "samples": len(client.labels),
                    "local_epochs": local_history,
                }
            )

        global_state = sample_weighted_fedavg(local_states, sample_counts)
        global_model = CLOSRModel(**model_config)
        global_model.load_state_dict(global_state)
        centers = _centers(global_model, clients, device)
        similarities, scores = score_features(
            global_model, centers, prepared.selection_features, device
        )
        _ = f1_score(prepared.selection_labels, np.argmax(similarities, axis=1), average="macro")
        snapshots.append(
            RoundSnapshot(
                round_index=server_round,
                round_start_state=round_start_state,
                model=global_model,
                centers=centers,
                similarities=similarities,
                scores=scores,
                global_state=clone_state_dict(global_state),
                local_states=local_states,
                client_records=client_records,
                gradient_rows=gradient_rows,
            )
        )
    return Trajectory(
        round0_model=initial_model,
        round0_centers=round0_centers,
        round0_similarities=round0_similarities,
        round0_scores=round0_scores,
        rounds=snapshots,
        rng_state=_rng_snapshot(),
    )


def _aggregate_round(snapshot: RoundSnapshot, labels: np.ndarray, num_classes: int) -> dict[str, Any]:
    epochs = [
        epoch
        for client in snapshot.client_records
        for epoch in client["local_epochs"]
    ]
    optimized_rows = sum(int(record["optimized_rows"]) for record in epochs)
    keys = [
        "active_batches",
        "positive_available_batches",
        "negative_available_batches",
        "target_examples_drawn",
        "positive_pairs",
        "negative_pairs",
        "positive_active_pairs",
        "negative_active_pairs",
    ]
    per_head = []
    for class_index in range(num_classes):
        per_head.append(
            {
                "class_index": class_index,
                **{
                    key: int(sum(record["per_head_signal"][class_index][key] for record in epochs))
                    for key in keys
                },
            }
        )
    predictions = np.argmax(snapshot.similarities, axis=1)
    return {
        "aggregation_sample_counts": [client["samples"] for client in snapshot.client_records],
        "examples_drawn": int(sum(record["examples_drawn"] for record in epochs)),
        "optimizer_steps": int(sum(record["optimizer_steps"] for record in epochs)),
        "optimized_rows": optimized_rows,
        "drawn_class_counts": np.sum(
            [record["drawn_class_counts"] for record in epochs], axis=0
        ).astype(np.int64).tolist(),
        "mean_local_loss": float(
            sum(float(record["mean_loss"]) * int(record["optimized_rows"]) for record in epochs)
            / optimized_rows
        ),
        "per_head_signal": per_head,
        **_state_group_deltas(snapshot.round_start_state, snapshot.global_state, num_classes),
        "selection_accuracy": float(accuracy_score(labels, predictions)),
        "selection_macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def _without_runtime(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_runtime(item)
            for key, item in value.items()
            if key != "runtime_seconds"
        }
    if isinstance(value, list):
        return [_without_runtime(item) for item in value]
    return value


def _summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "rows": int(len(values)),
        "mean": float(np.mean(values)),
        "sample_sd": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _analyze_selection(
    seed: int,
    mode: str,
    round_index: int,
    model: CLOSRModel,
    centers: torch.Tensor,
    similarities: np.ndarray,
    scores: np.ndarray,
    prepared,
    config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    class_names = list(prepared.class_names)
    labels = prepared.selection_labels
    device = torch.device(config["training"]["device"])
    calibration_similarities, calibration_scores = score_features(
        model, centers, prepared.calibration_features, device
    )
    threshold, observed_calibration_rejection = calibrate_known_threshold(
        calibration_scores,
        float(config["threshold_calibration"]["known_rejection_rate_target"]),
    )
    closed_predictions = np.argmax(similarities, axis=1).astype(np.int64)
    rejected = scores > threshold
    thresholded_predictions = closed_predictions.copy()
    thresholded_predictions[rejected] = -1
    closed_matrix = confusion_matrix(labels, closed_predictions, labels=range(len(class_names)))
    thresholded = known_open_set_diagnostics(
        similarities,
        scores,
        labels,
        threshold,
        class_names,
        class_names.index(config["benign_class"]),
    )

    round_metrics = [
        {
            "seed": seed,
            "mode": mode,
            "round": round_index,
            "closed_set_accuracy": float(accuracy_score(labels, closed_predictions)),
            "closed_set_macro_f1": float(
                f1_score(
                    labels,
                    closed_predictions,
                    labels=range(len(class_names)),
                    average="macro",
                    zero_division=0,
                )
            ),
            "known_calibration_threshold": threshold,
            "known_calibration_rejection_rate": observed_calibration_rejection,
            "thresholded_selection_macro_f1": float(thresholded["macro_f1_after_rejection"]),
            "thresholded_selection_known_rejection_rate": float(
                thresholded["known_rejection_rate"]
            ),
            "thresholded_selection_benign_rejection_rate": float(
                thresholded["benign_rejection_rate"]
            ),
            "thresholded_selection_benign_total_alert_fpr": float(
                thresholded["benign_total_alert_fpr"]
            ),
            "calibration_use": "known-only threshold procedure; excluded from configuration search",
        }
    ]

    closed_confusion = []
    for true_index, true_name in enumerate(class_names):
        for pred_index, pred_name in enumerate(class_names):
            closed_confusion.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "round": round_index,
                    "true_label": true_name,
                    "predicted_label": pred_name,
                    "count": int(closed_matrix[true_index, pred_index]),
                }
            )
    thresholded_confusion = []
    thresholded_matrix = np.asarray(thresholded["confusion_matrix"], dtype=np.int64)
    threshold_names = class_names + ["UNKNOWN"]
    for true_index, true_name in enumerate(threshold_names):
        for pred_index, pred_name in enumerate(threshold_names):
            thresholded_confusion.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "round": round_index,
                    "true_label": true_name,
                    "predicted_label": pred_name,
                    "count": int(thresholded_matrix[true_index, pred_index]),
                }
            )

    focus_indices = [class_names.index("Normal Traffic"), class_names.index("Sybil Attack")]
    class_metrics = []
    similarity_summaries = []
    margin_summaries = []
    record_rows = []
    for true_index in focus_indices:
        true_name = class_names[true_index]
        true_mask = labels == true_index
        support = int(true_mask.sum())
        for variant, predictions in (
            ("closed_set", closed_predictions),
            ("thresholded", thresholded_predictions),
        ):
            predicted_count = int(np.sum(predictions == true_index))
            true_positive = int(np.sum(true_mask & (predictions == true_index)))
            class_metrics.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "round": round_index,
                    "variant": variant,
                    "class_name": true_name,
                    "support": support,
                    "true_positive": true_positive,
                    "prediction_count": predicted_count,
                    "recall": true_positive / support,
                    "precision": true_positive / predicted_count if predicted_count else np.nan,
                }
            )
        for center_index, center_name in enumerate(class_names):
            similarity_summaries.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "round": round_index,
                    "true_class": true_name,
                    "measure": f"similarity_to::{center_name}",
                    **_summary(similarities[true_mask, center_index]),
                }
            )

        normal_index = class_names.index("Normal Traffic")
        sybil_index = class_names.index("Sybil Attack")
        canonical_margin = similarities[true_mask, normal_index] - similarities[true_mask, sybil_index]
        oriented_ns_margin = canonical_margin if true_index == normal_index else -canonical_margin
        wrong = similarities[true_mask].copy()
        wrong[:, true_index] = -np.inf
        true_vs_largest_wrong = similarities[true_mask, true_index] - np.max(wrong, axis=1)
        for measure, values in (
            ("normal_minus_sybil_similarity", canonical_margin),
            ("true_class_minus_other_normal_sybil", oriented_ns_margin),
            ("true_class_minus_largest_wrong_class", true_vs_largest_wrong),
        ):
            margin_summaries.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "round": round_index,
                    "true_class": true_name,
                    "measure": measure,
                    **_summary(values),
                }
            )

        selected_indices = np.flatnonzero(true_mask)
        for local_position, selection_index in enumerate(selected_indices):
            record = {
                "seed": seed,
                "mode": mode,
                "round": round_index,
                "selection_row_index": int(selection_index),
                "true_class": true_name,
                "closed_prediction": class_names[int(closed_predictions[selection_index])],
                "thresholded_prediction": (
                    "UNKNOWN"
                    if thresholded_predictions[selection_index] == -1
                    else class_names[int(thresholded_predictions[selection_index])]
                ),
                "unknown_score": float(scores[selection_index]),
                "known_threshold": threshold,
                "normal_minus_sybil_similarity": float(canonical_margin[local_position]),
                "true_class_minus_other_normal_sybil": float(oriented_ns_margin[local_position]),
                "true_class_minus_largest_wrong_class": float(true_vs_largest_wrong[local_position]),
            }
            for center_index, center_name in enumerate(class_names):
                record[f"similarity_to::{center_name}"] = float(
                    similarities[selection_index, center_index]
                )
            record_rows.append(record)

    return {
        "round_metrics": round_metrics,
        "closed_confusion": closed_confusion,
        "thresholded_confusion": thresholded_confusion,
        "class_metrics": class_metrics,
        "similarity_summaries": similarity_summaries,
        "margin_summaries": margin_summaries,
        "record_rows": record_rows,
    }


def _changes(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    statistic_columns = ["mean", "sample_sd", "min", "p05", "p25", "p50", "p75", "p95", "max"]
    rows = []
    for _, group in frame.groupby(group_columns, sort=False):
        group = group.sort_values("round")
        baseline = group.iloc[0]
        previous = None
        for _, row in group.iterrows():
            output = {column: row[column] for column in group_columns}
            output["round"] = int(row["round"])
            for column in statistic_columns:
                output[f"{column}_change_from_round0"] = float(row[column] - baseline[column])
                output[f"{column}_change_from_previous_round"] = (
                    np.nan if previous is None else float(row[column] - previous[column])
                )
            rows.append(output)
            previous = row
    return pd.DataFrame(rows)


def _class_metric_changes(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = ["seed", "mode", "variant", "class_name"]
    for _, group in frame.groupby(groups, sort=False):
        group = group.sort_values("round")
        baseline = group.iloc[0]
        previous = None
        for _, row in group.iterrows():
            output = {column: row[column] for column in groups}
            output["round"] = int(row["round"])
            for column in ("recall", "precision", "prediction_count"):
                output[f"{column}_change_from_round0"] = float(row[column] - baseline[column])
                output[f"{column}_change_from_previous_round"] = (
                    np.nan if previous is None else float(row[column] - previous[column])
                )
            rows.append(output)
            previous = row
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "contribution1_baseline.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "contribution1-rounds0-3-exploratory-240520-240523",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output_dir}")
    output_dir.mkdir(parents=True)

    reproduction: dict[str, Any] = {}
    round_metrics: list[dict] = []
    closed_confusion: list[dict] = []
    thresholded_confusion: list[dict] = []
    class_metrics: list[dict] = []
    similarity_summaries: list[dict] = []
    margin_summaries: list[dict] = []
    record_rows: list[dict] = []
    signal_rows: list[dict] = []
    gradient_rows: list[dict] = []
    update_rows: list[dict] = []
    pairwise_rows: list[dict] = []
    cancellation_rows: list[dict] = []
    verification_rows: list[dict] = []
    round0_agreement: list[dict] = []
    delta_artifact: dict[str, Any] = {
        "description": "Actual per-round client deltas, weighted contributions, and aggregate deltas for rounds 1-3",
        "cosine_near_zero_norm": COSINE_NEAR_ZERO_NORM,
        "seeds": {},
    }

    for seed in SEEDS:
        config = resolve_config(args.config.resolve(), smoke=False)
        config["training"]["seed"] = seed
        set_deterministic(seed, int(config["training"]["torch_num_threads"]))
        prepared = prepare_data(config)
        iid_clients, _ = make_iid_control_clients(
            prepared.clients, int(config["iid_control"]["partition_seed"])
        )
        model_config = {
            "input_dim": len(prepared.feature_names),
            "hidden_dims": config["model"]["hidden_dims"],
            "projection_dim": config["model"]["projection_dim"],
            "num_classes": len(prepared.class_names),
            "dropout": config["model"]["dropout"],
        }
        template = CLOSRModel(**model_config)
        initial_state = clone_state_dict(template)
        initial_hash = _state_sha256(initial_state)
        original_run = ROOT / "runs" / f"contribution1-three-way-seed-{seed}"
        saved_rng = json.loads((original_run / "rng_audit.json").read_text())
        if initial_hash != saved_rng["initial_state_sha256"]:
            raise AssertionError(f"initial-state mismatch for seed {seed}")
        original_summary = json.loads((original_run / "run_summary.json").read_text())
        reproduction[str(seed)] = {}
        delta_artifact["seeds"][seed] = {"initial_state_sha256": initial_hash, "modes": {}}
        trajectories = {}

        for mode, clients in (
            ("iid_federated", iid_clients),
            ("natural_non_iid_federated", prepared.clients),
        ):
            set_deterministic(seed, int(config["training"]["torch_num_threads"]))
            ordinary = _replay_trajectory(
                model_config, initial_state, clients, prepared, config, instrument=False
            )
            set_deterministic(seed, int(config["training"]["torch_num_threads"]))
            instrumented = _replay_trajectory(
                model_config, initial_state, clients, prepared, config, instrument=True
            )
            trajectories[mode] = instrumented
            checks = {
                "initial_state_sha256": initial_hash,
                **_rng_equal(ordinary.rng_state, instrumented.rng_state),
                "rounds": {},
            }
            for ordinary_round, observed_round in zip(ordinary.rounds, instrumented.rounds):
                state_check = _state_comparison(
                    ordinary_round.global_state, observed_round.global_state
                )
                local_checks = [
                    _state_comparison(left, right)
                    for left, right in zip(
                        ordinary_round.local_states, observed_round.local_states
                    )
                ]
                observed = _aggregate_round(
                    observed_round, prepared.selection_labels, len(prepared.class_names)
                )
                existing = original_summary[mode]["history"][observed_round.round_index - 1]
                keys = [
                    "aggregation_sample_counts",
                    "examples_drawn",
                    "optimizer_steps",
                    "optimized_rows",
                    "drawn_class_counts",
                    "mean_local_loss",
                    "per_head_signal",
                    "encoder_parameter_delta_l2",
                    "projection_head_parameter_delta_l2",
                    "selection_accuracy",
                    "selection_macro_f1",
                ]
                history_checks = {f"{key}_exact": observed[key] == existing[key] for key in keys}
                client_history_exact = _without_runtime(
                    [
                        {
                            "client_id": item["client_id"],
                            "samples": item["samples"],
                            "local_epochs": item["local_epochs"],
                        }
                        for item in observed_round.client_records
                    ]
                ) == _without_runtime(existing["clients"])
                round_checks = {
                    "global_state_bitwise_equal": state_check["bitwise_equal"],
                    "global_state_max_abs_difference": state_check["max_abs_difference"],
                    "all_client_states_bitwise_equal": all(
                        item["bitwise_equal"] for item in local_checks
                    ),
                    "client_history_exact_excluding_runtime": client_history_exact,
                    **history_checks,
                    "expected_macro_f1": existing["selection_macro_f1"],
                    "replayed_macro_f1": observed["selection_macro_f1"],
                }
                round_checks["all_checks_passed"] = all(
                    value
                    for key, value in round_checks.items()
                    if key.endswith("_equal") or key.endswith("_exact")
                )
                if not round_checks["all_checks_passed"]:
                    raise AssertionError(
                        f"round reproduction discrepancy for {seed}/{mode}/r{observed_round.round_index}: {round_checks}"
                    )
                checks["rounds"][str(observed_round.round_index)] = round_checks
            checks["all_checks_passed"] = (
                checks["python_rng_state_equal"]
                and checks["numpy_rng_state_equal"]
                and checks["torch_rng_state_equal"]
                and all(item["all_checks_passed"] for item in checks["rounds"].values())
            )
            reproduction[str(seed)][mode] = checks

            snapshots = [
                (
                    0,
                    instrumented.round0_model,
                    instrumented.round0_centers,
                    instrumented.round0_similarities,
                    instrumented.round0_scores,
                )
            ] + [
                (
                    item.round_index,
                    item.model,
                    item.centers,
                    item.similarities,
                    item.scores,
                )
                for item in instrumented.rounds
            ]
            for round_index, model, centers, similarities, scores in snapshots:
                analysis = _analyze_selection(
                    seed,
                    mode,
                    round_index,
                    model,
                    centers,
                    similarities,
                    scores,
                    prepared,
                    config,
                )
                round_metrics.extend(analysis["round_metrics"])
                closed_confusion.extend(analysis["closed_confusion"])
                thresholded_confusion.extend(analysis["thresholded_confusion"])
                class_metrics.extend(analysis["class_metrics"])
                similarity_summaries.extend(analysis["similarity_summaries"])
                margin_summaries.extend(analysis["margin_summaries"])
                record_rows.extend(analysis["record_rows"])

            delta_artifact["seeds"][seed]["modes"][mode] = {"rounds": {}}
            for item in instrumented.rounds:
                total_samples = sum(client["samples"] for client in item.client_records)
                for client in item.client_records:
                    epoch = client["local_epochs"][0]
                    for class_index, class_name in enumerate(prepared.class_names):
                        signal_rows.append(
                            {
                                "seed": seed,
                                "mode": mode,
                                "round": item.round_index,
                                "client_id": client["client_id"],
                                "samples": client["samples"],
                                "fedavg_weight": client["samples"] / total_samples,
                                "class_name": class_name,
                                "realized_class_draws": epoch["drawn_class_counts"][class_index],
                                **{
                                    key: epoch["per_head_signal"][class_index][key]
                                    for key in (
                                        "active_batches",
                                        "positive_available_batches",
                                        "negative_available_batches",
                                        "target_examples_drawn",
                                        "positive_pairs",
                                        "negative_pairs",
                                        "positive_active_pairs",
                                        "negative_active_pairs",
                                    )
                                },
                            }
                        )
                for row in item.gradient_rows:
                    gradient_rows.append(
                        {"seed": seed, "mode": mode, "round": item.round_index, **row}
                    )
                proxy = SimpleNamespace(
                    round1_model=item.model,
                    client_records=[
                        {
                            "client_id": client["client_id"],
                            "samples": client["samples"],
                        }
                        for client in item.client_records
                    ],
                    local_states=item.local_states,
                    global_state=item.global_state,
                )
                updates, pairwise, cancellation, verification, tensors = _update_diagnostics(
                    seed,
                    mode,
                    proxy,
                    item.round_start_state,
                    list(prepared.class_names),
                )
                for collection in (updates, pairwise, cancellation, verification):
                    for row in collection:
                        row["round"] = item.round_index
                update_rows.extend(updates)
                pairwise_rows.extend(pairwise)
                cancellation_rows.extend(cancellation)
                verification_rows.extend(verification)
                delta_artifact["seeds"][seed]["modes"][mode]["rounds"][
                    item.round_index
                ] = tensors

        iid = trajectories["iid_federated"]
        natural = trajectories["natural_non_iid_federated"]
        iid_predictions = np.argmax(iid.round0_similarities, axis=1)
        natural_predictions = np.argmax(natural.round0_similarities, axis=1)
        round0_agreement.append(
            {
                "seed": seed,
                "center_max_abs_difference": float(
                    torch.max(torch.abs(iid.round0_centers - natural.round0_centers)).item()
                ),
                "similarity_max_abs_difference": float(
                    np.max(np.abs(iid.round0_similarities - natural.round0_similarities))
                ),
                "unknown_score_max_abs_difference": float(
                    np.max(np.abs(iid.round0_scores - natural.round0_scores))
                ),
                "closed_prediction_mismatches": int(
                    np.sum(iid_predictions != natural_predictions)
                ),
            }
        )

    round_frame = pd.DataFrame(round_metrics)
    class_frame = pd.DataFrame(class_metrics)
    similarity_frame = pd.DataFrame(similarity_summaries)
    margin_frame = pd.DataFrame(margin_summaries)
    write_json(output_dir / "reproduction_checks.json", reproduction)
    pd.DataFrame(round0_agreement).to_csv(output_dir / "round0_iid_natural_agreement.csv", index=False)
    round_frame.to_csv(output_dir / "round0_to_round3_metrics.csv", index=False)
    pd.DataFrame(closed_confusion).to_csv(output_dir / "closed_confusion_round0_to_round3.csv", index=False)
    pd.DataFrame(thresholded_confusion).to_csv(
        output_dir / "thresholded_confusion_round0_to_round3.csv", index=False
    )
    class_frame.to_csv(output_dir / "normal_sybil_class_metrics.csv", index=False)
    _class_metric_changes(class_frame).to_csv(
        output_dir / "normal_sybil_class_metric_changes.csv", index=False
    )
    similarity_frame.to_csv(output_dir / "similarity_distribution_summary.csv", index=False)
    _changes(similarity_frame, ["seed", "mode", "true_class", "measure"]).to_csv(
        output_dir / "similarity_distribution_changes.csv", index=False
    )
    margin_frame.to_csv(output_dir / "margin_distribution_summary.csv", index=False)
    _changes(margin_frame, ["seed", "mode", "true_class", "measure"]).to_csv(
        output_dir / "margin_distribution_changes.csv", index=False
    )
    pd.DataFrame(record_rows).to_csv(
        output_dir / "normal_sybil_record_scores.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(signal_rows).to_csv(output_dir / "per_client_class_signal.csv", index=False)
    pd.DataFrame(gradient_rows).to_csv(output_dir / "loss_gradient_summary.csv", index=False)
    pd.DataFrame(update_rows).to_csv(output_dir / "client_update_norms.csv", index=False)
    pd.DataFrame(pairwise_rows).to_csv(output_dir / "pairwise_update_cosines.csv", index=False)
    pd.DataFrame(cancellation_rows).to_csv(output_dir / "weighted_update_cancellation.csv", index=False)
    pd.DataFrame(verification_rows).to_csv(output_dir / "aggregation_verification.csv", index=False)
    torch.save(delta_artifact, output_dir / "actual_client_parameter_deltas_rounds1_to3.pt")
    write_json(
        output_dir / "scope_and_integrity.json",
        {
            "seeds": SEEDS,
            "modes": MODES,
            "rounds": [0, 1, 2, 3],
            "post_hoc_exploratory_examples": True,
            "known_selection_subset_only": True,
            "calibration_use": "known-only threshold calculation; never used to search or choose a configuration",
            "unknown_examples_evaluated": False,
            "final_test_performance_evaluated": False,
            "baseline_method_modified": False,
            "cosine_near_zero_norm": COSINE_NEAR_ZERO_NORM,
            "gradient_zero_norm": GRADIENT_ZERO_NORM,
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "reproduction": "passed"}, indent=2))


if __name__ == "__main__":
    main()
