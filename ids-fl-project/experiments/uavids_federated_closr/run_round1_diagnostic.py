"""RNG-neutral, known-only round-1 diagnostic replay for fixed CLOSR FedAvg runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from uavids_closr.artifacts import write_json
from uavids_closr.data import ClientDataset, make_iid_control_clients, prepare_data, resolve_config
from uavids_closr.federated import clone_state_dict, sample_weighted_fedavg
from uavids_closr.inference import (
    aggregate_center_statistics,
    calibrate_known_threshold,
    centers_from_statistics,
    known_open_set_diagnostics,
    local_center_statistics,
    score_features,
)
from uavids_closr.model import CLOSRModel
from uavids_closr.training import make_criterion, make_loader, set_deterministic, train_epoch


ROOT = Path(__file__).resolve().parent
SEEDS = [240520, 240523]
MODES = ["iid_federated", "natural_non_iid_federated"]
GRADIENT_ZERO_NORM = 1e-12
COSINE_NEAR_ZERO_NORM = 1e-5


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _parameter_groups(model: CLOSRModel, class_names: list[str]) -> OrderedDict[str, list[str]]:
    names = list(dict(model.named_parameters()))
    groups: OrderedDict[str, list[str]] = OrderedDict()
    groups["encoder"] = [name for name in names if name.startswith("encoder.")]
    for class_index, class_name in enumerate(class_names):
        prefix = f"projection_heads.{class_index}."
        groups[f"head::{class_name}"] = [name for name in names if name.startswith(prefix)]
    if set().union(*map(set, groups.values())) != set(names):
        raise AssertionError("parameter grouping did not cover the model exactly")
    return groups


def _group_vector(
    state_or_delta: dict[str, torch.Tensor], names: list[str], dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    return torch.cat([state_or_delta[name].detach().cpu().to(dtype=dtype).reshape(-1) for name in names])


def _state_delta(
    after: dict[str, torch.Tensor], before: dict[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, after[name].detach().cpu() - before[name].detach().cpu())
        for name in before
        if before[name].is_floating_point()
    )


def _rng_snapshot() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
    }


def _rng_equal(left: dict[str, Any], right: dict[str, Any]) -> dict[str, bool]:
    numpy_left = left["numpy"]
    numpy_right = right["numpy"]
    return {
        "python_rng_state_equal": left["python"] == right["python"],
        "numpy_rng_state_equal": (
            numpy_left[0] == numpy_right[0]
            and np.array_equal(numpy_left[1], numpy_right[1])
            and numpy_left[2:] == numpy_right[2:]
        ),
        "torch_rng_state_equal": torch.equal(left["torch"], right["torch"]),
    }


def _module_delta(module: torch.nn.Module, before: list[torch.Tensor]) -> float:
    squared = 0.0
    for parameter, previous in zip(module.parameters(), before):
        difference = parameter.detach().cpu() - previous
        squared += float(torch.sum(difference.square()).item())
    return float(np.sqrt(squared))


def instrumented_train_epoch(
    model: CLOSRModel,
    features: np.ndarray,
    labels: np.ndarray,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    batch_size: int,
    seed: int,
    balanced: bool,
    class_names: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Mirror ``train_epoch`` while observing tensors without drawing randomness."""
    started = time.perf_counter()
    encoder_before = [parameter.detach().cpu().clone() for parameter in model.encoder.parameters()]
    heads_before = [
        [parameter.detach().cpu().clone() for parameter in head.parameters()]
        for head in model.projection_heads
    ]
    loader = make_loader(features, labels, batch_size, seed, balanced)
    model.to(device).train()
    groups = _parameter_groups(model, class_names)
    parameters = dict(model.named_parameters())
    gradient = {
        group: {
            "optimized_batches": 0,
            "nonzero_gradient_batches": 0,
            "zero_gradient_batches": 0,
            "all_missing_gradient_batches": 0,
            "partially_missing_gradient_batches": 0,
            "parameter_gradient_observations": 0,
            "missing_parameter_gradients": 0,
            "zero_parameter_gradients": 0,
            "nonzero_parameter_gradients": 0,
            "gradient_norm_sum": 0.0,
            "gradient_norm_max": 0.0,
            "optimizer_step_path_l2_sum": 0.0,
            "movement_on_nonzero_gradient_batches_l2_sum": 0.0,
            "movement_on_zero_gradient_batches_l2_sum": 0.0,
            "movement_on_missing_gradient_batches_l2_sum": 0.0,
        }
        for group in groups
    }
    weighted_loss = 0.0
    seen = 0
    optimized_batches = 0
    skipped_batches = 0
    examples_drawn = 0
    batches_drawn = 0
    drawn_class_counts = np.zeros(model.num_classes, dtype=np.int64)
    head_signal = [
        {
            "class_index": class_index,
            "active_batches": 0,
            "positive_available_batches": 0,
            "negative_available_batches": 0,
            "target_examples_drawn": 0,
            "positive_pairs": 0,
            "negative_pairs": 0,
            "positive_active_pairs": 0,
            "negative_active_pairs": 0,
        }
        for class_index in range(model.num_classes)
    ]
    for batch_features, batch_labels in loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(batch_features), batch_labels)
        batches_drawn += 1
        examples_drawn += len(batch_labels)
        drawn_class_counts += np.bincount(
            batch_labels.detach().cpu().numpy(), minlength=model.num_classes
        )
        for item in criterion.last_diagnostics:
            record = head_signal[item.class_index]
            record["active_batches"] += int(item.active)
            record["positive_available_batches"] += int(item.positive_available)
            record["negative_available_batches"] += int(item.negative_available)
            record["target_examples_drawn"] += item.target_samples
            record["positive_pairs"] += item.positive_pairs
            record["negative_pairs"] += item.negative_pairs
            record["positive_active_pairs"] += item.positive_active_pairs
            record["negative_active_pairs"] += item.negative_active_pairs
        active_heads = sum(item.active for item in criterion.last_diagnostics)
        if active_heads == 0:
            skipped_batches += 1
            continue
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite CLOSR loss")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("non-finite CLOSR gradient")

        before_step = {
            name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
        }
        batch_status: dict[str, str] = {}
        for group_name, parameter_names in groups.items():
            row = gradient[group_name]
            row["optimized_batches"] += 1
            squared_norm = 0.0
            missing = 0
            zero_parameters = 0
            nonzero_parameters = 0
            for name in parameter_names:
                grad = parameters[name].grad
                row["parameter_gradient_observations"] += 1
                if grad is None:
                    missing += 1
                    row["missing_parameter_gradients"] += 1
                    continue
                norm = float(torch.linalg.vector_norm(grad.detach().to(torch.float64)).item())
                squared_norm += norm * norm
                if norm <= GRADIENT_ZERO_NORM:
                    zero_parameters += 1
                    row["zero_parameter_gradients"] += 1
                else:
                    nonzero_parameters += 1
                    row["nonzero_parameter_gradients"] += 1
            group_norm = float(np.sqrt(squared_norm))
            row["gradient_norm_sum"] += group_norm
            row["gradient_norm_max"] = max(row["gradient_norm_max"], group_norm)
            if missing == len(parameter_names):
                status = "missing"
                row["all_missing_gradient_batches"] += 1
            elif group_norm <= GRADIENT_ZERO_NORM:
                status = "zero"
                row["zero_gradient_batches"] += 1
            else:
                status = "nonzero"
                row["nonzero_gradient_batches"] += 1
            if 0 < missing < len(parameter_names):
                row["partially_missing_gradient_batches"] += 1
            batch_status[group_name] = status

        optimizer.step()
        for group_name, parameter_names in groups.items():
            squared_movement = 0.0
            for name in parameter_names:
                difference = parameters[name].detach().cpu() - before_step[name]
                squared_movement += float(torch.sum(difference.square()).item())
            movement = float(np.sqrt(squared_movement))
            row = gradient[group_name]
            row["optimizer_step_path_l2_sum"] += movement
            row[f"movement_on_{batch_status[group_name]}_gradient_batches_l2_sum"] += movement

        weighted_loss += float(loss.item()) * len(batch_labels)
        seen += len(batch_labels)
        optimized_batches += 1

    gradient_rows = []
    for group_name, values in gradient.items():
        count = max(int(values["optimized_batches"]), 1)
        gradient_rows.append(
            {
                "parameter_group": group_name,
                **values,
                "gradient_norm_mean": float(values["gradient_norm_sum"] / count),
                "zero_gradient_definition": f"L2 <= {GRADIENT_ZERO_NORM:g}",
            }
        )
    return (
        {
            "mean_loss": weighted_loss / max(seen, 1),
            "runtime_seconds": float(time.perf_counter() - started),
            "examples_drawn": examples_drawn,
            "batches_drawn": batches_drawn,
            "optimized_batches": optimized_batches,
            "optimizer_steps": optimized_batches,
            "skipped_batches": skipped_batches,
            "optimized_rows": seen,
            "drawn_class_counts": drawn_class_counts.tolist(),
            "per_head_signal": head_signal,
            "encoder_parameter_delta_l2": _module_delta(model.encoder, encoder_before),
            "projection_head_parameter_delta_l2": [
                _module_delta(head, previous)
                for head, previous in zip(model.projection_heads, heads_before)
            ],
        },
        gradient_rows,
    )


def _centers(model: CLOSRModel, clients: dict[str, ClientDataset], device: torch.device) -> torch.Tensor:
    model.to(device).eval()
    statistics = [
        local_center_statistics(model, client.features, client.labels, model.num_classes, device)
        for client in clients.values()
    ]
    return centers_from_statistics(aggregate_center_statistics(statistics))


@dataclass
class Replay:
    initial_model: CLOSRModel
    round1_model: CLOSRModel
    round0_centers: torch.Tensor
    round1_centers: torch.Tensor
    round0_similarities: np.ndarray
    round0_scores: np.ndarray
    round1_similarities: np.ndarray
    round1_scores: np.ndarray
    local_states: list[OrderedDict[str, torch.Tensor]]
    global_state: OrderedDict[str, torch.Tensor]
    client_records: list[dict[str, Any]]
    gradient_rows: list[dict[str, Any]]
    rng_state: dict[str, Any]


def _replay_round_one(
    model_config: dict[str, Any],
    initial_state: OrderedDict[str, torch.Tensor],
    clients: dict[str, ClientDataset],
    prepared,
    config: dict[str, Any],
    instrument: bool,
) -> Replay:
    training = config["training"]
    device = torch.device(training["device"])
    initial_model = CLOSRModel(**model_config)
    initial_model.load_state_dict(initial_state)
    round0_centers = _centers(initial_model, clients, device)
    round0_similarities, round0_scores = score_features(
        initial_model, round0_centers, prepared.selection_features, device
    )
    # Match the baseline call that computes the initial closed-set diagnostic.
    _ = f1_score(prepared.selection_labels, np.argmax(round0_similarities, axis=1), average="macro")

    local_states = []
    sample_counts = []
    client_records = []
    gradient_rows = []
    for client_index, client in enumerate(clients.values()):
        local_model = CLOSRModel(**model_config)
        local_model.load_state_dict(initial_state)
        criterion = make_criterion(config["loss"], local_model.num_classes)
        optimizer = torch.optim.AdamW(
            local_model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        epoch_seed = int(training["seed"]) + 1000 + client_index * 100 + 1
        if instrument:
            epoch_record, client_gradient_rows = instrumented_train_epoch(
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
            for row in client_gradient_rows:
                row["client_id"] = client.client_id
            gradient_rows.extend(client_gradient_rows)
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
        local_states.append(clone_state_dict(local_model))
        sample_counts.append(len(client.labels))
        client_records.append(
            {"client_id": client.client_id, "samples": len(client.labels), "epoch": epoch_record}
        )

    global_state = sample_weighted_fedavg(local_states, sample_counts)
    round1_model = CLOSRModel(**model_config)
    round1_model.load_state_dict(global_state)
    round1_centers = _centers(round1_model, clients, device)
    round1_similarities, round1_scores = score_features(
        round1_model, round1_centers, prepared.selection_features, device
    )
    _ = f1_score(prepared.selection_labels, np.argmax(round1_similarities, axis=1), average="macro")
    return Replay(
        initial_model=initial_model,
        round1_model=round1_model,
        round0_centers=round0_centers,
        round1_centers=round1_centers,
        round0_similarities=round0_similarities,
        round0_scores=round0_scores,
        round1_similarities=round1_similarities,
        round1_scores=round1_scores,
        local_states=local_states,
        global_state=global_state,
        client_records=client_records,
        gradient_rows=gradient_rows,
        rng_state=_rng_snapshot(),
    )


def _state_comparison(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> dict[str, Any]:
    max_abs = 0.0
    exact = True
    for name in left:
        exact = exact and torch.equal(left[name], right[name])
        max_abs = max(
            max_abs,
            float(torch.max(torch.abs(left[name].to(torch.float64) - right[name].to(torch.float64))).item()),
        )
    return {"bitwise_equal": exact, "max_abs_difference": max_abs}


def _known_round_diagnostics(
    seed: int,
    mode: str,
    round_index: int,
    model: CLOSRModel,
    centers: torch.Tensor,
    similarities: np.ndarray,
    scores: np.ndarray,
    prepared,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    device = torch.device(config["training"]["device"])
    calibration_similarities, calibration_scores = score_features(
        model, centers, prepared.calibration_features, device
    )
    threshold, observed = calibrate_known_threshold(
        calibration_scores, float(config["threshold_calibration"]["known_rejection_rate_target"])
    )
    class_names = list(prepared.class_names)
    benign_index = class_names.index(config["benign_class"])
    thresholded = known_open_set_diagnostics(
        similarities,
        scores,
        prepared.selection_labels,
        threshold,
        class_names,
        benign_index,
    )
    closed_predictions = np.argmax(similarities, axis=1).astype(np.int64)
    closed_matrix = confusion_matrix(
        prepared.selection_labels, closed_predictions, labels=list(range(len(class_names)))
    )
    row = {
        "seed": seed,
        "mode": mode,
        "round": round_index,
        "closed_set_accuracy": float(accuracy_score(prepared.selection_labels, closed_predictions)),
        "closed_set_macro_f1": float(
            f1_score(
                prepared.selection_labels,
                closed_predictions,
                labels=list(range(len(class_names))),
                average="macro",
                zero_division=0,
            )
        ),
        "known_calibration_threshold": threshold,
        "known_calibration_rejection_rate": observed,
        "thresholded_selection_macro_f1": float(thresholded["macro_f1_after_rejection"]),
        "thresholded_selection_known_rejection_rate": float(thresholded["known_rejection_rate"]),
        "thresholded_selection_benign_rejection_rate": float(thresholded["benign_rejection_rate"]),
        "thresholded_selection_benign_total_alert_fpr": float(
            thresholded["benign_total_alert_fpr"]
        ),
        "calibration_group_use": "known-only threshold calibration; not performance evaluation",
    }
    confusion_rows = []
    for variant, matrix, columns in (
        ("closed_set", closed_matrix, class_names),
        (
            "thresholded",
            np.asarray(thresholded["confusion_matrix"], dtype=np.int64),
            class_names + ["UNKNOWN"],
        ),
    ):
        true_rows = class_names if variant == "closed_set" else class_names + ["UNKNOWN"]
        for true_index, true_name in enumerate(true_rows):
            for pred_index, pred_name in enumerate(columns):
                confusion_rows.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "round": round_index,
                        "variant": variant,
                        "true_label": true_name,
                        "predicted_label": pred_name,
                        "count": int(matrix[true_index, pred_index]),
                    }
                )
    return row, confusion_rows


def _update_diagnostics(
    seed: int,
    mode: str,
    replay: Replay,
    initial_state: OrderedDict[str, torch.Tensor],
    class_names: list[str],
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    groups = _parameter_groups(replay.round1_model, class_names)
    total_samples = sum(client["samples"] for client in replay.client_records)
    client_deltas = [_state_delta(state, initial_state) for state in replay.local_states]
    aggregate_delta = _state_delta(replay.global_state, initial_state)
    update_rows = []
    pairwise_rows = []
    cancellation_rows = []
    verification_rows = []
    tensor_artifact = {"clients": {}, "aggregate_delta": aggregate_delta}

    for client, delta in zip(replay.client_records, client_deltas):
        weight = client["samples"] / total_samples
        weighted_delta = OrderedDict(
            (name, value.to(torch.float64) * weight) for name, value in delta.items()
        )
        tensor_artifact["clients"][client["client_id"]] = {
            "samples": client["samples"],
            "fedavg_weight": weight,
            "delta": delta,
            "weighted_delta": weighted_delta,
        }
        for group_name, names in groups.items():
            vector = _group_vector(delta, names)
            contribution = vector * weight
            update_rows.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "client_id": client["client_id"],
                    "samples": client["samples"],
                    "fedavg_weight": weight,
                    "parameter_group": group_name,
                    "client_delta_l2": float(torch.linalg.vector_norm(vector).item()),
                    "weighted_contribution_l2": float(
                        torch.linalg.vector_norm(contribution).item()
                    ),
                }
            )

    for group_name, names in groups.items():
        vectors = [_group_vector(delta, names) for delta in client_deltas]
        weights = [client["samples"] / total_samples for client in replay.client_records]
        weighted_vectors = [vector * weight for vector, weight in zip(vectors, weights)]
        weighted_sum = torch.stack(weighted_vectors).sum(dim=0)
        contribution_norm_sum = sum(float(torch.linalg.vector_norm(item).item()) for item in weighted_vectors)
        aggregate_norm = float(torch.linalg.vector_norm(weighted_sum).item())
        defined = contribution_norm_sum > COSINE_NEAR_ZERO_NORM
        retention = aggregate_norm / contribution_norm_sum if defined else None
        cancellation_rows.append(
            {
                "seed": seed,
                "mode": mode,
                "parameter_group": group_name,
                "sum_weighted_client_update_norms": contribution_norm_sum,
                "weighted_aggregate_update_norm": aggregate_norm,
                "retention_ratio": retention,
                "cancellation_fraction": (1.0 - retention) if retention is not None else None,
                "defined": defined,
                "definition": "retention=norm(weighted sum)/sum(norm(weighted client delta)); cancellation=1-retention",
            }
        )
        actual = _group_vector(aggregate_delta, names)
        error = actual - weighted_sum
        max_abs = float(torch.max(torch.abs(error)).item())
        tolerance = 2e-7 + 1e-6 * float(torch.max(torch.abs(weighted_sum)).item())
        verification_rows.append(
            {
                "seed": seed,
                "mode": mode,
                "parameter_group": group_name,
                "max_abs_actual_minus_weighted_delta": max_abs,
                "l2_actual_minus_weighted_delta": float(torch.linalg.vector_norm(error).item()),
                "floating_cast_tolerance": tolerance,
                "matches_within_float32_fedavg_tolerance": max_abs <= tolerance,
            }
        )
        for left_index in range(len(vectors)):
            for right_index in range(left_index + 1, len(vectors)):
                left = vectors[left_index]
                right = vectors[right_index]
                left_norm = float(torch.linalg.vector_norm(left).item())
                right_norm = float(torch.linalg.vector_norm(right).item())
                cosine_defined = (
                    left_norm > COSINE_NEAR_ZERO_NORM
                    and right_norm > COSINE_NEAR_ZERO_NORM
                )
                cosine = (
                    float(torch.dot(left, right).item() / (left_norm * right_norm))
                    if cosine_defined
                    else None
                )
                pairwise_rows.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "parameter_group": group_name,
                        "left_client": replay.client_records[left_index]["client_id"],
                        "right_client": replay.client_records[right_index]["client_id"],
                        "left_norm": left_norm,
                        "right_norm": right_norm,
                        "cosine_defined": cosine_defined,
                        "cosine_similarity": cosine,
                        "near_zero_threshold": COSINE_NEAR_ZERO_NORM,
                    }
                )
    return update_rows, pairwise_rows, cancellation_rows, verification_rows, tensor_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "contribution1_baseline.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "contribution1-round1-exploratory-240520-240523-final",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output_dir}")
    output_dir.mkdir(parents=True)

    reproduction = {}
    round_rows = []
    confusion_rows = []
    agreement_rows = []
    signal_rows = []
    gradient_rows = []
    update_rows = []
    pairwise_rows = []
    cancellation_rows = []
    verification_rows = []
    delta_artifact = {
        "description": "Actual round-1 client deltas from the common global start and FedAvg-weighted contributions",
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
        saved_rng_audit = json.loads(
            (ROOT / "runs" / f"contribution1-three-way-seed-{seed}" / "rng_audit.json").read_text()
        )
        if initial_hash != saved_rng_audit["initial_state_sha256"]:
            raise AssertionError(f"initial-state hash mismatch for seed {seed}")
        delta_artifact["seeds"][seed] = {"initial_state_sha256": initial_hash, "modes": {}}
        mode_replays = {}
        reproduction[str(seed)] = {}
        for mode, clients in (
            ("iid_federated", iid_clients),
            ("natural_non_iid_federated", prepared.clients),
        ):
            set_deterministic(seed, int(config["training"]["torch_num_threads"]))
            baseline = _replay_round_one(
                model_config, initial_state, clients, prepared, config, instrument=False
            )
            set_deterministic(seed, int(config["training"]["torch_num_threads"]))
            instrumented = _replay_round_one(
                model_config, initial_state, clients, prepared, config, instrument=True
            )
            mode_replays[mode] = instrumented
            global_comparison = _state_comparison(baseline.global_state, instrumented.global_state)
            local_comparisons = [
                _state_comparison(left, right)
                for left, right in zip(baseline.local_states, instrumented.local_states)
            ]
            rng_checks = _rng_equal(baseline.rng_state, instrumented.rng_state)
            existing = json.loads(
                (
                    ROOT
                    / "runs"
                    / f"contribution1-three-way-seed-{seed}"
                    / "run_summary.json"
                ).read_text()
            )[mode]["history"][0]
            observed_predictions = np.argmax(instrumented.round1_similarities, axis=1)
            observed_accuracy = float(accuracy_score(prepared.selection_labels, observed_predictions))
            observed_macro = float(
                f1_score(prepared.selection_labels, observed_predictions, average="macro", zero_division=0)
            )
            aggregate_epoch_records = [record["epoch"] for record in instrumented.client_records]
            optimized_rows = sum(record["optimized_rows"] for record in aggregate_epoch_records)
            observed_mean_loss = float(
                sum(record["mean_loss"] * record["optimized_rows"] for record in aggregate_epoch_records)
                / optimized_rows
            )
            observed_draws = np.sum(
                [record["drawn_class_counts"] for record in aggregate_epoch_records], axis=0
            ).astype(np.int64).tolist()
            existing_checks = {
                "selection_accuracy_exact": observed_accuracy == float(existing["selection_accuracy"]),
                "selection_macro_f1_exact": observed_macro == float(existing["selection_macro_f1"]),
                "mean_local_loss_exact": observed_mean_loss == float(existing["mean_local_loss"]),
                "drawn_class_counts_exact": observed_draws == existing["drawn_class_counts"],
                "optimizer_steps_exact": sum(
                    int(record["optimizer_steps"]) for record in aggregate_epoch_records
                )
                == int(existing["optimizer_steps"]),
            }
            checks = {
                "initial_state_sha256": initial_hash,
                "global_state": global_comparison,
                "all_client_states_bitwise_equal": all(item["bitwise_equal"] for item in local_comparisons),
                "max_client_state_abs_difference": max(
                    item["max_abs_difference"] for item in local_comparisons
                ),
                **rng_checks,
                "existing_full_run_round1": {
                    **existing_checks,
                    "expected_selection_macro_f1": float(existing["selection_macro_f1"]),
                    "replayed_selection_macro_f1": observed_macro,
                },
            }
            boolean_checks = [
                global_comparison["bitwise_equal"],
                checks["all_client_states_bitwise_equal"],
                *rng_checks.values(),
                *existing_checks.values(),
            ]
            checks["all_reproduction_checks_passed"] = all(boolean_checks)
            if not checks["all_reproduction_checks_passed"]:
                raise AssertionError(f"round-1 reproduction discrepancy for {seed}/{mode}: {checks}")
            reproduction[str(seed)][mode] = checks

            for client, gradients in zip(
                instrumented.client_records,
                [
                    [row for row in instrumented.gradient_rows if row["client_id"] == item["client_id"]]
                    for item in instrumented.client_records
                ],
            ):
                weight = client["samples"] / sum(item["samples"] for item in instrumented.client_records)
                epoch = client["epoch"]
                for class_index, class_name in enumerate(prepared.class_names):
                    signal_rows.append(
                        {
                            "seed": seed,
                            "mode": mode,
                            "client_id": client["client_id"],
                            "samples": client["samples"],
                            "fedavg_weight": weight,
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
                            "pair_count_note": "positive_pairs includes target diagonals as in pinned upstream; active counts exclude zero-distance/inactive hinge values",
                        }
                    )
                for row in gradients:
                    gradient_rows.append({"seed": seed, "mode": mode, **row})

            updates, pairwise, cancellation, verification, tensors = _update_diagnostics(
                seed, mode, instrumented, initial_state, list(prepared.class_names)
            )
            update_rows.extend(updates)
            pairwise_rows.extend(pairwise)
            cancellation_rows.extend(cancellation)
            verification_rows.extend(verification)
            delta_artifact["seeds"][seed]["modes"][mode] = tensors

            for round_index, model, centers, similarities, scores in (
                (
                    0,
                    instrumented.initial_model,
                    instrumented.round0_centers,
                    instrumented.round0_similarities,
                    instrumented.round0_scores,
                ),
                (
                    1,
                    instrumented.round1_model,
                    instrumented.round1_centers,
                    instrumented.round1_similarities,
                    instrumented.round1_scores,
                ),
            ):
                metric_row, matrix_rows = _known_round_diagnostics(
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
                round_rows.append(metric_row)
                confusion_rows.extend(matrix_rows)

        iid = mode_replays["iid_federated"]
        natural = mode_replays["natural_non_iid_federated"]
        iid_predictions = np.argmax(iid.round0_similarities, axis=1)
        natural_predictions = np.argmax(natural.round0_similarities, axis=1)
        agreement_rows.append(
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
                "closed_prediction_mismatches": int(np.sum(iid_predictions != natural_predictions)),
                "selection_rows": len(prepared.selection_labels),
                "centers_close": bool(
                    torch.allclose(iid.round0_centers, natural.round0_centers, atol=2e-7, rtol=1e-6)
                ),
                "scores_close": bool(
                    np.allclose(iid.round0_scores, natural.round0_scores, atol=2e-7, rtol=1e-6)
                ),
            }
        )

    write_json(output_dir / "reproduction_checks.json", reproduction)
    pd.DataFrame(agreement_rows).to_csv(output_dir / "round0_iid_natural_agreement.csv", index=False)
    pd.DataFrame(signal_rows).to_csv(output_dir / "per_client_class_signal.csv", index=False)
    pd.DataFrame(gradient_rows).to_csv(output_dir / "loss_gradient_summary.csv", index=False)
    pd.DataFrame(update_rows).to_csv(output_dir / "client_update_norms.csv", index=False)
    pd.DataFrame(pairwise_rows).to_csv(output_dir / "pairwise_update_cosines.csv", index=False)
    pd.DataFrame(cancellation_rows).to_csv(output_dir / "weighted_update_cancellation.csv", index=False)
    pd.DataFrame(verification_rows).to_csv(output_dir / "aggregation_verification.csv", index=False)
    pd.DataFrame(round_rows).to_csv(output_dir / "round0_round1_metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(output_dir / "round0_round1_confusion_long.csv", index=False)
    torch.save(delta_artifact, output_dir / "actual_client_parameter_deltas.pt")
    write_json(
        output_dir / "scope_and_integrity.json",
        {
            "seeds": SEEDS,
            "modes": MODES,
            "post_hoc_exploratory_examples": True,
            "rounds_replayed": [1],
            "known_only": True,
            "unknown_validation_evaluated": False,
            "final_test_performance_evaluated": False,
            "baseline_method_modified": False,
            "near_zero_vector_cosine_rule": (
                f"undefined when either L2 norm <= {COSINE_NEAR_ZERO_NORM:g}; "
                "this excludes absent-head AdamW weight-decay-only movements that are over four "
                "orders of magnitude smaller than trained client updates"
            ),
            "zero_gradient_rule": f"zero when L2 norm <= {GRADIENT_ZERO_NORM:g}",
            "instrumentation_rng_rule": "tensor reads, clones, reductions, and serialization only; no random draw in instrumentation",
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "reproduction": "passed"}, indent=2))


if __name__ == "__main__":
    main()
