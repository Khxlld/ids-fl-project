"""Centralized and explicit five-client FedAvg CLOSR training loops."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .data import ClientDataset
from .federated import clone_state_dict, sample_weighted_fedavg
from .inference import (
    aggregate_center_statistics,
    calibrate_known_threshold,
    calibration_metrics,
    centers_from_statistics,
    known_class_metrics,
    known_open_set_diagnostics,
    local_center_statistics,
    score_features,
)
from .loss import CLOSRLoss
from .model import CLOSRModel


@dataclass
class BaselineResult:
    model: CLOSRModel
    centers: torch.Tensor
    threshold: float
    selection_metrics: dict[str, Any]
    calibration_metrics: dict[str, Any]
    history: list[dict[str, Any]]
    best_step: int
    initial_selection_metrics: dict[str, Any]
    post_calibration_selection_metrics: dict[str, Any]
    runtime_seconds: float


def set_deterministic(seed: int, torch_num_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(torch_num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    seed: int,
    balanced: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.array(features, dtype=np.float32, order="C", copy=True)),
        torch.from_numpy(np.array(labels, dtype=np.int64, order="C", copy=True)),
    )
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    shuffle = not balanced
    if balanced:
        counts = np.bincount(labels)
        weights = np.asarray([1.0 / counts[label] for label in labels], dtype=np.float64)
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights), num_samples=len(weights), replacement=True, generator=generator
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


def make_criterion(loss_config: dict[str, Any], num_classes: int) -> CLOSRLoss:
    return CLOSRLoss(
        num_classes=num_classes,
        margin=loss_config["margin"],
        squared=loss_config["squared"],
        alpha=loss_config["alpha"],
        normalization=loss_config["normalization"],
        head_reduction=loss_config["head_reduction"],
        epsilon=loss_config["epsilon"],
    )


def train_epoch(
    model: CLOSRModel,
    features: np.ndarray,
    labels: np.ndarray,
    optimizer: torch.optim.Optimizer,
    criterion: CLOSRLoss,
    device: torch.device,
    batch_size: int,
    seed: int,
    balanced: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    encoder_before = [parameter.detach().cpu().clone() for parameter in model.encoder.parameters()]
    heads_before = [
        [parameter.detach().cpu().clone() for parameter in head.parameters()]
        for head in model.projection_heads
    ]
    loader = make_loader(features, labels, batch_size, seed, balanced)
    model.to(device).train()
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
        optimizer.step()
        weighted_loss += float(loss.item()) * len(batch_labels)
        seen += len(batch_labels)
        optimized_batches += 1
    def module_delta(module: torch.nn.Module, before: list[torch.Tensor]) -> float:
        squared_delta = 0.0
        for parameter, previous in zip(module.parameters(), before):
            difference = parameter.detach().cpu() - previous
            squared_delta += float(torch.sum(difference.square()).item())
        return float(np.sqrt(squared_delta))

    return {
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
        "encoder_parameter_delta_l2": module_delta(model.encoder, encoder_before),
        "projection_head_parameter_delta_l2": [
            module_delta(head, previous)
            for head, previous in zip(model.projection_heads, heads_before)
        ],
    }


def _centers_for_datasets(
    model: CLOSRModel,
    datasets: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> torch.Tensor:
    # Candidate selection must use centers from the exact candidate model in
    # inference mode.  Callers pass known training data only.
    model.to(device).eval()
    statistics = [
        local_center_statistics(model, features, labels, model.num_classes, device)
        for features, labels in datasets
    ]
    return centers_from_statistics(aggregate_center_statistics(statistics))


def _selection_metrics(
    model: CLOSRModel,
    centers: torch.Tensor,
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> dict[str, float | int]:
    similarities, _ = score_features(model, centers, features, device)
    return known_class_metrics(similarities, labels)


def _post_calibration_known_diagnostics(
    model: CLOSRModel,
    centers: torch.Tensor,
    features: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    class_names: list[str],
    benign_index: int,
    device: torch.device,
) -> dict[str, Any]:
    similarities, scores = score_features(model, centers, features, device)
    return known_open_set_diagnostics(
        similarities,
        scores,
        labels,
        threshold,
        class_names=class_names,
        benign_index=benign_index,
    )


def _state_group_deltas(before, after, num_classes: int) -> dict[str, Any]:
    def delta_for(prefix: str) -> float:
        squared = 0.0
        for name in before:
            if name.startswith(prefix):
                difference = after[name].to(dtype=torch.float64) - before[name].to(dtype=torch.float64)
                squared += float(torch.sum(difference.square()).item())
        return float(np.sqrt(squared))

    return {
        "encoder_parameter_delta_l2": delta_for("encoder."),
        "projection_head_parameter_delta_l2": [
            delta_for(f"projection_heads.{class_index}.")
            for class_index in range(num_classes)
        ],
    }


def _calibration_metrics(
    model: CLOSRModel,
    centers: torch.Tensor,
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    rejection_target: float,
    class_names: list[str],
    benign_index: int,
) -> tuple[float, dict[str, Any]]:
    similarities, scores = score_features(model, centers, features, device)
    threshold, observed_rejection = calibrate_known_threshold(scores, rejection_target)
    metrics = known_class_metrics(similarities, labels)
    metrics.update(
        {
            "known_rejection_rate_target": float(rejection_target),
            "known_rejection_rate_observed": float(observed_rejection),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
        }
    )
    metrics.update(
        calibration_metrics(
            similarities,
            scores,
            labels,
            threshold,
            class_names=class_names,
            benign_index=benign_index,
        )
    )
    metrics.update(
        known_open_set_diagnostics(
            similarities,
            scores,
            labels,
            threshold,
            class_names=class_names,
            benign_index=benign_index,
        )
    )
    return threshold, metrics


def fit_centralized(
    model: CLOSRModel,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    selection_features: np.ndarray,
    selection_labels: np.ndarray,
    calibration_features: np.ndarray,
    calibration_labels: np.ndarray,
    config: dict[str, Any],
) -> BaselineResult:
    fit_started = time.perf_counter()
    training = config["training"]
    device = torch.device(training["device"])
    criterion = make_criterion(config["loss"], model.num_classes)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    best_state = clone_state_dict(model)
    best_score = -float("inf")
    best_step = 0
    stale = 0
    history: list[dict[str, Any]] = []
    initial_centers = _centers_for_datasets(model, [(train_features, train_labels)], device)
    initial_selection_metrics = _selection_metrics(
        model, initial_centers, selection_features, selection_labels, device
    )
    for epoch in range(1, int(training["centralized_epochs"]) + 1):
        train_metrics = train_epoch(
            model,
            train_features,
            train_labels,
            optimizer,
            criterion,
            device,
            int(training["batch_size"]),
            int(training["seed"]) + epoch,
            bool(training["balanced_sampling"]),
        )
        centers = _centers_for_datasets(model, [(train_features, train_labels)], device)
        validation = _selection_metrics(model, centers, selection_features, selection_labels, device)
        history.append({"epoch": epoch, **train_metrics, **{f"selection_{k}": v for k, v in validation.items()}})
        score = float(validation["macro_f1"])
        if score > best_score:
            best_score = score
            best_state = clone_state_dict(model)
            best_step = epoch
            stale = 0
        else:
            stale += 1
        if stale >= int(training["centralized_patience"]):
            break

    model.load_state_dict(best_state)
    centers = _centers_for_datasets(model, [(train_features, train_labels)], device)
    selection_metrics = _selection_metrics(model, centers, selection_features, selection_labels, device)
    threshold, calibration_metrics = _calibration_metrics(
        model,
        centers,
        calibration_features,
        calibration_labels,
        device,
        float(config["threshold_calibration"]["known_rejection_rate_target"]),
        list(config["known_classes"]),
        list(config["known_classes"]).index(config["benign_class"]),
    )
    post_calibration_selection_metrics = _post_calibration_known_diagnostics(
        model,
        centers,
        selection_features,
        selection_labels,
        threshold,
        list(config["known_classes"]),
        list(config["known_classes"]).index(config["benign_class"]),
        device,
    )
    return BaselineResult(
        model=model,
        centers=centers,
        threshold=threshold,
        selection_metrics=selection_metrics,
        calibration_metrics=calibration_metrics,
        history=history,
        best_step=best_step,
        initial_selection_metrics=initial_selection_metrics,
        post_calibration_selection_metrics=post_calibration_selection_metrics,
        runtime_seconds=float(time.perf_counter() - fit_started),
    )


def fit_federated(
    model_config: dict[str, Any],
    initial_state,
    clients: dict[str, ClientDataset],
    selection_features: np.ndarray,
    selection_labels: np.ndarray,
    calibration_features: np.ndarray,
    calibration_labels: np.ndarray,
    config: dict[str, Any],
) -> BaselineResult:
    fit_started = time.perf_counter()
    training = config["training"]
    device = torch.device(training["device"])
    global_state = clone_state_dict(initial_state)
    best_state = clone_state_dict(initial_state)
    best_score = -float("inf")
    best_step = 0
    history: list[dict[str, Any]] = []
    ordered_clients = list(clients.values())
    initial_model = CLOSRModel(**model_config)
    initial_model.load_state_dict(initial_state)
    initial_centers = _centers_for_datasets(
        initial_model,
        [(client.features, client.labels) for client in ordered_clients],
        device,
    )
    initial_selection_metrics = _selection_metrics(
        initial_model, initial_centers, selection_features, selection_labels, device
    )

    for server_round in range(1, int(training["federated_rounds"]) + 1):
        round_started = time.perf_counter()
        round_start_state = clone_state_dict(global_state)
        local_states = []
        sample_counts = []
        client_records: list[dict[str, Any]] = []
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
                local_history.append(
                    train_epoch(
                        local_model,
                        client.features,
                        client.labels,
                        optimizer,
                        criterion,
                        device,
                        int(training["batch_size"]),
                        int(training["seed"]) + server_round * 1000 + client_index * 100 + local_epoch,
                        bool(training["balanced_sampling"]),
                    )
                )
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
        centers = _centers_for_datasets(
            global_model,
            [(client.features, client.labels) for client in ordered_clients],
            device,
        )
        validation = _selection_metrics(
            global_model, centers, selection_features, selection_labels, device
        )
        flat_local_epochs = [
            epoch_record
            for client_record in client_records
            for epoch_record in client_record["local_epochs"]
        ]
        total_optimized_rows = sum(int(record["optimized_rows"]) for record in flat_local_epochs)
        aggregate_drawn_counts = np.sum(
            [record["drawn_class_counts"] for record in flat_local_epochs], axis=0
        ).astype(np.int64)
        aggregate_head_signal: list[dict[str, int]] = []
        for class_index in range(global_model.num_classes):
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
            aggregate_head_signal.append(
                {
                    "class_index": class_index,
                    **{
                        key: int(
                            sum(record["per_head_signal"][class_index][key] for record in flat_local_epochs)
                        )
                        for key in keys
                    },
                }
            )
        parameter_deltas = _state_group_deltas(
            round_start_state, global_state, global_model.num_classes
        )
        history.append(
            {
                "round": server_round,
                "runtime_seconds": float(time.perf_counter() - round_started),
                "aggregation_sample_counts": sample_counts,
                "examples_drawn": int(sum(record["examples_drawn"] for record in flat_local_epochs)),
                "optimizer_steps": int(sum(record["optimizer_steps"] for record in flat_local_epochs)),
                "optimized_rows": int(total_optimized_rows),
                "drawn_class_counts": aggregate_drawn_counts.tolist(),
                "mean_local_loss": float(
                    sum(float(record["mean_loss"]) * int(record["optimized_rows"]) for record in flat_local_epochs)
                    / max(total_optimized_rows, 1)
                ),
                "per_head_signal": aggregate_head_signal,
                **parameter_deltas,
                "clients": client_records,
                **{f"selection_{key}": value for key, value in validation.items()},
            }
        )
        score = float(validation["macro_f1"])
        if score > best_score:
            best_score = score
            best_state = clone_state_dict(global_state)
            best_step = server_round

    selected_model = CLOSRModel(**model_config)
    selected_model.load_state_dict(best_state)
    centers = _centers_for_datasets(
        selected_model,
        [(client.features, client.labels) for client in ordered_clients],
        device,
    )
    selection_metrics = _selection_metrics(
        selected_model, centers, selection_features, selection_labels, device
    )
    threshold, calibration_metrics = _calibration_metrics(
        selected_model,
        centers,
        calibration_features,
        calibration_labels,
        device,
        float(config["threshold_calibration"]["known_rejection_rate_target"]),
        list(config["known_classes"]),
        list(config["known_classes"]).index(config["benign_class"]),
    )
    post_calibration_selection_metrics = _post_calibration_known_diagnostics(
        selected_model,
        centers,
        selection_features,
        selection_labels,
        threshold,
        list(config["known_classes"]),
        list(config["known_classes"]).index(config["benign_class"]),
        device,
    )
    return BaselineResult(
        model=selected_model,
        centers=centers,
        threshold=threshold,
        selection_metrics=selection_metrics,
        calibration_metrics=calibration_metrics,
        history=history,
        best_step=best_step,
        initial_selection_metrics=initial_selection_metrics,
        post_calibration_selection_metrics=post_calibration_selection_metrics,
        runtime_seconds=float(time.perf_counter() - fit_started),
    )
