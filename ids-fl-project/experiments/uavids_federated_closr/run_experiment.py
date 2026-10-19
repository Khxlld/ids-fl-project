"""Run centralized and FedAvg CLOSR without touching the final test partition."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from uavids_closr.artifacts import save_model_bundle, write_json
from uavids_closr.data import make_iid_control_clients, prepare_data, resolve_config
from uavids_closr.federated import clone_state_dict
from uavids_closr.model import CLOSRModel
from uavids_closr.training import fit_centralized, fit_federated, set_deterministic


ROOT = Path(__file__).resolve().parent


def _counts(labels: np.ndarray, class_names: list[str]) -> dict[str, int]:
    return {name: int(np.sum(labels == index)) for index, name in enumerate(class_names)}


def _sampling_plan(prepared, iid_clients, config: dict) -> dict:
    training = config["training"]
    batch_size = int(training["batch_size"])
    local_epochs = int(training["local_epochs"])
    class_names = list(prepared.class_names)
    pooled_features, pooled_labels = prepared.pooled_training()

    def client_policy(clients) -> tuple[dict, list[float]]:
        records = {}
        aggregate_expected = np.zeros(len(class_names), dtype=np.float64)
        for client_id, client in clients.items():
            raw_counts = np.bincount(client.labels, minlength=len(class_names))
            present = raw_counts > 0
            probabilities = present.astype(np.float64) / present.sum()
            expected_draws = probabilities * len(client.labels) * local_epochs
            aggregate_expected += expected_draws
            records[client_id] = {
                "retained_rows": int(len(client.labels)),
                "raw_class_counts": _counts(client.labels, class_names),
                "classes_present": [class_names[index] for index in np.flatnonzero(present)],
                "sampling": "inverse-local-class-frequency WeightedRandomSampler",
                "replacement": True,
                "examples_drawn_per_local_epoch": int(len(client.labels)),
                "expected_class_probability_per_draw": {
                    name: float(probabilities[index]) for index, name in enumerate(class_names)
                },
                "optimizer_steps_per_local_epoch": int(math.ceil(len(client.labels) / batch_size)),
            }
        return records, aggregate_expected.tolist()

    natural_records, natural_expected = client_policy(prepared.clients)
    iid_records, iid_expected = client_policy(iid_clients)
    central_counts = np.bincount(pooled_labels, minlength=len(class_names))
    central_present = central_counts > 0
    central_probabilities = central_present.astype(np.float64) / central_present.sum()
    central_steps = math.ceil(len(pooled_labels) / batch_size)
    federated_steps_per_round = sum(
        math.ceil(len(client.labels) / batch_size) * local_epochs
        for client in prepared.clients.values()
    )
    return {
        "seed": int(training["seed"]),
        "frozen_unknown_exclusion": config["globally_withheld_unknown_classes"],
        "batch_size": batch_size,
        "drop_last": False,
        "centralized": {
            "retained_rows": int(len(pooled_labels)),
            "raw_class_counts": _counts(pooled_labels, class_names),
            "sampling": "inverse-pooled-class-frequency WeightedRandomSampler",
            "replacement": True,
            "examples_drawn_per_epoch": int(len(pooled_labels)),
            "expected_class_probability_per_draw": {
                name: float(central_probabilities[index]) for index, name in enumerate(class_names)
            },
            "optimizer_steps_per_epoch": int(central_steps),
            "configured_max_epochs": int(training["centralized_epochs"]),
            "early_stopping_patience": int(training["centralized_patience"]),
            "configured_max_examples_drawn": int(len(pooled_labels) * training["centralized_epochs"]),
            "configured_max_optimizer_steps": int(central_steps * training["centralized_epochs"]),
        },
        "iid_fedavg": {
            "clients": iid_records,
            "expected_aggregate_class_draws_per_round": {
                name: float(iid_expected[index]) for index, name in enumerate(class_names)
            },
            "configured_rounds": int(training["federated_rounds"]),
            "local_epochs": local_epochs,
            "examples_drawn_per_round": int(len(pooled_labels) * local_epochs),
            "optimizer_steps_per_round": int(federated_steps_per_round),
            "configured_examples_drawn": int(
                len(pooled_labels) * local_epochs * training["federated_rounds"]
            ),
            "configured_optimizer_steps": int(
                federated_steps_per_round * training["federated_rounds"]
            ),
        },
        "natural_non_iid_fedavg": {
            "clients": natural_records,
            "expected_aggregate_class_draws_per_round": {
                name: float(natural_expected[index]) for index, name in enumerate(class_names)
            },
            "configured_rounds": int(training["federated_rounds"]),
            "local_epochs": local_epochs,
            "examples_drawn_per_round": int(len(pooled_labels) * local_epochs),
            "optimizer_steps_per_round": int(federated_steps_per_round),
            "configured_examples_drawn": int(
                len(pooled_labels) * local_epochs * training["federated_rounds"]
            ),
            "configured_optimizer_steps": int(
                federated_steps_per_round * training["federated_rounds"]
            ),
        },
        "effective_objective_warning": (
            "The CLOSR loss formula is shared, but class-balanced replacement sampling is applied "
            "globally for centralized training and separately within each FL client. Missing local "
            "classes receive zero draws, so natural Non-IID FedAvg has different effective class "
            "weighting and local optimization trajectories from centralized and IID FedAvg."
        ),
    }


def _period_effort(records: list[dict], num_classes: int) -> dict:
    drawn_counts = np.sum(
        [record["drawn_class_counts"] for record in records],
        axis=0,
        dtype=np.int64,
    )
    head_keys = [
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
                    key: int(sum(record["per_head_signal"][class_index][key] for record in records))
                    for key in head_keys
                },
                "cumulative_parameter_path_l2": float(
                    sum(record["projection_head_parameter_delta_l2"][class_index] for record in records)
                ),
            }
        )
    return {
        "completed_steps": len(records),
        "examples_drawn": int(sum(record["examples_drawn"] for record in records)),
        "optimizer_steps": int(sum(record["optimizer_steps"] for record in records)),
        "optimized_rows": int(sum(record["optimized_rows"] for record in records)),
        "drawn_class_counts": drawn_counts.tolist(),
        "recorded_step_runtime_seconds": float(sum(record["runtime_seconds"] for record in records)),
        "per_head_training_signal": per_head,
    }


def _actual_effort(result, initial_state) -> dict:
    records = result.history
    selected_records = records[: int(result.best_step)]
    selected_period = _period_effort(selected_records, result.model.num_classes)
    selected_state = result.model.state_dict()
    selected_deltas = []
    for class_index in range(result.model.num_classes):
        squared = 0.0
        prefix = f"projection_heads.{class_index}."
        for name in initial_state:
            if name.startswith(prefix):
                difference = selected_state[name].detach().cpu().to(torch.float64) - initial_state[name].to(torch.float64)
                squared += float(torch.sum(difference.square()).item())
        selected_deltas.append(float(np.sqrt(squared)))
    selected_period["projection_head_net_delta_from_initial_l2"] = selected_deltas
    return {
        "best_step": int(result.best_step),
        "total_completed_steps": len(records),
        "initial_selection_macro_f1": float(result.initial_selection_metrics["macro_f1"]),
        "best_selection_macro_f1": float(result.selection_metrics["macro_f1"]),
        "selected_checkpoint_period": selected_period,
        "total_completed_period": _period_effort(records, result.model.num_classes),
        "total_wall_runtime_seconds": float(result.runtime_seconds),
        "measurement_note": (
            "Target-signal counts and cumulative head path lengths share the stated period. "
            "Net-from-initial head movement is available only for the selected checkpoint and "
            "is therefore paired only with selected-checkpoint-period signals."
        ),
    }


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _save_diagnostics(output_dir: Path, results: dict, class_names: list[str], initial_state) -> None:
    curves_dir = output_dir / "learning_curves"
    matrices_dir = output_dir / "known_confusion_matrices"
    curves_dir.mkdir()
    matrices_dir.mkdir()
    effort = {}
    for mode, result in results.items():
        if mode == "centralized":
            rows = [
                {
                    "epoch": record["epoch"],
                    "mean_loss": record["mean_loss"],
                    "selection_accuracy": record["selection_accuracy"],
                    "selection_macro_f1": record["selection_macro_f1"],
                    "examples_drawn": record["examples_drawn"],
                    "optimizer_steps": record["optimizer_steps"],
                    "runtime_seconds": record["runtime_seconds"],
                }
                for record in result.history
            ]
        else:
            rows = [
                {
                    "round": record["round"],
                    "mean_local_loss": record["mean_local_loss"],
                    "selection_accuracy": record["selection_accuracy"],
                    "selection_macro_f1": record["selection_macro_f1"],
                    "examples_drawn": record["examples_drawn"],
                    "optimizer_steps": record["optimizer_steps"],
                    "runtime_seconds": record["runtime_seconds"],
                }
                for record in result.history
            ]
        pd.DataFrame(rows).to_csv(curves_dir / f"{mode}.csv", index=False)
        for split_name, diagnostics in {
            "selection_after_calibration": result.post_calibration_selection_metrics,
            "calibration": result.calibration_metrics,
        }.items():
            pd.DataFrame(
                diagnostics["confusion_matrix"],
                index=diagnostics["true_rows"],
                columns=diagnostics["prediction_columns"],
            ).to_csv(matrices_dir / f"{mode}__{split_name}.csv", index_label="true_label")
        effort[mode] = _actual_effort(result, initial_state)
    write_json(output_dir / "actual_training_effort.json", effort)

    matched_rows = []
    for mode in ("iid_federated", "natural_non_iid_federated"):
        cumulative_signal = [
            {
                "target_examples_drawn": 0,
                "active_batches": 0,
                "cumulative_parameter_path_l2": 0.0,
            }
            for _ in class_names
        ]
        for record in results[mode].history:
            for class_index, class_name in enumerate(class_names):
                signal = cumulative_signal[class_index]
                signal["target_examples_drawn"] += int(
                    record["per_head_signal"][class_index]["target_examples_drawn"]
                )
                signal["active_batches"] += int(
                    record["per_head_signal"][class_index]["active_batches"]
                )
                signal["cumulative_parameter_path_l2"] += float(
                    record["projection_head_parameter_delta_l2"][class_index]
                )
                matched_rows.append(
                    {
                        "round": int(record["round"]),
                        "mode": mode,
                        "class_index": class_index,
                        "class_name": class_name,
                        **signal,
                    }
                )
    pd.DataFrame(matched_rows).to_csv(
        output_dir / "matched_federated_round_diagnostics.csv", index=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "contribution1_baseline.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "baseline")
    parser.add_argument("--smoke", action="store_true", help="run the bounded implementation smoke profile")
    parser.add_argument(
        "--training-seed",
        type=int,
        help="override only training.seed; fixed data and IID partition seeds remain unchanged",
    )
    return parser.parse_args()


def main() -> None:
    run_started = time.perf_counter()
    args = parse_args()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {output_dir}")
    output_dir.mkdir(parents=True)

    config = resolve_config(config_path, smoke=args.smoke)
    if args.training_seed is not None:
        config["training"]["seed"] = int(args.training_seed)
    training = config["training"]
    if training["device"] != "cpu":
        raise ValueError("this validated starting baseline currently requires explicit CPU execution")
    set_deterministic(int(training["seed"]), int(training["torch_num_threads"]))
    prepared = prepare_data(config)
    iid_clients, iid_audit = make_iid_control_clients(
        prepared.clients, int(config["iid_control"]["partition_seed"])
    )
    prepared.audit["iid_control"] = iid_audit
    train_features, train_labels = prepared.pooled_training()
    training_plan = _sampling_plan(prepared, iid_clients, config)
    write_json(output_dir / "training_plan.json", training_plan)

    model_config = {
        "input_dim": len(prepared.feature_names),
        "hidden_dims": config["model"]["hidden_dims"],
        "projection_dim": config["model"]["projection_dim"],
        "num_classes": len(prepared.class_names),
        "dropout": config["model"]["dropout"],
    }
    template = CLOSRModel(**model_config)
    initial_state = clone_state_dict(template)
    initial_state_sha256 = _state_sha256(initial_state)
    write_json(
        output_dir / "rng_audit.json",
        {
            "training_seed": int(training["seed"]),
            "training_seed_override": args.training_seed,
            "changed_rngs": [
                "Python random",
                "NumPy global RNG",
                "PyTorch CPU RNG",
                "PyTorch CUDA RNGs when CUDA exists (execution remains CPU)",
                "WeightedRandomSampler generators derived from the training seed",
                "dropout draws from the reproducibly reset PyTorch RNG",
            ],
            "fixed_rngs": {
                "checkpoint_and_calibration_group_split_seed": int(
                    config["validation_group_split"]["seed"]
                ),
                "iid_repartition_seed": int(config["iid_control"]["partition_seed"]),
            },
            "mode_start_seed": int(training["seed"]),
            "centralized_sampler_seed_formula": "training_seed + epoch",
            "federated_sampler_seed_formula": (
                "training_seed + round * 1000 + client_index * 100 + local_epoch"
            ),
            "identical_initial_state_for_all_three_configurations": True,
            "initial_state_sha256": initial_state_sha256,
        },
    )

    centralized_model = CLOSRModel(**model_config)
    centralized_model.load_state_dict(initial_state)
    set_deterministic(int(training["seed"]), int(training["torch_num_threads"]))
    centralized = fit_centralized(
        centralized_model,
        train_features,
        train_labels,
        prepared.selection_features,
        prepared.selection_labels,
        prepared.calibration_features,
        prepared.calibration_labels,
        config,
    )
    set_deterministic(int(training["seed"]), int(training["torch_num_threads"]))
    iid_federated = fit_federated(
        model_config,
        initial_state,
        iid_clients,
        prepared.selection_features,
        prepared.selection_labels,
        prepared.calibration_features,
        prepared.calibration_labels,
        config,
    )
    set_deterministic(int(training["seed"]), int(training["torch_num_threads"]))
    non_iid_federated = fit_federated(
        model_config,
        initial_state,
        prepared.clients,
        prepared.selection_features,
        prepared.selection_labels,
        prepared.calibration_features,
        prepared.calibration_labels,
        config,
    )
    results = {
        "centralized": centralized,
        "iid_federated": iid_federated,
        "natural_non_iid_federated": non_iid_federated,
    }
    _save_diagnostics(output_dir, results, prepared.class_names, initial_state)

    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    write_json(output_dir / "resolved_config.json", config)
    write_json(output_dir / "data_audit.json", prepared.audit)
    write_json(output_dir / "label_mapping.json", prepared.label_to_index)
    joblib.dump(prepared.preprocessor, output_dir / "training_only_preprocessor.joblib")

    common_metadata = {
        "experiment_id": config["experiment_id"],
        "execution_mode": config["execution_mode"],
        "config_source": str(config_path),
        "config_source_sha256": config_sha256,
        "final_test_loaded": False,
        "unknown_validation_evaluated": False,
        "purpose": "implementation validation only" if args.smoke else "baseline training without final evaluation",
        "sampling_policy_record": "training_plan.json",
        "actual_training_effort_record": "actual_training_effort.json",
        "rng_audit_record": "rng_audit.json",
        "initial_state_sha256": initial_state_sha256,
    }
    save_model_bundle(
        output_dir / "centralized_model.pt",
        centralized.model,
        centralized.centers,
        centralized.threshold,
        prepared.class_names,
        prepared.feature_names,
        {**common_metadata, "training_mode": "centralized", "best_epoch": centralized.best_step},
    )
    save_model_bundle(
        output_dir / "iid_federated_model.pt",
        iid_federated.model,
        iid_federated.centers,
        iid_federated.threshold,
        prepared.class_names,
        prepared.feature_names,
        {**common_metadata, "training_mode": "iid_sample_weighted_fedavg", "best_round": iid_federated.best_step},
    )
    save_model_bundle(
        output_dir / "natural_non_iid_federated_model.pt",
        non_iid_federated.model,
        non_iid_federated.centers,
        non_iid_federated.threshold,
        prepared.class_names,
        prepared.feature_names,
        {**common_metadata, "training_mode": "natural_non_iid_sample_weighted_fedavg", "best_round": non_iid_federated.best_step},
    )

    summary = {
        **common_metadata,
        "torch": torch.__version__,
        "device": "cpu",
        "known_classes": prepared.class_names,
        "globally_withheld_unknown_classes": config["globally_withheld_unknown_classes"],
        "centralized": {
            "best_epoch": centralized.best_step,
            "runtime_seconds": centralized.runtime_seconds,
            "initial_selection": centralized.initial_selection_metrics,
            "selection": centralized.selection_metrics,
            "selection_after_calibration": centralized.post_calibration_selection_metrics,
            "calibration": centralized.calibration_metrics,
            "threshold": centralized.threshold,
            "history": centralized.history,
        },
        "iid_federated": {
            "best_round": iid_federated.best_step,
            "runtime_seconds": iid_federated.runtime_seconds,
            "initial_selection": iid_federated.initial_selection_metrics,
            "selection": iid_federated.selection_metrics,
            "selection_after_calibration": iid_federated.post_calibration_selection_metrics,
            "calibration": iid_federated.calibration_metrics,
            "threshold": iid_federated.threshold,
            "history": iid_federated.history,
        },
        "natural_non_iid_federated": {
            "best_round": non_iid_federated.best_step,
            "runtime_seconds": non_iid_federated.runtime_seconds,
            "initial_selection": non_iid_federated.initial_selection_metrics,
            "selection": non_iid_federated.selection_metrics,
            "selection_after_calibration": non_iid_federated.post_calibration_selection_metrics,
            "calibration": non_iid_federated.calibration_metrics,
            "threshold": non_iid_federated.threshold,
            "history": non_iid_federated.history,
        },
        "total_runtime_seconds": float(time.perf_counter() - run_started),
        "interpretation": "Development-run known-only diagnostics are not final-test performance, causal evidence of a Non-IID effect, or novelty.",
    }
    write_json(output_dir / "run_summary.json", summary)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "execution_mode": config["execution_mode"],
                "final_test_loaded": False,
                "unknown_validation_evaluated": False,
                "total_runtime_seconds": summary["total_runtime_seconds"],
                "centralized": {
                    "best_epoch": centralized.best_step,
                    "initial_selection_macro_f1": centralized.initial_selection_metrics["macro_f1"],
                    "selected_selection_macro_f1": centralized.selection_metrics["macro_f1"],
                },
                "iid_federated": {
                    "best_round": iid_federated.best_step,
                    "initial_selection_macro_f1": iid_federated.initial_selection_metrics["macro_f1"],
                    "selected_selection_macro_f1": iid_federated.selection_metrics["macro_f1"],
                },
                "natural_non_iid_federated": {
                    "best_round": non_iid_federated.best_step,
                    "initial_selection_macro_f1": non_iid_federated.initial_selection_metrics["macro_f1"],
                    "selected_selection_macro_f1": non_iid_federated.selection_metrics["macro_f1"],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
