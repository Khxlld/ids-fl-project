"""Aggregate the fixed Contribution 1 development replication without loading test data."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SEEDS = [240517, 240519, 240520, 240521, 240522, 240523]
MODES = ["centralized", "iid_federated", "natural_non_iid_federated"]


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def _without_training_seed(value: dict) -> dict:
    normalized = copy.deepcopy(value)
    normalized["training"]["seed"] = "<replicated-training-seed>"
    return normalized


def _period(history: list[dict], count: int, num_classes: int) -> dict:
    records = history[:count]
    per_head = []
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
    for class_index in range(num_classes):
        per_head.append(
            {
                "class_index": class_index,
                **{
                    key: int(sum(record["per_head_signal"][class_index][key] for record in records))
                    for key in keys
                },
                "cumulative_parameter_path_l2": float(
                    sum(
                        record["projection_head_parameter_delta_l2"][class_index]
                        for record in records
                    )
                ),
            }
        )
    return {
        "completed_steps": count,
        "examples_drawn": int(sum(record["examples_drawn"] for record in records)),
        "optimizer_steps": int(sum(record["optimizer_steps"] for record in records)),
        "optimized_rows": int(sum(record["optimized_rows"] for record in records)),
        "recorded_step_runtime_seconds": float(sum(record["runtime_seconds"] for record in records)),
        "per_head_training_signal": per_head,
    }


def _mode_metrics(seed: int, role: str, mode: str, result: dict, class_names: list[str]) -> dict:
    selection = result["selection_after_calibration"]
    calibration = result["calibration"]
    row = {
        "seed": seed,
        "seed_role": role,
        "mode": mode,
        "selected_step": int(result.get("best_epoch", result.get("best_round"))),
        "total_completed_steps": len(result["history"]),
        "selection_closed_macro_f1": float(result["selection"]["macro_f1"]),
        "selection_thresholded_macro_f1": float(selection["macro_f1_after_rejection"]),
        "selection_known_rejection_rate": float(selection["known_rejection_rate"]),
        "selection_benign_rejection_rate": float(selection["benign_rejection_rate"]),
        "selection_benign_total_alert_fpr": float(selection["benign_total_alert_fpr"]),
        "calibration_thresholded_macro_f1": float(calibration["macro_f1_after_rejection"]),
        "calibration_known_rejection_rate": float(calibration["known_rejection_rate_overall"]),
        "calibration_benign_rejection_rate": float(calibration["benign_rejection_rate"]),
        "calibration_benign_total_alert_fpr": float(calibration["benign_total_alert_fpr"]),
    }
    for split_name, metrics in (("selection", selection), ("calibration", calibration)):
        for class_name in class_names:
            values = metrics["per_class"][class_name]
            prefix = f"{split_name}_{_slug(class_name)}"
            row[f"{prefix}_closed_set_recall"] = float(values["closed_set_recall"])
            row[f"{prefix}_recall_after_rejection"] = float(values["recall_after_rejection"])
            row[f"{prefix}_rejection_rate"] = float(
                values.get("rejection_rate", values.get("known_rejection_rate"))
            )
    return row


def _confusion_rows(seed: int, role: str, mode: str, split: str, metrics: dict) -> list[dict]:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    output = []
    for row_index, true_label in enumerate(metrics["true_rows"][:-1]):
        total = int(matrix[row_index].sum())
        for column_index, predicted_label in enumerate(metrics["prediction_columns"]):
            count = int(matrix[row_index, column_index])
            output.append(
                {
                    "seed": seed,
                    "seed_role": role,
                    "mode": mode,
                    "measurement_group": split,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": count,
                    "row_rate": float(count / total),
                    "true_row_total": total,
                }
            )
    return output


def _write_mean_std(frame: pd.DataFrame, group_columns: list[str], value_columns: list[str], path: Path) -> None:
    rows = []
    for group, values in frame.groupby(group_columns, sort=False):
        group_values = group if isinstance(group, tuple) else (group,)
        row = dict(zip(group_columns, group_values))
        for column in value_columns:
            row[f"{column}__mean"] = float(values[column].mean())
            row[f"{column}__sample_sd"] = float(values[column].std(ddof=1))
        row["n_seeds"] = int(len(values))
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "contribution1-multiseed-240517-240523",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite aggregate directory: {output_dir}")
    output_dir.mkdir(parents=True)

    summaries: dict[int, dict] = {}
    configs: dict[int, dict] = {}
    audits: dict[int, dict] = {}
    plans: dict[int, dict] = {}
    effort_files: dict[int, dict] = {}
    for seed in SEEDS:
        run_dir = ROOT / "runs" / f"contribution1-three-way-seed-{seed}"
        summaries[seed] = _read(run_dir / "run_summary.json")
        configs[seed] = _read(run_dir / "resolved_config.json")
        audits[seed] = _read(run_dir / "data_audit.json")
        plans[seed] = _read(run_dir / "training_plan.json")
        effort_files[seed] = _read(run_dir / "actual_training_effort.json")

    reference_config = _without_training_seed(configs[SEEDS[0]])
    reference_audit = audits[SEEDS[0]]
    reference_plan = copy.deepcopy(plans[SEEDS[0]])
    reference_plan["seed"] = "<replicated-training-seed>"
    integrity = {
        "expected_seeds": SEEDS,
        "checks": {},
        "final_test_performance_evaluated": False,
        "unknown_validation_performance_evaluated": False,
    }
    for seed in SEEDS:
        summary = summaries[seed]
        plan = copy.deepcopy(plans[seed])
        plan["seed"] = "<replicated-training-seed>"
        checks = {
            "resolved_training_seed_matches": int(configs[seed]["training"]["seed"]) == seed,
            "only_training_seed_changed_in_resolved_config": _without_training_seed(configs[seed])
            == reference_config,
            "data_audit_identical_to_pilot": audits[seed] == reference_audit,
            "sampling_plan_identical_except_training_seed": plan == reference_plan,
            "final_test_not_loaded": summary["final_test_loaded"] is False,
            "unknown_validation_not_evaluated": summary["unknown_validation_evaluated"] is False,
            "three_modes_share_initial_selection_metric": len(
                {
                    float(summary[mode]["initial_selection"]["macro_f1"])
                    for mode in MODES
                }
            )
            == 1,
        }
        if seed != SEEDS[0]:
            rng_audit = _read(
                ROOT / "runs" / f"contribution1-three-way-seed-{seed}" / "rng_audit.json"
            )
            checks["rng_audit_seed_matches"] = int(rng_audit["training_seed"]) == seed
            checks["identical_initial_state_declared_and_hashed"] = bool(
                rng_audit["identical_initial_state_for_all_three_configurations"]
                and rng_audit["initial_state_sha256"]
            )
        if not all(checks.values()):
            raise RuntimeError(f"integrity check failed for seed {seed}: {checks}")
        integrity["checks"][str(seed)] = checks
    (output_dir / "integrity_audit.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    class_names = summaries[SEEDS[0]]["known_classes"]
    metric_rows = []
    confusion_rows = []
    effort_rows = []
    head_rows = []
    matched_rows = []
    for seed in SEEDS:
        role = "pilot" if seed == 240517 else "replication"
        summary = summaries[seed]
        for mode in MODES:
            result = summary[mode]
            metric_rows.append(_mode_metrics(seed, role, mode, result, class_names))
            confusion_rows.extend(
                _confusion_rows(
                    seed, role, mode, "checkpoint_selection", result["selection_after_calibration"]
                )
            )
            confusion_rows.extend(
                _confusion_rows(seed, role, mode, "threshold_calibration", result["calibration"])
            )
            selected_step = int(result.get("best_epoch", result.get("best_round")))
            selected_period = _period(result["history"], selected_step, len(class_names))
            total_period = _period(result["history"], len(result["history"]), len(class_names))
            effort_rows.append(
                {
                    "seed": seed,
                    "seed_role": role,
                    "mode": mode,
                    "selected_step": selected_step,
                    "total_completed_steps": len(result["history"]),
                    "selected_examples_drawn": selected_period["examples_drawn"],
                    "total_examples_drawn": total_period["examples_drawn"],
                    "selected_optimizer_steps": selected_period["optimizer_steps"],
                    "total_optimizer_steps": total_period["optimizer_steps"],
                    "selected_recorded_step_runtime_seconds": selected_period[
                        "recorded_step_runtime_seconds"
                    ],
                    "total_recorded_step_runtime_seconds": total_period[
                        "recorded_step_runtime_seconds"
                    ],
                    "total_wall_runtime_seconds": float(result["runtime_seconds"]),
                    "stopping_description": (
                        "patience criterion reached"
                        if mode == "centralized" and len(result["history"]) < configs[seed]["training"]["centralized_epochs"]
                        else "configured budget completed"
                    ),
                }
            )
            old_effort = effort_files[seed][mode]
            if "selected_checkpoint_period" in old_effort:
                selected_net = old_effort["selected_checkpoint_period"][
                    "projection_head_net_delta_from_initial_l2"
                ]
            else:
                selected_net = old_effort["selected_projection_head_delta_from_initial_l2"]
            for period_name, period in (
                ("selected_checkpoint", selected_period), ("total_completed", total_period)
            ):
                for class_index, signal in enumerate(period["per_head_training_signal"]):
                    head_rows.append(
                        {
                            "seed": seed,
                            "seed_role": role,
                            "mode": mode,
                            "period": period_name,
                            "completed_steps": period["completed_steps"],
                            "class_name": class_names[class_index],
                            **signal,
                            "net_delta_from_initial_l2": (
                                float(selected_net[class_index])
                                if period_name == "selected_checkpoint"
                                else np.nan
                            ),
                        }
                    )
            if mode != "centralized":
                cumulative = [
                    {"target_examples_drawn": 0, "active_batches": 0, "parameter_path_l2": 0.0}
                    for _ in class_names
                ]
                for record in result["history"]:
                    for class_index, class_name in enumerate(class_names):
                        cumulative[class_index]["target_examples_drawn"] += int(
                            record["per_head_signal"][class_index]["target_examples_drawn"]
                        )
                        cumulative[class_index]["active_batches"] += int(
                            record["per_head_signal"][class_index]["active_batches"]
                        )
                        cumulative[class_index]["parameter_path_l2"] += float(
                            record["projection_head_parameter_delta_l2"][class_index]
                        )
                        matched_rows.append(
                            {
                                "seed": seed,
                                "seed_role": role,
                                "mode": mode,
                                "round": int(record["round"]),
                                "class_name": class_name,
                                **cumulative[class_index],
                            }
                        )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "individual_seed_results.csv", index=False)
    metric_columns = [
        column
        for column in metrics.columns
        if column not in {"seed", "seed_role", "mode", "selected_step", "total_completed_steps"}
    ]
    _write_mean_std(metrics, ["mode"], metric_columns, output_dir / "aggregate_mean_sd.csv")

    paired_columns = [
        column
        for column in metric_columns
        if "macro_f1" in column
        or "recall_after_rejection" in column
        or "rejection_rate" in column
        or "benign_total_alert_fpr" in column
    ]
    paired_rows = []
    for seed in SEEDS:
        indexed = metrics[metrics["seed"] == seed].set_index("mode")
        row = {"seed": seed, "seed_role": "pilot" if seed == 240517 else "replication"}
        for column in paired_columns:
            row[f"{column}__iid_minus_natural"] = float(
                indexed.loc["iid_federated", column]
                - indexed.loc["natural_non_iid_federated", column]
            )
        paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output_dir / "paired_iid_minus_natural_individual.csv", index=False)
    paired_value_columns = [column for column in paired.columns if column.endswith("__iid_minus_natural")]
    summary_row = {"n_seeds": len(SEEDS)}
    for column in paired_value_columns:
        summary_row[f"{column}__mean"] = float(paired[column].mean())
        summary_row[f"{column}__sample_sd"] = float(paired[column].std(ddof=1))
        summary_row[f"{column}__positive_seed_count"] = int((paired[column] > 0).sum())
    pd.DataFrame([summary_row]).to_csv(
        output_dir / "paired_iid_minus_natural_summary.csv", index=False
    )

    confusion = pd.DataFrame(confusion_rows)
    confusion.to_csv(output_dir / "known_confusion_long.csv", index=False)
    benign = confusion[confusion["true_label"] == configs[SEEDS[0]]["benign_class"]].copy()
    benign.to_csv(output_dir / "benign_confusion_patterns.csv", index=False)
    _write_mean_std(
        benign,
        ["mode", "measurement_group", "predicted_label"],
        ["row_rate"],
        output_dir / "benign_confusion_patterns_mean_sd.csv",
    )

    pd.DataFrame(effort_rows).to_csv(output_dir / "selected_vs_total_effort.csv", index=False)
    pd.DataFrame(head_rows).to_csv(output_dir / "per_head_period_diagnostics.csv", index=False)
    matched = pd.DataFrame(matched_rows)
    matched.to_csv(output_dir / "matched_round_federated_diagnostics.csv", index=False)
    matched_pivot = matched.pivot_table(
        index=["seed", "round", "class_name"],
        columns="mode",
        values=["target_examples_drawn", "active_batches", "parameter_path_l2"],
    )
    matched_pivot.columns = [f"{metric}__{mode}" for metric, mode in matched_pivot.columns]
    matched_pivot = matched_pivot.reset_index()
    for metric in ("target_examples_drawn", "active_batches", "parameter_path_l2"):
        matched_pivot[f"{metric}__iid_minus_natural"] = (
            matched_pivot[f"{metric}__iid_federated"]
            - matched_pivot[f"{metric}__natural_non_iid_federated"]
        )
    matched_pivot.to_csv(output_dir / "matched_round_iid_minus_natural.csv", index=False)
    _write_mean_std(
        matched_pivot,
        ["round", "class_name"],
        [
            "target_examples_drawn__iid_minus_natural",
            "active_batches__iid_minus_natural",
            "parameter_path_l2__iid_minus_natural",
        ],
        output_dir / "matched_round_iid_minus_natural_mean_sd.csv",
    )

    print(json.dumps({"output_dir": str(output_dir), "seeds": SEEDS, "integrity": "passed"}, indent=2))


if __name__ == "__main__":
    main()
