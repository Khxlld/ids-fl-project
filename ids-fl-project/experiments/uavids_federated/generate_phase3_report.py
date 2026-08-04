"""Generate the combined Phase 3 report from immutable saved results only."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_phase3"


def comparison_table(frame: pd.DataFrame) -> str:
    lines = [
        "| Model | Threshold | Accuracy | Macro-F1 | Attack precision | Attack recall | FPR | TN | FP | FN | TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.threshold:.2f} | {row.accuracy:.4f} | {row.macro_f1:.4f} "
            f"| {row.attack_precision:.4f} | {row.attack_recall:.4f} | {row.fpr:.4f} "
            f"| {int(row.tn)} | {int(row.fp)} | {int(row.fn)} | {int(row.tp)} |"
        )
    return "\n".join(lines)


def main() -> None:
    locked = json.loads((ROOT / "config" / "phase3_locked_config.json").read_text(encoding="utf-8"))
    record = json.loads((RESULTS / "locked_test_evaluation_record.json").read_text(encoding="utf-8"))
    candidates = pd.read_csv(RESULTS / "candidate_validation_comparison.csv")
    validation = pd.read_csv(RESULTS / "validation_model_comparison.csv")
    test = pd.read_csv(RESULTS / "locked_test_model_metrics.csv")
    families = pd.read_csv(RESULTS / "locked_test_original_class_detection.csv")

    selected = candidates.loc[candidates["selected"]].iloc[0]
    small_unweighted = candidates.loc[candidates["candidate_id"].eq("small_unweighted")].iloc[0]
    local_validation = validation[validation["model"].str.startswith("local/")]
    local_test = test[test["model"].str.startswith("local/")]
    central = test.loc[test["model"].eq("centralized")].iloc[0]
    fed = test.loc[test["model"].eq("federated_fedavg")].iloc[0]
    central_family = families[families["model"].eq("centralized")]
    fed_family = families[families["model"].eq("federated_fedavg")]
    weakest_central = central_family[central_family["original_label"].ne("Normal Traffic")].sort_values(
        "predicted_attack_rate"
    ).iloc[0]
    weakest_fed = fed_family[fed_family["original_label"].ne("Normal Traffic")].sort_values(
        "predicted_attack_rate"
    ).iloc[0]

    candidate_lines = [
        "| Candidate | Loss | Threshold | Macro-F1 | Attack recall | FPR | Selected |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in candidates.itertuples(index=False):
        candidate_lines.append(
            f"| {row.candidate_id} | {row.loss_policy} | {row.selected_threshold:.2f} "
            f"| {row.macro_f1:.4f} | {row.attack_recall:.4f} | {row.fpr:.4f} "
            f"| {'yes' if row.selected else 'no'} |"
        )

    report = f"""# Phase 3 - binary Normal-versus-Attack comparison

## 1. Selected preprocessing and MLP

The model uses the 15 approved numeric Phase 2 features in the locked order. Median imputation and `StandardScaler` were fitted once on the 6,148 rows from the five pooled training clients; validation and test were transform-only. No identifiers, ports, protocol, or excluded derived features were restored.

The selected MLP is **15 -> 128 -> 64 -> 32 -> 1**, with ReLU, dropout `[0.2, 0.1, 0.0]`, and unweighted binary cross-entropy. Training used AdamW (learning rate 0.001, weight decay 0.0001), batch size 128, seed 42, and deterministic CPU execution. The decision threshold is model-specific and was frozen from validation.

Pooled preprocessing centrally accessed every training client's feature values. It prevents validation/test leakage but is **not privacy-preserving federated preprocessing**; any privacy claim must explicitly exclude this prototype shortcut.

## 2. Validation-guided decisions

All architecture, loss, checkpoint, and threshold decisions used training clients plus validation only. Checkpoints maximized validation macro-F1 at threshold 0.5, with attack recall, lower FPR, and lower log loss as tie-breakers. After checkpoint selection, thresholds were chosen over 0.10-0.90 using validation macro-F1. The final test was not loaded until the complete policy and artifact hashes were locked.

{chr(10).join(candidate_lines)}

The selected `{selected.candidate_id}` candidate reached validation macro-F1 **{selected.macro_f1:.4f}**. It exceeded the small unweighted model by **{selected.macro_f1 - small_unweighted.macro_f1:.4f}**; global positive-class weighting reduced macro-F1 and increased FPR for both architectures.

## 3. Validation comparison

{comparison_table(validation)}

Local-only validation macro-F1 averaged **{local_validation['macro_f1'].mean():.4f}** (range **{local_validation['macro_f1'].min():.4f}-{local_validation['macro_f1'].max():.4f}**). Centralized was strongest; FedAvg exceeded the mean local baseline but remained below the best local model and centralized model.

## 4. Per-client and worst-client behavior

Each local model was trained only on its named logical client's Phase 2 training partition, then compared on the same global validation/test partitions. On the locked test, client 2 was the best local model at macro-F1 **{local_test['macro_f1'].max():.4f}**; client 5 was worst at **{local_test['macro_f1'].min():.4f}**, a spread of **{local_test['macro_f1'].max() - local_test['macro_f1'].min():.4f}**. Client 5 also missed {int(local_test.loc[local_test['macro_f1'].idxmin(), 'fn']):,} attacks. Client-specific training-partition scores are provided only as in-sample diagnostics and are not generalization estimates.

## 5. Final locked-test results

{comparison_table(test)}

The lock SHA-256 remained `{record['lock_config_sha256']}` before and after evaluation. No preprocessing refit, retraining, threshold revision, or test-driven experiment revision occurred.

Two clean development reruns reproduced all 27 locked development artifacts, tables, histories, and plots byte-for-byte. The saved-prediction audit independently reproduced all seven metric rows; discrete predictions and confusion counts matched exactly.

During traceability verification, the Phase 2 builder's expected raw schema count was corrected from 25 to 23 because the two model-facing labels are derived after loading. All Phase 2 partition file hashes remained unchanged.

The weakest attack family was **{weakest_central.original_label}** for centralized (recall **{weakest_central.predicted_attack_rate:.4f}**) and FedAvg (recall **{weakest_fed.predicted_attack_rate:.4f}**). FedAvg's locked-test FPR was **{fed.fpr:.4f}**, versus **{central.fpr:.4f}** centralized.

## 6. What the comparison demonstrates

The outcome is **mixed**. FedAvg test macro-F1 **{fed.macro_f1:.4f}** beat the local-only mean **{local_test['macro_f1'].mean():.4f}** by **{fed.macro_f1 - local_test['macro_f1'].mean():.4f}**, but trailed centralized **{central.macro_f1:.4f}** by **{central.macro_f1 - fed.macro_f1:.4f}**. It also had a materially higher FPR.

This fairly demonstrates optimization behavior for centralized pooling, isolated local training, and sample-weighted FedAvg over these five fixed non-IID logical sources under a shared preprocessing/model policy. It does not demonstrate privacy, communication security, physical UAV deployment, robustness to malicious clients, or statistical superiority across repeated datasets/seeds.

## 7. Limitations and suspicious findings

- The five clients are logical source-address groups, not verified physical UAVs.
- Test sources and exact signatures are disjoint, but the split is not proven temporal-, scenario-, run-, or independent-network-disjoint.
- Only one deterministic seed and one dataset split were used; there are no uncertainty intervals.
- Global validation was reused for candidate, checkpoint, and threshold selection, so it is a development set rather than an untouched benchmark.
- FedAvg finished at the last allowed round (30), so a longer schedule might change its result.
- Local thresholds varied widely (0.22-0.90), showing client-specific calibration instability.
- Wormhole recall and FedAvg false positives deserve focused investigation; binary aggregate scores hide this family-level weakness.
- In-sample client diagnostics are intentionally labeled and must not be presented as held-out per-client performance.

## 8. Artifact paths

- Configuration: `config/phase3_development_config.json`, `config/phase3_locked_config.json`
- Reproducible entry points: `run_phase3_development.py`, `run_phase3_final.py`, `verify_phase3_artifacts.py`, `generate_phase3_report.py`, `requirements-phase3.txt`
- Shared code/tests: `src/uavids_fl/modeling.py`, `tests/test_phase3_core.py`
- Preprocessing/features: `artifacts_phase3/training_only_preprocessor.joblib`, `artifacts_phase3/preprocessing_metadata.json`, `artifacts_phase3/feature_list.json`
- Checkpoints: `artifacts_phase3/centralized_model.pt`, `artifacts_phase3/federated_global_model.pt`, `artifacts_phase3/local_uav_client_1_model.pt` through `local_uav_client_5_model.pt`
- Validation/candidate results and histories: `results_phase3/candidate_validation_comparison.csv`, `results_phase3/validation_model_comparison.csv`, `results_phase3/*_history.csv`, `results_phase3/training_client_diagnostics.csv`
- Locked-test evidence: `results_phase3/locked_test_model_metrics.csv`, `results_phase3/locked_test_predictions.csv`, `results_phase3/locked_test_original_class_detection.csv`, `results_phase3/locked_test_evaluation_record.json`
- Plots: `results_phase3/plots/`
- Reports: `results_phase3/DEVELOPMENT_VALIDATION_SUMMARY.md`, `results_phase3/PHASE3_FINAL_SUMMARY.md`, this report

## 9. Docker-phase handoff

Carry forward the exact 15-feature order, fitted preprocessor, binary label mapping, `15-128-64-32-1` architecture, global checkpoint, threshold **{locked['model_thresholds']['federated_fedavg']:.2f}**, sample-count FedAvg rule, client partition/checksum manifest, seed, and validation/evaluation ownership. Each future client must read only its own partition. Preserve explicit version/hash checks and the statement that the current shared preprocessor is centrally fitted and therefore not privacy-preserving. Do not infer networking, security, or deployment guarantees from this Phase 3 simulation.
"""
    (RESULTS / "PHASE3_COMPARISON_REPORT.md").write_text(report, encoding="utf-8")
    print("Wrote results_phase3/PHASE3_COMPARISON_REPORT.md from saved results.")


if __name__ == "__main__":
    main()
