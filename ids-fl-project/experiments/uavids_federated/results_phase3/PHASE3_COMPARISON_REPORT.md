# Phase 3 - binary Normal-versus-Attack comparison

## 1. Selected preprocessing and MLP

The model uses the 15 approved numeric Phase 2 features in the locked order. Median imputation and `StandardScaler` were fitted once on the 6,148 rows from the five pooled training clients; validation and test were transform-only. No identifiers, ports, protocol, or excluded derived features were restored.

The selected MLP is **15 -> 128 -> 64 -> 32 -> 1**, with ReLU, dropout `[0.2, 0.1, 0.0]`, and unweighted binary cross-entropy. Training used AdamW (learning rate 0.001, weight decay 0.0001), batch size 128, seed 42, and deterministic CPU execution. The decision threshold is model-specific and was frozen from validation.

Pooled preprocessing centrally accessed every training client's feature values. It prevents validation/test leakage but is **not privacy-preserving federated preprocessing**; any privacy claim must explicitly exclude this prototype shortcut.

## 2. Validation-guided decisions

All architecture, loss, checkpoint, and threshold decisions used training clients plus validation only. Checkpoints maximized validation macro-F1 at threshold 0.5, with attack recall, lower FPR, and lower log loss as tie-breakers. After checkpoint selection, thresholds were chosen over 0.10-0.90 using validation macro-F1. The final test was not loaded until the complete policy and artifact hashes were locked.

| Candidate | Loss | Threshold | Macro-F1 | Attack recall | FPR | Selected |
|---|---|---:|---:|---:|---:|---|
| small_unweighted | unweighted | 0.58 | 0.9744 | 0.9858 | 0.0309 | no |
| small_balanced | global_training_pos_weight | 0.40 | 0.9709 | 0.9867 | 0.0444 | no |
| edge_style_unweighted | unweighted | 0.46 | 0.9775 | 0.9868 | 0.0246 | yes |
| edge_style_balanced | global_training_pos_weight | 0.30 | 0.9728 | 0.9868 | 0.0391 | no |

The selected `edge_style_unweighted` candidate reached validation macro-F1 **0.9775**. It exceeded the small unweighted model by **0.0031**; global positive-class weighting reduced macro-F1 and increased FPR for both architectures.

## 3. Validation comparison

| Model | Threshold | Accuracy | Macro-F1 | Attack precision | Attack recall | FPR | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| centralized | 0.46 | 0.9842 | 0.9775 | 0.9928 | 0.9868 | 0.0246 | 11831 | 299 | 551 | 41105 |
| federated_fedavg | 0.42 | 0.9628 | 0.9467 | 0.9757 | 0.9762 | 0.0833 | 11119 | 1011 | 990 | 40666 |
| local/uav-client-1 | 0.28 | 0.9499 | 0.9266 | 0.9592 | 0.9769 | 0.1429 | 10397 | 1733 | 964 | 40692 |
| local/uav-client-2 | 0.49 | 0.9601 | 0.9420 | 0.9679 | 0.9811 | 0.1117 | 10775 | 1355 | 789 | 40867 |
| local/uav-client-3 | 0.90 | 0.9286 | 0.8905 | 0.9313 | 0.9801 | 0.2485 | 9116 | 3014 | 829 | 40827 |
| local/uav-client-4 | 0.48 | 0.9430 | 0.9208 | 0.9752 | 0.9506 | 0.0830 | 11123 | 1007 | 2057 | 39599 |
| local/uav-client-5 | 0.22 | 0.8364 | 0.7926 | 0.9458 | 0.8368 | 0.1648 | 10131 | 1999 | 6798 | 34858 |

Local-only validation macro-F1 averaged **0.8945** (range **0.7926-0.9420**). Centralized was strongest; FedAvg exceeded the mean local baseline but remained below the best local model and centralized model.

## 4. Per-client and worst-client behavior

Each local model was trained only on its named logical client's Phase 2 training partition, then compared on the same global validation/test partitions. On the locked test, client 2 was the best local model at macro-F1 **0.9444**; client 5 was worst at **0.7872**, a spread of **0.1572**. Client 5 also missed 7,227 attacks. Client-specific training-partition scores are provided only as in-sample diagnostics and are not generalization estimates.

## 5. Final locked-test results

| Model | Threshold | Accuracy | Macro-F1 | Attack precision | Attack recall | FPR | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| centralized | 0.46 | 0.9841 | 0.9775 | 0.9923 | 0.9871 | 0.0260 | 11898 | 318 | 538 | 41195 |
| federated_fedavg | 0.42 | 0.9650 | 0.9502 | 0.9780 | 0.9767 | 0.0749 | 11301 | 915 | 971 | 40762 |
| local/uav-client-1 | 0.28 | 0.9519 | 0.9295 | 0.9595 | 0.9791 | 0.1410 | 10493 | 1723 | 874 | 40859 |
| local/uav-client-2 | 0.49 | 0.9618 | 0.9444 | 0.9680 | 0.9831 | 0.1111 | 10859 | 1357 | 704 | 41029 |
| local/uav-client-3 | 0.90 | 0.9322 | 0.8977 | 0.9372 | 0.9778 | 0.2237 | 9483 | 2733 | 927 | 40806 |
| local/uav-client-4 | 0.48 | 0.9462 | 0.9256 | 0.9786 | 0.9512 | 0.0711 | 11347 | 869 | 2036 | 39697 |
| local/uav-client-5 | 0.22 | 0.8301 | 0.7872 | 0.9468 | 0.8268 | 0.1586 | 10279 | 1937 | 7227 | 34506 |

The lock SHA-256 remained `5a4f00def41ee1ba5e827471179ffd0e1395f26587b0149439a70cd9168bbca5` before and after evaluation. No preprocessing refit, retraining, threshold revision, or test-driven experiment revision occurred.

Two clean development reruns reproduced all 27 locked development artifacts, tables, histories, and plots byte-for-byte. The saved-prediction audit independently reproduced all seven metric rows; discrete predictions and confusion counts matched exactly.

During traceability verification, the Phase 2 builder's expected raw schema count was corrected from 25 to 23 because the two model-facing labels are derived after loading. All Phase 2 partition file hashes remained unchanged.

The weakest attack family was **Wormhole Attack** for centralized (recall **0.9587**) and FedAvg (recall **0.9264**). FedAvg's locked-test FPR was **0.0749**, versus **0.0260** centralized.

## 6. What the comparison demonstrates

The outcome is **mixed**. FedAvg test macro-F1 **0.9502** beat the local-only mean **0.8969** by **0.0533**, but trailed centralized **0.9775** by **0.0273**. It also had a materially higher FPR.

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

Carry forward the exact 15-feature order, fitted preprocessor, binary label mapping, `15-128-64-32-1` architecture, global checkpoint, threshold **0.42**, sample-count FedAvg rule, client partition/checksum manifest, seed, and validation/evaluation ownership. Each future client must read only its own partition. Preserve explicit version/hash checks and the statement that the current shared preprocessor is centrally fitted and therefore not privacy-preserving. Do not infer networking, security, or deployment guarantees from this Phase 3 simulation.
