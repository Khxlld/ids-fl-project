# UAVIDS-2025 federated intrusion-detection experiment

This directory is the separate continuation of the existing Edge-IIoT work.
Phase 1 contains only a raw-dataset audit; it does not clean data, create final
partitions, or train a model.

## Phase 1 files

- `notebooks/04_uavids_dataset_audit.ipynb` — executed dataset audit.
- `results_audit/PHASE1_AUDIT_SUMMARY.md` — concise findings and Phase 2 gate.
- `results_audit/*.csv` and `audit_summary.json` — code-generated audit tables.
- `build_notebook_04.py` — reproducibly rebuilds the notebook source.

The raw dataset remains in the sibling `../UAVIDS-2025/` directory.

## Phase 2 files

- `prepare_phase2_partitions.py` — deterministic leakage-controlled partition builder.
- `config/phase2_partition_config.json` — locked sources, labels, feature policy, and split rules.
- `partitions/phase2/` — five model-facing client CSVs plus validation and final-test CSVs.
- `results_phase2/partition_manifest.json` — source assignments, distributions, checksums, and validation results.
- `results_phase2/row_assignments.csv` — raw-FlowID-level inclusion/exclusion traceability.
- `results_phase2/source_assignments.csv` — logical-source-to-partition mapping and counts.
- `results_phase2/PHASE2_PARTITION_SUMMARY.md` — concise design rationale and limitations.

## Phase 3 files

- `run_phase3_development.py` — reproducible train/validation development; it does not load the final test.
- `run_phase3_final.py` — one-way evaluation of frozen checkpoints and thresholds on the locked test.
- `verify_phase3_artifacts.py` — verifies hashes and recomputes metrics from saved predictions without reopening test data.
- `generate_phase3_report.py` — rebuilds the combined report from saved Phase 3 outputs.
- `requirements-phase3.txt` — exact packages used by the deterministic reference run.
- `config/phase3_locked_config.json` — frozen feature, preprocessing, model, optimization, threshold, and artifact policy.
- `artifacts_phase3/` — fitted training-only preprocessor, feature metadata, and model checkpoints.
- `results_phase3/PHASE3_COMPARISON_REPORT.md` — concise validation and locked-test conclusions.
- `results_phase3/*.csv`, `*.json`, and `plots/` — histories, metrics, predictions, hashes, and figures.
