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
- `artifacts_phase3/` — feature metadata plus the hash-locked FedAvg checkpoint and fitted preprocessor required for deployment; other research checkpoints remain local.
- `results_phase3/PHASE3_COMPARISON_REPORT.md` — concise validation and locked-test conclusions.
- `results_phase3/*.csv`, `*.json`, and `plots/` — histories, metrics, predictions, hashes, and figures.

## Phase 3, Phase 4, and Phase 5 guide notebooks

- `notebooks/07_phase5_post_quantum_security_guide.ipynb` — executed threat-model, protected-exchange, attack-rejection, equivalence, and overhead guide.
- `build_phase5_notebook.py` and `execute_phase5_notebook.py` — deterministic source rebuild and clean-kernel execution.

- `notebooks/05_phase3_binary_modeling_guide.ipynb` — read-only walkthrough of the frozen preprocessing, MLP, validation decisions, three training paths, and locked-test results.
- `notebooks/06_phase4_docker_federated_demo_guide.ipynb` — walkthrough of the six-service protocol, client profiles, one round, safe aggregation, GUI events, and measured verification evidence.
- `build_guide_notebooks.py` — deterministically rebuilds both notebook sources.
- `execute_guide_notebooks.py` — executes both guides from clean kernels with the experiment root as their working directory.
- `requirements-guides.txt` — Phase 3 runtime plus notebook build/execution dependencies.

From this directory, rebuild and execute the guides with:

```powershell
python .\build_guide_notebooks.py
python .\execute_guide_notebooks.py
```

## Phase 5 files

- `phase5/app/` — ML-KEM/ML-DSA handshake, HKDF/AES-GCM sessions, secure server adapter, and client transport.
- `phase5/docker-compose.yml` and `phase5/Dockerfile` — pinned OQS six-service secure demonstration with runtime-mounted identities.
- `phase5/README.md` and `phase5/SECURITY_PROTOCOL.md` — commands, threat/trust model, exact protocol, and limitations.
- `phase5/tests/` and `phase5/results/` — positive/negative tests and secret-free equivalence, timing, size, resource, and rejection evidence.
- `notebooks/07_phase5_post_quantum_security_guide.ipynb` — executed educational guide based only on saved public evidence.

## GUI integration handoff

- `gui_integration/backend.py` — stable binary inference and federated-telemetry HTTP adapter for the presentation frontend.
- `gui_integration/GUI_MINIMUM_REQUIREMENTS.md` — minimum honest, presentation-ready information requirements.
- `gui_integration/api/openapi.json` — machine-readable frontend contract.
- `gui_integration/examples/` — six label-free replay inputs and actual adapter response/event examples.
- `gui_integration/frontend/vanilla_integration.js` — framework-neutral connection and polling example.
- `gui_integration/scripts/` — start and end-to-end verification commands.

## Phase 4 files

- `phase4/docker-compose.yml` and `phase4/Dockerfile` — six-service CPU-only demo topology and image.
- `phase4/app/` — HTTP control center, isolated client, compatibility contract, and safe update serialization.
- `phase4/scripts/` — build/run/observe/stop commands for Windows PowerShell.
- `phase4/README.md` — operator guide and troubleshooting.
- `phase4/DEVICE_PROFILES.md` — authoritative device evidence and measured Docker limits.
- `phase4/EVENT_CONTRACT.md` — GUI-facing status/event API.
- `phase4/results/PHASE4_RESULTS_SUMMARY.md` — measured demonstration results, separate from Phase 3.
