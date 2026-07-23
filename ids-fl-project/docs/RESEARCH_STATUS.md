# Research status and experiment record

- **Last updated:** 2026-07-23
- **Task:** Binary intrusion detection on DNN-EdgeIIoT
- **Current baseline:** Leakage-aware centralized and federated MLP experiment
- **Code:** [`../experiments/edgeiiot_federated/`](../experiments/edgeiiot_federated/)

## Where the work stands

The project has a working end-to-end centralized baseline and a five-client
Flower FedAvg simulation. A first experiment produced perfect test metrics, but
a follow-up leakage audit showed that those results should not be used as the
main scientific claim. The current baseline is the second, leakage-aware run,
which uses capture-group-disjoint splits and a reduced set of 25 numeric network
features.

## Dated record

### 2026-07-13 — first-generation pipeline

- Built reusable modules for loading, cleaning, preprocessing, IID client
  partitioning, PyTorch training/evaluation, Flower clients/server, inference,
  and notebook generation.
- Used a balanced 10% subset: 221,838 rows split into 155,286 train,
  33,276 validation, and 33,276 test rows.
- Used 72 processed features, five balanced IID clients, ten FedAvg rounds, and
  seed 42.
- Both centralized and federated models reported 1.0000 for accuracy, precision,
  recall, F1, ROC-AUC, and PR-AUC.
- This result is now **historical/diagnostic only**. Later review found
  source-format and duplicate-pattern leakage capable of making the task
  artificially easy.

Evidence is retained in `notebooks/01_binary_federated_pipeline.ipynb`,
`artifacts/`, and `results/`.

### 2026-07-14 — leakage-aware experiment

- Removed label proxies and capture-specific identity, timestamp, payload,
  checksum, sequence, port, transaction, and contaminated source-format fields.
- Normalized semantic values, removed conflicting-label feature patterns, and
  deduplicated identical final-feature patterns.
- Created capture groups from attack/capture blocks, time resets, five-minute
  windows, and bounded contiguous chunks.
- Enforced zero capture-group overlap and zero exact-feature overlap across
  train, validation, and test.
- Kept 25 numeric features and used a balanced 1,582-row modeling sample:
  1,108 train rows across five IID clients and 238 held-out test rows. The
  remaining rows were used for validation.
- Used a shared MLP, seed 42, two local epochs, ten FedAvg rounds, and
  sample-count-weighted aggregation. The best federated checkpoint was round 3.

Saved held-out test results:

| Metric | Centralized | Federated global |
|---|---:|---:|
| Accuracy | 0.970588 | 0.970588 |
| Precision | 0.991228 | 0.991228 |
| Recall | 0.949580 | 0.949580 |
| F1 | 0.969957 | 0.969957 |
| ROC-AUC | 0.991667 | 0.992373 |
| PR-AUC | 0.993486 | 0.994311 |
| False-positive rate | 0.008403 | 0.008403 |
| False-negative rate | 0.050420 | 0.050420 |
| TN / FP / FN / TP | 118 / 1 / 6 / 113 | 118 / 1 / 6 / 113 |

Evidence is retained in `notebooks/02_simple_clean_federated_pipeline.ipynb`,
`artifacts_clean/`, and `results_clean/`.

### 2026-07-23 — repository integration

- Migrated the experiment into the shared repository as a self-contained
  directory under `experiments/`.
- Preserved both generations of results and labeled their scientific status.
- Kept small metrics and feature/drop summaries; excluded raw data, cached
  arrays, fitted preprocessors, and model binaries.
- Made dataset discovery compatible with the repository-level `data/` folder
  while retaining the `EDGEIIOT_DATASET` override.

## Interpretation

The leakage-aware run shows that a centralized MLP and an IID FedAvg simulation
can reach the same classification threshold metrics on this sampled split. It
does **not** show that federation improves detection quality, generalizes to
unseen operational environments, or provides privacy. The small ROC-AUC and
PR-AUC differences should not be interpreted without repeated runs and
uncertainty estimates.

## Known limitations

- One random seed and one test split; there are no confidence intervals.
- Balanced sampling does not reflect deployment prevalence.
- Only 1,582 sampled unique rows remain after the clean representation and
  filtering, so the effective evaluation is small.
- IID, class-balanced clients do not model cross-site heterogeneity.
- Local sequential simulation does not measure communication, failures,
  network latency, secure aggregation, or privacy.
- Capture grouping uses attack/capture-block information. It is excluded from
  model features, but the grouping design must be disclosed and sensitivity
  tested.
- The clean pipeline is notebook-local; the modular package and tests currently
  cover the historical first-generation pipeline.
- Results are from a lab-generated dataset and lack external or temporal
  validation.

## Next paper-oriented milestones

1. Extract the leakage-aware pipeline into reusable modules and add tests for
   leakage exclusions, group disjointness, and reproducible splits.
2. Repeat at multiple seeds and report mean, standard deviation, and confidence
   intervals.
3. Evaluate the natural class distribution and report calibration, per-class
   error costs, FPR, FNR, ROC-AUC, and PR-AUC.
4. Add realistic non-IID partitions based on sites, devices, attack families, or
   time, and compare them with IID results.
5. Add classical centralized baselines and ablations for grouping,
   deduplication, feature exclusions, and sampling.
6. Use temporal or external validation and document training time,
   communication volume, and hardware.

## Reproduction pointers

- Environment pins: `experiments/edgeiiot_federated/requirements.txt`
- Current executable notebook:
  `experiments/edgeiiot_federated/notebooks/02_simple_clean_federated_pipeline.ipynb`
- Machine-readable current summary:
  `experiments/edgeiiot_federated/results_clean/run_summary.json`
- Exact retained features:
  `experiments/edgeiiot_federated/artifacts_clean/feature_list.json`
- Feature exclusions and reasons:
  `experiments/edgeiiot_federated/artifacts_clean/dropped_columns.csv`
