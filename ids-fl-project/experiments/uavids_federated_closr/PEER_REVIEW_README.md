# Contribution 1 peer-review guide

This branch contains the current development-only Contribution 1 baseline for federated CLOSR on the existing UAVIDS Phase 2 partition design. It is intended for implementation and experimental-protocol review. It does **not** contain a final zero-day result.

## What is compared

The experiment contains three CLOSR-only configurations:

1. **Centralized CLOSR** trains on the pooled known-only training rows.
2. **IID FedAvg CLOSR** deterministically repartitions those same rows across five clients while preserving the natural client sizes and sample-weighted aggregation weights.
3. **Natural Non-IID FedAvg CLOSR** keeps the five saved client partitions after globally withholding Wormhole. Natural clients 4 and 5 have no Sybil rows.

All three use the same shared MLP, class-specific normalized projection heads, original class-specific CLAD/CLOSR objective, candidate-specific training centers, Equation 16 score, Equation 17 strict `score > threshold` rule, known classes, initialization policy, and fixed selection/calibration partitions. CLAD is the mathematical per-head objective inside CLOSR; there is no separate CLAD detector, supervised classifier, or cascade.

The loss defaults to the pinned official CLOSR behavior at upstream revision `79da4d1f2a6f4e5ecf92c4588195aaf163df40f8`: active-pair normalization and a mean across heads. Tests compare values and embedding gradients with a literal fixture of that revision, including ordinary, negative-only, positive-only, absent-class, all-target, and singleton cases. Intentional paper-style normalization remains a non-default option and is documented in `EXPERIMENT_SPEC.md` and `UPSTREAM_ATTRIBUTION.md`.

## Repository contents

- `src/uavids_closr/`: model, loss, data contract, centralized/FedAvg training, centers, inference, and artifact handling.
- `run_experiment.py`: the three-way known-only development runner.
- `aggregate_multiseed.py`: six-seed aggregation and matched-effort summaries.
- `run_round1_diagnostic.py` and `run_rounds23_diagnostic.py`: RNG-neutral exploratory instrumentation.
- `config/`: frozen baseline, six-seed replication protocol, and the proposed-but-unexecuted tuning plan.
- `tests/`: numerical, data-integrity, center/checkpoint, aggregation, inference, threshold, and instrumentation checks.
- `peer_review_evidence/`: deliberately selected aggregate metrics, learning curves, confusion matrices, diagnostic summaries, hashes, environment record, and test log.
- `EXPERIMENT_SPEC.md`: detailed method and evaluation contract.
- `research/PROJECT_CONTEXT.md` and `research/STATUS.md` at the repository root: research handover and current state.

Datasets, row-level traffic records, virtual environments, credentials, model/preprocessor weights, publisher PDFs, raw record-level diagnostic scores, and client-delta tensors are intentionally excluded.

## External data contract

The code expects an external populated Phase 2 root containing:

- `results_phase2/partition_manifest.json`;
- `results_phase2/row_assignments.csv`; and
- every artifact referenced by the manifest, including five train partitions, validation, and final test.

The committed baseline JSON retains the path used for the recorded run for provenance. Reviewers should create an uncommitted local configuration with their own external path. From PowerShell in this directory:

```powershell
$env:UAVIDS_PHASE2_ROOT = 'D:\path\to\external\uavids_federated'
$localConfig = Join-Path $env:TEMP 'contribution1_baseline.local.json'
$config = Get-Content -Raw .\config\contribution1_baseline.json | ConvertFrom-Json
$config.data_root = $env:UAVIDS_PHASE2_ROOT
$json = $config | ConvertTo-Json -Depth 100
[System.IO.File]::WriteAllText($localConfig, $json, [System.Text.UTF8Encoding]::new($false))
```

The runner verifies all manifest artifact hashes before training. It reads final-test file bytes only for the manifest SHA-256 check; it does not parse or evaluate final-test rows. The targeted integrity test separately parses final-test schema/features/labels only for missingness and cross-partition duplicate-signature checks. This access is integrity auditing, not model selection or performance evaluation.

## Environment setup

The recorded environment was Python 3.12.1 and CPU-only PyTorch 2.5.1. Python 3.11 or newer is declared.

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
& '.\.venv\Scripts\python.exe' -m pip install -e . --no-deps --no-build-isolation
```

Run the checks against the configured external Phase 2 root:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest
```

A bounded reproduction smoke run, which still does not evaluate unknown validation or final-test performance, is:

```powershell
& '.\.venv\Scripts\python.exe' .\run_experiment.py `
  --config $localConfig `
  --smoke `
  --output-dir (Join-Path $env:TEMP 'closr-review-smoke')
```

For a declared development seed, omit `--smoke` and add `--training-seed <seed>`. The six development seeds are `240517`, `240519`, `240520`, `240521`, `240522`, and `240523`. Use a new output directory because the runner refuses to overwrite existing artifacts. Do not run `config/contribution1_known_only_tuning_plan.json` as though it were a completed or validated configuration.

## Implemented correctness controls

- Wormhole is removed before preprocessing, model training, center calculation, checkpoint selection, and threshold calibration.
- Validation is split by `SrcAddr` into disjoint known-selection and known-calibration groups.
- Every checkpoint candidate is placed in evaluation mode and receives centers computed only from known training rows using that exact candidate model. Final centers are recomputed after restoring the selected checkpoint.
- Validation rows never contribute to centers.
- Both FedAvg controls use all clients, sample-count weights, fresh local AdamW state per client/round, and fixed-global-model federated center aggregation.
- IID repartitioning preserves every pooled known training row exactly once and preserves natural client sizes.
- Calibration is known-only and follows the actual strict-greater rejection rule. Unknown examples never tune the threshold.
- Instrumentation was verified not to change tensors, recorded histories, or RNG consumption.

## Sampling and comparison caveat

The loss formula is shared, but the effective empirical objectives are not identical. Centralized balancing uses inverse pooled-class-frequency replacement sampling. Each client balances with replacement only among classes present locally. Natural clients without Sybil cannot draw Sybil targets. Centralized optimizer state persists across epochs; each FL client starts fresh AdamW state every round. Centralized epochs and federated rounds therefore are not optimization-equivalent budgets. Realized draws, optimizer steps, target signals, and selected-versus-total effort are included in the evidence bundle.

## Known-only development findings

Across six development seeds, mean ± sample SD selection closed-set Macro-F1 was:

| Configuration | Macro-F1 |
|---|---:|
| Centralized | 0.7558 ± 0.0231 |
| IID FedAvg | 0.7133 ± 0.0123 |
| Natural Non-IID FedAvg | 0.6276 ± 0.0607 |

Paired IID-minus-natural Macro-F1 averaged `+0.0857 ± 0.0530` and was positive in five of six seeds. This development association does not prove that Non-IID data caused the gap.

The two-seed rounds-0-to-3 replay found that a prominent natural benign-to-Sybil spike at round 1 was transient. By round 3, natural Sybil recall was zero in both examples and most true Sybil rows were predicted as Flooding. IID also showed substantial Sybil-to-Flooding confusion. The evidence therefore supports a broader and unstable known-class weakness, not a threshold remedy or a single established cause.

These metrics come from checkpoint-selection or known-calibration data. Selection participated in checkpoint choice, and calibration set the declared threshold. Neither is an untouched final-test estimate.

## Limitations and prohibited interpretation

- Pooled known-only median imputation and standardization are a disclosed baseline limitation: the server has centralized access to training features during preprocessing. This is not a client-local raw-data or privacy-preserving preprocessing protocol.
- The implementation is a single-process FL simulation, not a privacy guarantee.
- `SrcAddr` defines logical source groups; it does not prove physical-UAV, session, scenario, or temporal separation.
- There is no valid held-out test set mapped back to the five training clients. Training-client summaries are in-sample diagnostics, not client-level generalization results.
- Unknown-validation and final-test performance remain unevaluated. No zero-day effectiveness or novelty claim follows from this branch.
- The tuning plan is only a proposal. It has not been executed and is not proven effective.

## Requested reviewer focus

Please examine:

1. fidelity of the loss, architecture, score, and strict rejection rule to the pinned upstream implementation and cited equations;
2. label ordering, feature mapping, manifest/schema enforcement, and global unknown exclusion;
3. source separation and the distinction between integrity access, checkpoint selection, calibration, and final evaluation;
4. Sybil/Flooding confusion and whether the reported interpretation is supported by the saved distributions;
5. centralized versus client-local replacement sampling, missing local classes, effective class weighting, and optimizer-state differences;
6. candidate-specific center computation, evaluation mode, final-center recomputation, and exclusion of validation from centers;
7. Macro-F1, per-class recall/rejection, benign rejection, benign total alert FPR, quantile/tie handling, paired-seed summaries, and training-effort calculations; and
8. whether the centralized-first tuning proposal cleanly distinguishes a shared baseline weakness from an additional federated weakness without using calibration performance for configuration search.

The intended next development step, only after review, is the centralized-only four-candidate known-selection screen declared in `config/contribution1_known_only_tuning_plan.json`. Final-test evaluation remains blocked until the development protocol is accepted and frozen.
