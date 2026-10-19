# Current research status

Last handover update: 2026-10-19.

## Branch and working location

- Active branch: `review/federated-closr-contribution1`.
- Base revision and current committed HEAD: `8e48e981afc001b2d406e659f1409c38aa77e1b4` from `Kaust-Final-submission`.
- Git worktree and VS Code folder: `C:\Users\feras\Downloads\KAUST 4th stage\glasgow\Project\useCases\smartCity\ids-fl-project-federated-closr`.
- Experiment directory: `ids-fl-project\experiments\uavids_federated_closr`.
- The separate pre-existing checkout at `...\smartCity\ids-fl-project-crowdsensing-iot` remains untouched on `crowdsensing-scope-remediation` at `ee51469be590195b5e7d2b96fd3391377ee951db`, with its prior tracked and untracked changes preserved.
- The Contribution 1 peer-review branch has been published. Publisher PDFs remain local and excluded. Publication details are recorded below.

## Scope and fixed initial design

Work remains limited to Contribution 1. The corrected baseline compares centralized CLOSR, IID five-client sample-weighted FedAvg CLOSR, and natural Non-IID five-client sample-weighted FedAvg CLOSR. Every mode uses the same CLOSR architecture and loss formula, initialization, center-based inference, known-only selection/calibration data, and pooled training rows. Their effective weighted empirical objectives are not identical because balancing is global centrally but client-local in FL, and missing local classes receive no target draws. No standalone CLAD detector, extra classifier, cascade, or prior binary IDS component was added. Knowledge distillation, unknown-to-known adaptation, server-side generators, deployment, Docker, cryptography, and dashboard changes were not added.

The initial configurable experiment is defined in `ids-fl-project/experiments/uavids_federated_closr/EXPERIMENT_SPEC.md` and `config/contribution1_baseline.json`:

- Known classes: Normal Traffic, Blackhole Attack, Flooding Attack, and Sybil Attack.
- Globally withheld class: Wormhole Attack.
- This choice preserves Sybil as a globally known class that is locally absent on clients 4 and 5, while withholding a well-populated attack class. It is a baseline design choice, not a result or novelty claim.
- Wormhole rows are excluded before preprocessing fit, training, centers, checkpoint selection, and threshold calibration.
- Validation known rows are split by `SrcAddr` with seed `240517` into source-disjoint checkpoint-selection and threshold-calibration subsets. Before smoke capping, these contain 39 sources/18,146 rows and 40 sources/23,204 rows, respectively, with every known class on both sides. Eight validation sources containing only the withheld class do not enter either subset.
- The final test CSV is hash-checked but not parsed by the training/smoke runner and has not been evaluated. The separate integrity test has parsed its schema, approved features, and labels solely for missingness and cross-partition duplicate checks, as disclosed below.
- There is no valid held-out data for the five training clients, so no per-client or worst-client generalization claim is made.

## Primary sources verified

- The local primary and supporting PDFs remain in `research/references/`.
- The official CLOSR repository was verified at `79da4d1f2a6f4e5ecf92c4588195aaf163df40f8`, which is still its `main`/HEAD revision.
- Inspected upstream files: `losses/closr_loss.py`, `losses/clad_loss.py`, `model/model.py`, `util/distance.py`, `data/loaders.py`, `train_closr.py`, and `eval_closr.py`.
- The local baseline documents the paper/code difference: Equations 11-12 describe paper pair averages and a sum across heads, while the official code normalizes by active nonzero/hinge pairs and averages across heads. The configured baseline defaults to the official active-pair/all-head mean behavior.
- The pinned `losses/closr_loss.py` SHA-256 is `b07464f66dace6e7f32767f79ea1470e31a9ca4c62b06332d1d3fa0d47541035`. Numerical tests now compare loss values and embedding gradients with a literal fixture of that revision.
- The earlier whole-head guard was removed because it discarded valid negative-only signal for one target plus negatives and valid positive-only signal for all-target batches. Terms are now handled independently, matching the pinned default behavior. The reference path also preserves upstream target diagonals and float32 active-pair counts. Paper-style normalization remains an explicit non-default deviation.
- The score implements Equation 16, `s(x) = -sum_c softmax(a)_c * a_c^2`, and follows Equation 17: reject when `s(x) > threshold`. The contradictory surrounding prose is not followed.
- The Kodama et al. method uses supervised contrastive pretraining plus EVT/OpenMax-style scoring and was not substituted for CLOSR.
- Upstream attribution and a copy of the Apache 2.0 license are retained in `UPSTREAM_ATTRIBUTION.md` and `THIRD_PARTY_LICENSES/CLOSR-Apache-2.0.txt`.

## Environment prepared

Dedicated ignored environment:

`C:\Users\feras\Downloads\KAUST 4th stage\glasgow\Project\useCases\smartCity\ids-fl-project-federated-closr\ids-fl-project\experiments\uavids_federated_closr\.venv`

Verified versions:

- Windows 11 / PowerShell.
- Python 3.12.1. Python 3.11 and global `python`/`py` launchers were unavailable.
- PyTorch 2.5.1+cpu; CUDA build `None`; `torch.cuda.is_available()` is false; `nvidia-smi` is unavailable.
- NumPy 2.3.1, pandas 3.0.3, scikit-learn 1.9.0, joblib 1.5.3, pytest 9.1.1.

Setup commands actually used from the experiment directory:

```powershell
& 'C:\Users\feras\Downloads\KAUST 4th stage\glasgow\Project\useCases\military theme datasets\experiments\uavids_federated\.venv\Scripts\python.exe' -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --force-reinstall -r requirements.txt
& '.\.venv\Scripts\python.exe' -m pip install -e . --no-deps --no-build-isolation
```

The first dependency command exceeded its command timeout after partially unpacking wheels. No process remained; the force-reinstall command completed and repaired the environment. Imports and exact versions were then verified.

## Dataset and partition verification

- Raw dataset: `C:\Users\feras\Downloads\KAUST 4th stage\glasgow\Project\useCases\military theme datasets\experiments\UAVIDS-2025\UAVIDS-2025.csv`.
- Populated Phase 2 root: `C:\Users\feras\Downloads\KAUST 4th stage\glasgow\Project\useCases\military theme datasets\experiments\uavids_federated`.
- The raw dataset SHA-256 matches the manifest: `d50d339f68be7b23f0bf089dd438b20a1835c13182d8641220538121440164d0`.
- All 11 populated-manifest artifact records match, including the seven client/validation/test partition CSVs.
- The test suite reloaded all 113,883 retained partition rows and confirmed zero repeated approved-feature signatures across partitions.
- Schema is the 15 approved numeric features plus `original_label` and `binary_label`.
- Withheld Wormhole training rows removed by client: 338, 348, 120, 68, and 368 for clients 1-5.
- The 12,436 withheld Wormhole validation rows were not used by the smoke runner.

## Baseline implementation created

The new package is under `ids-fl-project/experiments/uavids_federated_closr/src/uavids_closr/` and includes:

- Shared MLP encoder and class-specific normalized linear projection heads.
- Class-specific CLAD/CLOSR loss with rescaled cosine distance, independent unavailable-term handling, finite zero loss for absent/singleton cases, and configurable official/paper normalization.
- Class-balanced deterministic sampling with `drop_last=False`.
- Centralized training with known-only grouped checkpoint selection. Every candidate is put in evaluation mode and gets centers computed only from known training rows with that exact candidate. Centers are recomputed after restoring the selected checkpoint.
- Explicit IID and natural Non-IID five-client local training with sample-count-weighted FedAvg using retained known-row counts.
- The IID control uses deterministic sequential stratified repartitioning (seed `240518`) of the identical 4,906 known pooled rows. Each row occurs exactly once and natural client sizes/aggregation weights are preserved: 892, 698, 471, 1,959, and 886. Full-data IID class counts by client are `[332,248,274,38]`, `[260,194,214,30]`, `[175,131,145,20]`, `[729,545,601,84]`, and `[330,246,272,38]` in configured class order.
- Federated class centers computed by aggregating per-class embedding sums/counts after distributing the fixed selected global model. Stale local-model centers are never averaged.
- Equation 16 scoring and Equation 17 strict `score > threshold` rejection. Known-only order-statistic calibration permits at most `floor(target*n)` known rejections; ties are not split and can lower the achieved rate. Reports separate overall/per-class known rejection, benign rejection, benign closed-set attack prediction, and benign total alert FPR.
- Model/center/threshold bundles, preprocessing, label mapping, configuration, and audit persistence/reload.

Preprocessing remains a centrally fitted median imputer plus `StandardScaler` over pooled known-only training rows. This is disclosed central access to training features, not a privacy-preserving preprocessing protocol. The FL code is a single-process simulation and not a privacy guarantee.

## Checks actually run

Targeted suite:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest
```

Outcome after adding aligned multi-seed effort diagnostics: `18 passed in 8.26s`. Checks cover manifest hashes/schema/feature overlap, unknown exclusion, grouped validation/calibration, training-only preprocessing, exact pinned-upstream loss values and gradients for ordinary/negative-only/positive-only/absent/singleton cases, unit embeddings, strict-threshold and tie behavior, separate calibration/final-metric benign alerts, candidate-model/center consistency (including the untrained diagnostic), validation exclusion from centers, IID row conservation/client sizes, fixed-global-model federated center aggregation, exact sample-weighted FedAvg, bundle reload, and same-period aggregation of per-head signal and parameter-path measurements.

Clean bounded smoke command:

```powershell
& '.\.venv\Scripts\python.exe' .\run_experiment.py --smoke --output-dir .\runs\smoke-review-bundle-final-20261018
```

Outcome: completed in `15.9s` with one centralized epoch and one round for each FedAvg control. All 20 centralized, 20 IID local, and 20 natural Non-IID local batches optimized with zero skipped batches. All three saved bundles were reloaded successfully; centers were finite/unit-normalized; `final_test_loaded=false`; `unknown_validation_evaluated=false`. The strict-rule known calibration rejection rate was `0.046875` (12/256) for all three modes against a 0.05 target. Per-class and benign rejection/alert diagnostics were emitted.

Smoke selection Macro-F1 was `0.6038` centralized, `0.3735` IID FedAvg, and `0.3731` natural Non-IID FedAvg on capped known-only fixtures. These are implementation diagnostics from one epoch/round and 128 capped rows per client, not baseline performance, evidence of a Non-IID failure, zero-day results, or novelty.

Authoritative smoke artifacts are in:

`ids-fl-project\experiments\uavids_federated_closr\runs\smoke-review-bundle-final-20261018`

The `.venv` and `runs` directories are ignored. Earlier smoke directories remain untouched as historical implementation checks.

The review bundle is at `ids-fl-project\experiments\uavids_federated_closr\review_bundle\contribution1-pre-full-audit-20261018` with a sibling ZIP of the same name. It contains copied source, configuration, tests, dependency versions, documentation, and test/smoke logs. It excludes datasets, `.venv`, credentials, `.joblib` preprocessing objects, and `.pt` model weights.

## Single-seed development run completed

The approved full known-only development run completed for seed `240517`:

```powershell
& '.\.venv\Scripts\python.exe' .\run_experiment.py --output-dir .\runs\contribution1-three-way-seed-240517
```

Output directory:

`ids-fl-project\experiments\uavids_federated_closr\runs\contribution1-three-way-seed-240517`

The run finished without stderr in `68.3892s`. `final_test_loaded=false` and `unknown_validation_evaluated=false`. All three selected bundles reload with finite, unit-normalized centers. Three learning-curve CSVs and six known-class confusion-matrix CSVs were validated.

Sampling and effort were recorded before and during training:

- Centralized balancing is inverse pooled-class-frequency sampling with replacement. It draws 4,906 examples and takes 39 optimizer steps per epoch. It ran 20 epochs before patience-8 early stopping: 98,120 examples and 780 steps.
- Each FL client balances with replacement only among classes locally present. Both FedAvg modes ran 30 rounds × one local epoch: 147,180 examples and 1,200 client optimizer steps each.
- IID expected aggregate draws are 1,226.5 per class/round. Natural Non-IID expected draws are 1,463.58 each for Normal/Blackhole/Flooding but 515.25 for Sybil because clients 4/5 lack Sybil.
- Realized class draws `[Normal, Blackhole, Flooding, Sybil]` were centralized `[24484,24658,24548,24430]`, IID `[36580,36789,36691,37120]`, and natural Non-IID `[43742,43745,44222,15471]`. Natural Sybil exposure was 10.51%, so the effective objectives are not identical despite the shared loss function.

Compact known-only development results:

| Mode | Initial selection Macro-F1 | Selected checkpoint | Selected selection Macro-F1 | Selection Macro-F1 after rejection | Calibration Macro-F1 after rejection | Calibration benign total alert FPR |
|---|---:|---:|---:|---:|---:|---:|
| Centralized | 0.4924 | epoch 12 | 0.7219 | 0.7201 | 0.7725 | 0.0519 |
| IID FedAvg | 0.4924 | round 10 | 0.7041 | 0.6939 | 0.7417 | 0.0651 |
| Natural Non-IID FedAvg | 0.4924 | round 1 | 0.6097 | 0.6093 | 0.6317 | 0.2081 |

All calibration known-rejection rates were `0.0499914` under the strict `score > threshold` rule. These are development diagnostics: selection participated in model choice, and calibration set the threshold. They are not final-test estimates.

Pilot learning/stopping observations:

- Centralized improved over its initialization and stopped at epoch 20 because the configured patience criterion was reached after selecting epoch 12. The patience event is a stopping rule, not proof of optimal convergence.
- IID FedAvg improved over initialization, selected round 10, and completed all 30 configured rounds. Its later trajectory varied while local loss fell; this does not establish convergence to an optimum.
- Natural Non-IID improved over initialization, selected round 1, and completed all 30 configured rounds. Its later known-selection trajectory and local loss diverged. This single-seed observation was not treated as proof that Non-IID data caused the gap.

Natural clients gave the Sybil head direct target signal in 510 of 1,200 client batches; the other heads had signal in all 1,200. Selected projection-head L2 changes from the common initialization were centralized `[1.9928,1.7888,1.7035,1.7250]`, IID `[1.3471,1.4541,1.2068,1.0931]`, and natural `[0.4025,0.3870,0.4003,0.1194]`. These are diagnostics, not normalized effect sizes.

Primary artifacts:

- `DEVELOPMENT_RUN_REPORT.md` and `compact_results.csv` — readable report/table.
- `training_plan.json` — sampling policies and configured effort.
- `actual_training_effort.json` — realized draws, steps, runtime, per-head signal, and parameter changes.
- `learning_curves\*.csv` — centralized/IID/natural learning curves.
- `known_confusion_matrices\*.csv` — selection-after-calibration and calibration matrices.
- `centralized_model.pt`, `iid_federated_model.pt`, `natural_non_iid_federated_model.pt` — selected checkpoints.
- `run_summary.json`, `data_audit.json`, and `resolved_config.json` — full metrics, histories, frozen split, and configuration.

## Fixed multi-seed development replication completed

Seed `240517` remains labeled as the pilot. Five additional full training seeds completed: `240519`, `240520`, `240521`, `240522`, and `240523`. No detector, loss, sampler, preprocessing, budget, or stopping-rule change was made during replication. Each run contains all three CLOSR configurations and keeps `final_test_loaded=false` and `unknown_validation_evaluated=false`.

The fixed protocol is recorded in `config/contribution1_multiseed_replication.json`. Only `training.seed` changes. Source partitions, Wormhole exclusion, checkpoint/calibration group split seed `240517`, and IID repartition seed `240518` remain fixed. Changing the training seed changes Python `random`, NumPy global RNG, PyTorch CPU/CUDA RNG state, dropout, and class-balanced sampler draws. Centralized loader seeds are `training_seed + epoch`; federated loader seeds are `training_seed + round*1000 + client_index*100 + local_epoch`. The RNG state is reset before each mode and every mode within a seed receives identical initial tensors; new runs record their initial-state SHA-256 in `rng_audit.json`.

The aggregate integrity audit passed for all six seeds: resolved configurations differ only at `training.seed`; `data_audit.json` is identical; `training_plan.json` is identical except for the training seed; and protected evaluation flags are false. Targeted tests remain `18 passed`.

Six-seed known-only development summary (mean ± sample SD):

| Mode | Selection closed Macro-F1 | Selection thresholded Macro-F1 | Calibration thresholded Macro-F1 | Selection benign rejection | Selection benign total alert FPR |
|---|---:|---:|---:|---:|---:|
| Centralized | 0.7558 ± 0.0231 | 0.7501 ± 0.0225 | 0.8051 ± 0.0245 | 0.0196 ± 0.0068 | 0.0544 ± 0.0194 |
| IID FedAvg | 0.7133 ± 0.0123 | 0.7044 ± 0.0139 | 0.7555 ± 0.0155 | 0.0284 ± 0.0148 | 0.0697 ± 0.0234 |
| Natural Non-IID FedAvg | 0.6276 ± 0.0607 | 0.6236 ± 0.0638 | 0.6613 ± 0.0772 | 0.0373 ± 0.0106 | 0.1492 ± 0.0693 |

Checkpoint-selection measurements participated in model selection. Calibration measurements come from the separate known-only group that set the threshold. Neither is an untouched final-test estimate. All 18 seed/mode thresholds achieved a known calibration rejection rate of `0.0499914`, with one score tied at the threshold under the strict `score > threshold` rule.

Paired IID-minus-natural selection closed-set Macro-F1 was positive for five of six seeds and averaged `+0.0857 ± 0.0530`; seed `240520` was effectively tied/slightly reversed at `-0.0018`. Selection benign total alert FPR was lower for IID in five of six seeds, with paired difference `-0.0795 ± 0.0642`, but reversed for seed `240522`. The gap is directionally common, not universal, and no causal claim is made.

Paired selection recall-after-rejection differences were `+0.0795 ± 0.0642` Normal (IID higher in 5/6), `+0.1667 ± 0.0734` Blackhole (6/6), `+0.1679 ± 0.1007` Flooding (5/6), and `-0.0069 ± 0.1826` Sybil (IID higher in 2/6). Missing local Sybil targets therefore do not explain all deterioration. Natural benign errors were also seed-variable: mean benign destinations were 0.0412 Flooding and 0.0633 Sybil, with Flooding prominent in seeds `240517`/`240519` and Sybil prominent in `240523`. No threshold change is proposed.

Effort diagnostics now separate training through the selected checkpoint from total completed effort. Centralized selected epochs 12–25 and stopped at epochs 20–33 under patience; this is not evidence of optimal convergence. Both federated modes completed 30 rounds and 1,200 client optimizer steps in every seed. IID selected rounds 6–26; natural selected round 1 in five seeds and round 18 for `240522`. Head movement and target signals are reported over identical periods and at matched federated rounds. At round 1, the natural Sybil head averaged about 525 direct target draws and 17 active client batches versus about 1,214 draws and 40 batches for IID; non-Sybil confusion and recall still varied, so this does not establish a cause.

Aggregate results and the readable report are in:

`ids-fl-project\experiments\uavids_federated_closr\runs\contribution1-multiseed-240517-240523`

Key files are `MULTISEED_DEVELOPMENT_REPORT.md`, `integrity_audit.json`, `individual_seed_results.csv`, `aggregate_mean_sd.csv`, both paired-difference CSVs, confusion-pattern CSVs, `selected_vs_total_effort.csv`, `per_head_period_diagnostics.csv`, and matched-round diagnostic CSVs. Per-seed runs remain in `runs\contribution1-three-way-seed-<seed>` with learning curves, selected checkpoints, known confusion matrices, configuration, RNG/data audits, and full histories.

## Exploratory round-1 replay completed

The requested post-hoc round-1 replay completed for seeds `240520` and `240523`, using IID and natural Non-IID FedAvg CLOSR only. Baseline behavior was not changed. Each case was first replayed through the ordinary trainer and then through RNG-neutral instrumentation after resetting the same seed. Client and aggregate tensors were bitwise identical, maximum tensor difference was zero, Python/NumPy/PyTorch RNG end states matched, and the prior full-run round-1 accuracy, Macro-F1, mean loss, class draws, and optimizer-step counts were reproduced exactly. The suite now reports `19 passed in 4.70s`. There are no reproduction discrepancies.

At round 0, IID and natural centers differed by at most `7.45e-09`, selection scores by at most `1.19e-07`, and closed predictions by 0 of 18,146 rows. The tiny differences are attributable to floating-point summation order across alternate client partitions; metrics were identical.

The selected-checkpoint contrast was not present at matched round 1:

| Seed | IID closed Macro-F1 | Natural closed Macro-F1 | Natural minus IID | IID thresholded Macro-F1 | Natural thresholded Macro-F1 |
|---:|---:|---:|---:|---:|---:|
| 240520 | 0.5337 | 0.7265 | +0.1927 | 0.5374 | 0.7242 |
| 240523 | 0.5932 | 0.5983 | +0.0051 | 0.5859 | 0.5972 |

For seed `240523`, similar round-1 Macro-F1 concealed different errors: natural classified 1,140/5,292 benign rows as Sybil versus 17 for IID, producing benign total alert FPR `0.2462` versus `0.1306`, while natural also had higher Sybil recall (`0.3186` versus `0.1747`). This is a closed-class tradeoff, not evidence for a threshold change.

Natural clients 4 and 5 had zero Sybil draws, zero valid positive/negative Sybil pairs, and zero Sybil-head loss gradients in all 23 of their combined batches. No gradient was missing (`grad is None`). AdamW still produced weight-decay-only Sybil-head path movement of approximately `5.6e-06` and `2.5e-06`, over four orders smaller than trained-client Sybil deltas. Their near-zero update vectors are excluded from cosine reporting with a declared `1e-5` norm rule.

Natural encoder cancellation was `0.1636`/`0.1534` for the two seeds versus IID `0.0789`/`0.0678`; natural encoder mean pairwise cosine was `0.5627`/`0.5747` versus IID `0.7810`/`0.8030`. Natural also had lower alignment and higher cancellation for each non-Sybil head. Natural Sybil aggregate update norm was `0.1163`/`0.1209` versus IID `0.3342`/`0.3724`. These properties occur even though natural is better or tied at round 1, so they do not by themselves explain the later selected-checkpoint gap.

Every observed aggregate encoder/head delta matched the sample-weighted sum of actual client deltas within float32 FedAvg cast tolerance; all 20 checks passed and the largest elementwise difference was `1.49e-08`. Fourteen near-zero Sybil pairwise cosine measurements involving natural client 4 or 5 were correctly marked undefined.

The authoritative report and tables are in:

`ids-fl-project\experiments\uavids_federated_closr\runs\contribution1-round1-exploratory-240520-240523-final`

`ROUND1_DIAGNOSTIC_REPORT.md` is the readable report. Exact artifacts include `reproduction_checks.json`, `round0_iid_natural_agreement.csv`, `per_client_class_signal.csv`, `loss_gradient_summary.csv`, `actual_client_parameter_deltas.pt`, `client_update_norms.csv`, `pairwise_update_cosines.csv`, `weighted_update_cancellation.csv`, `aggregation_verification.csv`, `round0_round1_metrics.csv`, and `round0_round1_confusion_long.csv`. The earlier same-named directory without the `-final` suffix is a preserved draft generated before the explicit near-zero cosine rule was tightened; use the `-final` directory as authoritative.

## Exploratory rounds 2-3 extension completed

The bounded diagnostic extension for seeds `240520` and `240523` completed through round 3 for IID and natural Non-IID FedAvg CLOSR. It replayed rounds 1-3 from the common initial tensors using both the ordinary and instrumented paths. For all four seed/mode cases, client states and global states were bitwise identical, saved non-runtime history fields matched exactly, and Python/NumPy/PyTorch RNG end states matched. The current suite reports `22 passed`. There are no reproduction discrepancies.

Only the fixed known checkpoint-selection subset and the declared known-only calibration group were scored. Unknown validation and final-test performance remained disabled. Thresholded results are stored separately and were not used to explain the closed-class confusion.

Closed-set selection Macro-F1 trajectories were:

| Seed | Mode | Round 0 | Round 1 | Round 2 | Round 3 |
|---:|---|---:|---:|---:|---:|
| 240520 | IID | 0.5255 | 0.5337 | 0.5589 | 0.6680 |
| 240520 | Natural | 0.5255 | 0.7265 | 0.5384 | 0.5443 |
| 240523 | IID | 0.5419 | 0.5932 | 0.5523 | 0.5729 |
| 240523 | Natural | 0.5419 | 0.5983 | 0.5328 | 0.5434 |

The natural-minus-IID contrast is positive at round 1 but negative at rounds 2 and 3 for both seeds. This is an early trajectory observation from two post-hoc examples, not a causal Non-IID result.

The prominent seed-240523 natural benign-to-Sybil count falls from 1,140 at round 1 to 19 at round 2 and 45 at round 3. Seed 240520 falls from 331 to 58 to 6. This reduction does not mean Sybil classification recovered: natural Sybil recall is zero at round 3 in both seeds, with 4,745/5,352 and 4,856/5,352 Sybil rows predicted as Flooding. IID round-3 Sybil recall is 0.2513 for seed 240520 and 0.0372 for seed 240523, and IID also has substantial Sybil-to-Flooding error. The known-class problem is therefore broader and more unstable than a persistent Normal-to-Sybil boundary error.

Normal true-class-minus-largest-wrong-class mean margins become strongly positive in all four trajectories. Sybil margins are negative at round 3 in all four trajectories. For true natural Sybil rows at round 3, mean `[Normal, Sybil, Flooding]` similarities are `[-0.5884, 0.4733, 0.7106]` for seed 240520 and `[-0.7362, 0.7085, 0.8372]` for seed 240523. Full means, dispersions, quantiles, record-level scores, round-0 changes, and preceding-round changes are saved; means alone are not treated as recall surrogates.

Across rounds 1-3, natural encoder cancellation rises from `0.1636` to `0.2983` to `0.4245` for seed 240520 and from `0.1534` to `0.2371` to `0.3892` for seed 240523, remaining above IID in these cases. Natural Sybil-head aggregate update norms also remain below IID. These are descriptive diagnostics only. No update-conflict correction, missing-class compensation, or threshold change is proposed.

All 60 encoder/head aggregation checks match the weighted sums of actual client deltas within float32 FedAvg tolerance; maximum elementwise difference is `1.49e-08`. Expected artifact row counts were validated, including 170,304 record-level Normal/Sybil score rows. `scope_and_integrity.json` records `baseline_method_modified=false`, `unknown_examples_evaluated=false`, and `final_test_performance_evaluated=false`.

The authoritative directory is:

`ids-fl-project\experiments\uavids_federated_closr\runs\contribution1-rounds0-3-exploratory-240520-240523`

`ROUNDS0_TO_3_DIAGNOSTIC_REPORT.md` is the readable report. Exact tables include reproduction, round metrics, separate closed/thresholded confusion matrices, Normal/Sybil recall/precision/prediction counts, every class-center similarity distribution, both requested margin distributions and their changes, record-level scores, client signals, gradients, update norms, cosines, cancellation, and weighted-aggregation verification.

## Bounded known-only baseline-tuning plan (not run)

The inspected baseline uses AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size 128, globally balanced replacement sampling centrally, client-local balanced replacement sampling in FedAvg, at most 40 centralized epochs with patience 8, and 30 federated rounds with one local epoch. Existing centralized selected checkpoints occur at epochs 12-25 and patience stops occur at epochs 20-33; these stopping events are not proof of optimal convergence.

The predeclared plan is in `config/contribution1_known_only_tuning_plan.json` and has status `planned_not_run`:

1. Begin with centralized CLOSR across the existing six development seeds. Compare the frozen baseline with an 80-epoch/patience-16 window at learning rates `1e-3`, `5e-4`, and `2.5e-4`. Keep the architecture, original CLOSR loss, weight decay, batch size, sampler, pooled preprocessing, centers, inference, split, and unknown exclusion fixed.
2. Rank configurations using only known-selection closed-set Macro-F1. Predeclared tie-breakers within 0.005 Macro-F1 use Normal/Sybil recall, benign-to-Sybil error, and selected-checkpoint effort. Calibration performance cannot select or search a configuration.
3. Freeze the centralized winner and apply its learning rate unchanged to both FedAvg controls. First confirm at the current 30-round budget. Only if the centralized winner is a longer-window candidate, run one predeclared 60-round confirmation for both FedAvg modes. Do not tune IID and natural separately.
4. If centralized and IID improve consistently, that supports a shared baseline optimization/budget weakness. If centralized improves but IID does not, that identifies an additional FedAvg optimization weakness. If centralized and IID improve but a paired natural-versus-IID gap persists across seeds, that identifies an additional partition-associated federated gap under the controls, not causation. If centralized does not improve, stop this tuning branch rather than automatically modifying CLOSR.

The known-calibration group remains reserved for the declared threshold procedure after checkpoint/configuration choice. No configuration is selected using calibration rejection or calibration Macro-F1.

## Final-test integrity access already performed

- The runner hashes the bytes of `partitions/phase2/test.csv` but does not parse its rows or use them for fitting, centers, selection, calibration, inference, or metrics.
- The runner parses all nine columns and all records in `results_phase2/row_assignments.csv`, including final-test traceability records; only validation records and the fields `FlowID`, `status`, `partition`, `original_label`, and `SrcAddr` affect preparation.
- The integrity test parses all seven partition CSVs, including the final test's 15 approved features, `original_label`, and `binary_label`, only for schema, missingness, row-count, and cross-partition duplicate-signature checks. No final-test prediction or performance evaluation has occurred.

## Client-level evaluation protocol

The fixed final test remains the primary source-disjoint evaluation. It cannot support per-training-client generalization because held-out sources are not mapped back to the five training clients. Training-client metrics are in-sample diagnostics only. A supplementary future protocol may treat sufficiently populated held-out final-test `SrcAddr` groups as logical client/source proxies and report support, per-source metrics, macro-source averages, and worst-source values. These must be labeled held-out source proxies, not physical-UAV or training-client results; temporal, session, scenario, and device separation remain unproven.

## Remaining limitations and decisions

- No unknown-validation or final-test performance evaluation has run. The six-seed study is development-only and cannot estimate zero-day effectiveness.
- Wormhole is the frozen initial unknown class for this baseline, but robustness across other frozen unknown-class choices remains a later experimental question.
- The full training budget is an initial CPU-feasible configuration, not paper-optimized hyperparameters. The paper's Lycos2017 results cannot be transferred to UAVIDS.
- The threshold controls known rejection on a known-only calibration split; it is not an unknown-tuned optimal threshold.
- The source-disjoint split does not establish scenario, session, temporal, or physical-UAV separation.
- Centralized preprocessing remains a disclosed privacy limitation.
- Satisfying client-local raw-data handling requires a separate preprocessing protocol in which clients retain row-level features and expose only declared aggregate scaling statistics, or use evaluated local normalization. No unrelated cryptography is required for this Contribution 1 correction.
- The IID control is approximately class-IID under exact natural-client size constraints; it is not mathematically identical finite-sample class proportions.
- No genuine blocker remains for the next bounded Contribution 1 diagnostic.

## Peer-review publication preparation

An isolated publication branch, `review/federated-closr-contribution1`, was created from the current research work at base revision `8e48e981afc001b2d406e659f1409c38aa77e1b4`. Existing files and uncommitted work were preserved; no reset, automatic stash, overwrite, or merge was performed.

`ids-fl-project/experiments/uavids_federated_closr/PEER_REVIEW_README.md` is the reviewer entry point. It explains the three CLOSR-only configurations, external-data-root setup, reproduction commands, implemented checks, known-only findings, pooled-preprocessing and evaluation limitations, and requested review questions. It explicitly states that the tuning plan is proposed rather than executed or proven effective.

`peer_review_evidence/` contains a deliberately selected compact evidence set: six-seed individual/aggregate metrics, all six seeds' learning curves and known confusion matrices, matched-effort summaries, round-1 and rounds-0-to-3 diagnostic summaries, integrity/reproduction checks, configuration/data hashes, dependency versions, and the publication test log. It excludes datasets, row-level traffic or score records, virtual environments, credentials, checkpoints, preprocessors, client-delta tensors, publisher PDFs, ZIP archives, and the stale duplicated pre-full review bundle. Existing broad CSV ignore rules were not changed; only selected evidence CSVs are intended to be force-added explicitly.

Publication verification on 2026-10-19 used Python 3.12.1 and PyTorch 2.5.1+cpu with CUDA unavailable. The command `& '.\.venv\Scripts\python.exe' -m pytest -q` completed successfully with `22 passed`. No smoke training, tuning, unknown-validation evaluation, or final-test performance evaluation was started for publication.

The main review content commit is `7e08b2559f0733cb377ebcef027c6840a9fb9c93` (`research: publish federated CLOSR contribution 1 review`). The previously unconfigured `origin` remote now points to `https://github.com/Khxlld/ids-fl-project`, and the isolated branch was pushed successfully without force and without merging:

`https://github.com/Khxlld/ids-fl-project/tree/review/federated-closr-contribution1`

The first elevated push attempt stopped locally before contacting GitHub because Git detected different Windows ownership for the sandbox-created worktree. The successful retry used a one-command `safe.directory` exception limited to this exact repository; no global Git configuration was changed. The branch tracks `origin/review/federated-closr-contribution1`.

## Exact next justified step

Do not launch another diagnostic series or open the final test yet. Execute stage 1 of `config/contribution1_known_only_tuning_plan.json`: the four-candidate centralized-only, six-seed known-selection screen. Preserve the original CLOSR loss/inference, frozen data, known-only selection, and declared calibration separation. Record learning curves, selected-checkpoint effort, closed confusion, per-class recall/precision, and Normal-to-Sybil error. Do not use calibration performance for candidate selection.

After the bounded centralized screen and any predeclared three-way confirmation, freeze the selected configuration, source revision, dependencies, seeds, partition/manifest hashes, selection rule, calibration rule, output schema, and one-shot final-evaluation command. Only after that protocol freeze may the source-disjoint final test be evaluated once for the declared known metrics and frozen Wormhole unknown metrics. The threshold must still be derived only from the known-calibration group, and final results must not trigger retuning. Knowledge distillation, adaptation, another detector, deployment, and unrelated privacy/cryptography changes remain out of scope.
