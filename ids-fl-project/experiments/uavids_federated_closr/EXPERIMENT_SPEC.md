# Contribution 1 CLOSR baseline specification

This experiment compares three training arrangements for the same CLOSR method:

1. centralized CLOSR;
2. IID sample-weighted FedAvg CLOSR; and
3. natural Non-IID sample-weighted FedAvg CLOSR.

All three use the same shared MLP, class-specific normalized projection heads, class-specific CLAD/CLOSR objective, center-based CLOSR inference score, fresh initial tensors, known-only selection/calibration partitions, and training pool. CLAD is the mathematical class-specific objective inside CLOSR, not a standalone detector. This experiment does **not** create or run a separate CLAD binary detector, supervised classifier, hybrid/cascade, or the previous binary IDS. The prior binary pipeline remains untouched and outside this experiment.

Distillation, unknown-to-known adaptation, deployment, server generators, Docker, cryptography, and dashboard work are outside Contribution 1.

## Frozen task and data separation

- Known classes, in index order: Normal Traffic, Blackhole Attack, Flooding Attack, and Sybil Attack.
- Globally withheld class: Wormhole Attack.
- Retaining Sybil as known preserves local absence on natural clients 4 and 5. Withholding Wormhole supplies a populated global-unknown class. This is a configurable baseline choice, not a novelty or optimality claim.
- The five Phase 2 training-source partitions are logical clients. `SrcAddr` is a grouping key, not proof of a physical UAV.
- Every Wormhole row is excluded before preprocessing fit, model fitting, centers, checkpoint selection, and threshold calibration.
- Existing binary weights, classifiers, and preprocessors are never loaded.

Known validation rows are divided by `SrcAddr` with deterministic `GroupShuffleSplit` into checkpoint-selection and threshold-calibration subsets. Sources cannot cross those subsets, and every known class must occur on both sides. Evaluation partitions are fixed and identical across all three CLOSR modes.

## Loss fidelity audit

The reference is official `jackwilkie/CLOSR` revision `79da4d1f2a6f4e5ecf92c4588195aaf163df40f8`, specifically `losses/closr_loss.py` (SHA-256 `b07464f66dace6e7f32767f79ea1470e31a9ca4c62b06332d1d3fa0d47541035`). The default local loss numerically matches its behavior:

- rescaled cosine distance `(1 - cosine_similarity) / 2`;
- squared positive distances and squared active margin violations;
- active-pair denominators with epsilon `1e-6`;
- `alpha=0.5`, margin `1.0`; and
- mean over every class head.

The earlier whole-head guard was incorrect for the reference baseline because it discarded a negative-only term when a class had one target plus negatives, and a positive-only term in an all-target batch. Terms are now handled independently. An unavailable term is a differentiable zero; an available counterpart remains active. An absent class and a singleton batch with no usable pair signal remain finite differentiable zeros. Tests compare loss values and embedding gradients with a literal pinned-revision numerical fixture for ordinary, one-target-plus-negatives, all-target, absent-class, and singleton cases.

Intentional alternatives are not the default: `paper_all_pairs` excludes self-pairs and divides by all defined pairs, and non-default head reductions are retained only for controlled analysis. The upstream source includes target diagonals and converts active counts to float32; the reference path preserves both details for numerical parity.

Class-balanced sampling and `drop_last=False` remain deliberate dataset adaptations. Upstream's large batch/default dropping behavior would produce no full batch for these clients. Architecture width and training budget are CPU-feasible UAVIDS choices, not a numerical reproduction of Lycos2017.

## Checkpoint selection and centers

Checkpoint selection uses known closed-set Macro-F1 from CLOSR center similarities; rejection is calibrated only after selection. For every epoch/round candidate:

1. the exact candidate model is put in evaluation mode;
2. centers are recomputed solely from known training rows embedded by that candidate;
3. selection rows are scored with those candidate-specific centers; and
4. neither selection nor calibration rows contribute to a center.

After choosing the checkpoint, centers are recomputed again from known training rows with the selected model. Centralized CLOSR uses the pooled known training rows. Each federated mode distributes the fixed selected global model, computes per-class embedding sums/counts locally, aggregates those sufficient statistics, then normalizes. It never averages stale centers from local models.

## IID and natural Non-IID FedAvg controls

The natural control keeps the saved five client partitions after unknown removal. The IID control deterministically pools those same transformed known rows and repartitions them by repeated stratified sampling with seed `240518`. Every row appears exactly once, no row is synthesized, and the five natural client row counts are preserved exactly. Thus both federated controls have the same sample weights and number of local batches, while IID clients approximate the global class distribution and natural clients retain their observed class skew/local absence.

All three modes load the same fresh model state. RNG state is reset to the configured seed before each mode. Both federated controls use every client every round, one configurable local epoch, fresh AdamW optimizer state for each client at the start of each round, and sample-count aggregation weights based on known rows after unknown exclusion. Optimizer state persists across centralized epochs but is not federated. Balanced replacement sampling is used in every local loader; realized draws and dropout paths cannot be perfectly identical once client class composition/order differs. Centralized optimization also cannot have the same optimizer-step trajectory as FedAvg, so the comparison controls data, architecture, objective, initialization, nominal epochs/rounds, and evaluation—not optimization equivalence.

More specifically, centralized balancing is computed once over the pooled training labels, whereas each FL client computes inverse-frequency weights from its own labels. A client samples `len(client)` examples with replacement per local epoch and balances only among locally present classes. Consequently, a globally known but locally absent class has zero direct target draws on that client. The CLOSR loss formula is identical across modes, but their effective class weighting and optimized empirical objectives are not. Each run saves expected policies in `training_plan.json`, realized class draws and optimizer steps in `actual_training_effort.json`, and per-client/per-head signal in `run_summary.json`.

## Inference and known-only calibration

For similarities `a_c`, CLOSR uses:

`s(x) = -sum_c softmax(a)_c * a_c^2`

Higher scores are more unknown-like. Following Equation 17, a row is rejected only when `s(x) > threshold`.

The threshold is an observed order statistic from known-only calibration data. It allows at most `floor(target_rate * n)` rejected known rows under the actual strict-greater rule. Scores tied at the threshold remain accepted, so an exact target can be unattainable and the observed rate may be lower. Unknown rows never tune the threshold.

Calibration reports overall known rejection, per-class known rejection, benign rejection, benign closed-set attack prediction, and benign total alert FPR separately. Benign total alert means either UNKNOWN rejection or an attack-class output on a benign row.

## Evaluation protocol and claims

The primary future evaluation remains the fixed source-disjoint Phase 2 final test and will be run once only after the development run is accepted. Overall metrics are known-class Macro-F1 counting known rejection as error, unknown recall, overall/per-class known rejection, benign rejection, benign total alert FPR, and known-versus-unknown AUROC.

The current data do not map held-out sources back to the five training logical clients. Therefore no per-training-client or worst-training-client generalization result is defensible. Training-partition scores, if inspected, are labeled in-sample diagnostics. A supplementary client-level protocol is defined as follows: treat each sufficiently populated held-out final-test `SrcAddr` as an evaluation unit, report its metrics plus macro/worst-source summaries with support counts, and call these *held-out source/client proxies*, not results for training clients or physical UAVs. This preserves source disjointness but does not establish temporal, session, scenario, device, or physical-UAV generalization. No final-test records are evaluated during this audit.

## Pooled preprocessing limitation and required successor

Median imputation and `StandardScaler` are fitted once on pooled known-only training features so all three baseline modes use identical inputs. This centrally exposes raw training feature statistics and violates the intended client-local raw-data requirement; the single-process simulation is not a privacy guarantee.

To satisfy client-local raw-data handling in a successor baseline, each client must fit/retain local imputation state and contribute only explicitly defined aggregate sufficient statistics for a common scaling transform (counts, sums/means, and variance terms), or use an agreed client-local normalization whose model implications are evaluated. The server must never receive row-level raw features, and train/selection/calibration boundaries must remain unchanged. This is preprocessing protocol work; it does not require encryption, secure aggregation, differential privacy, or other unrelated cryptography in Contribution 1.

## Final-test integrity access disclosure

Integrity auditing is distinct from selection and performance evaluation:

- The training/smoke runner reads bytes of `partitions/phase2/test.csv` only to verify its manifest SHA-256; it does not parse that CSV or pass any test row to preprocessing, centers, selection, calibration, or inference.
- The runner parses all nine columns of `results_phase2/row_assignments.csv` for all records, including final-test traceability rows. Only validation records' `FlowID`, `status`, `partition`, `original_label`, and `SrcAddr` affect validation alignment/splitting.
- The targeted integrity test parses all seven partition CSVs, including the final test's 15 approved features plus `original_label` and `binary_label`, to verify schema, missingness, row counts, and cross-partition duplicate-feature signatures. It does not generate predictions or metrics.

The source-disjoint controls still do not prove scenario, session, temporal, or physical-UAV separation.

## Fixed multi-seed development replication

Seed `240517` is the pilot. The fixed additional training seeds are `240519`, `240520`, `240521`, `240522`, and `240523`, as registered in `config/contribution1_multiseed_replication.json`. Only `training.seed` changes. The source partitions, unknown exclusion, checkpoint-selection/calibration group split seed (`240517`), IID repartition seed (`240518`), CLOSR implementation, sampling, preprocessing, budgets, and stopping rules remain fixed.

Changing the training seed changes Python `random`, the NumPy global RNG, PyTorch CPU/CUDA RNG state, dropout, and the class-balanced sampler generators. The centralized sampler uses `training_seed + epoch`; federated samplers use `training_seed + server_round * 1000 + client_index * 100 + local_epoch`. Before every configuration the global RNG state is reset to the same within-seed value, and all three configurations load the same hashed initial model tensors. Natural and IID sampling streams remain reproducibly controlled but cannot be draw-identical because their client label compositions differ.

Checkpoint selection remains known-only closed-set Macro-F1. Calibration remains a separate known-only group and uses no unknown examples. Unknown-validation and final-test performance are prohibited during this replication. Checkpoint-selection measurements must be distinguished from calibration-group measurements; neither is an untouched final evaluation.

Effort is reported through each selected checkpoint separately from total completed effort. Per-head signal counts and parameter movement must cover the same period, and IID/natural comparisons must use matched federated rounds. Reaching the centralized patience criterion describes stopping only and is not proof of optimal convergence.

The post-hoc replay for seeds `240520` and `240523` is complete through round 3 for IID and natural FedAvg. Training behavior remained frozen; instrumentation reproduced the original histories and records client-wise class draws, per-head gradient/update norms, update-to-aggregate cosine alignment, class-center similarity distributions, Normal/Sybil margins, and round-0-to-round-3 known-selection confusion. This diagnostic is known-only and did not modify the detector, loss, sampler, optimizer, preprocessing, partitions, or threshold procedure.

The next bounded development step is the centralized-first known-only screen declared in `config/contribution1_known_only_tuning_plan.json`. It prioritizes learning rate and training window while preserving the CLOSR architecture, original loss, center-similarity inference, sampling, preprocessing, frozen splits, and unknown exclusion. Candidate selection uses only the known checkpoint-selection subset; calibration data remains reserved for the declared threshold procedure and cannot be used to search configurations. After any predeclared three-way confirmation, the protocol must be frozen before the one-time source-disjoint final known/unknown evaluation.
