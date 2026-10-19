# Contribution 1 rounds 0-3 known-only diagnostic

## Scope and reproduction

This is a bounded, post-hoc exploratory replay of IID and natural Non-IID FedAvg CLOSR for seeds `240520` and `240523`. It evaluates the fixed known checkpoint-selection subset only. The detector, CLOSR loss, sampler, preprocessing, partitions, initialization, optimizer, center construction, calibration procedure, and training behavior are unchanged. Unknown validation and final-test performance were not evaluated.

For every seed/mode, ordinary and instrumented replays reproduced rounds 1, 2, and 3 exactly. Client states and aggregate states were bitwise identical; saved history fields matched exactly after excluding runtime; and Python, NumPy, and Torch RNG end states matched. The existing round-0 IID/natural closed predictions agree on all 18,146 selection rows. The largest round-0 center and score differences are `7.45e-09` and `1.19e-07`, respectively, from alternate client summation order.

All 60 encoder/head aggregation checks match the sample-weighted sum of actual client deltas within the declared float32 FedAvg tolerance. The largest elementwise discrepancy is `1.49e-08`.

## Known-selection trajectories

Closed-set results are primary in this diagnostic. Thresholded results are stored separately and were not used to explain the closed-class confusion.

| Seed | Mode | R0 Macro-F1 | R1 | R2 | R3 |
|---:|---|---:|---:|---:|---:|
| 240520 | IID | 0.5255 | 0.5337 | 0.5589 | 0.6680 |
| 240520 | Natural | 0.5255 | 0.7265 | 0.5384 | 0.5443 |
| 240523 | IID | 0.5419 | 0.5932 | 0.5523 | 0.5729 |
| 240523 | Natural | 0.5419 | 0.5983 | 0.5328 | 0.5434 |

The natural-minus-IID round-1 contrast is positive (`+0.1927`, `+0.0051`) but becomes negative at round 2 (`-0.0205`, `-0.0195`) and round 3 (`-0.1237`, `-0.0295`). These two examples show a changing early trajectory, not a causal Non-IID result.

### Normal and Sybil closed-set behavior

`Pred` is the total number of rows predicted as the named class. `N->S`, `S->N`, and `S->F` are confusion counts. Normal and Sybil supports are 5,292 and 5,352.

| Seed | Mode | Round | Normal recall | Normal precision | Normal pred | N->S | Sybil recall | Sybil precision | Sybil pred | S->N | S->F |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 240520 | IID | 0 | .9424 | .6516 | 7654 | 10 | .0340 | .5583 | 326 | 1910 | 744 |
| 240520 | IID | 1 | .9333 | .9146 | 5400 | 105 | .0250 | .3066 | 437 | 138 | 4664 |
| 240520 | IID | 2 | .9514 | .9703 | 5189 | 50 | .0273 | .4220 | 346 | 95 | 4800 |
| 240520 | IID | 3 | .9579 | .9754 | 5197 | 81 | .2513 | .8728 | 1541 | 58 | 3795 |
| 240520 | Natural | 0 | .9424 | .6516 | 7654 | 10 | .0340 | .5583 | 326 | 1910 | 744 |
| 240520 | Natural | 1 | .9189 | .9153 | 5313 | 331 | .5787 | .7393 | 4189 | 109 | 1698 |
| 240520 | Natural | 2 | .9495 | .9555 | 5259 | 58 | .0028 | .1327 | 113 | 118 | 4722 |
| 240520 | Natural | 3 | .9737 | .9499 | 5425 | 6 | .0000 | .0000 | 19 | 160 | 4745 |
| 240523 | IID | 0 | .9044 | .7764 | 6164 | 19 | .1050 | .6854 | 820 | 904 | 3559 |
| 240523 | IID | 1 | .9006 | .9425 | 5057 | 17 | .1747 | .7620 | 1227 | 80 | 3910 |
| 240523 | IID | 2 | .9405 | .9689 | 5137 | 75 | .0342 | .3853 | 475 | 111 | 4698 |
| 240523 | IID | 3 | .9492 | .9870 | 5089 | 79 | .0372 | .5169 | 385 | 31 | 4911 |
| 240523 | Natural | 0 | .9044 | .7764 | 6164 | 19 | .1050 | .6854 | 820 | 904 | 3559 |
| 240523 | Natural | 1 | .7553 | .9369 | 4266 | 1140 | .3186 | .4376 | 3896 | 60 | 3314 |
| 240523 | Natural | 2 | .8985 | .9678 | 4913 | 19 | .0015 | .2105 | 38 | 111 | 4865 |
| 240523 | Natural | 3 | .9422 | .9594 | 5197 | 45 | .0000 | .0000 | 55 | 136 | 4856 |

The seed-240523 round-1 benign-to-Sybil spike (1,140 rows) does not persist: it falls to 19 at round 2 and 45 at round 3. Seed 240520 follows the same qualitative rise/fall at smaller magnitude (331, 58, 6). However, this is not recovery of Sybil classification: by round 3 natural Sybil recall is zero in both seeds, and most true Sybil rows are predicted as Flooding. IID also has serious Sybil-to-Flooding confusion, although seed 240520 reaches 25.1% Sybil recall at round 3. The known-class weakness therefore cannot be reduced to a stable Normal-to-Sybil boundary problem.

## Center-similarity geometry

The full distributions (row count, mean, sample SD, min, p05, p25, median, p75, p95, max) for every class center and for both true classes are in `similarity_distribution_summary.csv`; record-level scores are in `normal_sybil_record_scores.csv.gz`. `similarity_distribution_changes.csv` records changes from round 0 and from the preceding round.

The mean true-class-minus-largest-wrong-class margin develops as follows:

| Seed | Mode | True class | R0 | R1 | R2 | R3 |
|---:|---|---|---:|---:|---:|---:|
| 240520 | IID | Normal | .0283 | .6885 | .8996 | 1.3038 |
| 240520 | Natural | Normal | .0283 | .2833 | .5616 | .7689 |
| 240523 | IID | Normal | .0162 | .7644 | 1.2099 | 1.3122 |
| 240523 | Natural | Normal | .0162 | .1924 | .6574 | .9145 |
| 240520 | IID | Sybil | -.0927 | -.3261 | -.5065 | -.6734 |
| 240520 | Natural | Sybil | -.0927 | .0629 | -.2301 | -.3645 |
| 240523 | IID | Sybil | -.0524 | -.2737 | -.3952 | -.6604 |
| 240523 | Natural | Sybil | -.0524 | -.0113 | -.2050 | -.2489 |

Normal margins become strongly positive in every trajectory. Sybil margins are negative at round 3 in every trajectory, agreeing with the broad Sybil-to-Flooding confusion. Distribution summaries, rather than means alone, must be used because a mean margin does not determine recall.

For true Normal, the mean Normal-minus-Sybil similarity margin grows from `.1224` to `1.4111` (IID) and `.8103` (natural) for seed 240520, and from `.0794` to `1.4681` (IID) and `.9824` (natural) for seed 240523. For true Sybil, the same signed margin becomes negative as the Sybil center overtakes the Normal center, but the Flooding center commonly remains larger. At natural round 3, mean `[Normal, Sybil, Flooding]` similarities for true Sybil are `[-.5884, .4733, .7106]` for seed 240520 and `[-.7362, .7085, .8372]` for seed 240523.

## Update and gradient diagnostics

The existing update summaries were extended without changing the trajectory. Natural encoder cancellation rises from `.1636` to `.2983` to `.4245` across rounds 1-3 for seed 240520 and from `.1534` to `.2371` to `.3892` for seed 240523; IID values are `.0789/.1466/.1961` and `.0678/.1473/.1847`. Natural Sybil-head aggregate update norms are `.1163/.1015/.0847` and `.1209/.1130/.0970`, below IID `.3342/.2247/.2184` and `.3724/.2388/.2472`. These are observations only; update-conflict correction and missing-class compensation are not proposed.

The natural Sybil head has nonzero loss gradients in 17 of 40 client batches per round and zero gradients in 23; IID has nonzero gradients in all 40. No gradient is missing (`grad is None`). Per-client draws, pair counts, gradient categories, parameter paths, weighted contributions, pairwise cosines, and cancellation values remain available in their respective CSV files.

## Evidence-supported conclusion

The initial concern is real but incomplete. A large benign-to-Sybil error occurs at natural round 1, especially for seed 240523, then rapidly recedes. At the same time, Sybil recall collapses and Sybil-to-Flooding becomes dominant. The two seeds also differ substantially in IID recovery by round 3. This supports a shared early-training/known-class baseline weakness plus a possible additional federated-partition-associated difference to test under controlled tuning; it does not establish a cause, a threshold remedy, a missing-class remedy, or an update-conflict remedy.

## Next bounded development plan (not executed)

The executable plan is recorded in `../../config/contribution1_known_only_tuning_plan.json` with status `planned_not_run`.

1. Screen four centralized-only settings across the six existing development seeds: the frozen baseline; a longer 80-epoch/patience-16 window at `1e-3`; and the same longer window at `5e-4` and `2.5e-4`. Keep AdamW weight decay `1e-4`, batch size 128, balanced replacement sampling, architecture, loss, centers, and inference unchanged.
2. Rank candidates using only known-selection closed-set Macro-F1, with predeclared Normal/Sybil confusion and effort tie-breakers. Calibration performance cannot select a configuration.
3. Freeze the centralized winner. Apply its learning rate to both FedAvg controls for a three-way confirmation at the existing 30 rounds. Only if the centralized winner required the longer window, run one predeclared 60-round FedAvg confirmation. Do not search separate IID or natural hyperparameters.
4. A consistent centralized and IID gain indicates a shared baseline optimization weakness. A centralized gain without an IID gain identifies an additional FedAvg optimization issue. A persistent paired natural-versus-IID gap after centralized/IID improvement identifies an additional partition-associated federated gap under these controls, not causation.
5. After development, freeze code/configuration/dependencies/seeds/hashes/selection/calibration/output schema and the one-shot command. Only then perform the declared source-disjoint final evaluation, including the frozen Wormhole unknown class, without retuning.

No further diagnostic series or tuning run was launched here.

## Artifact index

- `reproduction_checks.json`: per-round state/history/RNG reproduction checks.
- `round0_to_round3_metrics.csv`: closed and separately thresholded known-selection metrics.
- `closed_confusion_round0_to_round3.csv` and `thresholded_confusion_round0_to_round3.csv`: separate confusion matrices.
- `normal_sybil_class_metrics.csv` and `normal_sybil_class_metric_changes.csv`: recall, precision, prediction count, and changes.
- `similarity_distribution_summary.csv` and `similarity_distribution_changes.csv`: every center similarity distribution.
- `margin_distribution_summary.csv` and `margin_distribution_changes.csv`: Normal/Sybil and largest-wrong margins.
- `normal_sybil_record_scores.csv.gz`: record-level fixed-selection scores and margins.
- `per_client_class_signal.csv`, `loss_gradient_summary.csv`, `client_update_norms.csv`, `pairwise_update_cosines.csv`, and `weighted_update_cancellation.csv`: rounds 1-3 training diagnostics.
- `aggregation_verification.csv` and `actual_client_parameter_deltas_rounds1_to3.pt`: exact weighted-delta checks and tensors.
- `scope_and_integrity.json`: protected-evaluation and method-change flags.
