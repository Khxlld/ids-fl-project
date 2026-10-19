# Contribution 1 fixed multi-seed development replication

## Scope and integrity

Seed `240517` remains the pilot. The additional training seeds are `240519`, `240520`, `240521`, `240522`, and `240523`. All six runs used centralized CLOSR, IID sample-weighted FedAvg CLOSR, and natural Non-IID sample-weighted FedAvg CLOSR only. The detector, official-aligned class-specific CLOSR loss, architecture, inference, preprocessing, sampling policies, budgets, and stopping rules were not changed.

The cross-run integrity audit passed. The resolved configurations differ only at `training.seed`; the source partitions, Wormhole exclusion, source-group-disjoint checkpoint-selection/calibration split (seed `240517`), and IID repartition (seed `240518`) are byte-for-byte/configuration-identical. Within each seed, all modes loaded the same initial tensor state and the global RNGs were reset to that seed before each mode. The five new runs record the initial-state SHA-256 in `rng_audit.json`.

Changing `training.seed` changes Python `random`, NumPy's global RNG, PyTorch CPU RNG, PyTorch CUDA RNG state if present (execution remained CPU), dropout, and balanced-sampler draws. Centralized loader seeds are `training_seed + epoch`; federated loader seeds are `training_seed + round*1000 + client_index*100 + local_epoch`.

No run loaded the final test or evaluated unknown validation. All values below are known-only development measurements. Checkpoint-selection measurements participated in model choice; calibration measurements come from the separate known group used to set the threshold and are not an untouched performance estimate.

## Individual results

`Sel closed` is known closed-set Macro-F1 used for checkpoint selection. `Sel thresholded` applies the known-only calibrated CLOSR rejection threshold back to the selection group. `Cal thresholded` is measured on the group that calibrated that threshold.

| Seed | Role | Mode | Selected / completed | Sel closed | Sel thresholded | Cal thresholded | Sel benign rejection | Sel benign total alert FPR |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 240517 | pilot | Centralized | 12 / 20 | 0.7219 | 0.7201 | 0.7725 | 0.0206 | 0.0455 |
| 240517 | pilot | IID FedAvg | 10 / 30 | 0.7041 | 0.6939 | 0.7417 | 0.0195 | 0.0627 |
| 240517 | pilot | Natural Non-IID | 1 / 30 | 0.6097 | 0.6093 | 0.6317 | 0.0374 | 0.1978 |
| 240519 | replication | Centralized | 16 / 24 | 0.7455 | 0.7368 | 0.7883 | 0.0108 | 0.0563 |
| 240519 | replication | IID FedAvg | 22 / 30 | 0.7196 | 0.7087 | 0.7607 | 0.0506 | 0.1087 |
| 240519 | replication | Natural Non-IID | 1 / 30 | 0.5771 | 0.5666 | 0.5946 | 0.0248 | 0.1721 |
| 240520 | replication | Centralized | 25 / 33 | 0.7794 | 0.7706 | 0.8314 | 0.0276 | 0.0444 |
| 240520 | replication | IID FedAvg | 12 / 30 | 0.7247 | 0.7142 | 0.7622 | 0.0106 | 0.0389 |
| 240520 | replication | Natural Non-IID | 1 / 30 | 0.7265 | 0.7242 | 0.7839 | 0.0242 | 0.0869 |
| 240521 | replication | Centralized | 22 / 30 | 0.7833 | 0.7810 | 0.8349 | 0.0231 | 0.0469 |
| 240521 | replication | IID FedAvg | 26 / 30 | 0.7252 | 0.7239 | 0.7786 | 0.0191 | 0.0571 |
| 240521 | replication | Natural Non-IID | 1 / 30 | 0.6766 | 0.6772 | 0.7310 | 0.0472 | 0.1293 |
| 240522 | replication | Centralized | 13 / 21 | 0.7460 | 0.7415 | 0.7967 | 0.0117 | 0.0408 |
| 240522 | replication | IID FedAvg | 6 / 30 | 0.6945 | 0.6862 | 0.7355 | 0.0329 | 0.0739 |
| 240522 | replication | Natural Non-IID | 18 / 30 | 0.5773 | 0.5670 | 0.6115 | 0.0474 | 0.0629 |
| 240523 | replication | Centralized | 12 / 20 | 0.7586 | 0.7505 | 0.8070 | 0.0236 | 0.0926 |
| 240523 | replication | IID FedAvg | 24 / 30 | 0.7114 | 0.6997 | 0.7541 | 0.0380 | 0.0771 |
| 240523 | replication | Natural Non-IID | 1 / 30 | 0.5983 | 0.5972 | 0.6151 | 0.0425 | 0.2462 |

## Six-seed mean and sample standard deviation

| Mode | Sel closed Macro-F1 | Sel thresholded Macro-F1 | Cal thresholded Macro-F1 | Sel benign rejection | Sel benign total alert FPR |
|---|---:|---:|---:|---:|---:|
| Centralized | 0.7558 ± 0.0231 | 0.7501 ± 0.0225 | 0.8051 ± 0.0245 | 0.0196 ± 0.0068 | 0.0544 ± 0.0194 |
| IID FedAvg | 0.7133 ± 0.0123 | 0.7044 ± 0.0139 | 0.7555 ± 0.0155 | 0.0284 ± 0.0148 | 0.0697 ± 0.0234 |
| Natural Non-IID FedAvg | 0.6276 ± 0.0607 | 0.6236 ± 0.0638 | 0.6613 ± 0.0772 | 0.0373 ± 0.0106 | 0.1492 ± 0.0693 |

Calibration benign total alert FPR means were `0.0593 ± 0.0212`, `0.0711 ± 0.0244`, and `0.1581 ± 0.0721` for centralized, IID, and natural Non-IID respectively. Every threshold achieved known calibration rejection `0.0499914` with one score tied at the threshold under the strict `score > threshold` rule.

## Paired IID-minus-natural observations

The paired selection closed-set Macro-F1 difference was positive in five of six seeds and averaged `+0.0857 ± 0.0530`; seed `240520` was the exception at `-0.0018`. The thresholded selection difference averaged `+0.0809 ± 0.0550` and the calibration-group difference averaged `+0.0942 ± 0.0692`, also positive in five of six seeds. The selection benign total alert FPR difference averaged `-0.0795 ± 0.0642` (IID lower in five of six seeds), but reversed for seed `240522` (`+0.0110`). The development gap is therefore directionally common but not universal.

Paired selection recall-after-rejection differences were:

| Class | IID minus natural mean ± SD | Seeds with positive difference |
|---|---:|---:|
| Normal Traffic | +0.0795 ± 0.0642 | 5 / 6 |
| Blackhole Attack | +0.1667 ± 0.0734 | 6 / 6 |
| Flooding Attack | +0.1679 ± 0.1007 | 5 / 6 |
| Sybil Attack | -0.0069 ± 0.1826 | 2 / 6 |

This pattern does not support treating missing local Sybil targets as a complete explanation. Natural Non-IID also had lower and more variable Blackhole and Flooding recall, while Sybil differences changed sign substantially across seeds.

## Benign-to-known-attack confusion

On the selection group, the mean natural Non-IID benign row was split into `0.8508` correct Normal, `0.0075` Blackhole, `0.0412` Flooding, `0.0633` Sybil, and `0.0373` UNKNOWN. IID averaged `0.9303` correct, `0.0123` Blackhole, `0.0057` Flooding, `0.0232` Sybil, and `0.0284` UNKNOWN. The dominant natural benign attack destination was seed-dependent: Flooding was large for `240517`/`240519`, while Sybil was large for `240523`. A threshold change is not proposed; these are both closed-class confusion and rejection effects that require training-dynamics diagnosis.

## Training effort and matched-round diagnostics

Centralized runs selected epochs 12–25 and stopped at epochs 20–33 because the configured patience criterion was reached. This states the stopping rule only; it is not proof of optimal convergence. Both federated modes completed all 30 configured rounds and 1,200 client optimizer steps in every seed. IID selected rounds ranged 6–26. Natural selected round 1 in five seeds and round 18 in seed `240522`. All configurations improved their selected known closed-set Macro-F1 over their same-seed initial model, but the budget does not establish convergence to an optimum.

Effort is now reported separately through the selected checkpoint and through completed training. For example, the pilot natural checkpoint used 4,906 draws/40 client optimizer steps at selected round 1, while the completed run used 147,180 draws/1,200 steps. Parameter-path measurements and target-signal counts are always paired over the same period.

At matched round 1, averaged across seeds, IID target draws were approximately `[1239, 1224, 1229, 1214]` for Normal/Blackhole/Flooding/Sybil, with all 40 client batches active for every head. Natural target draws were approximately `[1463, 1466, 1453, 525]`; its Sybil head was active in 17 of 40 batches because two clients lack Sybil, while the other heads were active in all 40. Natural Sybil head path length was lower, but non-Sybil recall and benign confusion also varied materially, so this diagnostic alone does not establish a cause.

## Conclusion and smallest next diagnostic

The IID-over-natural development gap is common across these fixed seeds but not fully consistent: five of six seeds favor IID on known closed-set Macro-F1, one is effectively tied/slightly reversed, and error destinations vary by seed. These results do not establish that Non-IID data caused the gap, do not evaluate zero-day effectiveness, and are not a novelty claim.

The smallest justified next diagnostic is a two-seed, round-1 replay for seeds `240520` (no IID Macro-F1 advantage) and `240523` (large IID advantage), running only IID and natural FedAvg with training behavior frozen. Instrument, but do not alter, each client's class draws, per-head gradient/update norms, client-update cosine alignment with the aggregate, and the round-0/round-1 known selection confusion. This contrast is the minimum useful check of whether client-update conflict and head-specific exposure track the divergent outcomes. It must remain known-only and must not change thresholds, detector, sampler, or loss.

## Artifact index

- `integrity_audit.json`: cross-run frozen-component and evaluation-access checks.
- `individual_seed_results.csv`: all seed/mode known-only selection and calibration measurements.
- `aggregate_mean_sd.csv`: six-seed mean and sample SD.
- `paired_iid_minus_natural_individual.csv` and `paired_iid_minus_natural_summary.csv`: paired differences.
- `known_confusion_long.csv`, `benign_confusion_patterns.csv`, and `benign_confusion_patterns_mean_sd.csv`: confusion analysis.
- `selected_vs_total_effort.csv`: checkpoint-period versus completed effort.
- `per_head_period_diagnostics.csv`: period-aligned signal and movement.
- `matched_round_federated_diagnostics.csv`, `matched_round_iid_minus_natural.csv`, and `matched_round_iid_minus_natural_mean_sd.csv`: matched-round comparisons.

Each per-seed run remains in `../contribution1-three-way-seed-<seed>/` with its resolved configuration, learning curves, selected checkpoint, confusion matrices, audit records, and full history.
