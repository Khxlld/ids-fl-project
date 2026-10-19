# Post-hoc round-1 FedAvg CLOSR diagnostic

## Scope

This is an exploratory replay of IID and natural Non-IID FedAvg CLOSR at round 1 for seeds `240520` and `240523`. It is not a new performance experiment. The detector, class-specific CLOSR loss, sampler, preprocessing, partitions, initialization procedure, AdamW optimizer, FedAvg, center computation, checkpoint/calibration groups, and rejection rule are unchanged. No unknown-validation or final-test performance was evaluated.

## Reproduction and instrumentation preservation

Each seed/mode was replayed once through the ordinary training function and again through an instrumented mirror after resetting the same RNGs. For all four cases:

- every client tensor and aggregated tensor was bitwise identical;
- maximum instrumented-versus-ordinary tensor difference was exactly zero;
- Python, NumPy, and PyTorch RNG states after the replay were identical;
- round-1 accuracy, Macro-F1, mean local loss, class draws, and optimizer steps exactly matched the prior 30-round run history; and
- the reconstructed initial-state SHA-256 matched the saved baseline RNG audit.

There are no reproduction discrepancies. The instrumentation only reads/clones existing tensors and computes deterministic reductions after backward; it makes no random draw.

## Round-0 IID/natural agreement

The two controls use the same initial tensors and pooled rows, but their center sufficient statistics are summed in different client/row orders. The resulting floating-point differences were negligible and caused no prediction mismatch.

| Seed | Center max abs diff | Similarity max abs diff | Score max abs diff | Closed prediction mismatches / 18,146 |
|---:|---:|---:|---:|---:|
| 240520 | 7.45e-09 | 1.19e-07 | 1.19e-07 | 0 |
| 240523 | 1.86e-09 | 1.19e-07 | 1.19e-07 | 0 |

Round-0 closed and thresholded metrics were identical between IID and natural within each seed. Calibration was known-only and used only to report the thresholded selection diagnostics.

## Matched round-1 result

| Seed | Mode | Closed Macro-F1 | Thresholded Macro-F1 | Benign rejection | Benign total alert FPR |
|---:|---|---:|---:|---:|---:|
| 240520 | IID | 0.5337 | 0.5374 | 0.0187 | 0.0771 |
| 240520 | Natural | 0.7265 | 0.7242 | 0.0242 | 0.0869 |
| 240523 | IID | 0.5932 | 0.5859 | 0.0554 | 0.1306 |
| 240523 | Natural | 0.5983 | 0.5972 | 0.0425 | 0.2462 |

The proposed selected-checkpoint contrast is absent at matched round 1. Natural minus IID closed Macro-F1 is `+0.1927` for seed `240520` and `+0.0051` for seed `240523`; thresholded differences are `+0.1868` and `+0.0114`. In the existing full histories the sign changes at round 2 for both seeds. By round 3 IID leads by `0.1237` for `240520` but only `0.0295` for `240523`; the latter reaches a similar gap later. Therefore a round-1-only update diagnostic cannot explain the later selected-checkpoint comparison.

## Round-1 closed-set class behavior

| Seed | Mode | Normal recall | Blackhole recall | Flooding recall | Sybil recall |
|---:|---|---:|---:|---:|---:|
| 240520 | IID | 0.9333 | 0.9442 | 0.6373 | 0.0250 |
| 240520 | Natural | 0.9189 | 0.8325 | 0.7136 | 0.5787 |
| 240523 | IID | 0.9006 | 0.9657 | 0.5727 | 0.1747 |
| 240523 | Natural | 0.7553 | 0.8648 | 0.5774 | 0.3186 |

For seed `240523`, natural and IID have similar Macro-F1 but very different errors. Of 5,292 benign selection rows, natural predicts Sybil for 1,140, compared with 17 for IID. This explains much of natural's higher benign total alert FPR, while natural simultaneously has higher Sybil recall. A threshold change would not resolve this closed-class tradeoff.

## Sampling and valid pair signal

Aggregate realized draws and defined/active CLOSR pair counts are in `per_client_class_signal.csv`. The round-1 aggregate class draws were:

| Seed | Mode | Normal | Blackhole | Flooding | Sybil |
|---:|---|---:|---:|---:|---:|
| 240520 | IID | 1,205 | 1,177 | 1,277 | 1,247 |
| 240520 | Natural | 1,419 | 1,439 | 1,479 | 569 |
| 240523 | IID | 1,292 | 1,248 | 1,184 | 1,182 |
| 240523 | Natural | 1,476 | 1,482 | 1,448 | 500 |

Natural clients 4 and 5 had zero Sybil draws and zero positive/negative pairs for the Sybil head. Clients 1–3 had valid positive and negative Sybil pairs. The saved positive-pair count includes target diagonals to match the pinned upstream implementation; active positive-pair counts exclude zero-distance entries. Negative active-pair counts apply the hinge condition.

FedAvg sample counts and weights were identical between IID and natural:

| Client | Rows | Weight |
|---:|---:|---:|
| 1 | 892 | 0.181818 |
| 2 | 698 | 0.142275 |
| 3 | 471 | 0.096005 |
| 4 | 1,959 | 0.399307 |
| 5 | 886 | 0.180595 |

## Gradients versus optimizer movement

Every encoder and non-Sybil head had nonzero gradients in all 40 client batches for both controls and seeds. IID Sybil heads also had nonzero gradients in all 40 batches. Natural Sybil heads had nonzero gradients in the 17 batches from clients 1–3 and exactly zero gradient tensors in the 23 batches from clients 4–5. No parameter group had a missing (`grad is None`) gradient.

Because AdamW applies decoupled weight decay to a present zero gradient tensor, the absent-Sybil client heads moved slightly despite zero loss gradient:

| Seed | Client | Zero-gradient batches | Sybil-head optimizer path movement |
|---:|---:|---:|---:|
| 240520 | 4 | 16 | 5.68e-06 |
| 240520 | 5 | 7 | 2.49e-06 |
| 240523 | 4 | 16 | 5.62e-06 |
| 240523 | 5 | 7 | 2.46e-06 |

These movements are optimizer-induced, not CLOSR loss signal. They are more than four orders of magnitude smaller than trained Sybil-head client deltas (`0.21–0.35`).

## Update norms, alignment, and cancellation

`Retention` is `||sum_i w_i Δ_i|| / sum_i ||w_i Δ_i||`. `Cancellation` is `1 - retention`; higher values indicate more cancellation. Pairwise cosine is computed on unweighted client deltas. A cosine is undefined when either vector norm is at most `1e-5`, which excludes the absent-head, weight-decay-only movements.

| Seed | Mode | Group | Aggregate delta norm | Mean pairwise cosine | Defined pairs | Cancellation |
|---:|---|---|---:|---:|---:|---:|
| 240520 | IID | Encoder | 0.9724 | 0.7810 | 10/10 | 0.0789 |
| 240520 | Natural | Encoder | 0.9305 | 0.5627 | 10/10 | 0.1636 |
| 240520 | IID | Normal head | 0.3887 | 0.8948 | 10/10 | 0.0418 |
| 240520 | Natural | Normal head | 0.3855 | 0.7982 | 10/10 | 0.0792 |
| 240520 | IID | Blackhole head | 0.3994 | 0.8533 | 10/10 | 0.0571 |
| 240520 | Natural | Blackhole head | 0.3946 | 0.7583 | 10/10 | 0.0894 |
| 240520 | IID | Flooding head | 0.3719 | 0.8664 | 10/10 | 0.0550 |
| 240520 | Natural | Flooding head | 0.3802 | 0.7672 | 10/10 | 0.0871 |
| 240520 | IID | Sybil head | 0.3342 | 0.8114 | 10/10 | 0.0790 |
| 240520 | Natural | Sybil head | 0.1163 | 0.7524 | 3/10 | 0.0637 |
| 240523 | IID | Encoder | 1.0600 | 0.8030 | 10/10 | 0.0678 |
| 240523 | Natural | Encoder | 0.9963 | 0.5747 | 10/10 | 0.1534 |
| 240523 | IID | Normal head | 0.4220 | 0.8978 | 10/10 | 0.0395 |
| 240523 | Natural | Normal head | 0.4024 | 0.7773 | 10/10 | 0.0806 |
| 240523 | IID | Blackhole head | 0.4358 | 0.8851 | 10/10 | 0.0416 |
| 240523 | Natural | Blackhole head | 0.4111 | 0.7805 | 10/10 | 0.0825 |
| 240523 | IID | Flooding head | 0.3869 | 0.8540 | 10/10 | 0.0599 |
| 240523 | Natural | Flooding head | 0.3822 | 0.7072 | 10/10 | 0.1068 |
| 240523 | IID | Sybil head | 0.3724 | 0.8362 | 10/10 | 0.0693 |
| 240523 | Natural | Sybil head | 0.1209 | 0.7155 | 3/10 | 0.0722 |

Natural updates are less aligned and cancel more for the encoder and all three non-Sybil heads in both examples. Natural Sybil aggregate movement is about one third of IID movement because clients 4–5 contribute only negligible weight-decay movement. However, natural is not worse on matched round-1 Macro-F1 and has higher Sybil recall in both seeds. Consequently these round-1 update properties do not, by themselves, explain the later IID advantage.

Fourteen Sybil-head pairwise cosines are marked undefined: the seven pairs involving natural client 4 or 5 in each seed. All other cosine vectors exceed the declared norm threshold.

## FedAvg arithmetic verification

For every seed, mode, encoder, and individual head, the observed aggregated parameter delta matches the weighted sum of actual client deltas within the expected float64-accumulation/float32-cast tolerance. All 20 checks pass. The maximum elementwise discrepancy is `1.49e-08`; no aggregation discrepancy was found.

The binary tensor artifact `actual_client_parameter_deltas.pt` contains each actual client delta, each weighted contribution, and the observed aggregate delta from the common global start. `client_update_norms.csv` provides readable group norms and weights.

## Interpretation and smallest evidence-supported follow-up

These two post-hoc examples show that natural partitions produce lower client-update alignment, greater encoder/non-Sybil cancellation, and strongly reduced direct Sybil-head movement at round 1. They do not show that these properties cause poorer matched-round-1 Macro-F1: natural is better for `240520` and approximately tied for `240523`. They also do not support a round-1 contrast between the two selected examples.

The smallest justified follow-up is to extend the same non-invasive instrumentation through rounds 2 and 3 for both seeds and both controls, starting from the exactly reproduced round-1 states. The full histories show that the IID-minus-natural sign flips at round 2 in both seeds, while by round 3 the gap is `0.1237` for `240520` but only `0.0295` for `240523`. Measuring the same client deltas, alignment, cancellation, and known confusion during those two transitions can test whether changes in update geometry track the emerging gap. No detector, loss, sampler, optimizer, threshold, preprocessing, or partition change is justified by the present evidence.

## Artifact index

- `reproduction_checks.json`: exact prior-run and instrumented/uninstrumented checks.
- `round0_iid_natural_agreement.csv`: center, score, and prediction agreement.
- `per_client_class_signal.csv`: draws and positive/negative defined and active pairs.
- `loss_gradient_summary.csv`: zero/missing/nonzero gradient and optimizer-movement summaries.
- `actual_client_parameter_deltas.pt`: actual client, weighted, and aggregate delta tensors.
- `client_update_norms.csv`: FedAvg weights and encoder/head delta/contribution norms.
- `pairwise_update_cosines.csv`: pairwise client-update cosines and undefined flags.
- `weighted_update_cancellation.csv`: per-group retention/cancellation.
- `aggregation_verification.csv`: weighted-sum arithmetic checks.
- `round0_round1_metrics.csv`: closed and separately thresholded known-only metrics.
- `round0_round1_confusion_long.csv`: closed and thresholded confusion matrices.
- `scope_and_integrity.json`: scope and protected-evaluation declarations.
