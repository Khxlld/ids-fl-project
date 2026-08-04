# Phase 2 — trustworthy UAVIDS-2025 partitions

Generated deterministically by `prepare_phase2_partitions.py` from the official hash-verified CSV. No preprocessing was fitted and no model was trained.

## Selected design

- `SrcAddr` is accepted only as a **source-based logical client** key. It is not claimed to identify a verified physical UAV.
- The provisional five sources were retained after applying the final quality/signature rules. Selecting the five largest sources was rejected because that set is dominated by similar flooding-heavy sources and contains no Sybil observations; aggregating many sources into each client was rejected because it weakens the one-source logical-client interpretation.
- The five training sources remain atomic. Every other retained source is assigned whole to validation or test by a deterministic largest-source-first greedy algorithm that balances original class counts and total rows.
- Validation and test use all remaining retained data with natural post-policy prevalence. Neither is balanced, sampled, fitted, or used for hyperparameter decisions.

## Row policy and accounting

- Raw rows: **122,171**.
- Excluded suspicious packet-drop rows: **81**. These have `PacketDropRate` outside [0,1] or `LostPackets > TxPackets`; no values were clipped or imputed.
- Conflicting approved-signature rows: **0**.
- Excluded same-label repeated approved-feature signatures: **8,207**. The lowest `FlowID` representative was retained globally, before source assignment.
- Retained unique rows: **113,883**.

## Partition distributions

| partition | sources | rows | normal | attack | normal_pct | attack_pct | Blackhole Attack | Flooding Attack | Sybil Attack | Wormhole Attack |
|---|---|---|---|---|---|---|---|---|---|---|
| train/uav-client-1 | 1 | 1,230 | 597 | 633 | 48.54% | 51.46% | 255 | 17 | 23 | 338 |
| train/uav-client-2 | 1 | 1,046 | 264 | 782 | 25.24% | 74.76% | 367 | 42 | 25 | 348 |
| train/uav-client-3 | 1 | 591 | 145 | 446 | 24.53% | 75.47% | 161 | 3 | 162 | 120 |
| train/uav-client-4 | 1 | 2,027 | 201 | 1,826 | 9.92% | 90.08% | 329 | 1,429 | 0 | 68 |
| train/uav-client-5 | 1 | 1,254 | 619 | 635 | 49.36% | 50.64% | 252 | 15 | 0 | 368 |
| validation | 87 | 53,786 | 12,130 | 41,656 | 22.55% | 77.45% | 12,370 | 4,993 | 11,857 | 12,436 |
| test | 84 | 53,949 | 12,216 | 41,733 | 22.64% | 77.36% | 12,351 | 5,251 | 11,741 | 12,390 |

All five clients contain normal and attack rows. Their natural attack prevalence remains deliberately non-IID. If Phase 3 needs imbalance mitigation, loss weights or a sampler must be computed from training-client labels only; validation and test must remain unchanged.

## Approved model inputs

`FlowDuration/s`, `TxPackets`, `RxPackets`, `LostPackets`, `TxBytes`, `RxBytes`, `TxPacketRate/s`, `RxPacketRate/s`, `TxByteRate/s`, `RxByteRate/s`, `MeanDelay/s`, `MeanJitter/s`, `Throughput/Kbps`, `PacketDropRate`, `AverageHopCount`

The model-facing partition files contain only these 15 numeric features plus `original_label` and `binary_label`.

## Excluded fields

- `FlowID`, `SrcAddr`, and `DstAddr`: row/source/endpoint identifiers and shortcut risks.
- `SrcPort` and `DstPort`: two-valued, identical to one another, and strongly scenario-associated.
- `Protocol`: constant UDP.
- `MeanPacketSize`: exposes the same port/application regime and is redundant with packet/byte counts.
- Raw `label`: retained under `original_label` for traceability and mapped to binary `binary_label`; it is never an input.

## What this split can test

It tests generalization from five selected logical source addresses to many unseen source addresses under a shared simulated dataset, while preventing exact approved-feature signature overlap. It does **not** establish scenario-, simulation-run-, temporal-, capture-session-, network-, or verified-physical-UAV generalization because those metadata are absent.

## Remaining limitations

- Source addresses can represent attack-specific or forged identities, particularly under Sybil attacks.
- Destination and latent scenario membership can connect otherwise source-disjoint rows, although destination identity is excluded from model inputs.
- Feature engineering was defined using the Phase 1 whole-dataset audit. Validation/test labels were used only to ensure viable group-level distributions, not for model fitting or performance selection.
- The dataset-paper normalization statement conflicts with the raw-scale official CSV; Phase 3 must fit preprocessing on combined training-client rows only.

## Phase 3 boundary

Phase 3 should fit training-only preprocessing, build the binary MLP, and run centralized/local/federated comparisons using this locked manifest. It must select checkpoints with validation data and evaluate the final test partition only after all model and hyperparameter decisions are fixed.
