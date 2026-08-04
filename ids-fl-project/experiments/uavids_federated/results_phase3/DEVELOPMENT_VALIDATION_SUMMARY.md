# Phase 3 development and validation summary

The final-test CSV was not opened during this run. All decisions below were made from the five Phase 2 training clients and the validation partition.

## Locked policy

- Preprocessing: median imputation plus `StandardScaler`, fitted once on all **6,148 pooled training-client rows** and only transformed on validation.
- Selected candidate: **edge_style_unweighted**, hidden layers `[128, 64, 32]`, dropout `[0.2, 0.1, 0.0]`.
- Loss: **unweighted** with attack `pos_weight=1.000000`. This weight was calculated from pooled training labels only.
- Optimizer: AdamW, learning rate 0.001, weight decay 0.0001, batch size 128.
- Checkpoints: highest validation macro-F1 at threshold 0.5, then validation attack recall, lower FPR, and lower log loss.
- Final decision thresholds: selected separately on validation macro-F1 and now locked.
- FedAvg: all five clients each round, 2 local epochs, 30 rounds, aggregation weighted by retained client sample count.

Pooled preprocessing is not privacy-preserving: raw training features were centrally available to fit global medians and scaling statistics. This is acceptable only as a research-prototype baseline and must be replaced or explicitly disclosed in a distributed deployment.

## Validation results

| Model | Threshold | Accuracy | Macro-F1 | Attack precision | Attack recall | FPR | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Centralized | 0.46 | 0.9842 | 0.9775 | 0.9928 | 0.9868 | 0.0246 | 11831 | 299 | 551 | 41105 |
| Federated FedAvg | 0.42 | 0.9628 | 0.9467 | 0.9757 | 0.9762 | 0.0833 | 11119 | 1011 | 990 | 40666 |

The five local-only validation macro-F1 scores have mean **0.8945**; the worst is **local/uav-client-5 at 0.7926**. Full local and candidate tables are saved as CSV.

The centralized checkpoint was epoch **52**. The selected federated checkpoint was round **30**.

## Test lock

`phase3_locked_config.json` records all feature, architecture, loss, optimizer, round, threshold, checkpoint, and artifact hashes. Phase 3 final evaluation must consume that file without revising it.
