# Phase 3 — binary model comparison and locked-test results

The policy was locked from validation before `test.csv` was opened. The locked configuration hash remained unchanged during final evaluation; preprocessing was not refitted and no model, threshold, or training setting was revised from test performance.

## Selected preprocessing and MLP

- Training-only pooled median imputation and `StandardScaler` over the 15 approved Phase 2 features.
- MLP: **15 → 128 → 64 → 32 → 1**, ReLU activations and dropout `[0.2, 0.1, 0.0]`.
- Loss: `unweighted`; optimizer: AdamW at learning rate 0.001 and weight decay 0.0001.
- FedAvg: all five logical clients, 2 local epochs per round, 30 rounds, sample-count weighting. The validation-selected global checkpoint is recorded in the federated checkpoint.

## Final locked-test comparison

| Model | Threshold | Accuracy | Macro-F1 | Attack precision | Attack recall | FPR | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| centralized | 0.46 | 0.9841 | 0.9775 | 0.9923 | 0.9871 | 0.0260 | 11898 | 318 | 538 | 41195 |
| federated_fedavg | 0.42 | 0.9650 | 0.9502 | 0.9780 | 0.9767 | 0.0749 | 11301 | 915 | 971 | 40762 |
| local/uav-client-1 | 0.28 | 0.9519 | 0.9295 | 0.9595 | 0.9791 | 0.1410 | 10493 | 1723 | 874 | 40859 |
| local/uav-client-2 | 0.49 | 0.9618 | 0.9444 | 0.9680 | 0.9831 | 0.1111 | 10859 | 1357 | 704 | 41029 |
| local/uav-client-3 | 0.90 | 0.9322 | 0.8977 | 0.9372 | 0.9778 | 0.2237 | 9483 | 2733 | 927 | 40806 |
| local/uav-client-4 | 0.48 | 0.9462 | 0.9256 | 0.9786 | 0.9512 | 0.0711 | 11347 | 869 | 2036 | 39697 |
| local/uav-client-5 | 0.22 | 0.8301 | 0.7872 | 0.9468 | 0.8268 | 0.1586 | 10279 | 1937 | 7227 | 34506 |

Local-only mean macro-F1 is **0.8969**. Best local model: **local/uav-client-2 (0.9444)**. Worst local model: **local/uav-client-5 (0.7872)**.

Federation assessment: **mixed: FedAvg improves on the mean local-only model but trails centralized pooling**.

## Interpretation and limitations

- Centralized training measures the benefit of pooled training rows under the shared train-only preprocessor.
- Local-only results show how strongly generalization depends on one logical source's data distribution.
- FedAvg measures distributed optimization over the five fixed non-IID logical sources, not privacy protection or a networked deployment.
- Pooled preprocessing centrally accessed all training-client feature values; it is not a private federated preprocessing protocol.
- The held-out test is source-address-disjoint and signature-disjoint, but not scenario-, run-, temporal-, independent-network-, or verified-physical-UAV-disjoint.
- Original-class detection rates are saved separately to expose attack-family weaknesses hidden by the binary aggregate.

## Docker handoff

Carry forward the locked feature order, fitted preprocessor, MLP architecture, global checkpoint, decision threshold, binary label mapping, and exact client partition/checksum manifest. Docker clients must load only their own Phase 2 CSV, while the control center owns global validation/evaluation and weighted FedAvg. The deployment must continue to describe clients as logical sources and must not claim that pooled preprocessing is privacy-preserving.
