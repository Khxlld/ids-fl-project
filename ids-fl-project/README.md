# Enhancing Intrusion Detection Systems with Federated Learning

This repository studies centralized and federated intrusion detection for
IoT/IIoT traffic. The current implemented experiment is binary classification
(`0 = Normal`, `1 = Attack`) on DNN-EdgeIIoT with a PyTorch MLP and five-client
Flower FedAvg simulation.

## Current status

As of **2026-07-23**, a leakage-aware, capture-group-disjoint experiment is the
current baseline. Its saved test results are:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Centralized MLP | 0.97059 | 0.99123 | 0.94958 | 0.96996 | 0.99167 |
| Federated global MLP | 0.97059 | 0.99123 | 0.94958 | 0.96996 | 0.99237 |

The earlier 100% result is retained for traceability, but it is **not** the
paper-ready baseline because later analysis found source-format and
duplicate-pattern leakage. See [Research status](docs/RESEARCH_STATUS.md) for
the dated record, interpretation, limitations, and next steps.

## Repository layout

```text
ids-fl-project/
├── data/                                  # local raw data; not committed
├── docs/RESEARCH_STATUS.md                # dated research and result log
└── experiments/edgeiiot_federated/
    ├── notebooks/                         # executed experiment notebooks
    ├── src/                               # modular first-generation pipeline
    ├── tests/                             # fast unit tests
    ├── results/                           # historical first-generation metrics
    ├── results_clean/                     # current leakage-aware metrics
    └── artifacts*/                        # small feature/drop summaries only
```

The self-contained experiment README has setup, dataset, and run instructions:
[experiments/edgeiiot_federated/README.md](experiments/edgeiiot_federated/README.md).

## Dataset

The experiment uses `DNN-EdgeIIoT-dataset.csv` from
[Edge-IIoTset](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot).
Place it at:

```text
data/raw/DNN-EdgeIIoT-dataset.csv
```

The raw dataset and generated model binaries are intentionally excluded from
Git. You can also set `EDGEIIOT_DATASET` to an absolute local CSV path.

## Broader roadmap

- [x] Implement centralized and five-client IID federated binary MLP baselines
- [x] Identify leakage in the first experiment and produce a group-disjoint run
- [ ] Modularize and test the leakage-aware notebook pipeline
- [ ] Repeat across seeds and report uncertainty
- [ ] Add natural-imbalance and non-IID client experiments
- [ ] Add classical baselines, ablations, and external/temporal validation
- [ ] Extend to multiclass classification
