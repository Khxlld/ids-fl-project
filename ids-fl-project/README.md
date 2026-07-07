# Enhancing Intrusion Detection Systems Using Federated Learning for Decentralized Model Training

## Overview
This project explores intrusion detection systems (IDS) for IoT/IIoT networks, starting with
centralized baseline machine learning models and extending toward federated learning for
decentralized model training.

**Current phase:** Baseline model development and optimization on the Edge-IIoTset dataset,
prior to the federated learning phase.

## Team
This is a collaborative 5-person project. Each member independently builds the same full
baseline pipeline (for consistency across results), and additionally owns deep hyperparameter
tuning for one assigned model.

## Dataset
- **Source:** [Edge-IIoTset](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot) (Kaggle)
- **File used:** `Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv`
- **Size:** ~157,000+ rows, 15 attack classes + Normal traffic
- **Download command:**
  ```bash
  kaggle datasets download -d mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot \
    -f "Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv"
  ```
- Data files are **not committed** to this repo (see `.gitignore`) — download locally or in Colab.

## Tasks
Two classification tasks, run for each of five baseline models:
- **Binary classification** — `Attack_label`
- **Multiclass classification** — `Attack_type` (15 attack types + Normal)

## Models
1. Logistic Regression
2. Random Forest
3. LightGBM
4. XGBoost
5. Simple MLP

## Evaluation Metrics
- Accuracy
- Precision (macro & weighted)
- Recall (macro & weighted)
- F1-score (macro & weighted)

## Project Structure
```
ids-fl-project/
├── data/                   # Raw/processed data (gitignored)
├── notebooks/               # Colab/Jupyter notebooks
├── src/
│   ├── preprocessing/       # Shared preprocessing pipeline
│   ├── models/               # Model training scripts
│   └── evaluation/           # Metrics & evaluation utilities
├── results/                  # Saved metrics, plots, model outputs
├── docs/                      # Notes, writeups
├── requirements.txt
└── README.md
```

## Setup
```bash
git clone <your-repo-url>
cd ids-fl-project
pip install -r requirements.txt
```

## Preprocessing Notes
- Drop non-generalizable columns (IP addresses, timestamps, etc.)
- Encode categorical features
- Scale numerical features
- Stratified train/test split (important given class imbalance across 15 attack types)
- **Shared across the team** to ensure comparable results before individual tuning

## Roadmap
- [x] Resolve correct dataset file/version
- [ ] Shared preprocessing pipeline
- [ ] Train 5 baseline models × 2 tasks (10 runs)
- [ ] Hyperparameter tuning (per-member assigned model)
- [ ] Federated learning implementation
