"""Central configuration for the Edge-IIoT federated intrusion-detection pipeline.

Every tunable knob lives here so it is easy to find and change. The dataset path
is defined exactly once (with an environment-variable override) and never
hard-coded elsewhere in the source tree.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# PROJECT_ROOT points at the `edgeiiot_federated/` experiment root, regardless
# of where a notebook or script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _find_data_root(project_root: Path) -> Path:
    """Return the nearest project/ancestor containing the shared data folder."""
    for candidate in (project_root, *project_root.parents):
        if (candidate / "data").is_dir():
            return candidate
    return project_root


REPO_ROOT = _find_data_root(PROJECT_ROOT)

# The dataset path is defined ONCE here. Override with the EDGEIIOT_DATASET env
# variable if the CSV lives somewhere else.
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "raw" / "DNN-EdgeIIoT-dataset.csv"
DATASET_PATH = Path(os.environ.get("EDGEIIOT_DATASET", str(DEFAULT_DATASET_PATH)))

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"

# Artifact file locations (created on demand).
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.json"
DATA_SUMMARY_PATH = ARTIFACTS_DIR / "data_summary.json"
DROPPED_COLUMNS_PATH = ARTIFACTS_DIR / "dropped_columns.csv"
CENTRALIZED_MODEL_PATH = ARTIFACTS_DIR / "centralized_model.pt"
GLOBAL_MODEL_PATH = ARTIFACTS_DIR / "global_model.pt"

# Result file locations.
FED_ROUND_METRICS_PATH = RESULTS_DIR / "federated_round_metrics.csv"
CLIENT_METRICS_PATH = RESULTS_DIR / "client_metrics.csv"
FINAL_FED_METRICS_PATH = RESULTS_DIR / "final_federated_metrics.json"
FINAL_CENTRALIZED_METRICS_PATH = RESULTS_DIR / "final_centralized_metrics.json"
CLIENT_DISTRIBUTIONS_PATH = RESULTS_DIR / "client_distributions.csv"

# Human-readable class names for the binary task.
CLASS_NAMES: Dict[int, str] = {0: "Normal", 1: "Attack"}

# Target / leakage columns.
TARGET_COLUMN = "Attack_label"
MULTICLASS_COLUMN = "Attack_type"  # dropped from features (leaks the answer).


@dataclass
class Config:
    """All hyperparameters and experiment settings in one place."""

    # Reproducibility
    random_seed: int = 42

    # Sampling: take ~10% of the cleaned dataset, balanced 50/50 Normal/Attack.
    dataset_fraction: float = 0.10
    normal_attack_ratio: float = 0.50  # fraction of the sample that is Normal

    # Splits (of the balanced subset). Train pool is the remainder.
    validation_fraction: float = 0.15
    test_fraction: float = 0.15

    # Federated setup
    num_clients: int = 5
    client_names: List[str] = field(
        default_factory=lambda: [
            "Factory_A",
            "Factory_B",
            "Factory_C",
            "Factory_D",
            "Factory_E",
        ]
    )

    # Model architecture (shared by centralized baseline and all clients).
    hidden_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout_rates: List[float] = field(default_factory=lambda: [0.20, 0.10])

    # Optimisation
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    batch_size: int = 256

    # Centralized training
    centralized_epochs: int = 15
    early_stopping_patience: int = 5  # epochs w/o val-F1 improvement -> stop

    # Federated training
    local_epochs: int = 2
    federated_rounds: int = 10
    fraction_fit: float = 1.0  # all clients train every round
    fraction_evaluate: float = 0.0  # server-side central evaluation instead

    # Decision threshold for turning probability into a 0/1 label.
    decision_threshold: float = 0.5

    # SMOTE is OFF by default: the subset is already balanced 50/50.
    use_smote: bool = False

    # Categorical columns with cardinality above this are treated as
    # high-cardinality / identifier-like and dropped instead of one-hot encoded.
    max_categorical_cardinality: int = 50

    def to_dict(self) -> dict:
        return asdict(self)


# A single shared instance used across the pipeline.
CONFIG = Config()


def ensure_dirs() -> None:
    """Create output directories if they do not yet exist."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
