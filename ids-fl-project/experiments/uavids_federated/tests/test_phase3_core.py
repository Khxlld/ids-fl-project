import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uavids_fl import BinaryMLP, fedavg, metric_bundle, select_threshold


def test_metric_bundle_confusion_and_macro_f1():
    metrics = metric_bundle(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.7, 0.8, 0.4]),
        threshold=0.5,
    )
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (1, 1, 1, 1)
    assert metrics["accuracy"] == 0.5
    assert metrics["macro_f1"] == 0.5
    assert metrics["fpr"] == 0.5


def test_threshold_selection_is_deterministic():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    first = select_threshold(y, p, 0.1, 0.9, 0.1)
    second = select_threshold(y, p, 0.1, 0.9, 0.1)
    assert first == second
    assert first[1]["macro_f1"] == 1.0


def test_fedavg_is_sample_weighted():
    first = BinaryMLP(2, [2], [0.0])
    second = BinaryMLP(2, [2], [0.0])
    with torch.no_grad():
        for value in first.parameters():
            value.fill_(0.0)
        for value in second.parameters():
            value.fill_(2.0)
    averaged = fedavg([first.state_dict(), second.state_dict()], [1, 3])
    assert all(torch.allclose(value, torch.full_like(value, 1.5)) for value in averaged.values())
