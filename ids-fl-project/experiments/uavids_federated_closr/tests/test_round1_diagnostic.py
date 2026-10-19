from __future__ import annotations

import random

import numpy as np
import torch

from run_round1_diagnostic import instrumented_train_epoch
from uavids_closr.federated import clone_state_dict
from uavids_closr.model import CLOSRModel
from uavids_closr.training import make_criterion, set_deterministic, train_epoch


def _run(instrumented: bool):
    seed = 816
    set_deterministic(seed, 1)
    features = np.random.default_rng(4).normal(size=(16, 3)).astype(np.float32)
    labels = np.asarray([0, 1] * 8, dtype=np.int64)
    model = CLOSRModel(3, [7], 4, 2, dropout=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = make_criterion(
        {
            "margin": 1.0,
            "squared": True,
            "alpha": 0.5,
            "normalization": "official_active_pairs",
            "head_reduction": "mean_all",
            "epsilon": 1e-6,
        },
        2,
    )
    if instrumented:
        metrics, gradients = instrumented_train_epoch(
            model,
            features,
            labels,
            optimizer,
            criterion,
            torch.device("cpu"),
            4,
            seed + 1,
            True,
            ["benign", "attack"],
        )
    else:
        metrics = train_epoch(
            model,
            features,
            labels,
            optimizer,
            criterion,
            torch.device("cpu"),
            4,
            seed + 1,
            True,
        )
        gradients = []
    return {
        "state": clone_state_dict(model),
        "metrics": {key: value for key, value in metrics.items() if key != "runtime_seconds"},
        "gradients": gradients,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state().clone(),
    }


def test_instrumentation_preserves_parameters_metrics_and_rng_state() -> None:
    plain = _run(False)
    observed = _run(True)

    assert plain["metrics"] == observed["metrics"]
    assert all(torch.equal(plain["state"][name], observed["state"][name]) for name in plain["state"])
    assert plain["python_rng"] == observed["python_rng"]
    assert plain["numpy_rng"][0] == observed["numpy_rng"][0]
    assert np.array_equal(plain["numpy_rng"][1], observed["numpy_rng"][1])
    assert plain["numpy_rng"][2:] == observed["numpy_rng"][2:]
    assert torch.equal(plain["torch_rng"], observed["torch_rng"])
    assert len(observed["gradients"]) == 3
    assert {row["parameter_group"] for row in observed["gradients"]} == {
        "encoder",
        "head::benign",
        "head::attack",
    }
