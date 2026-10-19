from __future__ import annotations

import numpy as np
import torch

import uavids_closr.training as training_module
from run_experiment import _period_effort
from uavids_closr.inference import centers_from_statistics
from uavids_closr.model import CLOSRModel
from uavids_closr.training import fit_centralized


def _state_snapshot(model: torch.nn.Module) -> tuple[torch.Tensor, ...]:
    return tuple(value.detach().cpu().clone() for value in model.state_dict().values())


def _same_state(left: tuple[torch.Tensor, ...], right: tuple[torch.Tensor, ...]) -> bool:
    return all(torch.equal(a, b) for a, b in zip(left, right))


def test_candidate_and_selected_centers_use_exact_eval_model_and_training_only(monkeypatch) -> None:
    rng = np.random.default_rng(71)
    train_features = rng.normal(size=(12, 3)).astype(np.float32)
    train_labels = np.asarray([0, 1] * 6, dtype=np.int64)
    selection_features = rng.normal(size=(6, 3)).astype(np.float32)
    selection_labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    calibration_features = rng.normal(size=(6, 3)).astype(np.float32)
    calibration_labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    config = {
        "known_classes": ["benign", "attack"],
        "benign_class": "benign",
        "loss": {
            "margin": 1.0,
            "squared": True,
            "alpha": 0.5,
            "normalization": "official_active_pairs",
            "head_reduction": "mean_all",
            "epsilon": 1e-6,
        },
        "training": {
            "device": "cpu",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "centralized_epochs": 2,
            "centralized_patience": 5,
            "batch_size": 6,
            "seed": 71,
            "balanced_sampling": True,
        },
        "threshold_calibration": {"known_rejection_rate_target": 0.2},
    }
    model = CLOSRModel(3, [6], 4, 2, dropout=0.2)
    original_center_statistics = training_module.local_center_statistics
    original_score_features = training_module.score_features
    center_states: list[tuple[torch.Tensor, ...]] = []
    scoring_states: list[tuple[torch.Tensor, ...]] = []

    def audited_center_statistics(candidate, features, labels, num_classes, device):
        assert candidate.training is False
        assert features is train_features
        assert labels is train_labels
        center_states.append(_state_snapshot(candidate))
        return original_center_statistics(candidate, features, labels, num_classes, device)

    def audited_score_features(candidate, centers, features, device):
        assert candidate.training is False
        assert features is selection_features or features is calibration_features
        scoring_states.append(_state_snapshot(candidate))
        return original_score_features(candidate, centers, features, device)

    monkeypatch.setattr(training_module, "local_center_statistics", audited_center_statistics)
    monkeypatch.setattr(training_module, "score_features", audited_score_features)
    result = fit_centralized(
        model,
        train_features,
        train_labels,
        selection_features,
        selection_labels,
        calibration_features,
        calibration_labels,
        config,
    )

    assert len(center_states) == 4  # initial diagnostic, two candidates, selected recomputation
    assert len(scoring_states) == 6  # initial/candidates, final selection, calibration, post-calibration selection
    assert _same_state(center_states[0], scoring_states[0])
    assert _same_state(center_states[1], scoring_states[1])
    assert _same_state(center_states[2], scoring_states[2])
    assert _same_state(center_states[3], scoring_states[3])
    assert _same_state(center_states[3], scoring_states[4])
    assert _same_state(center_states[3], scoring_states[5])
    final_statistics = original_center_statistics(
        result.model, train_features, train_labels, 2, torch.device("cpu")
    )
    torch.testing.assert_close(result.centers, centers_from_statistics(final_statistics))


def test_period_effort_aligns_signal_and_parameter_path_to_the_same_records() -> None:
    def record(step: int) -> dict:
        return {
            "examples_drawn": 10 * step,
            "optimizer_steps": step,
            "optimized_rows": 10 * step,
            "runtime_seconds": 0.5 * step,
            "drawn_class_counts": [4 * step, 6 * step],
            "per_head_signal": [
                {
                    "active_batches": step,
                    "positive_available_batches": step,
                    "negative_available_batches": step,
                    "target_examples_drawn": 4 * step,
                    "positive_pairs": 2 * step,
                    "negative_pairs": 3 * step,
                    "positive_active_pairs": step,
                    "negative_active_pairs": step,
                },
                {
                    "active_batches": 2 * step,
                    "positive_available_batches": step,
                    "negative_available_batches": step,
                    "target_examples_drawn": 6 * step,
                    "positive_pairs": 3 * step,
                    "negative_pairs": 2 * step,
                    "positive_active_pairs": step,
                    "negative_active_pairs": step,
                },
            ],
            "projection_head_parameter_delta_l2": [0.1 * step, 0.2 * step],
        }

    selected = _period_effort([record(1)], num_classes=2)
    completed = _period_effort([record(1), record(2)], num_classes=2)

    assert selected["examples_drawn"] == 10
    assert completed["examples_drawn"] == 30
    assert selected["per_head_training_signal"][0]["target_examples_drawn"] == 4
    assert completed["per_head_training_signal"][0]["target_examples_drawn"] == 12
    assert selected["per_head_training_signal"][0]["cumulative_parameter_path_l2"] == 0.1
    assert completed["per_head_training_signal"][0]["cumulative_parameter_path_l2"] == 0.30000000000000004
