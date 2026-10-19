from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch

from uavids_closr.artifacts import load_model_bundle, save_model_bundle
from uavids_closr.federated import sample_weighted_fedavg
from uavids_closr.inference import (
    aggregate_center_statistics,
    calibrate_known_threshold,
    calibration_metrics,
    centers_from_statistics,
    local_center_statistics,
    open_set_metrics,
    similarities_and_unknown_scores,
)
from uavids_closr.model import CLOSRModel


def test_score_direction_rejects_orthogonal_embedding() -> None:
    centers = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    embeddings = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        ]
    )
    _, scores = similarities_and_unknown_scores(embeddings, centers)
    assert scores[1] > scores[0]
    threshold = -0.2
    assert not bool(scores[0] > threshold)
    assert bool(scores[1] > threshold)


def test_calibration_uses_strict_greater_rule_and_respects_ties() -> None:
    unique_scores = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    threshold, observed = calibrate_known_threshold(unique_scores, 0.4)
    assert threshold == 2.0
    assert observed == 0.4
    assert int(np.sum(unique_scores > threshold)) == 2

    tied_scores = np.asarray([0.0, 1.0, 1.0, 1.0, 2.0])
    threshold, observed = calibrate_known_threshold(tied_scores, 0.4)
    assert threshold == 1.0
    assert observed == 0.2
    assert int(np.sum(tied_scores > threshold)) == 1

    small_scores = np.asarray([0.0, 1.0, 2.0])
    threshold, observed = calibrate_known_threshold(small_scores, 0.05)
    assert threshold == 2.0
    assert observed == 0.0


def test_calibration_reports_rejection_and_benign_alerts_separately() -> None:
    similarities = np.asarray(
        [[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.8, 0.2]], dtype=np.float64
    )
    scores = np.asarray([-0.8, -0.1, -0.7, -0.2], dtype=np.float64)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    metrics = calibration_metrics(
        similarities, scores, labels, threshold=-0.15, class_names=["benign", "attack"]
    )
    assert metrics["known_rejection_rate_overall"] == 0.25
    assert metrics["per_class"]["benign"]["known_rejection_rate"] == 0.5
    assert metrics["per_class"]["attack"]["known_rejection_rate"] == 0.0
    assert metrics["benign_rejection_rate"] == 0.5
    assert metrics["benign_closed_set_attack_prediction_rate"] == 0.0
    assert metrics["benign_total_alert_fpr"] == 0.5


def test_open_set_metrics_report_known_and_benign_rejection_separately() -> None:
    similarities = np.asarray(
        [[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.2, 0.3]], dtype=np.float64
    )
    scores = np.asarray([-0.8, -0.1, -0.7, -0.05], dtype=np.float64)
    labels = np.asarray([0, 0, 1, -1], dtype=np.int64)
    metrics = open_set_metrics(
        similarities,
        scores,
        labels,
        threshold=-0.15,
        num_known_classes=2,
        class_names=["benign", "attack"],
    )
    assert metrics["known_rejection_rate_overall"] == 1 / 3
    assert metrics["per_class"]["benign"]["known_rejection_rate"] == 0.5
    assert metrics["per_class"]["attack"]["known_rejection_rate"] == 0.0
    assert metrics["benign_rejection_rate"] == 0.5
    assert metrics["benign_total_alert_fpr"] == 0.5
    assert metrics["unknown_recall"] == 1.0


def test_federated_center_sufficient_statistics_equal_pooled_fixed_model() -> None:
    torch.manual_seed(11)
    model = CLOSRModel(3, [6], 4, 2, dropout=0.0)
    features = np.random.default_rng(11).normal(size=(8, 3)).astype(np.float32)
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    device = torch.device("cpu")

    first = local_center_statistics(model, features[:3], labels[:3], 2, device)
    second = local_center_statistics(model, features[3:], labels[3:], 2, device)
    federated = centers_from_statistics(aggregate_center_statistics([first, second]))
    pooled = centers_from_statistics(local_center_statistics(model, features, labels, 2, device))
    torch.testing.assert_close(federated, pooled)


def test_fedavg_uses_exact_sample_weights() -> None:
    states = [
        OrderedDict(weight=torch.tensor([1.0, 3.0])),
        OrderedDict(weight=torch.tensor([5.0, 7.0])),
    ]
    averaged = sample_weighted_fedavg(states, [1, 3])
    torch.testing.assert_close(averaged["weight"], torch.tensor([4.0, 6.0]))


def test_model_bundle_round_trip(tmp_path) -> None:
    torch.manual_seed(12)
    model = CLOSRModel(3, [5], 2, 2, dropout=0.0).eval()
    centers = torch.nn.functional.normalize(torch.randn(2, 2), dim=-1)
    path = tmp_path / "bundle.pt"
    save_model_bundle(
        path,
        model,
        centers,
        -0.25,
        ["normal", "attack"],
        ["a", "b", "c"],
        {"final_test_loaded": False},
    )
    loaded, loaded_centers, threshold, metadata = load_model_bundle(path)
    values = torch.randn(4, 3)
    torch.testing.assert_close(model(values), loaded(values))
    torch.testing.assert_close(centers, loaded_centers)
    assert threshold == -0.25
    assert metadata["metadata"]["final_test_loaded"] is False
