from __future__ import annotations

import torch
from torch.nn import functional as F

from uavids_closr.loss import CLOSRLoss
from uavids_closr.model import CLOSRModel


def _pinned_official_closr_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Literal numerical fixture for upstream closr_loss.py at 79da4d1f... .

    The source file SHA-256 is
    b07464f66dace6e7f32767f79ea1470e31a9ca4c62b06332d1d3fa0d47541035.
    """
    normalized = F.normalize(embeddings, p=2, dim=-1)
    by_head = normalized.transpose(0, 1)
    distances = (1.0 - torch.bmm(by_head, by_head.transpose(1, 2))) / 2.0
    head_losses = []
    for class_index in range(embeddings.shape[1]):
        equal_pairs = labels.unsqueeze(0).eq(labels.unsqueeze(1))
        similar_mask = torch.logical_and(
            equal_pairs, labels.unsqueeze(0) == class_index
        )
        dissimilar_mask = torch.logical_and(
            ~equal_pairs, labels.unsqueeze(1) == class_index
        )
        similar_distances = distances[class_index] * similar_mask
        dissimilar_distances = torch.where(
            dissimilar_mask,
            distances[class_index],
            torch.full_like(distances[class_index], margin),
        )
        n_similar = torch.sum(torch.greater(similar_distances, eps).float())
        similar_loss = torch.sum(similar_distances.square()) / (n_similar + eps)
        dissimilar_loss = F.relu(margin - dissimilar_distances).square()
        n_dissimilar = torch.sum(torch.greater(dissimilar_loss, eps).float())
        dissimilar_loss = torch.sum(dissimilar_loss) / (n_dissimilar + eps)
        head_losses.append(0.5 * similar_loss + 0.5 * dissimilar_loss)
    return torch.mean(torch.stack(head_losses))


def _assert_loss_and_gradient_parity(labels: list[int], num_classes: int) -> CLOSRLoss:
    torch.manual_seed(101 + len(labels) + num_classes)
    candidate = torch.randn(
        len(labels), num_classes, 5, dtype=torch.float64, requires_grad=True
    )
    reference = candidate.detach().clone().requires_grad_(True)
    encoded_labels = torch.tensor(labels, dtype=torch.long)
    criterion = CLOSRLoss(num_classes=num_classes, epsilon=1e-6)
    actual_loss = criterion(candidate, encoded_labels)
    expected_loss = _pinned_official_closr_loss(reference, encoded_labels)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-10, atol=1e-12)
    actual_loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(candidate.grad, reference.grad, rtol=1e-9, atol=1e-11)
    return criterion


def test_pinned_upstream_parity_for_ordinary_batch() -> None:
    criterion = _assert_loss_and_gradient_parity([0, 0, 1, 1, 2, 2], 3)
    assert all(item.positive_available and item.negative_available for item in criterion.last_diagnostics)


def test_one_target_plus_negatives_keeps_negative_only_signal() -> None:
    criterion = _assert_loss_and_gradient_parity([0, 1, 1], 2)
    first = criterion.last_diagnostics[0]
    assert not first.positive_available
    assert first.negative_available
    assert first.active


def test_all_target_batch_keeps_positive_only_signal() -> None:
    criterion = _assert_loss_and_gradient_parity([0, 0, 0], 2)
    first, absent = criterion.last_diagnostics
    assert first.positive_available
    assert not first.negative_available
    assert first.active
    assert not absent.active


def test_absent_head_is_finite_and_receives_zero_gradient() -> None:
    torch.manual_seed(7)
    model = CLOSRModel(4, [8], 3, 3, dropout=0.0)
    criterion = CLOSRLoss(num_classes=3)
    features = torch.randn(4, 4)
    labels = torch.tensor([0, 0, 1, 1])

    loss = criterion(model(features), labels)
    assert torch.isfinite(loss)
    loss.backward()

    assert [item.active for item in criterion.last_diagnostics] == [True, True, False]
    absent_gradient = model.projection_heads[2].weight.grad
    assert absent_gradient is not None
    assert torch.count_nonzero(absent_gradient) == 0
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.encoder.parameters()
    )


def test_singleton_batch_is_a_differentiable_zero_not_nan() -> None:
    torch.manual_seed(8)
    model = CLOSRModel(4, [8], 3, 3, dropout=0.0)
    criterion = CLOSRLoss(num_classes=3)
    loss = criterion(model(torch.randn(1, 4)), torch.tensor([0]))
    assert torch.isfinite(loss)
    assert float(loss.item()) == 0.0
    assert not any(item.active for item in criterion.last_diagnostics)
    loss.backward()
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in model.parameters()
    )
    _assert_loss_and_gradient_parity([0], 3)


def test_model_outputs_one_unit_vector_per_class() -> None:
    model = CLOSRModel(5, [7, 7], 4, 3, dropout=0.0)
    output = model(torch.randn(6, 5))
    assert output.shape == (6, 3, 4)
    torch.testing.assert_close(torch.linalg.vector_norm(output, dim=-1), torch.ones(6, 3))
