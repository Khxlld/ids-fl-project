"""Explicit sample-weighted FedAvg helpers."""

from __future__ import annotations

from collections import OrderedDict

import torch


def clone_state_dict(model_or_state) -> OrderedDict[str, torch.Tensor]:
    state = model_or_state.state_dict() if hasattr(model_or_state, "state_dict") else model_or_state
    return OrderedDict((name, value.detach().cpu().clone()) for name, value in state.items())


def sample_weighted_fedavg(
    states: list[OrderedDict[str, torch.Tensor]], sample_counts: list[int]
) -> OrderedDict[str, torch.Tensor]:
    if not states or len(states) != len(sample_counts) or any(count <= 0 for count in sample_counts):
        raise ValueError("matching non-empty states and positive sample counts are required")
    keys = list(states[0])
    if any(list(state) != keys for state in states):
        raise ValueError("state dictionaries do not share identical keys")
    total = float(sum(sample_counts))
    averaged: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in keys:
        reference = states[0][key]
        if reference.is_floating_point():
            accumulator = torch.zeros_like(reference, dtype=torch.float64)
            for state, count in zip(states, sample_counts):
                accumulator += state[key].to(dtype=torch.float64) * (count / total)
            averaged[key] = accumulator.to(dtype=reference.dtype)
        else:
            if any(not torch.equal(state[key], reference) for state in states[1:]):
                raise ValueError(f"non-floating buffer differs across clients: {key}")
            averaged[key] = reference.clone()
    return averaged
