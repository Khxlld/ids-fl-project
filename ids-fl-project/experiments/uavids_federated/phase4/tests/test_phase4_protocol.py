from __future__ import annotations

import base64
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from phase4.app.common import ProtocolError, decode_state, encode_state, state_spec
from phase4.app.server import Coordinator
from uavids_fl import BinaryMLP, fedavg


def test_update_round_trip_and_sample_weighted_aggregation_agree():
    first = BinaryMLP(2, [2], [0.0])
    second = BinaryMLP(2, [2], [0.0])
    with torch.no_grad():
        for value in first.parameters():
            value.fill_(0.0)
        for value in second.parameters():
            value.fill_(2.0)
    specification = state_spec(first.state_dict())
    first_encoded, _, _ = encode_state(first.state_dict())
    second_encoded, _, _ = encode_state(second.state_dict())
    first_state, _, _ = decode_state(first_encoded, specification, 1_000_000)
    second_state, _, _ = decode_state(second_encoded, specification, 1_000_000)

    actual = fedavg([first_state, second_state], [1, 3])
    assert all(torch.equal(value, torch.full_like(value, 1.5)) for value in actual.values())


def test_incompatible_or_nonfinite_update_is_rejected():
    model = BinaryMLP(2, [2], [0.0])
    specification = state_spec(model.state_dict())
    wrong = BinaryMLP(3, [2], [0.0])
    encoded, _, _ = encode_state(wrong.state_dict())
    with pytest.raises(ProtocolError, match="shape"):
        decode_state(encoded, specification, 1_000_000)

    bad_state = model.state_dict()
    bad_state[next(iter(bad_state))][0, 0] = float("nan")
    encoded, _, _ = encode_state(bad_state)
    with pytest.raises(ProtocolError, match="non-finite"):
        decode_state(encoded, specification, 1_000_000)


def test_malformed_and_oversized_payloads_are_rejected():
    model = BinaryMLP(2, [2], [0.0])
    specification = state_spec(model.state_dict())
    with pytest.raises(ProtocolError, match="base64"):
        decode_state("not base64!", specification, 1_000_000)
    oversized = base64.b64encode(b"x" * 100).decode("ascii")
    with pytest.raises(ProtocolError, match="maximum"):
        decode_state(oversized, specification, 50)


def test_unavailable_client_timeout_names_every_missing_client():
    coordinator = Coordinator.__new__(Coordinator)
    coordinator._lock = threading.RLock()
    coordinator.state = "waiting_for_clients"
    coordinator.startup_deadline_perf = 0.0
    coordinator.clients = {"uav-client-1": {}, "uav-client-2": {}}
    coordinator.registered = {"uav-client-1": {}}
    captured = {}
    coordinator._fail = lambda reason, missing: captured.update(reason=reason, missing=missing)

    coordinator.check_timeouts()
    assert captured == {"reason": "startup_timeout", "missing": ["uav-client-2"]}
