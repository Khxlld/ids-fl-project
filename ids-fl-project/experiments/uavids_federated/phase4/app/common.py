"""Shared protocol, compatibility, serialization, and HTTP helpers."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


class ProtocolError(ValueError):
    """A safely rejectable protocol or compatibility error."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def build_contract(lock: dict, demo: dict) -> dict:
    """Return the cross-container contract derived only from frozen policy/config."""
    candidate = lock["selected_candidate"]
    contract = {
        "protocol_version": demo["protocol_version"],
        "config_version": demo["config_version"],
        "phase3_design_version": lock["design_version"],
        "phase3_lock_sha256": demo["phase3_lock_sha256"],
        "feature_order": lock["features"],
        "preprocessor_sha256": lock["preprocessor"]["sha256"],
        "initial_checkpoint_sha256": lock["model_artifacts"]["federated_fedavg"]["sha256"],
        "input_dim": len(lock["features"]),
        "hidden_layers": candidate["hidden_layers"],
        "dropout": candidate["dropout"],
        "loss_policy": candidate["loss_policy"],
        "optimizer": lock["optimizer"],
        "batch_size": lock["batch_size"],
        "local_epochs": demo["local_epochs"],
        "aggregation": lock["federated_training"]["aggregation_weight"],
        "decision_threshold": lock["model_thresholds"]["federated_fedavg"],
    }
    return {**contract, "contract_hash": canonical_sha256(contract)}


def state_spec(state: Mapping[str, torch.Tensor]) -> list[dict]:
    return [
        {"name": name, "shape": list(value.shape), "dtype": "float32"}
        for name, value in state.items()
    ]


def encode_state(state: Mapping[str, torch.Tensor]) -> tuple[str, int, str]:
    arrays = {
        name: value.detach().cpu().numpy().astype(np.float32, copy=False)
        for name, value in state.items()
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    raw = buffer.getvalue()
    return base64.b64encode(raw).decode("ascii"), len(raw), sha256_bytes(raw)


def decode_state(
    encoded: str,
    expected_spec: list[dict],
    maximum_bytes: int,
) -> tuple[OrderedDict[str, torch.Tensor], bytes, str]:
    if not isinstance(encoded, str):
        raise ProtocolError("weights must be base64 text")
    if len(encoded) > ((maximum_bytes + 2) // 3) * 4 + 8:
        raise ProtocolError("encoded update exceeds maximum size")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ProtocolError("weights are not valid base64") from exc
    if len(raw) > maximum_bytes:
        raise ProtocolError("decoded update exceeds maximum size")

    expected_names = [item["name"] for item in expected_spec]
    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if archive.files != expected_names:
                raise ProtocolError("model parameter names/order are incompatible")
            for item in expected_spec:
                array = archive[item["name"]]
                if list(array.shape) != item["shape"]:
                    raise ProtocolError(f"incompatible shape for {item['name']}")
                if array.dtype != np.dtype(item["dtype"]):
                    raise ProtocolError(f"incompatible dtype for {item['name']}")
                if not np.isfinite(array).all():
                    raise ProtocolError(f"non-finite values in {item['name']}")
                state[item["name"]] = torch.from_numpy(array.copy())
    except ProtocolError:
        raise
    except Exception as exc:
        raise ProtocolError("weights are not a valid NumPy archive") from exc
    return state, raw, sha256_bytes(raw)


def json_request(
    method: str,
    url: str,
    payload: dict | None = None,
    timeout: float = 30.0,
) -> tuple[dict, int]:
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return json.loads(raw) if raw else {}, int(response.status)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        detail = raw.decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
            detail = parsed.get("error", detail)
        except json.JSONDecodeError:
            pass
        raise ProtocolError(f"HTTP {exc.code}: {detail}") from exc
