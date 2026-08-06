"""Small HTTP adapter for binary IDS inference and federated-demo telemetry.

The frontend talks only to this API.  Model/checkpoint, preprocessing, Docker,
and cryptographic details stay behind the adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uavids_fl import BinaryMLP, predict_proba, set_deterministic  # noqa: E402


API_VERSION = "uavids-gui-api-v1"
DEFAULT_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000")
CLIENT_IDS = tuple(f"uav-client-{index}" for index in range(1, 6))


class RequestError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenBinaryIDS:
    """Hash-checked inference wrapper around the frozen Phase 3 global model."""

    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.root = project_root
        self.lock_path = self.root / "config" / "phase3_locked_config.json"
        self.lock = read_json(self.lock_path)
        self.features = tuple(self.lock["features"])
        self.threshold = float(self.lock["model_thresholds"]["federated_fedavg"])
        self.lock_sha256 = sha256_path(self.lock_path)

        preprocessor_path = self.root.joinpath(
            *self.lock["preprocessor"]["path"].replace("\\", "/").split("/")
        )
        checkpoint_record = self.lock["model_artifacts"]["federated_fedavg"]
        checkpoint_path = self.root.joinpath(
            *checkpoint_record["path"].replace("\\", "/").split("/")
        )
        if sha256_path(preprocessor_path) != self.lock["preprocessor"]["sha256"]:
            raise RuntimeError("frozen preprocessor hash mismatch")
        if sha256_path(checkpoint_path) != checkpoint_record["sha256"]:
            raise RuntimeError("frozen federated checkpoint hash mismatch")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        candidate = self.lock["selected_candidate"]
        if checkpoint["input_dim"] != len(self.features):
            raise RuntimeError("checkpoint input dimension mismatch")
        if checkpoint["hidden_layers"] != candidate["hidden_layers"]:
            raise RuntimeError("checkpoint hidden layers mismatch")
        if checkpoint["dropout"] != candidate["dropout"]:
            raise RuntimeError("checkpoint dropout mismatch")
        if float(checkpoint["threshold"]) != self.threshold:
            raise RuntimeError("checkpoint threshold mismatch")

        set_deterministic(int(self.lock["seed"]), 1)
        self.model = BinaryMLP(len(self.features), candidate["hidden_layers"], candidate["dropout"])
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()
        self.preprocessor = joblib.load(preprocessor_path)
        self.checkpoint_sha256 = checkpoint_record["sha256"]
        self.model_id = f"federated_fedavg:{self.checkpoint_sha256[:12]}"

    def _feature_frame(self, payload: Any) -> tuple[pd.DataFrame, int]:
        if not isinstance(payload, dict):
            raise RequestError("invalid_features", "features must be a JSON object")
        expected, received = set(self.features), set(payload)
        if received != expected:
            missing = sorted(expected - received)
            unexpected = sorted(received - expected)
            detail = []
            if missing:
                detail.append(f"missing: {', '.join(missing)}")
            if unexpected:
                detail.append(f"unexpected: {', '.join(unexpected)}")
            raise RequestError("invalid_features", "; ".join(detail))
        values: dict[str, float] = {}
        missing_count = 0
        for name in self.features:
            value = payload[name]
            if value is None:
                values[name] = np.nan
                missing_count += 1
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RequestError("invalid_features", f"{name} must be numeric or null")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise RequestError("invalid_features", f"{name} must be finite")
            values[name] = numeric
        return pd.DataFrame([values], columns=self.features), missing_count

    def predict(self, request: Any) -> dict:
        if not isinstance(request, dict):
            raise RequestError("invalid_request", "request body must be a JSON object")
        allowed = {"record_id", "source", "features"}
        if set(request) - allowed:
            raise RequestError("invalid_request", "request contains unsupported fields")
        record_id = request.get("record_id")
        if record_id is None:
            record_id = str(uuid.uuid4())
        if not isinstance(record_id, str) or not 1 <= len(record_id) <= 128:
            raise RequestError("invalid_request", "record_id must be 1-128 characters")
        source = request.get("source")
        if source is not None and (not isinstance(source, str) or len(source) > 128):
            raise RequestError("invalid_request", "source must be null or at most 128 characters")

        frame, missing_count = self._feature_frame(request.get("features"))
        transformed = np.asarray(self.preprocessor.transform(frame), dtype=np.float32)
        probability = float(predict_proba(self.model, transformed, torch.device("cpu"))[0])
        is_attack = probability >= self.threshold
        return {
            "schema_version": "uavids-gui-prediction-v1",
            "prediction_id": str(uuid.uuid4()),
            "timestamp_utc": utc_now(),
            "record_id": record_id,
            "source": source,
            "label": "Attack" if is_attack else "Normal",
            "confidence": probability if is_attack else 1.0 - probability,
            "attack_probability": probability,
            "decision_threshold": self.threshold,
            "missing_features_imputed": missing_count,
            "model_id": self.model_id,
            "model_version": self.lock["design_version"],
            "inference_mode": "live_model",
        }

    def public_metadata(self) -> dict:
        validation = self.lock["validation_metrics"]["federated_fedavg"]
        return {
            "available": True,
            "model_id": self.model_id,
            "model_version": self.lock["design_version"],
            "task": "binary_intrusion_detection",
            "labels": ["Normal", "Attack"],
            "positive_class": "Attack",
            "feature_count": len(self.features),
            "features": list(self.features),
            "decision_threshold": self.threshold,
            "frozen_validation_metrics": {
                "macro_f1": validation["macro_f1"],
                "attack_precision": validation["attack_precision"],
                "attack_recall": validation["attack_recall"],
                "fpr": validation["fpr"],
            },
            "metrics_note": "Saved validation evidence; not live prediction accuracy.",
        }


class FederatedTelemetry:
    """Normalize live Phase 4/5 telemetry or use clearly marked saved evidence."""

    def __init__(self, project_root: Path, upstream_url: str | None):
        self.root = project_root
        self.upstream_url = upstream_url.rstrip("/") if upstream_url else None
        self.benchmark = read_json(self.root / "phase5" / "results" / "benchmark_comparison.json")
        self.saved_phase4_events = read_json(self.root / "phase4" / "results" / "event_excerpt.json")
        self.saved_security_events = read_json(
            self.root / "phase5" / "results" / "security_rejection_events.json"
        )

    def _upstream(self, path: str) -> dict:
        if not self.upstream_url:
            raise OSError("live federated backend not configured")
        request = urllib.request.Request(
            f"{self.upstream_url}{path}", headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=0.75) as response:
            return json.loads(response.read())

    @staticmethod
    def _client_states(status: dict, events: list[dict]) -> list[dict]:
        states = {client_id: "waiting" for client_id in status.get("expected_clients", CLIENT_IDS)}
        mapping = {
            "client_registered": "registered",
            "client_authenticated": "authenticated",
            "client_ready": "ready",
            "global_model_distributed": "model_received",
            "client_training_started": "training",
            "client_training_completed": "training_complete",
            "client_update_received": "update_received",
            "client_update_acknowledged": "round_complete",
            "client_security_summary": "complete",
            "client_protocol_error": "error",
        }
        for event in events:
            client_id = event.get("client_id")
            if client_id in states and event.get("event_type") in mapping:
                states[client_id] = mapping[event["event_type"]]
        if status.get("state") == "completed":
            states = {client_id: "complete" for client_id in states}
        return [{"client_id": client_id, "state": states[client_id]} for client_id in sorted(states)]

    def _saved_snapshot(self) -> dict:
        metrics = self.benchmark["final_validation_metrics"]
        rejected = int(self.benchmark["attack_test"]["rejections"])
        return {
            "data_mode": "replay",
            "upstream_available": False,
            "run_id": self.benchmark["secure_run_id"],
            "state": "completed",
            "current_round": int(self.benchmark["rounds"]),
            "total_rounds": int(self.benchmark["rounds"]),
            "updates_received": 5,
            "updates_expected": 5,
            "clients": [{"client_id": client_id, "state": "complete"} for client_id in CLIENT_IDS],
            "local_data_statement": "Each client trains from its own mounted partition; raw training rows are not sent to the server.",
            "global_model_metrics": {
                "macro_f1": metrics["macro_f1"],
                "attack_precision": metrics["attack_precision"],
                "attack_recall": metrics["attack_recall"],
                "fpr": metrics["fpr"],
                "source": "recorded_phase5_demo_validation_telemetry",
                "note": "Recorded validation telemetry, not live prediction accuracy.",
            },
            "security": {
                "mode": "secure",
                "status": "recorded_verified",
                "algorithms": self.benchmark["algorithms"],
                "authenticated_clients": 5,
                "rejected_messages": rejected,
                "note": "ML-KEM establishes key material; ML-DSA authenticates clients; AES-GCM protects model exchanges.",
            },
        }

    def snapshot(self) -> dict:
        try:
            status = self._upstream("/api/v1/status")
            event_response = self._upstream("/api/v1/events?after_seq=0")
            events = event_response.get("events", [])
        except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError):
            return self._saved_snapshot()

        security_status = status.get("security")
        secure = isinstance(security_status, dict)
        authenticated = len(security_status.get("authenticated_clients", [])) if secure else 0
        rejected = sum(event.get("event_type") == "security_message_rejected" for event in events)
        return {
            "data_mode": "live",
            "upstream_available": True,
            "run_id": status.get("run_id"),
            "state": status.get("state", "unknown"),
            "current_round": int(status.get("current_round", 0)),
            "total_rounds": int(status.get("total_rounds", 0)),
            "updates_received": len(status.get("received_updates", [])),
            "updates_expected": len(status.get("expected_clients", CLIENT_IDS)),
            "clients": self._client_states(status, events),
            "local_data_statement": "Each client trains from its own mounted partition; raw training rows are not sent to the server.",
            "global_model_metrics": {
                **(status.get("last_metrics") or {}),
                "source": "live_demo_validation_telemetry",
                "note": "Validation telemetry, not live prediction accuracy.",
            },
            "security": {
                "mode": "secure" if secure else "plain",
                "status": "live",
                "algorithms": self.benchmark["algorithms"] if secure else None,
                "authenticated_clients": authenticated,
                "rejected_messages": rejected,
                "note": (
                    "ML-KEM establishes key material; ML-DSA authenticates clients; AES-GCM protects model exchanges."
                    if secure
                    else "Phase 4 comparison mode uses plain application messages."
                ),
            },
        }

    def events(self, after_seq: int) -> dict:
        try:
            response = self._upstream(f"/api/v1/events?after_seq={after_seq}")
            return {
                "schema_version": "uavids-gui-federated-events-v1",
                "data_mode": "live",
                "run_id": response.get("run_id"),
                "events": response.get("events", []),
                "last_seq": int(response.get("last_seq", after_seq)),
            }
        except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError):
            selected = []
            combined = [*self.saved_phase4_events, *self.saved_security_events]
            for sequence, source in enumerate(combined, start=1):
                if sequence <= after_seq:
                    continue
                selected.append(
                    {
                        "schema_version": "uavids-gui-recorded-federated-event-v1",
                        "seq": sequence,
                        "recorded": True,
                        "run_id": self.benchmark["secure_run_id"],
                        "source": source.get("source", "control-center"),
                        "event_type": source.get("event_type"),
                        "severity": source.get("severity", "info"),
                        "round": source.get("round", 0),
                        "client_id": source.get("client_id"),
                        "payload": source.get("payload", {}),
                    }
                )
            return {
                "schema_version": "uavids-gui-federated-events-v1",
                "data_mode": "replay",
                "run_id": self.benchmark["secure_run_id"],
                "events": selected,
                "last_seq": len(combined),
            }


class GuiBackend:
    def __init__(
        self,
        project_root: Path = PROJECT_ROOT,
        *,
        upstream_url: str | None = None,
        replay_path: Path | None = None,
    ):
        self.started_utc = utc_now()
        self.ids = FrozenBinaryIDS(project_root)
        self.telemetry = FederatedTelemetry(project_root, upstream_url)
        path = replay_path or Path(__file__).resolve().parent / "examples" / "replay_records.json"
        records = read_json(path)
        if not isinstance(records, list) or not records:
            raise RuntimeError("replay fixture must contain a non-empty JSON array")
        self.replay_records = records
        self.replay_index = 0
        self._lock = threading.RLock()
        self.records_processed = 0
        self.normal_count = 0
        self.attack_count = 0
        self.latest_prediction: dict | None = None
        self.alerts: deque[dict] = deque(maxlen=20)
        self.events: deque[dict] = deque(maxlen=1000)
        self.sequence = 0

    def _record_prediction(self, prediction: dict, *, replayed: bool) -> dict:
        with self._lock:
            self.records_processed += 1
            if prediction["label"] == "Attack":
                self.attack_count += 1
            else:
                self.normal_count += 1
            prediction = {**prediction, "replayed": replayed}
            self.latest_prediction = prediction
            self.sequence += 1
            event = {
                "schema_version": "uavids-gui-event-v1",
                "seq": self.sequence,
                "timestamp_utc": prediction["timestamp_utc"],
                "event_type": "prediction_completed",
                "severity": "warning" if prediction["label"] == "Attack" else "info",
                "source": prediction["source"],
                "payload": prediction,
            }
            self.events.append(event)
            if prediction["label"] == "Attack":
                self.alerts.appendleft(prediction)
            return prediction

    def predict(self, request: Any, *, replayed: bool = False) -> dict:
        return self._record_prediction(self.ids.predict(request), replayed=replayed)

    def replay_next(self) -> dict:
        with self._lock:
            position = self.replay_index
            request = self.replay_records[position]
            self.replay_index = (position + 1) % len(self.replay_records)
        prediction = self.predict(request, replayed=True)
        return {
            "schema_version": "uavids-gui-replay-v1",
            "position": position + 1,
            "total_records": len(self.replay_records),
            "wrapped": self.replay_index == 0,
            "prediction": prediction,
        }

    def snapshot(self) -> dict:
        federated = self.telemetry.snapshot()
        with self._lock:
            inference = {
                "records_processed": self.records_processed,
                "normal_count": self.normal_count,
                "attack_count": self.attack_count,
                "latest_prediction": self.latest_prediction,
                "recent_alerts": list(self.alerts),
            }
        return {
            "schema_version": "uavids-gui-snapshot-v1",
            "generated_utc": utc_now(),
            "api_version": API_VERSION,
            "presentation_mode": federated["data_mode"],
            "backend": {
                "available": True,
                "started_utc": self.started_utc,
                "inference_mode": "live_model",
                "federated_upstream_available": federated["upstream_available"],
            },
            "model": self.ids.public_metadata(),
            "inference": inference,
            "federated": federated,
        }

    def inference_events(self, after_seq: int) -> dict:
        with self._lock:
            return {
                "schema_version": "uavids-gui-events-v1",
                "events": [event for event in self.events if event["seq"] > after_seq],
                "last_seq": self.sequence,
            }


class GuiHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, backend: GuiBackend, allowed_origins: tuple[str, ...]):
        super().__init__(address, GuiRequestHandler)
        self.backend = backend
        self.allowed_origins = allowed_origins


class GuiRequestHandler(BaseHTTPRequestHandler):
    server: GuiHTTPServer
    protocol_version = "HTTP/1.1"

    def _cors_origin(self) -> str | None:
        origin = self.headers.get("Origin")
        if origin is None:
            return None
        if "*" in self.server.allowed_origins:
            return "*"
        return origin if origin in self.server.allowed_origins else ""

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        cors = self._cors_origin()
        if cors:
            self.send_header("Access-Control-Allow-Origin", cors)
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, code: str, message: str) -> None:
        self._send(
            status,
            {"schema_version": "uavids-gui-error-v1", "error": {"code": code, "message": message}},
        )

    def _require_origin(self) -> bool:
        if self._cors_origin() == "":
            self._error(HTTPStatus.FORBIDDEN, "origin_not_allowed", "frontend origin is not allowed")
            return False
        return True

    def _json_body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise RequestError("invalid_request", "invalid Content-Length") from exc
        if length < 0 or length > 262144:
            raise RequestError("invalid_request", "request body exceeds 256 KiB")
        if length == 0:
            return {}
        try:
            value = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RequestError("invalid_json", "request body is not valid JSON") from exc
        if not isinstance(value, dict):
            raise RequestError("invalid_request", "request body must be a JSON object")
        return value

    def do_OPTIONS(self) -> None:  # noqa: N802
        if not self._require_origin():
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Content-Length", "0")
        cors = self._cors_origin()
        if cors:
            self.send_header("Access-Control-Allow-Origin", cors)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if not self._require_origin():
            return
        parsed = urlparse(self.path)
        try:
            if parsed.path in {"/health", "/api/gui/v1/health"}:
                self._send(
                    HTTPStatus.OK,
                    {
                        "schema_version": "uavids-gui-health-v1",
                        "ok": True,
                        "api_version": API_VERSION,
                        "model_available": True,
                    },
                )
            elif parsed.path == "/api/gui/v1/snapshot":
                self._send(HTTPStatus.OK, self.server.backend.snapshot())
            elif parsed.path == "/api/gui/v1/events":
                after = int(parse_qs(parsed.query).get("after_seq", ["0"])[0])
                if after < 0:
                    raise RequestError("invalid_cursor", "after_seq cannot be negative")
                self._send(HTTPStatus.OK, self.server.backend.inference_events(after))
            elif parsed.path == "/api/gui/v1/federated/events":
                after = int(parse_qs(parsed.query).get("after_seq", ["0"])[0])
                if after < 0:
                    raise RequestError("invalid_cursor", "after_seq cannot be negative")
                self._send(HTTPStatus.OK, self.server.backend.telemetry.events(after))
            else:
                self._error(HTTPStatus.NOT_FOUND, "not_found", "endpoint not found")
        except (RequestError, ValueError) as exc:
            code = exc.code if isinstance(exc, RequestError) else "invalid_cursor"
            self._error(HTTPStatus.BAD_REQUEST, code, str(exc))
        except Exception:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", "request failed safely")

    def do_POST(self) -> None:  # noqa: N802
        if not self._require_origin():
            return
        try:
            payload = self._json_body()
            if self.path == "/api/gui/v1/predictions":
                result = self.server.backend.predict(payload)
            elif self.path == "/api/gui/v1/replay/next":
                if payload:
                    raise RequestError("invalid_request", "replay request body must be empty")
                result = self.server.backend.replay_next()
            else:
                self._error(HTTPStatus.NOT_FOUND, "not_found", "endpoint not found")
                return
            self._send(HTTPStatus.OK, result)
        except RequestError as exc:
            self._error(HTTPStatus.BAD_REQUEST, exc.code, str(exc))
        except Exception:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", "request failed safely")

    def log_message(self, format_string: str, *args) -> None:
        return


def create_server(
    backend: GuiBackend,
    host: str,
    port: int,
    allowed_origins: tuple[str, ...],
) -> GuiHTTPServer:
    return GuiHTTPServer((host, port), backend, allowed_origins)


def main() -> None:
    parser = argparse.ArgumentParser(description="UAVIDS stable GUI integration backend")
    parser.add_argument("--host", default=os.environ.get("GUI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GUI_PORT", "8090")))
    parser.add_argument(
        "--allowed-origins",
        default=os.environ.get("GUI_ALLOWED_ORIGINS", ",".join(DEFAULT_ORIGINS)),
    )
    parser.add_argument(
        "--federated-backend",
        default=os.environ.get("FEDERATED_BACKEND_URL", ""),
        help="Optional Phase 4/5 URL, normally http://127.0.0.1:8080",
    )
    args = parser.parse_args()
    origins = tuple(value.strip() for value in args.allowed_origins.split(",") if value.strip())
    if not origins:
        raise SystemExit("at least one GUI origin must be configured")
    backend = GuiBackend(upstream_url=args.federated_backend or None)
    server = create_server(backend, args.host, args.port, origins)
    print(
        json.dumps(
            {
                "event": "gui_backend_ready",
                "url": f"http://{args.host}:{server.server_address[1]}",
                "allowed_origins": origins,
                "federated_backend": args.federated_backend or None,
                "model_id": backend.ids.model_id,
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
