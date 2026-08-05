"""HTTP control center for the five-container Phase 4 FedAvg demonstration."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
import pandas as pd
import torch

from uavids_fl import BinaryMLP, clone_state_dict, fedavg, metric_bundle, predict_proba, set_deterministic

from .common import (
    ProtocolError,
    build_contract,
    decode_state,
    encode_state,
    read_json,
    sha256_path,
    state_spec,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class Coordinator:
    """Thread-safe run state and protocol validation for one demo execution."""

    CLIENT_EVENT_TYPES = {
        "client_ready",
        "client_training_started",
        "client_training_completed",
        "client_update_acknowledged",
        "client_protocol_error",
    }

    def __init__(self) -> None:
        self.started_perf = time.perf_counter()
        self.started_utc = utc_now()
        self.run_id = str(uuid.uuid4())
        self._lock = threading.RLock()
        self.events: list[dict] = []
        self.sequence = 0
        self.state = "waiting_for_clients"
        self.failure: str | None = None
        self.current_round = 0
        self.round_started_perf: float | None = None
        self.round_deadline_perf: float | None = None
        self.registered: dict[str, dict] = {}
        self.delivered: set[tuple[int, str]] = set()
        self.updates: dict[str, dict] = {}
        self.round_summaries: list[dict] = []
        self.last_metrics: dict | None = None

        self.demo = read_json(os.environ["DEMO_CONFIG_PATH"])
        self.locked = read_json(os.environ["PHASE3_LOCK_PATH"])
        if sha256_path(os.environ["PHASE3_LOCK_PATH"]) != self.demo["phase3_lock_sha256"]:
            raise RuntimeError("Phase 3 lock hash does not match the Phase 4 demo contract")
        self.contract = build_contract(self.locked, self.demo)
        self.clients = {item["client_id"]: item for item in self.demo["clients"]}
        if len(self.clients) != 5:
            raise RuntimeError("Phase 4 requires exactly five distinct clients")

        preprocessor_path = Path(os.environ["PREPROCESSOR_PATH"])
        if sha256_path(preprocessor_path) != self.locked["preprocessor"]["sha256"]:
            raise RuntimeError("preprocessor hash mismatch")
        checkpoint_path = Path(os.environ["INITIAL_CHECKPOINT_PATH"])
        checkpoint_record = self.locked["model_artifacts"]["federated_fedavg"]
        if sha256_path(checkpoint_path) != checkpoint_record["sha256"]:
            raise RuntimeError("initial checkpoint hash mismatch")

        candidate = self.locked["selected_candidate"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["input_dim"] != len(self.locked["features"]):
            raise RuntimeError("checkpoint input dimension is incompatible")
        if checkpoint["hidden_layers"] != candidate["hidden_layers"] or checkpoint["dropout"] != candidate["dropout"]:
            raise RuntimeError("checkpoint architecture is incompatible")
        if float(checkpoint["threshold"]) != float(self.contract["decision_threshold"]):
            raise RuntimeError("checkpoint threshold is incompatible")

        set_deterministic(int(self.locked["seed"]), 1)
        self.model = BinaryMLP(
            len(self.locked["features"]),
            candidate["hidden_layers"],
            candidate["dropout"],
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.global_state = clone_state_dict(self.model)
        self.expected_spec = state_spec(self.global_state)
        self._refresh_model_payload()

        validation_path = Path(os.environ["VALIDATION_PATH"])
        validation_cfg = self.demo["validation"]
        if sha256_path(validation_path) != validation_cfg["sha256"]:
            raise RuntimeError("validation partition hash mismatch")
        validation = pd.read_csv(validation_path)
        expected_columns = self.locked["features"] + ["original_label", "binary_label"]
        if validation.columns.tolist() != expected_columns or len(validation) != validation_cfg["rows"]:
            raise RuntimeError("validation partition schema/row count mismatch")
        preprocessor = joblib.load(preprocessor_path)
        self.validation_X = np.ascontiguousarray(
            preprocessor.transform(validation[self.locked["features"]]), dtype=np.float32
        )
        self.validation_y = validation["binary_label"].to_numpy(dtype=np.int64)

        self.runtime_root = Path(os.environ.get("RUNTIME_DIR", "/runtime"))
        self.run_dir = self.runtime_root / "runs" / self.run_id
        self.audit_dir = self.run_dir / "aggregation_audit"
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.event_path = self.run_dir / "events.jsonl"
        self.status_path = self.run_dir / "status.json"
        (self.runtime_root / "latest_run.json").write_text(
            json.dumps({"run_id": self.run_id, "path": str(self.run_dir)}, indent=2) + "\n",
            encoding="utf-8",
        )
        self.startup_deadline_perf = time.perf_counter() + float(self.demo["startup_timeout_seconds"])
        self.emit(
            "server_started",
            payload={
                "mode": self.demo["mode"],
                "expected_clients": sorted(self.clients),
                "rounds": self.demo["rounds"],
                "local_epochs": self.demo["local_epochs"],
                "contract_hash": self.contract["contract_hash"],
                "initial_model_bytes": self.model_payload_bytes,
            },
        )

    def _refresh_model_payload(self) -> None:
        encoded, payload_bytes, payload_hash = encode_state(self.global_state)
        self.model_payload = encoded
        self.model_payload_bytes = payload_bytes
        self.model_payload_sha256 = payload_hash

    def emit(
        self,
        event_type: str,
        *,
        source: str = "control-center",
        client_id: str | None = None,
        server_round: int | None = None,
        severity: str = "info",
        payload: dict | None = None,
    ) -> dict:
        with self._lock:
            self.sequence += 1
            event = {
                "schema_version": self.demo["event_schema_version"],
                "seq": self.sequence,
                "timestamp_utc": utc_now(),
                "elapsed_ms": round((time.perf_counter() - self.started_perf) * 1000, 3),
                "run_id": self.run_id,
                "source": source,
                "event_type": event_type,
                "severity": severity,
                "round": self.current_round if server_round is None else server_round,
                "client_id": client_id,
                "payload": payload or {},
            }
            self.events.append(event)
            with self.event_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            print(json.dumps(event, sort_keys=True), flush=True)
            self._write_status()
            return event

    def _status_unlocked(self) -> dict:
        expected = set(self.clients)
        received = set(self.updates)
        return {
            "schema_version": "phase4-status-v1",
            "run_id": self.run_id,
            "mode": self.demo["mode"],
            "state": self.state,
            "failure": self.failure,
            "contract_hash": self.contract["contract_hash"],
            "current_round": self.current_round,
            "total_rounds": int(self.demo["rounds"]),
            "local_epochs": int(self.demo["local_epochs"]),
            "registered_clients": sorted(self.registered),
            "expected_clients": sorted(expected),
            "received_updates": sorted(received),
            "waiting_for_clients": sorted(expected - received) if self.state == "running" else [],
            "last_metrics": self.last_metrics,
            "round_summaries": self.round_summaries,
            "event_count": len(self.events),
            "started_utc": self.started_utc,
            "elapsed_seconds": round(time.perf_counter() - self.started_perf, 6),
        }

    def _write_status(self) -> None:
        if not hasattr(self, "status_path"):
            return
        temporary = self.status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._status_unlocked(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.status_path)

    def status(self) -> dict:
        with self._lock:
            return self._status_unlocked()

    def events_after(self, after_seq: int) -> dict:
        with self._lock:
            selected = [event for event in self.events if event["seq"] > after_seq]
            return {"run_id": self.run_id, "events": selected[-1000:], "last_seq": self.sequence}

    def register(self, payload: dict) -> dict:
        with self._lock:
            if self.state in {"failed", "completed"}:
                raise ProtocolError(f"demo is already {self.state}")
            client_id = payload.get("client_id")
            if client_id not in self.clients:
                raise ProtocolError("unknown client_id")
            expected = self.clients[client_id]
            checks = {
                "protocol_version": self.demo["protocol_version"],
                "config_version": self.demo["config_version"],
                "contract_hash": self.contract["contract_hash"],
                "partition_sha256": expected["partition_sha256"],
                "samples": expected["samples"],
            }
            for name, value in checks.items():
                if payload.get(name) != value:
                    self.emit(
                        "client_registration_rejected",
                        client_id=client_id,
                        severity="error",
                        payload={"field": name, "reason": "incompatible_value"},
                    )
                    raise ProtocolError(f"registration {name} is incompatible")
            if client_id not in self.registered:
                self.registered[client_id] = {
                    "registered_utc": utc_now(),
                    "profile": expected["profile"],
                    "samples": expected["samples"],
                }
                self.emit("client_registered", client_id=client_id, payload=self.registered[client_id])
            if len(self.registered) == len(self.clients) and self.state == "waiting_for_clients":
                self.emit("all_clients_ready", payload={"client_count": len(self.registered)})
                self._start_round(1)
            return {"accepted": True, "run_id": self.run_id, "status": self._status_unlocked()}

    def _start_round(self, server_round: int) -> None:
        self.state = "running"
        self.current_round = server_round
        self.updates = {}
        self.round_started_perf = time.perf_counter()
        self.round_deadline_perf = self.round_started_perf + float(self.demo["round_timeout_seconds"])
        self.emit(
            "round_started",
            server_round=server_round,
            payload={
                "expected_clients": sorted(self.clients),
                "model_sha256": self.model_payload_sha256,
                "model_bytes": self.model_payload_bytes,
            },
        )

    def get_model(self, client_id: str) -> dict:
        with self._lock:
            if client_id not in self.registered:
                raise ProtocolError("client must register before requesting a model")
            if self.state != "running":
                return {"available": False, "status": self._status_unlocked()}
            if client_id in self.updates:
                return {"available": False, "reason": "update_already_received", "status": self._status_unlocked()}
            delivery_key = (self.current_round, client_id)
            if delivery_key not in self.delivered:
                self.delivered.add(delivery_key)
                self.emit(
                    "global_model_distributed",
                    client_id=client_id,
                    payload={"model_bytes": self.model_payload_bytes, "model_sha256": self.model_payload_sha256},
                )
            return {
                "available": True,
                "run_id": self.run_id,
                "round": self.current_round,
                "contract": self.contract,
                "state_spec": self.expected_spec,
                "weights": self.model_payload,
                "weights_bytes": self.model_payload_bytes,
                "weights_sha256": self.model_payload_sha256,
                "training": {
                    "epochs": int(self.demo["local_epochs"]),
                    "batch_size": int(self.locked["batch_size"]),
                    "learning_rate": float(self.locked["optimizer"]["learning_rate"]),
                    "weight_decay": float(self.locked["optimizer"]["weight_decay"]),
                    "seed": int(self.locked["seed"]),
                },
            }

    def client_event(self, payload: dict) -> dict:
        with self._lock:
            client_id = payload.get("client_id")
            event_type = payload.get("event_type")
            if client_id not in self.registered:
                raise ProtocolError("unregistered event source")
            if event_type not in self.CLIENT_EVENT_TYPES:
                raise ProtocolError("client event type is not allowed")
            if payload.get("run_id") != self.run_id:
                raise ProtocolError("client event run_id is stale")
            event_round = int(payload.get("round", self.current_round))
            return self.emit(
                event_type,
                source=client_id,
                client_id=client_id,
                server_round=event_round,
                severity=str(payload.get("severity", "info")),
                payload=payload.get("payload", {}),
            )

    def submit_update(self, payload: dict, content_bytes: int, receive_ms: float) -> dict:
        with self._lock:
            client_id = payload.get("client_id")
            if self.state != "running":
                raise ProtocolError("no active round")
            if client_id not in self.registered:
                raise ProtocolError("unregistered update source")
            if payload.get("run_id") != self.run_id:
                raise ProtocolError("stale run_id")
            if int(payload.get("round", -1)) != self.current_round:
                raise ProtocolError("stale or future round")
            if payload.get("contract_hash") != self.contract["contract_hash"]:
                self.emit(
                    "update_rejected",
                    client_id=client_id,
                    severity="error",
                    payload={"reason": "incompatible contract_hash"},
                )
                raise ProtocolError("incompatible contract_hash")
            expected = self.clients[client_id]
            if int(payload.get("samples", -1)) != int(expected["samples"]):
                raise ProtocolError("reported sample count is incompatible")
            if client_id in self.updates:
                raise ProtocolError("duplicate update")

            try:
                state, raw, update_hash = decode_state(
                    payload.get("weights"), self.expected_spec, int(self.demo["maximum_update_bytes"])
                )
            except ProtocolError as exc:
                self.emit(
                    "update_rejected",
                    client_id=client_id,
                    severity="error",
                    payload={"reason": str(exc)},
                )
                raise
            claimed_hash = payload.get("weights_sha256")
            if claimed_hash != update_hash:
                self.emit(
                    "update_rejected",
                    client_id=client_id,
                    severity="error",
                    payload={"reason": "weights_sha256 mismatch"},
                )
                raise ProtocolError("weights_sha256 mismatch")

            round_dir = self.audit_dir / f"round_{self.current_round}"
            round_dir.mkdir(parents=True, exist_ok=True)
            (round_dir / f"{client_id}.npz").write_bytes(raw)
            self.updates[client_id] = {
                "state": state,
                "samples": int(expected["samples"]),
                "update_sha256": update_hash,
                "update_bytes": len(raw),
                "http_content_bytes": content_bytes,
                "receive_ms": receive_ms,
                "client_metrics": payload.get("metrics", {}),
            }
            self.emit(
                "client_update_received",
                client_id=client_id,
                payload={
                    "samples": expected["samples"],
                    "update_bytes": len(raw),
                    "http_content_bytes": content_bytes,
                    "receive_ms": round(receive_ms, 3),
                    "update_sha256": update_hash,
                },
            )
            missing = sorted(set(self.clients) - set(self.updates))
            if missing:
                self.emit("server_waiting_for_clients", payload={"missing_clients": missing})
            else:
                self._aggregate_round(round_dir)
            return {"accepted": True, "status": self._status_unlocked()}

    def _aggregate_round(self, round_dir: Path) -> None:
        round_number = self.current_round
        ordered_ids = sorted(self.clients)
        counts = [self.updates[client_id]["samples"] for client_id in ordered_ids]
        states = [self.updates[client_id]["state"] for client_id in ordered_ids]
        total_samples = sum(counts)
        self.emit(
            "aggregation_started",
            payload={"sample_counts": dict(zip(ordered_ids, counts)), "total_samples": total_samples},
        )
        aggregate_started = time.perf_counter()
        self.global_state = fedavg(states, counts)
        aggregation_ms = (time.perf_counter() - aggregate_started) * 1000
        self.model.load_state_dict(self.global_state)
        self._refresh_model_payload()
        _, aggregated_raw, _ = decode_state(
            self.model_payload, self.expected_spec, int(self.demo["maximum_update_bytes"])
        )
        (round_dir / "aggregated.npz").write_bytes(aggregated_raw)

        evaluation_started = time.perf_counter()
        probabilities = predict_proba(self.model, self.validation_X, torch.device("cpu"))
        metrics = metric_bundle(
            self.validation_y,
            probabilities,
            float(self.contract["decision_threshold"]),
        )
        evaluation_ms = (time.perf_counter() - evaluation_started) * 1000
        round_ms = (time.perf_counter() - (self.round_started_perf or time.perf_counter())) * 1000
        self.last_metrics = metrics
        summary = {
            "round": round_number,
            "sample_counts": dict(zip(ordered_ids, counts)),
            "aggregation_weights": {client_id: count / total_samples for client_id, count in zip(ordered_ids, counts)},
            "total_samples": total_samples,
            "aggregation_ms": round(aggregation_ms, 3),
            "evaluation_ms": round(evaluation_ms, 3),
            "round_ms": round(round_ms, 3),
            "model_bytes": self.model_payload_bytes,
            "model_sha256": self.model_payload_sha256,
            "metrics": metrics,
            "client_updates": {
                client_id: {
                    key: value
                    for key, value in self.updates[client_id].items()
                    if key != "state"
                }
                for client_id in ordered_ids
            },
        }
        self.round_summaries.append(summary)
        (round_dir / "round_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.emit(
            "aggregation_completed",
            payload={
                "aggregation_ms": summary["aggregation_ms"],
                "total_samples": total_samples,
                "model_sha256": self.model_payload_sha256,
            },
        )
        self.emit("round_metrics", payload=metrics)
        self.emit("round_completed", payload={"round_ms": summary["round_ms"]})

        if round_number >= int(self.demo["rounds"]):
            self.state = "completed"
            self.round_deadline_perf = None
            self.emit(
                "demo_completed",
                payload={
                    "rounds": len(self.round_summaries),
                    "total_runtime_seconds": round(time.perf_counter() - self.started_perf, 6),
                    "final_metrics": self.last_metrics,
                },
            )
            (self.run_dir / "final_summary.json").write_text(
                json.dumps(self._status_unlocked(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        else:
            self._start_round(round_number + 1)

    def check_timeouts(self) -> None:
        with self._lock:
            now = time.perf_counter()
            if self.state == "waiting_for_clients" and now > self.startup_deadline_perf:
                missing = sorted(set(self.clients) - set(self.registered))
                self._fail("startup_timeout", missing)
            elif self.state == "running" and self.round_deadline_perf and now > self.round_deadline_perf:
                missing = sorted(set(self.clients) - set(self.updates))
                self._fail("round_timeout", missing)

    def _fail(self, reason: str, missing: list[str]) -> None:
        self.state = "failed"
        self.failure = f"{reason}: missing {', '.join(missing)}"
        self.round_deadline_perf = None
        self.emit(
            "demo_failed",
            severity="error",
            payload={"reason": reason, "missing_clients": missing},
        )
        (self.run_dir / "failure_summary.json").write_text(
            json.dumps(self._status_unlocked(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


class DemoHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, coordinator: Coordinator):
        super().__init__(address, RequestHandler)
        self.coordinator = coordinator


class RequestHandler(BaseHTTPRequestHandler):
    server: DemoHTTPServer
    protocol_version = "HTTP/1.1"

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json_body(self) -> tuple[dict, int, float]:
        length = int(self.headers.get("Content-Length", "0"))
        maximum = int(self.server.coordinator.demo["maximum_update_bytes"]) * 2
        if length <= 0 or length > maximum:
            raise ProtocolError("request body size is invalid")
        started = time.perf_counter()
        raw = self.rfile.read(length)
        receive_ms = (time.perf_counter() - started) * 1000
        try:
            return json.loads(raw), length, receive_ms
        except json.JSONDecodeError as exc:
            raise ProtocolError("request body is not valid JSON") from exc

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/health":
                self._send(HTTPStatus.OK, {"ok": True, "state": self.server.coordinator.state})
            elif parsed.path == "/api/v1/status":
                self._send(HTTPStatus.OK, self.server.coordinator.status())
            elif parsed.path == "/api/v1/events":
                after = int(parse_qs(parsed.query).get("after_seq", ["0"])[0])
                self._send(HTTPStatus.OK, self.server.coordinator.events_after(after))
            elif parsed.path == "/api/v1/model":
                client_id = parse_qs(parsed.query).get("client_id", [""])[0]
                self._send(HTTPStatus.OK, self.server.coordinator.get_model(client_id))
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except (ProtocolError, ValueError) as exc:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload, content_bytes, receive_ms = self._json_body()
            if self.path == "/api/v1/register":
                result = self.server.coordinator.register(payload)
            elif self.path == "/api/v1/events":
                result = self.server.coordinator.client_event(payload)
            elif self.path == "/api/v1/updates":
                result = self.server.coordinator.submit_update(payload, content_bytes, receive_ms)
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            self._send(HTTPStatus.OK, result)
        except ProtocolError as exc:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:
            self.server.coordinator.emit(
                "server_protocol_error", severity="error", payload={"error": type(exc).__name__, "detail": str(exc)}
            )
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal server error"})

    def log_message(self, format_string: str, *args) -> None:
        return


def watchdog(coordinator: Coordinator) -> None:
    while coordinator.state not in {"completed", "failed"}:
        coordinator.check_timeouts()
        time.sleep(0.5)


def main() -> None:
    coordinator = Coordinator()
    host = os.environ.get("SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("SERVER_PORT", "8080"))
    threading.Thread(target=watchdog, args=(coordinator,), daemon=True).start()
    server = DemoHTTPServer((host, port), coordinator)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
