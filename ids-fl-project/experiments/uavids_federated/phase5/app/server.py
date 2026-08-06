"""Secure HTTP adapter around the verified Phase 4 coordinator."""

from __future__ import annotations

import json
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    from phase4_app.server import Coordinator, watchdog
except ModuleNotFoundError:  # host-side tests
    from phase4.app.server import Coordinator, watchdog

from .security import (
    SERVER_ID,
    SecurityError,
    ServerSecurityManager,
    canonical_bytes,
    load_security_material,
)


class SecureCoordinator:
    """Keep Phase 4 model behavior intact and add a secure transport facade."""

    def __init__(self) -> None:
        self.phase4 = Coordinator()
        config, trust, secret = load_security_material(
            os.environ["SECURITY_CONFIG_PATH"],
            os.environ["TRUST_STORE_PATH"],
            os.environ["SERVER_SIGN_SECRET_PATH"],
        )
        self.security = ServerSecurityManager(self.phase4, config, trust, secret)
        self.maximum_secure_json_bytes = int(config["maximum_secure_json_bytes"])
        self.security_path = self.phase4.run_dir / "security_summary.json"
        # ThreadingHTTPServer may finish several client requests together.  Keep
        # the atomic summary-file replacement itself single-writer so concurrent
        # requests cannot rename the same temporary path out from under another.
        self._security_summary_lock = threading.Lock()
        self.phase4.emit(
            "security_ready",
            payload={
                "security_version": config["security_version"],
                "kem": config["kem_algorithm"],
                "signature": config["signature_algorithm"],
                "kdf": config["kdf"],
                "aead": config["aead"],
                "provisioning": "controlled_academic_trust_store",
            },
        )
        self.write_security_summary()

    @property
    def state(self):
        return self.phase4.state

    def write_security_summary(self) -> None:
        with self._security_summary_lock:
            temporary = self.security_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(self.security.safe_summary(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.security_path)

    def reject(self, endpoint: str, exc: SecurityError, claimed_client_id: str | None = None) -> None:
        self.phase4.emit(
            "security_message_rejected",
            client_id=claimed_client_id if claimed_client_id in self.phase4.clients else None,
            severity="error",
            payload={"endpoint": endpoint, "category": exc.category},
        )
        self.write_security_summary()

    def begin_session(self, payload: dict) -> dict:
        result = self.security.begin(payload)
        self.write_security_summary()
        return result

    def finish_session(self, payload: dict) -> dict:
        client_id, registration, confirmation = self.security.finish(payload)
        registration_result = self.phase4.register(registration)
        self.write_security_summary()
        return {"confirmation": confirmation, "status": registration_result["status"]}

    @staticmethod
    def _claimed_client(envelope: object) -> str | None:
        if isinstance(envelope, dict) and isinstance(envelope.get("header"), dict):
            sender = envelope["header"].get("sender")
            return sender if isinstance(sender, str) else None
        return None

    def model(self, envelope: dict) -> dict:
        client_id = self._claimed_client(envelope)
        if client_id is None:
            raise SecurityError("malformed_payload", "secure sender is missing")
        session = self.security.session(client_id)
        request, header = session.open_json(
            envelope,
            message_type="model_request",
            maximum_plaintext_bytes=4096,
        )
        if request != {
            "client_id": client_id,
            "run_id": self.phase4.run_id,
            "completed_round": request.get("completed_round"),
        } or not isinstance(request.get("completed_round"), int):
            raise SecurityError("metadata_mismatch", "model request payload is inconsistent")
        if header["round"] != request["completed_round"] + 1:
            raise SecurityError("metadata_mismatch", "model request round is incorrect")
        response = self.phase4.get_model(client_id)
        response_round = int(response.get("round", self.phase4.current_round))
        model_hash = response.get("weights_sha256", "")
        protected = session.seal_json(
            response,
            message_type="global_model",
            server_round=response_round,
            model_sha256=model_hash,
        )
        if response.get("available"):
            self.phase4.emit(
                "security_message_protected",
                client_id=client_id,
                server_round=response_round,
                payload={
                    "message_type": "global_model",
                    "sequence": protected["header"]["sequence"],
                    "plaintext_bytes": len(canonical_bytes(response)),
                    "protected_bytes": len(canonical_bytes(protected)),
                },
            )
        self.write_security_summary()
        return {"secure_envelope": protected}

    def client_event(self, envelope: dict) -> dict:
        client_id = self._claimed_client(envelope)
        if client_id is None:
            raise SecurityError("malformed_payload", "secure sender is missing")
        session = self.security.session(client_id)
        event, header = session.open_json(
            envelope,
            message_type="client_event",
            maximum_plaintext_bytes=65536,
        )
        if event.get("client_id") != client_id or event.get("run_id") != self.phase4.run_id:
            raise SecurityError("metadata_mismatch", "client event identity or run is incorrect")
        if header["round"] != int(event.get("round", -1)):
            raise SecurityError("metadata_mismatch", "client event round is incorrect")
        result = self.phase4.client_event(event)
        self.write_security_summary()
        return {"accepted": True, "event_seq": result["seq"]}

    def update(self, envelope: dict, content_bytes: int, receive_ms: float) -> dict:
        client_id = self._claimed_client(envelope)
        if client_id is None:
            raise SecurityError("malformed_payload", "secure sender is missing")
        session = self.security.session(client_id)
        update, header = session.open_json(
            envelope,
            message_type="client_update",
            server_round=self.phase4.current_round,
            maximum_plaintext_bytes=self.maximum_secure_json_bytes,
        )
        if update.get("client_id") != client_id or update.get("run_id") != self.phase4.run_id:
            raise SecurityError("metadata_mismatch", "update identity or run is incorrect")
        if header["model_sha256"] != update.get("weights_sha256"):
            raise SecurityError("metadata_mismatch", "protected model hash is incorrect")
        result = self.phase4.submit_update(update, content_bytes, receive_ms)
        self.phase4.emit(
            "security_message_unprotected",
            client_id=client_id,
            payload={
                "message_type": "client_update",
                "sequence": header["sequence"],
                "protected_bytes": content_bytes,
                "plaintext_bytes": len(canonical_bytes(update)),
            },
        )
        self.write_security_summary()
        return result

    def probe(self, envelope: dict) -> dict:
        client_id = self._claimed_client(envelope)
        if client_id is None:
            raise SecurityError("malformed_payload", "secure sender is missing")
        session = self.security.session(client_id)
        probe, header = session.open_json(
            envelope,
            message_type="attack_probe",
            maximum_plaintext_bytes=4096,
        )
        if probe != {"purpose": "controlled_rejection_test"}:
            raise SecurityError("metadata_mismatch", "attack probe payload is incorrect")
        acknowledgement = session.seal_json(
            {"accepted": True, "purpose": "controlled_rejection_test"},
            message_type="attack_probe_ack",
            server_round=int(header["round"]),
        )
        self.phase4.emit(
            "security_probe_accepted",
            client_id=client_id,
            server_round=int(header["round"]),
            payload={"sequence": header["sequence"]},
        )
        self.write_security_summary()
        return {"secure_envelope": acknowledgement}


class SecureHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, coordinator: SecureCoordinator):
        super().__init__(address, SecureRequestHandler)
        self.coordinator = coordinator


class SecureRequestHandler(BaseHTTPRequestHandler):
    server: SecureHTTPServer
    protocol_version = "HTTP/1.1"

    def _send(self, status: int, payload: dict) -> None:
        body = canonical_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json_body(self) -> tuple[dict, int, float]:
        length = int(self.headers.get("Content-Length", "0"))
        maximum = self.server.coordinator.maximum_secure_json_bytes
        if length <= 0 or length > maximum:
            raise SecurityError("malformed_payload", "request body size is invalid")
        started = time.perf_counter()
        raw = self.rfile.read(length)
        receive_ms = (time.perf_counter() - started) * 1000
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SecurityError("malformed_payload", "request body is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise SecurityError("malformed_payload", "request JSON must be an object")
        return payload, length, receive_ms

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/health":
                self._send(HTTPStatus.OK, {"ok": True, "state": self.server.coordinator.state})
            elif parsed.path == "/api/v1/status":
                status = self.server.coordinator.phase4.status()
                status["security"] = {
                    "mode": "pq_secure",
                    "authenticated_clients": sorted(self.server.coordinator.security.sessions),
                }
                self._send(HTTPStatus.OK, status)
            elif parsed.path == "/api/v1/events":
                after = int(parse_qs(parsed.query).get("after_seq", ["0"])[0])
                self._send(HTTPStatus.OK, self.server.coordinator.phase4.events_after(after))
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except (SecurityError, ValueError) as exc:
            self._send(HTTPStatus.BAD_REQUEST, {"error": "request rejected"})

    def do_POST(self) -> None:  # noqa: N802
        endpoint = self.path
        claimed_client_id: str | None = None
        try:
            payload, content_bytes, receive_ms = self._json_body()
            claimed_client_id = SecureCoordinator._claimed_client(payload)
            if endpoint == "/api/v2/secure/hello":
                result = self.server.coordinator.begin_session(payload)
            elif endpoint == "/api/v2/secure/session":
                result = self.server.coordinator.finish_session(payload)
            elif endpoint == "/api/v2/secure/model":
                result = self.server.coordinator.model(payload)
            elif endpoint == "/api/v2/secure/events":
                result = self.server.coordinator.client_event(payload)
            elif endpoint == "/api/v2/secure/updates":
                result = self.server.coordinator.update(payload, content_bytes, receive_ms)
            elif endpoint == "/api/v2/secure/probe":
                result = self.server.coordinator.probe(payload)
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            self._send(HTTPStatus.OK, result)
        except SecurityError as exc:
            self.server.coordinator.reject(endpoint, exc, claimed_client_id)
            self._send(HTTPStatus.BAD_REQUEST, {"error": "security check failed", "category": exc.category})
        except Exception as exc:
            self.server.coordinator.phase4.emit(
                "server_protocol_error",
                severity="error",
                payload={"error": type(exc).__name__, "detail": "internal secure protocol error"},
            )
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal server error"})

    def log_message(self, format_string: str, *args) -> None:
        return


def main() -> None:
    coordinator = SecureCoordinator()
    host = os.environ.get("SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("SERVER_PORT", "8080"))
    threading.Thread(target=watchdog, args=(coordinator.phase4,), daemon=True).start()
    server = SecureHTTPServer((host, port), coordinator)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
