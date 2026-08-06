"""End-to-end verification for the GUI adapter, replay path, CORS, and artifact safety."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from backend import FederatedTelemetry, GuiBackend, create_server, sha256_path


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = Path(__file__).resolve().parent
ORIGIN = "http://localhost:5173"


class RecordedContractHandler(BaseHTTPRequestHandler):
    """Serve actual saved Phase 4 event shapes as a lightweight live-upstream check."""

    status_payload: dict = {}
    event_payload: dict = {}

    def do_GET(self):  # noqa: N802
        if self.path == "/api/v1/status":
            value = self.status_payload
        elif self.path.startswith("/api/v1/events"):
            value = self.event_payload
        else:
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps(value).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string, *args):
        return


def request(base: str, path: str, *, method: str = "GET", payload: dict | None = None, origin: str = ORIGIN):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json", "Origin": origin}
    if body is not None:
        headers["Content-Type"] = "application/json"
    value = urllib.request.Request(base + path, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(value, timeout=5) as response:
            return response.status, dict(response.headers), json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), json.loads(exc.read())


def main() -> None:
    lock_path = ROOT / "config" / "phase3_locked_config.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checkpoint = ROOT.joinpath(*lock["model_artifacts"]["federated_fedavg"]["path"].replace("\\", "/").split("/"))
    preprocessor = ROOT.joinpath(*lock["preprocessor"]["path"].replace("\\", "/").split("/"))
    before = {path.name: sha256_path(path) for path in (lock_path, checkpoint, preprocessor)}

    backend = GuiBackend(upstream_url="http://127.0.0.1:1")
    server = create_server(backend, "127.0.0.1", 0, (ORIGIN,))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        status, headers, health = request(base, "/api/gui/v1/health")
        assert status == 200 and health["ok"] and health["model_available"]
        assert headers["Access-Control-Allow-Origin"] == ORIGIN

        status, _, snapshot = request(base, "/api/gui/v1/snapshot")
        assert status == 200 and snapshot["presentation_mode"] == "replay"
        assert snapshot["backend"]["inference_mode"] == "live_model"
        assert snapshot["model"]["labels"] == ["Normal", "Attack"]
        assert len(snapshot["federated"]["clients"]) == 5
        assert snapshot["federated"]["security"]["mode"] == "secure"

        labels = []
        for _ in backend.replay_records:
            status, _, replay = request(base, "/api/gui/v1/replay/next", method="POST", payload={})
            assert status == 200 and replay["prediction"]["replayed"] is True
            labels.append(replay["prediction"]["label"])
        assert labels.count("Normal") == 3 and labels.count("Attack") == 3

        status, _, snapshot = request(base, "/api/gui/v1/snapshot")
        assert status == 200
        assert snapshot["inference"]["records_processed"] == 6
        assert snapshot["inference"]["normal_count"] == 3
        assert snapshot["inference"]["attack_count"] == 3
        assert len(snapshot["inference"]["recent_alerts"]) == 3

        status, _, events = request(base, "/api/gui/v1/events?after_seq=0")
        assert status == 200 and len(events["events"]) == 6 and events["last_seq"] == 6
        status, _, federated = request(base, "/api/gui/v1/federated/events?after_seq=0")
        assert status == 200 and federated["data_mode"] == "replay" and federated["events"]

        status, _, error = request(
            base,
            "/api/gui/v1/predictions",
            method="POST",
            payload={"record_id": "bad", "features": {}},
        )
        assert status == 400 and error["error"]["code"] == "invalid_features"
        status, _, error = request(base, "/api/gui/v1/health", origin="http://untrusted.invalid")
        assert status == 403 and error["error"]["code"] == "origin_not_allowed"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    saved_events = json.loads((ROOT / "phase4" / "results" / "event_excerpt.json").read_text(encoding="utf-8"))
    RecordedContractHandler.status_payload = {
        "run_id": "recorded-contract-check",
        "state": "running",
        "current_round": 2,
        "total_rounds": 3,
        "expected_clients": [f"uav-client-{i}" for i in range(1, 6)],
        "received_updates": ["uav-client-1", "uav-client-4"],
        "last_metrics": backend.telemetry.benchmark["final_validation_metrics"],
        "security": {"mode": "pq_secure", "authenticated_clients": [f"uav-client-{i}" for i in range(1, 6)]},
    }
    RecordedContractHandler.event_payload = {
        "run_id": "recorded-contract-check",
        "events": saved_events,
        "last_seq": max(event["seq"] for event in saved_events),
    }
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), RecordedContractHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    try:
        telemetry = FederatedTelemetry(ROOT, f"http://127.0.0.1:{upstream.server_address[1]}")
        live = telemetry.snapshot()
        assert live["data_mode"] == "live" and live["upstream_available"] is True
        assert live["current_round"] == 2 and live["updates_received"] == 2
        assert live["security"]["mode"] == "secure" and live["security"]["authenticated_clients"] == 5
        live_events = telemetry.events(0)
        assert live_events["data_mode"] == "live" and live_events["events"] == saved_events
    finally:
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)

    after = {path.name: sha256_path(path) for path in (lock_path, checkpoint, preprocessor)}
    assert after == before
    forbidden_suffixes = {".csv", ".pt", ".pth", ".pkl", ".joblib", ".key", ".pem"}
    files = [path for path in HANDOFF.rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    assert not [path for path in files if path.suffix.lower() in forbidden_suffixes]
    assert max(path.stat().st_size for path in files) < 2 * 1024 * 1024
    private_key_marker = b"-----BEGIN " + b"PRIVATE KEY-----"
    for path in files:
        assert private_key_marker not in path.read_bytes()

    replay = json.loads((HANDOFF / "examples" / "replay_records.json").read_text(encoding="utf-8"))
    assert len(replay) == 6
    for record in replay:
        assert set(record) == {"record_id", "source", "features"}
        assert set(record["features"]) == set(lock["features"])
        assert "binary_label" not in record and "original_label" not in record

    required_examples = {
        "health_response.json",
        "prediction_normal_response.json",
        "prediction_attack_response.json",
        "snapshot_replay_response.json",
        "federated_events_replay_response.json",
    }
    assert required_examples <= {path.name for path in (HANDOFF / "examples").iterdir()}
    print(
        json.dumps(
            {
                "verified": True,
                "http_flow": "passed",
                "cors": "passed",
                "replay_predictions": {"Normal": 3, "Attack": 3},
                "federated_fallback": "passed",
                "live_federated_contract": "passed",
                "frozen_artifacts_unchanged": True,
                "handoff_files_checked": len(files),
                "forbidden_payload_files": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
