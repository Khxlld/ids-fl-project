"""DEVELOPMENT MOCK - NOT THE REAL BACKEND. NEVER PRESENT ON THIS.

=============================================================================
This is not `backend.py`. It does not load the project's model. The verdicts
it returns are produced by a crude two-feature logistic that exists only so the
dashboard renders varying probabilities during frontend development. Its
numbers carry no scientific meaning whatsoever.
=============================================================================

Why it exists
-------------
`backend.py` hash-verifies and loads the frozen Phase 3 preprocessor and
checkpoint from `artifacts_phase3/`. Those are `*.joblib` / `*.pt` files
excluded by the repository `.gitignore`, so on a clean checkout the real
adapter cannot start:

    FileNotFoundError: ...artifacts_phase3/training_only_preprocessor.joblib

Until those artifacts are present, this mock is the only way to open the
dashboard at all. It serves the same API surface from the committed fixtures in
`examples/`, so layout, polling, cursors, counters and the simulator can be
exercised end to end.

Usage
-----
    python gui_integration/dev_mock_backend.py

Then start the dashboard on port 3000. Swap back to the real adapter with:

    python -m gui_integration.backend --host 127.0.0.1 --port 8090 \
        --allowed-origins http://localhost:3000

Telling them apart
------------------
The real adapter prints a `gui_backend_ready` JSON line containing the true
`model_id` on startup. This mock prints a loud MOCK banner instead. The
`/health` and `/snapshot` payloads are deliberately fixture-faithful, so the
UI cannot distinguish them - the startup banner is the only reliable signal.
"""

import itertools
import json
import math
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent / "examples"
ORIGIN = "http://localhost:3000"
THRESHOLD = 0.42


def load(name: str):
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


HEALTH = load("health_response.json")
SNAPSHOT = load("snapshot_replay_response.json")
FED_EVENTS = load("federated_events_replay_response.json")["events"]
REPLAY_RECORDS = load("replay_records.json")

MODEL_ID = SNAPSHOT["model"]["model_id"]
MODEL_VERSION = SNAPSHOT["model"]["model_version"]

replay_cycle = itertools.cycle(enumerate(REPLAY_RECORDS))

STATE = {
    "processed": 0,
    "normal": 0,
    "attack": 0,
    "latest": None,
    "alerts": [],
    "events": [],
    "seq": 0,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def score(features: dict) -> float:
    """Stand-in discriminator separating the two clusters visible in the fixture.

    NOT the project's model. Keyed off packet count and mean delay only, because
    those two axes separate the recorded Normal rows (01/03/05) from the
    recorded Attack rows (02/04/06) cleanly enough to drive the UI.
    """
    packets = max(1.0, float(features.get("TxPackets") or 1.0))
    delay = max(1e-6, float(features.get("MeanDelay/s") or 1e-6))
    packet_term = (math.log10(packets) - 2.2) / 0.55
    delay_term = (math.log10(0.15) - math.log10(delay)) / 0.6
    return 1.0 / (1.0 + math.exp(-2.2 * (0.5 * packet_term + 0.5 * delay_term)))


def predict(body: dict, replayed: bool) -> dict:
    features = body.get("features") or {}
    probability = score(features)
    is_attack = probability >= THRESHOLD
    prediction = {
        "schema_version": "uavids-gui-prediction-v1",
        "prediction_id": str(uuid.uuid4()),
        "timestamp_utc": utc_now(),
        "record_id": body.get("record_id") or str(uuid.uuid4()),
        "source": body.get("source"),
        "label": "Attack" if is_attack else "Normal",
        "confidence": probability if is_attack else 1.0 - probability,
        "attack_probability": probability,
        "decision_threshold": THRESHOLD,
        "missing_features_imputed": sum(1 for value in features.values() if value is None),
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "inference_mode": "live_model",
        "replayed": replayed,
    }
    STATE["processed"] += 1
    STATE["latest"] = prediction
    if is_attack:
        STATE["attack"] += 1
        STATE["alerts"].insert(0, prediction)
        del STATE["alerts"][20:]
    else:
        STATE["normal"] += 1
    STATE["seq"] += 1
    STATE["events"].append(
        {
            "schema_version": "uavids-gui-event-v1",
            "seq": STATE["seq"],
            "timestamp_utc": prediction["timestamp_utc"],
            "event_type": "prediction_completed",
            "severity": "warning" if is_attack else "info",
            "source": prediction["source"],
            "payload": prediction,
        }
    )
    del STATE["events"][:-200]
    return prediction


def snapshot() -> dict:
    payload = json.loads(json.dumps(SNAPSHOT))
    payload["generated_utc"] = utc_now()
    payload["inference"] = {
        "records_processed": STATE["processed"],
        "normal_count": STATE["normal"],
        "attack_count": STATE["attack"],
        "latest_prediction": STATE["latest"],
        "recent_alerts": STATE["alerts"],
    }
    return payload


def after_seq(path: str) -> int:
    return int(path.split("after_seq=")[-1]) if "after_seq=" in path else 0


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", ORIGIN)
        self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(body)

    def _not_found(self) -> None:
        self._send(
            HTTPStatus.NOT_FOUND,
            {
                "schema_version": "uavids-gui-error-v1",
                "error": {"code": "not_found", "message": "endpoint not found"},
            },
        )

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Content-Length", "0")
        self.send_header("Access-Control-Allow-Origin", ORIGIN)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path
        if path.startswith("/api/gui/v1/health") or path == "/health":
            self._send(HTTPStatus.OK, HEALTH)
        elif path.startswith("/api/gui/v1/snapshot"):
            self._send(HTTPStatus.OK, snapshot())
        elif path.startswith("/api/gui/v1/federated/events"):
            cursor = after_seq(path)
            self._send(
                HTTPStatus.OK,
                {
                    "schema_version": "uavids-gui-federated-events-v1",
                    "data_mode": "replay",
                    "run_id": SNAPSHOT["federated"]["run_id"],
                    "events": [e for e in FED_EVENTS if e["seq"] > cursor],
                    "last_seq": len(FED_EVENTS),
                },
            )
        elif path.startswith("/api/gui/v1/events"):
            cursor = after_seq(path)
            self._send(
                HTTPStatus.OK,
                {
                    "schema_version": "uavids-gui-events-v1",
                    "events": [e for e in STATE["events"] if e["seq"] > cursor],
                    "last_seq": STATE["seq"],
                },
            )
        else:
            self._not_found()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send(
                HTTPStatus.BAD_REQUEST,
                {
                    "schema_version": "uavids-gui-error-v1",
                    "error": {"code": "invalid_json", "message": "request body is not valid JSON"},
                },
            )
            return

        if self.path == "/api/gui/v1/predictions":
            self._send(HTTPStatus.OK, predict(body, replayed=False))
        elif self.path == "/api/gui/v1/replay/next":
            position, record = next(replay_cycle)
            self._send(
                HTTPStatus.OK,
                {
                    "schema_version": "uavids-gui-replay-v1",
                    "position": position + 1,
                    "total_records": len(REPLAY_RECORDS),
                    "wrapped": position == len(REPLAY_RECORDS) - 1,
                    "prediction": predict(record, replayed=True),
                },
            )
        else:
            self._not_found()

    def log_message(self, format_string: str, *args) -> None:
        return


def main() -> None:
    print("=" * 72, flush=True)
    print("  MOCK BACKEND - development only. This is NOT the project's model.", flush=True)
    print("  Verdicts are from a stand-in scorer and mean nothing. Do not present.", flush=True)
    print("=" * 72, flush=True)
    print("  http://127.0.0.1:8090  (allowed origin: %s)" % ORIGIN, flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8090), Handler).serve_forever()


if __name__ == "__main__":
    main()
