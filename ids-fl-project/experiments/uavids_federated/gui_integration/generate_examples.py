"""Generate actual GUI API response examples from the frozen model and saved telemetry."""

from __future__ import annotations

import json
from pathlib import Path

from backend import API_VERSION, GuiBackend


OUTPUT = Path(__file__).resolve().parent / "examples"


def write(name: str, value: dict) -> None:
    (OUTPUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    backend = GuiBackend(upstream_url=None)
    normal = attack = None
    for _ in range(len(backend.replay_records)):
        result = backend.replay_next()["prediction"]
        if result["label"] == "Normal" and normal is None:
            normal = result
        if result["label"] == "Attack" and attack is None:
            attack = result
    assert normal is not None and attack is not None
    write(
        "health_response.json",
        {
            "schema_version": "uavids-gui-health-v1",
            "ok": True,
            "api_version": API_VERSION,
            "model_available": True,
        },
    )
    write("prediction_normal_response.json", normal)
    write("prediction_attack_response.json", attack)
    write("snapshot_replay_response.json", backend.snapshot())
    write("federated_events_replay_response.json", backend.telemetry.events(0))
    print(json.dumps({"written": sorted(path.name for path in OUTPUT.glob("*_response.json"))}))


if __name__ == "__main__":
    main()
