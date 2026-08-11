"""Presentation launcher for the existing GUI adapter with 24 live demo inputs.

The verified adapter implementation is reused unchanged. This launcher replaces
its six-item replay fixture in memory with 24 distinct validation rows selected
by frozen-model probability. Feature values are never written, logged, or
returned by the API; each flow is still evaluated by the frozen model when the
frontend requests it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from gui_integration.backend import DEFAULT_ORIGINS, GuiBackend, create_server  # noqa: E402
DEMO_CANDIDATE_INDICES = (
    40110, 41157, 7491, 44023, 43002, 2107, 42001, 29897,
    4365, 30392, 28139, 47170, 30228, 29133, 21694, 32827,
    33982, 17416, 36948, 35263, 22973, 46733, 45975, 14553,
)
EXPECTED_LABELS = "NNANNANNANNANNANNANNANNA"


def build_live_demo_records(backend: GuiBackend) -> list[dict]:
    validation_path = EXPERIMENT_ROOT / "partitions" / "phase2" / "validation.csv"
    validation = pd.read_csv(validation_path)
    feature_names = list(backend.ids.features)
    feature_values = validation[feature_names].to_numpy(dtype=float)
    finite = np.isfinite(feature_values).all(axis=1)
    candidates = validation.loc[finite, feature_names].reset_index(drop=True)
    if max(DEMO_CANDIDATE_INDICES) >= len(candidates):
        raise RuntimeError("validation partition is incompatible with the verified live demo selection")
    records = []
    for sequence, row_index in enumerate(DEMO_CANDIDATE_INDICES, start=1):
        records.append(
            {
                "record_id": f"live-demo-flow-{sequence:03d}",
                "source": "approved-validation-replay",
                "features": {
                    name: float(candidates.iloc[row_index][name])
                    for name in feature_names
                },
            }
        )
    labels = "".join(
        "A" if backend.ids.predict(record)["label"] == "Attack" else "N"
        for record in records
    )
    if labels != EXPECTED_LABELS:
        raise RuntimeError("frozen model or validation inputs no longer match the verified live demo")
    return records


class PresentationGuiBackend(GuiBackend):
    def snapshot(self) -> dict:
        payload = super().snapshot()
        payload["inference"]["demo_total_records"] = len(self.replay_records)
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="UAVIDS presentation GUI adapter")
    parser.add_argument("--host", default=os.environ.get("GUI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GUI_PORT", "8090")))
    parser.add_argument(
        "--allowed-origins",
        default=os.environ.get("GUI_ALLOWED_ORIGINS", ",".join(DEFAULT_ORIGINS)),
    )
    parser.add_argument(
        "--federated-backend",
        default=os.environ.get("FEDERATED_BACKEND_URL", ""),
    )
    args = parser.parse_args()
    origins = tuple(value.strip() for value in args.allowed_origins.split(",") if value.strip())
    if not origins:
        raise SystemExit("at least one GUI origin must be configured")

    backend = PresentationGuiBackend(upstream_url=args.federated_backend or None)
    backend.replay_records = build_live_demo_records(backend)
    backend.replay_index = 0
    server = create_server(backend, args.host, args.port, origins)
    print(
        json.dumps(
            {
                "event": "presentation_gui_backend_ready",
                "url": f"http://{args.host}:{server.server_address[1]}",
                "allowed_origins": origins,
                "federated_backend": args.federated_backend or None,
                "model_id": backend.ids.model_id,
                "live_demo_records": len(backend.replay_records),
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
