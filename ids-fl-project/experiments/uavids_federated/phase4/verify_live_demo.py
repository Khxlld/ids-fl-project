"""Verify a completed Compose run and independently recompute every FedAvg."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase4.app.common import read_json, sha256_path


def main() -> None:
    runtime = ROOT / "phase4" / "runtime"
    latest = read_json(runtime / "latest_run.json")
    run_dir = runtime / "runs" / latest["run_id"]
    status = read_json(run_dir / "final_summary.json")
    demo = read_json(ROOT / "phase4" / "config" / "demo_config.json")
    locked = read_json(ROOT / "config" / "phase3_locked_config.json")

    assert status["state"] == "completed"
    assert status["run_id"] == latest["run_id"]
    expected_clients = sorted(item["client_id"] for item in demo["clients"])
    assert status["registered_clients"] == expected_clients
    assert len(status["round_summaries"]) == int(demo["rounds"])
    assert sha256_path(ROOT / "config" / "phase3_locked_config.json") == demo["phase3_lock_sha256"]
    assert sha256_path(ROOT / "artifacts_phase3" / "training_only_preprocessor.joblib") == locked["preprocessor"]["sha256"]
    assert sha256_path(ROOT / "artifacts_phase3" / "federated_global_model.pt") == locked["model_artifacts"]["federated_fedavg"]["sha256"]

    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    event_types = {event["event_type"] for event in events}
    required = {
        "server_started", "all_clients_ready", "round_started", "global_model_distributed",
        "client_training_started", "client_training_completed", "client_update_received",
        "server_waiting_for_clients", "aggregation_started", "aggregation_completed",
        "round_metrics", "round_completed", "demo_completed",
    }
    assert required <= event_types

    max_difference = 0.0
    for summary in status["round_summaries"]:
        round_number = int(summary["round"])
        round_dir = run_dir / "aggregation_audit" / f"round_{round_number}"
        counts = summary["sample_counts"]
        assert sorted(counts) == expected_clients
        assert sum(counts.values()) == 6148
        with np.load(round_dir / "aggregated.npz", allow_pickle=False) as aggregate:
            names = aggregate.files
            for name in names:
                expected = np.zeros_like(aggregate[name], dtype=np.float64)
                for client_id in expected_clients:
                    with np.load(round_dir / f"{client_id}.npz", allow_pickle=False) as update:
                        expected += update[name].astype(np.float64) * (counts[client_id] / 6148.0)
                expected = expected.astype(np.float32)
                difference = float(np.max(np.abs(expected - aggregate[name])))
                max_difference = max(max_difference, difference)
                np.testing.assert_allclose(aggregate[name], expected, rtol=0, atol=1e-7)

    print(
        json.dumps(
            {
                "verified": True,
                "run_id": latest["run_id"],
                "rounds": len(status["round_summaries"]),
                "clients_per_round": len(expected_clients),
                "total_samples_per_round": 6148,
                "maximum_aggregation_absolute_difference": max_difference,
                "frozen_phase3_hashes_unchanged": True,
                "event_count": len(events),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
