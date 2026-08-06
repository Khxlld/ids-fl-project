"""Verify secure training, FedAvg, plain equivalence, and secret non-disclosure."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase4.app.common import read_json, sha256_path


def latest_run(runtime: Path) -> tuple[str, Path]:
    run_id = read_json(runtime / "latest_run.json")["run_id"]
    return run_id, runtime / "runs" / run_id


def verify_aggregates(run_dir: Path, status: dict, expected_clients: list[str]) -> float:
    maximum = 0.0
    for summary in status["round_summaries"]:
        round_number = int(summary["round"])
        round_dir = run_dir / "aggregation_audit" / f"round_{round_number}"
        counts = summary["sample_counts"]
        assert sorted(counts) == expected_clients
        assert sum(counts.values()) == 6148
        with np.load(round_dir / "aggregated.npz", allow_pickle=False) as aggregate:
            for name in aggregate.files:
                expected = np.zeros_like(aggregate[name], dtype=np.float64)
                for client_id in expected_clients:
                    with np.load(round_dir / f"{client_id}.npz", allow_pickle=False) as update:
                        expected += update[name].astype(np.float64) * (counts[client_id] / 6148.0)
                expected = expected.astype(np.float32)
                difference = float(np.max(np.abs(expected - aggregate[name])))
                maximum = max(maximum, difference)
                np.testing.assert_allclose(aggregate[name], expected, rtol=0, atol=1e-7)
    return maximum


def compare_plain(secure_run: Path, plain_run: Path, rounds: int) -> float:
    maximum = 0.0
    for round_number in range(1, rounds + 1):
        secure_path = secure_run / "aggregation_audit" / f"round_{round_number}" / "aggregated.npz"
        plain_path = plain_run / "aggregation_audit" / f"round_{round_number}" / "aggregated.npz"
        with np.load(secure_path, allow_pickle=False) as secure, np.load(plain_path, allow_pickle=False) as plain:
            assert secure.files == plain.files
            for name in secure.files:
                difference = float(np.max(np.abs(secure[name] - plain[name])))
                maximum = max(maximum, difference)
                np.testing.assert_allclose(secure[name], plain[name], rtol=0, atol=1e-7)
    return maximum


def assert_secrets_absent(runtime: Path, run_dir: Path) -> int:
    secret_paths = sorted((runtime / "keys").glob("**/sign_secret.key"))
    assert len(secret_paths) == 6
    evidence_paths = [
        path
        for path in [
            *run_dir.glob("*.json"),
            *run_dir.glob("*.jsonl"),
            runtime / "host_run_timing.json",
            runtime / "host_container_stats.jsonl",
            runtime / "latest_verification.json",
        ]
        if path.is_file()
    ]
    evidence = b"\n".join(path.read_bytes() for path in evidence_paths)
    for secret_path in secret_paths:
        secret = secret_path.read_bytes()
        assert secret not in evidence
        assert base64.b64encode(secret) not in evidence
    forbidden_names = {"shared_secret", "derived_key", "aes_key", "private_key", "secret_key"}
    lowered = evidence.lower()
    for name in forbidden_names:
        assert name.encode("ascii") not in lowered
    return len(evidence_paths)


def main() -> None:
    runtime = ROOT / "phase5" / "runtime"
    run_id, run_dir = latest_run(runtime)
    status = read_json(run_dir / "final_summary.json")
    security = read_json(run_dir / "security_summary.json")
    demo = read_json(ROOT / "phase5" / "config" / "demo_config.json")
    security_config = read_json(ROOT / "phase5" / "config" / "security_config.json")
    locked = read_json(ROOT / "config" / "phase3_locked_config.json")

    assert status["state"] == "completed"
    assert status["mode"] == "pq_secure_demo_nonresearch"
    assert status["run_id"] == run_id
    expected_clients = sorted(item["client_id"] for item in demo["clients"])
    assert status["registered_clients"] == expected_clients
    assert len(status["round_summaries"]) == int(demo["rounds"])
    assert security["authenticated_clients"] == expected_clients
    assert security["algorithms"] == {
        "kem": security_config["kem_algorithm"],
        "signature": security_config["signature_algorithm"],
        "kdf": security_config["kdf"],
        "aead": security_config["aead"],
    }
    assert sha256_path(ROOT / "config" / "phase3_locked_config.json") == demo["phase3_lock_sha256"]
    assert sha256_path(ROOT / "artifacts_phase3" / "training_only_preprocessor.joblib") == locked["preprocessor"]["sha256"]
    assert sha256_path(ROOT / "artifacts_phase3" / "federated_global_model.pt") == locked["model_artifacts"]["federated_fedavg"]["sha256"]

    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    event_types = {event["event_type"] for event in events}
    required = {
        "server_started", "security_ready", "security_handshake_started", "client_authenticated",
        "all_clients_ready", "round_started", "global_model_distributed",
        "security_message_protected", "client_training_started", "client_training_completed",
        "security_message_unprotected", "client_update_received", "aggregation_started",
        "aggregation_completed", "round_completed", "demo_completed", "client_security_summary",
    }
    assert required <= event_types
    client_summaries = [event for event in events if event["event_type"] == "client_security_summary"]
    assert sorted(event["client_id"] for event in client_summaries) == expected_clients

    aggregation_difference = verify_aggregates(run_dir, status, expected_clients)
    plain_runtime = ROOT / "phase4" / "runtime"
    plain_run_id, plain_run_dir = latest_run(plain_runtime)
    plain_status = read_json(plain_run_dir / "final_summary.json")
    assert plain_status["state"] == "completed"
    equivalence_difference = compare_plain(run_dir, plain_run_dir, int(demo["rounds"]))
    evidence_files = assert_secrets_absent(runtime, run_dir)

    rejected = [event for event in events if event["event_type"] == "security_message_rejected"]
    print(
        json.dumps(
            {
                "verified": True,
                "secure_run_id": run_id,
                "plain_run_id": plain_run_id,
                "rounds": len(status["round_summaries"]),
                "authenticated_clients": len(expected_clients),
                "total_samples_per_round": 6148,
                "maximum_independent_aggregation_difference": aggregation_difference,
                "maximum_plain_secure_aggregation_difference": equivalence_difference,
                "aggregation_tolerance": 1e-7,
                "security_rejections": len(rejected),
                "rejection_categories": sorted({event["payload"]["category"] for event in rejected}),
                "frozen_phase3_hashes_unchanged": True,
                "secret_free_evidence_files_checked": evidence_files,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
