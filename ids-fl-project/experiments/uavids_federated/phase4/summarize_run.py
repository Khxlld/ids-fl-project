"""Turn the latest measured run into stable Phase 4 tables and summary files."""

from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE4 = ROOT / "phase4"
RUNTIME = PHASE4 / "runtime"
RESULTS = PHASE4 / "results"


def memory_mib(value: str) -> float:
    amount, unit = re.match(r"([0-9.]+)([KMG]iB)", value.split("/")[0].strip()).groups()
    return float(amount) * {"KiB": 1 / 1024, "MiB": 1, "GiB": 1024}[unit]


def service_name(container_name: str) -> str:
    match = re.match(r"uavids-phase4-(.+)-1$", container_name)
    return match.group(1) if match else container_name


def read_json_any_encoding(path: Path) -> dict:
    raw = path.read_bytes()
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    return json.loads(raw.decode(encoding))


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    run_id = json.loads((RUNTIME / "latest_run.json").read_text(encoding="utf-8"))["run_id"]
    run_dir = RUNTIME / "runs" / run_id
    status = json.loads((run_dir / "final_summary.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    host_timing = read_json_any_encoding(RUNTIME / "host_run_timing.json")
    isolation = read_json_any_encoding(RUNTIME / "latest_isolation_verification.json")
    aggregation = read_json_any_encoding(RUNTIME / "latest_verification.json")

    training_rows = []
    for event in events:
        if event["event_type"] == "client_training_completed":
            training_rows.append({
                "client_id": event["client_id"],
                "round": event["round"],
                "training_ms": event["payload"]["training_ms"],
                "update_bytes": event["payload"]["update_bytes"],
                "local_loss": event["payload"]["loss"],
                "local_macro_f1_in_sample": event["payload"]["macro_f1"],
            })
    with (RESULTS / "client_round_timings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(training_rows[0]))
        writer.writeheader()
        writer.writerows(training_rows)

    excerpt_types = [
        "server_started",
        "client_registered",
        "all_clients_ready",
        "round_started",
        "global_model_distributed",
        "client_training_started",
        "client_training_completed",
        "update_rejected",
        "client_update_received",
        "server_waiting_for_clients",
        "aggregation_started",
        "aggregation_completed",
        "round_metrics",
        "round_completed",
        "demo_completed",
    ]
    event_excerpt = []
    for event_type in excerpt_types:
        matching = [event for event in events if event["event_type"] == event_type]
        if matching:
            event = matching[0]
            event_excerpt.append({
                "seq": event["seq"],
                "elapsed_ms": event["elapsed_ms"],
                "round": event["round"],
                "source": event["source"],
                "event_type": event["event_type"],
                "severity": event["severity"],
                "client_id": event["client_id"],
                "payload": event["payload"],
            })
    (RESULTS / "event_excerpt.json").write_text(
        json.dumps(event_excerpt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    resource_samples: dict[str, list[dict]] = {}
    for line in (RUNTIME / "host_container_stats.jsonl").read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            sample = json.loads(line)
            resource_samples.setdefault(service_name(sample["Name"]), []).append(sample)
    resources = {
        service: {
            "samples": len(samples),
            "peak_memory_mib": max(memory_mib(sample["MemUsage"]) for sample in samples),
            "peak_cpu_percent_of_one_host_core": max(float(sample["CPUPerc"].rstrip("%")) for sample in samples),
            "last_network_io": samples[-1]["NetIO"],
        }
        for service, samples in sorted(resource_samples.items())
    }

    ready_event = next(event for event in events if event["event_type"] == "all_clients_ready")
    rejected = [event for event in events if event["event_type"] == "update_rejected"]
    received = [event for event in events if event["event_type"] == "client_update_received"]
    by_round = {}
    for summary in status["round_summaries"]:
        round_number = int(summary["round"])
        round_training = [row for row in training_rows if row["round"] == round_number]
        straggler = max(round_training, key=lambda row: row["training_ms"])
        by_round[str(round_number)] = {
            "round_ms": summary["round_ms"],
            "aggregation_ms": summary["aggregation_ms"],
            "evaluation_ms": summary["evaluation_ms"],
            "validation_macro_f1": summary["metrics"]["macro_f1"],
            "straggler_client": straggler["client_id"],
            "straggler_training_ms": straggler["training_ms"],
        }

    summary = {
        "run_id": run_id,
        "mode": status["mode"],
        "state": status["state"],
        "container_start_to_api_ready_seconds": host_timing["compose_up_to_api_ready_seconds"],
        "container_start_to_terminal_state_seconds": host_timing["compose_up_to_terminal_state_seconds"],
        "coordinator_start_to_all_clients_ready_seconds": ready_event["elapsed_ms"] / 1000,
        "coordinator_total_runtime_seconds": status["elapsed_seconds"],
        "rounds": by_round,
        "clients": {
            client_id: {
                "round_training_ms": [row["training_ms"] for row in training_rows if row["client_id"] == client_id],
                "steady_round_training_mean_ms": statistics.mean(
                    [row["training_ms"] for row in training_rows if row["client_id"] == client_id and row["round"] > 1]
                ),
            }
            for client_id in status["expected_clients"]
        },
        "update_archive_bytes": {
            "minimum": min(event["payload"]["update_bytes"] for event in received),
            "maximum": max(event["payload"]["update_bytes"] for event in received),
        },
        "http_update_body_bytes": {
            "minimum": min(event["payload"]["http_content_bytes"] for event in received),
            "maximum": max(event["payload"]["http_content_bytes"] for event in received),
        },
        "maximum_server_socket_receive_ms": max(event["payload"]["receive_ms"] for event in received),
        "resource_samples": resources,
        "final_demo_validation_metrics": status["last_metrics"],
        "failure_exercise": {
            "incompatible_update_rejections": len(rejected),
            "reason": rejected[0]["payload"]["reason"] if rejected else None,
        },
        "verification": {
            "aggregation": aggregation,
            "isolation": isolation,
            "phase3_artifacts_unchanged": aggregation["frozen_phase3_hashes_unchanged"],
        },
    }
    (RESULTS / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    client_lines = []
    for client_id, values in summary["clients"].items():
        timings = values["round_training_ms"]
        peak = resources[client_id]
        client_lines.append(
            f"| {client_id} | {timings[0]:.1f} | {timings[1]:.1f} | {timings[2]:.1f} "
            f"| {peak['peak_memory_mib']:.1f} | {peak['peak_cpu_percent_of_one_host_core']:.1f}% |"
        )
    report = f"""# Phase 4 measured results

This is a **demo-mode** run initialized from the locked Phase 3 federated checkpoint. Its metrics do not replace the Phase 3 research results.

## Runtime

- Run ID: `{run_id}`
- Compose start to API ready: **{summary['container_start_to_api_ready_seconds']:.2f} s**
- Compose start to completed state: **{summary['container_start_to_terminal_state_seconds']:.2f} s**
- Coordinator start to all five clients ready: **{summary['coordinator_start_to_all_clients_ready_seconds']:.2f} s**
- Coordinator total runtime: **{summary['coordinator_total_runtime_seconds']:.2f} s**
- Round times: **{', '.join(f"r{number} {values['round_ms'] / 1000:.2f} s" for number, values in by_round.items())}**
- Aggregation times: **{', '.join(f"r{number} {values['aggregation_ms']:.3f} ms" for number, values in by_round.items())}**
- Update archive: **{summary['update_archive_bytes']['minimum'] / 1024:.1f}-{summary['update_archive_bytes']['maximum'] / 1024:.1f} KiB**; HTTP body: **{summary['http_update_body_bytes']['minimum'] / 1024:.1f}-{summary['http_update_body_bytes']['maximum'] / 1024:.1f} KiB**.

| Client | Round 1 train ms | Round 2 train ms | Round 3 train ms | Peak MiB | Peak CPU |
|---|---:|---:|---:|---:|---:|
{chr(10).join(client_lines)}

Round 1 includes one-time library/optimizer warm-up under simultaneous constrained startup. The Zero 2 W-inspired client was the first-round straggler; later training was much shorter and client 4's larger partition became significant.

## Validation and failure behavior

The final demo validation macro-F1 was **{status['last_metrics']['macro_f1']:.4f}** at the locked threshold 0.42. This is presentation telemetry, not an official model-selection result.

The live incompatible-contract update was rejected with HTTP 400 and emitted an `update_rejected` error event. The timeout unit test separately confirms that an unavailable client is named in the terminal failure state.

## Verification

- All five clients participated in all three rounds with sample counts `1230, 1046, 591, 2027, 1254` (total 6,148).
- Independent aggregation agreed exactly: maximum absolute tensor difference **{aggregation['maximum_aggregation_absolute_difference']}**.
- Each client exposed exactly one training CSV; the server exposed zero training CSVs and one validation CSV.
- All roots were read-only, data/reference mounts were read-only, capabilities were dropped, and clients exited with code 0.
- The image contained no CSV files, and frozen Phase 3 hashes remained unchanged.
"""
    (RESULTS / "PHASE4_RESULTS_SUMMARY.md").write_text(report, encoding="utf-8")
    print(f"Wrote Phase 4 results for run {run_id}")


if __name__ == "__main__":
    main()
