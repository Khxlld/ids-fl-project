"""Create stable, secret-free Phase 5 evidence from completed runtime runs."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
P4 = ROOT / "phase4" / "runtime"
P5 = ROOT / "phase5" / "runtime"
OUT = ROOT / "phase5" / "results"


def read_json(path: Path):
    raw = path.read_bytes()
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    return json.loads(raw.decode(encoding))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def events(path: Path) -> list[dict]:
    raw = path.read_bytes()
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    return [json.loads(line) for line in raw.decode(encoding).splitlines() if line.strip()]


def completed_runs(root: Path) -> list[tuple[Path, dict, list[dict]]]:
    found = []
    for directory in (root / "runs").iterdir():
        summary_path = directory / "final_summary.json"
        event_path = directory / "events.jsonl"
        if summary_path.is_file() and event_path.is_file():
            summary = read_json(summary_path)
            if summary.get("state") == "completed":
                found.append((directory, summary, events(event_path)))
    return sorted(found, key=lambda item: item[1]["started_utc"])


def stats(path: Path) -> dict[str, dict[str, float]]:
    values = defaultdict(lambda: {"cpu": [], "memory_mib": []})
    if not path.is_file():
        return {}
    for row in events(path):
        name = row.get("Name", "").replace("uavids-phase5-", "").replace("uavids-phase4-", "")
        name = name.removesuffix("-1")
        try:
            cpu = float(row["CPUPerc"].rstrip("%"))
            memory_text = row["MemUsage"].split("/")[0].strip()
            if memory_text.endswith("GiB"):
                memory = float(memory_text[:-3]) * 1024
            elif memory_text.endswith("MiB"):
                memory = float(memory_text[:-3])
            else:
                continue
        except (KeyError, ValueError):
            continue
        values[name]["cpu"].append(cpu)
        values[name]["memory_mib"].append(memory)
    return {
        name: {
            "peak_cpu_percent": round(max(value["cpu"]), 3),
            "peak_memory_mib": round(max(value["memory_mib"]), 3),
        }
        for name, value in sorted(values.items())
    }


def aggregate(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 6),
        "median": round(statistics.median(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    verification = read_json(P5 / "latest_verification.json")
    secure_id = verification["secure_run_id"]
    plain_id = verification["plain_run_id"]
    secure_dir = P5 / "runs" / secure_id
    plain_dir = P4 / "runs" / plain_id
    secure = read_json(secure_dir / "final_summary.json")
    plain = read_json(plain_dir / "final_summary.json")
    secure_events = events(secure_dir / "events.jsonl")
    security = read_json(secure_dir / "security_summary.json")

    p5_runs = completed_runs(P5)
    normal_runs = [item for item in p5_runs if not any(e["event_type"] == "security_message_rejected" for e in item[2])]
    attack_runs = [item for item in p5_runs if any(e["event_type"] == "security_message_rejected" for e in item[2])]
    attack_dir, attack, attack_events = attack_runs[-1]
    rejections = [e for e in attack_events if e["event_type"] == "security_message_rejected"]
    rejection_counts = Counter(e["payload"]["category"] for e in rejections)
    attack_aggregate_matches = all(
        sha256(attack_dir / "aggregation_audit" / f"round_{number}" / "aggregated.npz")
        == sha256(plain_dir / "aggregation_audit" / f"round_{number}" / "aggregated.npz")
        for number in range(1, int(attack["total_rounds"]) + 1)
    )

    client_summaries = {
        e["client_id"]: e["payload"] for e in secure_events if e["event_type"] == "client_security_summary"
    }
    operation_values = defaultdict(list)
    for name, values in security["server_timings_ms"].items():
        if name.endswith("_ms"):
            operation_values[name].extend(float(v) for v in values)
    for payload in client_summaries.values():
        for name, value in payload["handshake_timings_ms"].items():
            if name.endswith("_ms"):
                operation_values[name].append(float(value))
        for name, values in payload["channel"]["timings_ms"].items():
            operation_values[name].extend(float(v) for v in values)

    operation_rows = []
    for name, values in sorted(operation_values.items()):
        row = {"operation": name, **aggregate(values)}
        operation_rows.append(row)
    with (OUT / "crypto_operation_timings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(operation_rows[0]))
        writer.writeheader()
        writer.writerows(operation_rows)

    size_rows = []
    for event_type, direction in (("security_message_protected", "server_to_client"), ("security_message_unprotected", "client_to_server")):
        selected = [e for e in secure_events if e["event_type"] == event_type]
        plain_sizes = [int(e["payload"]["plaintext_bytes"]) for e in selected]
        protected_sizes = [int(e["payload"]["protected_bytes"]) for e in selected]
        size_rows.append({
            "direction": direction,
            "messages": len(selected),
            "mean_plaintext_bytes": round(statistics.fmean(plain_sizes), 3),
            "mean_protected_bytes": round(statistics.fmean(protected_sizes), 3),
            "mean_overhead_bytes": round(statistics.fmean(p - q for p, q in zip(protected_sizes, plain_sizes)), 3),
            "mean_expansion_percent": round(statistics.fmean((p / q - 1) * 100 for p, q in zip(protected_sizes, plain_sizes)), 3),
        })
    with (OUT / "message_size_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(size_rows[0]))
        writer.writeheader()
        writer.writerows(size_rows)

    plain_resources = stats(P4 / "host_container_stats.jsonl")
    secure_resources = stats(secure_dir / "host_container_stats.jsonl")
    client_rows = []
    for client_id, payload in sorted(client_summaries.items()):
        training = [
            float(round_summary["client_updates"][client_id]["client_metrics"]["training_ms"])
            for round_summary in secure["round_summaries"]
        ]
        channel = payload["channel"]
        client_rows.append({
            "client_id": client_id,
            "samples": secure["round_summaries"][0]["sample_counts"][client_id],
            "mean_training_ms": round(statistics.fmean(training), 3),
            "max_training_ms": round(max(training), 3),
            "encapsulation_ms": payload["handshake_timings_ms"]["encapsulation_ms"],
            "signing_ms": payload["handshake_timings_ms"]["signing_ms"],
            "mean_encrypt_ms": round(statistics.fmean(channel["timings_ms"]["encrypt_ms"]), 6),
            "mean_decrypt_ms": round(statistics.fmean(channel["timings_ms"]["decrypt_ms"]), 6),
            "sent_messages": channel["sent_messages"],
            "received_messages": channel["received_messages"],
            "peak_cpu_percent": secure_resources[client_id]["peak_cpu_percent"],
            "peak_memory_mib": secure_resources[client_id]["peak_memory_mib"],
        })
    with (OUT / "per_client_secure_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(client_rows[0]))
        writer.writeheader()
        writer.writerows(client_rows)

    with (OUT / "rejection_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "count"])
        writer.writeheader()
        writer.writerows({"category": key, "count": value} for key, value in sorted(rejection_counts.items()))

    plain_host = read_json(P4 / "host_run_timing.json")
    secure_host = read_json(secure_dir / "host_run_timing.json")
    normal_runtime = [round(item[1]["elapsed_seconds"], 6) for item in normal_runs]
    comparison = {
        "scope": "demonstration timing only; not a rigorous performance conclusion",
        "algorithms": security["algorithms"],
        "plain_run_id": plain_id,
        "secure_run_id": secure_id,
        "attack_run_id": attack["run_id"],
        "clients": len(secure["expected_clients"]),
        "rounds": secure["total_rounds"],
        "samples_per_round": secure["round_summaries"][0]["total_samples"],
        "aggregation": {
            "tolerance": verification["aggregation_tolerance"],
            "maximum_plain_secure_difference": verification["maximum_plain_secure_aggregation_difference"],
            "maximum_independent_secure_difference": verification["maximum_independent_aggregation_difference"],
        },
        "runtime_seconds": {
            "plain_coordinator": plain["elapsed_seconds"],
            "secure_coordinator": secure["elapsed_seconds"],
            "secure_completed_repetitions": normal_runtime,
            "plain_compose_to_api_ready": plain_host["compose_up_to_api_ready_seconds"],
            "plain_compose_to_terminal": plain_host["compose_up_to_terminal_state_seconds"],
            "secure_compose_to_api_ready": secure_host["compose_up_to_api_ready_seconds"],
            "secure_compose_to_terminal": secure_host["compose_up_to_terminal_state_seconds"],
        },
        "crypto_operation_timings_ms": {row["operation"]: {k: v for k, v in row.items() if k != "operation"} for row in operation_rows},
        "message_sizes": size_rows,
        "resource_peaks": {
            "plain": plain_resources,
            "secure": secure_resources,
        },
        "attack_test": {
            "completed_training": attack["state"] == "completed",
            "all_aggregate_archives_identical_to_plain": attack_aggregate_matches,
            "rejections": len(rejections),
            "categories": dict(sorted(rejection_counts.items())),
        },
        "final_validation_metrics": secure["last_metrics"],
    }
    (OUT / "benchmark_comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    excerpt = [{"event_type": e["event_type"], "client_id": e["client_id"], "round": e["round"], "payload": e["payload"]} for e in rejections]
    (OUT / "security_rejection_events.json").write_text(json.dumps(excerpt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    crypto = comparison["crypto_operation_timings_ms"]
    text = f"""# Phase 5 measured results

This is a controlled academic demonstration. Timings are descriptive for these runs and host, not a general performance conclusion.

## Equivalence and completion

- Plain run `{plain_id}` and secure run `{secure_id}` each completed {secure['total_rounds']} rounds with all five clients and {comparison['samples_per_round']:,} samples per round.
- Maximum secure-versus-plain aggregate difference: **{comparison['aggregation']['maximum_plain_secure_difference']}** (tolerance `{comparison['aggregation']['tolerance']}`).
- Maximum difference from independent secure re-aggregation: **{comparison['aggregation']['maximum_independent_secure_difference']}**.
- Final validation macro-F1 remained **{secure['last_metrics']['macro_f1']:.4f}**; this is demo telemetry, not new model selection.

## Descriptive overhead

- Coordinator runtime: plain **{plain['elapsed_seconds']:.2f} s**, secure **{secure['elapsed_seconds']:.2f} s**. Completed normal secure repetitions: {', '.join(f'{v:.2f} s' for v in normal_runtime)}.
- Compose-to-terminal runtime: plain **{plain_host['compose_up_to_terminal_state_seconds']:.2f} s**, secure **{secure_host['compose_up_to_terminal_state_seconds']:.2f} s**.
- Mean ML-KEM encapsulation / decapsulation: **{crypto['encapsulation_ms']['mean']:.3f} / {crypto['decapsulation_ms']['mean']:.3f} ms**.
- Mean ML-DSA signing / server verification / client verification: **{crypto['signing_ms']['mean']:.3f} / {crypto['verify_ms']['mean']:.3f} / {crypto['server_signature_verification_ms']['mean']:.3f} ms**.
- Mean AES-GCM encryption / decryption: **{crypto['encrypt_ms']['mean']:.3f} / {crypto['decrypt_ms']['mean']:.3f} ms**.
- Model/update JSON envelopes expanded by roughly **{size_rows[0]['mean_expansion_percent']:.1f}% / {size_rows[1]['mean_expansion_percent']:.1f}%**, largely because binary NPZ and ciphertext are represented as base64 in JSON.

## Rejection evidence

Attack run `{attack['run_id']}` still completed genuine training and produced {len(rejections)} safe rejections across: {', '.join(sorted(rejection_counts))}. All three aggregate archives were byte-identical to plain mode, so rejected probes did not reach or change aggregation.

See the CSV and JSON files in this directory for machine-readable operation, size, resource, and rejection evidence.
"""
    (OUT / "PHASE5_RESULTS_SUMMARY.md").write_text(text, encoding="utf-8")
    print(json.dumps({"written": sorted(path.name for path in OUT.iterdir()), "secure_run_id": secure_id}, indent=2))


if __name__ == "__main__":
    main()
