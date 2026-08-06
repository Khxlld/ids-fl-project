"""Build the Phase 5 educational guide from stable, secret-free evidence."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent


def md(value):
    return nbf.v4.new_markdown_cell(value.strip())


def code(value):
    return nbf.v4.new_code_cell(value.strip())


cells = [
    md("""
# Phase 5 guide — post-quantum-secured federated communication

This read-only guide explains the implemented security layer using saved, secret-free evidence. The Docker implementation remains in `phase5/app/`; this notebook does not provision identities, load private keys, perform training, or duplicate the cryptographic protocol.
"""),
    code("""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display

HERE = Path.cwd().resolve()
PROJECT_ROOT = HERE if (HERE / "phase5" / "results" / "benchmark_comparison.json").is_file() else HERE.parent
if not (PROJECT_ROOT / "phase5" / "results" / "benchmark_comparison.json").is_file():
    raise RuntimeError("Run from the uavids_federated root or notebooks directory")

RESULTS = PROJECT_ROOT / "phase5" / "results"
CONFIG = json.loads((PROJECT_ROOT / "phase5" / "config" / "security_config.json").read_text(encoding="utf-8"))
BENCHMARK = json.loads((RESULTS / "benchmark_comparison.json").read_text(encoding="utf-8"))
print("Loaded stable Phase 5 evidence; no runtime key directory was accessed.")
print(BENCHMARK["algorithms"])
"""),
    md("""
## Threat and trust model

The network attacker can observe, alter, inject, copy, delay, and replay HTTP application messages or claim another client name. The host, Docker engine, pinned libraries, training implementation, mounted secrets, and provisioned public trust store are trusted. Host compromise, denial of service, traffic analysis, key theft, and malicious updates from an already authenticated client are outside scope.
"""),
    md("""
## What each primitive contributes

| Primitive | Contribution |
|---|---|
| ML-KEM-768 | Establishes shared cryptographic material without sending it |
| HKDF-SHA-256 | Derives independent directional AES keys and nonce prefixes |
| AES-256-GCM | Encrypts and authenticates model requests, models, events, updates, and acknowledgements |
| ML-DSA-65 | Authenticates provisioned server/client identities and binds the handshake |

Both selected post-quantum parameter sets are NIST security category 3. Final standardized ML-KEM/ML-DSA names are used, not legacy Kyber/Dilithium names.
"""),
    md("""
## One protected exchange

```text
client                              control center
  |-- identity + registration -------->|
  |<-- signed ephemeral ML-KEM key -----|
  |-- signed KEM ciphertext ----------->|
  |<-- AES-GCM confirmation ------------|
  |-- encrypted model request ---------->|
  |<-- encrypted global model ----------|
  |   local training                    |
  |-- encrypted client update --------->|
  |   verify before model decoding      |
  |                         safe FedAvg  |
```

The signed handshake binds the client, run, frozen contract, algorithms, challenges, and KEM key/ciphertext hashes. Each AES-GCM header binds protocol version, session, run, round, sender, recipient, message type, sequence, nonce, contract hash, plaintext hash, and model hash as canonical authenticated data.
"""),
    code("""
safe_header_example = {
    "security_version": CONFIG["security_version"],
    "session_id": "example-session",
    "run_id": "example-run",
    "round": 2,
    "sender": "uav-client-1",
    "recipient": "control-center",
    "message_type": "client_update",
    "sequence": 7,
    "nonce": "<derived-prefix || uint64-sequence>",
    "contract_hash": "<frozen-contract-hash>",
    "plaintext_sha256": "<update-plaintext-hash>",
    "model_sha256": "<model-hash>",
}
display(pd.DataFrame(safe_header_example.items(), columns=["bound field", "illustrative value"]))
"""),
    md("""
## Replay, tampering, and impersonation

Each direction uses a different key. A 96-bit nonce combines a direction-specific four-byte HKDF output with an eight-byte sequence. The receiver requires the next sequence and exact nonce; AES-GCM authentication must pass before plaintext JSON or model archives are decoded. The public trust store maps one approved ML-DSA identity to each logical client, so a signature or key cannot silently migrate to another name.
"""),
    code("""
rejections = pd.read_csv(RESULTS / "rejection_summary.csv")
assert set(rejections["category"]) == {
    "authentication_failed", "malformed_payload", "metadata_mismatch",
    "nonce_invalid", "replay", "signature_invalid", "untrusted_identity"
}
display(rejections)
print(f"Controlled attack run completed with {int(rejections['count'].sum())} rejected messages; valid training still completed.")
assert BENCHMARK["attack_test"]["all_aggregate_archives_identical_to_plain"]
print("All attack-run aggregate archives were byte-identical to plain mode.")
"""),
    md("""
## Plain versus secure equivalence

The comparison is meaningful because the security adapter calls the unchanged Phase 4 coordinator. It demonstrates that protecting transport did not alter client sample counts, local updates, or FedAvg. It does not demonstrate robustness against poisoned—but correctly authenticated—updates.
"""),
    code("""
aggregation = pd.DataFrame([{
    "plain run": BENCHMARK["plain_run_id"],
    "secure run": BENCHMARK["secure_run_id"],
    "clients": BENCHMARK["clients"],
    "rounds": BENCHMARK["rounds"],
    "samples/round": BENCHMARK["samples_per_round"],
    "plain-secure max abs diff": BENCHMARK["aggregation"]["maximum_plain_secure_difference"],
    "independent secure max abs diff": BENCHMARK["aggregation"]["maximum_independent_secure_difference"],
    "tolerance": BENCHMARK["aggregation"]["tolerance"],
}])
display(aggregation)
assert BENCHMARK["aggregation"]["maximum_plain_secure_difference"] <= BENCHMARK["aggregation"]["tolerance"]
"""),
    md("""
## Measured cryptographic and message overhead

Operation timings isolate the primitives from training and container startup. These are small repeated measurements within one final run, not hardware-independent benchmarks. Full runtimes vary with container startup and concurrent CPU scheduling.
"""),
    code("""
timings = pd.read_csv(RESULTS / "crypto_operation_timings.csv")
sizes = pd.read_csv(RESULTS / "message_size_comparison.csv")
clients = pd.read_csv(RESULTS / "per_client_secure_metrics.csv")
display(timings.round(4))
display(sizes.round(2))
display(clients.round(3))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
timings.plot.bar(x="operation", y="mean", legend=False, ax=axes[0], color="#3b82f6")
axes[0].set_ylabel("Mean milliseconds")
axes[0].set_title("Cryptographic operations")
sizes.plot.bar(x="direction", y=["mean_plaintext_bytes", "mean_protected_bytes"], ax=axes[1], color=["#94a3b8", "#0f766e"])
axes[1].set_ylabel("Mean bytes")
axes[1].set_title("Model-bearing JSON messages")
axes[1].legend(["plaintext", "protected"])
plt.tight_layout()
plt.show()
"""),
    md("""
Client 3 remained the first-round training straggler despite having the smallest partition, consistent with its tighter CPU profile and one-time startup effects. AES-GCM times stayed sub-millisecond for every client; single-handshake ML-KEM/ML-DSA differences are descriptive noise, not evidence of device-specific cryptographic performance.
"""),
    code("""
runtime = BENCHMARK["runtime_seconds"]
display(pd.DataFrame({
    "mode": ["plain", "secure"],
    "coordinator seconds": [runtime["plain_coordinator"], runtime["secure_coordinator"]],
    "compose-to-terminal seconds": [runtime["plain_compose_to_terminal"], runtime["secure_compose_to_terminal"]],
}))
print("Completed normal secure coordinator repetitions:", runtime["secure_completed_repetitions"])
print(BENCHMARK["scope"])
"""),
    md("""
## Run and verify

```powershell
# unchanged plain baseline
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\\phase4\\scripts\\run_demo.ps1

# secure demo and controlled attacks
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\\phase5\\scripts\\run_secure_demo.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\\phase5\\scripts\\run_attack_tests.ps1

# regressions and observation
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\\phase5\\scripts\\run_security_tests.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\\phase5\\scripts\\observe_secure_demo.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\\phase5\\scripts\\stop_secure_demo.ps1
```
"""),
    md("""
## Valid claims and limitations

This controlled prototype demonstrates confidential, authenticated, tamper-evident, context-bound, replay-resistant model exchange and aggregation-equivalent FL. Provisioning is an academic allowlist, not production PKI; HTTP metadata is visible; keys lack certificate/revocation/HSM lifecycle; the server sees plaintext updates; and encryption does not detect model poisoning. It implements neither secure aggregation nor differential privacy and must not be described as production-ready.

A later packet-capture comparison should show readable model-bearing JSON in plain mode and only headers plus ciphertext/base64 in secure mode, while carefully stating that IP/HTTP sizes and timing remain observable.
"""),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
)
path = ROOT / "notebooks" / "07_phase5_post_quantum_security_guide.ipynb"
nbf.write(notebook, path)
print(f"Built {path}")
