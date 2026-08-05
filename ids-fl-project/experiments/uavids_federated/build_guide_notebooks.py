"""Build the concise Phase 3 and Phase 4 educational guide notebooks."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOKS = ROOT / "notebooks"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def notebook(cells):
    return nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
    )


phase3_cells = [
    markdown(
        """
# Phase 3 guide — binary local, centralized, and federated models

This notebook explains the completed Normal-versus-Attack experiment from its
saved configuration, histories, checkpoints, and results. It does **not** train,
refit preprocessing, reopen model selection, or load the final-test CSV. The
reliable implementation remains in `run_phase3_development.py`,
`run_phase3_final.py`, and `src/uavids_fl/`.
"""
    ),
    markdown(
        """
## 1. Configuration and portable paths

Run from either `experiments/uavids_federated` or its `notebooks` directory.
The path helper also normalizes Windows paths stored in manifests when the
notebook is opened on Linux.
"""
    ),
    code(
        """
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import Markdown, display

HERE = Path.cwd().resolve()
if (HERE / "config" / "phase3_locked_config.json").is_file():
    PROJECT_ROOT = HERE
elif HERE.name == "notebooks" and (HERE.parent / "config" / "phase3_locked_config.json").is_file():
    PROJECT_ROOT = HERE.parent
else:
    raise RuntimeError(f"Run from the uavids_federated root or notebooks directory, not {HERE}")

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from uavids_fl import BinaryMLP, fedavg

def repo_path(value):
    return PROJECT_ROOT.joinpath(*str(value).replace("\\\\", "/").split("/"))

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

lock = json.loads((PROJECT_ROOT / "config" / "phase3_locked_config.json").read_text(encoding="utf-8"))
phase2 = json.loads((PROJECT_ROOT / "results_phase2" / "partition_manifest.json").read_text(encoding="utf-8"))
features = lock["features"]

print("Project root resolved successfully; all experiment inputs use repository-relative paths.")
print(f"Frozen design: {lock['design_version']}; features: {len(features)}; seed: {lock['seed']}")
print("Final test policy: read saved metrics only — test.csv is not loaded by this notebook.")
"""
    ),
    markdown(
        """
## 2. Verify the Phase 2 handoff

Hashes protect the exact five client partitions, validation set, and locked test
file. Hashing the test file verifies identity without parsing its records.
"""
    ),
    code(
        """
partition_checks = []
for name, record in phase2["artifact_checksums"].items():
    if not name.startswith("partition::"):
        continue
    path = repo_path(record["path"])
    actual = sha256(path)
    partition_checks.append({
        "partition": name.removeprefix("partition::"),
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256_ok": actual == record["sha256"],
    })
partition_checks = pd.DataFrame(partition_checks)
assert partition_checks["sha256_ok"].all()
display(partition_checks)

distribution = pd.DataFrame(phase2["partition_distributions"])[
    ["partition", "rows", "normal", "attack", "attack_pct", "sources"]
]
display(distribution.round({"attack_pct": 2}))
"""
    ),
    markdown(
        """
The training clients are deliberately non-IID logical source groups: they range
from 591 to 2,027 rows and roughly 50% to 90% attack traffic. They are not
verified physical UAV identities.
"""
    ),
    markdown(
        """
## 3. Training-only preprocessing

Median imputation and `StandardScaler` were fitted once on the pooled 6,148
training-client rows. Validation and test were transform-only. Pooling prevents
evaluation leakage, but centrally accessing client features is not a private
federated-statistics protocol.
"""
    ),
    code(
        """
metadata = json.loads((PROJECT_ROOT / "artifacts_phase3" / "preprocessing_metadata.json").read_text(encoding="utf-8"))
preprocessor_path = repo_path(lock["preprocessor"]["path"])
assert sha256(preprocessor_path) == lock["preprocessor"]["sha256"]
preprocessor = joblib.load(preprocessor_path)

preprocessing_table = pd.DataFrame({
    "feature": features,
    "training_median": metadata["imputer_statistics"],
    "training_mean": metadata["scaler_mean"],
    "training_scale": metadata["scaler_scale"],
})
display(preprocessing_table.round(5))

first_client_record = phase2["artifact_checksums"]["partition::train/uav-client-1"]
sample = pd.read_csv(repo_path(first_client_record["path"]), nrows=3)
transformed = pd.DataFrame(preprocessor.transform(sample[features]), columns=features)
demonstration = pd.concat(
    [sample[features[:5]].add_prefix("raw: "), transformed[features[:5]].add_prefix("scaled: ")], axis=1
)
display(demonstration.round(4))
"""
    ),
    markdown(
        """
The demonstration applies the saved transformer; it never calls `fit`. The
feature order is part of the compatibility contract and identifiers, addresses,
ports, protocol, and `MeanPacketSize` remain excluded.
"""
    ),
    markdown(
        """
## 4. Selected MLP and optimization policy

The model is intentionally small enough for CPU clients while retaining the
reliable Edge-IIoT hidden-layer pattern.
"""
    ),
    code(
        """
candidate = lock["selected_candidate"]
model = BinaryMLP(len(features), candidate["hidden_layers"], candidate["dropout"])
checkpoint_record = lock["model_artifacts"]["federated_fedavg"]
federated_checkpoint = torch.load(repo_path(checkpoint_record["path"]), map_location="cpu", weights_only=True)
model.load_state_dict(federated_checkpoint["state_dict"])

model_policy = pd.DataFrame([
    ("architecture", f"15 -> {' -> '.join(map(str, candidate['hidden_layers']))} -> 1"),
    ("dropout", str(candidate["dropout"])),
    ("loss", candidate["loss_policy"]),
    ("optimizer", lock["optimizer"]["name"]),
    ("learning rate", lock["optimizer"]["learning_rate"]),
    ("weight decay", lock["optimizer"]["weight_decay"]),
    ("batch size", lock["batch_size"]),
    ("trainable parameters", sum(parameter.numel() for parameter in model.parameters())),
], columns=["setting", "value"])
display(model_policy)
print(model)
"""
    ),
    markdown(
        """
## 5. How validation selected the design

Four controlled candidates compared two widths and unweighted versus globally
weighted loss. Checkpoints maximized validation macro-F1 at threshold 0.5;
model-specific decision thresholds were selected only after checkpoint choice.
"""
    ),
    code(
        """
candidates = pd.read_csv(PROJECT_ROOT / "results_phase3" / "candidate_validation_comparison.csv")
candidate_view = candidates[[
    "candidate_id", "loss_policy", "best_epoch", "selected_threshold",
    "macro_f1", "attack_recall", "fpr", "selected",
]].copy()
display(candidate_view.round(4))

fig, axis = plt.subplots(figsize=(8, 3.5))
colors = ["#2a9d8f" if value else "#9aa0a6" for value in candidates["selected"]]
axis.bar(candidates["candidate_id"], candidates["macro_f1"], color=colors)
axis.set_ylim(0.965, 0.98)
axis.set_ylabel("validation macro-F1")
axis.tick_params(axis="x", rotation=20)
axis.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        """
The selected unweighted 128–64–32 model improved macro-F1 and reduced false
positives. Positive-class weighting slightly increased recall in one candidate,
but its higher FPR reduced the balanced macro-F1 objective.
"""
    ),
    markdown(
        """
## 6. Three fair training paths

- **Local-only:** each model sees one client partition and uses the common
  validation set for checkpoint/threshold selection.
- **Centralized:** the same initialization and optimizer see all pooled training
  rows.
- **FedAvg:** every round sends the same global state to all five clients; each
  trains for two epochs with reset optimizer state; sample counts weight updates.
"""
    ),
    code(
        """
central_history = pd.read_csv(PROJECT_ROOT / "results_phase3" / "centralized_training_history.csv")
local_history = pd.read_csv(PROJECT_ROOT / "results_phase3" / "local_only_training_history.csv")
federated_history = pd.read_csv(PROJECT_ROOT / "results_phase3" / "federated_round_history.csv")
central_checkpoint = torch.load(
    repo_path(lock["model_artifacts"]["centralized"]["path"]), map_location="cpu", weights_only=True
)

history_summary = pd.DataFrame([
    {
        "path": "centralized",
        "units": central_checkpoint["epochs_executed"],
        "selected_unit": central_checkpoint["best_epoch"],
        "unit_name": "epoch",
    },
    *[
        {
            "path": f"local/{client_id}",
            "units": lock["validation_metrics"][f"local/{client_id}"]["epochs_executed"],
            "selected_unit": lock["validation_metrics"][f"local/{client_id}"]["best_epoch"],
            "unit_name": "epoch",
        }
        for client_id, group in local_history.groupby("client_id")
    ],
    {
        "path": "federated_fedavg",
        "units": federated_checkpoint["rounds_executed"],
        "selected_unit": federated_checkpoint["best_round"],
        "unit_name": "round",
    },
])
display(history_summary)
"""
    ),
    markdown(
        """
### A small FedAvg calculation

The production helper is demonstrated below on one scalar per client using the
real sample counts. The same operation is applied tensor-by-tensor to the MLP.
"""
    ),
    code(
        """
counts = distribution.loc[distribution["partition"].str.startswith("train/"), "rows"].astype(int).tolist()
toy_states = [OrderedDict(weight=torch.tensor([float(index)])) for index in range(1, 6)]
toy_average = fedavg(toy_states, counts)["weight"].item()
manual_average = np.average(np.arange(1, 6, dtype=float), weights=counts)

display(pd.DataFrame({"client": range(1, 6), "sample_count": counts, "toy_update": range(1, 6)}))
print(f"FedAvg helper: {toy_average:.6f}; independent weighted mean: {manual_average:.6f}")
assert np.isclose(toy_average, manual_average, atol=1e-7)
"""
    ),
    markdown(
        """
## 7. Frozen validation comparison

Validation finalized the architecture, checkpoints, and thresholds. The final
test played no role in these choices.
"""
    ),
    code(
        """
validation = pd.read_csv(PROJECT_ROOT / "results_phase3" / "validation_model_comparison.csv")
metrics = ["model", "threshold", "accuracy", "macro_f1", "attack_precision", "attack_recall", "fpr"]
display(validation[metrics].round(4))

local_validation = validation[validation["model"].str.startswith("local/")]
print(f"Local mean macro-F1: {local_validation['macro_f1'].mean():.4f}")
print(f"Centralized – FedAvg macro-F1 gap: {validation.iloc[0]['macro_f1'] - validation.iloc[1]['macro_f1']:.4f}")
"""
    ),
    markdown(
        """
## 8. Locked-test evaluation — reported separately

This cell reads the saved result table, not `test.csv`. Frozen thresholds and
checkpoints were applied once after the lock was written.
"""
    ),
    code(
        """
test_metrics = pd.read_csv(PROJECT_ROOT / "results_phase3" / "locked_test_model_metrics.csv")
display(test_metrics[
    ["model", "threshold", "accuracy", "macro_f1", "attack_precision", "attack_recall", "fpr", "tn", "fp", "fn", "tp"]
].round(4))

local_test = test_metrics[test_metrics["model"].str.startswith("local/")]
central = test_metrics.loc[test_metrics["model"].eq("centralized")].iloc[0]
fed = test_metrics.loc[test_metrics["model"].eq("federated_fedavg")].iloc[0]
print(f"FedAvg vs local mean macro-F1: {fed.macro_f1 - local_test.macro_f1.mean():+.4f}")
print(f"FedAvg vs centralized macro-F1: {fed.macro_f1 - central.macro_f1:+.4f}")
"""
    ),
    markdown(
        """
The result is mixed: federation improves on the average isolated client, but
centralized pooling remains stronger and has a much lower false-positive rate.
This demonstrates behavior for these fixed logical sources—not privacy,
network security, physical UAV deployment, or statistical superiority over
multiple datasets and seeds.
"""
    ),
    markdown(
        """
## 9. Client and attack-family weaknesses

Aggregate binary scores can hide calibration and family-level failures.
"""
    ),
    code(
        """
family = pd.read_csv(PROJECT_ROOT / "results_phase3" / "locked_test_original_class_detection.csv")
family_view = family[family["model"].isin(["centralized", "federated_fedavg"])].pivot(
    index="original_label", columns="model", values="predicted_attack_rate"
)
display(family_view.round(4))

worst_local = local_test.loc[local_test["macro_f1"].idxmin()]
best_local = local_test.loc[local_test["macro_f1"].idxmax()]
print(f"Best local: {best_local.model}, macro-F1={best_local.macro_f1:.4f}")
print(f"Worst local: {worst_local.model}, macro-F1={worst_local.macro_f1:.4f}, FN={int(worst_local.fn):,}")
"""
    ),
    markdown(
        """
Wormhole is the weakest attack family for centralized and FedAvg. Local
thresholds range from 0.22 to 0.90, and client 5 is substantially weaker on the
common held-out data—evidence of non-IID sensitivity and unstable calibration.
"""
    ),
    markdown(
        """
## 10. Reproducibility check

The project verifier checks the frozen lock, model/preprocessor hashes, saved
prediction columns, recomputed metrics, confusion counts, and plots without
opening the test partition.
"""
    ),
    code(
        """
import verify_phase3_artifacts

verify_phase3_artifacts.main()
"""
    ),
    markdown(
        """
## Phase 3 takeaway

Carry the exact feature order, fitted transformer, architecture, global
checkpoint, threshold, sample-weighted aggregation rule, and version/hash checks
forward. Keep calling the clients logical source groups and disclose that the
shared preprocessing statistics were fitted centrally.
"""
    ),
]


phase4_cells = [
    markdown(
        """
# Phase 4 guide — six-container federated-learning demonstration

This notebook explains the completed Docker Compose demo from checked-in
configuration and measured results. It does not replace the control-center or
client processes, and it runs successfully even when Docker is stopped. Live
operation remains in `phase4/app/`, `phase4/docker-compose.yml`, and
`phase4/scripts/`.
"""
    ),
    markdown(
        """
## 1. Configuration and saved evidence

The guide uses stable benchmark and event excerpts generated from the definitive
run. Raw runtime logs remain disposable operational evidence.
"""
    ),
    code(
        """
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import Markdown, display

HERE = Path.cwd().resolve()
if (HERE / "phase4" / "docker-compose.yml").is_file():
    PROJECT_ROOT = HERE
elif HERE.name == "notebooks" and (HERE.parent / "phase4" / "docker-compose.yml").is_file():
    PROJECT_ROOT = HERE.parent
else:
    raise RuntimeError(f"Run from the uavids_federated root or notebooks directory, not {HERE}")

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from phase4.app.common import build_contract, decode_state, encode_state, sha256_path, state_spec
from uavids_fl import BinaryMLP

demo = json.loads((PROJECT_ROOT / "phase4" / "config" / "demo_config.json").read_text(encoding="utf-8"))
lock = json.loads((PROJECT_ROOT / "config" / "phase3_locked_config.json").read_text(encoding="utf-8"))
benchmark = json.loads((PROJECT_ROOT / "phase4" / "results" / "benchmark_summary.json").read_text(encoding="utf-8"))
events = json.loads((PROJECT_ROOT / "phase4" / "results" / "event_excerpt.json").read_text(encoding="utf-8"))
timings = pd.read_csv(PROJECT_ROOT / "phase4" / "results" / "client_round_timings.csv")

assert demo["mode"] == "demo_nonresearch"
assert benchmark["state"] == "completed"
print(f"Configuration: {demo['config_version']}; protocol: {demo['protocol_version']}")
print(f"Definitive run: {benchmark['run_id']}; state: {benchmark['state']}")
print("Docker does not need to be running to execute this guide.")
"""
    ),
    markdown(
        """
## 2. Six-container architecture

```text
five isolated client containers             control-center container
one read-only train.csv each                 validation.csv only
        |                                            |
        +-- register and verify contract ---------->|
        |<----------- global model -----------------+
        +-- real local CPU training                 |
        +-- validated model update ---------------->|
                                                     +-- sample-weighted FedAvg
                                                     +-- metrics and GUI events
```

All communication is ordinary HTTP on one Compose bridge network. The server
never reads a client training CSV; clients never share their CSVs with one
another.
"""
    ),
    code(
        """
service_table = pd.DataFrame([
    {"service": "control-center", "role": "coordinate, aggregate, validate", "data": "validation.csv", **demo["server"]},
    *[
        {
            "service": item["client_id"],
            "role": "local training",
            "data": item["partition_filename"],
            "cpus": item["cpus"],
            "memory": item["memory"],
        }
        for item in demo["clients"]
    ],
])
display(service_table)
"""
    ),
    markdown(
        """
## 3. Device-inspired profiles

The names provide an understandable range of edge classes. Docker limits are
process allocations, not the products' full specifications or hardware
emulation. Product sources and the exact mapping rationale are recorded in
[`phase4/DEVICE_PROFILES.md`](../phase4/DEVICE_PROFILES.md).
"""
    ),
    code(
        """
profile_rows = []
for item in demo["clients"]:
    measured = benchmark["resource_samples"][item["client_id"]]
    profile_rows.append({
        "client": item["client_id"],
        "logical_rows": item["samples"],
        "profile": item["profile"],
        "CPU limit": item["cpus"],
        "memory limit MiB": int(item["memory"].removesuffix("m")),
        "measured peak MiB": measured["peak_memory_mib"],
        "measured peak CPU %": measured["peak_cpu_percent_of_one_host_core"],
    })
profiles = pd.DataFrame(profile_rows)
display(profiles.round(1))

fig, axis = plt.subplots(figsize=(8, 3.5))
axis.bar(profiles["client"], profiles["memory limit MiB"], label="limit", color="#b7c9e2")
axis.bar(profiles["client"], profiles["measured peak MiB"], label="measured peak", color="#2a9d8f")
axis.set_ylabel("MiB")
axis.legend()
axis.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        """
The five inspirations are Raspberry Pi 4, Jetson Nano, Raspberry Pi Zero 2 W,
Jetson Orin Nano, and NXP NavQPlus. The definitive run needed no increase from
the initial limits. The x86 containers cannot support claims about ARM timing,
GPU/NPU acceleration, power, thermal behavior, radio links, or flight hardware.
"""
    ),
    markdown(
        """
## 4. Compatibility contract

Every service independently derives the same contract from the frozen Phase 3
lock and the Phase 4 demo configuration.
"""
    ),
    code(
        """
contract = build_contract(lock, demo)
contract_view = pd.DataFrame([
    ("contract hash", contract["contract_hash"]),
    ("Phase 3 design", contract["phase3_design_version"]),
    ("features", len(contract["feature_order"])),
    ("architecture", f"{contract['input_dim']} -> {' -> '.join(map(str, contract['hidden_layers']))} -> 1"),
    ("local epochs", contract["local_epochs"]),
    ("aggregation", contract["aggregation"]),
    ("decision threshold", contract["decision_threshold"]),
], columns=["contract field", "value"])
display(contract_view)

assert sha256_path(PROJECT_ROOT / "config" / "phase3_locked_config.json") == demo["phase3_lock_sha256"]
assert sha256_path(PROJECT_ROOT / "artifacts_phase3" / "training_only_preprocessor.joblib") == lock["preprocessor"]["sha256"]
print("Frozen lock and preprocessor hashes verified.")
"""
    ),
    markdown(
        """
## 5. One federated round, event by event

The excerpt below is derived from the saved JSONL event stream. Sequence numbers
allow a future GUI to reconnect with `after_seq` without replaying everything.
"""
    ),
    code(
        """
event_table = pd.DataFrame(events).sort_values("seq")
event_table["payload summary"] = event_table["payload"].apply(
    lambda payload: ", ".join(f"{key}={value}" for key, value in list(payload.items())[:3])
)
display(event_table[["seq", "elapsed_ms", "round", "source", "event_type", "severity", "payload summary"]])
"""
    ),
    markdown(
        """
A round begins only after all clients register. The server distributes one
global state, clients report training start/finish, updates are checked and
stored in a runtime audit, the server names missing clients while waiting, and
aggregation starts only after all five valid updates arrive.
"""
    ),
    markdown(
        """
## 6. Sample-weighted aggregation

The server uses the Phase 3 `fedavg` helper and expected manifest counts—not a
client's untrusted choice of weight.
"""
    ),
    code(
        """
counts = pd.DataFrame(demo["clients"])[["client_id", "samples"]]
counts["FedAvg weight"] = counts["samples"] / counts["samples"].sum()
display(counts.round({"FedAvg weight": 4}))
print(f"Rows aggregated each round: {counts['samples'].sum():,}; weights sum: {counts['FedAvg weight'].sum():.1f}")
"""
    ),
    markdown(
        """
The live verifier independently reopened each saved client update and recomputed
every tensor. The maximum difference from the containerized aggregate was zero.
"""
    ),
    markdown(
        """
## 7. Safe model transport demonstration

Model updates are compressed NumPy archives, never network-supplied pickle.
Names, order, shapes, `float32` dtypes, finite values, hash, and size are checked.
"""
    ),
    code(
        """
candidate = lock["selected_candidate"]
toy_model = BinaryMLP(len(lock["features"]), candidate["hidden_layers"], candidate["dropout"])
specification = state_spec(toy_model.state_dict())
encoded, archive_bytes, archive_hash = encode_state(toy_model.state_dict())
decoded, _, decoded_hash = decode_state(encoded, specification, demo["maximum_update_bytes"])
maximum_difference = max(
    float(torch.max(torch.abs(decoded[name] - value)).item())
    for name, value in toy_model.state_dict().items()
)

display(pd.DataFrame(specification).head())
print(f"Parameters: {len(specification)} tensors; archive: {archive_bytes / 1024:.1f} KiB")
print(f"SHA-256 round trip: {archive_hash == decoded_hash}; maximum tensor difference: {maximum_difference}")
"""
    ),
    markdown(
        """
## 8. Measured runtime and stragglers

Round 1 includes simultaneous import and optimizer warm-up. Later rounds better
show data-size and CPU-limit effects.
"""
    ),
    code(
        """
timing_pivot = timings.pivot(index="client_id", columns="round", values="training_ms")
timing_pivot.columns = [f"round {value} ms" for value in timing_pivot.columns]
display(timing_pivot.round(1))

rounds = pd.DataFrame.from_dict(benchmark["rounds"], orient="index").reset_index(names="round")
display(rounds[["round", "round_ms", "aggregation_ms", "evaluation_ms", "straggler_client", "straggler_training_ms", "validation_macro_f1"]].round(3))

print(f"Compose start → API ready: {benchmark['container_start_to_api_ready_seconds']:.2f} s")
print(f"Compose start → completed: {benchmark['container_start_to_terminal_state_seconds']:.2f} s")
print(f"Update archive range: {benchmark['update_archive_bytes']['minimum']/1024:.1f}–{benchmark['update_archive_bytes']['maximum']/1024:.1f} KiB")
"""
    ),
    markdown(
        """
The Zero 2 W-inspired client was the first-round straggler. By round 3, the
largest partition on client 4 took longest. These are Docker scheduling
observations, not predictions of real device latency.
"""
    ),
    markdown(
        """
## 9. Isolation, aggregation, and failure evidence

The result below was produced by Docker inspection plus an independent NumPy
aggregation check.
"""
    ),
    code(
        """
isolation = benchmark["verification"]["isolation"]
isolation_rows = []
for service, values in isolation["services"].items():
    isolation_rows.append({
        "service": service,
        "data mounts": ", ".join(values["data_mounts"]),
        "read-only root": values["read_only_root"],
        "CPU limit": values["cpus"],
        "memory MiB": values["memory_bytes"] / 1024**2,
        "exit code": values["exit_code"],
    })
display(pd.DataFrame(isolation_rows))

checks = pd.DataFrame([
    ("all clients/round", benchmark["verification"]["aggregation"]["clients_per_round"] == 5),
    ("rows/round", benchmark["verification"]["aggregation"]["total_samples_per_round"] == 6148),
    ("independent aggregate", benchmark["verification"]["aggregation"]["maximum_aggregation_absolute_difference"] == 0.0),
    ("server training CSV visibility", isolation["server_training_partitions_visible"] == 0),
    ("one training CSV/client", isolation["client_training_partitions_visible_each"] == 1),
    ("CSV files embedded in image", isolation["training_csv_files_embedded_in_image"] == 0),
    ("frozen Phase 3 hashes", benchmark["verification"]["phase3_artifacts_unchanged"]),
    ("incompatible update rejected", benchmark["failure_exercise"]["incompatible_update_rejections"] == 1),
], columns=["check", "passed"])
assert checks["passed"].all()
display(checks)
"""
    ),
    markdown(
        """
The live incompatible update returned HTTP 400 and emitted `update_rejected`.
The automated timeout test separately confirms that an unavailable client is
named and the round fails instead of silently aggregating four updates.
"""
    ),
    markdown(
        """
## 10. GUI-facing API

- `GET /health`
- `GET /api/v1/status`
- `GET /api/v1/events?after_seq=N`
- `POST /api/v1/register`
- `GET /api/v1/model?client_id=...`
- `POST /api/v1/updates`
- `POST /api/v1/events`

Each event carries schema version, sequence, UTC time, run ID, source, type,
severity, round, client ID, and a typed payload. The full contract is in
`phase4/EVENT_CONTRACT.md`.
"""
    ),
    markdown(
        """
## 11. Build, run, observe, verify, and stop

These commands are executed from `experiments/uavids_federated`. The notebook
shows them rather than starting background containers automatically.
"""
    ),
    code(
        """
commands = pd.DataFrame([
    ("clean build and run", r"powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\\phase4\\scripts\\run_demo.ps1"),
    ("reuse image", r"powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\\phase4\\scripts\\run_demo.ps1 -SkipBuild"),
    ("observe", r"powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\\phase4\\scripts\\observe_demo.ps1"),
    ("verify aggregation", r"python .\\phase4\\verify_live_demo.py"),
    ("verify isolation", r"python .\\phase4\\verify_container_isolation.py"),
    ("stop", r"powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\\phase4\\scripts\\stop_demo.ps1"),
], columns=["action", "command"])
display(commands)
"""
    ),
    markdown(
        """
## Phase 4 takeaway and limits

The demo performs real local PyTorch training and real networked aggregation,
but it is not hardware emulation. It uses ordinary unauthenticated HTTP, logical
source clients, centrally fitted preprocessing, one x86 host, and a short
three-round presentation schedule. Its validation telemetry does not replace
the Phase 3 research results.

The Python services remain essential: notebooks cannot replace container entry
points, long-running coordination, protocol checks, automated verification, or
repeatable command-line operation. The next security step should protect this
working protocol with authenticated transport and replay-resistant client/update
identity; no cryptography is implemented in this guide.
"""
    ),
]


NOTEBOOKS.mkdir(parents=True, exist_ok=True)
outputs = {
    NOTEBOOKS / "05_phase3_binary_modeling_guide.ipynb": notebook(phase3_cells),
    NOTEBOOKS / "06_phase4_docker_federated_demo_guide.ipynb": notebook(phase4_cells),
}
for path, value in outputs.items():
    nbf.write(value, path)
    print(f"Wrote {path}")
