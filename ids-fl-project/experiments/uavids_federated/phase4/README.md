# Phase 4 containerized federated-learning demo

This is a short CPU-only presentation demo. It initializes from the immutable Phase 3 FedAvg checkpoint, distributes the exact MLP over ordinary HTTP, performs three rounds of two local epochs on five isolated clients, aggregates by the locked sample-count rule, and evaluates on validation at threshold `0.42`. Runtime output is written only under `phase4/runtime/`; Phase 3 artifacts are read-only.

```text
five client containers                 control-center container
one read-only train.csv each           validation.csv only
        |                                   |
        +-- register/verify contract ------>|
        |<--------- global NPZ model -------+
        +-- local CPU training              |
        +-- validated NPZ update ---------->|
                                            +-- sample-weighted FedAvg
                                            +-- validation metrics/events
```

## Run

From `experiments/uavids_federated` in PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\phase4\scripts\run_demo.ps1
```

The command performs a clean Compose build, starts all six services, exercises one incompatible-update rejection, prints lifecycle events, samples `docker stats`, waits for completion, and independently verifies every aggregate. To reuse the existing local image:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\phase4\scripts\run_demo.ps1 -SkipBuild
```

Observe status/events in another terminal:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\phase4\scripts\observe_demo.ps1
docker compose -f .\phase4\docker-compose.yml logs -f --no-color
```

Direct GUI/backend endpoints are `http://localhost:8080/api/v1/status` and `http://localhost:8080/api/v1/events?after_seq=0`.

Stop containers and the demo network while preserving evidence:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\phase4\scripts\stop_demo.ps1
```

Add `-RemoveRuntime` only when the generated run evidence should also be deleted.

## What is reused from Phase 3

- Frozen 15-feature order and binary label mapping.
- Read-only fitted imputer/scaler; it is never refitted.
- `15 -> 128 -> 64 -> 32 -> 1` MLP and `uavids_fl` training/metric code.
- Unweighted loss, AdamW settings, batch size, seed policy, and two local epochs.
- All-five-client, sample-count-weighted FedAvg.
- Frozen federated checkpoint as demo initialization and threshold `0.42` for telemetry.

Demo mode uses three rounds instead of the 30-round Phase 3 research schedule. Its results must never overwrite or be reported as Phase 3 results.

## Reliability and isolation

Compose mounts one exact training file into each client and no training file into the server. All data/reference mounts and container roots are read-only; containers run unprivileged with all Linux capabilities dropped and a small temporary filesystem. Updates use safe `allow_pickle=False` NumPy archives and strict version, shape, dtype, finiteness, hash, sample-count, round, and size checks. A 120-second startup/round timeout reports missing clients explicitly.

Run the host checks independently while the completed server remains up:

```powershell
python .\phase4\verify_live_demo.py
python .\phase4\verify_container_isolation.py
python -m pytest -q tests phase4\tests
```

See [DEVICE_PROFILES.md](DEVICE_PROFILES.md), [EVENT_CONTRACT.md](EVENT_CONTRACT.md), and [results/PHASE4_RESULTS_SUMMARY.md](results/PHASE4_RESULTS_SUMMARY.md).

## Troubleshooting

- Start Docker Desktop and wait for the `desktop-linux` engine before running the script.
- If `docker` is not on `PATH`, the scripts use this installation's per-user Docker Desktop CLI automatically and add its credential-helper directory for builds.
- If port 8080 is occupied, stop that process or change only the published host port; keep the container/server port at 8080.
- Inspect `docker compose -f .\phase4\docker-compose.yml ps -a` and `logs --no-color` after a failure.
- A client exit code other than zero is a real demo failure. Do not reduce `min` participation or aggregate a partial round to hide it.
- Memory/OOM failures appear in `docker inspect`; retain the runtime directory and logs before changing a profile.

## Scope and security handoff

The bridge network is ordinary unauthenticated HTTP. The next security phase should first add mutually authenticated transport and update-message identity/replay protection around this frozen protocol contract. Post-quantum cryptography, AES-GCM, packet capture, attack simulation, and GUI implementation are deliberately absent here.
