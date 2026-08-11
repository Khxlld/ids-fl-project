# RASID — dataset-backed presentation dashboard

This GUI demonstrates the frozen binary UAV intrusion-detection model, its
five-client federated training context, and protected model communication.
All dashboard traffic now comes from a traceable subset of the final UAVIDS-2025
test split. The browser never fabricates model features or computes a verdict.

## Start it on Windows

Open two PowerShell windows.

Terminal 1 — real adapter and frozen model, from `experiments\uavids_federated`:

```powershell
& .\.venv-gui\Scripts\python.exe -m gui_integration.backend `
  --host 127.0.0.1 `
  --port 8090 `
  --allowed-origins http://localhost:3000,http://127.0.0.1:3000
```

Wait for `"event": "gui_backend_ready"` and model ID
`federated_fedavg:5652da686897`.

Terminal 2 — static GUI, from the same directory:

```powershell
& .\.venv-gui\Scripts\python.exe .\gui_integration\Ammar_Attempt\serve.py --no-browser
```

Open **http://localhost:3000**. Stop both processes with `Ctrl+C`.

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8090/api/gui/v1/health
```

No Docker stack is required for genuine frozen-model inference. Without a live
Phase 4/5 upstream, federation and security telemetry remains clearly labelled
recorded evidence.

## Recommended presentation sequence

1. Open Dashboard and press **Restart Test Stream** for a clean session.
2. Press **Start Traffic**. Ground-truth Normal test flows appear automatically.
3. Point out the five-minute probability timeline and frozen 0.42 threshold.
4. Press **Attacker View** in the sidebar. Allow the local popup if prompted.
5. Select any logical client and inspect its complete configured profile.
6. Choose Blackhole, Flooding, Sybil, or Wormhole and press
   **Send Next Held-out Attack Flow**.
7. Compare dataset ground truth with the model's binary Normal/Attack result.
   The main timeline receives a marked controlled-attack event automatically.
8. Return to Dashboard and inspect the Attack Alert Log. Export it if useful.
9. Use **AI Data** for the concise locked-test comparison and verified security evidence. Open **Technical details** only if the audience asks about model inputs, preprocessing, or exact client scores.

**Start Traffic** toggles between running and paused. Replay pace is adjustable
from 0.5 to 3 seconds. The 250 Normal records never wrap silently; restart the
session to replay them again.

## What the data means

The committed demonstration pool contains 450 genuine held-out records:

- 250 Normal Traffic
- 50 Blackhole Attack
- 50 Flooding Attack
- 50 Sybil Attack
- 50 Wormhole Attack

Rows are the first entries for each ground-truth label in Phase 2 test-partition
order. Selection never consults the model output. At startup the adapter checks
the pool's test-split provenance and feature contract; runtime predictions are
also checked against saved locked-test evidence.

Only the approved 15 numeric features enter preprocessing and the model. The
recorded addresses, ports, protocol, MeanPacketSize, FlowID, and dataset label
are display/export metadata only.

The five clients are federated-training participants on device-inspired Docker
profiles. Selecting a client in Attacker View assigns a controlled presentation
target; it does not deliver traffic to that container and does not perform
per-client inference. The held-out record always keeps its real recorded source.

The model is binary. The attack family shown is dataset ground truth, never an
attack type predicted by the model. This is controlled dataset replay—not live
packet capture, a real cyberattack, or a new accuracy evaluation.

## AI Data evidence

AI Data deliberately does not show live accuracy or the three-round Docker demo
as research performance. Its primary values come from the final Phase 3 locked
test of 53,949 flows: FedAvg macro-F1 95.02%, attack precision 97.80%, attack
recall 97.67%, and false-positive rate 7.49%. The comparison uses the same locked
test for the local-only mean, FedAvg, and centralized models.

The security panel uses the controlled Phase 5 benchmark: five authenticated
clients, 12 safely rejected malicious messages, and plain/secure aggregation
agreement within the recorded tolerance. If either authoritative evidence file
is missing or inconsistent, the page shows **Evidence unavailable** instead of
substituting validation or live-stream values.

## Detection timeline and alert export

The timeline retains up to 240 predictions from the rolling last five minutes.
Controlled attacks have a vertical marker and tooltip containing target, ground
truth, model verdict, and Detected/Missed outcome. A missed attack remains on the
timeline even though its model verdict is Normal.

The Attack Alert Log contains only model-generated Attack alerts. It shows the
latest 10 by default and can expand to the full current demo session. A Normal
test row predicted as Attack is shown honestly as a false alarm. A missed attack
is not an alert.

**Export CSV** downloads the entire original 23-column UAVIDS row for every
detected alert, plus partition index, controlled target, timestamp, model ID,
probability, threshold, predicted label, and outcome. Complete rows are never
included in normal JSON snapshots, events, or browser logs.

## Troubleshooting

- **Adapter unreachable:** confirm Terminal 1 is still running and port 8090 is
  free: `Test-NetConnection 127.0.0.1 -Port 8090`.
- **CORS/origin error:** use `http://localhost:3000` or include the exact URL in
  `--allowed-origins`, then restart the adapter.
- **Attacker View did not open:** allow popups for `localhost:3000` and press the
  sidebar button again.
- **Torch or joblib import error:** use `.venv-gui\Scripts\python.exe`, not the
  system Python. The verified environment uses the pinned Torch 2.5.1 CPU build.
- **Pool exhausted:** press **Restart Test Stream**. This resets only controlled
  demo cursors, counts, and alert history; it does not modify any model artifact.
- **Federated telemetry says recorded evidence:** expected unless a Phase 4/5
  server URL was supplied to the adapter with `--federated-backend`.

To point the GUI at another compatible adapter, open:

```text
http://localhost:3000/?api=http://127.0.0.1:9000/api/gui/v1
```

The Attacker View inherits that adapter URL automatically.

## Verification command

From `experiments\uavids_federated`:

```powershell
& .\.venv-gui\Scripts\python.exe .\gui_integration\verify_integration.py
& .\.venv-gui\Scripts\python.exe .\verify_phase3_artifacts.py
```

The integration verifier covers existing endpoints, CORS, replay fallback,
dataset provenance, runtime locked-test agreement, concurrent attack requests,
a genuine missed attack, detected-only alert behavior, complete CSV export, and
frozen-artifact immutability.
