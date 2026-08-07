# UAVIDS presentation dashboard

React + Vite operations console for the live demo. It talks only to the GUI
adapter described in [`../../API_CONTRACT.md`](../../API_CONTRACT.md) — it never
loads checkpoints, preprocessing objects, datasets, or cryptographic material.

## Prerequisite: the adapter must be running

The dashboard is a pure client. Start the backend first, from
`experiments/uavids_federated`:

```powershell
py -3.11 -m venv .venv-gui
.\.venv-gui\Scripts\python.exe -m pip install -r .\gui_integration\requirements.txt
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\gui_integration\scripts\start_backend.ps1 `
  -Python .\.venv-gui\Scripts\python.exe `
  -FrontendOrigin http://localhost:3000
```

> **Blocker on a fresh clone.** `backend.py` verifies and loads the frozen Phase 3
> checkpoint and preprocessor from `artifacts_phase3/`. Those are `*.pt` /
> `*.joblib` files excluded by the repository `.gitignore`, so a clean checkout
> fails at startup with `FileNotFoundError`. Obtain them from whoever produced the
> Phase 3 run, or regenerate them, before presenting. Without them there is no
> real inference.

Optional live federated telemetry (otherwise the adapter reports
`presentation_mode: "replay"` and serves recorded evidence):

```powershell
... start_backend.ps1 ... -FederatedBackend http://127.0.0.1:8080
```

### Working without the artifacts

Until the frozen artifacts are available, `../../dev_mock_backend.py` serves the
same API from the committed fixtures so the UI can be developed and inspected:

```powershell
python ..\..\dev_mock_backend.py     # from this directory
```

**It is not the model.** Its verdicts come from a two-feature stand-in scorer and
mean nothing — it exists so the console renders. It prints a loud MOCK banner on
startup; the real adapter prints a `gui_backend_ready` line with the true
`model_id`. That banner is the only reliable way to tell which one you are on,
because the mock's `/health` and `/snapshot` payloads are fixture-faithful by
design. Never demo on it.

## Run the dashboard

```powershell
npm install
npm run dev      # http://localhost:3000
```

Port 3000 is pinned in `vite.config.ts` because it is the adapter's default
allowed CORS origin. If you change it, pass the matching `-FrontendOrigin`.

Point at a non-default adapter with `VITE_API_BASE`:

```powershell
$env:VITE_API_BASE = "http://127.0.0.1:9000/api/gui/v1"; npm run dev
```

## Suggested demo sequence

1. Open with the console idle — command bar shows link, detection, FL telemetry,
   crypto and injector state at a glance.
2. **NOMINAL LINK** → start stream. Bars stay low and green, well under the
   threshold rule.
3. Switch to **TRANSITIONAL** mid-stream. Verdicts genuinely mix — this is the
   honest moment, showing the 0.42 boundary doing real work rather than a rigged
   binary.
4. Switch to **SUSTAINED FLOOD**. Verdict flips to ATTACK, bars spike above the
   threshold, alert log and counters move.
5. Drop to the federated and security panels for the FedAvg and post-quantum
   story, then the event logs.

Chrome zoom (`Ctrl -`) widens the CSS viewport if the projector is low
resolution; the layout re-flows from three columns to one on its own.

## What the traffic injector does and does not do

`src/sim/profiles.ts` generates **model input only**. Sampling envelopes come
from the two clusters in the committed fixture
`gui_integration/examples/replay_records.json` — rows 01/03/05 (scored Normal)
and 02/04/06 (scored Attack). Derived features are computed, not sampled, using
relationships that hold exactly in that fixture:

```
TxPacketRate = TxPackets / FlowDuration        Throughput/Kbps = RxBytes * 8 / 1000 / FlowDuration
TxByteRate   = TxBytes   / FlowDuration        PacketDropRate  = LostPackets / TxPackets
```

so an emitted vector is internally self-consistent rather than 15 unrelated
random numbers. Every verdict on screen is produced by the real frozen model via
`POST /predictions`. Nothing in the frontend fabricates a prediction, a
probability, or a metric.

A profile name describes the traffic shape **the operator is injecting**. The
model is binary and never identifies an attack family.

## Claims the UI must keep intact

These are enforced in the components and restated in the footer; keep them if you
edit the layout.

- Binary Normal/Attack only — no attack family, technique, or actor.
- `presentation_mode`, `inference_mode` and per-prediction `replayed` stay visible
  and distinct.
- Session counters (this tab) and backend counters (adapter process lifetime) are
  shown separately and never summed.
- Saved validation metrics are labelled as saved evidence, not live accuracy.
  There is no ground truth for injected or replayed input, so no accuracy figure
  is computed from this screen.
- Five logical clients are dataset partitions on device-inspired container
  profiles, not verified physical UAVs.
- Security covers federated model exchange in transit only — not poisoning by an
  authenticated client, not HTTP metadata, not production readiness.

## Layout

```
src/
  api/         typed client + response types mirroring api/openapi.json
  sim/         traffic profile envelopes and vector generation
  hooks/       snapshot polling, event feeds, session history, injector loop
  components/  console panels
```
