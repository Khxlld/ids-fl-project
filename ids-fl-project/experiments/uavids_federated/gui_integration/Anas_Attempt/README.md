# UAVIDS Federated Defense Console

Presentation-ready browser GUI for the verified binary UAV intrusion-detection, five-client federated-learning, and post-quantum communication demonstrations. It was implemented independently inside `Anas_Attempt` and uses only the documented GUI adapter and saved evidence contracts.

## What is included

- **IDS Monitor:** real or recorded `Normal`/`Attack` verdicts, confidence, model identity, counters, probability history, and recent Attack alerts.
- **Federation:** five logical clients, local sample counts, client state, round/update progress, secure exchange, FedAvg state, validation telemetry, and event trail.
- **Security:** ML-KEM-768, HKDF-SHA-256, ML-DSA-65 and AES-256-GCM roles; authentication and rejection status; rejection categories; honest security boundaries.
- **Evidence:** locked-test centralized/local/federated comparison, federated binary metrics, secure/plain aggregation equivalence, measured overhead, and limitations.
- **Recorded Demo:** start, pause, restart, change speed, and move through the verified presentation sequence without Docker.

No model checkpoint, preprocessing object, dataset, private partition, model tensor, key, signature, or ciphertext is loaded by this GUI.

## Prerequisites

- Windows PowerShell.
- Python 3.10 or newer. No package installation is needed.
- A current browser such as Microsoft Edge or Chrome.
- Optional for Live Adapter mode: the existing GUI backend running on port `8090`.

## Start the GUI

From this folder:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\start.ps1
```

The browser opens at:

```text
http://127.0.0.1:3000
```

Press `Ctrl+C` in the PowerShell window to stop the GUI.

To start without opening a browser:

```powershell
.\start.ps1 -NoBrowser
```

To use another port or adapter URL:

```powershell
.\start.ps1 -Port 3001 -ApiBase http://127.0.0.1:8090/api/gui/v1
```

## Recorded Demo mode

Recorded Demo is selected automatically when the documented adapter cannot be reached. It does not need Docker, model files, or network access.

1. Click **Recorded demo** in the header.
2. Use **Play**, **Pause**, **Previous**, **Next**, **Restart**, or the speed selector.
3. Use **Analyze next verified record** to step through six saved real-model responses: three Normal and three Attack.

The interface always labels this mode `Recorded Demo`. Its client sequence, metrics, predictions and rejection counts come from committed, presentation-safe project evidence. Recorded prediction results are not recomputed in the browser.

## Live Adapter mode

Live Adapter means the browser is connected to the existing `uavids-gui-api-v1` adapter. It polls `/snapshot` and keeps independent sequence cursors for inference and federated events. The **Analyze next verified record** button calls the adapter's `/replay/next` endpoint, so the live frozen model evaluates the approved replay input at request time.

One-time backend environment, from `experiments\uavids_federated`:

```powershell
py -3.11 -m venv .venv-gui
.\.venv-gui\Scripts\python.exe -m pip install -r .\gui_integration\requirements.txt
```

Start the adapter in a separate PowerShell window:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\gui_integration\scripts\start_backend.ps1 `
  -Python .\.venv-gui\Scripts\python.exe `
  -FrontendOrigin http://127.0.0.1:3000
```

Then start this GUI and click **Live adapter**. If a Phase 4 or Phase 5 upstream is configured for the adapter, federated telemetry is live. Otherwise the adapter truthfully reports recorded federated telemetry while its binary inference remains the live frozen model.

For live secure federation, launch the existing Phase 5 demonstration separately, then point the adapter at its HTTP service:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\gui_integration\scripts\start_backend.ps1 `
  -Python .\.venv-gui\Scripts\python.exe `
  -FrontendOrigin http://127.0.0.1:3000 `
  -FederatedBackend http://127.0.0.1:8080
```

This GUI never connects directly to Docker services.

## Configuration

Defaults are shown in `.env.example`. Set values in PowerShell before starting if needed:

```powershell
$env:GUI_PORT = "3000"
$env:GUI_API_BASE = "http://127.0.0.1:8090/api/gui/v1"
$env:GUI_POLL_MS = "1200"
$env:GUI_REQUEST_TIMEOUT_MS = "1800"
.\start.ps1
```

These values contain no credentials or secrets. `.env.example` is a template; no dependency-heavy `.env` loader is required.

## Verification

Run the offline structural and HTTP checks:

```powershell
py -3 .\verify.py
```

The verifier checks required files, safe replay contents, the five locked clients and sample counts, the six real-model results, locked metrics, security algorithms, rejection totals, prohibited payload types, and local HTTP serving.

## Recommended presentation flow

1. **IDS Monitor:** analyze a Normal record, then an Attack record; point out the frozen threshold, confidence, counters and alert.
2. **Federation:** play the recorded sequence or show the live round; explain that five clients keep their rows local and send model updates.
3. **Security:** advance to the rejection stage; explain authentication, key establishment, protected payloads, replay checks, and why rejected updates never enter FedAvg.
4. **Evidence:** compare 89.69% local-only mean, 95.02% FedAvg and 97.75% centralized locked-test Macro-F1; show the exact secure/plain aggregation match.

Use browser full-screen mode (`F11`) for the presentation. At low projector resolutions, browser zoom around 80-90% gives more vertical room.

## Common problems

- **Live adapter unavailable:** Recorded Demo remains functional. Check that the backend is running on port `8090`, then click **Live adapter** again.
- **CORS rejection:** restart the existing adapter with `-FrontendOrigin http://127.0.0.1:3000`. `localhost` and `127.0.0.1` are different origins.
- **Port 3000 is busy:** run `.\start.ps1 -Port 3001` and use the same exact origin when launching the adapter.
- **Model unavailable:** use Recorded Demo for the presentation, or verify the existing deployment checkpoint/preprocessor with the project integration verifier. This GUI does not load them directly.
- **Page appears stale:** use `Ctrl+F5`; the local server sends no-store headers for presentation assets.

## Genuine limitations

- The model is binary and does not identify an attack family.
- Replay is recorded evidence and is never presented as live.
- Saved validation or locked-test metrics are context, not current-stream accuracy.
- The five clients are source-based logical partitions, not verified physical UAV identities.
- The shared preprocessor was fitted centrally on pooled training-client features.
- Communication security does not detect model poisoning by an authenticated client, hide HTTP metadata, provide secure aggregation, or make the prototype production-ready.
