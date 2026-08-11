# UAVIDS Federated Defense Console

Presentation dashboard for the frozen binary UAV intrusion-detection model, five-client federated training, and post-quantum-secured model communication. The four sections are **IDS Monitor**, **Federation**, **Security**, and **Evidence**.

The GUI uses only presentation-safe API fields and saved evidence. It does not load model checkpoints, preprocessing objects, training CSVs, client partitions, tensors, keys, signatures, or ciphertext.

## Morning-of-presentation: Recorded Demo

Recorded Demo needs only Python 3.10 or newer and a current browser. No installation, Docker, model loading, or network connection is required.

```powershell
cd C:\Projects\ids-fl-project\ids-fl-project\experiments\uavids_federated\gui_integration\Anas_Attempt
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\start.ps1
```

Open `http://127.0.0.1:3000`. Press `Ctrl+C` in the PowerShell window to stop it.

The dashboard intentionally starts on **IDS Monitor** in **Recorded Demo** mode. It does not attempt Live mode or display a connection failure unless the presenter clicks **Live adapter**.

### Recorded presentation controls

- **Start IDS Demo:** starts the automatic network-flow inference stream in either Recorded or Live mode.
- **Pause / Resume:** stops and continues without losing counters or history.
- **Restart:** resets the Recorded Demo stream, alerts, Federation walkthrough, and presentation section. Restart `start_live.ps1` for a fresh Live session.
- **Speed:** selects 0.5x, 1x, or 2x pacing.
- **Analyze next verified flow:** manually advances one flow.
- **Next section:** deliberately moves from IDS Monitor to Federation, Security, and Evidence. The GUI never changes tabs automatically.
- **Advance FL stage:** steps through client readiness, local training, updates, FedAvg, and completion within Federation.

The recorded stream contains **24 distinct frozen-model inference responses**: 16 Normal and 8 Attack. They were generated without retraining from distinct finite-feature rows in the locked Phase 2 validation partition using the pinned Phase 3 model and threshold. No feature values or labels are stored in the GUI. The stream is curated for presentation and is not an attack-prevalence estimate.

## Recommended sponsor sequence

1. Stay on **IDS Monitor** and press **Start IDS Demo**. Point out the operational status, binary verdict, fixed 0.42 threshold, probability history, counters, chronological flow activity, and alerts.
2. Pause after enough activity is visible, or let all 24 flows finish. Press **Next section: Federation**.
3. Use **Advance FL stage** to show five clients training locally, sending model updates, and contributing to FedAvg. Raw training rows are not uploaded to the coordinator.
4. Press **Next section: Security**. Explain ML-DSA-65 authentication, ML-KEM-768 key establishment, HKDF-SHA-256 derivation, AES-256-GCM protection, and the controlled rejection evidence.
5. Press **Next section: Evidence**. Compare the 89.69% local-only mean, 95.02% federated, and 97.75% centralized locked-test Macro-F1, then show the exact plain/secure aggregation match.

Use `F11` for full screen. Around 80-90% browser zoom provides more vertical room on a low-resolution projector.

## Live frozen-model inference

Live mode requires the existing `uavids-gui-api-v1` adapter. Its first startup imports PyTorch, verifies artifact hashes, and loads the frozen model, so readiness can take materially longer than a normal API poll.

### One-time setup

Run from `experiments\uavids_federated`:

```powershell
cd C:\Projects\ids-fl-project\ids-fl-project\experiments\uavids_federated
py -3.12 -m venv .venv-gui
.\.venv-gui\Scripts\python.exe -m pip install -r .\gui_integration\requirements.txt
```

Use Python 3.12 (or the reference Python 3.11), not Python 3.13. The repository pins `torch==2.5.1`, for which the required Windows Python 3.13 wheel is unavailable. The environment belongs directly under `experiments\uavids_federated`; a `.venv-gui` accidentally created inside `gui_integration` is not used by `start_live.ps1`.

### Simplest Live command

From `Anas_Attempt`, the wrapper starts the existing adapter through the presentation helper, waits up to 60 seconds for the frozen model, then starts the dashboard. The helper prepares 24 distinct approved validation inputs in memory and exposes only their prediction responses; it does not write, log, or return feature values. Each input is evaluated by the real frozen model when the GUI requests it. `Ctrl+C` stops the dashboard and its adapter process.

```powershell
.\start_live.cmd
```

The `.cmd` launcher applies PowerShell's execution-policy bypass only to this one dashboard process. It does not change the system or user execution policy.

Open `http://127.0.0.1:3000`, then click **Live adapter**. The visible connection sequence is: Checking adapter, Adapter reached, Checking model, Model ready, Live mode connected. Press **Start IDS Demo** to run all 24 genuine frozen-model requests automatically; Pause, Resume, Speed, and manual next-flow analysis work in Live mode too. The sequence uses a readable Normal–Normal–Attack rhythm (16 Normal and 8 Attack), not the original six-input alternating loop.

To reset Live-mode counters and return to the first approved input, press `Ctrl+C` and run `start_live.ps1` again. The in-page **Restart** button remains a Recorded Demo control because the verified backend intentionally exposes no state-reset endpoint.

To attach live Phase 4/5 telemetry as well:

```powershell
.\start_live.ps1 -FederatedBackend http://127.0.0.1:8080
```

The Phase 4 or Phase 5 service must already be running at that URL. Without it, frozen-model inference is still live while Federation is clearly labelled as recorded telemetry.

### Two-window Live alternative

Window 1, from `experiments\uavids_federated`:

```powershell
.\.venv-gui\Scripts\python.exe .\gui_integration\Anas_Attempt\live_adapter.py `
  --host 127.0.0.1 `
  --port 8090 `
  --allowed-origins http://127.0.0.1:3000
```

Window 2, from `Anas_Attempt`:

```powershell
.\start.ps1
```

Confirm adapter readiness:

```powershell
Invoke-RestMethod http://127.0.0.1:8090/api/gui/v1/health
```

Expected fields include `ok: true`, `api_version: uavids-gui-api-v1`, and `model_available: true`. Stop both visible windows with `Ctrl+C`.

## Live connection behavior

The dashboard keeps Recorded Demo unchanged while checking Live mode. It uses a 20-second configurable readiness window, a 4.5-second ordinary request timeout, and readiness retries. It switches modes only after both health and snapshot contracts confirm the frozen model is available.

If connection fails, the sponsor-facing banner remains short. Expand **Connection details and setup help** to distinguish an absent adapter, wrong API path, startup timeout, model unavailability, likely CORS mismatch, or incompatible response. Recorded Demo continues from its previous state.

## Configuration

Defaults are in `.env.example`:

```powershell
$env:GUI_PORT = "3000"
$env:GUI_API_BASE = "http://127.0.0.1:8090/api/gui/v1"
$env:GUI_POLL_MS = "1200"
$env:GUI_REQUEST_TIMEOUT_MS = "4500"
$env:GUI_CONNECT_TIMEOUT_MS = "20000"
$env:GUI_LIVE_RETRY_MS = "750"
.\start.ps1
```

Use the exact dashboard origin when starting the adapter. `http://localhost:3000` and `http://127.0.0.1:3000` are different CORS origins.

## Common fixes

- **Adapter not running:** start it with `start_live.ps1` or the documented backend script, then confirm `/health`.
- **`No matching distribution found for torch==2.5.1`:** the environment was created with Python 3.13. Return to `experiments\uavids_federated` and recreate the project-level `.venv-gui` with `py -3.12 -m venv .venv-gui`.
- **Wrong API URL:** the base must end in `/api/gui/v1`; set `GUI_API_BASE` or use `start.ps1 -ApiBase ...`.
- **Startup timeout:** watch the adapter terminal and allow frozen-model loading to finish. Increase `GUI_CONNECT_TIMEOUT_MS` if the machine is unusually slow.
- **CORS mismatch:** restart the adapter with `-FrontendOrigin` exactly matching the browser address and port.
- **Port 3000 busy:** run `.\start.ps1 -Port 3001`; use `http://127.0.0.1:3001` as the adapter origin.
- **Model unavailable:** run the existing GUI integration verifier and confirm the frozen artifact hashes. Recorded Demo remains available.
- **Stale page:** press `Ctrl+F5`; the local server sends no-store headers.

## Verification

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
py -3.12 .\verify.py
```

The verifier checks the 24 distinct safe responses, class counts, ordering, model identity, threshold, absence of features and secrets, five-client evidence, locked metrics, security evidence, presentation controls, diagnostic route, and local HTTP serving.

`generate_recorded_stream.py` documents how the saved stream was produced. It is an offline evidence helper, not a runtime requirement, and requires the repository-pinned Phase 3 environment.

## Genuine limitations

- The model is binary and does not identify an attack family or severity.
- The saved stream is curated recorded evidence, not live traffic and not a prevalence estimate.
- Validation and locked-test metrics are context, not current-stream accuracy.
- The saved API evidence supplies `recorded-validation-sample` as the source; it does not provide a defensible physical UAV identity for each flow.
- The five clients are source-based logical partitions, not verified physical UAV identities.
- The shared preprocessor was fitted centrally on pooled training-client features.
- Communication security does not detect poisoning by an authenticated client, hide HTTP metadata, provide secure aggregation, or make the prototype production-ready.
