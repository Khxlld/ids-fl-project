# UAVIDS GUI integration handoff

This folder is the stable boundary for the presentation GUI. The frontend does not load checkpoints, preprocessing objects, datasets, Docker files, or cryptographic material. It calls one small HTTP/JSON adapter on port `8090`.

The primary path performs genuine binary inference with the frozen Phase 3 FedAvg model and training-only preprocessor. The secondary path normalizes the existing Phase 4/5 status and event APIs. If Docker is unavailable, federated telemetry switches explicitly to verified recorded evidence while inference continues to use the real model.

## Quick start: dependable replay presentation

Run from `experiments/uavids_federated`:

```powershell
py -3.11 -m venv .venv-gui
.\.venv-gui\Scripts\python.exe -m pip install -r .\gui_integration\requirements.txt
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\gui_integration\scripts\start_backend.ps1 `
  -Python .\.venv-gui\Scripts\python.exe `
  -FrontendOrigin http://localhost:3000
```

The API is then available at `http://127.0.0.1:8090/api/gui/v1`. The default frontend origins are not guessed: pass the exact origin used by the GUI, including its port.

For the presentation, call `POST /replay/next` every one or two seconds and poll `GET /snapshot`. The six-record fixture alternates Normal and Attack predictions, wraps safely, and contains only the 15 approved model features from selected validation rows. It has no labels, identifiers from the source data, training rows, or test rows.

## Live federated telemetry

Start the GUI adapter with the optional upstream URL:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\gui_integration\scripts\start_backend.ps1 `
  -Python .\.venv-gui\Scripts\python.exe `
  -FrontendOrigin http://localhost:3000 `
  -FederatedBackend http://127.0.0.1:8080
```

Then run either backend in another terminal:

```powershell
# Secure Phase 5 demonstration
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\phase5\scripts\run_secure_demo.ps1

# Or the plain comparison baseline
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\phase4\scripts\run_demo.ps1
```

The adapter automatically reports `presentation_mode: "live"` while the upstream is reachable and `"replay"` otherwise. Do not hide that indicator. Record inference always remains `inference_mode: "live_model"`; the prediction's `replayed` field says whether its input came from the fallback fixture.

## Frontend flow

1. Poll `GET /health`; show the backend/model as unavailable if it fails.
2. Poll `GET /snapshot` about once per second. This one response supplies counters, latest prediction, alerts, model identity, five-client state, round/update progress, performance context, and security state.
3. For incoming records, send `POST /predictions` with the exact feature object.
4. For the dependable fallback, send `POST /replay/next` with `{}`.
5. Optionally poll `/events` and `/federated/events` using their independent `after_seq` cursors for activity timelines.
6. Ignore unknown response fields so compatible additions do not break the GUI.

Use [frontend/vanilla_integration.js](frontend/vanilla_integration.js) as a framework-neutral example. The minimum information to display is in [GUI_MINIMUM_REQUIREMENTS.md](GUI_MINIMUM_REQUIREMENTS.md), and precise semantics are in [API_CONTRACT.md](API_CONTRACT.md). [api/openapi.json](api/openapi.json) is the machine-readable contract.

## Verify

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\gui_integration\scripts\verify_integration.ps1 `
  -Python .\.venv-gui\Scripts\python.exe
```

This launches the adapter on an ephemeral local port, exercises health, snapshot, six real-model replay predictions, counters, alerts, both event feeds, invalid input, and CORS rejection, then confirms frozen hashes and scans the handoff for prohibited payload files.

## Important boundaries

- The model is binary. Display only `Normal` or `Attack`; no attack family is inferred.
- `confidence` is the probability assigned to the displayed class, not a guarantee of correctness.
- Saved validation metrics are context, not live accuracy and not a production guarantee.
- Logical clients are source partitions/device profiles, not verified physical UAV identities.
- The shared preprocessor was fitted centrally on pooled training-client features; it is not privacy-preserving federated preprocessing.
- Security protects federated model exchange. It does not detect poisoning by an authenticated client, hide HTTP metadata, or make this production infrastructure.
