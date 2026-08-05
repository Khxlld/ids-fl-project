# GUI-facing status and event contract

The control center exposes ordinary HTTP/JSON on `http://localhost:8080`. Phase 4 intentionally adds no authentication, encryption, cryptography, or attack simulation.

## Endpoints

- `GET /health` - lightweight container health check.
- `GET /api/v1/status` - current run state, clients, pending updates, round summaries, and latest validation metrics.
- `GET /api/v1/events?after_seq=N` - ordered events after sequence `N`, capped at the latest 1,000 returned events.
- `POST /api/v1/register` - client compatibility and partition registration.
- `GET /api/v1/model?client_id=...` - current round model and training instructions.
- `POST /api/v1/updates` - validated model update and client telemetry.
- `POST /api/v1/events` - allow-listed client lifecycle event.

## Event envelope

```json
{
  "schema_version": "phase4-event-v1",
  "seq": 42,
  "timestamp_utc": "2026-08-04T16:06:20.000000Z",
  "elapsed_ms": 28000.0,
  "run_id": "uuid",
  "source": "uav-client-3",
  "event_type": "client_training_completed",
  "severity": "info",
  "round": 2,
  "client_id": "uav-client-3",
  "payload": {
    "training_ms": 27.3,
    "update_bytes": 47873
  }
}
```

`seq` is strictly increasing within one control-center run. Consumers should retain the last seen sequence and reconnect with `after_seq`. Unknown payload fields should be ignored so the envelope can evolve without breaking a GUI.

## Event types

- Startup: `server_started`, `client_registered`, `client_ready`, `all_clients_ready`.
- Distribution/training: `round_started`, `global_model_distributed`, `client_training_started`, `client_training_completed`.
- Upload/waiting: `client_update_received`, `client_update_acknowledged`, `server_waiting_for_clients`.
- Aggregation/evaluation: `aggregation_started`, `aggregation_completed`, `round_metrics`, `round_completed`.
- Terminal/error: `demo_completed`, `client_registration_rejected`, `update_rejected`, `client_protocol_error`, `server_protocol_error`, `demo_failed`.

The GUI should treat `demo_completed` and `demo_failed` as terminal for that `run_id`. A timeout failure names every missing client. Inference metrics use the locked federated threshold `0.42`; they are demo telemetry, not new official research results.

## Update safety checks

Registration and every update are bound to the run ID, protocol/config versions, canonical contract hash, expected client ID, exact partition hash, and exact sample count. Model archives must contain the expected parameter names in order, `float32` dtypes, exact tensor shapes, finite values, a matching payload hash, and remain below the configured size limit. Stale, future, duplicate, malformed, non-finite, or incompatible updates are rejected with HTTP 400 and an error event where applicable.
