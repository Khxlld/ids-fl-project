# GUI API contract

Base URL: `http://127.0.0.1:8090/api/gui/v1`

Contract version: `uavids-gui-api-v1`

Transport: JSON over HTTP; no cookies or frontend credentials.

## Endpoints

| Method and path | Purpose |
|---|---|
| `GET /health` | Backend and frozen-model availability |
| `GET /snapshot` | Complete presentation state for simple polling |
| `POST /predictions` | Classify one supplied 15-feature record |
| `POST /replay/next` | Classify the next recorded fallback input |
| `GET /events?after_seq=N` | New inference activity events |
| `GET /federated/events?after_seq=N` | New live or recorded FL/security events |

`after_seq` starts at zero. Keep separate inference and federated cursors. Unknown fields must be ignored.

## Prediction request

```json
{
  "record_id": "capture-row-104",
  "source": "uav-client-2",
  "features": {
    "FlowDuration/s": 1.2,
    "TxPackets": 5,
    "RxPackets": 4,
    "LostPackets": 1,
    "TxBytes": 200,
    "RxBytes": 160,
    "TxPacketRate/s": 4.17,
    "RxPacketRate/s": 3.33,
    "TxByteRate/s": 166.67,
    "RxByteRate/s": 133.33,
    "MeanDelay/s": 0.02,
    "MeanJitter/s": 0.003,
    "Throughput/Kbps": 2.4,
    "PacketDropRate": 0.2,
    "AverageHopCount": 1
  }
}
```

All 15 feature names are required exactly; extra names are rejected. Values must be finite numbers or `null`. A null value is handled by the frozen training-only imputer and counted in `missing_features_imputed`. `record_id` may be omitted and generated. `source` is optional and is returned unchanged; do not infer it when absent.

## Prediction semantics

- `attack_probability` is the sigmoid probability for the positive `Attack` class.
- The frozen decision threshold is `0.42`.
- `label` is `Attack` at or above the threshold and `Normal` below it.
- `confidence` is `attack_probability` for Attack and `1 - attack_probability` for Normal.
- `replayed` indicates the input source. Even replayed inputs are evaluated at request time by the real frozen model.
- No feature values or raw tensors are returned in prediction responses or alerts.

## Snapshot semantics

- `presentation_mode` describes federated telemetry: `live` when Phase 4/5 is reachable, otherwise `replay`.
- `backend.inference_mode` is `live_model` when this adapter has loaded the verified checkpoint.
- `inference` contains process-lifetime counters, latest result, and up to 20 recent Attack alerts.
- `federated` contains five client states, round/update progress, local-data statement, validation telemetry, and security summary.
- `global_model_metrics.source` and `note` must remain visible wherever metrics are shown.

## Errors and CORS

Errors use `uavids-gui-error-v1` with a stable code and safe message. Invalid features and JSON return HTTP 400; unapproved browser origins return 403; unknown endpoints return 404. Internal exceptions return a generic HTTP 500 without implementation details.

Configure an exact frontend origin with `-FrontendOrigin` or `GUI_ALLOWED_ORIGINS`. Multiple origins may be comma-separated when launching Python directly. Wildcard origin is supported only for a credential-free local demonstration and is not recommended. Restart the adapter after changing CORS configuration.
