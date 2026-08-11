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
| `GET /demo/catalog` | Locked-test demonstration provenance, pool counts, and session state |
| `POST /demo/traffic/next` | Score the next unused ground-truth Normal test flow |
| `POST /demo/attacks` | Score the next unused test flow for a selected attack family and controlled target |
| `POST /demo/reset` | Reset only the controlled demonstration session |
| `GET /demo/alerts.csv` | Export detected demo alerts with complete original rows |

`after_seq` starts at zero. Keep separate inference and federated cursors. Unknown fields must be ignored.

## Locked-test demonstration

The dataset-demo endpoints use a checksum-traceable 450-row subset of the final
Phase 2 test partition: the first 250 Normal rows and first 50 rows for each of
Blackhole, Flooding, Sybil, and Wormhole in test-partition order. Selection does
not inspect saved or runtime model outcomes. Every request is scored at runtime
by the same frozen model used by `/predictions`.

`POST /demo/traffic/next` accepts an empty object. `POST /demo/attacks` accepts
exactly:

```json
{"client_id":"uav-client-3","attack_type":"Wormhole Attack"}
```

The selected client is controlled presentation context, not the recorded source
and not a per-client inference endpoint. Dataset-backed predictions add limited
`dataset` and `demo` objects. `dataset.ground_truth_label` is source ground truth;
the model remains binary and returns only `Normal` or `Attack`. The demo outcome
is one of `correct_normal`, `false_alarm`, `detected`, or `missed`.

The normal JSON API never returns a complete dataset row. The explicit CSV
export contains the original 23 UAVIDS-2025 columns followed by partition,
target, and prediction metadata for model-generated Attack alerts in the current
demo session. Identifiers, ports, protocol, `MeanPacketSize`, and the dataset
label are export/display metadata only and never enter preprocessing or inference.
Pool exhaustion returns HTTP 409 and rows never wrap silently.

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
- Dataset-demo responses expose only a small presentation projection. Complete
  original rows are available only through the explicit alert CSV export.

## Snapshot semantics

- `presentation_mode` describes federated telemetry: `live` when Phase 4/5 is reachable, otherwise `replay`.
- `backend.inference_mode` is `live_model` when this adapter has loaded the verified checkpoint.
- `evidence` is a read-only public projection of the authoritative Phase 3 locked-test evaluation and Phase 5 controlled security benchmark. It is independent of live inference and demonstration-round telemetry.
- `evidence.available` is `false` when those saved artifacts are missing, malformed, or inconsistent; consumers must show the evidence as unavailable and must not substitute other metrics.
- `inference` contains process-lifetime counters, latest result, and up to 20 recent Attack alerts.
- `federated` contains five client states, round/update progress, local-data statement, validation telemetry, and security summary.
- `federated.global_model_metrics` is demonstration validation telemetry. Its `source` and `note` must remain visible wherever it is shown, and it must not be presented as locked-test performance or live accuracy.

## Errors and CORS

Errors use `uavids-gui-error-v1` with a stable code and safe message. Invalid features and JSON return HTTP 400; unapproved browser origins return 403; unknown endpoints return 404. Internal exceptions return a generic HTTP 500 without implementation details.

Configure an exact frontend origin with `-FrontendOrigin` or `GUI_ALLOWED_ORIGINS`. Multiple origins may be comma-separated when launching Python directly. Wildcard origin is supported only for a credential-free local demonstration and is not recommended. Restart the adapter after changing CORS configuration.
