# Minimum presentation-ready GUI requirements

The GUI team controls layout and visual design. A presentation-ready implementation must communicate the following without inventing values.

## Deployed IDS — primary

- Backend/model available or disconnected.
- Presentation mode: live or replayed.
- Latest binary result: `Normal` or `Attack`.
- Confidence, record source when supplied, and model ID/version.
- Recent Attack alerts.
- Records processed plus Normal and Attack counters.
- Loading, empty, invalid-data, and backend-error states that do not crash the UI.

## Federated learning — secondary

- Exactly five logical clients and each client's reported state.
- Current round and total rounds.
- Updates received versus expected and aggregation activity when events are used.
- A clear statement that each client reads its own local partition; raw training rows are not uploaded.
- Available global validation telemetry with a visible note that it is saved/live validation evidence, not live prediction accuracy.

## Communication security

- Plain or secure communication mode.
- In secure mode: ML-KEM establishes key material, ML-DSA authenticates clients, and AES-GCM protects federated model messages.
- Authenticated-client count when available.
- Security rejection events and rejected-message count.

## Honesty and safety

- Keep `presentation_mode`, `inference_mode`, and prediction `replayed` indicators visible and distinct.
- Never fabricate missing source, metric, client, or security values.
- Never claim multiclass attack identification, production readiness, malicious-update protection, private federated preprocessing, or live accuracy.
- Never request or display private keys, shared secrets, signatures, ciphertext, raw tensors, checkpoint paths, or dataset paths.
