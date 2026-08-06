# Phase 5 event-contract extensions

Phase 5 keeps the Phase 4 `phase4-event-v1` envelope and status endpoints. Consumers that ignore unknown event types remain compatible. The following additive event types never contain keys, secrets, signatures, ciphertext, nonces, or model plaintext:

| Event | Meaning | Safe payload fields |
|---|---|---|
| `security_ready` | Server security layer initialized | version, selected algorithms, provisioning model |
| `security_handshake_started` | Signed server challenge created | algorithm names, public identity fingerprint |
| `client_authenticated` | Trusted signature verified and KEM session established | client ID, public fingerprint, algorithms |
| `security_message_protected` | Available global model encrypted | type, sequence, plaintext/protected byte counts |
| `security_message_unprotected` | Client update authenticated before coordinator entry | type, sequence, plaintext/protected byte counts |
| `security_message_rejected` | Message rejected before model processing | endpoint, safe category |
| `client_security_summary` | Client emitted secret-free operation evidence after completion | public fingerprint, timing/size arrays |

`security_message_rejected` has severity `error`, but controlled attack-test rejections do not fail the training run. Ordinary Phase 4 lifecycle, aggregation, metric, and failure events are unchanged.
