# Phase 5 security protocol

## Threat and trust model

The attacker may read, copy, delay, replay, alter, or inject HTTP application messages and may claim another logical client name. The server host, mounted runtime secrets, OQS implementation, Docker engine, training code, and provisioned public trust store are trusted. Host compromise, traffic analysis, denial of service, malicious-but-authenticated training, key exfiltration, and rollback of trusted software/configuration are outside scope.

## Algorithms

- ML-KEM-768: NIST security category 3 key encapsulation; 1,184-byte public key, 1,088-byte ciphertext, 32-byte shared secret.
- ML-DSA-65: NIST security category 3 signatures; 1,952-byte public key and 3,309-byte signature.
- HKDF-SHA-256: derives two AES-256 keys and two four-byte nonce prefixes from the KEM secret with domain-separated salt/info containing the session context.
- AES-256-GCM: protects every post-handshake application message with a unique 96-bit nonce and authenticated associated data (AAD).

Level 3 was selected as a defensible middle ground: stronger than the level-1 variants while avoiding the larger level-5 handshake objects. The final standardized names are used; no legacy Kyber/Dilithium identifiers are accepted by configuration.

## Message flow

```text
client                                            server
  |-- registration + public-key fingerprint ------>|
  |<-- signed server hello + ephemeral ML-KEM pk ---|
  | verify trust/name/run/contract/signature         |
  | ML-KEM encapsulate; sign canonical handshake     |
  |-- KEM ciphertext + signed bound metadata ------>|
  |                     verify allowlist/signature   |
  |                     ML-KEM decapsulate; HKDF     |
  |<-- AES-GCM session confirmation ----------------|
  |                                                  |
  |-- AES-GCM(model request / event / update) ------>|
  |<-- AES-GCM(global model / acknowledgement) ------|
```

The server hello binds security version, run/session/client/server IDs, frozen contract hash, algorithm names, server challenge, KEM-public-key hash, server identity fingerprint, and expiry. The client handshake repeats these fields and binds a client challenge, complete approved registration, KEM ciphertext/hash, and client identity fingerprint. Signatures cover a fixed domain label, an internal message-type label with a null delimiter, and canonical JSON (ASCII-escaped UTF-8, sorted keys, no insignificant whitespace, finite JSON values only). Because the message type is selected internally and the JSON object begins after the delimiter, the representation is unambiguous.

## Protected-message contract

The exact header field set is: security version, session ID, run ID, round, sender, recipient, message type, sequence, nonce, contract hash, plaintext SHA-256, and model SHA-256. Canonical header bytes are AES-GCM AAD. The encrypted plaintext is canonical JSON and its length/hash are checked after authentication. Unexpected or missing fields are rejected.

Each direction has a distinct AES key and HKDF-derived four-byte prefix. The remaining eight nonce bytes are a big-endian sequence number. Senders never reuse a sequence within a session; receivers require the exact next sequence and matching nonce. Invalid ciphertext does not advance receiver state, so the genuine message can still be accepted after a tampered copy. Authentication and metadata checks finish before JSON/model decoding; model archives retain Phase 4's `allow_pickle=False`, strict names/shapes/dtypes/finiteness/hash/size checks.

## Failure handling and limitations

Failures return a generic HTTP error and emit `security_message_rejected` with only endpoint and category. Keys, secrets, signatures, ciphertext, nonces, tensors, and detailed cryptographic diagnostics are never emitted. Rejected updates are not passed to the Phase 4 coordinator and therefore cannot enter FedAvg.

The implementation demonstrates authenticated, confidential, tamper-evident, context-bound, replay-resistant application messages in the controlled provisioned environment. It is not production PKI, authenticated clients may still send poisoned updates, HTTP metadata remains observable, availability is not guaranteed, server-side aggregation sees plaintext updates, and the benchmark is host-specific.
