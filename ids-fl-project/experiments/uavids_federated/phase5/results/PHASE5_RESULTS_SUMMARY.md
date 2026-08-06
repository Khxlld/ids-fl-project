# Phase 5 measured results

This is a controlled academic demonstration. Timings are descriptive for these runs and host, not a general performance conclusion.

## Equivalence and completion

- Plain run `00647f41-ff4f-49e1-8101-1867e3683453` and secure run `23795a6c-4d27-4d27-96e1-1da6954d7c9c` each completed 3 rounds with all five clients and 6,148 samples per round.
- Maximum secure-versus-plain aggregate difference: **0.0** (tolerance `1e-07`).
- Maximum difference from independent secure re-aggregation: **0.0**.
- Final validation macro-F1 remained **0.9517**; this is demo telemetry, not new model selection.

## Descriptive overhead

- Coordinator runtime: plain **32.35 s**, secure **27.58 s**. Completed normal secure repetitions: 24.70 s, 27.58 s.
- Compose-to-terminal runtime: plain **44.94 s**, secure **36.90 s**.
- Mean ML-KEM encapsulation / decapsulation: **2.371 / 0.144 ms**.
- Mean ML-DSA signing / server verification / client verification: **0.495 / 0.628 / 0.480 ms**.
- Mean AES-GCM encryption / decryption: **0.065 / 0.097 ms**.
- Model/update JSON envelopes expanded by roughly **34.2% / 34.2%**, largely because binary NPZ and ciphertext are represented as base64 in JSON.

## Rejection evidence

Attack run `5b65ef3c-a9db-4a7d-ab29-21a50f7dbbdf` still completed genuine training and produced 12 safe rejections across: authentication_failed, malformed_payload, metadata_mismatch, nonce_invalid, replay, signature_invalid, untrusted_identity. All three aggregate archives were byte-identical to plain mode, so rejected probes did not reach or change aggregation.

See the CSV and JSON files in this directory for machine-readable operation, size, resource, and rejection evidence.
