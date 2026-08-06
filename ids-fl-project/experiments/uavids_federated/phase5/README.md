# Phase 5 post-quantum-secured federated demo

Phase 5 wraps the verified Phase 4 HTTP protocol without changing its data, preprocessing, MLP, local training, sample counts, or FedAvg. The selected level-3 algorithms are ML-KEM-768 for shared-secret establishment, HKDF-SHA-256 for separate directional keys and nonce prefixes, AES-256-GCM for model-message confidentiality and authentication, and ML-DSA-65 for provisioned client/server identities and signed handshakes. `liboqs` and `liboqs-python` are pinned to 0.16.0; `cryptography` is pinned to 50.0.0.

## Modes and commands

Run these from `experiments/uavids_federated` in PowerShell.

Plain Phase 4 baseline:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\phase4\scripts\run_demo.ps1
```

Provision runtime-only demonstration identities, build, run all five secure clients, and verify aggregation:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\phase5\scripts\run_secure_demo.ps1
```

Run controlled tampering, identity, metadata, replay, duplicate-nonce, and malformed-message probes while genuine training continues:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\phase5\scripts\run_attack_tests.ps1
```

Run the Phase 3, Phase 4, and Phase 5 automated tests inside the OQS image:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\phase5\scripts\run_security_tests.ps1
```

Observe and stop without deleting evidence:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\phase5\scripts\observe_secure_demo.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\phase5\scripts\stop_secure_demo.ps1
```

## Provisioning and trust

The run script creates six ML-DSA identities under ignored `phase5/runtime/keys/`. Each client receives its own private key as a read-only runtime mount; the server receives only its private key. Every service receives the public trust store. Private keys are not copied into the image, logs, events, or saved results. This is an explicit controlled academic allowlist, not a production PKI: it has no certificates, revocation service, HSM, automated rotation, or organizational identity proofing.

The server uses a fresh in-memory ML-KEM key pair for each process. Its signed hello binds that key to the run, client, contract, algorithms, challenge, and expiry. The authenticated client encapsulates to it and signs the complete canonical handshake. HKDF derives independent client-to-server/server-to-client AES keys and four-byte nonce prefixes. Each 96-bit nonce is `directional_prefix || uint64_sequence`; strict sequence and nonce checks reject replay before decryption.

See [SECURITY_PROTOCOL.md](SECURITY_PROTOCOL.md) for the precise flow and [EVENT_CONTRACT_EXTENSIONS.md](EVENT_CONTRACT_EXTENSIONS.md) for safe observable events.

## Evidence and scope

`verify_secure_demo.py` independently recomputes every aggregate, compares plain and secure tensors at `1e-7`, checks frozen Phase 3 hashes, and scans evidence for secret material. `verify_container_isolation.py` checks mounts, read-only roots, limits, and the built image. `summarize_results.py` turns ignored runtime output into committed, secret-free tables in `results/`.

This protects application model messages against passive payload inspection, impersonation by identities outside the allowlist, tampering, misaddressing, and replay under the stated trust model. It does not hide HTTP metadata, provide forward secrecy after ML-KEM session compromise, validate whether an authenticated client's model update is benign, implement secure aggregation or differential privacy, or make the prototype production-ready. HTTP should be replaced or complemented by production TLS and a real identity lifecycle in a deployed system.

Authoritative design references: [NIST FIPS 203](https://csrc.nist.gov/pubs/fips/203/final), [NIST FIPS 204](https://csrc.nist.gov/pubs/fips/204/final), [OQS ML-KEM](https://openquantumsafe.org/liboqs/algorithms/kem/ml-kem.html), [OQS ML-DSA](https://openquantumsafe.org/liboqs/algorithms/sig/ml-dsa.html), [liboqs-python](https://github.com/open-quantum-safe/liboqs-python), and [AES-GCM API guidance](https://cryptography.io/en/latest/hazmat/primitives/aead/).
