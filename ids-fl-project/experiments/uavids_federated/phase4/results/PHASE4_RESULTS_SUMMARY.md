# Phase 4 measured results

This is a **demo-mode** run initialized from the locked Phase 3 federated checkpoint. Its metrics do not replace the Phase 3 research results.

## Runtime

- Run ID: `9ec38846-0983-4f1e-a343-f7f8a320ba46`
- Compose start to API ready: **6.98 s**
- Compose start to completed state: **28.16 s**
- Coordinator start to all five clients ready: **14.33 s**
- Coordinator total runtime: **20.58 s**
- Round times: **r1 3.45 s, r2 1.19 s, r3 1.24 s**
- Aggregation times: **r1 2.052 ms, r2 1.670 ms, r3 1.677 ms**
- Update archive: **46.7-46.8 KiB**; HTTP body: **62.7-62.8 KiB**.

| Client | Round 1 train ms | Round 2 train ms | Round 3 train ms | Peak MiB | Peak CPU |
|---|---:|---:|---:|---:|---:|
| uav-client-1 | 1373.6 | 72.6 | 72.1 | 270.1 | 65.1% |
| uav-client-2 | 1819.5 | 101.5 | 66.7 | 270.7 | 54.6% |
| uav-client-3 | 2588.3 | 34.1 | 33.9 | 269.2 | 36.1% |
| uav-client-4 | 1049.6 | 92.8 | 165.9 | 272.4 | 99.0% |
| uav-client-5 | 2203.1 | 73.0 | 91.7 | 270.4 | 45.5% |

Round 1 includes one-time library/optimizer warm-up under simultaneous constrained startup. The Zero 2 W-inspired client was the first-round straggler; later training was much shorter and client 4's larger partition became significant.

## Validation and failure behavior

The final demo validation macro-F1 was **0.9517** at the locked threshold 0.42. This is presentation telemetry, not an official model-selection result.

The live incompatible-contract update was rejected with HTTP 400 and emitted an `update_rejected` error event. The timeout unit test separately confirms that an unavailable client is named in the terminal failure state.

## Verification

- All five clients participated in all three rounds with sample counts `1230, 1046, 591, 2027, 1254` (total 6,148).
- Independent aggregation agreed exactly: maximum absolute tensor difference **0.0**.
- Each client exposed exactly one training CSV; the server exposed zero training CSVs and one validation CSV.
- All roots were read-only, data/reference mounts were read-only, capabilities were dropped, and clients exited with code 0.
- The image contained no CSV files, and frozen Phase 3 hashes remained unchanged.
