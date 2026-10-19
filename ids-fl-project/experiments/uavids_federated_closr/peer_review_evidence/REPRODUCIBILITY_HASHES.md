# Reproducibility and data-integrity hashes

This file records hashes used by the completed development runs. It contains no dataset records.

## Committed specifications

| File | SHA-256 |
|---|---|
| `config/contribution1_baseline.json` | `7f478d341040a984e8043bd28b8fe7bfc34132e0b5b0e6dc317494697bf6a3cb` |
| `config/contribution1_multiseed_replication.json` | `7c75d1bebcca57ae259edda3418e973074f2bb4c215a1a1d75f7aa8d66a0de40` |
| `requirements.txt` | `eee09016bd47f6e904fc2eb9551f67e1fefba5066e2efa90d808e25364d60660` |
| `pyproject.toml` | `c6bd633a28df8e33ecf3d2b45a7cc42952af6819d06992600602d22a5001b461` |

The baseline-config hash above matches `config_source_sha256` in all six completed run summaries.

## Pinned upstream reference

- Official CLOSR revision: `79da4d1f2a6f4e5ecf92c4588195aaf163df40f8`
- Pinned upstream `losses/closr_loss.py`: `b07464f66dace6e7f32767f79ea1470e31a9ca4c62b06332d1d3fa0d47541035`

## External Phase 2 manifest artifacts

These hashes were reverified before the development runs:

| Manifest artifact | SHA-256 |
|---|---|
| `partition::test` | `fa094088a9ee44858401d58064b35cd76e20e7962eb1c5b68c2a3136c8b95e03` |
| `partition::train/uav-client-1` | `d30b8217542c83370f6bdec82c7b2a3c21d269b5a50179d2380f70bef648c5a5` |
| `partition::train/uav-client-2` | `4f93be7c7ae45c5ddc5a1fe9c4ecd78c38af0317347afad7f2a0e2a2be91d0a2` |
| `partition::train/uav-client-3` | `a615376eabce812e42f8e18def75a8190733027bbddbadfcbf4e7ff5229a8f6a` |
| `partition::train/uav-client-4` | `bc87fe1f5c2e300edf63eb910b01ad84e8ef42d514edf576ca29fac71408d0ae` |
| `partition::train/uav-client-5` | `57f35a61833eb300ea72e2a5eb338687998839c4887f98a5752503b26f5ecd38` |
| `partition::validation` | `e401d919379f3611aca6369380584e139996c4c35313cc3f05551766bb5631dc` |
| `partition_distributions` | `a247f4384857711317f2108e89b122d5210e9adc457e787e9aaa9aadfd2c19c6` |
| `phase2_config` | `76f00ccc840af24f1a1c8aa1b9c54ed504b55b36294b688d182be58d461ec2e4` |
| `row_assignments` | `6e4c4c4e48b5ed44b3ffb183727f7f419f14574d03a9c69d912d021796284dc6` |
| `source_assignments` | `7fd138a9482f195f7c3cab01c2ecc6b05a917ee0e73a47201fab4523acbfc2a7` |

The final-test hash records integrity only. The training runner did not parse or evaluate final-test rows.

## Recorded initialization hashes

Later replication runs explicitly saved the common within-seed initial-model hash:

| Seed | Initial-state SHA-256 |
|---:|---|
| 240519 | `66115ad8940458524cb7a736d44b14b6c534b069ab36326f2ac12241f9abe28f` |
| 240520 | `95f98c673b6a6d5101fe5a10152142f377f8c270d79243367c10a8242bd20272` |
| 240521 | `1272ade95a737d2ff9b6466093b1a7991a77bd3369a4b4511eec58546f34ffe6` |
| 240522 | `8f28c71f76753416086821d7b8c4c421e543a4647b8a9774fdcee1a9ea1437ab` |
| 240523 | `d3d4c058e0c7eac35a52ab1fda0b74b5f72caaa17af3e16b0b3466c416ded330` |

Seed `240517` is the earlier pilot. Its run predates the dedicated `rng_audit.json`, so this table does not invent an unavailable initialization hash for it.
