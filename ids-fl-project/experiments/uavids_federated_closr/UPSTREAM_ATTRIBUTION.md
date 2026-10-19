# Upstream attribution and deviations

This baseline is a modified, federated adaptation informed by Jack Wilkie et al.'s official CLOSR implementation:

- Repository: https://github.com/jackwilkie/CLOSR
- Inspected revision: `79da4d1f2a6f4e5ecf92c4588195aaf163df40f8`
- Upstream license: Apache License 2.0; a copy is retained at `THIRD_PARTY_LICENSES/CLOSR-Apache-2.0.txt`.
- Primary files inspected: `losses/closr_loss.py`, `losses/clad_loss.py`, `model/model.py`, `util/distance.py`, `data/loaders.py`, `train_closr.py`, and `eval_closr.py`.
- Audited `losses/closr_loss.py` SHA-256: `b07464f66dace6e7f32767f79ea1470e31a9ca4c62b06332d1d3fa0d47541035`.

The local code was rewritten for the UAVIDS schema, deterministic CPU execution, independent unavailable-term handling, grouped validation/calibration, IID and natural Non-IID five-client FedAvg, post-aggregation center statistics, artifact round-trips, and targeted leakage tests. A literal numerical fixture in `tests/test_loss_and_model.py` checks default loss values and gradients against the pinned source behavior. Deliberate adaptations and non-default paper-style options are documented in `EXPERIMENT_SPEC.md`. This is not a byte-for-byte copy or a claim of reproducing the Lycos2017 results.

The supporting Kodama et al. MVA 2021 paper uses supervised contrastive pretraining followed by a classifier and EVT/OpenMax-style scoring. Its EVT mechanism is intentionally not used here because it is different from CLOSR's Equation 16 score.
