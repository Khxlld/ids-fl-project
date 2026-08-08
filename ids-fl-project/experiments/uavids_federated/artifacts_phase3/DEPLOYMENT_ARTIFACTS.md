# Versioned UAVIDS deployment artifacts

Two small frozen Phase 3 artifacts are intentionally versioned so a clean clone can run the GUI's binary inference backend and initialize the Docker demonstration without retraining:

| Artifact | Purpose | SHA-256 |
|---|---|---|
| `federated_global_model.pt` | Selected global FedAvg MLP checkpoint | `5652da68689708d7dda651b562df9d79f41bd67b6db345a91d3e9e2f66d4698b` |
| `training_only_preprocessor.joblib` | Pooled-training-only median imputer and scaler | `c68e56105b4f25462bedc34bd5b2c031d577fa76fef71e810f025841bab67256` |

These values are locked in `config/phase3_locked_config.json`. The GUI adapter and Docker coordinator verify the hashes before loading either file. The PyTorch checkpoint is loaded with `weights_only=True`. The joblib file must be treated as trusted repository content because joblib/pickle deserialization is not safe for untrusted files.

The centralized and five local-only research checkpoints remain excluded because the deployed GUI uses only the selected global federated model. Raw training, validation, and locked-test partitions also remain excluded from Git.

Versioning these files does not change the model, preprocessing, threshold, training results, or frozen artifact hashes.
