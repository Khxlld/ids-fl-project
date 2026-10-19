from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from uavids_closr.data import (
    ClientDataset,
    make_iid_control_clients,
    prepare_data,
    read_json,
    resolve_config,
    verify_manifest_artifacts,
    verify_partition_integrity,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "contribution1_baseline.json"


def test_manifest_schema_hashes_and_feature_overlap() -> None:
    config = read_json(CONFIG_PATH)
    data_root = Path(config["data_root"])
    manifest = read_json(data_root / config["manifest_relative_path"])
    verified = verify_manifest_artifacts(data_root, manifest)
    assert len(verified) == 11
    integrity = verify_partition_integrity(data_root, manifest)
    assert integrity["duplicate_feature_rows"] == 0
    assert integrity["total_partition_rows"] == 113883


def test_unknown_exclusion_group_split_and_training_only_preprocessing(tmp_path) -> None:
    config = resolve_config(CONFIG_PATH, smoke=True)
    prepared = prepare_data(config)
    assert prepared.audit["preprocessor_unknown_rows"] == 0
    assert prepared.audit["source_overlap"] == []
    assert prepared.audit["test_partition_hash_checked"] is True
    assert prepared.audit["test_partition_loaded"] is False
    assert prepared.audit["withheld_validation_rows_not_used"] > 0
    assert len(prepared.clients) == 5
    assert prepared.audit["client_known_class_counts"]["uav-client-4"]["Sybil Attack"] == 0
    assert prepared.audit["client_known_class_counts"]["uav-client-5"]["Sybil Attack"] == 0
    all_labels = np.concatenate([client.labels for client in prepared.clients.values()])
    assert set(all_labels) == set(range(len(prepared.class_names)))
    assert set(prepared.selection_labels) == set(range(len(prepared.class_names)))
    assert set(prepared.calibration_labels) == set(range(len(prepared.class_names)))

    preprocessor_path = tmp_path / "preprocessor.joblib"
    joblib.dump(prepared.preprocessor, preprocessor_path)
    reloaded = joblib.load(preprocessor_path)
    fixture = pd.DataFrame(np.zeros((2, len(prepared.feature_names))), columns=prepared.feature_names)
    np.testing.assert_allclose(prepared.preprocessor.transform(fixture), reloaded.transform(fixture))


def test_iid_control_preserves_every_pooled_row_and_natural_client_sizes() -> None:
    features = np.arange(36, dtype=np.float32).reshape(12, 3)
    labels = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)
    natural = {
        "a": ClientDataset("a", features[:6], labels[:6]),
        "b": ClientDataset("b", features[6:9], labels[6:9]),
        "c": ClientDataset("c", features[9:], labels[9:]),
    }
    iid, audit = make_iid_control_clients(natural, seed=44)
    assert [len(client.labels) for client in iid.values()] == [6, 3, 3]
    reassigned = np.concatenate([client.features for client in iid.values()], axis=0)
    np.testing.assert_array_equal(np.sort(reassigned[:, 0]), np.sort(features[:, 0]))
    assert audit["each_row_assigned_exactly_once"] is True
    assert audit["natural_client_sizes_preserved"] is True
