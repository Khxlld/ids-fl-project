"""Manifest-checked, leakage-aware data preparation for the CLOSR baseline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ClientDataset:
    client_id: str
    features: np.ndarray
    labels: np.ndarray


@dataclass
class PreparedData:
    clients: dict[str, ClientDataset]
    selection_features: np.ndarray
    selection_labels: np.ndarray
    calibration_features: np.ndarray
    calibration_labels: np.ndarray
    preprocessor: ColumnTransformer
    feature_names: list[str]
    class_names: list[str]
    label_to_index: dict[str, int]
    audit: dict[str, Any]

    def pooled_training(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.concatenate([client.features for client in self.clients.values()], axis=0),
            np.concatenate([client.labels for client in self.clients.values()], axis=0),
        )


def make_iid_control_clients(
    natural_clients: dict[str, ClientDataset], seed: int
) -> tuple[dict[str, ClientDataset], dict[str, Any]]:
    """Repartition the identical pooled rows approximately IID.

    Natural client row counts are preserved exactly so both FedAvg controls
    have identical total rows, client weights, local epochs, and batch-count
    budgets.  Repeated stratified sampling makes class proportions similar;
    it does not synthesize or duplicate observations.
    """
    ordered = list(natural_clients.values())
    if len(ordered) < 2:
        raise ValueError("an IID FedAvg control requires at least two clients")
    features = np.concatenate([client.features for client in ordered], axis=0)
    labels = np.concatenate([client.labels for client in ordered], axis=0)
    remaining = np.arange(len(labels), dtype=np.int64)
    assignments: list[np.ndarray] = []
    for client_index, client in enumerate(ordered[:-1]):
        desired = len(client.labels)
        chosen_relative, remaining_relative = train_test_split(
            np.arange(len(remaining), dtype=np.int64),
            train_size=desired,
            random_state=seed + client_index,
            shuffle=True,
            stratify=labels[remaining],
        )
        assignments.append(remaining[chosen_relative])
        remaining = remaining[remaining_relative]
    assignments.append(remaining)
    assigned = np.concatenate(assignments)
    if len(assigned) != len(labels) or not np.array_equal(np.sort(assigned), np.arange(len(labels))):
        raise AssertionError("IID assignment did not preserve the pooled training rows exactly once")

    iid_clients: dict[str, ClientDataset] = {}
    class_count_audit: dict[str, dict[str, int]] = {}
    for natural, indices in zip(ordered, assignments):
        client_id = f"iid::{natural.client_id}"
        iid_clients[client_id] = ClientDataset(
            client_id=client_id,
            features=np.ascontiguousarray(features[indices], dtype=np.float32),
            labels=np.ascontiguousarray(labels[indices], dtype=np.int64),
        )
        class_count_audit[client_id] = {
            str(class_index): int(np.sum(labels[indices] == class_index))
            for class_index in np.unique(labels)
        }
    return iid_clients, {
        "method": "sequential deterministic stratified repartition of the pooled known-only training rows",
        "seed": int(seed),
        "pooled_rows": int(len(labels)),
        "each_row_assigned_exactly_once": True,
        "natural_client_sizes_preserved": True,
        "client_class_counts_by_index": class_count_audit,
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest_artifacts(data_root: Path, manifest: dict[str, Any]) -> dict[str, str]:
    """Hash every recorded artifact, including but not loading the final test CSV."""
    verified: dict[str, str] = {}
    for name, record in manifest["artifact_checksums"].items():
        path = data_root / Path(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"manifest artifact is missing: {path}")
        actual = sha256(path)
        if actual != record["sha256"]:
            raise ValueError(f"checksum mismatch for {name}: {actual} != {record['sha256']}")
        verified[name] = actual
    return verified


def verify_partition_integrity(data_root: Path, manifest: dict[str, Any]) -> dict[str, int]:
    """Recheck schema and global feature-signature uniqueness across all partitions."""
    features = list(manifest["approved_model_features"])
    expected_columns = features + ["original_label", "binary_label"]
    frames: list[pd.DataFrame] = []
    row_counts: dict[str, int] = {}
    partition_records = {
        name: record
        for name, record in manifest["artifact_checksums"].items()
        if name.startswith("partition::")
    }
    for name, record in partition_records.items():
        path = data_root / Path(record["path"])
        frame = pd.read_csv(path)
        if frame.columns.tolist() != expected_columns:
            raise ValueError(f"unexpected schema in {path}")
        if frame.isna().any().any():
            raise ValueError(f"missing values found in {path}")
        frames.append(frame[features])
        row_counts[name] = len(frame)
    pooled = pd.concat(frames, ignore_index=True)
    duplicate_count = int(pooled.duplicated(features, keep=False).sum())
    if duplicate_count:
        raise ValueError(f"approved-feature overlap exists across partitions: {duplicate_count} rows")
    row_counts["duplicate_feature_rows"] = duplicate_count
    row_counts["total_partition_rows"] = len(pooled)
    return row_counts


def _stratified_cap(labels: np.ndarray, maximum: int | None, seed: int) -> np.ndarray:
    if maximum is None or len(labels) <= maximum:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    base = maximum // len(classes)
    chosen: list[int] = []
    for class_index in classes:
        candidates = np.flatnonzero(labels == class_index)
        take = min(len(candidates), base)
        chosen.extend(rng.choice(candidates, size=take, replace=False).tolist())
    remaining = maximum - len(chosen)
    if remaining:
        available = np.setdiff1d(np.arange(len(labels)), np.asarray(chosen), assume_unique=False)
        chosen.extend(rng.choice(available, size=remaining, replace=False).tolist())
    return np.sort(np.asarray(chosen, dtype=np.int64))


def _class_counts(labels: np.ndarray, class_names: list[str]) -> dict[str, int]:
    return {name: int(np.sum(labels == index)) for index, name in enumerate(class_names)}


def _apply_smoke_overrides(config: dict[str, Any]) -> dict[str, Any]:
    config = json.loads(json.dumps(config))
    smoke = config["smoke"]
    config["model"]["hidden_dims"] = smoke["hidden_dims"]
    config["model"]["projection_dim"] = smoke["projection_dim"]
    config["training"]["batch_size"] = smoke["batch_size"]
    config["training"]["centralized_epochs"] = smoke["centralized_epochs"]
    config["training"]["centralized_patience"] = smoke["centralized_epochs"]
    config["training"]["federated_rounds"] = smoke["federated_rounds"]
    config["training"]["local_epochs"] = smoke["local_epochs"]
    config["execution_mode"] = "smoke"
    return config


def resolve_config(config_path: Path, smoke: bool) -> dict[str, Any]:
    config = read_json(config_path)
    config["execution_mode"] = "full_baseline_without_final_test"
    return _apply_smoke_overrides(config) if smoke else config


def prepare_data(config: dict[str, Any]) -> PreparedData:
    data_root = Path(config["data_root"])
    manifest_path = data_root / Path(config["manifest_relative_path"])
    manifest = read_json(manifest_path)
    verified = verify_manifest_artifacts(data_root, manifest)

    feature_names = list(manifest["approved_model_features"])
    class_names = list(config["known_classes"])
    withheld = set(config["globally_withheld_unknown_classes"])
    all_labels = set(manifest["binary_label_mapping"])
    if set(class_names) | withheld != all_labels or set(class_names) & withheld:
        raise ValueError("known and withheld labels must be disjoint and cover the manifest labels")
    label_to_index = {name: index for index, name in enumerate(class_names)}
    expected_columns = feature_names + ["original_label", "binary_label"]
    seed = int(config["training"]["seed"])
    smoke_mode = config.get("execution_mode") == "smoke"
    max_client_rows = config["smoke"]["max_rows_per_client"] if smoke_mode else None

    raw_client_frames: dict[str, pd.DataFrame] = {}
    unknown_removed_by_client: dict[str, int] = {}
    for client_number, item in enumerate(manifest["training_clients"]):
        client_id = item["client_id"]
        record = manifest["artifact_checksums"][f"partition::train/{client_id}"]
        frame = pd.read_csv(data_root / Path(record["path"]))
        if frame.columns.tolist() != expected_columns:
            raise ValueError(f"unexpected schema for {client_id}")
        unknown_mask = frame["original_label"].isin(withheld)
        unknown_removed_by_client[client_id] = int(unknown_mask.sum())
        known_frame = frame.loc[~unknown_mask].copy()
        if known_frame["original_label"].isin(withheld).any():
            raise AssertionError("unknown row survived client filtering")
        mapped = known_frame["original_label"].map(label_to_index)
        if mapped.isna().any():
            raise ValueError(f"unmapped label in {client_id}")
        indices = _stratified_cap(mapped.to_numpy(dtype=np.int64), max_client_rows, seed + client_number)
        raw_client_frames[client_id] = known_frame.iloc[indices].reset_index(drop=True)

    pooled_training = pd.concat(raw_client_frames.values(), ignore_index=True)
    if pooled_training["original_label"].isin(withheld).any():
        raise AssertionError("unknown row reached preprocessing fit")
    preprocessor = ColumnTransformer(
        [("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), feature_names)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.fit(pooled_training[feature_names])

    clients: dict[str, ClientDataset] = {}
    for client_id, frame in raw_client_frames.items():
        client_features = np.ascontiguousarray(preprocessor.transform(frame[feature_names]), dtype=np.float32)
        client_labels = frame["original_label"].map(label_to_index).to_numpy(dtype=np.int64)
        clients[client_id] = ClientDataset(client_id, client_features, client_labels)

    validation_record = manifest["artifact_checksums"]["partition::validation"]
    validation = pd.read_csv(data_root / Path(validation_record["path"]))
    if validation.columns.tolist() != expected_columns:
        raise ValueError("unexpected validation schema")
    assignments = pd.read_csv(data_root / Path(config["row_assignments_relative_path"]))
    validation_meta = assignments.loc[
        assignments["status"].eq("included") & assignments["partition"].eq("validation")
    ].sort_values("FlowID").reset_index(drop=True)
    if len(validation_meta) != len(validation):
        raise ValueError("validation traceability row count does not align")
    if not validation["original_label"].reset_index(drop=True).equals(validation_meta["original_label"]):
        raise ValueError("validation traceability order does not align with partition generation")

    known_mask = ~validation["original_label"].isin(withheld)
    known_validation = validation.loc[known_mask].reset_index(drop=True)
    known_groups = validation_meta.loc[known_mask.to_numpy(), "SrcAddr"].reset_index(drop=True)
    known_labels = known_validation["original_label"].map(label_to_index).to_numpy(dtype=np.int64)
    splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=float(config["validation_group_split"]["selection_fraction"]),
        random_state=int(config["validation_group_split"]["seed"]),
    )
    selection_index, calibration_index = next(
        splitter.split(known_validation, known_labels, groups=known_groups)
    )
    selection_sources = set(known_groups.iloc[selection_index])
    calibration_sources = set(known_groups.iloc[calibration_index])
    if selection_sources & calibration_sources:
        raise AssertionError("selection and calibration sources overlap")
    expected_indices = set(range(len(class_names)))
    if set(known_labels[selection_index]) != expected_indices or set(known_labels[calibration_index]) != expected_indices:
        raise ValueError("every known class must occur in selection and calibration")

    if smoke_mode:
        selection_cap = _stratified_cap(
            known_labels[selection_index], config["smoke"]["max_selection_rows"], seed + 1000
        )
        calibration_cap = _stratified_cap(
            known_labels[calibration_index], config["smoke"]["max_calibration_rows"], seed + 2000
        )
        selection_index = selection_index[selection_cap]
        calibration_index = calibration_index[calibration_cap]

    selection_features = np.ascontiguousarray(
        preprocessor.transform(known_validation.iloc[selection_index][feature_names]), dtype=np.float32
    )
    calibration_features = np.ascontiguousarray(
        preprocessor.transform(known_validation.iloc[calibration_index][feature_names]), dtype=np.float32
    )
    selection_labels = known_labels[selection_index]
    calibration_labels = known_labels[calibration_index]
    if not np.isfinite(selection_features).all() or not np.isfinite(calibration_features).all():
        raise ValueError("non-finite values after preprocessing")

    audit: dict[str, Any] = {
        "data_root": str(data_root.resolve()),
        "manifest": str(manifest_path.resolve()),
        "verified_artifact_count": len(verified),
        "verified_artifacts": verified,
        "test_partition_hash_checked": "partition::test" in verified,
        "test_partition_loaded": False,
        "integrity_access_disclosure": {
            "training_runner_test_csv": "file bytes read only for SHA-256; CSV rows, features, and labels not parsed",
            "training_runner_row_assignments": "all nine columns of the combined traceability CSV are parsed for all records, including final-test records; only FlowID, status, partition, original_label, and SrcAddr from validation records affect the run",
            "targeted_integrity_test": "verify_partition_integrity parses all seven partition CSV schemas, 15 approved feature columns, original_label, and binary_label, including the final test partition, solely to check missing values and cross-partition feature-signature overlap",
            "model_selection_or_performance_use": False,
        },
        "preprocessor_fit_scope": config["preprocessing"]["fit_scope"],
        "preprocessor_fit_rows": len(pooled_training),
        "preprocessor_unknown_rows": int(pooled_training["original_label"].isin(withheld).sum()),
        "unknown_removed_by_client": unknown_removed_by_client,
        "client_known_class_counts": {
            client_id: _class_counts(client.labels, class_names) for client_id, client in clients.items()
        },
        "selection_rows": len(selection_labels),
        "calibration_rows": len(calibration_labels),
        "selection_class_counts": _class_counts(selection_labels, class_names),
        "calibration_class_counts": _class_counts(calibration_labels, class_names),
        "selection_source_count_before_smoke_cap": len(selection_sources),
        "calibration_source_count_before_smoke_cap": len(calibration_sources),
        "source_overlap": sorted(selection_sources & calibration_sources),
        "withheld_validation_rows_not_used": int((~known_mask).sum()),
        "known_classes": class_names,
        "globally_withheld_unknown_classes": sorted(withheld),
        "execution_mode": config["execution_mode"],
    }
    return PreparedData(
        clients=clients,
        selection_features=selection_features,
        selection_labels=selection_labels,
        calibration_features=calibration_features,
        calibration_labels=calibration_labels,
        preprocessor=preprocessor,
        feature_names=feature_names,
        class_names=class_names,
        label_to_index=label_to_index,
        audit=audit,
    )
