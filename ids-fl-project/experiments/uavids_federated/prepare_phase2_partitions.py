"""Materialize the Phase 2 leakage-controlled UAVIDS-2025 partitions.

This script performs data hygiene and partition construction only. It does not
fit preprocessing, select model hyperparameters, or train any model.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "phase2_partition_config.json"
PHASE1_AUDIT_PATH = ROOT / "results_audit" / "audit_summary.json"
PARTITION_DIR = ROOT / "partitions" / "phase2"
RESULTS_DIR = ROOT / "results_phase2"


def digest(path: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ip_sort_key(value: str) -> int:
    return int(ipaddress.ip_address(value))


def partition_filename(partition: str) -> str:
    if partition.startswith("train/"):
        return "train_" + partition.split("/", 1)[1].replace("-", "_") + ".csv"
    return partition + ".csv"


def markdown_distribution_table(distributions: pd.DataFrame) -> str:
    columns = [
        "partition", "sources", "rows", "normal", "attack", "normal_pct", "attack_pct",
        "Blackhole Attack", "Flooding Attack", "Sybil Attack", "Wormhole Attack",
    ]
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for record in distributions[columns].to_dict(orient="records"):
        values = []
        for column in columns:
            value = record[column]
            if column in {"normal_pct", "attack_pct"}:
                values.append(f"{value:.2f}%")
            elif isinstance(value, (int, np.integer)):
                values.append(f"{int(value):,}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *rows])


def assign_heldout_sources(source_counts: pd.DataFrame, class_order: list[str]) -> dict[str, str]:
    """Balance whole source groups across validation and test deterministically."""
    balance_columns = class_order + ["Total"]
    totals = source_counts[balance_columns].sum().to_numpy(dtype=float)
    assert np.all(totals > 0)

    order = sorted(
        source_counts.index,
        key=lambda source: (-int(source_counts.loc[source, "Total"]), ip_sort_key(source)),
    )
    validation_counts = np.zeros(len(balance_columns), dtype=float)
    test_counts = np.zeros(len(balance_columns), dtype=float)
    validation_sources: list[str] = []
    test_sources: list[str] = []

    for source in order:
        vector = source_counts.loc[source, balance_columns].to_numpy(dtype=float)
        validation_cost = np.sum(((validation_counts + vector - test_counts) / totals) ** 2)
        test_cost = np.sum(((validation_counts - (test_counts + vector)) / totals) ** 2)

        if validation_cost < test_cost - 1e-15:
            destination = "validation"
        elif test_cost < validation_cost - 1e-15:
            destination = "test"
        else:
            destination = "validation" if len(validation_sources) <= len(test_sources) else "test"

        if destination == "validation":
            validation_sources.append(source)
            validation_counts += vector
        else:
            test_sources.append(source)
            test_counts += vector

    assignment = {source: "validation" for source in validation_sources}
    assignment.update({source: "test" for source in test_sources})
    assert set(assignment) == set(source_counts.index)
    return assignment


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw_config = config["raw_dataset"]
    data_path = (ROOT / raw_config["relative_path"]).resolve()
    approved_features = list(config["approved_model_features"])
    excluded_fields = set(config["excluded_model_fields"])
    label_mapping = {key: int(value) for key, value in config["binary_label_mapping"].items()}
    class_order = list(label_mapping)
    training_clients = config["training_clients"]
    client_source_map = {item["client_id"]: item["source"] for item in training_clients}

    assert data_path.is_file(), data_path
    assert digest(data_path, "md5") == raw_config["expected_md5"]
    assert digest(data_path, "sha256") == raw_config["expected_sha256"]
    assert len(client_source_map) == 5
    assert len(set(client_source_map.values())) == 5
    assert set(approved_features).isdisjoint(excluded_fields)
    assert set(label_mapping.values()) == {0, 1}
    assert label_mapping["Normal Traffic"] == 0

    phase1 = json.loads(PHASE1_AUDIT_PATH.read_text(encoding="utf-8"))
    assert phase1["official_md5"] == raw_config["expected_md5"]
    assert phase1["shape"]["rows"] == raw_config["expected_rows"]

    raw = pd.read_csv(data_path, low_memory=False)
    assert raw.shape == (raw_config["expected_rows"], raw_config["expected_columns"])
    assert raw["FlowID"].is_unique
    assert set(raw["label"].unique()) == set(class_order)
    assert raw.isna().sum().sum() == 0
    assert np.isfinite(raw.select_dtypes(include=[np.number]).to_numpy()).all()

    raw["original_label"] = raw["label"]
    raw["binary_label"] = raw["original_label"].map(label_mapping).astype("int8")

    suspicious_mask = (
        ~raw["PacketDropRate"].between(0, 1)
        | raw["LostPackets"].gt(raw["TxPackets"])
    )
    eligible = raw.loc[~suspicious_mask].copy()
    eligible = eligible.sort_values("FlowID")

    signature_groups = eligible.groupby(approved_features, dropna=False, sort=True)
    eligible["signature_group_id"] = signature_groups.ngroup().astype("int64")
    original_label_counts = eligible.groupby("signature_group_id")["original_label"].nunique()
    binary_label_counts = eligible.groupby("signature_group_id")["binary_label"].nunique()
    assert int(original_label_counts.max()) == 1, "Approved feature signature has conflicting multiclass labels"
    assert int(binary_label_counts.max()) == 1, "Approved feature signature has conflicting binary labels"

    representative_flow = eligible.groupby("signature_group_id")["FlowID"].transform("min")
    eligible["representative_flow_id"] = representative_flow.astype("int64")
    duplicate_mask = eligible["FlowID"].ne(eligible["representative_flow_id"])
    unique_rows = eligible.loc[~duplicate_mask].copy()
    assert not unique_rows.duplicated(approved_features).any()

    training_sources = set(client_source_map.values())
    assert training_sources <= set(unique_rows["SrcAddr"])
    heldout = unique_rows.loc[~unique_rows["SrcAddr"].isin(training_sources)]
    heldout_source_counts = pd.crosstab(heldout["SrcAddr"], heldout["original_label"]).reindex(
        columns=class_order, fill_value=0
    )
    heldout_source_counts["Total"] = heldout_source_counts.sum(axis=1)
    heldout_assignment = assign_heldout_sources(heldout_source_counts, class_order)

    source_partition: dict[str, str] = {
        source: f"train/{client_id}" for client_id, source in client_source_map.items()
    }
    source_partition.update(heldout_assignment)
    assert set(source_partition) == set(unique_rows["SrcAddr"])
    unique_rows["partition"] = unique_rows["SrcAddr"].map(source_partition)
    assert unique_rows["partition"].notna().all()

    ordered_partitions = [f"train/{item['client_id']}" for item in training_clients] + ["validation", "test"]
    partition_frames = {
        partition: unique_rows.loc[unique_rows["partition"].eq(partition)].sort_values("FlowID").copy()
        for partition in ordered_partitions
    }

    # Leakage, coverage, and usability assertions on the in-memory design.
    partition_flow_ids = {name: set(frame["FlowID"]) for name, frame in partition_frames.items()}
    partition_sources = {name: set(frame["SrcAddr"]) for name, frame in partition_frames.items()}
    for i, left in enumerate(ordered_partitions):
        for right in ordered_partitions[i + 1:]:
            assert partition_flow_ids[left].isdisjoint(partition_flow_ids[right])
            assert partition_sources[left].isdisjoint(partition_sources[right])
    assert set().union(*partition_flow_ids.values()) == set(unique_rows["FlowID"])
    assert sum(len(frame) for frame in partition_frames.values()) == len(unique_rows)
    assert not pd.concat(partition_frames.values(), ignore_index=True).duplicated(approved_features).any()
    for partition, frame in partition_frames.items():
        assert set(frame["binary_label"].unique()) == {0, 1}, f"{partition} lacks one binary class"
        if partition.startswith("train/"):
            expected_source = client_source_map[partition.split("/", 1)[1]]
            assert set(frame["SrcAddr"]) == {expected_source}

    PARTITION_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_columns = approved_features + ["original_label", "binary_label"]
    prohibited_identifiers = set(config["excluded_model_fields"])
    assert set(output_columns).isdisjoint(prohibited_identifiers - {"label"})
    assert set(approved_features).isdisjoint(prohibited_identifiers)

    partition_paths: dict[str, Path] = {}
    for partition in ordered_partitions:
        output_path = PARTITION_DIR / partition_filename(partition)
        partition_frames[partition][output_columns].to_csv(
            output_path, index=False, float_format="%.17g", lineterminator="\n"
        )
        partition_paths[partition] = output_path

    expected_partition_files = {path.resolve() for path in partition_paths.values()}
    actual_partition_files = {path.resolve() for path in PARTITION_DIR.glob("*.csv")}
    assert actual_partition_files == expected_partition_files, (
        "Unexpected stale partition files exist", actual_partition_files - expected_partition_files
    )

    # Row-level traceability accounts for every raw FlowID without placing identifiers in model files.
    assignments = raw[["FlowID", "SrcAddr", "original_label", "binary_label"]].copy()
    assignments["signature_group_id"] = pd.Series(pd.NA, index=assignments.index, dtype="Int64")
    assignments["representative_flow_id"] = pd.Series(pd.NA, index=assignments.index, dtype="Int64")
    assignments.loc[eligible.index, "signature_group_id"] = eligible["signature_group_id"].astype("Int64")
    assignments.loc[eligible.index, "representative_flow_id"] = eligible["representative_flow_id"].astype("Int64")
    assignments["status"] = ""
    assignments["partition"] = "excluded"
    assignments["exclusion_reason"] = ""

    assignments.loc[suspicious_mask, "status"] = "excluded_suspicious_packet_drop"
    assignments.loc[suspicious_mask, "exclusion_reason"] = config["suspicious_row_policy"]["rule"]
    duplicate_indices = eligible.index[duplicate_mask]
    assignments.loc[duplicate_indices, "status"] = "excluded_same_label_repeated_signature"
    assignments.loc[duplicate_indices, "exclusion_reason"] = (
        "Repeated approved-feature signature; lowest FlowID retained globally"
    )
    retained_indices = unique_rows.index
    assignments.loc[retained_indices, "status"] = "included"
    assignments.loc[retained_indices, "partition"] = unique_rows["partition"]
    assert assignments["status"].ne("").all()
    assert len(assignments) == len(raw)

    row_assignment_path = RESULTS_DIR / "row_assignments.csv"
    assignments.to_csv(row_assignment_path, index=False, lineterminator="\n")

    source_rows = []
    raw_by_source = raw.groupby("SrcAddr")
    unique_by_source = unique_rows.groupby("SrcAddr")
    for source in sorted(raw["SrcAddr"].unique(), key=ip_sort_key):
        retained = unique_by_source.get_group(source)
        partition = source_partition[source]
        retained_counts = retained["original_label"].value_counts()
        source_rows.append({
            "SrcAddr": source,
            "physical_uav_verified": False,
            "assigned_partition": partition,
            "client_id": partition.split("/", 1)[1] if partition.startswith("train/") else "",
            "raw_rows": int(len(raw_by_source.get_group(source))),
            "retained_rows": int(len(retained)),
            "excluded_suspicious_rows": int((suspicious_mask & raw["SrcAddr"].eq(source)).sum()),
            "excluded_duplicate_rows": int(assignments["SrcAddr"].eq(source).mul(
                assignments["status"].eq("excluded_same_label_repeated_signature")
            ).sum()),
            "normal": int(retained_counts.get("Normal Traffic", 0)),
            "attack": int(retained["binary_label"].eq(1).sum()),
            **{label: int(retained_counts.get(label, 0)) for label in class_order},
        })
    source_assignments = pd.DataFrame(source_rows)
    source_assignment_path = RESULTS_DIR / "source_assignments.csv"
    source_assignments.to_csv(source_assignment_path, index=False, lineterminator="\n")

    distribution_rows = []
    for partition in ordered_partitions:
        frame = partition_frames[partition]
        counts = frame["original_label"].value_counts()
        normal = int(frame["binary_label"].eq(0).sum())
        attack = int(frame["binary_label"].eq(1).sum())
        distribution_rows.append({
            "partition": partition,
            "sources": int(frame["SrcAddr"].nunique()),
            "rows": int(len(frame)),
            "normal": normal,
            "attack": attack,
            "normal_pct": float(100 * normal / len(frame)),
            "attack_pct": float(100 * attack / len(frame)),
            **{label: int(counts.get(label, 0)) for label in class_order},
        })
    distributions = pd.DataFrame(distribution_rows)
    distribution_path = RESULTS_DIR / "partition_distributions.csv"
    distributions.to_csv(distribution_path, index=False, float_format="%.10g", lineterminator="\n")

    # Round-trip the model-facing files and prove that prohibited fields were not serialized.
    for partition, output_path in partition_paths.items():
        loaded = pd.read_csv(output_path)
        assert loaded.columns.tolist() == output_columns
        assert len(loaded) == len(partition_frames[partition])
        assert set(loaded["binary_label"].unique()) == {0, 1}
        assert set(loaded.columns).isdisjoint(prohibited_identifiers)
        assert loaded.isna().sum().sum() == 0

    row_accounting = {
        "raw_rows": int(len(raw)),
        "excluded_suspicious_rows": int(suspicious_mask.sum()),
        "excluded_conflicting_signature_rows": 0,
        "excluded_same_label_duplicate_rows": int(duplicate_mask.sum()),
        "retained_unique_rows": int(len(unique_rows)),
    }
    assert (
        row_accounting["excluded_suspicious_rows"]
        + row_accounting["excluded_same_label_duplicate_rows"]
        + row_accounting["retained_unique_rows"]
        == row_accounting["raw_rows"]
    )

    artifact_paths = {
        "phase2_config": CONFIG_PATH,
        "row_assignments": row_assignment_path,
        "source_assignments": source_assignment_path,
        "partition_distributions": distribution_path,
        **{f"partition::{name}": path for name, path in partition_paths.items()},
    }
    checksums = {
        name: {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
        for name, path in artifact_paths.items()
    }

    validation_sources = sorted(
        [source for source, partition in heldout_assignment.items() if partition == "validation"],
        key=ip_sort_key,
    )
    test_sources = sorted(
        [source for source, partition in heldout_assignment.items() if partition == "test"],
        key=ip_sort_key,
    )

    manifest = {
        "phase": 2,
        "design_version": config["design_version"],
        "source_interpretation": config["source_interpretation"],
        "raw_dataset": {
            "path": str(data_path),
            "rows": int(len(raw)),
            "columns": int(raw_config["expected_columns"]),
            "md5": digest(data_path, "md5"),
            "sha256": digest(data_path, "sha256"),
        },
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "row_accounting": row_accounting,
        "approved_model_features": approved_features,
        "excluded_model_fields": config["excluded_model_fields"],
        "binary_label_mapping": label_mapping,
        "training_clients": training_clients,
        "validation_sources": validation_sources,
        "test_sources": test_sources,
        "heldout_assignment": config["heldout_assignment"],
        "partition_distributions": distributions.to_dict(orient="records"),
        "artifact_checksums": checksums,
        "verification": {
            "five_training_clients": len(training_clients) == 5,
            "all_partitions_flowid_disjoint": True,
            "all_partitions_source_disjoint": True,
            "approved_signature_overlap_between_partitions": 0,
            "prohibited_identifiers_in_feature_list": [],
            "all_clients_have_both_binary_classes": True,
            "validation_has_both_binary_classes": True,
            "test_has_both_binary_classes": True,
            "raw_rows_fully_accounted": True,
            "preprocessor_fitted": False,
            "training_or_tuning_performed": False,
        },
        "evaluation_claim": config["heldout_assignment"]["claim"],
        "future_training_imbalance_policy": config["future_training_imbalance_policy"],
    }
    manifest_path = RESULTS_DIR / "partition_manifest.json"
    write_json(manifest_path, manifest)

    distribution_md = markdown_distribution_table(distributions)
    summary = f"""# Phase 2 — trustworthy UAVIDS-2025 partitions

Generated deterministically by `prepare_phase2_partitions.py` from the official hash-verified CSV. No preprocessing was fitted and no model was trained.

## Selected design

- `SrcAddr` is accepted only as a **source-based logical client** key. It is not claimed to identify a verified physical UAV.
- The provisional five sources were retained after applying the final quality/signature rules. Selecting the five largest sources was rejected because that set is dominated by similar flooding-heavy sources and contains no Sybil observations; aggregating many sources into each client was rejected because it weakens the one-source logical-client interpretation.
- The five training sources remain atomic. Every other retained source is assigned whole to validation or test by a deterministic largest-source-first greedy algorithm that balances original class counts and total rows.
- Validation and test use all remaining retained data with natural post-policy prevalence. Neither is balanced, sampled, fitted, or used for hyperparameter decisions.

## Row policy and accounting

- Raw rows: **{row_accounting['raw_rows']:,}**.
- Excluded suspicious packet-drop rows: **{row_accounting['excluded_suspicious_rows']:,}**. These have `PacketDropRate` outside [0,1] or `LostPackets > TxPackets`; no values were clipped or imputed.
- Conflicting approved-signature rows: **0**.
- Excluded same-label repeated approved-feature signatures: **{row_accounting['excluded_same_label_duplicate_rows']:,}**. The lowest `FlowID` representative was retained globally, before source assignment.
- Retained unique rows: **{row_accounting['retained_unique_rows']:,}**.

## Partition distributions

{distribution_md}

All five clients contain normal and attack rows. Their natural attack prevalence remains deliberately non-IID. If Phase 3 needs imbalance mitigation, loss weights or a sampler must be computed from training-client labels only; validation and test must remain unchanged.

## Approved model inputs

{', '.join(f'`{feature}`' for feature in approved_features)}

The model-facing partition files contain only these 15 numeric features plus `original_label` and `binary_label`.

## Excluded fields

- `FlowID`, `SrcAddr`, and `DstAddr`: row/source/endpoint identifiers and shortcut risks.
- `SrcPort` and `DstPort`: two-valued, identical to one another, and strongly scenario-associated.
- `Protocol`: constant UDP.
- `MeanPacketSize`: exposes the same port/application regime and is redundant with packet/byte counts.
- Raw `label`: retained under `original_label` for traceability and mapped to binary `binary_label`; it is never an input.

## What this split can test

It tests generalization from five selected logical source addresses to many unseen source addresses under a shared simulated dataset, while preventing exact approved-feature signature overlap. It does **not** establish scenario-, simulation-run-, temporal-, capture-session-, network-, or verified-physical-UAV generalization because those metadata are absent.

## Remaining limitations

- Source addresses can represent attack-specific or forged identities, particularly under Sybil attacks.
- Destination and latent scenario membership can connect otherwise source-disjoint rows, although destination identity is excluded from model inputs.
- Feature engineering was defined using the Phase 1 whole-dataset audit. Validation/test labels were used only to ensure viable group-level distributions, not for model fitting or performance selection.
- The dataset-paper normalization statement conflicts with the raw-scale official CSV; Phase 3 must fit preprocessing on combined training-client rows only.

## Phase 3 boundary

Phase 3 should fit training-only preprocessing, build the binary MLP, and run centralized/local/federated comparisons using this locked manifest. It must select checkpoints with validation data and evaluate the final test partition only after all model and hyperparameter decisions are fixed.
"""
    (RESULTS_DIR / "PHASE2_PARTITION_SUMMARY.md").write_text(summary, encoding="utf-8")

    print(distributions.to_string(index=False))
    print(json.dumps(row_accounting, indent=2))
    print(f"Wrote {len(partition_paths)} model-facing partitions to {PARTITION_DIR}")
    print(f"Manifest: {manifest_path}")
    print("All Phase 2 assertions passed; no preprocessing or training was performed.")


if __name__ == "__main__":
    main()
