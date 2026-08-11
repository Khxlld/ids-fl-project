"""Generate the small, traceable locked-test pool used by the presentation GUI.

The output deliberately contains original dataset rows, but it is consumed only
by the GUI adapter.  The browser receives a small metadata projection and the
frozen model receives only the 15 approved Phase 2 features.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = Path(__file__).resolve().parent
OUTPUT = HANDOFF / "examples" / "locked_test_demo_pool.json"
COUNTS = {
    "Normal Traffic": 250,
    "Blackhole Attack": 50,
    "Flooding Attack": 50,
    "Sybil Attack": 50,
    "Wormhole Attack": 50,
}


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def main() -> None:
    config_path = ROOT / "config" / "phase2_partition_config.json"
    manifest_path = ROOT / "results_phase2" / "partition_manifest.json"
    raw_path = ROOT.parent / "UAVIDS-2025" / "UAVIDS-2025.csv"
    test_path = ROOT / "partitions" / "phase2" / "test.csv"
    assignments_path = ROOT / "results_phase2" / "row_assignments.csv"
    predictions_path = ROOT / "results_phase3" / "locked_test_predictions.csv"

    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert digest(raw_path) == config["raw_dataset"]["expected_sha256"]
    assert digest(raw_path, "md5") == config["raw_dataset"]["expected_md5"]
    assert digest(test_path) == manifest["artifact_checksums"]["partition::test"]["sha256"]
    assert digest(assignments_path) == manifest["artifact_checksums"]["row_assignments"]["sha256"]

    features = config["approved_model_features"]
    raw = pd.read_csv(raw_path, low_memory=False)
    test = pd.read_csv(test_path)
    assignments = pd.read_csv(assignments_path)
    predictions = pd.read_csv(predictions_path)

    retained = assignments.loc[
        assignments["status"].eq("included") & assignments["partition"].eq("test")
    ].sort_values("FlowID").reset_index(drop=True)
    assert len(retained) == len(test) == len(predictions)
    assert predictions["partition_row_index"].tolist() == list(range(len(test)))
    assert retained["original_label"].tolist() == test["original_label"].tolist()
    assert retained["binary_label"].astype(int).tolist() == test["binary_label"].astype(int).tolist()
    assert set(retained["SrcAddr"]).isdisjoint(
        {item["source"] for item in config["training_clients"]}
    )

    raw_by_flow = raw.set_index("FlowID", drop=False)
    records = []
    for label, count in COUNTS.items():
        positions = test.index[test["original_label"].eq(label)].tolist()[:count]
        assert len(positions) == count
        for position in positions:
            assignment = retained.iloc[position]
            raw_row = raw_by_flow.loc[int(assignment["FlowID"])]
            model_row = test.iloc[position]
            assert raw_row["label"] == label == model_row["original_label"]
            assert np.allclose(
                raw_row[features].astype(float).to_numpy(),
                model_row[features].astype(float).to_numpy(),
                rtol=1e-12,
                atol=1e-12,
            )
            evidence = predictions.iloc[position]
            records.append(
                {
                    "partition_row_index": int(position),
                    "raw_row": {name: json_value(raw_row[name]) for name in raw.columns},
                    "locked_evidence": {
                        "attack_probability": float(evidence["federated_fedavg_probability"]),
                        "prediction": int(evidence["federated_fedavg_prediction"]),
                    },
                }
            )

    payload = {
        "schema_version": "uavids-locked-test-demo-pool-v1",
        "dataset": "UAVIDS-2025",
        "partition": "locked_test",
        "selection_policy": "first rows per original label in Phase 2 test-partition order; model outcomes not consulted",
        "source_hashes": {
            "raw_sha256": digest(raw_path),
            "test_sha256": digest(test_path),
            "row_assignments_sha256": digest(assignments_path),
        },
        "approved_model_features": features,
        "counts": COUNTS,
        "records": records,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "records": len(records), "sha256": digest(OUTPUT)}))


if __name__ == "__main__":
    main()
