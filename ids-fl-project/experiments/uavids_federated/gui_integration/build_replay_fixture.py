"""Build a tiny label-free replay fixture from high-confidence validation rows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from backend import FrozenBinaryIDS, PROJECT_ROOT
from uavids_fl import predict_proba


OUTPUT = Path(__file__).resolve().parent / "examples" / "replay_records.json"


def main() -> None:
    ids = FrozenBinaryIDS(PROJECT_ROOT)
    validation = pd.read_csv(PROJECT_ROOT / "partitions" / "phase2" / "validation.csv")
    finite = np.isfinite(validation[list(ids.features)].to_numpy(dtype=float)).all(axis=1)
    candidates = validation.loc[finite, list(ids.features)].reset_index(drop=True)
    transformed = np.asarray(ids.preprocessor.transform(candidates), dtype=np.float32)
    probability = predict_proba(ids.model, transformed, torch.device("cpu"))
    normal = np.flatnonzero(probability < ids.threshold)[np.argsort(probability[probability < ids.threshold])[:3]]
    attack_pool = np.flatnonzero(probability >= ids.threshold)
    attack = attack_pool[np.argsort(-probability[attack_pool])[:3]]
    selected = [value for pair in zip(normal.tolist(), attack.tolist()) for value in pair]
    records = []
    for index, row_index in enumerate(selected, start=1):
        records.append(
            {
                "record_id": f"replay-{index:02d}",
                "source": "recorded-validation-sample",
                "features": {
                    name: float(candidates.iloc[row_index][name]) for name in ids.features
                },
            }
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    labels = [ids.predict(record)["label"] for record in records]
    assert labels.count("Normal") == 3 and labels.count("Attack") == 3
    print(json.dumps({"written": str(OUTPUT), "records": len(records), "labels": labels}))


if __name__ == "__main__":
    main()
