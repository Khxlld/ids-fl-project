"""Inference: load the saved preprocessor + global model and score raw rows.

Usage as a script:
    python -m src.inference --csv path/to/rows.csv
    python -m src.inference            # demo on a few held-out test rows
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import torch

from .config import (
    CLASS_NAMES,
    CONFIG,
    FEATURE_NAMES_PATH,
    GLOBAL_MODEL_PATH,
    PREPROCESSOR_PATH,
)
from .model import build_model
from .preprocessing import transform_to_float32


class InferencePipeline:
    """Wraps the fitted preprocessor and a trained model for scoring raw rows."""

    def __init__(self, model_path: Path = GLOBAL_MODEL_PATH, threshold: float | None = None):
        self.preprocessor = joblib.load(PREPROCESSOR_PATH)
        with open(FEATURE_NAMES_PATH) as f:
            meta = json.load(f)
        self.feature_columns: List[str] = meta["input_columns"]
        input_dim = meta["n_output_features"]
        self.threshold = CONFIG.decision_threshold if threshold is None else threshold

        self.model = build_model(input_dim, CONFIG)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self.model.eval()

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a DataFrame of raw rows (extra columns are ignored)."""
        # Keep only the columns the preprocessor was fitted on, in order.
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing[:10]}...")
        X = self.preprocessor.transform(df[self.feature_columns])
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.ascontiguousarray(X, dtype=np.float32)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(torch.from_numpy(X))).numpy().ravel()
        preds = (probs >= self.threshold).astype(int)
        return pd.DataFrame(
            {
                "attack_probability": probs,
                "predicted_label": preds,
                "predicted_class": [CLASS_NAMES[int(p)] for p in preds],
                "threshold": self.threshold,
            }
        )


def _demo() -> None:
    """Score a handful of untouched test rows and print the result."""
    from . import config

    pipe = InferencePipeline()
    test_path = config.ARTIFACTS_DIR / "test_raw.csv"
    if not test_path.exists():
        print(f"No demo test rows at {test_path}. Run the notebook first.")
        return
    df = pd.read_csv(test_path).head(8)
    out = pipe.predict(df)
    for i, row in out.iterrows():
        true = df.iloc[i].get("Attack_label", "?")
        print(
            f"row {i}: prob={row['attack_probability']:.4f} "
            f"pred={int(row['predicted_label'])} ({row['predicted_class']}) "
            f"threshold={row['threshold']:.2f}  [true={true}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge-IIoT federated IDS inference")
    parser.add_argument("--csv", type=str, default=None, help="CSV of raw rows to score")
    args = parser.parse_args()
    if args.csv:
        pipe = InferencePipeline()
        df = pd.read_csv(args.csv)
        print(pipe.predict(df).to_string())
    else:
        _demo()


if __name__ == "__main__":
    main()
