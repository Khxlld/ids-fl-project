"""Preprocessing pipeline: fit only on the federated training pool.

Numeric features -> median imputation + StandardScaler.
Low-cardinality categorical features -> most-frequent imputation + OneHotEncoder.
High-cardinality / raw-text categorical features are dropped (with a reason)
before encoding so they cannot explode the feature matrix or leak content.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import Config


def _make_ohe() -> OneHotEncoder:
    """OneHotEncoder that tolerates unseen categories at inference time.

    `sparse_output` replaced `sparse` in scikit-learn >= 1.2; we fall back for
    older versions so the code runs across environments.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - very old sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def identify_feature_types(
    X: pd.DataFrame, cfg: Config
) -> Tuple[List[str], List[str], List[dict]]:
    """Split feature columns into numeric vs categorical, dropping high-card ones.

    Returns (numeric_cols, categorical_cols, extra_dropped_report).
    Cardinality is measured on the training pool only.
    """
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    extra_dropped: List[dict] = []

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            numeric_cols.append(col)
            continue
        # Object/categorical column: decide by cardinality.
        n_unique = X[col].nunique(dropna=True)
        if n_unique > cfg.max_categorical_cardinality:
            extra_dropped.append(
                {
                    "column": col,
                    "reason": (
                        f"High-cardinality categorical ({n_unique} unique values "
                        f"> {cfg.max_categorical_cardinality}); likely raw text/identifier"
                    ),
                    "category": "high-cardinality",
                }
            )
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols, extra_dropped


def build_preprocessor(
    numeric_cols: List[str], categorical_cols: List[str]
) -> ColumnTransformer:
    """Assemble the ColumnTransformer (unfitted)."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    transformers = [("num", numeric_pipe, numeric_cols)]

    if categorical_cols:
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _make_ohe()),
            ]
        )
        transformers.append(("cat", categorical_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_feature_names(
    preprocessor: ColumnTransformer,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> List[str]:
    """Recover output feature names after fitting."""
    names = list(numeric_cols)
    if categorical_cols:
        ohe: OneHotEncoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        names.extend(ohe.get_feature_names_out(categorical_cols).tolist())
    return names


def transform_to_float32(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    """Apply a fitted preprocessor and return a dense float32 array."""
    out = preprocessor.transform(X)
    if hasattr(out, "toarray"):  # safety: densify if a sparse matrix slips through
        out = out.toarray()
    return np.ascontiguousarray(out, dtype=np.float32)
