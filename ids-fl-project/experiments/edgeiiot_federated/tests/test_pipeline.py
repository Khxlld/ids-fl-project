"""Fast unit tests for the pipeline building blocks (no full dataset needed).

Run:  py -3.11 -m pytest -q   (from the edgeiiot_federated/ directory)
"""

import numpy as np
import pandas as pd
import torch

from src.config import CONFIG
from src.model import build_model, set_seed
from src.training import get_parameters, set_parameters
from src.partitioning import iid_partition, assert_valid_partitions, partition_distribution
from src import preprocessing as prep
from src.evaluation import compute_metrics


def test_model_output_shape():
    model = build_model(72, CONFIG)
    x = torch.randn(16, 72)
    out = model(x)
    assert out.shape == (16, 1), "model must output one logit per row"


def test_param_roundtrip():
    m1 = build_model(20, CONFIG)
    m2 = build_model(20, CONFIG)
    set_parameters(m2, get_parameters(m1))
    for p1, p2 in zip(m1.state_dict().values(), m2.state_dict().values()):
        assert torch.allclose(p1, p2), "set/get parameters must round-trip exactly"


def test_iid_partition_valid_and_balanced():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=5000)
    parts = iid_partition(y, CONFIG.num_clients, CONFIG.client_names, seed=42)
    assert_valid_partitions(parts, y, CONFIG.num_clients)  # raises if invalid
    dist = partition_distribution(parts, y)
    assert len(dist) == CONFIG.num_clients
    assert dist["total"].sum() == len(y)


def test_preprocessing_float32_no_nan():
    df = pd.DataFrame({
        "num_a": [1.0, 2.0, np.nan, 4.0],
        "num_b": [10.0, np.inf, 30.0, 40.0],
        "cat": ["x", "y", "x", None],
    }).replace([np.inf, -np.inf], np.nan)
    num_cols, cat_cols, dropped = prep.identify_feature_types(df, CONFIG)
    pre = prep.build_preprocessor(num_cols, cat_cols).fit(df[num_cols + cat_cols])
    X = prep.transform_to_float32(pre, df[num_cols + cat_cols])
    assert X.dtype == np.float32
    assert not np.isnan(X).any() and not np.isinf(X).any()


def test_metrics_zero_division_safe():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.15, 0.05])  # everything predicted Normal
    m = compute_metrics(y_true, y_prob, threshold=0.5)
    assert m["precision"] == 0.0  # no positive predictions -> guarded, not NaN
    assert m["tp"] == 0 and m["fn"] == 2
    assert 0.0 <= m["accuracy"] <= 1.0


def test_high_cardinality_categorical_dropped():
    df = pd.DataFrame({
        "num": np.arange(200, dtype=float),
        "id_like": [f"id_{i}" for i in range(200)],  # unique per row
    })
    num_cols, cat_cols, dropped = prep.identify_feature_types(df, CONFIG)
    assert "id_like" not in cat_cols
    assert any(d["column"] == "id_like" for d in dropped)
