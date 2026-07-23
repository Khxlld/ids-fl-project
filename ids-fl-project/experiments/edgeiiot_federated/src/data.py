"""Dataset loading, inspection, cleaning, balanced sampling and splitting.

The DNN-EdgeIIoT dataset is a lab-generated capture: specific attacks were run
from/against specific hosts, ports and times. Columns that encode those
identifiers (IP addresses, timestamps, ports, raw payloads/URIs/messages)
therefore correlate almost perfectly with the label *because of how the lab was
set up*, which is target leakage rather than a generalisable signal. We drop
them deliberately and record the reason for each.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config, TARGET_COLUMN, MULTICLASS_COLUMN


# Columns that identify the lab setup rather than describe traffic behaviour.
# Each entry maps column -> (reason, category) used to build the drop report.
# We only drop a column from this list if it is actually present in the file.
LEAKAGE_IDENTIFIER_COLUMNS: Dict[str, Tuple[str, str]] = {
    "frame.time": ("Packet capture timestamp; encodes when each attack was run", "identifier"),
    "ip.src_host": ("Source IP address; identifies the lab host, not behaviour", "identifier"),
    "ip.dst_host": ("Destination IP address; identifies the lab victim", "identifier"),
    "arp.src.proto_ipv4": ("Source IPv4 in ARP; host identifier", "identifier"),
    "arp.dst.proto_ipv4": ("Destination IPv4 in ARP; host identifier", "identifier"),
    "icmp.transmit_timestamp": ("ICMP timestamp; encodes capture time", "identifier"),
    "tcp.payload": ("Raw TCP payload bytes; leaks attack content verbatim", "leakage-prone"),
    "mqtt.msg": ("Raw MQTT message payload; leaks attack content", "leakage-prone"),
    "http.file_data": ("Raw HTTP body; leaks attack content", "leakage-prone"),
    "http.request.full_uri": ("Full request URI string; leaks attack content", "leakage-prone"),
    "http.request.uri.query": ("URI query string; leaks attack content", "leakage-prone"),
    "http.referer": ("HTTP referer string; leaks attack content", "leakage-prone"),
    "tcp.options": ("Raw TCP options hex string; high-cardinality identifier", "identifier"),
    "tcp.srcport": ("Source port; tied to the specific lab attack setup", "leakage-prone"),
    "tcp.dstport": ("Destination service port; tied to the lab attack target", "leakage-prone"),
    "udp.port": ("UDP port; tied to the specific lab attack setup", "leakage-prone"),
}


def load_dataset(path, nrows: int | None = None) -> pd.DataFrame:
    """Load the raw CSV. `nrows` limits rows for quick smoke tests."""
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Compute the schema/quality report requested for the raw dataset."""
    numeric = df.select_dtypes(include=[np.number])
    inf_counts = np.isinf(numeric.to_numpy(dtype="float64", na_value=np.nan)).sum()
    label_counts = (
        df[TARGET_COLUMN].value_counts(dropna=False).to_dict()
        if TARGET_COLUMN in df.columns
        else {}
    )
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 1),
        "missing_per_column": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "total_missing": int(df.isna().sum().sum()),
        "total_infinities": int(inf_counts),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
    }


def clean_dataset(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Clean the full dataset and return (cleaned_df, dropped_columns_report).

    Steps: drop duplicates, replace +/-inf with NaN, drop the multiclass target,
    drop leakage/identifier columns that are present, and drop constant columns.
    Remaining missing values are handled later inside the fitted preprocessor so
    that imputation statistics come only from the training pool.
    """
    report_rows: List[dict] = []

    # 1. Remove duplicate rows *before* splitting so identical rows cannot land
    #    in both train and test.
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dupes = n_before - len(df)

    # 2. Replace +/- infinity with NaN across numeric columns.
    df = df.replace([np.inf, -np.inf], np.nan)

    # 3. Drop the multiclass label (would leak the binary answer).
    if MULTICLASS_COLUMN in df.columns:
        report_rows.append(
            {
                "column": MULTICLASS_COLUMN,
                "reason": "Detailed attack category; directly reveals the binary label",
                "category": "target-leakage",
            }
        )
        df = df.drop(columns=[MULTICLASS_COLUMN])

    # 4. Drop known identifier / leakage-prone columns that are present.
    for col, (reason, category) in LEAKAGE_IDENTIFIER_COLUMNS.items():
        if col in df.columns:
            report_rows.append({"column": col, "reason": reason, "category": category})
            df = df.drop(columns=[col])

    # 5. Drop constant columns (a single value carries no signal).
    for col in list(df.columns):
        if col == TARGET_COLUMN:
            continue
        if df[col].nunique(dropna=False) <= 1:
            report_rows.append(
                {
                    "column": col,
                    "reason": "Constant column (<=1 unique value)",
                    "category": "constant",
                }
            )
            df = df.drop(columns=[col])

    dropped_report = pd.DataFrame(report_rows, columns=["column", "reason", "category"])
    # Attach the dedup count as metadata via the DataFrame attrs.
    df.attrs["n_duplicate_rows_removed"] = int(n_dupes)
    return df, dropped_report


def balanced_sample(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Draw a balanced ~`dataset_fraction` subset (50/50 Normal/Attack).

    If a class has too few rows for a perfectly balanced target, the largest
    feasible equal count is used and the resulting size is reported by the
    caller via `print` of the returned counts.
    """
    rng = cfg.random_seed
    target_total = int(round(len(df) * cfg.dataset_fraction))
    per_class = target_total // 2

    normal = df[df[TARGET_COLUMN] == 0]
    attack = df[df[TARGET_COLUMN] == 1]

    # Cap by the smaller class so both classes can supply `n_each` rows.
    n_each = min(per_class, len(normal), len(attack))

    normal_s = normal.sample(n=n_each, random_state=rng)
    attack_s = attack.sample(n=n_each, random_state=rng)

    sample = (
        pd.concat([normal_s, attack_s])
        .sample(frac=1.0, random_state=rng)  # shuffle
        .reset_index(drop=True)
    )
    return sample


def split_data(
    df: pd.DataFrame, cfg: Config
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 split into (train_pool, validation, test)."""
    y = df[TARGET_COLUMN]

    # First carve off the test set (test_fraction of the whole).
    train_val, test = train_test_split(
        df,
        test_size=cfg.test_fraction,
        stratify=y,
        random_state=cfg.random_seed,
    )
    # Then split the remainder into train pool and validation.
    val_ratio = cfg.validation_fraction / (1.0 - cfg.test_fraction)
    train_pool, val = train_test_split(
        train_val,
        test_size=val_ratio,
        stratify=train_val[TARGET_COLUMN],
        random_state=cfg.random_seed,
    )
    # Keep the (unique) sample-row indices so callers can prove the three splits
    # are disjoint by index. `balanced_sample` already reset them to 0..N-1.
    return train_pool, val, test


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Separate feature columns from the binary target."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].to_numpy(dtype="float32")
    return X, y
