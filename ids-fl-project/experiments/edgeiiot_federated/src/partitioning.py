"""IID partitioning of the federated training pool across clients.

IID here means every client receives a random, class-balanced slice of the same
underlying distribution: roughly equal row counts and roughly the same 50/50
Normal/Attack ratio as the pool. We achieve this by splitting each class evenly
across clients, which guarantees no overlap and full coverage by construction.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def iid_partition(y: np.ndarray, num_clients: int, client_names, seed: int) -> Dict[str, np.ndarray]:
    """Return {client_name: row_indices} for an IID, class-balanced split.

    `y` are the training-pool labels (0/1); indices returned are positions into
    that pool (0..len(y)-1).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int).ravel()

    # Split each class's shuffled indices into `num_clients` near-equal chunks.
    per_client_chunks = {name: [] for name in client_names}
    for cls in (0, 1):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        chunks = np.array_split(cls_idx, num_clients)
        for name, chunk in zip(client_names, chunks):
            per_client_chunks[name].append(chunk)

    partitions: Dict[str, np.ndarray] = {}
    for name in client_names:
        idx = np.concatenate(per_client_chunks[name])
        rng.shuffle(idx)  # mix classes within the client
        partitions[name] = idx
    return partitions


def partition_distribution(partitions: Dict[str, np.ndarray], y: np.ndarray) -> pd.DataFrame:
    """Build the per-client Normal/Attack distribution table."""
    y = np.asarray(y).astype(int).ravel()
    rows = []
    for name, idx in partitions.items():
        labels = y[idx]
        normal = int((labels == 0).sum())
        attack = int((labels == 1).sum())
        total = normal + attack
        rows.append(
            {
                "client": name,
                "total": total,
                "normal": normal,
                "attack": attack,
                "normal_pct": round(100 * normal / total, 2) if total else 0.0,
                "attack_pct": round(100 * attack / total, 2) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def assert_valid_partitions(partitions: Dict[str, np.ndarray], y: np.ndarray, num_clients: int) -> None:
    """Sanity checks: exactly N non-empty, disjoint partitions covering the pool."""
    n_total = len(y)
    assert len(partitions) == num_clients, f"expected {num_clients} clients, got {len(partitions)}"

    all_idx = np.concatenate([idx for idx in partitions.values()])
    # No empty partition.
    for name, idx in partitions.items():
        assert len(idx) > 0, f"partition {name} is empty"
    # No overlap.
    assert len(np.unique(all_idx)) == len(all_idx), "partitions overlap (duplicate indices)"
    # Full coverage of the training pool.
    assert len(all_idx) == n_total, f"partitions cover {len(all_idx)} of {n_total} rows"
    assert set(all_idx.tolist()) == set(range(n_total)), "partitions do not cover all pool indices"

    # Class ratios reasonably similar across clients (within 5 percentage points).
    dist = partition_distribution(partitions, y)
    spread = dist["attack_pct"].max() - dist["attack_pct"].min()
    assert spread <= 5.0, f"client attack% spread too large: {spread:.2f} pp"
