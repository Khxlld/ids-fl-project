"""Orchestration for the Flower simulation: cache shards, run, collect results."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from flwr.simulation import run_simulation

from .client_app import make_client_app
from .config import CONFIG, Config
from .server_app import make_server_app


def cache_partitions(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    partitions: Dict[str, np.ndarray],
    X_val: np.ndarray,
    y_val: np.ndarray,
    client_names,
    cache_dir: Path,
) -> None:
    """Write each client's shard and the validation set to disk as .npy files.

    Partition ids are assigned in the order of `client_names` so partition-id k
    always maps to client_names[k].
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for pid, name in enumerate(client_names):
        idx = partitions[name]
        np.save(cache_dir / f"client_{pid}_X.npy", np.ascontiguousarray(X_pool[idx], dtype=np.float32))
        np.save(cache_dir / f"client_{pid}_y.npy", np.ascontiguousarray(y_pool[idx], dtype=np.float32))
    np.save(cache_dir / "val_X.npy", np.ascontiguousarray(X_val, dtype=np.float32))
    np.save(cache_dir / "val_y.npy", np.ascontiguousarray(y_val, dtype=np.float32))


def run_federated(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    partitions: Dict[str, np.ndarray],
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    cfg: Config = CONFIG,
    cache_dir: Path | None = None,
    client_device: str = "cpu",
    num_cpus_per_client: int = 1,
    num_gpus_per_client: float = 0.0,
    verbose_logging: bool = False,
):
    """Run the full Flower FedAvg simulation and return collected results.

    Returns a dict with round_metrics (DataFrame), client_metrics (DataFrame),
    best (dict) and wall_time (seconds).
    """
    from .config import ARTIFACTS_DIR

    cache_dir = Path(cache_dir) if cache_dir else (ARTIFACTS_DIR / "federated_cache")
    client_names = list(cfg.client_names)

    # 1. Persist shards + validation set so Ray workers can load them.
    cache_partitions(X_pool, y_pool, partitions, X_val, y_val, client_names, cache_dir)

    # 2. Build the client and server apps.
    client_app = make_client_app(cache_dir, input_dim, client_names, cfg, device_str=client_device)
    server_app, records = make_server_app(X_val, y_val, input_dim, cfg, device_str=client_device)

    # 3. Backend resources. One CPU per client keeps five clients running
    #    reliably; GPUs default to 0 to avoid Ray+CUDA contention on the laptop.
    backend_config = {
        "client_resources": {
            "num_cpus": num_cpus_per_client,
            "num_gpus": num_gpus_per_client,
        }
    }

    t0 = time.time()
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=cfg.num_clients,
        backend_config=backend_config,
        verbose_logging=verbose_logging,
    )
    wall_time = time.time() - t0

    round_df = pd.DataFrame(records["rounds"]).sort_values("round").reset_index(drop=True)
    client_df = pd.DataFrame(records["clients"]).sort_values(["round", "partition_id"]).reset_index(drop=True)
    return {
        "round_metrics": round_df,
        "client_metrics": client_df,
        "best": records["best"],
        "wall_time": wall_time,
    }
