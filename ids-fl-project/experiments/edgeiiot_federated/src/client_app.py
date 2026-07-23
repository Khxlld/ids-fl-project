"""Flower ClientApp: one simulated client == one Factory partition.

Each client trains the shared MLP on ONLY its own data shard and returns updated
weights plus training metrics. In simulation the ClientApp is shipped to Ray
workers, so every client loads its shard from disk by `partition-id` rather than
capturing large arrays in a closure (which would not survive across processes).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from .config import CONFIG, Config
from .model import build_model
from .training import evaluate, get_parameters, make_loader, set_parameters, train_epochs


class FlowerClient(NumPyClient):
    """A single federated client backed by one on-disk data shard."""

    def __init__(self, model, X, y, cfg: Config, device, client_name: str, partition_id: int):
        self.model = model
        self.X = X
        self.y = y
        self.cfg = cfg
        self.device = device
        self.client_name = client_name
        self.partition_id = partition_id

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        """Load global weights, train locally, return updated weights + metrics."""
        set_parameters(self.model, parameters)
        local_epochs = int(config.get("local_epochs", self.cfg.local_epochs))
        loader = make_loader(self.X, self.y, self.cfg.batch_size, shuffle=True)
        train_epochs(
            self.model,
            loader,
            epochs=local_epochs,
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            device=self.device,
        )
        # Report metrics measured on this client's own data.
        m = evaluate(self.model, self.X, self.y, self.device, self.cfg.decision_threshold)
        metrics = {
            "loss": float(m["loss"]),
            "accuracy": float(m["accuracy"]),
            "f1": float(m["f1"]),
            "client_name": self.client_name,
            "partition_id": int(self.partition_id),
        }
        return get_parameters(self.model), len(self.X), metrics

    def evaluate(self, parameters, config):
        """Local evaluation (server-side central eval is the primary signal)."""
        set_parameters(self.model, parameters)
        m = evaluate(self.model, self.X, self.y, self.device, self.cfg.decision_threshold)
        return float(m["loss"]), len(self.X), {"accuracy": float(m["accuracy"]), "f1": float(m["f1"])}


def load_shard(cache_dir: Path, partition_id: int):
    """Load the (X, y) arrays for a given partition id from the cache dir."""
    X = np.load(cache_dir / f"client_{partition_id}_X.npy")
    y = np.load(cache_dir / f"client_{partition_id}_y.npy")
    return X, y


def make_client_app(cache_dir, input_dim: int, client_names, cfg: Config = CONFIG, device_str: str = "cpu") -> ClientApp:
    """Build a ClientApp whose clients load their shard by partition id.

    Clients run on CPU by default: the MLP is tiny, and keeping five concurrent
    simulated clients off the 6 GB laptop GPU avoids Ray+CUDA memory contention.
    """
    cache_dir = Path(cache_dir)
    device = torch.device(device_str)
    client_names = list(client_names)

    def client_fn(context: Context) -> "FlowerClient":
        partition_id = int(context.node_config["partition-id"])
        X, y = load_shard(cache_dir, partition_id)
        model = build_model(input_dim, cfg)
        name = client_names[partition_id] if partition_id < len(client_names) else f"client_{partition_id}"
        return FlowerClient(model, X, y, cfg, device, name, partition_id).to_client()

    return ClientApp(client_fn=client_fn)
