"""Flower ServerApp: FedAvg + server-side central validation.

The strategy runs in the driver process, so it can safely (a) evaluate the
aggregated global model on the held-out validation set each round, (b) record
per-round and per-client metrics, and (c) checkpoint the best-F1 global model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from .config import CONFIG, GLOBAL_MODEL_PATH, Config
from .model import build_model
from .training import evaluate, get_parameters, set_parameters


def make_server_components(
    val_X: np.ndarray,
    val_y: np.ndarray,
    input_dim: int,
    cfg: Config = CONFIG,
    device_str: str = "cpu",
    model_out_path: Path = GLOBAL_MODEL_PATH,
):
    """Return (build_fn, records) where build_fn(context)->ServerAppComponents.

    `records` is a mutable dict the caller reads after the simulation:
      - records["rounds"]:  list of per-round global validation metrics
      - records["clients"]: list of per-round per-client training metrics
      - records["best"]:    best round summary
    """
    device = torch.device(device_str)
    records: Dict[str, list] = {"rounds": [], "clients": [], "best": {"f1": -1.0, "round": -1}}

    # Round-0 initial parameters from a freshly built model.
    init_model = build_model(input_dim, cfg)
    initial_parameters = ndarrays_to_parameters(get_parameters(init_model))

    def evaluate_fn(server_round: int, parameters: List[np.ndarray], config: dict):
        """Central evaluation of the aggregated model on the validation set."""
        model = build_model(input_dim, cfg)
        set_parameters(model, parameters)
        m = evaluate(model, val_X, val_y, device, cfg.decision_threshold)

        records["rounds"].append(
            {
                "round": server_round,
                "val_loss": m["loss"],
                "val_acc": m["accuracy"],
                "val_precision": m["precision"],
                "val_recall": m["recall"],
                "val_f1": m["f1"],
                "val_roc_auc": m["roc_auc"],
            }
        )

        # Checkpoint the best global model by validation F1 (skip round 0 init).
        if server_round > 0 and m["f1"] > records["best"]["f1"]:
            records["best"] = {"round": server_round, "f1": m["f1"], **m}
            torch.save(model.state_dict(), model_out_path)

        return float(m["loss"]), {
            "val_acc": m["accuracy"],
            "val_f1": m["f1"],
            "val_precision": m["precision"],
            "val_recall": m["recall"],
        }

    def on_fit_config_fn(server_round: int) -> dict:
        return {"server_round": server_round, "local_epochs": cfg.local_epochs}

    def fit_metrics_aggregation_fn(results: List[tuple]) -> dict:
        """Weighted-average client training metrics; also log each client."""
        server_round = len(records["rounds"])  # rounds recorded so far == current
        total = sum(n for n, _ in results)
        agg = {"train_loss": 0.0, "train_acc": 0.0, "train_f1": 0.0}
        for n, m in results:
            w = n / total if total else 0.0
            agg["train_loss"] += w * float(m.get("loss", 0.0))
            agg["train_acc"] += w * float(m.get("accuracy", 0.0))
            agg["train_f1"] += w * float(m.get("f1", 0.0))
            records["clients"].append(
                {
                    "round": server_round,
                    "client_name": m.get("client_name", "?"),
                    "partition_id": int(m.get("partition_id", -1)),
                    "num_examples": int(n),
                    "train_loss": float(m.get("loss", 0.0)),
                    "train_acc": float(m.get("accuracy", 0.0)),
                    "train_f1": float(m.get("f1", 0.0)),
                }
            )
        return agg

    strategy = FedAvg(
        fraction_fit=cfg.fraction_fit,
        fraction_evaluate=cfg.fraction_evaluate,
        min_fit_clients=cfg.num_clients,
        min_available_clients=cfg.num_clients,
        min_evaluate_clients=cfg.num_clients if cfg.fraction_evaluate > 0 else 0,
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=on_fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    )

    def build_fn(context: Context) -> ServerAppComponents:
        # Honour a `flwr run --run-config num-server-rounds=...` override.
        num_rounds = int(context.run_config.get("num-server-rounds", cfg.federated_rounds))
        config = ServerConfig(num_rounds=num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    return build_fn, records


def make_server_app(*args, **kwargs):
    """Convenience wrapper returning (ServerApp, records)."""
    build_fn, records = make_server_components(*args, **kwargs)
    return ServerApp(server_fn=build_fn), records
