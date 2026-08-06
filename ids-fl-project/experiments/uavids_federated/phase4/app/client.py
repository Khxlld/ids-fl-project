"""One isolated Phase 4 client: load one partition, poll, train, and upload."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from uavids_fl import BinaryMLP, clone_state_dict, evaluate_model, set_deterministic, train_local_epochs

from .common import (
    ProtocolError,
    build_contract,
    decode_state,
    encode_state,
    json_request,
    read_json,
    sha256_path,
    state_spec,
)


def log(event_type: str, **payload) -> None:
    print(json.dumps({"event_type": event_type, **payload}, sort_keys=True), flush=True)


class DemoClient:
    def __init__(self) -> None:
        self.client_id = os.environ["CLIENT_ID"]
        self.server_url = os.environ.get("SERVER_URL", "http://control-center:8080").rstrip("/")
        self.demo = read_json(os.environ["DEMO_CONFIG_PATH"])
        self.locked = read_json(os.environ["PHASE3_LOCK_PATH"])
        if sha256_path(os.environ["PHASE3_LOCK_PATH"]) != self.demo["phase3_lock_sha256"]:
            raise RuntimeError("Phase 3 lock hash mismatch")
        self.contract = build_contract(self.locked, self.demo)
        records = {item["client_id"]: item for item in self.demo["clients"]}
        if self.client_id not in records:
            raise RuntimeError(f"unknown configured client: {self.client_id}")
        self.record = records[self.client_id]
        self.client_index = list(records).index(self.client_id)

        preprocessor_path = Path(os.environ["PREPROCESSOR_PATH"])
        if sha256_path(preprocessor_path) != self.locked["preprocessor"]["sha256"]:
            raise RuntimeError("preprocessor hash mismatch")
        partition_path = Path(os.environ["TRAINING_PARTITION_PATH"])
        if sha256_path(partition_path) != self.record["partition_sha256"]:
            raise RuntimeError("assigned partition hash mismatch")
        frame = pd.read_csv(partition_path)
        expected_columns = self.locked["features"] + ["original_label", "binary_label"]
        if frame.columns.tolist() != expected_columns:
            raise RuntimeError("assigned partition schema mismatch")
        if len(frame) != int(self.record["samples"]):
            raise RuntimeError("assigned partition sample count mismatch")
        preprocessor = joblib.load(preprocessor_path)
        self.X = np.ascontiguousarray(
            preprocessor.transform(frame[self.locked["features"]]), dtype=np.float32
        )
        self.y = frame["binary_label"].to_numpy(dtype=np.float32)
        if not np.isfinite(self.X).all() or set(np.unique(self.y)) != {0.0, 1.0}:
            raise RuntimeError("assigned partition transformed to invalid values")

        candidate = self.locked["selected_candidate"]
        set_deterministic(int(self.locked["seed"]), 1)
        self.model = BinaryMLP(
            len(self.locked["features"]), candidate["hidden_layers"], candidate["dropout"]
        )
        self.expected_spec = state_spec(self.model.state_dict())
        self.run_id: str | None = None

    def _register(self, payload: dict) -> tuple[dict, int]:
        return json_request(
            "POST", f"{self.server_url}/api/v1/register", payload, timeout=30
        )

    def _request_model(self, completed_round: int) -> tuple[dict, int]:
        return json_request(
            "GET", f"{self.server_url}/api/v1/model?client_id={self.client_id}", timeout=30
        )

    def _submit_update(self, payload: dict) -> tuple[dict, int]:
        return json_request(
            "POST",
            f"{self.server_url}/api/v1/updates",
            payload,
            timeout=float(self.demo["round_timeout_seconds"]) + 30,
        )

    def _post_event_payload(self, payload: dict) -> tuple[dict, int]:
        return json_request(
            "POST", f"{self.server_url}/api/v1/events", payload, timeout=30
        )

    def _on_complete(self, completed_round: int) -> None:
        """Transport-specific completion hook; plain mode needs no action."""

    def post_event(self, event_type: str, server_round: int, payload: dict, severity: str = "info") -> None:
        if self.run_id is None:
            return
        self._post_event_payload(
            {
                "run_id": self.run_id,
                "client_id": self.client_id,
                "event_type": event_type,
                "round": server_round,
                "severity": severity,
                "payload": payload,
            }
        )

    def run(self) -> None:
        registration, _ = self._register(
            {
                "client_id": self.client_id,
                "protocol_version": self.demo["protocol_version"],
                "config_version": self.demo["config_version"],
                "contract_hash": self.contract["contract_hash"],
                "partition_sha256": self.record["partition_sha256"],
                "samples": int(self.record["samples"]),
                "profile": self.record["profile"],
            }
        )
        self.run_id = registration["run_id"]
        self.post_event(
            "client_ready",
            0,
            {
                "profile": self.record["profile"],
                "samples": len(self.y),
                "transformed_shape": list(self.X.shape),
            },
        )
        log("client_ready", client_id=self.client_id, run_id=self.run_id, samples=len(self.y))

        completed_round = 0
        while completed_round < int(self.demo["rounds"]):
            response, _ = self._request_model(completed_round)
            if not response.get("available"):
                status = response.get("status", {})
                if status.get("state") == "failed":
                    raise RuntimeError(status.get("failure", "server reported demo failure"))
                if status.get("state") == "completed":
                    break
                time.sleep(float(self.demo["client_poll_seconds"]))
                continue

            server_round = int(response["round"])
            if server_round <= completed_round:
                time.sleep(float(self.demo["client_poll_seconds"]))
                continue
            if response["run_id"] != self.run_id:
                raise RuntimeError("server run_id changed")
            if response["contract"] != self.contract:
                raise RuntimeError("server model contract is incompatible")
            if response["state_spec"] != self.expected_spec:
                raise RuntimeError("server model state specification is incompatible")
            state, _, received_hash = decode_state(
                response["weights"], self.expected_spec, int(self.demo["maximum_update_bytes"])
            )
            if received_hash != response["weights_sha256"]:
                raise RuntimeError("global model payload hash mismatch")
            self.model.load_state_dict(state)

            training = response["training"]
            self.post_event(
                "client_training_started",
                server_round,
                {
                    "epochs": training["epochs"],
                    "samples": len(self.y),
                    "global_model_sha256": received_hash,
                },
            )
            training_started = time.perf_counter()
            final_loss = train_local_epochs(
                self.model,
                self.X,
                self.y,
                device=torch.device("cpu"),
                batch_size=int(training["batch_size"]),
                epochs=int(training["epochs"]),
                learning_rate=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
                pos_weight=None,
                seed=int(training["seed"]) + server_round * 100 + self.client_index,
            )
            training_ms = (time.perf_counter() - training_started) * 1000
            metrics, _ = evaluate_model(
                self.model,
                self.X,
                self.y.astype(np.int64),
                torch.device("cpu"),
                float(self.contract["decision_threshold"]),
            )
            encoded, update_bytes, update_hash = encode_state(clone_state_dict(self.model))
            client_metrics = {
                "loss": float(final_loss),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "attack_recall": metrics["attack_recall"],
                "fpr": metrics["fpr"],
                "training_ms": round(training_ms, 3),
            }
            self.post_event(
                "client_training_completed",
                server_round,
                {**client_metrics, "update_bytes": update_bytes, "update_sha256": update_hash},
            )

            request_started = time.perf_counter()
            result, _ = self._submit_update(
                {
                    "run_id": self.run_id,
                    "round": server_round,
                    "client_id": self.client_id,
                    "contract_hash": self.contract["contract_hash"],
                    "samples": len(self.y),
                    "weights": encoded,
                    "weights_sha256": update_hash,
                    "metrics": client_metrics,
                }
            )
            request_ack_ms = (time.perf_counter() - request_started) * 1000
            if not result.get("accepted"):
                raise RuntimeError("server did not accept update")
            self.post_event(
                "client_update_acknowledged",
                server_round,
                {
                    "request_ack_ms": round(request_ack_ms, 3),
                    "note": "includes server aggregation/evaluation when this was the final update",
                },
            )
            log(
                "round_update_accepted",
                client_id=self.client_id,
                round=server_round,
                training_ms=round(training_ms, 3),
                update_bytes=update_bytes,
            )
            completed_round = server_round

        self._on_complete(completed_round)
        log("client_complete", client_id=self.client_id, completed_rounds=completed_round)


def main() -> None:
    client: DemoClient | None = None
    try:
        client = DemoClient()
        client.run()
    except Exception as exc:
        log("client_fatal_error", client_id=os.environ.get("CLIENT_ID"), error=type(exc).__name__, detail=str(exc))
        if client is not None and client.run_id is not None:
            try:
                client.post_event(
                    "client_protocol_error",
                    0,
                    {"error": type(exc).__name__, "detail": str(exc)},
                    severity="error",
                )
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
