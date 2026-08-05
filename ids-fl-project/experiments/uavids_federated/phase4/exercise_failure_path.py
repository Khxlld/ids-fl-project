"""Send one intentionally incompatible update to demonstrate safe rejection."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


BASE = "http://127.0.0.1:8080"


def get_status() -> dict:
    with urllib.request.urlopen(f"{BASE}/api/v1/status", timeout=3) as response:
        return json.loads(response.read())


def main() -> None:
    deadline = time.time() + 120
    while time.time() < deadline:
        status = get_status()
        if status["state"] == "running" and "uav-client-1" in status["registered_clients"]:
            payload = {
                "run_id": status["run_id"],
                "round": status["current_round"],
                "client_id": "uav-client-1",
                "contract_hash": "intentionally-incompatible",
                "samples": 1230,
                "weights": "invalid",
                "weights_sha256": "invalid",
            }
            request = urllib.request.Request(
                f"{BASE}/api/v1/updates",
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                urllib.request.urlopen(request, timeout=5)
            except urllib.error.HTTPError as exc:
                body = json.loads(exc.read())
                assert exc.code == 400 and "contract_hash" in body["error"]
                print(json.dumps({"failure_path_verified": True, "response": body}, indent=2))
                return
            raise RuntimeError("incompatible update was unexpectedly accepted")
        if status["state"] in {"completed", "failed"}:
            break
        time.sleep(0.05)
    raise RuntimeError("no active round was available for failure-path exercise")


if __name__ == "__main__":
    main()
