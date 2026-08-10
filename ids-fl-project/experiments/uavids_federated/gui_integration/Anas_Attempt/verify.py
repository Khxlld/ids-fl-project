"""Offline structural and HTTP smoke checks for the presentation GUI."""

from __future__ import annotations

import json
import threading
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from serve import create_server


ROOT = Path(__file__).resolve().parent
REQUIRED_FILES = {
    "index.html",
    "styles.css",
    "app.js",
    "serve.py",
    "start.ps1",
    "README.md",
    ".env.example",
    "data/replay.json",
}


class AssetParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.assets: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "script" and values.get("src"):
            self.assets.add(values["src"])
        if tag == "link" and values.get("href"):
            self.assets.add(values["href"])


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_files() -> None:
    present = {str(path.relative_to(ROOT)).replace("\\", "/") for path in ROOT.rglob("*") if path.is_file()}
    missing = REQUIRED_FILES - present
    require(not missing, f"missing required files: {sorted(missing)}")
    prohibited_suffixes = {".pt", ".pth", ".joblib", ".pem", ".key", ".csv"}
    prohibited = [path.name for path in ROOT.rglob("*") if path.is_file() and path.suffix.lower() in prohibited_suffixes]
    require(not prohibited, f"prohibited payload files present: {prohibited}")
    large = [path.name for path in ROOT.rglob("*") if path.is_file() and path.stat().st_size > 2 * 1024 * 1024]
    require(not large, f"unexpected file larger than 2 MiB: {large}")


def verify_html() -> None:
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    parser = AssetParser()
    parser.feed(html)
    for asset in parser.assets:
        if asset.startswith(("http://", "https://", "//")):
            raise AssertionError(f"external runtime asset is not allowed: {asset}")
        require((ROOT / asset).is_file(), f"referenced asset does not exist: {asset}")
    for required_text in (
        "IDS Monitor", "Federation", "Security", "Evidence", "Recorded demo",
        "Normal", "Attack", "ML-KEM-768", "ML-DSA-65", "AES-256-GCM",
    ):
        require(required_text in html, f"required UI text missing: {required_text}")


def verify_replay() -> None:
    replay = json.loads((ROOT / "data" / "replay.json").read_text(encoding="utf-8"))
    require(replay["schema_version"] == "anas-attempt-recorded-demo-v1", "unexpected replay schema")
    clients = replay["clients"]
    require([client["client_id"] for client in clients] == [f"uav-client-{i}" for i in range(1, 6)], "client IDs are not the locked five-client set")
    require(sum(client["samples"] for client in clients) == 6148, "client samples do not total 6,148")
    predictions = replay["predictions"]
    require(len(predictions) == 6, "recorded inference set must contain six results")
    require(sum(item["label"] == "Normal" for item in predictions) == 3, "expected three Normal results")
    require(sum(item["label"] == "Attack" for item in predictions) == 3, "expected three Attack results")
    require(all(item["inference_mode"] == "live_model" and item["replayed"] for item in predictions), "recorded results must retain verified live-model/replayed semantics")
    require(all("features" not in item for item in predictions), "prediction evidence must not include feature vectors")

    locked = replay["evidence"]["locked_test"]
    require(locked["rows"] == 53949, "locked-test row count changed")
    require(locked["federated"]["macro_f1"] == 0.9502, "FedAvg locked-test macro-F1 changed")
    require(locked["federated"]["fpr"] == 0.0749, "FedAvg locked-test FPR changed")
    require(locked["local_mean_macro_f1"] == 0.8969, "local-only mean changed")
    require(locked["centralized"]["macro_f1"] == 0.9775, "centralized result changed")

    security = replay["evidence"]["security"]
    require(security["algorithms"] == {
        "kem": "ML-KEM-768",
        "kdf": "HKDF-SHA-256",
        "signature": "ML-DSA-65",
        "aead": "AES-256-GCM",
    }, "security algorithm selection changed")
    require(sum(security["rejection_categories"].values()) == 12, "security rejection categories do not total 12")
    require(security["maximum_plain_secure_difference"] == 0.0, "plain/secure aggregation equivalence changed")

    serialized = json.dumps(replay).lower()
    for forbidden in ("begin private key", '"private_key"', '"shared_secret"', '"ciphertext":', '"features":', "artifacts_phase3"):
        require(forbidden not in serialized, f"sensitive or prohibited replay content found: {forbidden}")


def verify_http() -> None:
    server = create_server(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(f"{base}/", timeout=3) as response:
            require(response.status == 200, "index HTTP response failed")
            require(b"Intrusion Detection Control Room" in response.read(), "index response content is wrong")
        with urllib.request.urlopen(f"{base}/runtime-config.json", timeout=3) as response:
            config = json.loads(response.read())
            require(config["apiBase"].endswith("/api/gui/v1"), "runtime API base is invalid")
        with urllib.request.urlopen(f"{base}/data/replay.json", timeout=3) as response:
            require(json.loads(response.read())["schema_version"] == "anas-attempt-recorded-demo-v1", "replay HTTP response failed")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def main() -> None:
    verify_files()
    verify_html()
    verify_replay()
    verify_http()
    print("Anas_Attempt verification passed: files, safe evidence, locked metrics, algorithms, and HTTP serving.")


if __name__ == "__main__":
    main()
