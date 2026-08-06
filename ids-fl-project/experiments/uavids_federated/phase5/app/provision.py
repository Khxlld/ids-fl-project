"""Generate demo-only ML-DSA identities outside the image and repository."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import oqs

from .security import encode_bytes, public_fingerprint, require_supported


CLIENT_IDS = [f"uav-client-{index}" for index in range(1, 6)]


def _write_secret(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _new_identity(algorithm: str) -> tuple[bytes, bytes]:
    with oqs.Signature(algorithm) as signer:
        public_key = signer.generate_keypair()
        return public_key, signer.export_secret_key()


def provision(output: Path, config: dict) -> dict:
    require_supported(config)
    trust_path = output / "trust_store.json"
    expected_secrets = [output / "server" / "sign_secret.key"] + [
        output / "clients" / client_id / "sign_secret.key" for client_id in CLIENT_IDS
    ]
    if trust_path.is_file() and all(path.is_file() for path in expected_secrets):
        trust = json.loads(trust_path.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "provisioned": True,
                    "reused": True,
                    "identities": 1 + len(CLIENT_IDS),
                    "algorithm": config["signature_algorithm"],
                },
                sort_keys=True,
            )
        )
        return trust
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            "provisioning directory is incomplete; remove only phase5/runtime/keys and retry"
        )

    output.mkdir(parents=True, exist_ok=True)
    server_public, server_secret = _new_identity(config["signature_algorithm"])
    _write_secret(output / "server" / "sign_secret.key", server_secret)
    trust = {
        "schema_version": "uavids-phase5-demo-trust-v1",
        "provisioning_model": "controlled_academic_static_mldsa_identities",
        "signature_algorithm": config["signature_algorithm"],
        "liboqs_version": oqs.oqs_version(),
        "liboqs_python_version": oqs.oqs_python_version(),
        "server": {
            "algorithm": config["signature_algorithm"],
            "fingerprint": public_fingerprint(server_public),
            "public_key": encode_bytes(server_public),
        },
        "clients": {},
    }
    for client_id in CLIENT_IDS:
        public_key, secret_key = _new_identity(config["signature_algorithm"])
        _write_secret(output / "clients" / client_id / "sign_secret.key", secret_key)
        trust["clients"][client_id] = {
            "algorithm": config["signature_algorithm"],
            "fingerprint": public_fingerprint(public_key),
            "public_key": encode_bytes(public_key),
        }
    trust_path.write_text(json.dumps(trust, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "provisioned": True,
                "reused": False,
                "identities": 1 + len(CLIENT_IDS),
                "algorithm": config["signature_algorithm"],
            },
            sort_keys=True,
        )
    )
    return trust


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    provision(args.output.resolve(), config)


if __name__ == "__main__":
    main()
