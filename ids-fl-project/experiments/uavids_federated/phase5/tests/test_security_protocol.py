from __future__ import annotations

import copy
import json
from pathlib import Path

import oqs
import pytest

from phase5_app.security import (
    ClientSecurityManager,
    SecurityError,
    ServerSecurityManager,
    encode_bytes,
    public_fingerprint,
    sha256_bytes,
    signed_bytes,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "security_config.json"
if not CONFIG_PATH.is_file():
    CONFIG_PATH = Path("/app/phase5_config/security_config.json")
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def identity():
    with oqs.Signature(CONFIG["signature_algorithm"]) as signer:
        public = signer.generate_keypair()
        return public, signer.export_secret_key()


class FakeCoordinator:
    def __init__(self):
        self.run_id = "test-run-id"
        self.demo = {
            "protocol_version": "phase5-pq-secure-fedavg-v1",
            "config_version": "uavids-phase5-secure-demo-v1",
        }
        self.contract = {"contract_hash": "c" * 64}
        self.clients = {
            "uav-client-1": {
                "partition_sha256": "p" * 64,
                "samples": 10,
                "profile": "test-profile",
            },
            "uav-client-2": {
                "partition_sha256": "q" * 64,
                "samples": 12,
                "profile": "test-profile-2",
            },
        }
        self.events = []

    def emit(self, event_type, **kwargs):
        self.events.append((event_type, kwargs))
        return {"seq": len(self.events)}


def registration(client_id="uav-client-1"):
    coordinator = FakeCoordinator()
    record = coordinator.clients[client_id]
    return {
        "client_id": client_id,
        "protocol_version": coordinator.demo["protocol_version"],
        "config_version": coordinator.demo["config_version"],
        "contract_hash": coordinator.contract["contract_hash"],
        "partition_sha256": record["partition_sha256"],
        "samples": record["samples"],
        "profile": record["profile"],
    }


def setup_channel():
    coordinator = FakeCoordinator()
    server_public, server_secret = identity()
    client_public, client_secret = identity()
    other_public, _ = identity()
    trust = {
        "server": {
            "algorithm": CONFIG["signature_algorithm"],
            "fingerprint": public_fingerprint(server_public),
            "public_key": encode_bytes(server_public),
        },
        "clients": {
            "uav-client-1": {
                "algorithm": CONFIG["signature_algorithm"],
                "fingerprint": public_fingerprint(client_public),
                "public_key": encode_bytes(client_public),
            },
            "uav-client-2": {
                "algorithm": CONFIG["signature_algorithm"],
                "fingerprint": public_fingerprint(other_public),
                "public_key": encode_bytes(other_public),
            },
        },
    }
    server = ServerSecurityManager(coordinator, CONFIG, trust, server_secret)
    client = ClientSecurityManager("uav-client-1", CONFIG, trust, client_secret)
    reg = registration()
    hello = server.begin(
        {"registration": reg, "client_identity_fingerprint": trust["clients"]["uav-client-1"]["fingerprint"]}
    )
    request, client_session = client.prepare(hello, reg)
    client_id, returned_registration, confirmation = server.finish(request)
    assert client_id == "uav-client-1"
    assert returned_registration == reg
    client.accept_confirmation(confirmation, client_session)
    return coordinator, server, client, request, hello


def assert_category(expected, callable_):
    with pytest.raises(SecurityError) as caught:
        callable_()
    assert caught.value.category == expected


def test_mldsa_authenticated_mlkem_establishment_and_aesgcm_round_trip():
    _, server, client, _, _ = setup_channel()
    outbound = client.session.seal(
        b"genuine-model-update",
        message_type="client_update",
        server_round=1,
        model_sha256="a" * 64,
    )
    plaintext, header = server.session("uav-client-1").open(
        outbound,
        message_type="client_update",
        server_round=1,
        maximum_plaintext_bytes=1024,
    )
    assert plaintext == b"genuine-model-update"
    assert header["sender"] == "uav-client-1"
    response = server.session("uav-client-1").seal(
        b"global-model",
        message_type="global_model",
        server_round=1,
        model_sha256="b" * 64,
    )
    assert client.session.open(
        response,
        message_type="global_model",
        server_round=1,
        maximum_plaintext_bytes=1024,
    )[0] == b"global-model"


def test_modified_server_hello_signature_is_rejected():
    _, server, client, _, hello = setup_channel()
    tampered = copy.deepcopy(hello)
    raw = bytearray(__import__("base64").b64decode(tampered["signature"]))
    raw[-1] ^= 1
    tampered["signature"] = encode_bytes(bytes(raw))
    assert_category("signature_invalid", lambda: client.prepare(tampered, registration()))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", "wrong-run"),
        ("round", 9),
        ("recipient", "wrong-recipient"),
        ("message_type", "global_model"),
        ("sender", "uav-client-2"),
    ],
)
def test_modified_protected_metadata_or_wrong_identity_is_rejected(field, value):
    _, server, client, _, _ = setup_channel()
    valid = client.session.seal(
        b"update", message_type="client_update", server_round=1, model_sha256="a" * 64
    )
    tampered = copy.deepcopy(valid)
    tampered["header"][field] = value
    assert_category(
        "metadata_mismatch",
        lambda: server.session("uav-client-1").open(
            tampered,
            message_type="client_update",
            server_round=1,
            maximum_plaintext_bytes=1024,
        ),
    )


def test_modified_ciphertext_tag_replay_and_duplicate_nonce_are_rejected():
    _, server, client, _, _ = setup_channel()
    server_session = server.session("uav-client-1")
    first = client.session.seal(
        b"first", message_type="client_update", server_round=1, model_sha256="a" * 64
    )
    tampered = copy.deepcopy(first)
    raw = bytearray(__import__("base64").b64decode(tampered["ciphertext"]))
    raw[-1] ^= 1
    tampered["ciphertext"] = encode_bytes(bytes(raw))
    assert_category(
        "authentication_failed",
        lambda: server_session.open(
            tampered,
            message_type="client_update",
            server_round=1,
            maximum_plaintext_bytes=1024,
        ),
    )
    assert server_session.open(
        first, message_type="client_update", server_round=1, maximum_plaintext_bytes=1024
    )[0] == b"first"
    assert_category(
        "replay",
        lambda: server_session.open(
            first,
            message_type="client_update",
            server_round=1,
            maximum_plaintext_bytes=1024,
        ),
    )

    second = client.session.seal(
        b"second", message_type="client_update", server_round=1, model_sha256="b" * 64
    )
    duplicate_nonce = copy.deepcopy(second)
    duplicate_nonce["header"]["nonce"] = first["header"]["nonce"]
    assert_category(
        "nonce_invalid",
        lambda: server_session.open(
            duplicate_nonce,
            message_type="client_update",
            server_round=1,
            maximum_plaintext_bytes=1024,
        ),
    )
    assert server_session.open(
        second, message_type="client_update", server_round=1, maximum_plaintext_bytes=1024
    )[0] == b"second"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: {**value, "extra": "field"},
        lambda value: {"header": value["header"], "ciphertext": "not-base64!"},
        lambda value: {"header": "not-an-object", "ciphertext": value["ciphertext"]},
    ],
)
def test_malformed_crypto_payloads_are_rejected(mutation):
    _, server, client, _, _ = setup_channel()
    valid = client.session.seal(
        b"update", message_type="client_update", server_round=1, model_sha256="a" * 64
    )
    assert_category(
        "malformed_payload",
        lambda: server.session("uav-client-1").open(
            mutation(copy.deepcopy(valid)),
            message_type="client_update",
            server_round=1,
            maximum_plaintext_bytes=1024,
        ),
    )


def test_modified_signed_handshake_wrong_signature_and_bad_kem_are_rejected():
    coordinator = FakeCoordinator()
    server_public, server_secret = identity()
    client_public, client_secret = identity()
    trust = {
        "server": {
            "algorithm": CONFIG["signature_algorithm"],
            "fingerprint": public_fingerprint(server_public),
            "public_key": encode_bytes(server_public),
        },
        "clients": {
            "uav-client-1": {
                "algorithm": CONFIG["signature_algorithm"],
                "fingerprint": public_fingerprint(client_public),
                "public_key": encode_bytes(client_public),
            }
        },
    }
    server = ServerSecurityManager(coordinator, CONFIG, trust, server_secret)
    client = ClientSecurityManager("uav-client-1", CONFIG, trust, client_secret)
    reg = registration()
    hello = server.begin(
        {"registration": reg, "client_identity_fingerprint": trust["clients"]["uav-client-1"]["fingerprint"]}
    )
    request, _ = client.prepare(hello, reg)

    changed = copy.deepcopy(request)
    changed["handshake"]["run_id"] = "wrong-run"
    assert_category("metadata_mismatch", lambda: server.finish(changed))

    bad_signature = copy.deepcopy(request)
    raw = bytearray(__import__("base64").b64decode(bad_signature["signature"]))
    raw[-1] ^= 1
    bad_signature["signature"] = encode_bytes(bytes(raw))
    assert_category("signature_invalid", lambda: server.finish(bad_signature))

    bad_kem = copy.deepcopy(request)
    ciphertext = b"short"
    bad_kem["handshake"]["kem_ciphertext"] = encode_bytes(ciphertext)
    bad_kem["handshake"]["kem_ciphertext_sha256"] = sha256_bytes(ciphertext)
    bad_kem["signature"] = encode_bytes(
        client._signer.sign(signed_bytes("client_handshake", bad_kem["handshake"]))
    )
    assert_category("malformed_payload", lambda: server.finish(bad_kem))


def test_missing_or_untrusted_public_identity_is_rejected():
    coordinator = FakeCoordinator()
    server_public, server_secret = identity()
    trust = {
        "server": {
            "algorithm": CONFIG["signature_algorithm"],
            "fingerprint": public_fingerprint(server_public),
            "public_key": encode_bytes(server_public),
        },
        "clients": {},
    }
    server = ServerSecurityManager(coordinator, CONFIG, trust, server_secret)
    assert_category(
        "untrusted_identity",
        lambda: server.begin(
            {"registration": registration(), "client_identity_fingerprint": "sha256:missing"}
        ),
    )
