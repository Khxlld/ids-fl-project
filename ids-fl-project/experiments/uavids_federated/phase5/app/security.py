"""Post-quantum identity, key establishment, AEAD, and replay protection.

Only public identifiers, hashes, byte counts, and timings leave this module.
Private keys, KEM shared secrets, derived keys, and plaintext model bytes are
never logged or written as runtime evidence.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import oqs
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


SIGNING_DOMAIN = b"UAVIDS-PHASE5-SIGNED-V1\x00"
KDF_DOMAIN = b"UAVIDS-PHASE5-SESSION-V1\x00"
SERVER_ID = "control-center"


class SecurityError(ValueError):
    """A cryptographic/protocol failure safe to report by category."""

    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise SecurityError("malformed_payload", "value is not canonical JSON data") from exc


def signed_bytes(message_type: str, value: Mapping[str, Any]) -> bytes:
    return SIGNING_DOMAIN + message_type.encode("ascii") + b"\x00" + canonical_bytes(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def encode_bytes(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def decode_bytes(value: Any, field_name: str, maximum_bytes: int) -> bytes:
    if not isinstance(value, str):
        raise SecurityError("malformed_payload", f"{field_name} must be base64 text")
    if len(value) > ((maximum_bytes + 2) // 3) * 4 + 8:
        raise SecurityError("malformed_payload", f"{field_name} exceeds its size limit")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise SecurityError("malformed_payload", f"{field_name} is not valid base64") from exc
    if len(decoded) > maximum_bytes:
        raise SecurityError("malformed_payload", f"{field_name} exceeds its size limit")
    return decoded


def public_fingerprint(public_key: bytes) -> str:
    return f"sha256:{sha256_bytes(public_key)}"


def require_supported(config: Mapping[str, Any]) -> None:
    kem = config["kem_algorithm"]
    signature = config["signature_algorithm"]
    if kem not in oqs.get_enabled_kem_mechanisms():
        raise RuntimeError(f"configured KEM is not enabled by liboqs: {kem}")
    if signature not in oqs.get_enabled_sig_mechanisms():
        raise RuntimeError(f"configured signature is not enabled by liboqs: {signature}")
    if config["kdf"] != "HKDF-SHA-256" or config["aead"] != "AES-256-GCM":
        raise RuntimeError("unsupported Phase 5 KDF or AEAD configuration")


def _derive_direction(shared_secret: bytes, salt: bytes, info: bytes) -> tuple[bytes, bytes]:
    material = HKDF(
        algorithm=hashes.SHA256(), length=36, salt=salt, info=KDF_DOMAIN + info
    ).derive(shared_secret)
    return material[:32], material[32:]


def derive_session_material(shared_secret: bytes, context: Mapping[str, Any]) -> dict[str, bytes]:
    context_bytes = canonical_bytes(context)
    salt = hashlib.sha256(KDF_DOMAIN + context_bytes).digest()
    client_to_server_key, client_to_server_prefix = _derive_direction(
        shared_secret, salt, b"client-to-server\x00" + context_bytes
    )
    server_to_client_key, server_to_client_prefix = _derive_direction(
        shared_secret, salt, b"server-to-client\x00" + context_bytes
    )
    return {
        "client_to_server_key": client_to_server_key,
        "client_to_server_prefix": client_to_server_prefix,
        "server_to_client_key": server_to_client_key,
        "server_to_client_prefix": server_to_client_prefix,
    }


def session_context(hello: Mapping[str, Any], handshake: Mapping[str, Any]) -> dict:
    return {
        "security_version": hello["security_version"],
        "run_id": hello["run_id"],
        "session_id": hello["session_id"],
        "client_id": hello["client_id"],
        "server_id": hello["server_id"],
        "contract_hash": hello["contract_hash"],
        "kem_algorithm": hello["kem_algorithm"],
        "signature_algorithm": hello["signature_algorithm"],
        "kdf": hello["kdf"],
        "aead": hello["aead"],
        "server_nonce": hello["server_nonce"],
        "client_nonce": handshake["client_nonce"],
        "kem_public_key_sha256": hello["kem_public_key_sha256"],
        "kem_ciphertext_sha256": handshake["kem_ciphertext_sha256"],
        "client_identity_fingerprint": handshake["client_identity_fingerprint"],
        "server_identity_fingerprint": hello["server_identity_fingerprint"],
    }


@dataclass
class SecureSession:
    security_version: str
    run_id: str
    session_id: str
    local_id: str
    peer_id: str
    contract_hash: str
    send_key: bytes = field(repr=False)
    receive_key: bytes = field(repr=False)
    send_nonce_prefix: bytes = field(repr=False)
    receive_nonce_prefix: bytes = field(repr=False)
    send_sequence: int = 0
    receive_sequence: int = 0
    received_nonces: set[bytes] = field(default_factory=set, repr=False)
    timings: dict[str, list[float]] = field(default_factory=dict, repr=False)
    sizes: dict[str, list[int]] = field(default_factory=dict, repr=False)

    def _record(self, name: str, elapsed_ms: float, size_name: str | None = None, size: int = 0) -> None:
        self.timings.setdefault(name, []).append(round(elapsed_ms, 6))
        if size_name:
            self.sizes.setdefault(size_name, []).append(int(size))

    def _nonce(self, prefix: bytes, sequence: int) -> bytes:
        if len(prefix) != 4 or sequence <= 0 or sequence >= 2**64:
            raise SecurityError("nonce_invalid", "nonce sequence is outside the allowed range")
        return prefix + sequence.to_bytes(8, "big")

    def seal(
        self,
        plaintext: bytes,
        *,
        message_type: str,
        server_round: int,
        model_sha256: str = "",
    ) -> dict:
        sequence = self.send_sequence + 1
        nonce = self._nonce(self.send_nonce_prefix, sequence)
        header = {
            "security_version": self.security_version,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "round": int(server_round),
            "sender": self.local_id,
            "recipient": self.peer_id,
            "message_type": message_type,
            "sequence": sequence,
            "nonce": encode_bytes(nonce),
            "contract_hash": self.contract_hash,
            "plaintext_sha256": sha256_bytes(plaintext),
            "model_sha256": str(model_sha256),
        }
        aad = canonical_bytes(header)
        started = time.perf_counter()
        ciphertext = AESGCM(self.send_key).encrypt(nonce, plaintext, aad)
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.send_sequence = sequence
        envelope = {"header": header, "ciphertext": encode_bytes(ciphertext)}
        protected_bytes = len(canonical_bytes(envelope))
        self._record("encrypt_ms", elapsed_ms, "protected_bytes", protected_bytes)
        self.sizes.setdefault("plaintext_bytes", []).append(len(plaintext))
        return envelope

    def open(
        self,
        envelope: Any,
        *,
        message_type: str,
        server_round: int | None = None,
        maximum_plaintext_bytes: int,
    ) -> tuple[bytes, dict]:
        if not isinstance(envelope, dict) or set(envelope) != {"header", "ciphertext"}:
            raise SecurityError("malformed_payload", "secure envelope fields are invalid")
        header = envelope.get("header")
        if not isinstance(header, dict):
            raise SecurityError("malformed_payload", "secure envelope header is invalid")
        required = {
            "security_version", "session_id", "run_id", "round", "sender", "recipient",
            "message_type", "sequence", "nonce", "contract_hash", "plaintext_sha256",
            "model_sha256",
        }
        if set(header) != required:
            raise SecurityError("malformed_payload", "secure envelope header fields are invalid")
        expected = {
            "security_version": self.security_version,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "sender": self.peer_id,
            "recipient": self.local_id,
            "message_type": message_type,
            "contract_hash": self.contract_hash,
        }
        for name, value in expected.items():
            if header.get(name) != value:
                raise SecurityError("metadata_mismatch", f"protected {name} is incorrect")
        if server_round is not None and header.get("round") != int(server_round):
            raise SecurityError("metadata_mismatch", "protected round is incorrect")
        sequence = header.get("sequence")
        if not isinstance(sequence, int):
            raise SecurityError("malformed_payload", "protected sequence is invalid")
        if sequence <= self.receive_sequence:
            raise SecurityError("replay", "message sequence is duplicated or stale")
        if sequence != self.receive_sequence + 1:
            raise SecurityError("sequence_gap", "message sequence is not the next expected value")
        nonce = decode_bytes(header.get("nonce"), "nonce", 12)
        if len(nonce) != 12 or nonce != self._nonce(self.receive_nonce_prefix, sequence):
            raise SecurityError("nonce_invalid", "message nonce is not the expected unique nonce")
        if nonce in self.received_nonces:
            raise SecurityError("replay", "message nonce was already accepted")
        maximum_ciphertext = maximum_plaintext_bytes + 16
        ciphertext = decode_bytes(envelope.get("ciphertext"), "ciphertext", maximum_ciphertext)
        aad = canonical_bytes(header)
        started = time.perf_counter()
        try:
            plaintext = AESGCM(self.receive_key).decrypt(nonce, ciphertext, aad)
        except InvalidTag as exc:
            raise SecurityError("authentication_failed", "ciphertext authentication failed") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        if len(plaintext) > maximum_plaintext_bytes:
            raise SecurityError("malformed_payload", "decrypted payload exceeds its size limit")
        if header.get("plaintext_sha256") != sha256_bytes(plaintext):
            raise SecurityError("authentication_failed", "decrypted payload hash is inconsistent")
        self.receive_sequence = sequence
        self.received_nonces.add(nonce)
        self._record("decrypt_ms", elapsed_ms, "received_protected_bytes", len(canonical_bytes(envelope)))
        self.sizes.setdefault("received_plaintext_bytes", []).append(len(plaintext))
        return plaintext, header

    def seal_json(self, value: Mapping[str, Any], **kwargs) -> dict:
        return self.seal(canonical_bytes(value), **kwargs)

    def open_json(self, envelope: Any, **kwargs) -> tuple[dict, dict]:
        raw, header = self.open(envelope, **kwargs)
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SecurityError("malformed_payload", "decrypted payload is not valid JSON") from exc
        if not isinstance(value, dict):
            raise SecurityError("malformed_payload", "decrypted JSON must be an object")
        return value, header

    def safe_summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "sent_messages": self.send_sequence,
            "received_messages": self.receive_sequence,
            "timings_ms": self.timings,
            "sizes_bytes": self.sizes,
        }


def make_session(
    *,
    role: str,
    config: Mapping[str, Any],
    shared_secret: bytes,
    context: Mapping[str, Any],
) -> SecureSession:
    material = derive_session_material(shared_secret, context)
    if role == "client":
        local_id, peer_id = context["client_id"], context["server_id"]
        send_key, receive_key = material["client_to_server_key"], material["server_to_client_key"]
        send_prefix = material["client_to_server_prefix"]
        receive_prefix = material["server_to_client_prefix"]
    elif role == "server":
        local_id, peer_id = context["server_id"], context["client_id"]
        send_key, receive_key = material["server_to_client_key"], material["client_to_server_key"]
        send_prefix = material["server_to_client_prefix"]
        receive_prefix = material["client_to_server_prefix"]
    else:
        raise ValueError("role must be client or server")
    return SecureSession(
        security_version=config["security_version"],
        run_id=context["run_id"],
        session_id=context["session_id"],
        local_id=local_id,
        peer_id=peer_id,
        contract_hash=context["contract_hash"],
        send_key=send_key,
        receive_key=receive_key,
        send_nonce_prefix=send_prefix,
        receive_nonce_prefix=receive_prefix,
    )


class ServerSecurityManager:
    """Own the server identity, ephemeral KEM key, challenges, and sessions."""

    def __init__(self, coordinator, config: dict, trust_store: dict, secret_key: bytes):
        require_supported(config)
        self.coordinator = coordinator
        self.config = config
        self.trust_store = trust_store
        self._signer = oqs.Signature(config["signature_algorithm"], secret_key)
        self._kem = oqs.KeyEncapsulation(config["kem_algorithm"])
        self.kem_public_key = self._kem.generate_keypair()
        self._lock = threading.RLock()
        self.challenges: dict[str, dict] = {}
        self.sessions: dict[str, SecureSession] = {}
        self.server_metrics: dict[str, list[float | int]] = {}
        expected_server = trust_store["server"]
        if expected_server["algorithm"] != config["signature_algorithm"]:
            raise RuntimeError("server trust record algorithm mismatch")
        self.server_fingerprint = expected_server["fingerprint"]

    def _record(self, name: str, value: float | int) -> None:
        self.server_metrics.setdefault(name, []).append(round(value, 6) if isinstance(value, float) else value)

    def _validate_registration(self, registration: Any) -> tuple[str, dict]:
        if not isinstance(registration, dict):
            raise SecurityError("malformed_payload", "registration is invalid")
        client_id = registration.get("client_id")
        if client_id not in self.coordinator.clients:
            raise SecurityError("untrusted_identity", "client identity is not approved")
        expected_client = self.coordinator.clients[client_id]
        expected = {
            "client_id": client_id,
            "protocol_version": self.coordinator.demo["protocol_version"],
            "config_version": self.coordinator.demo["config_version"],
            "contract_hash": self.coordinator.contract["contract_hash"],
            "partition_sha256": expected_client["partition_sha256"],
            "samples": expected_client["samples"],
            "profile": expected_client["profile"],
        }
        if registration != expected:
            raise SecurityError("metadata_mismatch", "registration does not match the approved client contract")
        return client_id, expected

    def begin(self, request: dict) -> dict:
        registration = request.get("registration")
        client_id, registration = self._validate_registration(registration)
        identity = self.trust_store["clients"].get(client_id)
        if identity is None:
            raise SecurityError("untrusted_identity", "client public identity is missing")
        if request.get("client_identity_fingerprint") != identity["fingerprint"]:
            raise SecurityError("identity_mismatch", "client identity fingerprint is incorrect")
        session_id = str(uuid.uuid4())
        hello = {
            "security_version": self.config["security_version"],
            "run_id": self.coordinator.run_id,
            "session_id": session_id,
            "client_id": client_id,
            "server_id": SERVER_ID,
            "contract_hash": self.coordinator.contract["contract_hash"],
            "kem_algorithm": self.config["kem_algorithm"],
            "signature_algorithm": self.config["signature_algorithm"],
            "kdf": self.config["kdf"],
            "aead": self.config["aead"],
            "server_nonce": encode_bytes(secrets.token_bytes(32)),
            "kem_public_key": encode_bytes(self.kem_public_key),
            "kem_public_key_sha256": sha256_bytes(self.kem_public_key),
            "server_identity_fingerprint": self.server_fingerprint,
            "expires_after_seconds": int(self.config["handshake_ttl_seconds"]),
        }
        started = time.perf_counter()
        with self._lock:
            signature = self._signer.sign(signed_bytes("server_hello", hello))
        self._record("sign_ms", (time.perf_counter() - started) * 1000)
        self._record("server_hello_bytes", len(canonical_bytes(hello)) + len(signature))
        self.challenges[session_id] = {
            "hello": hello,
            "registration": registration,
            "created": time.monotonic(),
            "used": False,
        }
        self.coordinator.emit(
            "security_handshake_started",
            client_id=client_id,
            payload={
                "kem": self.config["kem_algorithm"],
                "signature": self.config["signature_algorithm"],
                "server_identity": self.server_fingerprint,
            },
        )
        return {"hello": hello, "signature": encode_bytes(signature)}

    def finish(self, request: dict) -> tuple[str, dict, dict]:
        handshake = request.get("handshake")
        if not isinstance(handshake, dict):
            raise SecurityError("malformed_payload", "signed handshake is invalid")
        session_id = handshake.get("session_id")
        challenge = self.challenges.get(session_id)
        if challenge is None:
            raise SecurityError("stale_handshake", "handshake challenge is unknown or expired")
        if challenge["used"]:
            raise SecurityError("replay", "handshake challenge was already used")
        if time.monotonic() - challenge["created"] > int(self.config["handshake_ttl_seconds"]):
            raise SecurityError("stale_handshake", "handshake challenge expired")
        hello = challenge["hello"]
        client_id = hello["client_id"]
        expected_fields = {
            "security_version": hello["security_version"],
            "run_id": hello["run_id"],
            "session_id": hello["session_id"],
            "client_id": client_id,
            "server_id": hello["server_id"],
            "contract_hash": hello["contract_hash"],
            "kem_algorithm": hello["kem_algorithm"],
            "signature_algorithm": hello["signature_algorithm"],
            "kdf": hello["kdf"],
            "aead": hello["aead"],
            "server_nonce": hello["server_nonce"],
            "kem_public_key_sha256": hello["kem_public_key_sha256"],
            "client_identity_fingerprint": self.trust_store["clients"][client_id]["fingerprint"],
        }
        for name, value in expected_fields.items():
            if handshake.get(name) != value:
                raise SecurityError("metadata_mismatch", f"signed handshake {name} is incorrect")
        if set(handshake) != {*expected_fields, "client_nonce", "kem_ciphertext", "kem_ciphertext_sha256", "registration"}:
            raise SecurityError("malformed_payload", "signed handshake fields are invalid")
        if handshake["registration"] != challenge["registration"]:
            raise SecurityError("metadata_mismatch", "signed registration changed during handshake")
        identity = self.trust_store["clients"][client_id]
        public_key = decode_bytes(identity["public_key"], "trusted client public key", 8192)
        signature = decode_bytes(request.get("signature"), "signature", 8192)
        started = time.perf_counter()
        with oqs.Signature(self.config["signature_algorithm"]) as verifier:
            verified = verifier.verify(signed_bytes("client_handshake", handshake), signature, public_key)
        self._record("verify_ms", (time.perf_counter() - started) * 1000)
        if not verified:
            raise SecurityError("signature_invalid", "client handshake signature is invalid")
        ciphertext = decode_bytes(
            handshake["kem_ciphertext"], "KEM ciphertext", int(self._kem.details["length_ciphertext"])
        )
        if len(ciphertext) != int(self._kem.details["length_ciphertext"]):
            raise SecurityError("malformed_payload", "KEM ciphertext length is invalid")
        if sha256_bytes(ciphertext) != handshake["kem_ciphertext_sha256"]:
            raise SecurityError("metadata_mismatch", "KEM ciphertext hash is incorrect")
        started = time.perf_counter()
        with self._lock:
            shared_secret = self._kem.decap_secret(ciphertext)
        self._record("decapsulation_ms", (time.perf_counter() - started) * 1000)
        context = session_context(hello, handshake)
        session = make_session(
            role="server", config=self.config, shared_secret=shared_secret, context=context
        )
        challenge["used"] = True
        self.sessions[client_id] = session
        confirmation = session.seal_json(
            {"accepted": True, "run_id": self.coordinator.run_id, "client_id": client_id},
            message_type="session_confirmation",
            server_round=0,
        )
        self.coordinator.emit(
            "client_authenticated",
            client_id=client_id,
            payload={
                "identity": identity["fingerprint"],
                "kem": self.config["kem_algorithm"],
                "signature": self.config["signature_algorithm"],
                "aead": self.config["aead"],
            },
        )
        return client_id, challenge["registration"], confirmation

    def session(self, client_id: str) -> SecureSession:
        try:
            return self.sessions[client_id]
        except KeyError as exc:
            raise SecurityError("unauthenticated", "client has no authenticated secure session") from exc

    def safe_summary(self) -> dict:
        return {
            "security_version": self.config["security_version"],
            "algorithms": {
                "kem": self.config["kem_algorithm"],
                "signature": self.config["signature_algorithm"],
                "kdf": self.config["kdf"],
                "aead": self.config["aead"],
            },
            "authenticated_clients": sorted(self.sessions),
            "server_timings_ms": self.server_metrics,
            "sessions": {client_id: session.safe_summary() for client_id, session in self.sessions.items()},
        }


class ClientSecurityManager:
    """Own one client's provisioned identity and established secure session."""

    def __init__(self, client_id: str, config: dict, trust_store: dict, secret_key: bytes):
        require_supported(config)
        self.client_id = client_id
        self.config = config
        self.trust_store = trust_store
        self._signer = oqs.Signature(config["signature_algorithm"], secret_key)
        identity = trust_store["clients"].get(client_id)
        if identity is None:
            raise RuntimeError("client identity is absent from the trust store")
        self.identity = identity
        self.session: SecureSession | None = None
        self.handshake_timings: dict[str, float] = {}

    def prepare(self, response: dict, registration: dict) -> tuple[dict, SecureSession]:
        hello = response.get("hello")
        if not isinstance(hello, dict):
            raise SecurityError("malformed_payload", "server hello is invalid")
        expected = {
            "security_version": self.config["security_version"],
            "client_id": self.client_id,
            "server_id": SERVER_ID,
            "contract_hash": registration["contract_hash"],
            "kem_algorithm": self.config["kem_algorithm"],
            "signature_algorithm": self.config["signature_algorithm"],
            "kdf": self.config["kdf"],
            "aead": self.config["aead"],
            "server_identity_fingerprint": self.trust_store["server"]["fingerprint"],
        }
        for name, value in expected.items():
            if hello.get(name) != value:
                raise SecurityError("metadata_mismatch", f"server hello {name} is incorrect")
        server_public = decode_bytes(
            self.trust_store["server"]["public_key"], "trusted server public key", 8192
        )
        signature = decode_bytes(response.get("signature"), "server signature", 8192)
        started = time.perf_counter()
        with oqs.Signature(self.config["signature_algorithm"]) as verifier:
            verified = verifier.verify(signed_bytes("server_hello", hello), signature, server_public)
        self.handshake_timings["server_signature_verification_ms"] = round(
            (time.perf_counter() - started) * 1000, 6
        )
        if not verified:
            raise SecurityError("signature_invalid", "server hello signature is invalid")
        kem_public = decode_bytes(hello.get("kem_public_key"), "server KEM public key", 8192)
        if sha256_bytes(kem_public) != hello.get("kem_public_key_sha256"):
            raise SecurityError("metadata_mismatch", "server KEM public key hash is incorrect")
        started = time.perf_counter()
        with oqs.KeyEncapsulation(self.config["kem_algorithm"]) as encapsulator:
            ciphertext, shared_secret = encapsulator.encap_secret(kem_public)
        self.handshake_timings["encapsulation_ms"] = round(
            (time.perf_counter() - started) * 1000, 6
        )
        handshake = {
            "security_version": hello["security_version"],
            "run_id": hello["run_id"],
            "session_id": hello["session_id"],
            "client_id": self.client_id,
            "server_id": hello["server_id"],
            "contract_hash": hello["contract_hash"],
            "kem_algorithm": hello["kem_algorithm"],
            "signature_algorithm": hello["signature_algorithm"],
            "kdf": hello["kdf"],
            "aead": hello["aead"],
            "server_nonce": hello["server_nonce"],
            "client_nonce": encode_bytes(secrets.token_bytes(32)),
            "kem_public_key_sha256": hello["kem_public_key_sha256"],
            "kem_ciphertext": encode_bytes(ciphertext),
            "kem_ciphertext_sha256": sha256_bytes(ciphertext),
            "client_identity_fingerprint": self.identity["fingerprint"],
            "registration": registration,
        }
        started = time.perf_counter()
        signature = self._signer.sign(signed_bytes("client_handshake", handshake))
        self.handshake_timings["signing_ms"] = round((time.perf_counter() - started) * 1000, 6)
        context = session_context(hello, handshake)
        session = make_session(
            role="client", config=self.config, shared_secret=shared_secret, context=context
        )
        request = {"handshake": handshake, "signature": encode_bytes(signature)}
        self.handshake_timings["handshake_request_bytes"] = len(canonical_bytes(request))
        return request, session

    def accept_confirmation(self, envelope: dict, session: SecureSession) -> dict:
        confirmation, _ = session.open_json(
            envelope,
            message_type="session_confirmation",
            server_round=0,
            maximum_plaintext_bytes=4096,
        )
        if confirmation != {
            "accepted": True,
            "run_id": session.run_id,
            "client_id": self.client_id,
        }:
            raise SecurityError("metadata_mismatch", "session confirmation is incorrect")
        self.session = session
        return confirmation

    def safe_summary(self) -> dict:
        if self.session is None:
            raise RuntimeError("secure session has not been established")
        return {
            "client_id": self.client_id,
            "identity": self.identity["fingerprint"],
            "handshake_timings_ms": self.handshake_timings,
            "channel": self.session.safe_summary(),
        }


def load_security_material(config_path: str, trust_path: str, secret_path: str) -> tuple[dict, dict, bytes]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    trust = json.loads(Path(trust_path).read_text(encoding="utf-8"))
    secret = Path(secret_path).read_bytes()
    if not secret:
        raise RuntimeError("provisioned signing secret is empty")
    return config, trust, secret
