"""Secure transport subclass that preserves the Phase 4 training loop."""

from __future__ import annotations

import copy
import os

try:
    from phase4_app.client import DemoClient, log
    from phase4_app.common import ProtocolError, json_request
except ModuleNotFoundError:  # host-side tests
    from phase4.app.client import DemoClient, log
    from phase4.app.common import ProtocolError, json_request

from .security import (
    ClientSecurityManager,
    SecurityError,
    canonical_bytes,
    decode_bytes,
    encode_bytes,
    load_security_material,
)


class SecureDemoClient(DemoClient):
    def __init__(self) -> None:
        super().__init__()
        config, trust, secret = load_security_material(
            os.environ["SECURITY_CONFIG_PATH"],
            os.environ["TRUST_STORE_PATH"],
            os.environ["CLIENT_SIGN_SECRET_PATH"],
        )
        self.security = ClientSecurityManager(self.client_id, config, trust, secret)
        self.attack_tests = os.environ.get("SECURITY_ATTACK_TESTS", "0") == "1"

    def _expect_rejection(self, path: str, payload: dict, label: str) -> None:
        try:
            json_request("POST", f"{self.server_url}{path}", payload, timeout=30)
        except ProtocolError:
            log("controlled_attack_rejected", client_id=self.client_id, attack=label)
            return
        raise RuntimeError(f"controlled attack was unexpectedly accepted: {label}")

    def _exercise_handshake_rejections(self, request: dict) -> None:
        modified_metadata = copy.deepcopy(request)
        modified_metadata["handshake"]["run_id"] = "wrong-run-id"
        self._expect_rejection(
            "/api/v2/secure/session", modified_metadata, "modified_signed_metadata"
        )

        wrong_identity = copy.deepcopy(request)
        wrong_identity["handshake"]["client_id"] = "uav-client-2"
        self._expect_rejection(
            "/api/v2/secure/session", wrong_identity, "wrong_client_identity"
        )

        wrong_signature = copy.deepcopy(request)
        signature = bytearray(decode_bytes(wrong_signature["signature"], "signature", 8192))
        signature[-1] ^= 1
        wrong_signature["signature"] = encode_bytes(bytes(signature))
        self._expect_rejection(
            "/api/v2/secure/session", wrong_signature, "wrong_signature"
        )

    def _exercise_channel_rejections(self) -> None:
        session = self.security.session
        if session is None:
            raise RuntimeError("secure attack probes require an established session")
        valid = session.seal_json(
            {"purpose": "controlled_rejection_test"},
            message_type="attack_probe",
            server_round=0,
        )
        tampered = copy.deepcopy(valid)
        ciphertext = bytearray(decode_bytes(tampered["ciphertext"], "ciphertext", 4096))
        ciphertext[-1] ^= 1
        tampered["ciphertext"] = encode_bytes(bytes(ciphertext))
        self._expect_rejection("/api/v2/secure/probe", tampered, "modified_ciphertext_or_tag")

        for field, value, label in [
            ("run_id", "wrong-run-id", "wrong_run_id"),
            ("round", 99, "wrong_round"),
            ("recipient", "wrong-recipient", "wrong_recipient"),
            ("message_type", "client_update", "wrong_message_type"),
        ]:
            modified = copy.deepcopy(valid)
            modified["header"][field] = value
            self._expect_rejection("/api/v2/secure/probe", modified, label)

        malformed = copy.deepcopy(valid)
        malformed["ciphertext"] = "not-base64!"
        self._expect_rejection("/api/v2/secure/probe", malformed, "malformed_payload")

        accepted, _ = json_request(
            "POST", f"{self.server_url}/api/v2/secure/probe", valid, timeout=30
        )
        acknowledgement, _ = session.open_json(
            accepted["secure_envelope"],
            message_type="attack_probe_ack",
            server_round=0,
            maximum_plaintext_bytes=4096,
        )
        if not acknowledgement.get("accepted"):
            raise RuntimeError("controlled valid probe was not acknowledged")
        self._expect_rejection("/api/v2/secure/probe", valid, "replayed_message")

        second = session.seal_json(
            {"purpose": "controlled_rejection_test"},
            message_type="attack_probe",
            server_round=0,
        )
        duplicate_nonce = copy.deepcopy(second)
        duplicate_nonce["header"]["nonce"] = valid["header"]["nonce"]
        self._expect_rejection("/api/v2/secure/probe", duplicate_nonce, "duplicate_nonce")
        accepted, _ = json_request(
            "POST", f"{self.server_url}/api/v2/secure/probe", second, timeout=30
        )
        session.open_json(
            accepted["secure_envelope"],
            message_type="attack_probe_ack",
            server_round=0,
            maximum_plaintext_bytes=4096,
        )

    def _register(self, payload: dict) -> tuple[dict, int]:
        hello_request = {
            "registration": payload,
            "client_identity_fingerprint": self.security.identity["fingerprint"],
        }
        if self.attack_tests and self.client_id == "uav-client-1":
            unknown = copy.deepcopy(hello_request)
            unknown["registration"]["client_id"] = "untrusted-client"
            self._expect_rejection("/api/v2/secure/hello", unknown, "untrusted_identity")

        hello, _ = json_request(
            "POST", f"{self.server_url}/api/v2/secure/hello", hello_request, timeout=30
        )
        request, session = self.security.prepare(hello, payload)
        if self.attack_tests and self.client_id == "uav-client-1":
            self._exercise_handshake_rejections(request)
        result, status = json_request(
            "POST", f"{self.server_url}/api/v2/secure/session", request, timeout=30
        )
        confirmation = self.security.accept_confirmation(result["confirmation"], session)
        if self.attack_tests and self.client_id == "uav-client-1":
            self._exercise_channel_rejections()
        return {"accepted": True, "run_id": confirmation["run_id"], "status": result["status"]}, status

    def _request_model(self, completed_round: int) -> tuple[dict, int]:
        session = self.security.session
        if session is None:
            raise RuntimeError("secure session is unavailable")
        envelope = session.seal_json(
            {
                "client_id": self.client_id,
                "run_id": self.run_id,
                "completed_round": int(completed_round),
            },
            message_type="model_request",
            server_round=int(completed_round) + 1,
        )
        result, status = json_request(
            "POST", f"{self.server_url}/api/v2/secure/model", envelope, timeout=30
        )
        response, header = session.open_json(
            result["secure_envelope"],
            message_type="global_model",
            maximum_plaintext_bytes=int(self.security.config["maximum_secure_json_bytes"]),
        )
        if response.get("available"):
            if header["round"] != int(response["round"]):
                raise SecurityError("metadata_mismatch", "protected global model round is incorrect")
            if header["model_sha256"] != response.get("weights_sha256"):
                raise SecurityError("metadata_mismatch", "protected global model hash is incorrect")
        return response, status

    def _post_event_payload(self, payload: dict) -> tuple[dict, int]:
        session = self.security.session
        if session is None:
            raise RuntimeError("secure session is unavailable")
        envelope = session.seal_json(
            payload,
            message_type="client_event",
            server_round=int(payload["round"]),
        )
        return json_request(
            "POST", f"{self.server_url}/api/v2/secure/events", envelope, timeout=30
        )

    def _submit_update(self, payload: dict) -> tuple[dict, int]:
        session = self.security.session
        if session is None:
            raise RuntimeError("secure session is unavailable")
        envelope = session.seal_json(
            payload,
            message_type="client_update",
            server_round=int(payload["round"]),
            model_sha256=payload["weights_sha256"],
        )
        return json_request(
            "POST",
            f"{self.server_url}/api/v2/secure/updates",
            envelope,
            timeout=float(self.demo["round_timeout_seconds"]) + 30,
        )

    def _on_complete(self, completed_round: int) -> None:
        summary = self.security.safe_summary()
        self.post_event("client_security_summary", completed_round, summary)
        log(
            "client_security_complete",
            client_id=self.client_id,
            sent_messages=summary["channel"]["sent_messages"],
            received_messages=summary["channel"]["received_messages"],
        )


def main() -> None:
    client: SecureDemoClient | None = None
    try:
        client = SecureDemoClient()
        client.run()
    except Exception as exc:
        log(
            "client_fatal_error",
            client_id=os.environ.get("CLIENT_ID"),
            error=type(exc).__name__,
            detail=str(exc),
        )
        if client is not None and client.run_id is not None:
            try:
                client.post_event(
                    "client_protocol_error",
                    0,
                    {"error": type(exc).__name__, "detail": "secure client protocol failure"},
                    severity="error",
                )
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
