"""Dependency-free local server for the Anas_Attempt presentation GUI."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import urllib.error
import urllib.request
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_API_BASE = "http://127.0.0.1:8090/api/gui/v1"


def adapter_diagnostic(api_base: str, timeout_seconds: float = 3.5) -> dict[str, Any]:
    """Safely distinguish an absent adapter from a browser/CORS problem."""

    health_url = f"{api_base.rstrip('/')}/health"
    request = urllib.request.Request(health_url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read())
            return {
                "schema_version": "anas-adapter-diagnostic-v1",
                "adapter_reached": True,
                "http_status": response.status,
                "contract_ok": payload.get("api_version") == "uavids-gui-api-v1",
                "model_available": payload.get("model_available") is True,
                "category": "reachable",
            }
    except urllib.error.HTTPError as exc:
        return {
            "schema_version": "anas-adapter-diagnostic-v1",
            "adapter_reached": True,
            "http_status": exc.code,
            "contract_ok": False,
            "model_available": False,
            "category": "http_error",
        }
    except (urllib.error.URLError, TimeoutError, OSError):
        return {
            "schema_version": "anas-adapter-diagnostic-v1",
            "adapter_reached": False,
            "http_status": None,
            "contract_ok": False,
            "model_available": False,
            "category": "not_running_or_not_ready",
        }
    except (json.JSONDecodeError, ValueError):
        return {
            "schema_version": "anas-adapter-diagnostic-v1",
            "adapter_reached": True,
            "http_status": 200,
            "contract_ok": False,
            "model_available": False,
            "category": "invalid_contract",
        }


def positive_int(value: str, *, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{name} must be positive")
    return parsed


class PresentationHandler(SimpleHTTPRequestHandler):
    server_version = "UAVIDSPresentation/1.0"

    def __init__(self, *args: Any, directory: str, runtime_config: dict[str, Any], **kwargs: Any) -> None:
        self.runtime_config = runtime_config
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        request_path = self.path.split("?", 1)[0]
        if request_path == "/runtime-config.json":
            payload = json.dumps(self.runtime_config, separators=(",", ":")).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
            return
        if request_path == "/adapter-diagnostic.json":
            diagnostic = adapter_diagnostic(self.runtime_config["apiBase"])
            payload = json.dumps(diagnostic, separators=(",", ":")).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
            return
        super().do_GET()

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self' http://127.0.0.1:* http://localhost:*; "
            "font-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )
        if self.path.endswith((".html", ".js", ".css", ".json")) or self.path == "/":
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"[GUI] {self.address_string()} - {format_string % args}")


def create_server(
    *, host: str = "127.0.0.1", port: int = 3000, api_base: str = DEFAULT_API_BASE,
    poll_ms: int = 1200, request_timeout_ms: int = 4500,
    connect_timeout_ms: int = 20000, live_retry_ms: int = 750,
) -> ThreadingHTTPServer:
    runtime_config = {
        "apiBase": api_base.rstrip("/"),
        "pollMs": poll_ms,
        "requestTimeoutMs": request_timeout_ms,
        "connectTimeoutMs": connect_timeout_ms,
        "liveRetryMs": live_retry_ms,
    }
    handler = partial(
        PresentationHandler,
        directory=str(ROOT),
        runtime_config=runtime_config,
    )
    return ThreadingHTTPServer((host, port), handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the UAVIDS presentation GUI locally.")
    parser.add_argument("--host", default=os.environ.get("GUI_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=lambda value: positive_int(value, name="port"),
        default=positive_int(os.environ.get("GUI_PORT", "3000"), name="port"),
    )
    parser.add_argument("--api-base", default=os.environ.get("GUI_API_BASE", DEFAULT_API_BASE))
    parser.add_argument(
        "--poll-ms",
        type=lambda value: positive_int(value, name="poll-ms"),
        default=positive_int(os.environ.get("GUI_POLL_MS", "1200"), name="poll-ms"),
    )
    parser.add_argument(
        "--request-timeout-ms",
        type=lambda value: positive_int(value, name="request-timeout-ms"),
        default=positive_int(os.environ.get("GUI_REQUEST_TIMEOUT_MS", "4500"), name="request-timeout-ms"),
    )
    parser.add_argument(
        "--connect-timeout-ms",
        type=lambda value: positive_int(value, name="connect-timeout-ms"),
        default=positive_int(os.environ.get("GUI_CONNECT_TIMEOUT_MS", "20000"), name="connect-timeout-ms"),
    )
    parser.add_argument(
        "--live-retry-ms",
        type=lambda value: positive_int(value, name="live-retry-ms"),
        default=positive_int(os.environ.get("GUI_LIVE_RETRY_MS", "750"), name="live-retry-ms"),
    )
    parser.add_argument("--open", action="store_true", help="Open the local URL in the default browser.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        server = create_server(
            host=args.host,
            port=args.port,
            api_base=args.api_base,
            poll_ms=args.poll_ms,
            request_timeout_ms=args.request_timeout_ms,
            connect_timeout_ms=args.connect_timeout_ms,
            live_retry_ms=args.live_retry_ms,
        )
    except OSError as exc:
        print(f"Could not start the GUI server on {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 2

    url = f"http://{args.host}:{server.server_port}/"
    print("UAVIDS Federated Defense Console")
    print(f"Open: {url}")
    print(f"Live adapter: {args.api_base}")
    print("Recorded Demo works even when the live adapter is unavailable.")
    print("Press Ctrl+C to stop.")
    if args.open:
        threading.Timer(0.55, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping GUI server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
