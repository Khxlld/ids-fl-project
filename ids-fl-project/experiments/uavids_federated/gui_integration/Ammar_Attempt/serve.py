"""Static file server for the RASID console.

Serves this directory only. It performs no proxying, holds no credentials, and
never touches model, dataset, or cryptographic material — the browser talks
directly to the GUI adapter on port 8090.

Port 3000 is the default because both http://localhost:3000 and
http://127.0.0.1:3000 are already in the adapter's DEFAULT_ORIGINS, so the demo
works without passing --allowed-origins.

    python3 serve.py                 # http://127.0.0.1:3000
    python3 serve.py --port 3001     # then start the adapter with a matching
                                     # --allowed-origins http://127.0.0.1:3001
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import sys
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    """Static handler with presentation-friendly caching and quiet logging."""

    def end_headers(self) -> None:
        # A stale panel during a live demonstration is worse than a re-fetch.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def log_message(self, format_string: str, *args) -> None:
        if isinstance(args[1] if len(args) > 1 else "", str) and args[1].startswith("4"):
            sys.stderr.write("  %s %s\n" % (args[1], args[0]))


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the UAVIDS console")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    handler = functools.partial(Handler, directory=str(HERE))

    try:
        server = ReusableTCPServer((args.host, args.port), handler)
    except OSError as exc:
        raise SystemExit(
            f"cannot bind {args.host}:{args.port} ({exc}).\n"
            f"Use --port to pick another, then start the adapter with a matching "
            f"--allowed-origins http://{args.host}:<port>"
        )

    url = f"http://{args.host}:{args.port}"
    print(f"RASID console  ->  {url}")
    print("Adapter expected at http://127.0.0.1:8090/api/gui/v1")
    print("Override with a query string, e.g. ?api=http://127.0.0.1:9000/api/gui/v1")
    print("Ctrl+C to stop.")

    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
