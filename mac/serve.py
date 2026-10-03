"""Resident local HTTP server for the E2 LoRA on MLX.

  python -m mac.serve [--port 8765]

Loads JevMLX once and keeps it in memory so callers (e.g. omp hooks) avoid per-call model load.
Single-threaded on purpose: requests are served sequentially against one MLX model.
Inference uses the single canonical order (final inference config; see README).

  GET  /health   -> {"ok": true}
  POST /predict  {"state": str, "branches": [{"kind": "choice"|"score"|"noul", "instructions": str, "options": [str, ...]}]}
                 -> {"probabilities": [[float, ...], ...], "ms": float}
Noul branches use options ["No", "Yes"]; probabilities follow option order.
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from mac.jev_mlx import JevMLX
from model.packer import Branch


def parse_request(body: dict) -> tuple[str, list[Branch]]:
    state = body.get("state")
    branches = body.get("branches")
    if not isinstance(state, str) or not state:
        raise ValueError("state must be a non-empty string")
    if not isinstance(branches, list) or not branches:
        raise ValueError("branches must be a non-empty list")
    parsed = []
    for b in branches:
        if b.get("kind") not in ("choice", "score", "noul"):
            raise ValueError(f"invalid kind: {b.get('kind')!r}")
        options = b.get("options") or (["No", "Yes"] if b["kind"] == "noul" else [])
        if len(options) < 2:
            raise ValueError("each branch needs at least 2 options")
        parsed.append(Branch(b["kind"], str(b.get("instructions", "")), [str(o) for o in options]))
    return state, parsed


def make_handler(jev: JevMLX) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"ok": True})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/predict":
                self._send(404, {"error": "not found"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                state, branches = parse_request(body)
            except (ValueError, TypeError, AttributeError, KeyError) as e:
                self._send(400, {"error": str(e)})
                return
            start = time.perf_counter()
            probabilities = jev.predict_request(state, branches)
            self._send(200, {"probabilities": probabilities, "ms": round((time.perf_counter() - start) * 1000, 1)})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    jev = JevMLX()
    server = HTTPServer((args.host, args.port), make_handler(jev))
    print(f"jev serving on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
