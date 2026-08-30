"""Federated learning transport (Phase 2, P3-4 / audit 2026-08-30).

The simulation-only FL pipeline (`agent/fed/simulate.py`) is now complemented
with a real HTTP transport: gateways submit signed contributions to an
aggregator server; the aggregator verifies every signature, collects the
round, runs FedAvg + MAD screening (`agent.fed.aggregate.fed_avg`) and serves
the resulting global weights.

Wire format:
  POST /v1/contributions   body: { public_key, contribution } (signed)
  POST /v1/rounds/<round>/close  → aggregates the round, returns global weights
  GET  /v1/weights?round=N → stored global weights of a round
  GET  /health

Implements the stdlib-only HTTP server (no new dependencies) and an httpx
client, so it works offline and in CI.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from agent.fed.aggregate import fed_avg
from agent.fed.protocol import verify_contribution

#: Max request body (bytes) — contributions are small.
MAX_BODY = 1_000_000


class ContributionRejected(Exception):
    """Raised when a contribution fails signature/schema verification."""


class FLServer:
    """In-process federated aggregator (single host, stdlib HTTP)."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        min_samples: int = 1,
        z_threshold: float = 3.0,
        extra_weight: Optional[Dict[str, float]] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.min_samples = min_samples
        self.z_threshold = z_threshold
        self.extra_weight = extra_weight or {}

        self._lock = threading.Lock()
        # round -> list of accepted contributions {public_key, contribution}
        self._contributions: Dict[int, List[Dict[str, Any]]] = {}
        # round -> global weights {weights, loss, accepted_count, rejected}
        self._global: Dict[int, Dict[str, Any]] = {}
        self._httpd: Optional[ThreadingHTTPServer] = None

    # ── Storage API (also used by tests directly) ────────────────────────

    def submit(self, public_key: str, contribution: Dict[str, Any]) -> int:
        """Verify a signed contribution and store it for its round.

        Returns the round number the contribution joined.
        """
        if not verify_contribution(public_key, contribution):
            raise ContributionRejected("invalid signature")
        round_no = int(contribution.get("round", 0))
        with self._lock:
            self._contributions.setdefault(round_no, []).append(
                {"public_key": public_key, "contribution": contribution}
            )
        return round_no

    def contributions(self, round_no: int) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._contributions.get(round_no, []))

    def close_round(self, round_no: int) -> Dict[str, Any]:
        """Aggregate a round with FedAvg + MAD screening; cache the result."""
        with self._lock:
            if round_no in self._global:
                return self._global[round_no]
            items = self._contributions.get(round_no, [])

        contributions = []
        for it in items:
            c = dict(it["contribution"])
            # fed_avg/verify expects the key inside the contribution; it is NOT
            # part of SIGNED_FIELDS, so the canonical message (and signature)
            # is unchanged.
            c["public_key"] = it["public_key"]
            contributions.append(c)
        result = fed_avg(
            contributions,
            verify=True,
            min_samples=self.min_samples,
            z_threshold=self.z_threshold,
            extra_weight=self.extra_weight or None,
        )
        global_weights = {
            "round": round_no,
            "weights": result.weights,
            "loss": result.loss,
            "accepted_count": result.accepted_count,
            "rejected_count": result.rejected_count,
        }
        with self._lock:
            self._global[round_no] = global_weights
        return global_weights

    def weights(self, round_no: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._global.get(round_no)

    def start(self) -> int:
        """Start the HTTP server on a background thread; returns the port."""
        self._httpd = ThreadingHTTPServer((self.host, self.port), self._handler_class())
        self.port = self._httpd.server_address[1]
        t = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        t.start()
        return self.port

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    def _handler_class(self) -> type:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # silence stdlib logs
                pass

            def _send_json(self, status: int, obj: Any) -> None:
                body = json.dumps(obj).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/health":
                    return self._send_json(200, {"ok": True})
                if parsed.path == "/v1/weights":
                    qs = parse_qs(parsed.query)
                    try:
                        round_no = int(qs.get("round", ["0"])[0])
                    except ValueError:
                        return self._send_json(400, {"error": "bad round"})
                    w = server.weights(round_no)
                    if w is None:
                        return self._send_json(404, {"error": "round not aggregated"})
                    return self._send_json(200, w)
                return self._send_json(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/v1/contributions":
                    length = int(self.headers.get("Content-Length", "0"))
                    if length > MAX_BODY:
                        return self._send_json(413, {"error": "body too large"})
                    try:
                        payload = json.loads(self.rfile.read(length) or b"{}")
                        round_no = server.submit(
                            payload["public_key"], payload["contribution"]
                        )
                    except (KeyError, json.JSONDecodeError):
                        return self._send_json(400, {"error": "bad payload"})
                    except ContributionRejected as e:
                        return self._send_json(403, {"error": str(e)})
                    return self._send_json(201, {"accepted": True, "round": round_no})

                # /v1/rounds/<round>/close
                if parsed.path.startswith("/v1/rounds/") and parsed.path.endswith("/close"):
                    try:
                        round_no = int(parsed.path.split("/")[3])
                    except ValueError:
                        return self._send_json(400, {"error": "bad round"})
                    return self._send_json(200, server.close_round(round_no))

                return self._send_json(404, {"error": "not found"})

        return Handler


def submit_contribution(
    url: str,
    public_key: str,
    contribution: Dict[str, Any],
    timeout: float = 5.0,
) -> Dict[str, Any]:
    """POST a signed contribution to an FL aggregator (httpx client)."""
    import httpx

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            f"{url.rstrip('/')}/v1/contributions",
            json={"public_key": public_key, "contribution": contribution},
        )
        resp.raise_for_status()
        return resp.json()


def fetch_weights(url: str, round_no: int, timeout: float = 5.0) -> Dict[str, Any]:
    """GET the global weights of an aggregated round."""
    import httpx

    with httpx.Client(timeout=timeout) as client:
        resp = client.get(f"{url.rstrip('/')}/v1/weights", params={"round": round_no})
        resp.raise_for_status()
        return resp.json()


def close_round(url: str, round_no: int, timeout: float = 5.0) -> Dict[str, Any]:
    """Ask the aggregator to close and aggregate a round."""
    import httpx

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{url.rstrip('/')}/v1/rounds/{round_no}/close")
        resp.raise_for_status()
        return resp.json()

