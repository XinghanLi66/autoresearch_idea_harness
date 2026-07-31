#!/usr/bin/env python3
"""Anthropic /v1/messages -> RedNote Runway/MaaS rawPredict shim.

Lets any native Anthropic-SDK client (e.g. MLS-Bench's AnthropicClient) talk to
the internal Runway/MaaS gateways, which speak Vertex-style
``:rawPredict`` / ``:streamRawPredict`` with an ``api-key`` header instead of
the standard Anthropic wire protocol.

Routing (by request "model"):
    claude-fable-5    -> MaaS streamRawPredict/rawPredict (RUNWAY_FABLE5_MAAS_*)
    claude-opus-4-8   -> MaaS streamRawPredict/rawPredict (RUNWAY_OPUS48_MAAS_*)
    claude-opus-4-6   -> legacy Runway rawPredict (RUNWAY_OPUS46_API_KEY);
                         the key implies the model, so "model" is stripped.

Request/response bodies pass through verbatim apart from:
    - "anthropic_version": "vertex-2023-10-16" is added
    - "model" is replaced with the MaaS model name (or removed for legacy)
Streaming responses are proxied as raw SSE bytes (the gateway already emits
Anthropic-format events), so tools / thinking / cache_control all work.

Usage:
    python scripts/anthropic_maas_shim.py [--port 18791]
    # then: anthropic.Anthropic(base_url="http://127.0.0.1:18791", api_key="local")

Keys are read from autoresearch_idea_harness/.env (python-dotenv).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="[shim] %(asctime)s %(message)s")
log = logging.getLogger("anthropic_maas_shim")

HARNESS_ROOT = Path(__file__).resolve().parent.parent
LEGACY_RUNWAY_BASE = "https://runway.devops.rednote.life/openai/google/anthropic/v1"

# Upstream calls can legitimately take many minutes (large agent turns).
UPSTREAM_TIMEOUT = httpx.Timeout(connect=30.0, read=3600.0, write=120.0, pool=30.0)

# Empty-completion retry: under concurrency the MaaS gateway sometimes returns a
# near-empty completion (output_tokens ~2) for large-prompt requests — it ingests the
# prompt (cache_creation ~33k) but emits nothing. mlsbench's own retry is 3x within ~6s
# (too fast to let the gateway recover). The shim buffers each streamed response, and if
# the completion is degenerate (< MIN_OUTPUT_TOKENS) re-issues upstream with SPACED
# backoff so a transient/load-induced empty gets a real recovery window, then replays the
# good stream to the client. Tunable via env.
MIN_OUTPUT_TOKENS = int(os.environ.get("SHIM_MIN_OUTPUT_TOKENS", "5"))
EMPTY_RETRY_BACKOFF = [float(s) for s in
                       os.environ.get("SHIM_EMPTY_BACKOFF", "4,12,30").split(",") if s.strip()]


def build_routes() -> dict[str, dict]:
    """Model-name -> route table, resolved from .env at startup."""
    routes: dict[str, dict] = {}

    for alias, prefix in (("claude-fable-5", "RUNWAY_FABLE5"),
                          ("claude-opus-4-8", "RUNWAY_OPUS48")):
        stream_url = os.environ.get(f"{prefix}_MAAS_URL", "")
        key = (os.environ.get(f"{prefix}_MAAS_API_KEY")
               or os.environ.get(f"{prefix}_API_KEY", ""))
        maas_model = os.environ.get(f"{prefix}_MAAS_MODEL", "")
        if stream_url and key and maas_model:
            routes[alias] = {
                "stream_url": stream_url,
                "raw_url": stream_url.replace(":streamRawPredict", ":rawPredict"),
                "api_key": key,
                "upstream_model": maas_model,   # replace "model" with this
            }

    opus46_key = os.environ.get("RUNWAY_OPUS46_API_KEY", "")
    if opus46_key:
        routes["claude-opus-4-6"] = {
            "stream_url": f"{LEGACY_RUNWAY_BASE}:streamRawPredict",
            "raw_url": f"{LEGACY_RUNWAY_BASE}:rawPredict",
            "api_key": opus46_key,
            "upstream_model": None,             # legacy gateway rejects "model"
        }

    return routes


ROUTES: dict[str, dict] = {}


class ShimHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter default access log
        log.info("%s %s", self.address_string(), fmt % args)

    def _send_json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/health", "/"):
            self._send_json(200, {"ok": True, "models": sorted(ROUTES)})
        else:
            self._send_json(404, {"error": {"type": "not_found",
                                            "message": f"no route: GET {self.path}"}})

    def do_POST(self):
        if not self.path.startswith("/v1/messages"):
            self._send_json(404, {"error": {"type": "not_found",
                                            "message": f"no route: POST {self.path}"}})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": {"type": "invalid_request_error",
                                            "message": f"bad JSON body: {exc}"}})
            return

        model = str(body.get("model", ""))
        # Arm-tagging support: "claude-fable-5__base" routes like "claude-fable-5".
        # The full tagged name stays in the caller's leaderboard; only routing here
        # strips the suffix.
        route_model = model.split("__", 1)[0]
        route = ROUTES.get(route_model)
        if route is None:
            self._send_json(404, {
                "error": {"type": "not_found_error",
                          "message": f"model {model!r} (route {route_model!r}) not routed; "
                                     f"available: {sorted(ROUTES)}"}})
            return

        body["anthropic_version"] = "vertex-2023-10-16"
        if route["upstream_model"] is None:
            body.pop("model", None)
        else:
            body["model"] = route["upstream_model"]

        stream = bool(body.get("stream"))
        url = route["stream_url"] if stream else route["raw_url"]
        headers = {"api-key": route["api_key"], "Content-Type": "application/json"}
        if stream:
            headers["Accept"] = "text/event-stream"

        try:
            if stream:
                self._proxy_stream(url, headers, body, model)
            else:
                self._proxy_once(url, headers, body, model)
        except httpx.HTTPError as exc:
            log.error("upstream error for %s: %s", model, exc)
            try:
                self._send_json(502, {"error": {"type": "api_error",
                                                "message": f"upstream error: {exc}"}})
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _proxy_once(self, url: str, headers: dict, body: dict, model: str) -> None:
        with httpx.Client(timeout=UPSTREAM_TIMEOUT) as client:
            resp = client.post(url, headers=headers, json=body)
        data = resp.content
        # Gateway errors come back as HTTP 200 + {"Code":..,"Error":..}; convert
        # to a proper Anthropic-style error so the SDK raises cleanly.
        if resp.status_code == 200:
            try:
                parsed = json.loads(data)
                if "Code" in parsed and "Error" in parsed:
                    self._send_json(502, {"error": {
                        "type": "api_error",
                        "message": f"gateway Code={parsed['Code']}: {parsed['Error']}"}})
                    return
                # restore the alias so clients see the model they asked for
                if parsed.get("model"):
                    parsed["model"] = model
                    data = json.dumps(parsed).encode()
            except json.JSONDecodeError:
                pass
        self.send_response(resp.status_code)
        self.send_header("Content-Type",
                         resp.headers.get("content-type", "application/json"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    @staticmethod
    def _stream_output_tokens(raw: bytes) -> int:
        """Parse an Anthropic SSE byte buffer for the final usage.output_tokens.
        Returns -1 if not determinable (treated as non-empty -> no retry)."""
        best = -1
        for line in raw.split(b"\n"):
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == b"[DONE]":
                continue
            try:
                evt = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = evt.get("usage") or (evt.get("message") or {}).get("usage")
            if isinstance(usage, dict) and "output_tokens" in usage:
                try:
                    best = max(best, int(usage["output_tokens"]))
                except (TypeError, ValueError):
                    pass
        return best

    def _proxy_stream(self, url: str, headers: dict, body: dict, model: str) -> None:
        # Buffer each upstream stream so we can detect a degenerate/empty completion and
        # re-issue with spaced backoff before replaying to the client. Buffering (vs true
        # passthrough) is fine here: the caller is a non-interactive eval worker.
        attempts = len(EMPTY_RETRY_BACKOFF) + 1
        raw = b""
        status = 200
        ctype = "text/event-stream"
        for i in range(attempts):
            with httpx.Client(timeout=UPSTREAM_TIMEOUT) as client:
                with client.stream("POST", url, headers=headers, json=body) as resp:
                    status = resp.status_code
                    ctype = resp.headers.get("content-type", "")
                    raw = resp.read()
            if "text/event-stream" not in ctype:
                # Gateway error (HTTP 200 + {Code,Error}, or non-SSE). These are also
                # transient under load, so retry with backoff too; surface on last try.
                if i < attempts - 1:
                    log.warning("non-SSE upstream for %s (attempt %d/%d); retrying",
                                model, i + 1, attempts)
                    time.sleep(EMPTY_RETRY_BACKOFF[i])
                    continue
                try:
                    parsed = json.loads(raw)
                    msg = (f"gateway Code={parsed.get('Code')}: {parsed.get('Error')}"
                           if "Code" in parsed else str(parsed)[:500])
                except json.JSONDecodeError:
                    msg = raw[:500].decode("utf-8", errors="replace")
                self._send_json(502, {"error": {"type": "api_error", "message": msg}})
                return
            out_toks = self._stream_output_tokens(raw)
            if out_toks < 0 or out_toks >= MIN_OUTPUT_TOKENS or i == attempts - 1:
                if 0 <= out_toks < MIN_OUTPUT_TOKENS:
                    log.warning("empty completion (output_tokens=%d) for %s after %d "
                                "attempts; passing through", out_toks, model, i + 1)
                break
            log.warning("empty completion (output_tokens=%d) for %s (attempt %d/%d); "
                        "backing off %ss", out_toks, model, i + 1, attempts,
                        EMPTY_RETRY_BACKOFF[i])
            time.sleep(EMPTY_RETRY_BACKOFF[i])

        # Replay the buffered SSE to the client.
        self.send_response(status)
        self.send_header("Content-Type", ctype or "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(raw)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            log.warning("client disconnected before replay (%s)", model)
        self.close_connection = True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=18791)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--env-file", default=str(HARNESS_ROOT / ".env"))
    args = ap.parse_args()

    load_dotenv(args.env_file)
    global ROUTES
    ROUTES = build_routes()
    if not ROUTES:
        raise SystemExit(f"No routes resolvable from {args.env_file} — check keys.")
    log.info("routes: %s", sorted(ROUTES))

    server = ThreadingHTTPServer((args.host, args.port), ShimHandler)
    log.info("listening on http://%s:%d (threads=%s)",
             args.host, args.port, threading.active_count())
    server.serve_forever()


if __name__ == "__main__":
    main()
