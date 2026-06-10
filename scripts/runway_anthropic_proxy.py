#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx


DEFAULT_RUNWAY_BASE = "https://runway.devops.rednote.life"
DEFAULT_RAW_PREDICT_PATH = "/openai/google/anthropic/v1:rawPredict"
DEFAULT_FALLBACK_BASE = "http://10.39.10.241:10001"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _models_response() -> dict[str, Any]:
    models = [
        {
            "type": "model",
            "id": "claude-sonnet-4-6",
            "display_name": "Claude Sonnet 4.6",
            "created_at": "2026-01-01T00:00:00Z",
        },
        {
            "type": "model",
            "id": "claude-opus-4-8",
            "display_name": "Claude Opus 4.8",
            "created_at": "2026-01-01T00:00:00Z",
        },
        {
            "type": "model",
            "id": "claude-opus-4-7",
            "display_name": "Claude Opus 4.7",
            "created_at": "2026-01-01T00:00:00Z",
        },
        {
            "type": "model",
            "id": "claude-opus-4-6",
            "display_name": "Claude Opus 4.6",
            "created_at": "2026-01-01T00:00:00Z",
        },
    ]
    return {
        "type": "list",
        "data": models,
        "first_id": models[0]["id"],
        "last_id": models[-1]["id"],
        "has_more": False,
    }


def _sse_event(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n".encode(
        "utf-8"
    )


def _anthropic_to_raw_predict(req: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "max_tokens",
        "messages",
        "system",
        "tools",
        "tool_choice",
        "thinking",
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
    }
    payload = {k: v for k, v in req.items() if k in allowed and v is not None}
    payload["anthropic_version"] = "vertex-2023-10-16"
    # Claude Code may request very large max_tokens. Keep it, but ensure a sane
    # type because Vertex rejects stringified values.
    if "max_tokens" in payload:
        payload["max_tokens"] = int(payload["max_tokens"])
    return payload


def _uses_raw_predict_route(model: str | None) -> bool:
    if not model:
        return True
    normalized = model.lower().replace("_", "-")
    return "opus-4-8" in normalized or "opus-48" in normalized


def _message_for_start(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": data.get("id", f"msg_proxy_{int(time.time() * 1000)}"),
        "type": "message",
        "role": "assistant",
        "model": data.get("model", "claude-opus-4-8"),
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": data.get("usage", {}),
    }


def _stream_response(handler: BaseHTTPRequestHandler, data: dict[str, Any]) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "close")
    handler.end_headers()

    handler.wfile.write(_sse_event("message_start", {"type": "message_start", "message": _message_for_start(data)}))
    handler.wfile.flush()

    out_index = 0
    for block in data.get("content") or []:
        btype = block.get("type")
        if btype == "text":
            text = block.get("text") or ""
            handler.wfile.write(
                _sse_event(
                    "content_block_start",
                    {"type": "content_block_start", "index": out_index, "content_block": {"type": "text", "text": ""}},
                )
            )
            if text:
                handler.wfile.write(
                    _sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": out_index,
                            "delta": {"type": "text_delta", "text": text},
                        },
                    )
                )
            handler.wfile.write(_sse_event("content_block_stop", {"type": "content_block_stop", "index": out_index}))
            out_index += 1
        elif btype == "tool_use":
            tool_id = block.get("id") or f"toolu_proxy_{uuid.uuid4().hex}"
            name = block.get("name")
            tool_input = block.get("input") or {}
            handler.wfile.write(
                _sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": out_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": name,
                            "input": {},
                        },
                    },
                )
            )
            if tool_input:
                handler.wfile.write(
                    _sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": out_index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": json.dumps(tool_input, ensure_ascii=False, separators=(",", ":")),
                            },
                        },
                    )
                )
            handler.wfile.write(_sse_event("content_block_stop", {"type": "content_block_stop", "index": out_index}))
            out_index += 1
        elif btype == "thinking":
            thinking = block.get("thinking") or ""
            signature = block.get("signature")
            handler.wfile.write(
                _sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": out_index,
                        "content_block": {"type": "thinking", "thinking": ""},
                    },
                )
            )
            if thinking:
                handler.wfile.write(
                    _sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": out_index,
                            "delta": {"type": "thinking_delta", "thinking": thinking},
                        },
                    )
                )
            if signature:
                handler.wfile.write(
                    _sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": out_index,
                            "delta": {"type": "signature_delta", "signature": signature},
                        },
                    )
                )
            handler.wfile.write(_sse_event("content_block_stop", {"type": "content_block_stop", "index": out_index}))
            out_index += 1

    usage = data.get("usage") or {}
    handler.wfile.write(
        _sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": data.get("stop_reason", "end_turn"),
                    "stop_sequence": data.get("stop_sequence"),
                },
                "usage": {"output_tokens": usage.get("output_tokens", 0)},
            },
        )
    )
    handler.wfile.write(_sse_event("message_stop", {"type": "message_stop"}))
    handler.wfile.flush()


class ProxyHandler(BaseHTTPRequestHandler):
    server_version = "RunwayAnthropicProxy/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        if getattr(self.server, "verbose", False):
            sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), fmt % args))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/v1/models":
            self._send_json(_models_response())
            return
        if self.path == "/" or self.path.startswith("/health"):
            self._send_json({"status": "running", "target": "runway_google_anthropic_rawPredict"})
            return
        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/v1/messages":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            body = self.rfile.read(length)
            req = json.loads(body.decode("utf-8"))
            if getattr(self.server, "model_router", False) and not _uses_raw_predict_route(req.get("model")):
                self._forward_to_fallback(body, parsed)
                return
            payload = _anthropic_to_raw_predict(req)
            data = self._call_runway(payload)
            if req.get("stream"):
                _stream_response(self, data)
            else:
                self._send_json(data)
        except Exception as exc:
            self._send_json(
                {"type": "error", "error": {"type": "proxy_error", "message": str(exc)}},
                status=500,
            )

    def _send_json(self, obj: Any, status: int = 200) -> None:
        body = _json_bytes(obj)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _forward_to_fallback(self, body: bytes, parsed: Any) -> None:
        base = str(getattr(self.server, "fallback_base", DEFAULT_FALLBACK_BASE)).rstrip("/")
        url = base + parsed.path
        if parsed.query:
            url += "?" + parsed.query
        token = os.environ.get("CLAUDE_PROXY_AUTH_TOKEN", "123")
        headers: dict[str, str] = {}
        copy_headers = {
            "accept",
            "anthropic-beta",
            "anthropic-dangerous-direct-browser-access",
            "anthropic-version",
            "content-type",
            "user-agent",
            "x-app",
            "x-claude-code-session-id",
            "x-stainless-arch",
            "x-stainless-lang",
            "x-stainless-os",
            "x-stainless-package-version",
            "x-stainless-retry-count",
            "x-stainless-runtime",
            "x-stainless-runtime-version",
            "x-stainless-timeout",
        }
        for key, value in self.headers.items():
            if key.lower() in copy_headers:
                headers[key] = value
        headers["x-api-key"] = token
        headers["authorization"] = f"Bearer {token}"
        headers.setdefault("content-type", "application/json")
        timeout = float(os.environ.get("CLAUDE_ROUTER_FALLBACK_TIMEOUT", "600"))
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", url, headers=headers, content=body) as resp:
                self.send_response(resp.status_code)
                excluded = {
                    "connection",
                    "content-encoding",
                    "content-length",
                    "transfer-encoding",
                    "keep-alive",
                    "proxy-authenticate",
                    "proxy-authorization",
                    "te",
                    "trailers",
                    "upgrade",
                    "set-cookie",
                }
                for key, value in resp.headers.items():
                    if key.lower() not in excluded:
                        self.send_header(key, value)
                self.send_header("Connection", "close")
                self.end_headers()
                for chunk in resp.iter_bytes():
                    if chunk:
                        self.wfile.write(chunk)
                        self.wfile.flush()

    def _call_runway(self, payload: dict[str, Any]) -> dict[str, Any]:
        api_key = os.environ.get("RUNWAY_OPUS48_API_KEY")
        if not api_key:
            raise RuntimeError("RUNWAY_OPUS48_API_KEY is not set")
        base = os.environ.get("RUNWAY_BASE_URL", DEFAULT_RUNWAY_BASE).rstrip("/")
        url = base + DEFAULT_RAW_PREDICT_PATH
        headers = {"api-key": api_key, "Content-Type": "application/json"}
        timeout = float(os.environ.get("RUNWAY_PROXY_TIMEOUT", "600"))
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        if "Code" in data and "Error" in data:
            raise RuntimeError(str(data.get("Error")))
        return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--env-file", default="autoresearch_idea_harness/.env")
    parser.add_argument(
        "--model-router",
        action="store_true",
        help="Route Opus 4.8 to Runway Google rawPredict and other models to fallback Anthropic gateway.",
    )
    parser.add_argument("--fallback-base", default=DEFAULT_FALLBACK_BASE)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    load_env_file(Path(args.env_file))
    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    server.verbose = args.verbose  # type: ignore[attr-defined]
    server.model_router = args.model_router  # type: ignore[attr-defined]
    server.fallback_base = args.fallback_base  # type: ignore[attr-defined]
    print(f"runway anthropic proxy listening on http://{args.host}:{args.port}", flush=True)
    if args.model_router:
        print(f"model router enabled; fallback base {args.fallback_base}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
