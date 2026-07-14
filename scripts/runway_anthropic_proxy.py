#!/usr/bin/env python3
# INTERNAL-ONLY helper: local HTTP proxy that translates Anthropic Messages API
# requests onto the internal Runway LLM gateway. Not needed for external
# reproduction (see REPRODUCE.md) — external users call the Anthropic API
# directly with ANTHROPIC_API_KEY. Requires RUNWAY_BASE_URL to be set; there is
# intentionally no default gateway URL.
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


DEFAULT_RAW_PREDICT_PATH = "/openai/google/anthropic/v1:rawPredict"
DEFAULT_BEDROCK_INVOKE_PATH = "/openai/bedrock_runtime/model/invoke"
CONTEXT_1M_BETA_HEADER = "context-1m-2025-08-07"


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
            "id": "claude-sonnet-4-6[1m]",
            "display_name": "Claude Sonnet 4.6 (1M context)",
            "created_at": "2026-01-01T00:00:00Z",
        },
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


def _normalize_model(model: str | None) -> str:
    return (model or "").lower().replace("_", "-").replace("[1m]", "").strip()


def _route_for_model(model: str | None) -> tuple[str, str]:
    normalized = _normalize_model(model)
    if not normalized or normalized in {"opus", "opusplan"}:
        return "google", "RUNWAY_OPUS48_API_KEY"
    if normalized == "sonnet" or "sonnet-4-6" in normalized:
        return "bedrock", "RUNWAY_SONNET46_API_KEY"
    if "opus-4-8" in normalized or "opus-48" in normalized:
        return "google", "RUNWAY_OPUS48_API_KEY"
    if "opus-4-7" in normalized or "opus-47" in normalized:
        return "google", "RUNWAY_OPUS47_API_KEY"
    if "opus-4-6" in normalized or "opus-46" in normalized:
        return "google", "RUNWAY_OPUS46_API_KEY"
    raise ValueError(f"unsupported model for Runway Claude proxy: {model!r}")


def _split_beta_header(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _request_betas(req: dict[str, Any], headers: Any) -> list[str]:
    betas: list[str] = []
    body_betas = req.get("anthropic_beta")
    if isinstance(body_betas, list):
        betas.extend(str(item) for item in body_betas if item)
    elif isinstance(body_betas, str):
        betas.extend(_split_beta_header(body_betas))
    betas.extend(_split_beta_header(headers.get("anthropic-beta")))
    seen: set[str] = set()
    unique: list[str] = []
    for beta in betas:
        if beta not in seen:
            unique.append(beta)
            seen.add(beta)
    return unique


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


def _anthropic_to_bedrock(req: dict[str, Any], betas: list[str]) -> dict[str, Any]:
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
    payload["anthropic_version"] = "bedrock-2023-05-31"
    if "max_tokens" in payload:
        payload["max_tokens"] = int(payload["max_tokens"])
    bedrock_betas = [b for b in betas if b == CONTEXT_1M_BETA_HEADER]
    if bedrock_betas:
        payload["anthropic_beta"] = bedrock_betas
    return payload


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
            self._send_json({"status": "running", "target": "runway_anthropic_router"})
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
            provider, key_env = _route_for_model(req.get("model"))
            betas = _request_betas(req, self.headers)
            if provider == "bedrock":
                payload = _anthropic_to_bedrock(req, betas)
                data = self._call_runway_bedrock(payload, key_env)
            else:
                payload = _anthropic_to_raw_predict(req)
                data = self._call_runway_google(payload, key_env)
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

    def _call_runway_google(self, payload: dict[str, Any], key_env: str) -> dict[str, Any]:
        api_key = os.environ.get(key_env)
        if not api_key:
            raise RuntimeError(f"{key_env} is not set")
        base = os.environ.get("RUNWAY_BASE_URL", "").rstrip("/")
        if not base:
            raise RuntimeError("RUNWAY_BASE_URL is not set (internal-only Runway gateway)")
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

    def _call_runway_bedrock(self, payload: dict[str, Any], key_env: str) -> dict[str, Any]:
        api_key = os.environ.get(key_env)
        if not api_key:
            raise RuntimeError(f"{key_env} is not set")
        base = os.environ.get("RUNWAY_BASE_URL", "").rstrip("/")
        if not base:
            raise RuntimeError("RUNWAY_BASE_URL is not set (internal-only Runway gateway)")
        url = base + DEFAULT_BEDROCK_INVOKE_PATH
        headers = {"token": api_key, "Content-Type": "application/json"}
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
        help="Deprecated compatibility flag; routing is always enabled.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    load_env_file(Path(args.env_file))
    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    server.verbose = args.verbose  # type: ignore[attr-defined]
    print(f"runway anthropic proxy listening on http://{args.host}:{args.port}", flush=True)
    print("model router enabled: opus->google rawPredict, sonnet->amazon bedrock invoke", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
