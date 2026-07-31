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
DEFAULT_BEDROCK_INVOKE_PATH = "/openai/bedrock_runtime/model/invoke"
DEFAULT_MAAS_STREAM_RAW_PREDICT_URL = (
    "https://maas.devops.rednote.life/openai/openai/google/anthropic/v1:streamRawPredict"
)
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
    fable5_model = os.environ.get("RUNWAY_FABLE5_MODEL", "claude-fable-5")
    models = [
        {
            "type": "model",
            "id": fable5_model,
            "display_name": "Claude Fable 5",
            "created_at": "2026-01-01T00:00:00Z",
        },
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
    fable5_model = _normalize_model(os.environ.get("RUNWAY_FABLE5_MODEL", "claude-fable-5"))
    if not normalized or normalized in {"opus", "opusplan"}:
        return _opus48_route()
    if normalized == "fable" or "fable-5" in normalized or (fable5_model and normalized == fable5_model):
        return "maas_google_stream_fable5", "RUNWAY_FABLE5_API_KEY"
    if normalized == "sonnet" or "sonnet-4-6" in normalized:
        return "bedrock", "RUNWAY_SONNET46_API_KEY"
    if "opus-4-8" in normalized or "opus-48" in normalized:
        return _opus48_route()
    if "opus-4-7" in normalized or "opus-47" in normalized:
        return "google", "RUNWAY_OPUS47_API_KEY"
    if "opus-4-6" in normalized or "opus-46" in normalized:
        return "google", "RUNWAY_OPUS46_API_KEY"
    raise ValueError(f"unsupported model for Runway Claude proxy: {model!r}")


def _opus48_route() -> tuple[str, str]:
    if os.environ.get("RUNWAY_OPUS48_MAAS_MODEL") or os.environ.get("RUNWAY_OPUS48_MAAS_URL"):
        return "maas_google_stream_opus48", "RUNWAY_OPUS48_API_KEY"
    return "google", "RUNWAY_OPUS48_API_KEY"


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


def _maas_google_model_env(provider: str) -> tuple[str, str]:
    if provider == "maas_google_stream_opus48":
        return "RUNWAY_OPUS48_MAAS_MODEL", "claude opus 4.8"
    return "RUNWAY_FABLE5_MAAS_MODEL", "Claude Fable 5"


def _maas_google_api_key_env(provider: str, fallback_key_env: str) -> str:
    if provider == "maas_google_stream_opus48":
        return "RUNWAY_OPUS48_MAAS_API_KEY"
    if provider == "maas_google_stream_fable5":
        return "RUNWAY_FABLE5_MAAS_API_KEY"
    return fallback_key_env


def _anthropic_to_maas_google_stream(req: dict[str, Any], provider: str) -> dict[str, Any]:
    payload = _anthropic_to_raw_predict(req)
    model_env, default_model = _maas_google_model_env(provider)
    payload["model"] = os.environ.get(model_env, default_model)
    payload["stream"] = True
    return payload


def _merge_usage(base: dict[str, Any], delta: dict[str, Any] | None) -> dict[str, Any]:
    if not delta:
        return base
    out = dict(base)
    for key, value in delta.items():
        if isinstance(value, int) and isinstance(out.get(key), int):
            out[key] = max(out[key], value)
        else:
            out[key] = value
    return out


def _message_from_anthropic_sse(lines: list[str]) -> dict[str, Any]:
    message: dict[str, Any] = {
        "id": f"msg_proxy_{int(time.time() * 1000)}",
        "type": "message",
        "role": "assistant",
        "model": os.environ.get("RUNWAY_FABLE5_MODEL", "claude-fable-5"),
        "content": [],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {},
    }
    blocks: dict[int, dict[str, Any]] = {}
    tool_json_parts: dict[int, list[str]] = {}

    for line in lines:
        if not line.startswith("data:"):
            continue
        raw = line[len("data:") :].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue

        etype = event.get("type")
        if etype == "message_start":
            start_message = event.get("message") or {}
            for key in ("id", "type", "role", "model", "stop_reason", "stop_sequence"):
                if start_message.get(key) is not None:
                    message[key] = start_message[key]
            message["usage"] = _merge_usage(message.get("usage") or {}, start_message.get("usage"))
        elif etype == "content_block_start":
            idx = int(event.get("index", len(blocks)))
            block = dict(event.get("content_block") or {})
            blocks[idx] = block
            if block.get("type") == "tool_use":
                tool_json_parts[idx] = []
        elif etype == "content_block_delta":
            idx = int(event.get("index", len(blocks)))
            delta = event.get("delta") or {}
            block = blocks.setdefault(idx, {"type": "text", "text": ""})
            dtype = delta.get("type")
            if dtype == "text_delta":
                block["text"] = str(block.get("text") or "") + str(delta.get("text") or "")
            elif dtype == "thinking_delta":
                block["type"] = "thinking"
                block["thinking"] = str(block.get("thinking") or "") + str(delta.get("thinking") or "")
            elif dtype == "signature_delta":
                block["signature"] = str(delta.get("signature") or "")
            elif dtype == "input_json_delta":
                tool_json_parts.setdefault(idx, []).append(str(delta.get("partial_json") or ""))
        elif etype == "message_delta":
            delta = event.get("delta") or {}
            if delta.get("stop_reason") is not None:
                message["stop_reason"] = delta.get("stop_reason")
            if delta.get("stop_sequence") is not None:
                message["stop_sequence"] = delta.get("stop_sequence")
            message["usage"] = _merge_usage(message.get("usage") or {}, event.get("usage"))

    content: list[dict[str, Any]] = []
    for idx in sorted(blocks):
        block = blocks[idx]
        if block.get("type") == "tool_use":
            raw_input = "".join(tool_json_parts.get(idx) or [])
            if raw_input:
                try:
                    block["input"] = json.loads(raw_input)
                except json.JSONDecodeError:
                    block["input"] = {}
            else:
                block["input"] = block.get("input") or {}
        content.append(block)
    message["content"] = content
    return message


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
            elif provider.startswith("maas_google_stream"):
                payload = _anthropic_to_maas_google_stream(req, provider)
                data = self._call_maas_google_stream(payload, key_env, provider)
            else:
                payload = _anthropic_to_raw_predict(req)
                data = self._call_runway_google(payload, key_env)
            if req.get("stream"):
                _stream_response(self, data)
            else:
                self._send_json(data)
        except Exception as exc:
            try:
                sys.stderr.write(f"proxy_error model={req.get('model') if 'req' in locals() else '<unparsed>'}: {exc}\n")
                sys.stderr.flush()
            except Exception:
                pass
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

    def _call_runway_bedrock(self, payload: dict[str, Any], key_env: str) -> dict[str, Any]:
        api_key = os.environ.get(key_env)
        if not api_key:
            raise RuntimeError(f"{key_env} is not set")
        base = os.environ.get("RUNWAY_BASE_URL", DEFAULT_RUNWAY_BASE).rstrip("/")
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

    def _call_maas_google_stream(self, payload: dict[str, Any], key_env: str, provider: str) -> dict[str, Any]:
        maas_key_env = _maas_google_api_key_env(provider, key_env)
        api_key = os.environ.get(maas_key_env) or os.environ.get(key_env)
        if not api_key:
            raise RuntimeError(f"{maas_key_env} or {key_env} is not set")
        url_env = "RUNWAY_OPUS48_MAAS_URL" if provider == "maas_google_stream_opus48" else "RUNWAY_FABLE5_MAAS_URL"
        url = os.environ.get(url_env, os.environ.get("RUNWAY_FABLE5_MAAS_URL", DEFAULT_MAAS_STREAM_RAW_PREDICT_URL))
        headers = {
            "api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        timeout = float(os.environ.get("RUNWAY_PROXY_TIMEOUT", "600"))
        lines: list[str] = []
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "")
                if "text/event-stream" not in ctype:
                    body = resp.read().decode("utf-8", errors="replace")
                    try:
                        data = json.loads(body)
                    except json.JSONDecodeError:
                        raise RuntimeError(f"MaaS streamRawPredict returned non-SSE response: {body[:300]}")
                    if "Code" in data and "Error" in data:
                        raise RuntimeError(str(data.get("Error")))
                    if data.get("error"):
                        raise RuntimeError(str(data.get("error")))
                    raise RuntimeError(f"MaaS streamRawPredict returned non-SSE response: {body[:300]}")
                for line in resp.iter_lines():
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="replace")
                    if line:
                        lines.append(line)
        data = _message_from_anthropic_sse(lines)
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
    print(
        "model router enabled: fable/opus48(if configured)->maas google streamRawPredict, other opus->google rawPredict, sonnet->amazon bedrock invoke",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
