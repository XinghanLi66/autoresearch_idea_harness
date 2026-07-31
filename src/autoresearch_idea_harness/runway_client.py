from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx


DEFAULT_MAAS_ANTHROPIC_URL = "https://maas.devops.rednote.life/openai/openai/google/anthropic/v1:streamRawPredict"


@dataclass
class ChatResult:
    text: str
    model: str
    usage: dict[str, Any]
    raw_finish_reason: str | None = None


class RunwayClient:
    """Small OpenAI-compatible client for the internal Runway endpoint."""

    def __init__(self, cfg: dict[str, Any], key_env: str) -> None:
        llm_cfg = cfg.get("llm", {})
        self.base_url = str(llm_cfg.get("runway_base_url", "https://runway.devops.rednote.life")).rstrip("/")
        self.api_version = str(llm_cfg.get("chat_api_version", "2024-12-01-preview"))
        self.timeout = float(llm_cfg.get("timeout", 300))
        self.api_key = os.environ.get(key_env)
        self.key_env = key_env
        if not self.api_key:
            raise RuntimeError(
                f"Missing API key env var {key_env}. Put it in autoresearch_idea_harness/.env "
                "or export it in the shell."
            )

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int,
        stream: bool = True,
    ) -> ChatResult:
        url = f"{self.base_url}/openai/chat/completions?api-version={self.api_version}"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
            try:
                return self._chat_stream(url, headers, payload, model)
            except RuntimeError as exc:
                if "no SSE data lines" not in str(exc):
                    raise
                payload.pop("stream", None)
                payload.pop("stream_options", None)
                return self._chat_once(url, headers, payload, model)
        return self._chat_once(url, headers, payload, model)

    def responses(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int,
    ) -> ChatResult:
        maas_url = self._maas_chat_url(model)
        if maas_url:
            return self._maas_chat(
                url=maas_url,
                model=self._maas_model(model),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
        url = f"{self.base_url}/openai/v1/responses?api-version=preview"
        payload = {
            "model": model,
            "input": self._messages_to_input(messages),
            "max_output_tokens": max_tokens,
            "text": {"verbosity": "medium"},
        }
        if temperature is not None:
            payload["temperature"] = temperature
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        self._raise_runway_error(data)
        return ChatResult(
            text=self._extract_responses_text(data).strip(),
            model=data.get("model") or model,
            usage=data.get("usage") or {},
            raw_finish_reason=data.get("status"),
        )

    def complete(
        self,
        *,
        endpoint: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int,
        stream: bool = True,
    ) -> ChatResult:
        if endpoint == "responses":
            return self.responses(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if endpoint in {"google_anthropic", "anthropic_google", "raw_predict"}:
            return self.google_anthropic_raw_predict(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if endpoint in {"chat", "chat_completions"}:
            return self.chat(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )
        raise ValueError(f"Unknown Runway endpoint: {endpoint}")

    def google_anthropic_raw_predict(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int,
    ) -> ChatResult:
        maas_url = self._maas_anthropic_url(model)
        if maas_url:
            return self._maas_anthropic_stream(
                url=maas_url,
                model=self._maas_model(model),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        url = f"{self.base_url}/openai/google/anthropic/v1:rawPredict"
        anthropic_messages, system = self._messages_to_anthropic(messages)
        payload: dict[str, Any] = {
            "anthropic_version": "vertex-2023-10-16",
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        self._raise_runway_error(data)
        return ChatResult(
            text=self._extract_anthropic_text(data).strip(),
            model=data.get("model") or model,
            usage=data.get("usage") or {},
            raw_finish_reason=data.get("stop_reason"),
        )

    def _chat_once(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        model: str,
    ) -> ChatResult:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        self._raise_runway_error(data)
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return ChatResult(
            text=(message.get("content") or "").strip(),
            model=data.get("model") or model,
            usage=data.get("usage") or {},
            raw_finish_reason=choice.get("finish_reason"),
        )

    def _chat_stream(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        model: str,
    ) -> ChatResult:
        text_parts: list[str] = []
        usage: dict[str, Any] = {}
        finish_reason = None
        seen_any_sse = False
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="replace")
                    if not line.startswith("data:"):
                        continue
                    seen_any_sse = True
                    data_s = line[len("data:"):].strip()
                    if data_s == "[DONE]":
                        break
                    try:
                        event = json.loads(data_s)
                    except json.JSONDecodeError:
                        continue
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if content:
                            text_parts.append(content)
                        if choice.get("finish_reason"):
                            finish_reason = choice.get("finish_reason")
        if not seen_any_sse:
            raise RuntimeError("Runway stream returned no SSE data lines.")
        return ChatResult(
            text="".join(text_parts).strip(),
            model=model,
            usage=usage,
            raw_finish_reason=finish_reason,
        )

    def _maas_chat(
        self,
        *,
        url: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int,
        stream: bool,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if self._uses_max_completion_tokens(model):
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        headers = {
            "api-key": self._maas_api_key(),
            "Content-Type": "application/json",
        }
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
            try:
                return self._chat_stream(url, headers, payload, model)
            except RuntimeError as exc:
                if "no SSE data lines" not in str(exc):
                    raise
                payload.pop("stream", None)
                payload.pop("stream_options", None)
        return self._chat_once(url, headers, payload, model)

    def _maas_anthropic_stream(
        self,
        *,
        url: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int,
    ) -> ChatResult:
        anthropic_messages, system = self._messages_to_anthropic(messages)
        payload: dict[str, Any] = {
            "anthropic_version": "vertex-2023-10-16",
            "stream": True,
            "max_tokens": max_tokens,
            "model": model,
            "messages": anthropic_messages,
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        headers = {
            "api-key": self._maas_api_key(),
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        text_parts: list[str] = []
        usage: dict[str, Any] = {}
        finish_reason = None
        saw_sse = False
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "")
                if "text/event-stream" not in ctype:
                    body = resp.read().decode("utf-8", errors="replace")
                    try:
                        data = json.loads(body)
                    except json.JSONDecodeError:
                        raise RuntimeError(f"MaaS Anthropic returned non-SSE response: {body[:300]}")
                    self._raise_runway_error(data)
                    raise RuntimeError(f"MaaS Anthropic returned non-SSE response: {body[:300]}")
                for line in resp.iter_lines():
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="replace")
                    if not line.startswith("data:"):
                        continue
                    saw_sse = True
                    data_s = line[len("data:"):].strip()
                    if not data_s or data_s == "[DONE]":
                        continue
                    try:
                        event = json.loads(data_s)
                    except json.JSONDecodeError:
                        continue
                    etype = event.get("type")
                    if etype == "message_start":
                        usage = (event.get("message") or {}).get("usage") or usage
                    elif etype == "content_block_delta":
                        delta = event.get("delta") or {}
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            text_parts.append(str(delta["text"]))
                    elif etype == "message_delta":
                        delta = event.get("delta") or {}
                        finish_reason = delta.get("stop_reason") or finish_reason
                        if event.get("usage"):
                            usage.update(event["usage"])
        if not saw_sse:
            raise RuntimeError("MaaS Anthropic stream returned no SSE data lines.")
        return ChatResult(
            text="".join(text_parts).strip(),
            model=model,
            usage=usage,
            raw_finish_reason=finish_reason,
        )

    def _maas_api_key(self) -> str:
        candidate = os.environ.get(self._prefix_env("MAAS_API_KEY")) or self.api_key
        if not candidate:
            raise RuntimeError(f"Missing API key env var {self.key_env}.")
        return candidate

    def _maas_model(self, model: str) -> str:
        return os.environ.get(self._prefix_env("MAAS_MODEL")) or model

    def _maas_anthropic_url(self, model: str) -> str | None:
        explicit = os.environ.get(self._prefix_env("MAAS_URL"))
        if explicit:
            return explicit
        normalized = self._normalize_model(model)
        if "fable" in normalized or "opus-4-8" in normalized or "opus-48" in normalized:
            return os.environ.get("RUNWAY_DEFAULT_MAAS_ANTHROPIC_URL")
        return None

    def _maas_chat_url(self, model: str) -> str | None:
        return os.environ.get(self._prefix_env("MAAS_URL"))

    def _prefix_env(self, suffix: str) -> str:
        if self.key_env.endswith("_API_KEY"):
            return self.key_env[: -len("_API_KEY")] + "_" + suffix
        return self.key_env + "_" + suffix

    @staticmethod
    def _normalize_model(model: str | None) -> str:
        return (model or "").lower().replace("_", "-").replace(".", "-").strip()

    @staticmethod
    def _uses_max_completion_tokens(model: str | None) -> bool:
        normalized = (model or "").lower().replace("_", "-")
        return normalized.startswith("gpt-5") or normalized.startswith("o1") or normalized.startswith("o3")

    @staticmethod
    def _raise_runway_error(data: dict[str, Any]) -> None:
        if "Code" in data and "Error" in data:
            raise RuntimeError(f"Runway error Code={data.get('Code')}: {str(data.get('Error'))[:500]}")
        error = data.get("error")
        if error:
            raise RuntimeError(f"Runway error: {str(error)[:500]}")

    @staticmethod
    def _messages_to_input(messages: list[dict[str, str]]) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role.upper()}:\n{content}")
        return "\n\n".join(parts)

    @staticmethod
    def _extract_responses_text(data: dict[str, Any]) -> str:
        if data.get("output_text"):
            return str(data["output_text"])
        texts = []
        for item in data.get("output") or []:
            for content in item.get("content") or []:
                text = content.get("text")
                if text:
                    texts.append(text)
        return "".join(texts)

    @staticmethod
    def _messages_to_anthropic(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], str | None]:
        system_parts = []
        out = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role in {"user", "assistant"}:
                out.append({"role": role, "content": content})
            else:
                out.append({"role": "user", "content": content})
        return out, "\n\n".join(system_parts) if system_parts else None

    @staticmethod
    def _extract_anthropic_text(data: dict[str, Any]) -> str:
        texts = []
        for item in data.get("content") or []:
            if isinstance(item, dict) and item.get("text"):
                texts.append(str(item["text"]))
        return "".join(texts)
