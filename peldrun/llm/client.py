"""
PELDRUN Core Asynchronous LLM Client.
Unified OpenAI-compatible asynchronous HTTP client supporting streaming, function calling,
and resilient error logging and auto-healing retries.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

import httpx

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration parameters for LLM client connectivity and model execution."""
    api_base: str = "http://localhost:1234/v1"
    model: str = "local-model"
    api_key: str = "EMPTY"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 120.0
    max_retries: int = 3
    provider: str = "openai_compat"
    base_url: Optional[str] = None

    def __post_init__(self) -> None:
        if self.base_url and not self.api_base:
            self.api_base = self.base_url
        elif self.api_base and not self.base_url:
            self.base_url = self.api_base


@dataclass
class DeltaToolCall:
    """Incremental tool call fragment received during streaming."""
    index: int = 0
    id: Optional[str] = None
    name: Optional[str] = None
    arguments: str = ""


@dataclass
class StreamChunk:
    """Single decoded chunk yielded from LLM stream."""
    content: Optional[str] = None
    finish_reason: Optional[str] = None
    tool_calls: List[DeltaToolCall] = field(default_factory=list)


@dataclass
class LLMResponse:
    """Consolidated completion response returned from LLM generation."""
    content: Optional[str] = None
    finish_reason: Optional[str] = "stop"
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    thought: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)


async def accumulate_stream_chunks(
    stream: AsyncIterator[StreamChunk],
) -> Tuple[LLMResponse, List[StreamChunk]]:
    """Consolidate incremental stream chunks into a unified LLMResponse."""
    raw_chunks: List[StreamChunk] = []
    content_parts: List[str] = []
    finish_reason: Optional[str] = None
    tool_calls_map: Dict[int, Dict[str, Any]] = {}

    async for chunk in stream:
        raw_chunks.append(chunk)
        if chunk.content:
            content_parts.append(chunk.content)
        if chunk.finish_reason:
            finish_reason = chunk.finish_reason
        for tc in chunk.tool_calls:
            idx = tc.index
            if idx not in tool_calls_map:
                tool_calls_map[idx] = {
                    "id": tc.id or "",
                    "type": "function",
                    "function": {"name": tc.name or "", "arguments": ""},
                }
            if tc.id:
                tool_calls_map[idx]["id"] = tc.id
            if tc.name:
                tool_calls_map[idx]["function"]["name"] = tc.name
            if tc.arguments:
                tool_calls_map[idx]["function"]["arguments"] += tc.arguments

    final_content = "".join(content_parts) if content_parts else None
    sorted_calls = [tool_calls_map[k] for k in sorted(tool_calls_map.keys())]

    return LLMResponse(
        content=final_content,
        finish_reason=finish_reason or "stop",
        tool_calls=sorted_calls,
    ), raw_chunks


class AsyncLLMClient:
    """Asynchronous HTTP client managing LLM API interactions with adaptive fallbacks."""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.config: LLMConfig = config or LLMConfig()
        self.transport: Optional[httpx.AsyncBaseTransport] = transport
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        """Resolve or lazily initialize the underlying httpx.AsyncClient."""
        if self._client is None or self._client.is_closed:
            base_url = self.config.base_url or self.config.api_base
            self._client = httpx.AsyncClient(
                base_url=str(base_url),
                transport=self.transport,
                timeout=self.config.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the active HTTP client session."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def test_connection(self) -> bool:
        """Probe model discovery endpoint to verify provider connectivity."""
        client = self._get_client()
        endpoints = ["/v1/models", "/models", "models"]
        for ep in endpoints:
            try:
                resp = await client.get(ep)
                if resp.status_code == 200:
                    return True
            except Exception:
                continue
        return False

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute a non-streaming chat completion request with adaptive 422 recovery."""
        client = self._get_client()
        endpoint = "/v1/chat/completions" if "/v1" not in str(client.base_url) else "/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.config.api_key and self.config.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        payload.update(kwargs)

        last_err: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                resp = await client.post(endpoint, json=payload, headers=headers)
                if resp.status_code >= 400:
                    err_body = resp.text
                    print(f"[LLM ERROR {resp.status_code}] URL: {resp.url} | Details: {err_body}")

                    # Adaptive Auto-Healing for 422 Unprocessable Entity
                    if resp.status_code == 422 and attempt < self.config.max_retries:
                        if "tool_choice" in payload:
                            print("[LLM RECOVERY] Removing 'tool_choice' parameter and retrying...")
                            payload.pop("tool_choice", None)
                            continue
                        if "tools" in payload:
                            print("[LLM RECOVERY] Removing 'tools' parameter and retrying...")
                            payload.pop("tools", None)
                            continue
                        if payload.get("max_tokens", 0) > 4096:
                            print("[LLM RECOVERY] Clamping 'max_tokens' to 2048 and retrying...")
                            payload["max_tokens"] = 2048
                            continue

                resp.raise_for_status()
                data = resp.json()

                choice = data["choices"][0]
                msg = choice.get("message", {})
                content = msg.get("content")
                finish_reason = choice.get("finish_reason", "stop")
                tool_calls = msg.get("tool_calls", [])
                usage = data.get("usage", {})

                return LLMResponse(
                    content=content,
                    finish_reason=finish_reason,
                    tool_calls=tool_calls,
                    usage=usage,
                )
            except Exception as err:
                last_err = err
                logger.warning(f"LLM request attempt {attempt}/{self.config.max_retries} failed: {err}")

        raise RuntimeError(
            f"Failed to obtain chat completion after {self.config.max_retries} attempts: {last_err}"
        )

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat completion chunks asynchronously."""
        client = self._get_client()
        endpoint = "/v1/chat/completions" if "/v1" not in str(client.base_url) else "/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.config.api_key and self.config.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        payload.update(kwargs)

        async with client.stream("POST", endpoint, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                raw_data = line[len("data: "):].strip()
                if raw_data == "[DONE]":
                    break
                try:
                    chunk_json = json.loads(raw_data)
                    choice = chunk_json["choices"][0]
                    delta = choice.get("delta", {})
                    content = delta.get("content")
                    finish_reason = choice.get("finish_reason")
                    tc_deltas: List[DeltaToolCall] = []

                    if "tool_calls" in delta:
                        for raw_tc in delta["tool_calls"]:
                            tc_deltas.append(
                                DeltaToolCall(
                                    index=raw_tc.get("index", 0),
                                    id=raw_tc.get("id"),
                                    name=raw_tc.get("function", {}).get("name"),
                                    arguments=raw_tc.get("function", {}).get("arguments", ""),
                                )
                            )

                    yield StreamChunk(
                        content=content,
                        finish_reason=finish_reason,
                        tool_calls=tc_deltas,
                    )
                except Exception:
                    continue


__all__ = [
    "LLMConfig",
    "StreamChunk",
    "DeltaToolCall",
    "LLMResponse",
    "AsyncLLMClient",
    "accumulate_stream_chunks",
]