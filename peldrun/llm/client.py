"""
PELDRUN Core Unified LLM Client.
Provides an asynchronous, non-blocking HTTP client for OpenAI-compatible APIs,
supporting local providers (LM Studio, Ollama) and cloud inference endpoints.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

import httpx
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("peldrun.llm.client")


class LLMConfig(BaseModel):
    """Configuration parameters for the unified LLM client."""
    model_config = ConfigDict(extra="ignore")

    api_base: str = Field(
        default="http://localhost:1234/v1",
        description="Base URL for OpenAI-compatible endpoint",
    )
    model: str = Field(
        default="local-model",
        description="Target model identifier",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Optional API authorization key",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum generation token ceiling",
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter",
    )
    timeout: float = Field(
        default=120.0,
        description="HTTP request timeout ceiling in seconds",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts on connection drops or transient errors",
    )


class DeltaToolCall(BaseModel):
    """Chunk of an incremental tool call received during streaming."""
    index: int = Field(default=0)
    id: Optional[str] = Field(default=None)
    name: Optional[str] = Field(default=None)
    arguments: Optional[str] = Field(default=None)


class StreamChunk(BaseModel):
    """Single parsed token or delta fragment from a streaming completion."""
    content: Optional[str] = Field(default=None)
    tool_calls: Optional[List[DeltaToolCall]] = Field(default=None)
    finish_reason: Optional[str] = Field(default=None)
    usage: Optional[Dict[str, int]] = Field(default=None)


class LLMResponse(BaseModel):
    """Complete structured response returned from an LLM inference call."""
    content: Optional[str] = Field(default=None)
    tool_calls: Optional[List[Dict[str, Any]]] = Field(default=None)
    finish_reason: Optional[str] = Field(default=None)
    usage: Optional[Dict[str, int]] = Field(default=None)
    raw: Optional[Dict[str, Any]] = Field(default=None)


class AsyncLLMClient:
    """
    High-performance asynchronous client communicating with OpenAI-compatible backends.
    Supports connection pooling, streaming generation, and structured function calling.
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.config = config or LLMConfig()
        self._transport = transport
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-initialize and return the shared httpx.AsyncClient session."""
        async with self._lock:
            if self._client is None or self._client.is_closed:
                headers = {"Content-Type": "application/json"}
                if self.config.api_key:
                    headers["Authorization"] = f"Bearer {self.config.api_key}"
                else:
                    headers["Authorization"] = "Bearer not-needed"

                self._client = httpx.AsyncClient(
                    base_url=self.config.api_base.rstrip("/"),
                    headers=headers,
                    timeout=httpx.Timeout(self.config.timeout, connect=10.0),
                    limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
                    transport=self._transport,
                )
            return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client session."""
        async with self._lock:
            if self._client is not None and not self._client.is_closed:
                await self._client.aclose()
                self._client = None

    async def __aenter__(self) -> AsyncLLMClient:
        await self._get_client()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    def _prepare_payload(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Construct the standard OpenAI chat completion JSON payload."""
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature if temperature is not None else self.config.temperature,
        }

        if max_tokens is not None or self.config.max_tokens is not None:
            payload["max_tokens"] = max_tokens if max_tokens is not None else self.config.max_tokens

        if self.config.top_p is not None:
            payload["top_p"] = self.config.top_p

        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice

        return payload

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Execute a standard non-streaming chat completion request.
        Includes automatic retry mechanics for transient communication failures.
        """
        client = await self._get_client()
        payload = self._prepare_payload(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        last_exception: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = await client.post("/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()

                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})

                return LLMResponse(
                    content=message.get("content"),
                    tool_calls=message.get("tool_calls"),
                    finish_reason=choice.get("finish_reason"),
                    usage=data.get("usage"),
                    raw=data,
                )
            except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as ex:
                last_exception = ex
                logger.warning(
                    "LLM request attempt %d/%d failed: %s",
                    attempt,
                    self.config.max_retries,
                    str(ex),
                )
                if attempt < self.config.max_retries:
                    await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
                else:
                    break

        raise RuntimeError(
            f"Failed to obtain chat completion after {self.config.max_retries} attempts: {last_exception}"
        )

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream chat completion tokens and tool invocation fragments asynchronously.
        Yields StreamChunk instances until completion.
        """
        client = await self._get_client()
        payload = self._prepare_payload(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async with client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue

                raw_data = line[5:].strip()
                if raw_data == "[DONE]":
                    break

                try:
                    data = json.loads(raw_data)
                except json.JSONDecodeError:
                    continue

                choices = data.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason")

                content_delta = delta.get("content")
                tool_calls_delta = delta.get("tool_calls")

                parsed_tool_calls: Optional[List[DeltaToolCall]] = None
                if tool_calls_delta:
                    parsed_tool_calls = []
                    for tc in tool_calls_delta:
                        parsed_tool_calls.append(
                            DeltaToolCall(
                                index=tc.get("index", 0),
                                id=tc.get("id"),
                                name=tc.get("function", {}).get("name") if "function" in tc else None,
                                arguments=tc.get("function", {}).get("arguments") if "function" in tc else None,
                            )
                        )

                yield StreamChunk(
                    content=content_delta,
                    tool_calls=parsed_tool_calls,
                    finish_reason=finish_reason,
                    usage=data.get("usage"),
                )

    async def test_connection(self) -> bool:
        """Verify reachability and readiness of the configured LLM endpoint."""
        try:
            client = await self._get_client()
            resp = await client.get("/models")
            return resp.status_code == 200
        except Exception as ex:
            logger.debug("LLM endpoint connectivity check failed: %s", ex)
            return False


async def accumulate_stream_chunks(
    stream: AsyncIterator[StreamChunk],
    model: str = "",
) -> Tuple[LLMResponse, List[StreamChunk]]:
    """
    Consumes a stream of StreamChunk items and reconstructs the consolidated LLMResponse.
    Returns a tuple of (LLMResponse, List[StreamChunk]).
    """
    chunks: List[StreamChunk] = []
    content_parts: List[str] = []
    tool_calls_map: Dict[int, Dict[str, Any]] = {}
    finish_reason: Optional[str] = "stop"
    usage: Optional[Dict[str, int]] = None

    async for chunk in stream:
        chunks.append(chunk)

        if chunk.content:
            content_parts.append(chunk.content)

        if chunk.tool_calls:
            for delta in chunk.tool_calls:
                idx = delta.index
                if idx not in tool_calls_map:
                    tool_calls_map[idx] = {
                        "id": delta.id or f"call_{idx}",
                        "type": "function",
                        "function": {
                            "name": delta.name or "",
                            "arguments": "",
                        },
                    }
                else:
                    if delta.id:
                        tool_calls_map[idx]["id"] = delta.id
                    if delta.name:
                        tool_calls_map[idx]["function"]["name"] = delta.name

                if delta.arguments:
                    tool_calls_map[idx]["function"]["arguments"] += delta.arguments

        if chunk.finish_reason:
            finish_reason = chunk.finish_reason

        if chunk.usage:
            usage = chunk.usage

    sorted_indices = sorted(tool_calls_map.keys())
    consolidated_tool_calls = [tool_calls_map[i] for i in sorted_indices] if tool_calls_map else None
    final_content = "".join(content_parts) if content_parts else None

    response = LLMResponse(
        content=final_content,
        tool_calls=consolidated_tool_calls,
        finish_reason=finish_reason,
        usage=usage,
    )
    return response, chunks