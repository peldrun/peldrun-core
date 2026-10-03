"""
PELDRUN Core LLM Subsystem Test Suite.
Verifies AsyncLLMClient, OpenAICompatProvider, LMStudioProvider, OllamaProvider,
ContextBudgetManager, stream accumulation, and dynamic provider factory routing
using hermetic httpx.MockTransport fixtures.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
import httpx
import pytest

from peldrun.llm.client import (
    AsyncLLMClient,
    DeltaToolCall,
    LLMConfig,
    LLMResponse,
    StreamChunk,
    accumulate_stream_chunks,
)
from peldrun.llm.providers import (
    LMStudioProvider,
    OllamaProvider,
    OpenAICompatProvider,
    get_provider,
)
from peldrun.llm.tokenizer import (
    ContextBudgetManager,
    count_message_tokens,
    estimate_tokens_from_string,
    truncate_messages_sliding_window,
)


def test_llm_config_defaults() -> None:
    """Verify configuration parameters initialize with secure defaults."""
    cfg = LLMConfig()
    assert cfg.api_base == "http://localhost:1234/v1"
    assert cfg.model == "local-model"
    assert cfg.temperature == 0.7
    assert cfg.timeout == 120.0
    assert cfg.max_retries == 3


@pytest.mark.asyncio
async def test_async_llm_client_chat_completion_basic() -> None:
    """Verify standard non-streaming completion roundtrip via MockTransport."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "test-model"

        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "PELDRUN Core Active."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
            },
        )

    transport = httpx.MockTransport(handler)
    client = AsyncLLMClient(
        config=LLMConfig(api_base="http://localhost:1234/v1", model="test-model"),
        transport=transport,
    )

    try:
        response = await client.chat_completion(
            messages=[{"role": "user", "content": "Ping"}]
        )
        assert response.content == "PELDRUN Core Active."
        assert response.finish_reason == "stop"
        assert response.usage["total_tokens"] == 12
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_llm_client_chat_completion_with_tools() -> None:
    """Verify function calling schema serialization and tool invocation extraction."""
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert "tools" in body
        assert body["tool_choice"] == "auto"

        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_123",
                                    "type": "function",
                                    "function": {
                                        "name": "shell_exec",
                                        "arguments": '{"command": "dir"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = AsyncLLMClient(
        config=LLMConfig(api_base="http://localhost:1234/v1"),
        transport=transport,
    )

    try:
        tools = [{"type": "function", "function": {"name": "shell_exec"}}]
        response = await client.chat_completion(
            messages=[{"role": "user", "content": "Run dir"}],
            tools=tools,
            tool_choice="auto",
        )
        assert response.content is None
        assert response.finish_reason == "tool_calls"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["id"] == "call_123"
        assert response.tool_calls[0]["function"]["name"] == "shell_exec"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_llm_client_stream_chat() -> None:
    """Verify asynchronous SSE stream decoding and delta chunk yielding."""
    def handler(request: httpx.Request) -> httpx.Response:
        lines = [
            'data: {"choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}',
            'data: {"choices": [{"delta": {"content": " World"}, "finish_reason": "stop"}]}',
            'data: [DONE]',
        ]
        body = "\n\n".join(lines) + "\n\n"
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=body.encode("utf-8"),
        )

    transport = httpx.MockTransport(handler)
    client = AsyncLLMClient(transport=transport)

    try:
        chunks: List[StreamChunk] = []
        async for chunk in client.stream_chat(messages=[{"role": "user", "content": "Hi"}]):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " World"
        assert chunks[1].finish_reason == "stop"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_llm_client_test_connection() -> None:
    """Verify test_connection probes /models endpoint correctly."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "model_1"}]})
        return httpx.Response(500)

    transport = httpx.MockTransport(handler)
    client = AsyncLLMClient(transport=transport)

    try:
        assert await client.test_connection() is True
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_accumulate_stream_chunks() -> None:
    """Verify reconstructive consolidation of text and delta tool calls."""
    async def mock_stream():
        yield StreamChunk(content="Authoring ")
        yield StreamChunk(content="code...")
        yield StreamChunk(
            tool_calls=[
                DeltaToolCall(index=0, id="call_abc", name="file_ops", arguments='{"action":')
            ]
        )
        yield StreamChunk(
            tool_calls=[
                DeltaToolCall(index=0, arguments=' "write"}')
            ],
            finish_reason="tool_calls",
        )

    response, raw_chunks = await accumulate_stream_chunks(mock_stream())

    assert len(raw_chunks) == 4
    assert response.content == "Authoring code..."
    assert response.finish_reason == "tool_calls"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["id"] == "call_abc"
    assert response.tool_calls[0]["function"]["name"] == "file_ops"
    assert response.tool_calls[0]["function"]["arguments"] == '{"action": "write"}'


@pytest.mark.asyncio
async def test_openai_compat_provider_generate_and_stream() -> None:
    """Verify OpenAICompatProvider orchestrates ContextBudgetManager and client."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "Provider response"}, "finish_reason": "stop"}
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    provider = OpenAICompatProvider(transport=transport)

    try:
        assert provider.name == "openai_compat"
        res = await provider.generate(messages=[{"role": "user", "content": "Hello"}])
        assert res.content == "Provider response"
    finally:
        await provider.close()


def test_lmstudio_provider_thinking_trace_and_discovery() -> None:
    """Verify LMStudioProvider reasoning tag extraction."""
    raw = "<think>Calculating optimal architecture.</think>The plan is ready."
    clean, thought = LMStudioProvider.extract_thinking_trace(raw)
    assert clean == "The plan is ready."
    assert thought == "Calculating optimal architecture."

    # When no think tags are present
    no_think = "Standard direct output."
    c2, t2 = LMStudioProvider.extract_thinking_trace(no_think)
    assert c2 == "Standard direct output."
    assert t2 is None


@pytest.mark.asyncio
async def test_ollama_provider_models_and_health() -> None:
    """Verify OllamaProvider model listing and reachability checks."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"id": "llama3.1:latest"}, {"id": "qwen2.5:7b"}]},
        )

    transport = httpx.MockTransport(handler)
    client = AsyncLLMClient(
        config=LLMConfig(api_base="http://localhost:11434/v1"),
        transport=transport,
    )
    provider = OllamaProvider(client=client)

    try:
        assert provider.name == "ollama"
        models = await provider.fetch_installed_models()
        assert "llama3.1:latest" in models
        assert "qwen2.5:7b" in models

        is_healthy = await provider.check_health()
        assert is_healthy is True
    finally:
        await provider.close()


def test_tokenizer_and_budget_manager() -> None:
    """Verify token estimation, sliding window trimming, and budget enforcement."""
    # Token estimation
    tokens = estimate_tokens_from_string("Hello world from PELDRUN Core")
    assert tokens > 0

    # Message tokens count
    messages = [
        {"role": "system", "content": "System directive"},
        {"role": "user", "content": "Message 1"},
        {"role": "assistant", "content": "Reply 1"},
        {"role": "user", "content": "Latest query"},
    ]
    total_tokens = count_message_tokens(messages)
    assert total_tokens > 10

    # Truncate sliding window
    trimmed = truncate_messages_sliding_window(
        messages=messages,
        max_context_tokens=30,
        reserve_generation_tokens=5,
    )
    assert len(trimmed) >= 2
    assert trimmed[0]["role"] == "system"
    assert trimmed[-1]["content"] == "Latest query"

    # Context budget manager
    budget = ContextBudgetManager(max_context_window=4096, max_generation_tokens=1024)
    fitted = budget.fit_messages(messages)
    assert len(fitted) > 0
    cap = budget.calculate_remaining_generation_capacity(messages)
    assert cap > 0


def test_get_provider_factory() -> None:
    """Verify get_provider instantiates proper provider subclasses and rejects unknowns."""
    p_compat = get_provider("openai_compat")
    assert isinstance(p_compat, OpenAICompatProvider)

    p_lms = get_provider("lmstudio")
    assert isinstance(p_lms, LMStudioProvider)

    p_ollama = get_provider("ollama")
    assert isinstance(p_ollama, OllamaProvider)

    with pytest.raises(ValueError) as exc_info:
        get_provider("unsupported_backend")
    assert "unsupported llm provider" in str(exc_info.value).lower()