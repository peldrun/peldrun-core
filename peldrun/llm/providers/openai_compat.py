"""
PELDRUN Core OpenAI-Compatible LLM Provider.
Defines the BaseLLMProvider abstract interface and the OpenAICompatProvider
concrete implementation for standard OpenAI-compatible endpoints.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional
import httpx

from peldrun.llm.client import (
    AsyncLLMClient,
    LLMConfig,
    LLMResponse,
    StreamChunk,
)


class BaseLLMProvider(ABC):
    """Abstract base class for all PELDRUN LLM providers."""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        client: Optional[AsyncLLMClient] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        **kwargs: Any,
    ) -> None:
        if config is not None:
            self.config = config
        elif client is not None and hasattr(client, "config") and client.config is not None:
            self.config = client.config
        else:
            self.config = LLMConfig()

        self.name: str = "base"
        self._owned_http_client: Optional[httpx.AsyncClient] = None
        self.client: AsyncLLMClient = client or AsyncLLMClient(
            config=self.config,
            transport=transport,
            **kwargs,
        )

    def _get_http_client(self) -> httpx.AsyncClient:
        """Resolve or lazily initialize the active httpx.AsyncClient."""
        if isinstance(self.client, httpx.AsyncClient):
            return self.client

        for method_name in ["_get_client", "get_client", "_ensure_client"]:
            method = getattr(self.client, method_name, None)
            if callable(method):
                try:
                    c = method()
                    if isinstance(c, httpx.AsyncClient):
                        return c
                except Exception:
                    pass

        for attr in ["_client", "client", "_http_client", "http_client", "_session", "session", "_http"]:
            val = getattr(self.client, attr, None)
            if isinstance(val, httpx.AsyncClient):
                return val

        if self._owned_http_client is not None and not self._owned_http_client.is_closed:
            return self._owned_http_client

        transport = getattr(self.client, "transport", getattr(self.client, "_transport", None))
        base_url = getattr(self.config, "api_base", "http://localhost:1234/v1")
        self._owned_http_client = httpx.AsyncClient(
            base_url=str(base_url),
            transport=transport,
            timeout=getattr(self.config, "timeout", 120.0),
        )

        if hasattr(self.client, "_client"):
            self.client._client = self._owned_http_client

        return self._owned_http_client

    async def close(self) -> None:
        """Gracefully close underlying client sessions."""
        if self.client and hasattr(self.client, "close"):
            await self.client.close()
        if self._owned_http_client is not None and not self._owned_http_client.is_closed:
            await self._owned_http_client.aclose()

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute a non-streaming chat completion request."""
        pass

    @abstractmethod
    async def stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat completion chunks asynchronously."""
        pass


class OpenAICompatProvider(BaseLLMProvider):
    """OpenAI-compatible LLM provider with robust client binding."""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        client: Optional[AsyncLLMClient] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, client=client, transport=transport, **kwargs)
        self.name = "openai_compat"

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute a non-streaming chat completion request."""
        if "model" in kwargs:
            self.config.model = kwargs.pop("model")
            if hasattr(self.client, "config") and self.client.config:
                self.client.config.model = self.config.model
        if "temperature" in kwargs:
            self.config.temperature = kwargs.pop("temperature")
            if hasattr(self.client, "config") and self.client.config:
                self.client.config.temperature = self.config.temperature
        if "max_tokens" in kwargs:
            self.config.max_tokens = kwargs.pop("max_tokens")
            if hasattr(self.client, "config") and self.client.config:
                self.client.config.max_tokens = self.config.max_tokens

        return await self.client.chat_completion(
            messages=messages,
            tools=tools,
            **kwargs,
        )

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat completion chunks asynchronously."""
        if "model" in kwargs:
            self.config.model = kwargs.pop("model")
            if hasattr(self.client, "config") and self.client.config:
                self.client.config.model = self.config.model
        if "temperature" in kwargs:
            self.config.temperature = kwargs.pop("temperature")
            if hasattr(self.client, "config") and self.client.config:
                self.client.config.temperature = self.config.temperature
        if "max_tokens" in kwargs:
            self.config.max_tokens = kwargs.pop("max_tokens")
            if hasattr(self.client, "config") and self.client.config:
                self.client.config.max_tokens = self.config.max_tokens

        async for chunk in self.client.stream_chat(
            messages=messages,
            tools=tools,
            **kwargs,
        ):
            yield chunk


__all__ = ["BaseLLMProvider", "OpenAICompatProvider"]