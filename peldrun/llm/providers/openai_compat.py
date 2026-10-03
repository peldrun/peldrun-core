"""
PELDRUN Core OpenAI-Compatible Provider Adapter.
Defines the base LLM provider interface and the standard OpenAI protocol implementation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from peldrun.llm.client import AsyncLLMClient, LLMConfig, LLMResponse, StreamChunk
from peldrun.llm.tokenizer import ContextBudgetManager

logger = logging.getLogger("peldrun.llm.providers.openai_compat")


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM inference providers.
    Enforces unified interfaces for non-streaming completions, streaming responses,
    and connectivity health checks.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier string."""
        ...

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Execute a non-streaming completion request."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream completion tokens and tool invocation fragments asynchronously."""
        ...

    @abstractmethod
    async def check_health(self) -> bool:
        """Verify reachability and readiness of the provider endpoint."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release underlying HTTP connections and provider resources."""
        ...


class OpenAICompatProvider(BaseLLMProvider):
    """
    Standard OpenAI-compatible provider adapter.
    Operates against local servers (LM Studio, Ollama, vLLM) and compatible cloud endpoints.
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        budget_manager: Optional[ContextBudgetManager] = None,
        client: Optional[AsyncLLMClient] = None,
    ) -> None:
        self.config = config or LLMConfig()
        self.budget_manager = budget_manager or ContextBudgetManager(
            max_context_window=8192,
            max_generation_tokens=self.config.max_tokens or 2048,
            model_name=self.config.model,
        )
        self.client = client or AsyncLLMClient(config=self.config)

    @property
    def name(self) -> str:
        return "openai_compat"

    def _prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fit messages inside the allowed context window before dispatching."""
        return self.budget_manager.fit_messages(messages)

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Execute a non-streaming chat completion with budgeted messages."""
        fitted_messages = self._prepare_messages(messages)
        return await self.client.chat_completion(
            messages=fitted_messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream tokens asynchronously with budgeted messages."""
        fitted_messages = self._prepare_messages(messages)
        async for chunk in self.client.stream_chat(
            messages=fitted_messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield chunk

    async def check_health(self) -> bool:
        """Verify reachability of the endpoint."""
        return await self.client.test_connection()

    async def close(self) -> None:
        """Close client connection pool."""
        await self.client.close()