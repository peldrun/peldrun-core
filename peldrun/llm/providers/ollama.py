"""
PELDRUN Core Ollama Provider Adapter.
Dedicated adapter for local Ollama server (default port 11434),
featuring local model catalog discovery, keep-alive options, and context management.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from peldrun.llm.client import AsyncLLMClient, LLMConfig, LLMResponse, StreamChunk
from peldrun.llm.providers.openai_compat import OpenAICompatProvider
from peldrun.llm.tokenizer import ContextBudgetManager

logger = logging.getLogger("peldrun.llm.providers.ollama")


class OllamaProvider(OpenAICompatProvider):
    """
    Dedicated LLM provider adapter optimized for Ollama local inference.
    Connects to http://localhost:11434/v1 by default with auto-discovery of installed tags.
    """

    def __init__(
        self,
        api_base: str = "http://localhost:11434/v1",
        model: str = "llama3.1",
        max_context_window: int = 8192,
        max_generation_tokens: int = 2048,
        temperature: float = 0.7,
        timeout: float = 180.0,
        client: Optional[AsyncLLMClient] = None,
    ) -> None:
        config = LLMConfig(
            api_base=api_base,
            model=model,
            temperature=temperature,
            max_tokens=max_generation_tokens,
            timeout=timeout,
        )
        budget_manager = ContextBudgetManager(
            max_context_window=max_context_window,
            max_generation_tokens=max_generation_tokens,
            model_name=model,
        )
        super().__init__(config=config, budget_manager=budget_manager, client=client)
        self._available_models: List[str] = []

    @property
    def name(self) -> str:
        return "ollama"

    async def fetch_installed_models(self) -> List[str]:
        """
        Query Ollama endpoint to retrieve the list of pulled models.
        Supports both /v1/models (OpenAI-compatible) and /api/tags (native).
        """
        installed: List[str] = []
        try:
            http_client = await self.client._get_client()
            response = await http_client.get("/models")
            if response.status_code == 200:
                data = response.json()
                for item in data.get("data", []):
                    model_id = item.get("id")
                    if model_id:
                        installed.append(model_id)
            else:
                native_tags_url = self.config.api_base.replace("/v1", "") + "/api/tags"
                resp = await http_client.get(native_tags_url)
                if resp.status_code == 200:
                    data = resp.json()
                    for model_obj in data.get("models", []):
                        m_name = model_obj.get("name")
                        if m_name:
                            installed.append(m_name)
        except Exception as ex:
            logger.debug("Failed to query Ollama installed models: %s", ex)

        self._available_models = installed
        return installed

    async def check_health(self) -> bool:
        """
        Verify reachability and readiness of the Ollama server.
        Returns True if server responds and has at least one active or installed model.
        """
        models = await self.fetch_installed_models()
        if models:
            if self.config.model == "auto" or self.config.model not in models:
                self.config.model = models[0]
                self.budget_manager.model_name = models[0]
                logger.info("Defaulting Ollama model to available model: %s", self.config.model)
            return True

        return await self.client.test_connection()

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Execute completion via Ollama OpenAI-compatible endpoint with automatic fallback.
        """
        if self.config.model == "auto" and not self._available_models:
            await self.fetch_installed_models()

        return await super().generate(
            messages=messages,
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
        """
        Stream completion tokens via Ollama OpenAI-compatible endpoint.
        """
        if self.config.model == "auto" and not self._available_models:
            await self.fetch_installed_models()

        async for chunk in super().stream(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield chunk