"""
PELDRUN Core LM Studio Provider Adapter.
Dedicated adapter for local LM Studio inference (default port 1234),
featuring active model discovery, reasoning token extraction, and context optimization.
"""

from __future__ import annotations

import logging
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

from peldrun.llm.client import AsyncLLMClient, LLMConfig, LLMResponse, StreamChunk
from peldrun.llm.providers.openai_compat import OpenAICompatProvider
from peldrun.llm.tokenizer import ContextBudgetManager

logger = logging.getLogger("peldrun.llm.providers.lmstudio")


class LMStudioProvider(OpenAICompatProvider):
    """
    Dedicated LLM provider adapter optimized for local LM Studio instances.
    Connects to http://localhost:1234/v1 by default and provides automatic model resolution.
    """

    def __init__(
        self,
        api_base: str = "http://localhost:1234/v1",
        model: str = "auto",
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
        self._discovered_model_id: Optional[str] = None

    @property
    def name(self) -> str:
        return "lmstudio"

    async def discover_active_model(self) -> Optional[str]:
        """
        Query the LM Studio local endpoint to detect the currently loaded model ID.
        Returns the resolved model identifier, or None if server is unreachable.
        """
        try:
            http_client = await self.client._get_client()
            response = await http_client.get("/models")
            if response.status_code == 200:
                payload = response.json()
                models = payload.get("data", [])
                if models and len(models) > 0:
                    model_id = models[0].get("id", "local-model")
                    self._discovered_model_id = model_id
                    if self.config.model == "auto":
                        self.config.model = model_id
                        self.budget_manager.model_name = model_id
                    logger.info("Discovered active LM Studio model: %s", model_id)
                    return model_id
        except Exception as ex:
            logger.debug("Failed to auto-discover LM Studio model: %s", ex)
        return None

    async def check_health(self) -> bool:
        """Verify reachability and ensure LM Studio has an active endpoint responding."""
        model_id = await self.discover_active_model()
        return model_id is not None

    @staticmethod
    def extract_thinking_trace(content: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract reasoning/chain-of-thought traces wrapped in <think> tags.
        Returns a tuple of (clean_content, reasoning_thought).
        """
        if not content:
            return None, None

        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if think_match:
            thought = think_match.group(1).strip()
            clean_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return clean_content, thought

        return content, None

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Execute generation with automatic model resolution for LM Studio.
        """
        if self.config.model == "auto" and not self._discovered_model_id:
            await self.discover_active_model()

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
        Stream tokens with automatic model resolution.
        """
        if self.config.model == "auto" and not self._discovered_model_id:
            await self.discover_active_model()

        async for chunk in super().stream(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield chunk