"""
PELDRUN ReAct Agent Implementation.
Implements the core think-act loop satisfying BaseAgent abstract contracts.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from peldrun.agents.base import AgentConfig, BaseAgent
from peldrun.events.emitter import EventEmitter
from peldrun.events.schema import EventType, PeldrunEvent
from peldrun.llm.client import AsyncLLMClient
from peldrun.tools.collection import ToolCollection

logger = logging.getLogger(__name__)


class ReActAgent(BaseAgent):
    """Core ReAct Agent partitioning execution steps into think and act phases."""

    def __init__(
        self,
        config: AgentConfig,
        llm: Optional[AsyncLLMClient] = None,
        tool_collection: Optional[ToolCollection] = None,
        emitter: Optional[EventEmitter] = None,
        workspace_dir: Optional[str] = None,
    ) -> None:
        super().__init__(config=config)
        self.llm = llm
        self.tool_collection = tool_collection or ToolCollection()
        self.emitter = emitter or EventEmitter()
        self.workspace_dir = workspace_dir
        self.messages: List[Dict[str, Any]] = []

    def set_system_prompt(self, prompt: str) -> None:
        """Update system instruction prompt."""
        self.config.system_prompt = prompt

    @abstractmethod
    async def think(self) -> Any:
        """Analyze current state and formulate the next action decision."""
        pass

    @abstractmethod
    async def act(self, decision: Any) -> Any:
        """Execute the determined action decision."""
        pass

    async def _astep(self) -> bool:
        """Single execution step satisfying BaseAgent contract."""
        self.current_step += 1
        await self.emitter.emit(
            PeldrunEvent(
                type=EventType.STEP_START,
                step=self.current_step,
                payload={"step": self.current_step, "max_steps": self.config.max_steps},
            )
        )

        decision = await self.think()
        if decision is None:
            return False

        should_continue = await self.act(decision)
        return bool(should_continue)