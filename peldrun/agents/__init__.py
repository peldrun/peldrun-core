"""
PELDRUN Core Agents Subsystem.
Exports base agent contracts, specialized agent implementations (ReAct, Planning, Coding),
milestone planning data models, and a centralized agent factory.
"""

from __future__ import annotations

from typing import Any, Optional

from peldrun.agents.base import AgentConfig, BaseAgent
from peldrun.agents.coding_agent import CodingAgent, CodingAgentConfig
from peldrun.agents.planning_agent import (
    Plan,
    PlanningAgent,
    PlanningAgentConfig,
    PlanStep,
    StepStatus,
)
from peldrun.agents.react_agent import ReActAgent, ReActAgentConfig
from peldrun.events.emitter import EventEmitter
from peldrun.llm.providers.openai_compat import BaseLLMProvider
from peldrun.memory import MemoryManager
from peldrun.sandbox.base import BaseSandbox
from peldrun.tools.registry import ToolRegistry


def get_agent(
    agent_type: str = "react",
    config: Optional[AgentConfig] = None,
    llm_provider: Optional[BaseLLMProvider] = None,
    tool_registry: Optional[ToolRegistry] = None,
    memory_manager: Optional[MemoryManager] = None,
    sandbox: Optional[BaseSandbox] = None,
    emitter: Optional[EventEmitter] = None,
    **kwargs: Any,
) -> BaseAgent:
    """
    Factory function to instantiate an autonomous agent by identifier.
    Supports 'react', 'planning', and 'coding'.
    """
    key = agent_type.lower().strip()

    if key in ("react", "default", "reasoning"):
        cfg = (
            config
            if isinstance(config, ReActAgentConfig)
            else ReActAgentConfig(**kwargs) if kwargs else (config or ReActAgentConfig())
        )
        return ReActAgent(
            config=cfg,
            llm_provider=llm_provider,
            tool_registry=tool_registry,
            memory_manager=memory_manager,
            sandbox=sandbox,
            emitter=emitter,
        )

    elif key in ("plan", "planning", "planner"):
        cfg = (
            config
            if isinstance(config, PlanningAgentConfig)
            else PlanningAgentConfig(**kwargs) if kwargs else (config or PlanningAgentConfig())
        )
        return PlanningAgent(
            config=cfg,
            llm_provider=llm_provider,
            tool_registry=tool_registry,
            memory_manager=memory_manager,
            sandbox=sandbox,
            emitter=emitter,
        )

    elif key in ("code", "coding", "software_engineer"):
        cfg = (
            config
            if isinstance(config, CodingAgentConfig)
            else CodingAgentConfig(**kwargs) if kwargs else (config or CodingAgentConfig())
        )
        return CodingAgent(
            config=cfg,
            llm_provider=llm_provider,
            tool_registry=tool_registry,
            memory_manager=memory_manager,
            sandbox=sandbox,
            emitter=emitter,
        )

    else:
        raise ValueError(
            f"Unsupported agent type identifier: '{agent_type}'. "
            "Expected 'react', 'planning', or 'coding'."
        )


__all__ = [
    # Base abstractions
    "BaseAgent",
    "AgentConfig",
    # ReAct
    "ReActAgent",
    "ReActAgentConfig",
    # Planning
    "PlanningAgent",
    "PlanningAgentConfig",
    "Plan",
    "PlanStep",
    "StepStatus",
    # Coding
    "CodingAgent",
    "CodingAgentConfig",
    # Factory
    "get_agent",
]