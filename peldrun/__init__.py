"""
PELDRUN Core Agent Engine.

Production-grade autonomous agent framework and runtime.
Provides typed execution graphs, reactive agents, sandboxed tool
execution, unified event streaming, and checkpoint persistence.
"""

from typing import List

from peldrun.agents.base import AgentConfig, BaseAgent
from peldrun.agents.coding_agent import CodingAgent
from peldrun.agents.planning_agent import PlanningAgent
from peldrun.agents.react_agent import ReActAgent, ReActAgentConfig

# Aliases for backward compatibility and naming ergonomics
ReactAgent = ReActAgent
ReactAgentConfig = ReActAgentConfig

from peldrun.engine.runner import AgentRunner
from peldrun.engine.state import ExecutionState
from peldrun.events.emitter import EventEmitter
from peldrun.sandbox.base import BaseSandbox
from peldrun.sandbox.local_process import LocalProcessSandbox
from peldrun.security.policy import (
    ActionRiskLevel,
    SecurityPolicy,
    SecurityViolationError,
    validate_workspace_path,
)
from peldrun.tools.base import BaseTool, ToolResult
from peldrun.tools.registry import ToolRegistry

try:
    from peldrun.engine.graph import ExecutionGraph
except ImportError:
    ExecutionGraph = None  # type: ignore

try:
    from peldrun.engine.checkpoint import CheckpointManager
except ImportError:
    try:
        from peldrun.engine.checkpoint import Checkpoint as CheckpointManager  # type: ignore
    except ImportError:
        CheckpointManager = None  # type: ignore

try:
    from peldrun.events.schema import EventType, PeldrunEvent
except ImportError:
    try:
        from peldrun.events.schema import EventType, Event as PeldrunEvent  # type: ignore
    except ImportError:
        EventType = None  # type: ignore
        PeldrunEvent = None  # type: ignore

try:
    from peldrun.llm.client import LLMClient
except ImportError:
    LLMClient = None  # type: ignore

try:
    from peldrun.sandbox.docker_sandbox import DockerSandbox
except ImportError:
    DockerSandbox = None  # type: ignore

__version__ = "0.1.0"

__all__: List[str] = [
    "__version__",
    # Agents
    "BaseAgent",
    "AgentConfig",
    "ReActAgent",
    "ReactAgent",
    "ReActAgentConfig",
    "ReactAgentConfig",
    "CodingAgent",
    "PlanningAgent",
    # Engine & Execution
    "AgentRunner",
    "ExecutionState",
    "ExecutionGraph",
    "CheckpointManager",
    # Events Protocol
    "PeldrunEvent",
    "EventType",
    "EventEmitter",
    # Security & Governance
    "SecurityPolicy",
    "SecurityViolationError",
    "ActionRiskLevel",
    "validate_workspace_path",
    # Tools Subsystem
    "BaseTool",
    "ToolResult",
    "ToolRegistry",
    # LLM Gateway
    "LLMClient",
    # Sandbox Infrastructure
    "BaseSandbox",
    "LocalProcessSandbox",
    "DockerSandbox",
]