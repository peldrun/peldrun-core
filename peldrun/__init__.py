"""
PELDRUN Core Autonomous Agent Framework.
Enterprise-grade, standalone agent runtime engine supporting ReAct, Tool Calling,
Multi-Agent Orchestration, Sandboxing, MCP, and Event Streaming.
"""

from __future__ import annotations

__version__ = "0.2.0"

# Agent Architectures
from peldrun.agents.base import BaseAgent, AgentConfig
from peldrun.agents.tool_call_agent import ToolCallAgent
from peldrun.agents.multi.coordinator import MultiAgentCoordinator
from peldrun.agents.multi.protocol import DelegatedTask, DelegationResult

# Tools & Collections
from peldrun.tools.base import BaseTool, ToolResult
from peldrun.tools.collection import ToolCollection
from peldrun.tools.registry import ToolRegistry
from peldrun.tools.builtins.terminate import TerminateTool
from peldrun.tools.builtins.str_replace_editor import StrReplaceEditor
from peldrun.tools.builtins.human_input import HumanInputTool, HumanInputRegistry
from peldrun.tools.builtins.shell_exec import ShellExecTool
from peldrun.tools.builtins.delegate_tool import DelegateTool

# Events & Streaming
from peldrun.events.schema import PeldrunEvent, EventType
from peldrun.events.bus import EventBus
from peldrun.events.emitter import EventEmitter

# Flows & Planning
from peldrun.flows.base import BaseFlow, FlowResult
from peldrun.flows.planning import PlanState, PlanStep, StepStatus
from peldrun.flows.planning_flow import PlanningFlow

# Artifacts & Deliverables
from peldrun.artifacts.manifest import ArtifactManifest, ArtifactRef, ArtifactType
from peldrun.artifacts.manager import ArtifactManager

# Sandbox & Security
from peldrun.sandbox.base import BaseSandbox, SandboxConfig, SandboxResult
from peldrun.sandbox.local_process import LocalProcessSandbox
from peldrun.security.policy import SecurityPolicy, SecurityViolationError

# Memory Management
from peldrun.memory import MemoryManager
from peldrun.memory.short_term import ShortTermMemory, ShortTermMemoryConfig
from peldrun.memory.long_term import LongTermMemory, LongTermMemoryConfig, MemoryEntry

# LLM Client & Providers
from peldrun.llm.client import (
    AsyncLLMClient,
    LLMConfig,
    LLMResponse,
    StreamChunk,
    DeltaToolCall,
)
from peldrun.llm.providers import (
    BaseLLMProvider,
    OpenAICompatProvider,
    LMStudioProvider,
    OllamaProvider,
    get_provider,
)

# Backward-compatibility alias
LLMClient = AsyncLLMClient

__all__ = [
    "__version__",
    # Agents
    "BaseAgent",
    "AgentConfig",
    "ToolCallAgent",
    "MultiAgentCoordinator",
    "DelegatedTask",
    "DelegationResult",
    # Tools
    "BaseTool",
    "ToolResult",
    "ToolCollection",
    "ToolRegistry",
    "TerminateTool",
    "StrReplaceEditor",
    "HumanInputTool",
    "HumanInputRegistry",
    "ShellExecTool",
    "DelegateTool",
    # Events
    "PeldrunEvent",
    "EventType",
    "EventBus",
    "EventEmitter",
    # Flows
    "BaseFlow",
    "FlowResult",
    "PlanState",
    "PlanStep",
    "StepStatus",
    "PlanningFlow",
    # Artifacts
    "ArtifactManifest",
    "ArtifactRef",
    "ArtifactType",
    "ArtifactManager",
    # Sandbox & Security
    "BaseSandbox",
    "SandboxConfig",
    "SandboxResult",
    "LocalProcessSandbox",
    "SecurityPolicy",
    "SecurityViolationError",
    # Memory
    "MemoryManager",
    "ShortTermMemory",
    "ShortTermMemoryConfig",
    "LongTermMemory",
    "LongTermMemoryConfig",
    "MemoryEntry",
    # LLM
    "AsyncLLMClient",
    "LLMClient",
    "LLMConfig",
    "LLMResponse",
    "StreamChunk",
    "DeltaToolCall",
    "BaseLLMProvider",
    "OpenAICompatProvider",
    "LMStudioProvider",
    "OllamaProvider",
    "get_provider",
]