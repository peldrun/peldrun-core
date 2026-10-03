"""
PELDRUN Core Pytest Configuration and Global Fixtures.
Provides mock LLM providers, isolated temporary workspaces, event capture collectors,
and pre-configured tool registries for integration testing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional
import pytest

from peldrun.engine.state import ExecutionState
from peldrun.events.emitter import EventEmitter
from peldrun.events.schema import AgentEvent
from peldrun.llm.client import DeltaToolCall, LLMResponse, StreamChunk
from peldrun.llm.providers.openai_compat import BaseLLMProvider
from peldrun.memory import MemoryManager
from peldrun.sandbox.local_process import LocalProcessSandbox
from peldrun.tools.builtins import FileOpsTool, ShellExecTool
from peldrun.tools.registry import ToolRegistry


class MockLLMProvider(BaseLLMProvider):
    """
    Simulated LLM provider for unit and integration testing.
    Replays pre-configured responses, tool calls, and reasoning streams without network overhead.
    """

    def __init__(self, model_name: str = "mock-model") -> None:
        super().__init__(model_name=model_name, api_key="mock-key")
        self._canned_responses: List[LLMResponse] = []
        self._invocation_history: List[Dict[str, Any]] = []

    def queue_response(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        finish_reason: str = "stop",
    ) -> None:
        """Enqueue a canned response to be returned by next generate or stream call."""
        resp = LLMResponse(
            content=content,
            tool_calls=tool_calls or [],
            finish_reason=finish_reason,
            model=self.model_name,
            usage={"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
        )
        self._canned_responses.append(resp)

    @property
    def invocations(self) -> List[Dict[str, Any]]:
        """Return history of parameters passed to generate/stream."""
        return list(self._invocation_history)

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Return next queued canned response."""
        self._invocation_history.append({
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "kwargs": kwargs,
        })

        if self._canned_responses:
            return self._canned_responses.pop(0)

        return LLMResponse(
            content="Mock automated completion.",
            finish_reason="stop",
            model=self.model_name,
        )

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chunks matching the next queued canned response."""
        resp = await self.generate(
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Stream reasoning / content if present
        if resp.content:
            words = resp.content.split(" ")
            for i, word in enumerate(words):
                chunk_text = word + (" " if i < len(words) - 1 else "")
                yield StreamChunk(content=chunk_text)

        # Stream tool calls if present
        if resp.tool_calls:
            for idx, tc in enumerate(resp.tool_calls):
                func = tc.get("function", {})
                yield StreamChunk(
                    tool_calls=[
                        DeltaToolCall(
                            index=idx,
                            id=tc.get("id", f"call_{idx}"),
                            name=func.get("name", ""),
                            arguments=func.get("arguments", "{}"),
                        )
                    ]
                )

        yield StreamChunk(finish_reason=resp.finish_reason)


class CapturedEventEmitter(EventEmitter):
    """
    Extension of EventEmitter capturing all broadcast events in memory for test assertions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.captured_events: List[AgentEvent] = []

        async def _capture_listener(event: AgentEvent) -> None:
            self.captured_events.append(event)

        self.subscribe_all(_capture_listener)

    def get_events_by_type(self, event_type: str) -> List[AgentEvent]:
        """Filter captured events by their type identifier."""
        return [e for e in self.captured_events if e.type == event_type]


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Provide an isolated filesystem workspace for sandbox and tool execution."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """Provide a fresh mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def event_collector() -> CapturedEventEmitter:
    """Provide an EventEmitter that retains an inspection log of all fired events."""
    return CapturedEventEmitter()


@pytest.fixture
def tool_registry(temp_workspace: Path) -> ToolRegistry:
    """Provide a tool registry pre-loaded with workspace-scoped built-in tools."""
    registry = ToolRegistry(workspace_root=str(temp_workspace))
    registry.register(FileOpsTool(workspace_root=str(temp_workspace)))
    registry.register(ShellExecTool(workspace_root=str(temp_workspace)))
    return registry


@pytest.fixture
def memory_manager(temp_workspace: Path) -> MemoryManager:
    """Provide a MemoryManager instance bound to the isolated temporary workspace."""
    return MemoryManager(
        workspace_root=str(temp_workspace),
        system_prompt="You are a test agent.",
    )


@pytest.fixture
def local_sandbox(temp_workspace: Path) -> LocalProcessSandbox:
    """Provide a LocalProcessSandbox instance bound to the isolated workspace."""
    sandbox = LocalProcessSandbox()
    sandbox.set_workspace_root(str(temp_workspace))
    return sandbox