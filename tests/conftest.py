"""
PELDRUN Core Pytest Configuration and Global Fixtures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union
import pytest

from peldrun.events.emitter import EventEmitter
from peldrun.events.schema import AgentEvent
from peldrun.llm.client import DeltaToolCall, LLMResponse, StreamChunk
from peldrun.llm.providers.openai_compat import BaseLLMProvider
from peldrun.memory import MemoryManager
from peldrun.sandbox.local_process import LocalProcessSandbox
from peldrun.tools.builtins import FileOpsTool, ShellExecTool
from peldrun.tools.registry import ToolRegistry


class MockLLMProvider(BaseLLMProvider):
    """Simulated LLM provider for tests.

    IMPORTANT: `BaseLLMProvider` inherits directly from `object` and does NOT
    define `__init__`. We must NOT call `super().__init__(model_name=..., ...)`.
    """

    def __init__(self, model_name: str = "mock-model") -> None:
        self.model_name = model_name
        self._canned_responses: List[LLMResponse] = []
        self._invocation_history: List[Dict[str, Any]] = []

    # --------------------------------------------------------------- contract
    @property
    def name(self) -> str:
        return "mock_provider"

    async def check_health(self) -> bool:
        return True

    async def close(self) -> None:
        return None

    # ----------------------------------------------------------- test helpers
    def queue_response(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        finish_reason: str = "stop",
    ) -> None:
        self._canned_responses.append(
            LLMResponse(
                content=content,
                tool_calls=tool_calls or [],
                finish_reason=finish_reason,
                model=self.model_name,
                usage={"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
            )
        )

    @property
    def invocations(self) -> List[Dict[str, Any]]:
        return list(self._invocation_history)

    # ------------------------------------------------------------------ base
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self._invocation_history.append(
            {
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "kwargs": kwargs,
            }
        )
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
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        resp = await self.generate(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        if resp.content:
            words = resp.content.split(" ")
            for i, word in enumerate(words):
                yield StreamChunk(content=word + (" " if i < len(words) - 1 else ""))
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
    """EventEmitter that retains an in-memory log of all fired events.

    NOTE: `AgentRunner.run()` reads/writes `emitter.run_id`, so we must expose it.
    """

    def __init__(self) -> None:
        super().__init__()
        # `runner.py` requires this attribute to exist.
        self.run_id: Optional[str] = None
        self.captured_events: List[AgentEvent] = []

        async def _capture_listener(event: AgentEvent) -> None:
            self.captured_events.append(event)

        self.subscribe_all(_capture_listener)

    def get_events_by_type(self, event_type: str) -> List[AgentEvent]:
        target = event_type.value if hasattr(event_type, "value") else str(event_type)
        return [
            e
            for e in self.captured_events
            if (e.type.value if hasattr(e.type, "value") else str(e.type)) == target
        ]


# --------------------------------------------------------------------- fixtures
@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    return MockLLMProvider()


@pytest.fixture
def event_collector() -> CapturedEventEmitter:
    return CapturedEventEmitter()


@pytest.fixture
def tool_registry(temp_workspace: Path) -> ToolRegistry:
    registry = ToolRegistry(workspace_root=str(temp_workspace))
    registry.register(FileOpsTool(workspace_root=str(temp_workspace)))
    registry.register(ShellExecTool(workspace_root=str(temp_workspace)))
    return registry


@pytest.fixture
def memory_manager(temp_workspace: Path) -> MemoryManager:
    return MemoryManager(
        workspace_root=str(temp_workspace),
        system_prompt="You are a test agent.",
    )


@pytest.fixture
def local_sandbox(temp_workspace: Path) -> LocalProcessSandbox:
    sandbox = LocalProcessSandbox()
    sandbox.set_workspace_root(str(temp_workspace))
    return sandbox