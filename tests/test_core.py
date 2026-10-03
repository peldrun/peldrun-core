"""
PELDRUN Core End-to-End System Integration Test Suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import pytest

import peldrun
from peldrun.engine.runner import AgentRunner, RunnerConfig
from peldrun.engine.state import ExecutionState, ExecutionStatus, MessageRole
from peldrun.events.emitter import EventEmitter
from peldrun.events.schema import EventType
from peldrun.memory import MemoryManager
from peldrun.tools.builtins import FileOpsTool, ShellExecTool
from peldrun.tools.registry import ToolRegistry
from tests.conftest import CapturedEventEmitter, MockLLMProvider


def test_package_metadata() -> None:
    assert hasattr(peldrun, "__version__")
    assert peldrun.__version__ == "0.1.0"


class _IntegrationAgent:
    """
    Minimal integration agent implementing the StepExecutableAgent protocol.
    Drives the real loop through a mock LLM provider, dispatches tool calls
    via the real `ToolRegistry.aexecute(...)`, and records everything in
    ExecutionState.
    """

    name = "IntegrationAgent"

    def __init__(
        self,
        llm: MockLLMProvider,
        tool_registry: ToolRegistry,
        memory_manager: MemoryManager,
    ) -> None:
        self.llm = llm
        self.tool_registry = tool_registry
        self.memory_manager = memory_manager

    async def step(self, state: ExecutionState, emitter: EventEmitter) -> bool:
        # 1) Query the mock LLM with the current dialogue.
        response = await self.llm.generate(messages=state.get_llm_messages())

        # 2) Record the assistant turn in state and short-term memory.
        state.add_message(
            role=MessageRole.ASSISTANT,
            content=response.content,
            tool_calls=response.tool_calls or None,
        )
        if response.content:
            self.memory_manager.short_term.add_message("assistant", response.content)

        # 3) No tool calls -> task complete.
        if not response.tool_calls:
            state.status = ExecutionStatus.COMPLETED
            state.final_output = response.content or ""
            return True

        # 4) Execute each tool call through the real registry.
        for tc in response.tool_calls:
            func = tc.get("function", {}) or {}
            tool_name = func.get("name", "")
            raw_args = func.get("arguments", "{}")
            try:
                arguments: Dict[str, Any] = json.loads(raw_args)
            except json.JSONDecodeError:
                arguments = {}

            is_error = False
            exit_code = 0
            output: Any = ""
            artifacts: List[str] = []
            try:
                # Correct contract: ToolRegistry exposes `aexecute(name, **kwargs)`.
                result = await self.tool_registry.aexecute(tool_name, **arguments)
                output = getattr(result, "output", result)
                exit_code = getattr(result, "exit_code", 0)
                is_error = getattr(result, "is_error", False)
                artifacts = list(getattr(result, "artifacts", []) or [])
            except Exception as ex:
                output = f"{type(ex).__name__}: {ex}"
                is_error = True
                exit_code = 1

            state.record_tool_execution(
                tool_call_id=tc.get("id", ""),
                tool_name=tool_name,
                arguments=arguments,
                output=output,
                exit_code=exit_code,
                is_error=is_error,
                artifacts=artifacts,
            )
            state.add_message(
                role=MessageRole.TOOL,
                content=str(output),
                tool_call_id=tc.get("id"),
            )
            self.memory_manager.short_term.add_message("tool", str(output))

            # Register produced artifacts as deliverables.
            for art in artifacts:
                if art not in state.deliverables:
                    state.deliverables.append(art)

            # Fallback: `file_ops` writes produce the file path from args.
            if tool_name == "file_ops" and not is_error:
                action = arguments.get("action")
                path = arguments.get("path")
                if action in ("write", "create", "append") and path and path not in state.deliverables:
                    state.deliverables.append(path)

        return False


@pytest.mark.asyncio
async def test_end_to_end_multitool_agent_workflow(
    temp_workspace: Path,
    mock_llm: MockLLMProvider,
    event_collector: CapturedEventEmitter,
) -> None:
    """Full cycle: write a file, then execute it, then synthesize the result."""
    tool_registry = ToolRegistry(workspace_root=str(temp_workspace))
    tool_registry.register(FileOpsTool(workspace_root=str(temp_workspace)))
    tool_registry.register(ShellExecTool(workspace_root=str(temp_workspace)))

    memory_manager = MemoryManager(
        workspace_root=str(temp_workspace),
        system_prompt="You are PELDRUN Core.",
    )

    # Turn 1: write file
    mock_llm.queue_response(
        content="Writing hello.py...",
        tool_calls=[{
            "id": "call_write_1",
            "type": "function",
            "function": {
                "name": "file_ops",
                "arguments": json.dumps({
                    "action": "write",
                    "path": "hello.py",
                    "content": "print('PELDRUN_CORE_INTEGRATION_SUCCESS')",
                }),
            },
        }],
    )
    # Turn 2: execute file
    mock_llm.queue_response(
        content="Executing hello.py...",
        tool_calls=[{
            "id": "call_exec_2",
            "type": "function",
            "function": {
                "name": "shell_exec",
                "arguments": json.dumps({"command": "python hello.py"}),
            },
        }],
    )
    # Turn 3: done
    mock_llm.queue_response(content="Completed successfully.")

    agent = _IntegrationAgent(mock_llm, tool_registry, memory_manager)
    runner = AgentRunner(
        agent=agent,
        emitter=event_collector,
        config=RunnerConfig(max_steps=10, enable_checkpointing=False),
    )

    state = await runner.run(
        task_prompt="Create hello.py and run it.",
        workspace_root=str(temp_workspace),
    )

    # 1) Task completed with the expected number of steps.
    assert state.status == ExecutionStatus.COMPLETED
    assert state.current_step == 3

    # 2) Physical file created on disk.
    created_file = temp_workspace / "hello.py"
    assert created_file.exists()
    assert "PELDRUN_CORE_INTEGRATION_SUCCESS" in created_file.read_text(encoding="utf-8")
    assert "hello.py" in state.deliverables

    # 3) Tool history has both invocations with no errors.
    assert len(state.tool_history) == 2
    assert state.tool_history[0].tool_name == "file_ops"
    assert state.tool_history[0].is_error is False
    assert state.tool_history[1].tool_name == "shell_exec"
    assert state.tool_history[1].is_error is False

    # 4) Canonical event sequence.
    event_types = [e.type for e in event_collector.captured_events]
    assert EventType.SNAPSHOT in event_types
    assert EventType.STEP_START in event_types
    assert EventType.STEP_END in event_types


@pytest.mark.asyncio
async def test_end_to_end_error_recovery_workflow(
    temp_workspace: Path,
    mock_llm: MockLLMProvider,
    event_collector: CapturedEventEmitter,
) -> None:
    """Agent survives a tool failure and produces a diagnostic conclusion."""
    tool_registry = ToolRegistry(workspace_root=str(temp_workspace))
    tool_registry.register(ShellExecTool(workspace_root=str(temp_workspace)))
    memory_manager = MemoryManager(workspace_root=str(temp_workspace))

    mock_llm.queue_response(
        content="Attempting failing command.",
        tool_calls=[{
            "id": "err_call_1",
            "type": "function",
            "function": {
                "name": "shell_exec",
                "arguments": json.dumps({
                    "command": 'python -c "import sys; sys.exit(42)"',
                }),
            },
        }],
    )
    mock_llm.queue_response(content="Diagnosed exit code 42. Task concluded.")

    agent = _IntegrationAgent(mock_llm, tool_registry, memory_manager)
    runner = AgentRunner(
        agent=agent,
        emitter=event_collector,
        config=RunnerConfig(max_steps=5, enable_checkpointing=False),
    )

    state = await runner.run(
        task_prompt="Run error diagnostic test.",
        workspace_root=str(temp_workspace),
    )

    assert state.status == ExecutionStatus.COMPLETED
    assert len(state.tool_history) == 1
    # The command exits with code 42 -> is_error=True and exit_code=42.
    assert state.tool_history[0].is_error is True
    assert state.tool_history[0].exit_code == 42