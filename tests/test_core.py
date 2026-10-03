"""
PELDRUN Core End-to-End System Integration Test Suite.
Verifies complete multi-step autonomous workflows integrating AgentRunner,
ReActAgent, scoped tools (file_ops, shell_exec), MemoryManager, and event streams.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List
import pytest

import peldrun
from peldrun.engine.runner import AgentRunner, RunnerConfig
from peldrun.engine.state import ExecutionState
from peldrun.events.schema import EventType
from peldrun.memory import MemoryManager
from peldrun.sandbox.local_process import LocalProcessSandbox
from peldrun.tools.builtins import FileOpsTool, ShellExecTool
from peldrun.tools.registry import ToolRegistry
from tests.conftest import CapturedEventEmitter, MockLLMProvider


def test_package_metadata() -> None:
    """Verify package importability and published version string."""
    assert hasattr(peldrun, "__version__")
    assert peldrun.__version__ == "0.1.0"


@pytest.mark.asyncio
async def test_end_to_end_multitool_agent_workflow(
    temp_workspace: Path,
    mock_llm: MockLLMProvider,
    event_collector: CapturedEventEmitter,
) -> None:
    """
    Execute full end-to-end multi-step cycle:
    1. Step 1: Model reasons, invokes file_ops to write hello.py.
    2. Step 2: Model observes result, invokes shell_exec to run hello.py.
    3. Step 3: Model evaluates execution output and delivers final synthesis.
    """
    # 1. Setup isolated subsystems
    tool_registry = ToolRegistry(workspace_root=str(temp_workspace))
    tool_registry.register(FileOpsTool(workspace_root=str(temp_workspace)))
    tool_registry.register(ShellExecTool(workspace_root=str(temp_workspace)))

    memory_manager = MemoryManager(
        workspace_root=str(temp_workspace),
        system_prompt="You are PELDRUN Core, an autonomous software engineering engine.",
    )

    # 2. Queue simulated multi-step agent dialogue turns
    # Turn 1: Write python script
    mock_llm.queue_response(
        content="<think>I need to create a python greeting script first.</think>Writing hello.py...",
        tool_calls=[{
            "id": "call_write_1",
            "type": "function",
            "function": {
                "name": "file_ops",
                "arguments": '{"action": "write", "path": "hello.py", "content": "print(\'PELDRUN_CORE_INTEGRATION_SUCCESS\')"}',
            },
        }],
    )

    # Turn 2: Execute python script
    mock_llm.queue_response(
        content="<think>Now I must execute the created script to verify output.</think>Executing hello.py...",
        tool_calls=[{
            "id": "call_exec_2",
            "type": "function",
            "function": {
                "name": "shell_exec",
                "arguments": '{"command": "python hello.py"}',
            },
        }],
    )

    # Turn 3: Synthesize final output
    mock_llm.queue_response(
        content="The script was created and verified successfully. Output confirmed: PELDRUN_CORE_INTEGRATION_SUCCESS."
    )

    # 3. Instantiate and run orchestrator
    runner = AgentRunner(
        config=RunnerConfig(
            max_steps=10,
            workspace_root=str(temp_workspace),
            agent_type="react",
        ),
        llm_provider=mock_llm,
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        emitter=event_collector,
    )

    state = await runner.arun_task("Create hello.py and execute it to verify output.")

    # 4. Assert execution outcome
    assert state.is_completed is True
    assert not state.is_error
    assert state.step == 3
    assert "PELDRUN_CORE_INTEGRATION_SUCCESS" in str(state.final_output)

    # 5. Assert physical artifacts on host filesystem
    created_file = temp_workspace / "hello.py"
    assert created_file.exists()
    assert "PELDRUN_CORE_INTEGRATION_SUCCESS" in created_file.read_text(encoding="utf-8")
    assert "hello.py" in state.artifacts

    # 6. Assert complete tool execution history
    assert len(state.tool_history) == 2
    assert state.tool_history[0].tool_name == "file_ops"
    assert state.tool_history[0].is_error is False
    assert state.tool_history[1].tool_name == "shell_exec"
    assert state.tool_history[1].is_error is False

    # 7. Assert canonical event timeline stream
    event_types = [e.type for e in event_collector.captured_events]
    assert EventType.SNAPSHOT in event_types
    assert EventType.STEP_START in event_types
    assert EventType.THOUGHT in event_types
    assert EventType.TOOL_CALL in event_types
    assert EventType.OBSERVATION in event_types
    assert EventType.STEP_END in event_types
    assert EventType.FINAL in event_types

    # Validate thought contents
    thought_events = event_collector.get_events_by_type(EventType.THOUGHT.value)
    assert len(thought_events) == 2
    assert "create a python greeting script" in thought_events[0].data["thought"]
    assert "execute the created script" in thought_events[1].data["thought"]

    # 8. Assert long-term memory retention
    memories = await memory_manager.long_term.asearch("hello.py")
    assert len(memories) >= 1
    assert "Outcome:" in memories[0].content


@pytest.mark.asyncio
async def test_end_to_end_error_recovery_workflow(
    temp_workspace: Path,
    mock_llm: MockLLMProvider,
    event_collector: CapturedEventEmitter,
) -> None:
    """Verify orchestrator survives tool errors and captures diagnostic observation."""
    tool_registry = ToolRegistry(workspace_root=str(temp_workspace))
    tool_registry.register(ShellExecTool(workspace_root=str(temp_workspace)))
    memory_manager = MemoryManager(workspace_root=str(temp_workspace))

    # Turn 1: Invoke shell tool with non-zero exit command
    mock_llm.queue_response(
        content="Attempting shell command that fails.",
        tool_calls=[{
            "id": "err_call_1",
            "type": "function",
            "function": {
                "name": "shell_exec",
                "arguments": '{"command": "python -c \\"import sys; sys.exit(42)\\""}',
            },
        }],
    )

    # Turn 2: Recover and conclude
    mock_llm.queue_response(
        content="Detected exit code 42. Task concluded with diagnosed failure."
    )

    runner = AgentRunner(
        config=RunnerConfig(max_steps=5, workspace_root=str(temp_workspace)),
        llm_provider=mock_llm,
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        emitter=event_collector,
    )

    state = await runner.arun_task("Run error diagnostic test.")

    assert state.is_completed is True
    assert len(state.tool_history) == 1
    assert state.tool_history[0].is_error is True
    assert state.tool_history[0].exit_code == 42

    obs_events = event_collector.get_events_by_type(EventType.OBSERVATION.value)
    assert len(obs_events) == 1
    assert obs_events[0].data["is_error"] is True
    assert obs_events[0].data["exit_code"] == 42