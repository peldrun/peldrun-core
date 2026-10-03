"""
PELDRUN Core Engine Test Suite.
Verifies ExecutionState mutations, CheckpointManager serialization,
ExecutionGraph asynchronous pipelines, and AgentRunner lifecycle orchestration.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict
import pytest

from peldrun.engine.checkpoint import CheckpointManager, StateCheckpoint
from peldrun.engine.graph import ExecutionGraph
from peldrun.engine.runner import AgentRunner, RunnerConfig
from peldrun.engine.state import ExecutionState
from peldrun.events.schema import EventType
from peldrun.memory import MemoryManager
from peldrun.tools.registry import ToolRegistry
from tests.conftest import CapturedEventEmitter, MockLLMProvider


def test_execution_state_lifecycle_and_snapshots() -> None:
    """Verify state initialization, message tracking, status transitions, and snapshot serialization."""
    state = ExecutionState()
    assert state.step == 0
    assert state.status == "idle"
    assert not state.is_completed
    assert not state.is_error

    # Record interaction messages
    state.add_message(role="system", content="Initialize system instructions.")
    state.add_message(role="user", content="Deploy test database.")
    assert len(state.messages) == 2

    # Transition state to completed
    state.mark_completed(output="Database deployed successfully.")
    assert state.status == "completed"
    assert state.is_completed
    assert state.final_output == "Database deployed successfully."

    # Verify snapshot round-trip
    snapshot = state.to_snapshot_dict()
    restored_state = ExecutionState.from_snapshot_dict(snapshot)

    assert restored_state.status == "completed"
    assert restored_state.step == state.step
    assert len(restored_state.messages) == len(state.messages)
    assert restored_state.final_output == state.final_output


def test_execution_state_tool_recording_and_artifacts() -> None:
    """Verify tool execution history tracking, error flag recording, and artifact accumulation."""
    state = ExecutionState()

    state.record_tool_execution(
        tool_name="file_ops",
        arguments={"action": "write", "path": "output.txt"},
        output="Created file.",
        exit_code=0,
        is_error=False,
        artifacts=["output.txt"],
        tool_call_id="call_abc",
    )

    assert len(state.tool_history) == 1
    record = state.tool_history[0]
    assert record.tool_name == "file_ops"
    assert record.artifacts == ["output.txt"]
    assert "output.txt" in state.artifacts

    # Record secondary tool execution with error
    state.record_tool_execution(
        tool_name="shell_exec",
        arguments={"command": "invalid_cmd"},
        output="Command not found",
        exit_code=127,
        is_error=True,
        tool_call_id="call_err",
    )

    assert len(state.tool_history) == 2
    assert state.tool_history[1].is_error is True
    assert state.tool_history[1].exit_code == 127


@pytest.mark.asyncio
async def test_checkpoint_manager_save_and_restore(temp_workspace: Path) -> None:
    """Verify persisting and retrieving execution state checkpoints to disk asynchronously."""
    manager = CheckpointManager(workspace_root=str(temp_workspace))
    await manager.ainitialize()

    state = ExecutionState()
    state.step = 3
    state.add_message("user", "Perform step 3.")
    state.artifacts.append("report.pdf")

    # Save checkpoint
    checkpoint = await manager.asave_checkpoint(state=state, label="midway_checkpoint")
    assert checkpoint.step == 3
    assert checkpoint.label == "midway_checkpoint"
    assert Path(checkpoint.file_path).exists()

    # List checkpoints
    checkpoints = await manager.alist_checkpoints()
    assert len(checkpoints) == 1
    assert checkpoints[0].id == checkpoint.id

    # Load latest checkpoint
    loaded_checkpoint = await manager.aload_latest_checkpoint()
    assert loaded_checkpoint is not None
    assert loaded_checkpoint.id == checkpoint.id

    # Restore state
    restored_state = await manager.arestore_state(checkpoint.id)
    assert restored_state.step == 3
    assert "report.pdf" in restored_state.artifacts
    assert len(restored_state.messages) == 1


@pytest.mark.asyncio
async def test_execution_graph_linear_pipeline() -> None:
    """Verify ExecutionGraph defines and executes asynchronous node workflows."""
    graph = ExecutionGraph()

    async def step_one(state: ExecutionState) -> ExecutionState:
        state.step += 1
        state.metadata["node_1"] = "done"
        return state

    async def step_two(state: ExecutionState) -> ExecutionState:
        state.step += 1
        state.metadata["node_2"] = "done"
        state.mark_completed(output="Pipeline finished.")
        return state

    graph.add_node("first", step_one)
    graph.add_node("second", step_two)
    graph.add_edge("first", "second")
    graph.set_entry_point("first")

    initial_state = ExecutionState()
    final_state = await graph.arun(initial_state)

    assert final_state.step == 2
    assert final_state.metadata.get("node_1") == "done"
    assert final_state.metadata.get("node_2") == "done"
    assert final_state.is_completed is True
    assert final_state.final_output == "Pipeline finished."


@pytest.mark.asyncio
async def test_agent_runner_lifecycle_and_events(
    mock_llm: MockLLMProvider,
    event_collector: CapturedEventEmitter,
    tool_registry: ToolRegistry,
    memory_manager: MemoryManager,
    temp_workspace: Path,
) -> None:
    """Verify AgentRunner coordinates execution cycles and emits required canonical events."""
    mock_llm.queue_response(content="Automated run completed successfully.")

    runner = AgentRunner(
        config=RunnerConfig(
            max_steps=5,
            workspace_root=str(temp_workspace),
        ),
        llm_provider=mock_llm,
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        emitter=event_collector,
    )

    state = await runner.arun_task("Sample runner execution test")

    assert state.is_completed is True
    assert state.final_output == "Automated run completed successfully."

    # Validate emitted event sequence
    event_types = [e.type for e in event_collector.captured_events]
    assert EventType.SNAPSHOT in event_types
    assert EventType.STEP_START in event_types
    assert EventType.STEP_END in event_types
    assert EventType.FINAL in event_types


@pytest.mark.asyncio
async def test_agent_runner_stop_signal(
    mock_llm: MockLLMProvider,
    event_collector: CapturedEventEmitter,
    tool_registry: ToolRegistry,
    memory_manager: MemoryManager,
    temp_workspace: Path,
) -> None:
    """Verify AgentRunner halts gracefully upon receiving an asynchronous stop request."""
    # Queue multiple responses
    mock_llm.queue_response(content="Processing step 1...")
    mock_llm.queue_response(content="Processing step 2...")

    runner = AgentRunner(
        config=RunnerConfig(max_steps=10, workspace_root=str(temp_workspace)),
        llm_provider=mock_llm,
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        emitter=event_collector,
    )

    async def trigger_stop_later() -> None:
        await asyncio.sleep(0.05)
        runner.stop()

    stop_task = asyncio.create_task(trigger_stop_later())
    state = await runner.arun_task("Task to be interrupted")
    await stop_task

    # Execution should halt early before reaching max steps
    assert state.step < 10
    assert not runner.is_running