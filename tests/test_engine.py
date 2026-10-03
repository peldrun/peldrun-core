"""
PELDRUN Core Engine Test Suite.

Aligned with the current source contract:
  - ExecutionState uses `current_step`, `deliverables`, `task_prompt` (default="").
  - AgentRunner accepts only `agent=` (StepExecutableAgent protocol).
  - ExecutionGraph must be `.compile()`d then `.run()`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import pytest

from peldrun.engine.checkpoint import CheckpointManager
from peldrun.engine.graph import ExecutionGraph
from peldrun.engine.runner import AgentRunner, RunnerConfig
from peldrun.engine.state import (
    ExecutionState,
    ExecutionStatus,
    MessageRole,
)
from peldrun.events.schema import EventType
from tests.conftest import CapturedEventEmitter


# =============================================================================
# ExecutionState
# =============================================================================

def test_execution_state_initial_lifecycle() -> None:
    """Default fields, including task_prompt which defaults to an empty string."""
    state = ExecutionState()
    assert state.task_prompt == ""
    assert state.current_step == 0
    assert state.max_steps == 30
    assert state.agent_name == "PrimaryAgent"
    assert state.workspace_root is None
    assert state.status == ExecutionStatus.IDLE
    assert state.messages == []
    assert state.tool_history == []
    assert state.deliverables == []
    assert state.metadata == {}
    assert state.checkpoints == []


def test_execution_state_message_tracking() -> None:
    state = ExecutionState(task_prompt="Test")
    msg_sys = state.add_message(role=MessageRole.SYSTEM, content="Init system instructions.")
    msg_usr = state.add_message(role=MessageRole.USER, content="Deploy test database.")

    assert len(state.messages) == 2
    assert state.messages[0].role == MessageRole.SYSTEM
    assert state.messages[1].role == MessageRole.USER
    assert msg_sys is state.messages[0]
    assert msg_usr is state.messages[1]


def test_execution_state_llm_messages_export() -> None:
    state = ExecutionState(task_prompt="Test")
    state.add_message(role=MessageRole.SYSTEM, content="System directive")
    state.add_message(role=MessageRole.USER, content="Hello")
    assert state.get_llm_messages() == [
        {"role": "system", "content": "System directive"},
        {"role": "user", "content": "Hello"},
    ]


def test_execution_state_status_transitions() -> None:
    state = ExecutionState(task_prompt="Test")
    assert state.status == ExecutionStatus.IDLE
    state.status = ExecutionStatus.RUNNING
    assert state.status == ExecutionStatus.RUNNING
    state.status = ExecutionStatus.COMPLETED
    assert state.status == ExecutionStatus.COMPLETED
    state.status = ExecutionStatus.FAILED
    assert state.status == ExecutionStatus.FAILED


def test_execution_state_tool_recording_and_error() -> None:
    state = ExecutionState(task_prompt="Test")

    rec_ok = state.record_tool_execution(
        call_id="call_abc",
        tool_name="file_ops",
        arguments={"action": "write", "path": "output.txt"},
        output="Created file.",
        exit_code=0,
        is_error=False,
    )
    assert len(state.tool_history) == 1
    assert rec_ok.call_id == "call_abc"
    assert rec_ok.tool_name == "file_ops"
    assert rec_ok.is_error is False
    assert isinstance(rec_ok.timestamp, float)

    rec_err = state.record_tool_execution(
        call_id="call_err",
        tool_name="shell_exec",
        arguments={"command": "invalid_cmd"},
        output="Command not found",
        exit_code=127,
        is_error=True,
    )
    assert len(state.tool_history) == 2
    assert rec_err.is_error is True
    assert rec_err.exit_code == 127


def test_execution_state_deliverables_deduplication() -> None:
    state = ExecutionState(task_prompt="Test")
    state.add_deliverable("report.pdf")
    state.add_deliverable("report.pdf")
    state.add_deliverable("chart.png")
    assert state.deliverables == ["report.pdf", "chart.png"]


def test_execution_state_checkpoint_create_and_restore() -> None:
    state = ExecutionState(task_prompt="Test")
    state.current_step = 2
    state.metadata["phase"] = "early"

    cp = state.create_checkpoint()
    assert cp.step == 2
    assert isinstance(cp.state_dump, dict)
    assert len(state.checkpoints) == 1

    state.current_step = 10
    state.metadata["phase"] = "late"

    assert state.restore_checkpoint(cp.checkpoint_id) is True
    assert state.current_step == 2
    assert state.metadata["phase"] == "early"


def test_execution_state_restore_unknown_checkpoint() -> None:
    state = ExecutionState(task_prompt="Test")
    assert state.restore_checkpoint("does-not-exist") is False


# =============================================================================
# CheckpointManager
# =============================================================================

@pytest.mark.asyncio
async def test_checkpoint_manager_save_and_restore(temp_workspace: Path) -> None:
    """Persist and restore execution state via CheckpointManager."""
    manager = CheckpointManager(workspace_root=str(temp_workspace))
    await manager.ainitialize()

    state = ExecutionState(task_prompt="Test task")
    state.current_step = 3
    state.add_message(role=MessageRole.USER, content="Perform step 3.")
    state.add_deliverable("report.pdf")

    checkpoint = await manager.asave_checkpoint(state=state, label="midway_checkpoint")
    assert checkpoint.step == 3
    assert checkpoint.label == "midway_checkpoint"
    assert Path(checkpoint.file_path).exists()

    checkpoints = await manager.alist_checkpoints()
    assert len(checkpoints) == 1
    assert checkpoints[0].id == checkpoint.id

    latest = await manager.aload_latest_checkpoint()
    assert latest is not None
    assert latest.id == checkpoint.id

    restored_state = await manager.arestore_state(checkpoint.id)
    assert restored_state.current_step == 3
    assert "report.pdf" in restored_state.deliverables
    assert len(restored_state.messages) == 1


# =============================================================================
# ExecutionGraph
# =============================================================================

@pytest.mark.asyncio
async def test_execution_graph_linear_pipeline() -> None:
    graph = ExecutionGraph()

    async def step_one(state: ExecutionState) -> ExecutionState:
        state.current_step += 1
        state.metadata["node_1"] = "done"
        return state

    async def step_two(state: ExecutionState) -> ExecutionState:
        state.current_step += 1
        state.metadata["node_2"] = "done"
        state.status = ExecutionStatus.COMPLETED
        return state

    graph.add_node("first", step_one)
    graph.add_node("second", step_two)
    graph.add_edge("first", "second")
    graph.set_entry_point("first")

    compiled = graph.compile()
    initial = ExecutionState(task_prompt="Linear pipeline test")
    final = await compiled.run(initial)

    assert final.current_step == 2
    assert final.metadata["node_1"] == "done"
    assert final.metadata["node_2"] == "done"
    assert final.status == ExecutionStatus.COMPLETED


# =============================================================================
# AgentRunner
# =============================================================================

class _CompletingAgent:
    """StepExecutableAgent that completes on the first step."""

    name = "CompletingAgent"

    def __init__(self) -> None:
        self.calls = 0

    async def step(self, state: ExecutionState, emitter) -> bool:
        self.calls += 1
        state.add_message(role=MessageRole.ASSISTANT, content="done")
        state.status = ExecutionStatus.COMPLETED
        return True


class _NeverCompletingAgent:
    """StepExecutableAgent that never returns True and sleeps per step."""

    name = "NeverCompletingAgent"

    def __init__(self) -> None:
        self.calls = 0

    async def step(self, state: ExecutionState, emitter) -> bool:
        self.calls += 1
        await asyncio.sleep(0.05)
        return False


@pytest.mark.asyncio
async def test_agent_runner_lifecycle_and_events(
    event_collector: CapturedEventEmitter,
    temp_workspace: Path,
) -> None:
    agent = _CompletingAgent()
    runner = AgentRunner(
        agent=agent,
        emitter=event_collector,
        config=RunnerConfig(max_steps=5, enable_checkpointing=False),
    )

    state = await runner.run(
        task_prompt="Sample runner execution test",
        workspace_root=str(temp_workspace),
    )

    assert agent.calls == 1
    assert state.status == ExecutionStatus.COMPLETED
    assert state.current_step == 1
    assert state.task_prompt == "Sample runner execution test"
    assert state.agent_name == "CompletingAgent"

    event_types = [e.type for e in event_collector.captured_events]
    assert EventType.SNAPSHOT in event_types
    assert EventType.STEP_START in event_types
    assert EventType.STEP_END in event_types


@pytest.mark.asyncio
async def test_agent_runner_stop_signal(
    event_collector: CapturedEventEmitter,
    temp_workspace: Path,
) -> None:
    agent = _NeverCompletingAgent()
    runner = AgentRunner(
        agent=agent,
        emitter=event_collector,
        config=RunnerConfig(max_steps=10, enable_checkpointing=False),
    )

    async def trigger_cancel_later() -> None:
        await asyncio.sleep(0.08)
        runner.cancel()

    cancel_task = asyncio.create_task(trigger_cancel_later())
    state = await runner.run(
        task_prompt="Task to be interrupted",
        workspace_root=str(temp_workspace),
    )
    await cancel_task

    assert state.current_step < 10
    assert runner.is_cancelled is True
    assert state.status == ExecutionStatus.PAUSED