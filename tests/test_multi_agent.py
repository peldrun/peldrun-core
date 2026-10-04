"""
Automated Verification Suite for Phase 11 Multi-Agent Architecture.
Verifies specialist registration, parent-child event bridging, and sub-task delegation.
"""

import uuid
from typing import Any, Dict, List
import pytest

from peldrun.agents.base import AgentConfig, BaseAgent
from peldrun.agents.multi.coordinator import MultiAgentCoordinator
from peldrun.agents.multi.protocol import DelegatedTask
from peldrun.events.emitter import EventEmitter
from peldrun.events.schema import EventType, PeldrunEvent
from peldrun.tools.builtins.delegate_tool import DelegateTool


class MockSpecialistAgent(BaseAgent):
    """Simulated specialist agent executing specific sub-tasks."""

    async def run_task(self, prompt: str, max_steps: int | None = None) -> str:
        if self.emitter:
            await self.emitter.emit_thought(thought=f"Specialist thinking on: {prompt}", step=1)
        return f"Specialist output for: {prompt}"


@pytest.mark.asyncio
async def test_multi_agent_delegation_and_event_bridging():
    parent_run_id = uuid.uuid4()
    parent_emitter = EventEmitter(run_id=parent_run_id)

    captured_events: List[PeldrunEvent] = []
    parent_emitter.subscribe_all(lambda ev: captured_events.append(ev))

    coordinator = MultiAgentCoordinator(parent_emitter=parent_emitter)

    # 1. Register specialist
    specialist = MockSpecialistAgent(
        config=AgentConfig(name="coder", system_prompt="You are a coding expert.")
    )
    coordinator.register_specialist("coder", specialist)
    assert "coder" in coordinator.list_specialists()

    # 2. Delegate sub-task directly
    task = DelegatedTask(
        parent_run_id=parent_run_id,
        target_agent_id="coder",
        instruction="Implement sorting algorithm",
    )

    result = await coordinator.delegate(task)

    assert result.success is True
    assert "Specialist output for: Implement sorting algorithm" in result.output

    # 3. Verify event bridging: child events appeared on parent stream with lineage metadata
    delegation_events = [ev for ev in captured_events if ev.metadata.get("delegated_agent") == "coder"]
    assert len(delegation_events) > 0
    assert any(ev.type == EventType.THOUGHT for ev in delegation_events)

    # 4. Verify DelegateTool execution
    tool = DelegateTool(coordinator=coordinator, parent_run_id=parent_run_id)
    tool_res = await tool.execute(target_agent="coder", instruction="Write tests")

    assert tool_res.exit_code == 0
    assert "Specialist output for: Write tests" in tool_res.output
    assert tool_res.metadata.get("target_agent") == "coder"


@pytest.mark.asyncio
async def test_multi_agent_unregistered_specialist():
    parent_run_id = uuid.uuid4()
    parent_emitter = EventEmitter(run_id=parent_run_id)
    coordinator = MultiAgentCoordinator(parent_emitter=parent_emitter)

    task = DelegatedTask(
        parent_run_id=parent_run_id,
        target_agent_id="non_existent",
        instruction="Do something",
    )

    result = await coordinator.delegate(task)
    assert result.success is False
    assert "is not registered" in str(result.error)