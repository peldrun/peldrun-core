"""
PELDRUN Core Agents Subsystem Test Suite.
Verifies BaseAgent lifecycle boundaries, ReAct reasoning loops, PlanningAgent milestone decomposition,
CodingAgent self-repair and artifact collection, and the centralized get_agent factory.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List
import pytest

from peldrun.agents import (
    AgentConfig,
    BaseAgent,
    CodingAgent,
    CodingAgentConfig,
    Plan,
    PlanningAgent,
    PlanningAgentConfig,
    PlanStep,
    ReActAgent,
    ReActAgentConfig,
    get_agent,
)
from peldrun.events.schema import EventType
from peldrun.memory import MemoryManager
from peldrun.tools.builtins import FileOpsTool
from peldrun.tools.registry import ToolRegistry
from tests.conftest import CapturedEventEmitter, MockLLMProvider


class ConcreteTestAgent(BaseAgent):
    """Concrete minimal agent implementation for testing BaseAgent mechanics."""

    async def _astep(self) -> bool:
        if self.state.step >= 2:
            self.state.mark_completed(output="Concrete step limit reached.")
            return False
        return True


@pytest.mark.asyncio
async def test_base_agent_step_ceiling_and_completion(
    mock_llm: MockLLMProvider,
    event_collector: CapturedEventEmitter,
    tool_registry: ToolRegistry,
    memory_manager: MemoryManager,
) -> None:
    """Verify BaseAgent respects max_steps and manages execution lifecycle."""
    config = AgentConfig(name="test_base", max_steps=2)
    agent = ConcreteTestAgent(
        config=config,
        llm_provider=mock_llm,
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        emitter=event_collector,
    )

    state = await agent.arun(task="Execute concrete base test")
    assert state.is_completed is True
    assert state.step <= 2

    # Check event stream captured lifecycle
    events = [e.type for e in event_collector.captured_events]
    assert EventType.SNAPSHOT in events
    assert EventType.STEP_START in events
    assert EventType.FINAL in events


@pytest.mark.asyncio
async def test_base_agent_pause_and_resume(
    mock_llm: MockLLMProvider,
    event_collector: CapturedEventEmitter,
    tool_registry: ToolRegistry,
    memory_manager: MemoryManager,
) -> None:
    """Verify pausing and resuming BaseAgent execution loop."""
    config = AgentConfig(name="pause_test_agent", max_steps=5)
    agent = ConcreteTestAgent(
        config=config,
        llm_provider=mock_llm,
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        emitter=event_collector,
    )

    agent.pause()
    assert agent.is_paused is True

    # Start run in background
    run_task = asyncio.create_task(agent.arun("Paused task"))
    await asyncio.sleep(0.05)
    assert agent.is_running is True

    # Resume and allow to complete
    agent.resume()
    assert agent.is_paused is False

    await run_task
    assert agent.state.is_completed is True


@pytest.mark.asyncio
async def test_react_agent_reasoning_and_final_output(
    mock_llm: MockLLMProvider,
    event_collector: CapturedEventEmitter,
    tool_registry: ToolRegistry,
    memory_manager: MemoryManager,
) -> None:
    """Verify ReActAgent extracts thought tags, broadcasts thought events, and terminates on final output."""
    thought_response = "<think>Analyzing repository architecture requirements.</think>Plan finalized."
    mock_llm.queue_response(content=thought_response)

    agent = ReActAgent(
        config=ReActAgentConfig(name="react_test", max_steps=5),
        llm_provider=mock_llm,
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        emitter=event_collector,
    )

    state = await agent.arun("Architecture review task")

    assert state.is_completed is True
    assert "Plan finalized." in str(state.final_output)

    # Verify thought event was broadcast
    thought_events = event_collector.get_events_by_type(EventType.THOUGHT.value)
    assert len(thought_events) >= 1
    assert "Analyzing repository architecture" in thought_events[0].data["thought"]


@pytest.mark.asyncio
async def test_react_agent_tool_invocation_loop(
    mock_llm: MockLLMProvider,
    event_collector: CapturedEventEmitter,
    temp_workspace: Path,
) -> None:
    """Verify ReActAgent executes tool calls, records observations, and completes subsequent turn."""
    tool_reg = ToolRegistry(workspace_root=str(temp_workspace))
    tool_reg.register(FileOpsTool(workspace_root=str(temp_workspace)))

    mem_mgr = MemoryManager(workspace_root=str(temp_workspace))

    # Turn 1: LLM triggers file write tool call
    mock_llm.queue_response(
        content="Writing initial configuration file.",
        tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "file_ops",
                "arguments": '{"action": "write", "path": "config.json", "content": "{\\"env\\": \\"test\\"}"}',
            },
        }],
    )
    # Turn 2: LLM concludes after observing result
    mock_llm.queue_response(content="Config file successfully initialized.")

    agent = ReActAgent(
        config=ReActAgentConfig(max_steps=5),
        llm_provider=mock_llm,
        tool_registry=tool_reg,
        memory_manager=mem_mgr,
        emitter=event_collector,
    )

    state = await agent.arun("Initialize project config")

    assert state.is_completed is True
    assert (temp_workspace / "config.json").exists()

    # Validate tool_call and observation events were fired
    tool_events = event_collector.get_events_by_type(EventType.TOOL_CALL.value)
    obs_events = event_collector.get_events_by_type(EventType.OBSERVATION.value)

    assert len(tool_events) == 1
    assert tool_events[0].data["tool_name"] == "file_ops"
    assert len(obs_events) == 1
    assert obs_events[0].data["tool_name"] == "file_ops"


@pytest.mark.asyncio
async def test_planning_agent_progression(
    mock_llm: MockLLMProvider,
    event_collector: CapturedEventEmitter,
    tool_registry: ToolRegistry,
    memory_manager: MemoryManager,
) -> None:
    """Verify PlanningAgent formulates milestone plan and advances step-by-step."""
    # Step 1: Formulate plan
    plan_json = (
        '{"goal": "Deploy app", "steps": ['
        '{"id": 1, "title": "Check environment", "description": "Verify python runtime"},'
        '{"id": 2, "title": "Run tests", "description": "Execute pytest suite"}'
        ']}'
    )
    mock_llm.queue_response(content=plan_json)

    # Step 2: Conclude Step 1
    mock_llm.queue_response(content="Environment is verified and ready.")

    # Step 3: Conclude Step 2
    mock_llm.queue_response(content="All tests passed successfully.")

    agent = PlanningAgent(
        config=PlanningAgentConfig(max_steps=10),
        llm_provider=mock_llm,
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        emitter=event_collector,
    )

    state = await agent.arun("Deploy application pipeline")

    assert state.is_completed is True
    assert agent.plan is not None
    assert agent.plan.is_complete is True
    assert agent.plan.steps[0].status == "completed"
    assert agent.plan.steps[1].status == "completed"


@pytest.mark.asyncio
async def test_coding_agent_artifact_accumulation(
    mock_llm: MockLLMProvider,
    event_collector: CapturedEventEmitter,
    temp_workspace: Path,
) -> None:
    """Verify CodingAgent records modified files in state artifacts metadata."""
    tool_reg = ToolRegistry(workspace_root=str(temp_workspace))
    tool_reg.register(FileOpsTool(workspace_root=str(temp_workspace)))
    mem_mgr = MemoryManager(workspace_root=str(temp_workspace))

    # Turn 1: Write code file
    mock_llm.queue_response(
        content="Writing source module.",
        tool_calls=[{
            "id": "code_call_1",
            "type": "function",
            "function": {
                "name": "file_ops",
                "arguments": '{"action": "write", "path": "main.py", "content": "print(\'PELDRUN\')"}',
            },
        }],
    )
    # Turn 2: Finish
    mock_llm.queue_response(content="Code authored and verified.")

    agent = CodingAgent(
        config=CodingAgentConfig(max_steps=5),
        llm_provider=mock_llm,
        tool_registry=tool_reg,
        memory_manager=mem_mgr,
        emitter=event_collector,
    )

    state = await agent.arun("Create main module")

    assert state.is_completed is True
    assert "main.py" in state.metadata.get("modified_artifacts", [])


def test_agent_factory_routing(
    mock_llm: MockLLMProvider,
    tool_registry: ToolRegistry,
    memory_manager: MemoryManager,
) -> None:
    """Verify get_agent instantiates ReActAgent, PlanningAgent, and CodingAgent by identifier."""
    # ReAct
    react_agt = get_agent("react", llm_provider=mock_llm, tool_registry=tool_registry, memory_manager=memory_manager)
    assert isinstance(react_agt, ReActAgent)

    # Planning
    plan_agt = get_agent("planning", llm_provider=mock_llm, tool_registry=tool_registry, memory_manager=memory_manager)
    assert isinstance(plan_agt, PlanningAgent)

    # Coding
    code_agt = get_agent("coding", llm_provider=mock_llm, tool_registry=tool_registry, memory_manager=memory_manager)
    assert isinstance(code_agt, CodingAgent)

    # Invalid agent identifier raises ValueError
    with pytest.raises(ValueError) as exc_info:
        get_agent("unknown_agent_variant")

    assert "unsupported agent type" in str(exc_info.value).lower()