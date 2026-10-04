"""
Automated Verification Suite for Phase 10 Planning Flows.
Verifies plan formulation, step status progression, and full flow execution.
"""

import json
from typing import Any, Dict, List
import pytest

from peldrun.agents.base import AgentConfig
from peldrun.agents.tool_call_agent import ToolCallAgent
from peldrun.events.emitter import EventEmitter
from peldrun.events.schema import EventType, PeldrunEvent
from peldrun.flows.planning import PlanState, StepStatus
from peldrun.flows.planning_flow import PlanningFlow
from peldrun.tools.collection import ToolCollection


class MockPlanningLLM:
    """Simulates LLM plan formulation and step response generation."""

    def __init__(self):
        self.call_count = 0

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self.call_count += 1
        content = messages[-1].get("content", "")

        # Phase 1: Plan formulation request
        if "Break down the following task" in content:
            plan_json = {
                "steps": [
                    {"index": 1, "title": "Prepare Data", "description": "Create sample data file"},
                    {"index": 2, "title": "Analyze Data", "description": "Perform summary analysis"},
                ]
            }
            return {
                "choices": [{"message": {"role": "assistant", "content": json.dumps(plan_json)}}]
            }

        # Phase 2: Step execution
        return {
            "choices": [{"message": {"role": "assistant", "content": "Step executed successfully."}}]
        }


@pytest.mark.asyncio
async def test_planning_flow_execution():
    emitter = EventEmitter()
    captured_events: List[PeldrunEvent] = []
    emitter.subscribe_all(lambda ev: captured_events.append(ev))

    mock_llm = MockPlanningLLM()
    collection = ToolCollection()

    agent_config = AgentConfig(name="worker", system_prompt="Execute step instructions.")
    agent = ToolCallAgent(
        config=agent_config,
        llm=mock_llm,  # type: ignore
        tool_collection=collection,
        emitter=emitter,
    )

    flow = PlanningFlow(
        llm=mock_llm,  # type: ignore
        agent=agent,
        emitter=emitter,
    )

    result = await flow.aexecute(task="Process monthly report")

    assert result.success is True
    assert "Prepare Data" in result.output
    assert "Analyze Data" in result.output

    plan_meta = result.metadata.get("plan", {})
    steps = plan_meta.get("steps", [])
    assert len(steps) == 2
    assert all(s.get("status") == StepStatus.COMPLETED.value for s in steps)

    # Verify event broadcasts occurred
    event_types = [ev.type for ev in captured_events]
    assert EventType.SNAPSHOT in event_types
    assert EventType.AGENT_ACTIVITY in event_types
    assert EventType.FINAL in event_types