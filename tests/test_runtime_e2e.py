"""
PELDRUN Core End-to-End (E2E) Runtime Lifecycle & Recovery Test Suite.
Verifies the cohesive interaction of the execution loop, security boundaries,
tool invocations, unified event envelopes, and state checkpoint recovery.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional
import pytest

from peldrun.agents.react_agent import ReActAgent, ReActAgentConfig
from peldrun.engine.checkpoint import CheckpointManager
from peldrun.engine.runner import AgentRunner
from peldrun.engine.state import ExecutionState, ExecutionStatus, MessageRole
from peldrun.events.emitter import EventEmitter
from peldrun.events.schema import EventType, PeldrunEvent
from peldrun.llm.client import StreamChunk
from peldrun.llm.providers.openai_compat import BaseLLMProvider, LLMResponse
from peldrun.security.policy import SecurityPolicy
from peldrun.tools.builtins.file_ops import FileOpsTool
from peldrun.tools.registry import ToolRegistry


class MockE2ELLMProvider(BaseLLMProvider):
    """Deterministic LLM mock complying strictly with BaseLLMProvider contracts."""

    def __init__(self, step_responses: List[LLMResponse]) -> None:
        super().__init__()
        self.model = "mock-e2e-model"
        self.step_responses = list(step_responses)
        self.call_count = 0

    @property
    def name(self) -> str:
        return "mock_e2e_provider"

    async def check_health(self) -> bool:
        return True

    async def close(self) -> None:
        pass

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if self.call_count < len(self.step_responses):
            resp = self.step_responses[self.call_count]
            self.call_count += 1
            return resp
        return LLMResponse(content="Default mock final answer.", finish_reason="stop")

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        resp = await self.generate(messages=messages, tools=tools, **kwargs)
        if resp.content:
            yield StreamChunk(content=resp.content)


@pytest.mark.asyncio
async def test_e2e_runtime_execution_with_security_and_events(temp_workspace: Path) -> None:
    """
    Test a full E2E execution lifecycle:
    Prompt -> Model calls file_ops -> SecurityPolicy validates -> File written -> Final answer.
    """
    # 1. Prepare deterministic scripted LLM turns
    step1_response = LLMResponse(
        content="<think>I need to write the deliverable file into workspace.</think>",
        tool_calls=[
            {
                "id": "call_write_1",
                "function": {
                    "name": "file_ops",
                    "arguments": json.dumps({
                        "action": "write",
                        "path": "output/deliverable.txt",
                        "content": "PELDRUN Core Production Ready"
                    })
                }
            }
        ]
    )
    step2_response = LLMResponse(
        content="The deliverable file has been successfully written and verified.",
        finish_reason="stop"
    )

    mock_llm = MockE2ELLMProvider([step1_response, step2_response])

    # 2. Wire security policy, tools registry, and event emitter
    security_policy = SecurityPolicy(workspace_root=temp_workspace)
    tool_registry = ToolRegistry(workspace_root=str(temp_workspace))
    file_tool = FileOpsTool(workspace_root=str(temp_workspace), security_policy=security_policy)
    tool_registry.register(file_tool)

    emitter = EventEmitter()
    captured_events: List[PeldrunEvent] = []

    async def event_collector(event: PeldrunEvent) -> None:
        captured_events.append(event)

    emitter.subscribe_all(event_collector)

    # 3. Instantiate ReActAgent with non-streaming mock execution
    config = ReActAgentConfig(name="e2e_agent", max_steps=5, enable_streaming=False)
    agent = ReActAgent(
        config=config,
        llm_provider=mock_llm,
        tool_registry=tool_registry,
        emitter=emitter,
    )

    # 4. Execute via AgentRunner using the actual run() method
    runner = AgentRunner(agent=agent, emitter=emitter)
    state = await runner.run(task_prompt="Generate output/deliverable.txt in workspace")

    # 5. Assertions on business delivery
    assert state.status == ExecutionStatus.COMPLETED
    assert "deliverable.txt" in state.output or "verified" in state.output

    written_file = temp_workspace / "output" / "deliverable.txt"
    assert written_file.exists()
    assert written_file.read_text(encoding="utf-8") == "PELDRUN Core Production Ready"

    # 6. Assertions on strict Event Envelope Protocol
    assert len(captured_events) >= 4

    # Verify monotonic incrementation of sequence numbers
    sequences = [ev.sequence for ev in captured_events]
    assert sequences == list(range(1, len(captured_events) + 1))

    # Verify that thought, tool_call, observation, and final were dispatched
    event_types = [ev.type for ev in captured_events]
    assert EventType.THOUGHT in event_types
    assert EventType.TOOL_CALL in event_types
    assert EventType.OBSERVATION in event_types
    assert EventType.FINAL in event_types


@pytest.mark.asyncio
async def test_e2e_checkpoint_persistence_and_recovery(temp_workspace: Path) -> None:
    """
    Test runtime checkpointing: state snapshot saved to disk, restored,
    and verified for zero data loss using CheckpointManager contracts.
    """
    checkpoint_dir = temp_workspace / "checkpoints"
    manager = CheckpointManager(storage_dir=str(checkpoint_dir))

    # 1. Create an active execution state
    original_state = ExecutionState(task_prompt="Recoverable architecture deployment")
    original_state.current_step = 3
    original_state.add_message(role=MessageRole.USER, content="Start deployment")
    original_state.add_message(role=MessageRole.ASSISTANT, content="Step 1 & 2 completed")
    original_state.record_tool_execution(
        tool_name="file_ops",
        arguments={"action": "write", "path": "step1.log"},
        output="Logged successfully",
        exit_code=0,
    )

    # 2. Persist checkpoint using CheckpointManager.asave_checkpoint
    cp = await manager.asave_checkpoint(original_state, label="test_e2e_save")
    assert cp.id is not None
    assert Path(cp.file_path).exists()

    # 3. Restore state from checkpoint using CheckpointManager.arestore_state
    restored_state = await manager.arestore_state(cp.id)

    assert restored_state is not None
    assert restored_state.task_prompt == "Recoverable architecture deployment"
    assert restored_state.current_step == 3
    assert len(restored_state.messages) >= 2
    assert len(restored_state.tool_calls) == 1
    assert restored_state.tool_calls[0].tool_name == "file_ops"
    assert restored_state.tool_calls[0].output == "Logged successfully"


@pytest.mark.asyncio
async def test_e2e_direct_chat_stream_path() -> None:
    """
    Verify the lightweight direct chat interaction path without running ReAct loops.
    """
    emitter = EventEmitter()
    streamed_events: List[PeldrunEvent] = []

    async def chat_collector(event: PeldrunEvent) -> None:
        streamed_events.append(event)

    emitter.subscribe_all(chat_collector)

    # Direct conversation token stream simulation
    chat_chunks = ["Hello! ", "I am ", "PELDRUN Core.", " How can I help?"]
    for chunk in chat_chunks:
        await emitter.emit_agent_activity(message=chunk, phase="chat_stream", step=0)

    await emitter.emit_final(output="".join(chat_chunks), step=0)

    assert len(streamed_events) == 5
    assert all(ev.step == 0 for ev in streamed_events)
    assert streamed_events[-1].type == EventType.FINAL
    assert streamed_events[-1].payload["output"] == "Hello! I am PELDRUN Core. How can I help?"