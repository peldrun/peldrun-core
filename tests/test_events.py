"""
PELDRUN Core Event Subsystem Test Suite.
Verifies the nine canonical event types, Pydantic event schemas, SSE serialization,
asynchronous EventEmitter routing, and listener fault isolation.
"""

from __future__ import annotations

import asyncio
import json
import pytest

from peldrun.events.emitter import EventEmitter
from peldrun.events.schema import AgentEvent, EventType


def test_event_types_enum_completeness() -> None:
    """Verify all 9 mandatory canonical event types exist in the EventType enum."""
    expected_types = {
        "snapshot",
        "step_start",
        "thought",
        "tool_call",
        "observation",
        "step_end",
        "final",
        "error",
        "ask_human",
    }
    actual_types = {e.value for e in EventType}
    assert expected_types.issubset(actual_types), f"Missing event types: {expected_types - actual_types}"


def test_agent_event_schema_and_sse_formatting() -> None:
    """Verify AgentEvent structure, JSON serialization, and SSE wire protocol representation."""
    payload = {"thought": "Analyzing user requirements..."}
    event = AgentEvent(
        type=EventType.THOUGHT,
        data=payload,
        metadata={"step": 1},
    )

    assert event.type == EventType.THOUGHT
    assert event.data["thought"] == "Analyzing user requirements..."
    assert event.id is not None
    assert event.timestamp > 0

    sse_text = event.to_sse()
    assert sse_text.startswith(f"event: {EventType.THOUGHT.value}\n")
    assert "data: " in sse_text
    assert sse_text.endswith("\n\n")

    # Extract and parse json data from SSE line
    for line in sse_text.splitlines():
        if line.startswith("data: "):
            parsed_data = json.loads(line[len("data: "):])
            assert parsed_data["id"] == event.id
            assert parsed_data["type"] == EventType.THOUGHT.value
            assert parsed_data["data"]["thought"] == "Analyzing user requirements..."


@pytest.mark.asyncio
async def test_event_emitter_subscribe_and_emit() -> None:
    """Verify typed subscriptions receive matching events exclusively."""
    emitter = EventEmitter()
    received_thoughts: list[AgentEvent] = []
    received_errors: list[AgentEvent] = []

    async def thought_listener(event: AgentEvent) -> None:
        received_thoughts.append(event)

    async def error_listener(event: AgentEvent) -> None:
        received_errors.append(event)

    emitter.subscribe(EventType.THOUGHT, thought_listener)
    emitter.subscribe(EventType.ERROR, error_listener)

    # Emit a thought event
    await emitter.emit_thought(thought="Thinking about architecture.")
    assert len(received_thoughts) == 1
    assert len(received_errors) == 0
    assert received_thoughts[0].data["thought"] == "Thinking about architecture."

    # Emit an error event
    await emitter.emit_error(error="A simulated runtime error occurred.")
    assert len(received_thoughts) == 1
    assert len(received_errors) == 1
    assert received_errors[0].data["error"] == "A simulated runtime error occurred."


@pytest.mark.asyncio
async def test_event_emitter_subscribe_all_and_unsubscribe() -> None:
    """Verify subscribe_all captures all event variants and unsubscribe unregisters listeners."""
    emitter = EventEmitter()
    all_events: list[AgentEvent] = []

    async def global_listener(event: AgentEvent) -> None:
        all_events.append(event)

    emitter.subscribe_all(global_listener)

    await emitter.emit_step_start(step_number=1)
    await emitter.emit_step_end(step_number=1)

    assert len(all_events) == 2
    assert all_events[0].type == EventType.STEP_START
    assert all_events[1].type == EventType.STEP_END

    # Unsubscribe and verify no more events are received
    emitter.unsubscribe(EventType.STEP_START, global_listener)
    emitter.unsubscribe(EventType.STEP_END, global_listener)

    await emitter.emit_step_start(step_number=2)
    assert len(all_events) == 2


@pytest.mark.asyncio
async def test_event_emitter_all_convenience_helpers() -> None:
    """Verify each specialized helper emits the appropriate event type and structure."""
    emitter = EventEmitter()
    dispatched: list[AgentEvent] = []

    async def collector(event: AgentEvent) -> None:
        dispatched.append(event)

    emitter.subscribe_all(collector)

    # 1. snapshot
    await emitter.emit_snapshot(snapshot={"step": 0, "status": "idle"})
    assert dispatched[-1].type == EventType.SNAPSHOT

    # 2. step_start
    await emitter.emit_step_start(step_number=1)
    assert dispatched[-1].type == EventType.STEP_START
    assert dispatched[-1].data["step_number"] == 1

    # 3. thought
    await emitter.emit_thought(thought="Plan formulated.")
    assert dispatched[-1].type == EventType.THOUGHT

    # 4. tool_call
    await emitter.emit_tool_call(tool_name="file_ops", arguments={"action": "list", "path": "."}, tool_call_id="call_1")
    assert dispatched[-1].type == EventType.TOOL_CALL
    assert dispatched[-1].data["tool_name"] == "file_ops"

    # 5. observation
    await emitter.emit_observation(output="file1.txt", tool_name="file_ops", tool_call_id="call_1")
    assert dispatched[-1].type == EventType.OBSERVATION
    assert dispatched[-1].data["output"] == "file1.txt"

    # 6. step_end
    await emitter.emit_step_end(step_number=1)
    assert dispatched[-1].type == EventType.STEP_END

    # 7. ask_human
    await emitter.emit_ask_human(question="Confirm deletion?", options=["yes", "no"])
    assert dispatched[-1].type == EventType.ASK_HUMAN
    assert dispatched[-1].data["question"] == "Confirm deletion?"

    # 8. error
    await emitter.emit_error(error="Execution failure")
    assert dispatched[-1].type == EventType.ERROR

    # 9. final
    await emitter.emit_final(output="Task successfully accomplished.")
    assert dispatched[-1].type == EventType.FINAL
    assert dispatched[-1].data["output"] == "Task successfully accomplished."

    assert len(dispatched) == 9


@pytest.mark.asyncio
async def test_event_emitter_resilience_to_faulty_listener() -> None:
    """Verify an exception inside one listener does not prevent other listeners from executing."""
    emitter = EventEmitter()
    successful_calls: list[str] = []

    async def faulty_listener(event: AgentEvent) -> None:
        raise RuntimeError("Intentional listener crash!")

    async def healthy_listener(event: AgentEvent) -> None:
        successful_calls.append("healthy_ok")

    emitter.subscribe(EventType.THOUGHT, faulty_listener)
    emitter.subscribe(EventType.THOUGHT, healthy_listener)

    # Should not raise exception
    await emitter.emit_thought(thought="Resilience test.")
    assert len(successful_calls) == 1
    assert successful_calls[0] == "healthy_ok"


@pytest.mark.asyncio
async def test_event_emitter_astream_sse() -> None:
    """Verify that astream_sse yields SSE data chunks as events are published."""
    emitter = EventEmitter()
    stream_results: list[str] = []

    async def reader() -> None:
        async for sse_chunk in emitter.astream_sse():
            stream_results.append(sse_chunk)
            if "final" in sse_chunk:
                break

    reader_task = asyncio.create_task(reader())
    await asyncio.sleep(0.01)

    await emitter.emit_thought(thought="Streaming token 1")
    await emitter.emit_final(output="End of stream")

    await asyncio.wait_for(reader_task, timeout=2.0)
    assert len(stream_results) == 2
    assert "event: thought" in stream_results[0]
    assert "event: final" in stream_results[1]