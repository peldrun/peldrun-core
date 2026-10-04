"""
PELDRUN Core Event System Unit Tests.
Validates event types, canonical envelope serialization, subscriber dispatching,
fault isolation, and asynchronous SSE streaming.
"""

from __future__ import annotations

import asyncio
import json
from typing import List
from uuid import UUID

import pytest

from peldrun.events.emitter import EventEmitter
from peldrun.events.schema import AgentEvent, EventType, PeldrunEvent


def test_event_types_enum_completeness() -> None:
    """Verify all supported lifecycle event types exist in EventType enum."""
    expected_types = {
        "snapshot",
        "step_start",
        "thought",
        "agent_activity",
        "tool_call",
        "observation",
        "step_end",
        "final",
        "error",
        "ask_human",
    }
    actual_types = {e.value for e in EventType}
    assert expected_types.issubset(actual_types)


def test_agent_event_schema_and_sse_formatting() -> None:
    """Verify AgentEvent structure, canonical JSON serialization, and SSE wire protocol representation."""
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
            assert parsed_data["event_id"] == event.id
            assert parsed_data["type"] == EventType.THOUGHT.value
            assert parsed_data["payload"]["thought"] == "Analyzing user requirements..."
            assert parsed_data["metadata"]["step"] == 1
            # Wire purity check: redundant alias keys must not exist in canonical wire payload
            assert "id" not in parsed_data
            assert "data" not in parsed_data


@pytest.mark.asyncio
async def test_event_emitter_subscribe_and_emit() -> None:
    """Verify targeted subscription and event dispatching."""
    emitter = EventEmitter()
    received: List[PeldrunEvent] = []

    async def on_tool_call(event: PeldrunEvent) -> None:
        received.append(event)

    emitter.subscribe(EventType.TOOL_CALL, on_tool_call)

    event = emitter.create_event(
        EventType.TOOL_CALL,
        payload={"tool_name": "file_ops", "arguments": {"action": "read"}},
    )
    await emitter.emit(event)

    assert len(received) == 1
    assert received[0].payload["tool_name"] == "file_ops"
    assert received[0].sequence == 1


@pytest.mark.asyncio
async def test_event_emitter_subscribe_all_and_unsubscribe() -> None:
    """Verify global subscriber registration and clean unsubscription."""
    emitter = EventEmitter()
    all_events: List[PeldrunEvent] = []

    async def global_handler(event: PeldrunEvent) -> None:
        all_events.append(event)

    emitter.subscribe_all(global_handler)

    ev1 = emitter.create_event(EventType.THOUGHT, payload={"thought": "thinking"})
    await emitter.emit(ev1)
    assert len(all_events) == 1

    emitter.unsubscribe(EventType.THOUGHT, global_handler)
    ev2 = emitter.create_event(EventType.THOUGHT, payload={"thought": "more thought"})
    await emitter.emit(ev2)
    assert len(all_events) == 1


@pytest.mark.asyncio
async def test_event_emitter_all_convenience_helpers() -> None:
    """Verify helper emit methods produce valid events with correct payloads."""
    emitter = EventEmitter()
    collected: List[PeldrunEvent] = []

    async def capture(event: PeldrunEvent) -> None:
        collected.append(event)

    emitter.subscribe_all(capture)

    await emitter.emit_snapshot({"status": "ready"})
    await emitter.emit_step_start(1)
    await emitter.emit_thought("Planning execution")
    await emitter.emit_agent_activity("Inspecting project structure", phase="analysis")
    await emitter.emit_tool_call("web_search", {"query": "PELDRUN"})
    await emitter.emit_observation("Search results text", tool_name="web_search")
    await emitter.emit_step_end(1, elapsed_seconds=0.5)
    await emitter.emit_ask_human("Confirm proceed?", options=["yes", "no"])
    await emitter.emit_error("Minor warning")
    await emitter.emit_final("Execution completed successfully")

    assert len(collected) == 10
    assert [e.type for e in collected] == [
        EventType.SNAPSHOT,
        EventType.STEP_START,
        EventType.THOUGHT,
        EventType.AGENT_ACTIVITY,
        EventType.TOOL_CALL,
        EventType.OBSERVATION,
        EventType.STEP_END,
        EventType.ASK_HUMAN,
        EventType.ERROR,
        EventType.FINAL,
    ]


@pytest.mark.asyncio
async def test_event_emitter_resilience_to_faulty_listener() -> None:
    """Verify an exception inside a subscriber callback does not abort dispatch to other subscribers."""
    emitter = EventEmitter()
    healthy_received: List[PeldrunEvent] = []

    async def failing_listener(_: PeldrunEvent) -> None:
        raise RuntimeError("Subscriber explosion")

    async def healthy_listener(event: PeldrunEvent) -> None:
        healthy_received.append(event)

    emitter.subscribe(EventType.THOUGHT, failing_listener)
    emitter.subscribe(EventType.THOUGHT, healthy_listener)

    event = emitter.create_event(EventType.THOUGHT, payload={"thought": "resilience test"})
    await emitter.emit(event)

    assert len(healthy_received) == 1
    assert healthy_received[0].payload["thought"] == "resilience test"


@pytest.mark.asyncio
async def test_event_emitter_astream_sse() -> None:
    """Verify SSE streaming yields formatted frames and terminates gracefully upon FINAL event."""
    emitter = EventEmitter()
    stream_results: List[str] = []

    async def consume() -> None:
        async for frame in emitter.astream_sse():
            stream_results.append(frame)

    consumer_task = asyncio.create_task(consume())
    await asyncio.sleep(0.01)

    await emitter.emit_thought("Initiating stream")
    await emitter.emit_final("Stream ended")

    await asyncio.wait_for(consumer_task, timeout=2.0)
    assert len(stream_results) == 2
    assert "event: thought" in stream_results[0]
    assert "event: final" in stream_results[1]