"""
Automated Verification Suite for Phase 8 Event Bus & Replay Mechanics.
Verifies monotonic sequencing, replay after disconnection, and terminal event guarantee.
"""

import uuid
import pytest

from peldrun.events.emitter import EventEmitter
from peldrun.events.schema import EventType, PeldrunEvent


@pytest.mark.asyncio
async def test_event_bus_monotonic_sequence_and_replay():
    run_id = uuid.uuid4()
    emitter = EventEmitter(run_id=run_id)

    # Dispatch series of events
    await emitter.emit(EventType.STEP_START, payload={"step": 1}, step=1)
    await emitter.emit_activity(phase="planning", message="Analyzing task requirements", step=1)
    await emitter.emit(EventType.TOOL_CALLED, payload={"tool": "file_ops"}, step=1)
    await emitter.emit(EventType.RUN_COMPLETED, payload={"result": "Success"}, step=1)

    # Replay all events from sequence 0
    full_history = await emitter.replay(after_sequence=0)
    assert len(full_history) == 4
    assert [ev.sequence for ev in full_history] == [1, 2, 3, 4]

    # Replay reconnect simulation (client missed events after sequence 2)
    resumed_events = await emitter.replay(after_sequence=2)
    assert len(resumed_events) == 2
    assert resumed_events[0].sequence == 3
    assert resumed_events[0].type == EventType.TOOL_CALLED
    assert resumed_events[1].sequence == 4
    assert resumed_events[1].type == EventType.RUN_COMPLETED


@pytest.mark.asyncio
async def test_event_bus_terminal_event_guarantee():
    run_id = uuid.uuid4()
    emitter = EventEmitter(run_id=run_id)

    received_events: list[PeldrunEvent] = []
    emitter.subscribe_all(lambda ev: received_events.append(ev))  # type: ignore

    await emitter.emit(EventType.STEP_START, payload={"step": 1}, step=1)

    # Verify that ensure_terminated emits RUN_FAILED if execution broke abruptly
    await emitter.ensure_terminated(default_error="Agent crashed unexpectedly.")

    assert len(received_events) == 2
    assert received_events[-1].type == EventType.RUN_FAILED
    assert received_events[-1].payload.get("error") == "Agent crashed unexpectedly."
    assert emitter._is_terminated is True

    # Subsequent call to ensure_terminated should not emit duplicate terminal events
    await emitter.ensure_terminated(default_error="Another error")
    assert len(received_events) == 2