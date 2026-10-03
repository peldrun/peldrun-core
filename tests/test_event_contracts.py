"""
PELDRUN Core Event Protocol & Contract Verification Test Suite.
Validates the strict Envelope pattern, typed payloads, wire format purity,
concurrent thread-safety, and monotonic sequence allocation.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from uuid import UUID, uuid4
import pytest
from pydantic import ValidationError

from peldrun.events.emitter import EventEmitter
from peldrun.events.schema import (
    EventType,
    PeldrunEvent,
    SnapshotEvent,
    ToolCallEvent,
    AskHumanEvent,
)


def test_valid_canonical_envelope_creation() -> None:
    event = PeldrunEvent(
        version=1,
        step=2,
        type=EventType.THOUGHT,
        payload={"thought": "Processing request..."},
        metadata={"producer": "test-runner"},
    )
    assert event.version == 1
    assert event.step == 2
    assert event.type == EventType.THOUGHT
    assert event.payload["thought"] == "Processing request..."
    assert isinstance(event.event_id, UUID)
    assert isinstance(event.run_id, UUID)
    assert event.timestamp > 0


def test_strict_uuid_validation_no_silent_fallback() -> None:
    # Malformed event_id string must raise ValidationError instead of falling back to uuid5
    with pytest.raises(ValidationError):
        PeldrunEvent(
            type=EventType.THOUGHT,
            event_id="invalid-uuid-string",
            payload={"thought": "test"},
        )

    # Malformed run_id must raise ValidationError
    with pytest.raises(ValidationError):
        PeldrunEvent(
            type=EventType.THOUGHT,
            run_id="not-a-valid-uuid",
            payload={"thought": "test"},
        )

    # Valid UUID string must parse correctly
    valid_id = str(uuid4())
    event = PeldrunEvent(
        type=EventType.THOUGHT,
        event_id=valid_id,
        payload={"thought": "test"},
    )
    assert str(event.event_id) == valid_id


def test_extra_fields_forbidden_on_envelope() -> None:
    # Envelope must be closed (extra='forbid')
    with pytest.raises(ValidationError):
        PeldrunEvent(
            type=EventType.THOUGHT,
            payload={"thought": "test"},
            unknown_arbitrary_field="not_allowed",
        )


def test_wire_format_purity_no_redundant_id_or_data() -> None:
    event = PeldrunEvent(
        type=EventType.TOOL_CALL,
        payload={"tool_name": "shell", "arguments": {"command": "ls"}},
        metadata={"trace_id": "tr-123"},
    )

    wire_dict = event.to_sse_payload()

    # Canonical keys must exist
    expected_canonical_keys = {
        "version",
        "event_id",
        "sequence",
        "run_id",
        "timestamp",
        "step",
        "type",
        "payload",
        "metadata",
    }
    assert set(wire_dict.keys()) == expected_canonical_keys

    # Alias keys 'id' and 'data' must NOT be present in wire dictionary
    assert "id" not in wire_dict
    assert "data" not in wire_dict

    # Python backward-compatible property access must continue to function
    assert event.id == str(event.event_id)
    assert event.data == event.payload

    # SSE framing check
    sse_frame = event.to_sse()
    assert sse_frame.startswith("event: tool_call\ndata: {")
    assert sse_frame.endswith("}\n\n")

    parsed_sse_data = json.loads(sse_frame.split("data: ")[1].strip())
    assert "id" not in parsed_sse_data
    assert "data" not in parsed_sse_data
    assert parsed_sse_data["type"] == "tool_call"


def test_typed_payload_validation_success_and_failure() -> None:
    # TOOL_CALL requires 'tool_name'
    valid_tool_event = PeldrunEvent(
        type=EventType.TOOL_CALL,
        payload={"tool_name": "browser", "arguments": {"url": "https://example.com"}},
    )
    assert valid_tool_event.payload["tool_name"] == "browser"

    # TOOL_CALL with missing 'tool_name' must fail validation
    with pytest.raises(ValidationError):
        PeldrunEvent(
            type=EventType.TOOL_CALL,
            payload={"arguments": {"url": "https://example.com"}},
        )

    # ASK_HUMAN requires 'question'
    valid_human_event = PeldrunEvent(
        type=EventType.ASK_HUMAN,
        payload={"question": "Do you accept this action?"},
    )
    assert valid_human_event.payload["question"] == "Do you accept this action?"

    # ASK_HUMAN with missing 'question' must fail validation
    with pytest.raises(ValidationError):
        PeldrunEvent(
            type=EventType.ASK_HUMAN,
            payload={"options": ["yes", "no"]},
        )


def test_payload_allows_extra_extensible_fields() -> None:
    # Domain payloads allow extra fields for vendor-specific extensions (extra='allow')
    event = PeldrunEvent(
        type=EventType.TOOL_CALL,
        payload={
            "tool_name": "custom_mcp",
            "arguments": {},
            "provider_custom_attribute": "allowed_value",
        },
    )
    assert event.payload["provider_custom_attribute"] == "allowed_value"


def test_subclass_strictness_literal_enforcement() -> None:
    # Valid subclass creation
    snapshot = SnapshotEvent(payload={"status": "running"})
    assert snapshot.type == EventType.SNAPSHOT

    # Subclass with conflicting type must raise ValidationError
    with pytest.raises(ValidationError):
        SnapshotEvent(type=EventType.ERROR)


def test_legacy_id_data_input_normalization() -> None:
    # Legacy fields (id and data) accepted on input instantiation
    legacy_id = str(uuid4())
    event = PeldrunEvent(
        type=EventType.THOUGHT,
        id=legacy_id,
        data={"thought": "legacy payload"},
    )
    assert str(event.event_id) == legacy_id
    assert event.payload["thought"] == "legacy payload"
    wire = event.to_sse_payload()
    assert "id" not in wire
    assert "data" not in wire


def test_external_sequence_cannot_break_monotonicity() -> None:
    emitter = EventEmitter()

    # Explicit sequence without replay=True must raise ValueError
    with pytest.raises(ValueError):
        emitter.create_event(EventType.THOUGHT, payload={"thought": "x"}, sequence=100)

    # Valid replay sequence updates sequence counter
    ev100 = emitter.create_event(EventType.THOUGHT, payload={"thought": "replay"}, sequence=100, replay=True)
    assert ev100.sequence == 100
    assert emitter.current_sequence == 100

    # Lower replay sequence must be rejected as it violates monotonicity
    with pytest.raises(ValueError):
        emitter.create_event(EventType.THOUGHT, payload={"thought": "broken"}, sequence=2, replay=True)


@pytest.mark.asyncio
async def test_allow_external_sequence_actually_changes_behavior() -> None:
    emitter = EventEmitter()
    emitter.reset_sequence(start=50)

    # Historical event with lower sequence
    ev_ext = PeldrunEvent(
        type=EventType.THOUGHT,
        sequence=10,
        payload={"thought": "historical replay"},
    )

    # In external replay mode, sequence 10 is preserved as-is
    await emitter.emit(ev_ext, allow_external_sequence=True)
    assert ev_ext.sequence == 10
    assert emitter.current_sequence == 50

    # In canonical live mode, a lower sequence event is monotonically corrected
    ev_lower = PeldrunEvent(
        type=EventType.THOUGHT,
        sequence=10,
        payload={"thought": "out-of-order live event"},
    )
    await emitter.emit(ev_lower, allow_external_sequence=False)
    assert ev_lower.sequence == 51
    assert emitter.current_sequence == 51


@pytest.mark.asyncio
async def test_lower_sequence_event_is_rejected_or_corrected() -> None:
    emitter = EventEmitter()
    ev1 = emitter.create_event(EventType.THOUGHT, payload={"thought": "step 1"})
    assert ev1.sequence == 1
    await emitter.emit(ev1)

    # Unsequenced event (sequence=0) must be automatically allocated sequence 2
    ev2 = PeldrunEvent(type=EventType.THOUGHT, sequence=0, payload={"thought": "step 2"})
    await emitter.emit(ev2)
    assert ev2.sequence == 2

    # Lower sequence event (sequence=1) emitted after sequence 2 must be corrected to 3
    ev_lower = PeldrunEvent(type=EventType.THOUGHT, sequence=1, payload={"thought": "delayed"})
    await emitter.emit(ev_lower)
    assert ev_lower.sequence == 3


def test_thread_safety_sequence_allocation() -> None:
    emitter = EventEmitter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        futures = [
            ex.submit(
                emitter.create_event,
                EventType.AGENT_ACTIVITY,
                {"message": f"task {i}"},
            )
            for i in range(200)
        ]
        events = [f.result() for f in futures]

    sequences = [e.sequence for e in events]
    assert len(sequences) == 200
    assert len(set(sequences)) == 200
    assert min(sequences) == 1
    assert max(sequences) == 200


@pytest.mark.asyncio
async def test_concurrent_monotonic_sequence_allocation() -> None:
    emitter = EventEmitter()
    received_events: list[PeldrunEvent] = []

    async def on_event(ev: PeldrunEvent) -> None:
        received_events.append(ev)

    emitter.subscribe_all(on_event)

    # Dispatch 100 concurrent events from multiple asynchronous coroutines
    async def dispatch(idx: int) -> None:
        ev = emitter.create_event(
            EventType.AGENT_ACTIVITY,
            payload={"message": f"Step {idx}", "phase": "exec"},
        )
        await emitter.emit(ev)

    await asyncio.gather(*[dispatch(i) for i in range(100)])

    assert len(received_events) == 100

    # Ensure all sequence numbers are strictly positive, unique, and sequential
    sequences = [ev.sequence for ev in received_events]
    assert len(sequences) == len(set(sequences))
    assert min(sequences) == 1
    assert max(sequences) == 100


@pytest.mark.asyncio
async def test_astream_sse_termination_at_final() -> None:
    emitter = EventEmitter()
    stream_results: list[str] = []

    async def consume() -> None:
        async for frame in emitter.astream_sse():
            stream_results.append(frame)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.01)

    await emitter.emit_thought("Analyzing...")
    await emitter.emit_final("Done.")

    await asyncio.wait_for(task, timeout=2.0)
    assert len(stream_results) == 2
    assert "event: thought" in stream_results[0]
    assert "event: final" in stream_results[1]