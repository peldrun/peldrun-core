"""
PELDRUN Core Event Protocol & Contract Verification Test Suite.
Validates the strict Envelope pattern, typed payloads, wire format purity,
and concurrent sequence monotonicity.
"""

from __future__ import annotations

import asyncio
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
    ToolCallPayload,
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