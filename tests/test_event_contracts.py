"""
PELDRUN Core Event Protocol Contract Tests.
Ensures strict validation of the Envelope Pattern, monotonic sequences,
rejection of arbitrary fields via extra='forbid', and wire-level SSE compliance.
"""

from __future__ import annotations

import json
import uuid
import pytest
from pydantic import ValidationError

from peldrun.events.emitter import EventEmitter
from peldrun.events.schema import EventType, PeldrunEvent


def test_valid_event_contract_instantiation() -> None:
    """Verify that a compliant PeldrunEvent passes strict validation."""
    test_run_id = uuid.uuid4()
    event = PeldrunEvent(
        version=1,
        sequence=1,
        run_id=test_run_id,
        step=2,
        type=EventType.STEP_START,
        payload={"agent_name": "react_agent"},
        metadata={"priority": "high"},
    )

    assert event.version == 1
    assert event.sequence == 1
    assert event.run_id == test_run_id
    assert event.step == 2
    assert event.type == EventType.STEP_START
    assert event.payload["agent_name"] == "react_agent"
    assert event.metadata["priority"] == "high"
    assert isinstance(event.event_id, uuid.UUID)


def test_extra_fields_forbidden() -> None:
    """Verify that extra='forbid' strictly rejects unknown top-level envelope fields."""
    with pytest.raises(ValidationError) as excinfo:
        PeldrunEvent(
            type=EventType.TOOL_CALL,
            unauthorized_field="malicious_payload",
        )
    assert "Extra inputs are not permitted" in str(excinfo.value)


def test_negative_values_rejected() -> None:
    """Verify validation constraints on version, sequence, and step fields."""
    with pytest.raises(ValidationError):
        PeldrunEvent(
            type=EventType.THOUGHT,
            version=0,  # Must be >= 1
        )

    with pytest.raises(ValidationError):
        PeldrunEvent(
            type=EventType.THOUGHT,
            sequence=-1,  # Must be >= 0
        )

    with pytest.raises(ValidationError):
        PeldrunEvent(
            type=EventType.THOUGHT,
            step=-1,  # Must be >= 0
        )


def test_invalid_event_type_rejected() -> None:
    """Verify that unrecognized event types are rejected."""
    with pytest.raises(ValidationError):
        PeldrunEvent(
            type="unknown_event_type_that_does_not_exist",  # type: ignore
        )


def test_event_emitter_monotonic_sequence() -> None:
    """Verify that EventEmitter enforces monotonically incrementing sequence numbers."""
    emitter = EventEmitter()
    assert emitter.current_sequence == 0

    e1 = emitter.create_event(EventType.STEP_START, step=1)
    e2 = emitter.create_event(EventType.AGENT_ACTIVITY, payload={"message": "working"}, step=1)
    e3 = emitter.create_event(EventType.TOOL_CALL, payload={"tool_name": "shell_exec"}, step=1)
    e4 = emitter.create_event(EventType.OBSERVATION, payload={"output": "success"}, step=1)
    e5 = emitter.create_event(EventType.STEP_END, step=1)

    assert [e1.sequence, e2.sequence, e3.sequence, e4.sequence, e5.sequence] == [1, 2, 3, 4, 5]
    assert emitter.current_sequence == 5


def test_sse_wire_format_compatibility() -> None:
    """Verify that to_sse() produces wire format with both envelope and backward-compatible fields."""
    test_event_id = uuid.uuid4()
    test_run_id = uuid.uuid4()

    event = PeldrunEvent(
        event_id=test_event_id,
        run_id=test_run_id,
        sequence=42,
        step=3,
        type=EventType.FINAL,
        payload={"output": "Done!"},
    )

    sse_frame = event.to_sse()
    assert sse_frame.startswith("event: final\n")
    assert sse_frame.endswith("\n\n")

    # Extract JSON payload line
    for line in sse_frame.splitlines():
        if line.startswith("data: "):
            wire_dict = json.loads(line[len("data: "):])
            # Strict envelope verification
            assert wire_dict["event_id"] == str(test_event_id)
            assert wire_dict["run_id"] == str(test_run_id)
            assert wire_dict["sequence"] == 42
            assert wire_dict["step"] == 3
            assert wire_dict["payload"]["output"] == "Done!"
            # Backward-compatibility alias verification
            assert wire_dict["id"] == str(test_event_id)
            assert wire_dict["data"]["output"] == "Done!"


def test_roundtrip_json_serialization() -> None:
    """Verify lossless serialization and deserialization."""
    original = PeldrunEvent(
        sequence=10,
        type=EventType.OBSERVATION,
        payload={"tool_name": "read_file", "output": "file content", "exit_code": 0},
        metadata={"session_id": "test_sess"},
    )

    json_str = original.model_dump_json()
    reconstructed = PeldrunEvent.model_validate_json(json_str)

    assert reconstructed.event_id == original.event_id
    assert reconstructed.sequence == original.sequence
    assert reconstructed.run_id == original.run_id
    assert reconstructed.payload == original.payload
    assert reconstructed.metadata == original.metadata