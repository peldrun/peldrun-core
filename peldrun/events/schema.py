"""
PELDRUN Core Event Protocol Schemas.
Implements the strict Envelope Pattern using Pydantic v2 for the unified runtime lifecycle.
"""

from __future__ import annotations

import json
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EventType(str, Enum):
    """Supported protocol event types across agent tasks and direct chat interactions."""
    SNAPSHOT = "snapshot"
    STEP_START = "step_start"
    THOUGHT = "thought"
    AGENT_ACTIVITY = "agent_activity"
    TOOL_CALL = "tool_call"
    OBSERVATION = "observation"
    STEP_END = "step_end"
    FINAL = "final"
    ERROR = "error"
    ASK_HUMAN = "ask_human"


class PeldrunEvent(BaseModel):
    """
    Unified public event envelope for all PELDRUN runtime event dispatches.
    Enforces strict typing and forbids arbitrary top-level fields on the core model.
    """
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    version: int = Field(default=1, ge=1, description="Protocol envelope schema version")
    event_id: UUID = Field(default_factory=uuid4, description="Unique identifier for the event instance")
    sequence: int = Field(default=0, ge=0, description="Monotonically increasing sequence number within run")
    run_id: UUID = Field(default_factory=uuid4, description="Unique execution or session identifier")
    timestamp: float = Field(default_factory=time.time, description="Unix epoch timestamp in UTC seconds")
    step: int = Field(default=0, ge=0, description="Execution step index (0 for direct chat or initialization)")
    type: EventType = Field(..., description="Discriminator event type")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Event-specific payload content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Contextual tracing and producer metadata")

    @field_validator("event_id", "run_id", mode="before")
    @classmethod
    def _coerce_uuid(cls, value: Any) -> UUID:
        """Coerce strings or existing UUID instances into strict UUID objects."""
        if isinstance(value, UUID):
            return value
        if isinstance(value, str):
            try:
                return UUID(value)
            except ValueError:
                # Deterministic fallback for test fixtures and custom identifiers
                return uuid.uuid5(uuid.NAMESPACE_DNS, value)
        return uuid4()

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_fields(cls, values: Any) -> Any:
        """Normalize legacy fields (id -> event_id, data -> payload) while preserving envelope purity."""
        if not isinstance(values, dict):
            return values
        mutable = dict(values)
        if "id" in mutable and "event_id" not in mutable:
            mutable["event_id"] = mutable.pop("id")
        if "data" in mutable and "payload" not in mutable:
            mutable["payload"] = mutable.pop("data")
        return mutable

    @property
    def id(self) -> str:
        """String representation of event_id for backward compatibility."""
        return str(self.event_id)

    @property
    def data(self) -> Dict[str, Any]:
        """Dictionary access alias pointing to payload for backward compatibility."""
        return self.payload

    @data.setter
    def data(self, value: Dict[str, Any]) -> None:
        self.payload = value

    def to_sse_payload(self) -> Dict[str, Any]:
        """
        Return serialized JSON-safe dictionary representation with both
        standard envelope keys and backward-compatible wire aliases.
        """
        raw = self.model_dump(mode="json")
        raw["id"] = str(self.event_id)
        raw["data"] = self.payload
        return raw

    def to_sse(self) -> str:
        """Serialize event into standard SSE wire protocol frame."""
        serialized = self.to_sse_payload()
        event_name = self.type.value if hasattr(self.type, "value") else str(self.type)
        return f"event: {event_name}\ndata: {json.dumps(serialized, ensure_ascii=False)}\n\n"


# Aliases for backward compatibility
BaseEvent = PeldrunEvent
AgentEvent = PeldrunEvent


# Typed helper payloads for validation and domain modeling
class SnapshotPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str = Field(default="idle")
    agent_name: Optional[str] = None
    step: int = 0
    total_steps: Optional[int] = None
    workspace_files: List[str] = Field(default_factory=list)
    active_tools: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StepStartPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    step: int = 0
    agent_name: Optional[str] = ""
    input_prompt: Optional[str] = None


class ThoughtPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    thought: str = ""
    is_delta: bool = False


class AgentActivityPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    message: str = ""
    phase: str = "general"
    details: Dict[str, Any] = Field(default_factory=dict)


class ToolCallPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    tool_call_id: str = ""


class ObservationPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    tool_name: str = ""
    output: Any = ""
    exit_code: int = 0
    is_error: bool = False
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    tool_call_id: str = ""


class StepEndPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    step: int = 0
    elapsed_seconds: float = 0.0
    success: bool = True


class FinalPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    output: str = ""
    deliverables: List[str] = Field(default_factory=list)
    total_steps: int = 1
    total_tokens: Optional[int] = None


class ErrorPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    error: str = ""
    error_type: str = "RuntimeError"
    details: Optional[Dict[str, Any]] = None
    recoverable: bool = False


class AskHumanPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    options: List[str] = Field(default_factory=list)
    timeout_seconds: float = 300.0


# Specialized convenience envelope subclasses
class SnapshotEvent(PeldrunEvent):
    type: EventType = Field(default=EventType.SNAPSHOT)


class StepStartEvent(PeldrunEvent):
    type: EventType = Field(default=EventType.STEP_START)


class ThoughtEvent(PeldrunEvent):
    type: EventType = Field(default=EventType.THOUGHT)


class AgentActivityEvent(PeldrunEvent):
    type: EventType = Field(default=EventType.AGENT_ACTIVITY)


class ToolCallEvent(PeldrunEvent):
    type: EventType = Field(default=EventType.TOOL_CALL)


class ObservationEvent(PeldrunEvent):
    type: EventType = Field(default=EventType.OBSERVATION)


class StepEndEvent(PeldrunEvent):
    type: EventType = Field(default=EventType.STEP_END)


class FinalEvent(PeldrunEvent):
    type: EventType = Field(default=EventType.FINAL)


class ErrorEvent(PeldrunEvent):
    type: EventType = Field(default=EventType.ERROR)


class AskHumanEvent(PeldrunEvent):
    type: EventType = Field(default=EventType.ASK_HUMAN)


__all__ = [
    "EventType",
    "PeldrunEvent",
    "BaseEvent",
    "AgentEvent",
    "SnapshotPayload",
    "StepStartPayload",
    "ThoughtPayload",
    "AgentActivityPayload",
    "ToolCallPayload",
    "ObservationPayload",
    "StepEndPayload",
    "FinalPayload",
    "ErrorPayload",
    "AskHumanPayload",
    "SnapshotEvent",
    "StepStartEvent",
    "ThoughtEvent",
    "AgentActivityEvent",
    "ToolCallEvent",
    "ObservationEvent",
    "StepEndEvent",
    "FinalEvent",
    "ErrorEvent",
    "AskHumanEvent",
]