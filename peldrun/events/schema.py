"""
PELDRUN Core Event Protocol Schemas.
Defines strictly typed Pydantic v2 event models for the PELDRUN streaming lifecycle.
"""

from __future__ import annotations

import json
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class EventType(str, Enum):
    """Supported protocol event types for agent execution streaming."""
    SNAPSHOT = "snapshot"
    STEP_START = "step_start"
    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    OBSERVATION = "observation"
    STEP_END = "step_end"
    FINAL = "final"
    ERROR = "error"
    ASK_HUMAN = "ask_human"


class BaseEvent(BaseModel):
    """Base event model common to all PELDRUN protocol notifications."""
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the event instance."
    )
    type: EventType = Field(
        ...,
        description="Discriminator type of the event."
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Epoch timestamp when the event was emitted."
    )
    run_id: Optional[str] = Field(
        default=None,
        description="Unique execution or job run identifier."
    )
    step: Optional[int] = Field(
        default=None,
        description="Sequence index of the current execution step."
    )
    payload: Any = Field(
        default_factory=dict,
        description="Event specific structured payload."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Contextual metadata."
    )

    def __init__(self, **data: Any) -> None:
        if "id" in data and "event_id" not in data:
            data["event_id"] = data.pop("id")
        if "data" in data and "payload" not in data:
            data["payload"] = data.pop("data")
        super().__init__(**data)

    @property
    def id(self) -> str:
        """Alias for event_id."""
        return self.event_id

    @property
    def data(self) -> Dict[str, Any]:
        """Convenience dictionary representation of event data for tests and consumers."""
        if isinstance(self.payload, BaseModel):
            return self.payload.model_dump(mode="json")
        if isinstance(self.payload, dict):
            return self.payload
        return {"value": self.payload}

    @data.setter
    def data(self, val: Any) -> None:
        self.payload = val

    def to_sse(self) -> str:
        """Serialize event to standard SSE wire protocol format."""
        raw_dict = {
            "id": self.id,
            "type": self.type.value if hasattr(self.type, "value") else str(self.type),
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "step": self.step,
            "data": self.data,
            "metadata": self.metadata,
        }
        event_name = self.type.value if hasattr(self.type, "value") else str(self.type)
        return f"event: {event_name}\ndata: {json.dumps(raw_dict, ensure_ascii=False)}\n\n"

    def to_sse_payload(self) -> Dict[str, Any]:
        """Convert the event model to a standard dictionary payload for SSE streaming."""
        return self.model_dump(mode="json")


# Alias for backward compatibility and test fixtures
AgentEvent = BaseEvent


class SnapshotPayload(BaseModel):
    """Payload representing an instantaneous state snapshot."""
    model_config = ConfigDict(extra="allow")

    status: str = Field(default="idle", description="Current execution status (idle, running, paused, done, error)")
    agent_name: Optional[str] = Field(default=None, description="Active agent identifier")
    step: int = Field(default=0, description="Current execution step index")
    total_steps: Optional[int] = Field(default=None, description="Configured maximum execution steps")
    workspace_files: List[str] = Field(default_factory=list, description="Files discovered in current workspace")
    active_tools: List[str] = Field(default_factory=list, description="Enabled tool names")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary execution context metadata")


class SnapshotEvent(BaseEvent):
    """Event broadcasting a full execution and workspace state snapshot."""
    type: EventType = Field(default=EventType.SNAPSHOT)
    payload: Union[SnapshotPayload, Dict[str, Any]] = Field(default_factory=SnapshotPayload)


class StepStartPayload(BaseModel):
    """Payload marking the initiation of an agent action step."""
    model_config = ConfigDict(extra="allow")
    step: Optional[int] = Field(default=None, description="1-based index of the step being initiated")
    step_number: Optional[int] = Field(default=None, description="Alternative key for step index")
    agent_name: Optional[str] = Field(default="", description="Name of the agent taking action")
    input_prompt: Optional[str] = Field(default=None, description="Input query or current task directive")


class StepStartEvent(BaseEvent):
    """Event emitted at the onset of an execution step."""
    type: EventType = Field(default=EventType.STEP_START)
    payload: Union[StepStartPayload, Dict[str, Any]] = Field(default_factory=dict)


class ThoughtPayload(BaseModel):
    """Payload carrying reasoning traces or chain-of-thought tokens."""
    model_config = ConfigDict(extra="allow")
    content: Optional[str] = Field(default="", description="Reasoning text, hypothesis, or plan fragment")
    thought: Optional[str] = Field(default="", description="Thought content alias")
    is_delta: bool = Field(default=False, description="True if payload is a streaming delta token")


class ThoughtEvent(BaseEvent):
    """Event conveying internal monologue, planning, or reasoning."""
    type: EventType = Field(default=EventType.THOUGHT)
    payload: Union[ThoughtPayload, Dict[str, Any]] = Field(default_factory=dict)


class ToolCallPayload(BaseModel):
    """Payload identifying a selected tool and its invocation arguments."""
    model_config = ConfigDict(extra="allow")
    call_id: Optional[str] = Field(default="", description="Unique call identifier from the LLM engine")
    tool_call_id: Optional[str] = Field(default="", description="Call id alias")
    tool_name: str = Field(default="", description="Registered identifier of the target tool")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Structured arguments passed to the tool")


class ToolCallEvent(BaseEvent):
    """Event dispatched when an agent decides to invoke an external tool."""
    type: EventType = Field(default=EventType.TOOL_CALL)
    payload: Union[ToolCallPayload, Dict[str, Any]] = Field(default_factory=dict)


class ObservationPayload(BaseModel):
    """Payload delivering tool execution outcomes and observations."""
    model_config = ConfigDict(extra="allow")
    call_id: Optional[str] = Field(default="", description="Corresponding tool call identifier")
    tool_call_id: Optional[str] = Field(default="", description="Call id alias")
    tool_name: Optional[str] = Field(default="", description="Name of the tool that completed execution")
    output: Any = Field(default="", description="Raw or parsed result returned by the tool")
    exit_code: int = Field(default=0, description="Process exit code or status indicator (0 = success)")
    is_error: bool = Field(default=False, description="Flag indicating execution failure")


class ObservationEvent(BaseEvent):
    """Event detailing the result or observation acquired from a tool execution."""
    type: EventType = Field(default=EventType.OBSERVATION)
    payload: Union[ObservationPayload, Dict[str, Any]] = Field(default_factory=dict)


class StepEndPayload(BaseModel):
    """Payload signaling the conclusion of an individual step."""
    model_config = ConfigDict(extra="allow")
    step: Optional[int] = Field(default=None, description="Index of the step that completed")
    step_number: Optional[int] = Field(default=None, description="Step number alias")
    elapsed_seconds: float = Field(default=0.0, description="Duration of the step in seconds")
    success: bool = Field(default=True, description="Whether the step executed without unhandled errors")


class StepEndEvent(BaseEvent):
    """Event emitted when an execution step wraps up."""
    type: EventType = Field(default=EventType.STEP_END)
    payload: Union[StepEndPayload, Dict[str, Any]] = Field(default_factory=dict)


class FinalPayload(BaseModel):
    """Payload containing the final output and deliverable for the user."""
    model_config = ConfigDict(extra="allow")
    content: Optional[str] = Field(default="", description="Final markdown or textual response to the user prompt")
    output: Optional[str] = Field(default="", description="Output alias")
    deliverables: List[str] = Field(default_factory=list, description="Workspace file paths produced as final artifacts")
    total_steps: int = Field(default=1, description="Total steps executed to achieve the resolution")
    total_tokens: Optional[int] = Field(default=None, description="Aggregated token usage across all steps")


class FinalEvent(BaseEvent):
    """Event signaling successful resolution and final deliverable dispatch."""
    type: EventType = Field(default=EventType.FINAL)
    payload: Union[FinalPayload, Dict[str, Any]] = Field(default_factory=dict)


class ErrorPayload(BaseModel):
    """Payload documenting execution faults, runtime exceptions, or timeouts."""
    model_config = ConfigDict(extra="allow")
    message: Optional[str] = Field(default="", description="Human-readable error summary")
    error: Optional[str] = Field(default="", description="Error message alias")
    error_type: str = Field(default="RuntimeError", description="Exception class or category")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Detailed trace or contextual debug attributes")
    recoverable: bool = Field(default=False, description="Indicates whether the agent can recover and continue")


class ErrorEvent(BaseEvent):
    """Event emitted when an unrecoverable failure or tracked error takes place."""
    type: EventType = Field(default=EventType.ERROR)
    payload: Union[ErrorPayload, Dict[str, Any]] = Field(default_factory=dict)


class AskHumanPayload(BaseModel):
    """Payload requesting interactive human feedback, authorization, or input."""
    model_config = ConfigDict(extra="allow")
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique token referencing this interaction pause"
    )
    question: str = Field(..., description="The query or prompt displayed to the user")
    options: Optional[List[str]] = Field(
        default=None,
        description="Optional set of predefined selections for the user"
    )
    timeout_seconds: Optional[float] = Field(
        default=300.0,
        description="Maximum seconds to hold execution before timing out"
    )


class AskHumanEvent(BaseEvent):
    """Event emitted when the agent pauses execution to solicit human collaboration."""
    type: EventType = Field(default=EventType.ASK_HUMAN)
    payload: Union[AskHumanPayload, Dict[str, Any]] = Field(default_factory=dict)


# Type union representing any protocol event model
PeldrunEvent = Union[
    SnapshotEvent,
    StepStartEvent,
    ThoughtEvent,
    ToolCallEvent,
    ObservationEvent,
    StepEndEvent,
    FinalEvent,
    ErrorEvent,
    AskHumanEvent,
    BaseEvent,
]

__all__ = [
    "EventType",
    "BaseEvent",
    "AgentEvent",
    "PeldrunEvent",
    "SnapshotPayload",
    "SnapshotEvent",
    "StepStartPayload",
    "StepStartEvent",
    "ThoughtPayload",
    "ThoughtEvent",
    "ToolCallPayload",
    "ToolCallEvent",
    "ObservationPayload",
    "ObservationEvent",
    "StepEndPayload",
    "StepEndEvent",
    "FinalPayload",
    "FinalEvent",
    "ErrorPayload",
    "ErrorEvent",
    "AskHumanPayload",
    "AskHumanEvent",
]