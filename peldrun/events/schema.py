"""
PELDRUN Core Event Protocol Schemas.
Defines strictly typed Pydantic v2 event models for the PELDRUN streaming lifecycle.
"""

from __future__ import annotations

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
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

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

    def to_sse_payload(self) -> Dict[str, Any]:
        """Convert the event model to a standard dictionary payload for SSE streaming."""
        return self.model_dump(mode="json")


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
    payload: SnapshotPayload = Field(default_factory=SnapshotPayload)


class StepStartPayload(BaseModel):
    """Payload marking the initiation of an agent action step."""
    step: int = Field(..., description="1-based index of the step being initiated")
    agent_name: str = Field(..., description="Name of the agent taking action")
    input_prompt: Optional[str] = Field(default=None, description="Input query or current task directive")


class StepStartEvent(BaseEvent):
    """Event emitted at the onset of an execution step."""
    type: EventType = Field(default=EventType.STEP_START)
    payload: StepStartPayload


class ThoughtPayload(BaseModel):
    """Payload carrying reasoning traces or chain-of-thought tokens."""
    content: str = Field(..., description="Reasoning text, hypothesis, or plan fragment")
    is_delta: bool = Field(default=False, description="True if payload is a streaming delta token")


class ThoughtEvent(BaseEvent):
    """Event conveying internal monologue, planning, or reasoning."""
    type: EventType = Field(default=EventType.THOUGHT)
    payload: ThoughtPayload


class ToolCallPayload(BaseModel):
    """Payload identifying a selected tool and its invocation arguments."""
    call_id: str = Field(..., description="Unique call identifier from the LLM engine")
    tool_name: str = Field(..., description="Registered identifier of the target tool")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Structured arguments passed to the tool")


class ToolCallEvent(BaseEvent):
    """Event dispatched when an agent decides to invoke an external tool."""
    type: EventType = Field(default=EventType.TOOL_CALL)
    payload: ToolCallPayload


class ObservationPayload(BaseModel):
    """Payload delivering tool execution outcomes and observations."""
    call_id: str = Field(..., description="Corresponding tool call identifier")
    tool_name: str = Field(..., description="Name of the tool that completed execution")
    output: Any = Field(..., description="Raw or parsed result returned by the tool")
    exit_code: int = Field(default=0, description="Process exit code or status indicator (0 = success)")
    is_error: bool = Field(default=False, description="Flag indicating execution failure")


class ObservationEvent(BaseEvent):
    """Event detailing the result or observation acquired from a tool execution."""
    type: EventType = Field(default=EventType.OBSERVATION)
    payload: ObservationPayload


class StepEndPayload(BaseModel):
    """Payload signaling the conclusion of an individual step."""
    step: int = Field(..., description="Index of the step that completed")
    elapsed_seconds: float = Field(default=0.0, description="Duration of the step in seconds")
    success: bool = Field(default=True, description="Whether the step executed without unhandled errors")


class StepEndEvent(BaseEvent):
    """Event emitted when an execution step wraps up."""
    type: EventType = Field(default=EventType.STEP_END)
    payload: StepEndPayload


class FinalPayload(BaseModel):
    """Payload containing the final output and deliverable for the user."""
    content: str = Field(..., description="Final markdown or textual response to the user prompt")
    deliverables: List[str] = Field(default_factory=list, description="Workspace file paths produced as final artifacts")
    total_steps: int = Field(default=1, description="Total steps executed to achieve the resolution")
    total_tokens: Optional[int] = Field(default=None, description="Aggregated token usage across all steps")


class FinalEvent(BaseEvent):
    """Event signaling successful resolution and final deliverable dispatch."""
    type: EventType = Field(default=EventType.FINAL)
    payload: FinalPayload


class ErrorPayload(BaseModel):
    """Payload documenting execution faults, runtime exceptions, or timeouts."""
    message: str = Field(..., description="Human-readable error summary")
    error_type: str = Field(default="RuntimeError", description="Exception class or category")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Detailed trace or contextual debug attributes")
    recoverable: bool = Field(default=False, description="Indicates whether the agent can recover and continue")


class ErrorEvent(BaseEvent):
    """Event emitted when an unrecoverable failure or tracked error takes place."""
    type: EventType = Field(default=EventType.ERROR)
    payload: ErrorPayload


class AskHumanPayload(BaseModel):
    """Payload requesting interactive human feedback, authorization, or input."""
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
    payload: AskHumanPayload


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
]