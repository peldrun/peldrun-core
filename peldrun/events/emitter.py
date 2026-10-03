"""
PELDRUN Core Asynchronous Event Bus and Streaming Emitter.
Provides high-performance, non-blocking event distribution and SSE streaming bridges.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set, Union

from peldrun.events.schema import (
    AskHumanEvent,
    AskHumanPayload,
    BaseEvent,
    ErrorEvent,
    ErrorPayload,
    EventType,
    FinalEvent,
    FinalPayload,
    ObservationEvent,
    ObservationPayload,
    PeldrunEvent,
    SnapshotEvent,
    SnapshotPayload,
    StepEndEvent,
    StepEndPayload,
    StepStartEvent,
    StepStartPayload,
    ThoughtEvent,
    ThoughtPayload,
    ToolCallEvent,
    ToolCallPayload,
)

logger = logging.getLogger("peldrun.events.emitter")


class EventEmitter:
    """
    Central asynchronous event dispatcher for PELDRUN engine workflows.
    Enables zero monkey-patching event observation, history retention, and SSE generators.
    """

    def __init__(self, run_id: Optional[str] = None, max_history: int = 1000) -> None:
        self.run_id = run_id
        self.max_history = max_history
        self._history: List[PeldrunEvent] = []
        self._subscribers: Set[asyncio.Queue[PeldrunEvent]] = set()
        self._handlers: Dict[Optional[EventType], List[Callable[[PeldrunEvent], Any]]] = {}
        self._lock = asyncio.Lock()

    @property
    def history(self) -> List[PeldrunEvent]:
        """Return a copy of the recorded event timeline."""
        return list(self._history)

    def subscribe(self) -> asyncio.Queue[PeldrunEvent]:
        """
        Create and register a listener queue for streaming events.
        Caller is responsible for calling unsubscribe() when done.
        """
        queue: asyncio.Queue[PeldrunEvent] = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[PeldrunEvent]) -> None:
        """Remove an active subscriber queue."""
        self._subscribers.discard(queue)

    def on(
        self,
        event_type: Optional[EventType],
        handler: Callable[[PeldrunEvent], Any],
    ) -> None:
        """
        Register a callback handler for a specific event type, or None for all events.
        Supports both synchronous functions and async coroutines.
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    async def emit(self, event: PeldrunEvent) -> None:
        """
        Broadcast an event to all subscriber queues and registered callback handlers.
        Guarantees history persistence within the configured window.
        """
        if not event.run_id and self.run_id:
            event.run_id = self.run_id

        async with self._lock:
            self._history.append(event)
            if len(self._history) > self.max_history:
                self._history.pop(0)

        # Distribute to subscriber queues
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Subscriber queue is full. Event %s was dropped.", event.event_id)

        # Execute registered type-specific callbacks
        callbacks = list(self._handlers.get(event.type, []))
        # Execute global callbacks
        callbacks.extend(self._handlers.get(None, []))

        for cb in callbacks:
            try:
                if inspect.iscoroutinefunction(cb):
                    await cb(event)
                else:
                    cb(event)
            except Exception as ex:
                logger.exception("Error executing event callback for %s: %s", event.type, ex)

    async def sse_stream(
        self,
        replay_history: bool = False,
    ) -> AsyncIterator[str]:
        """
        Async generator yielding Server-Sent Events (SSE) compliant strings.
        Format: data: {"type": ..., ...}\\n\\n
        """
        queue = self.subscribe()
        try:
            if replay_history:
                for past_event in self.history:
                    payload_json = json.dumps(past_event.to_sse_payload(), ensure_ascii=False)
                    yield f"data: {payload_json}\n\n"

            while True:
                event = await queue.get()
                payload_json = json.dumps(event.to_sse_payload(), ensure_ascii=False)
                yield f"data: {payload_json}\n\n"
                queue.task_done()

                # Terminate stream on terminal execution events
                if event.type in (EventType.FINAL, EventType.ERROR):
                    break
        finally:
            self.unsubscribe(queue)

    # --- Convenience emission helpers ---

    async def emit_snapshot(
        self,
        status: str,
        step: int = 0,
        agent_name: Optional[str] = None,
        workspace_files: Optional[List[str]] = None,
        active_tools: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SnapshotEvent:
        event = SnapshotEvent(
            run_id=self.run_id,
            step=step,
            payload=SnapshotPayload(
                status=status,
                step=step,
                agent_name=agent_name,
                workspace_files=workspace_files or [],
                active_tools=active_tools or [],
                metadata=metadata or {},
            ),
        )
        await self.emit(event)
        return event

    async def emit_step_start(
        self,
        step: int,
        agent_name: str,
        input_prompt: Optional[str] = None,
    ) -> StepStartEvent:
        event = StepStartEvent(
            run_id=self.run_id,
            step=step,
            payload=StepStartPayload(
                step=step,
                agent_name=agent_name,
                input_prompt=input_prompt,
            ),
        )
        await self.emit(event)
        return event

    async def emit_thought(
        self,
        content: str,
        step: Optional[int] = None,
        is_delta: bool = False,
    ) -> ThoughtEvent:
        event = ThoughtEvent(
            run_id=self.run_id,
            step=step,
            payload=ThoughtPayload(content=content, is_delta=is_delta),
        )
        await self.emit(event)
        return event

    async def emit_tool_call(
        self,
        call_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        step: Optional[int] = None,
    ) -> ToolCallEvent:
        event = ToolCallEvent(
            run_id=self.run_id,
            step=step,
            payload=ToolCallPayload(
                call_id=call_id,
                tool_name=tool_name,
                arguments=arguments,
            ),
        )
        await self.emit(event)
        return event

    async def emit_observation(
        self,
        call_id: str,
        tool_name: str,
        output: Any,
        step: Optional[int] = None,
        exit_code: int = 0,
        is_error: bool = False,
    ) -> ObservationEvent:
        event = ObservationEvent(
            run_id=self.run_id,
            step=step,
            payload=ObservationPayload(
                call_id=call_id,
                tool_name=tool_name,
                output=output,
                exit_code=exit_code,
                is_error=is_error,
            ),
        )
        await self.emit(event)
        return event

    async def emit_step_end(
        self,
        step: int,
        elapsed_seconds: float = 0.0,
        success: bool = True,
    ) -> StepEndEvent:
        event = StepEndEvent(
            run_id=self.run_id,
            step=step,
            payload=StepEndPayload(
                step=step,
                elapsed_seconds=elapsed_seconds,
                success=success,
            ),
        )
        await self.emit(event)
        return event

    async def emit_final(
        self,
        content: str,
        deliverables: Optional[List[str]] = None,
        total_steps: int = 1,
        total_tokens: Optional[int] = None,
        step: Optional[int] = None,
    ) -> FinalEvent:
        event = FinalEvent(
            run_id=self.run_id,
            step=step,
            payload=FinalPayload(
                content=content,
                deliverables=deliverables or [],
                total_steps=total_steps,
                total_tokens=total_tokens,
            ),
        )
        await self.emit(event)
        return event

    async def emit_error(
        self,
        message: str,
        error_type: str = "RuntimeError",
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = False,
        step: Optional[int] = None,
    ) -> ErrorEvent:
        event = ErrorEvent(
            run_id=self.run_id,
            step=step,
            payload=ErrorPayload(
                message=message,
                error_type=error_type,
                details=details,
                recoverable=recoverable,
            ),
        )
        await self.emit(event)
        return event

    async def emit_ask_human(
        self,
        question: str,
        options: Optional[List[str]] = None,
        timeout_seconds: Optional[float] = 300.0,
        step: Optional[int] = None,
    ) -> AskHumanEvent:
        event = AskHumanEvent(
            run_id=self.run_id,
            step=step,
            payload=AskHumanPayload(
                question=question,
                options=options,
                timeout_seconds=timeout_seconds,
            ),
        )
        await self.emit(event)
        return event