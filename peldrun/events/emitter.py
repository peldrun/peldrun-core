"""
PELDRUN Core Asynchronous Event Emitter.
Dispatches strictly typed lifecycle events to registered subscribers and SSE streams
with thread-safe and coroutine-safe monotonic sequence numbering and multi-run isolation.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import defaultdict
from typing import Any, AsyncIterator, Callable, Coroutine, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4

from peldrun.events.schema import EventType, PeldrunEvent

EventListener = Callable[[PeldrunEvent], Coroutine[Any, Any, None]]


class EventEmitter:
    """Thread-safe and coroutine-safe event broadcaster supporting sequence tracking and SSE queues."""

    def __init__(self, run_id: Optional[Union[UUID, str]] = None) -> None:
        self._run_id: UUID = self._coerce_uuid(run_id) if run_id else uuid4()
        self._sequence_counter: int = 0
        self._last_emitted_sequence: int = 0
        self._seq_lock = threading.Lock()
        self._lock = asyncio.Lock()
        self._listeners: Dict[str, List[EventListener]] = defaultdict(list)
        self._global_listeners: List[EventListener] = []
        self._sse_queues: Set[asyncio.Queue[PeldrunEvent]] = set()

    @staticmethod
    def _coerce_uuid(value: Union[UUID, str]) -> UUID:
        """Convert string or UUID instance to strict UUID, raising ValueError on invalid format."""
        if isinstance(value, UUID):
            return value
        if isinstance(value, str):
            return UUID(value)
        raise ValueError(f"Value must be a valid UUID or UUID string, got {type(value).__name__}")

    @property
    def run_id(self) -> UUID:
        """Active execution or session run identifier."""
        return self._run_id

    @run_id.setter
    def run_id(self, val: Union[UUID, str]) -> None:
        self._run_id = self._coerce_uuid(val)

    @property
    def current_sequence(self) -> int:
        """Current monotonic sequence index."""
        with self._seq_lock:
            return self._sequence_counter

    def reset_sequence(self, start: int = 0) -> None:
        """Reset the sequence counter to a designated baseline."""
        with self._seq_lock:
            self._sequence_counter = start
            self._last_emitted_sequence = start

    def create_event(
        self,
        event_type: Union[EventType, str],
        payload: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        step: int = 0,
        run_id: Optional[Union[UUID, str]] = None,
        sequence: Optional[int] = None,
        replay: bool = False,
    ) -> PeldrunEvent:
        """
        Factory method producing a strictly valid PeldrunEvent.
        Explicit sequence override is forbidden unless replay=True and sequence is strictly monotonic.
        """
        with self._seq_lock:
            if sequence is not None:
                if not replay:
                    raise ValueError(
                        "Explicit sequence assignment is forbidden outside replay mode. Pass replay=True to import historical events."
                    )
                if sequence <= self._sequence_counter:
                    raise ValueError(
                        f"Replay sequence {sequence} violates monotonicity; current sequence counter is {self._sequence_counter}."
                    )
                self._sequence_counter = sequence
                assigned_sequence = sequence
            else:
                self._sequence_counter += 1
                assigned_sequence = self._sequence_counter

        resolved_type = EventType(event_type) if not isinstance(event_type, EventType) else event_type
        target_run_id = self._coerce_uuid(run_id) if run_id else self._run_id

        return PeldrunEvent(
            version=1,
            event_id=uuid4(),
            sequence=assigned_sequence,
            run_id=target_run_id,
            timestamp=time.time(),
            step=step,
            type=resolved_type,
            payload=payload or {},
            metadata=metadata or {},
        )

    def subscribe(self, event_type: Union[EventType, str], listener: EventListener) -> None:
        """Register an async callback for a specific event type."""
        key = event_type.value if isinstance(event_type, EventType) else str(event_type)
        if listener not in self._listeners[key]:
            self._listeners[key].append(listener)

    def subscribe_all(self, listener: EventListener) -> None:
        """Register an async callback to receive all dispatched events."""
        if listener not in self._global_listeners:
            self._global_listeners.append(listener)

    def unsubscribe(self, event_type: Union[EventType, str], listener: EventListener) -> None:
        """Unregister a listener from a specific type and global list."""
        key = event_type.value if isinstance(event_type, EventType) else str(event_type)
        if listener in self._listeners[key]:
            self._listeners[key].remove(listener)
        if listener in self._global_listeners:
            self._global_listeners.remove(listener)

    async def emit(self, event: PeldrunEvent, allow_external_sequence: bool = False) -> None:
        """
        Publish an event across matching listeners and active SSE queues.
        Guarantees thread-safe and coroutine-safe strictly monotonic sequence allocation.
        """
        async with self._lock:
            with self._seq_lock:
                if event.sequence <= 0:
                    self._sequence_counter += 1
                    event.sequence = self._sequence_counter
                elif not allow_external_sequence:
                    if event.sequence <= self._last_emitted_sequence:
                        self._sequence_counter = max(self._sequence_counter, self._last_emitted_sequence) + 1
                        event.sequence = self._sequence_counter
                    elif event.sequence > self._sequence_counter:
                        self._sequence_counter = event.sequence
                else:
                    if event.sequence > self._sequence_counter:
                        self._sequence_counter = event.sequence

                self._last_emitted_sequence = max(self._last_emitted_sequence, event.sequence)

            key = event.type.value if hasattr(event.type, "value") else str(event.type)
            targets = list(self._listeners.get(key, [])) + list(self._global_listeners)
            queues = list(self._sse_queues)

        for listener in targets:
            try:
                res = listener(event)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                # Listener resilience: isolate subscriber faults
                pass

        for q in queues:
            await q.put(event)

    async def astream_sse(self) -> AsyncIterator[str]:
        """Asynchronously yield formatted SSE string frames as events occur."""
        q: asyncio.Queue[PeldrunEvent] = asyncio.Queue()
        self._sse_queues.add(q)
        try:
            while True:
                event = await q.get()
                yield event.to_sse()
                if (event.type.value if hasattr(event.type, "value") else str(event.type)) == EventType.FINAL.value:
                    break
        finally:
            self._sse_queues.discard(q)

    # Convenience domain helpers with automatic envelope packaging
    async def emit_snapshot(self, snapshot: Dict[str, Any], step: int = 0, **kwargs: Any) -> None:
        payload = {"snapshot": snapshot, **kwargs}
        event = self.create_event(EventType.SNAPSHOT, payload=payload, step=step)
        await self.emit(event)

    async def emit_step_start(self, step_number: int, **kwargs: Any) -> None:
        payload = {"step_number": step_number, **kwargs}
        event = self.create_event(EventType.STEP_START, payload=payload, step=step_number)
        await self.emit(event)

    async def emit_thought(self, thought: str = "", step: int = 0, **kwargs: Any) -> None:
        payload = {"thought": thought, **kwargs}
        event = self.create_event(EventType.THOUGHT, payload=payload, step=step)
        await self.emit(event)

    async def emit_agent_activity(self, message: str, phase: str = "general", step: int = 0, **kwargs: Any) -> None:
        payload = {"message": message, "phase": phase, **kwargs}
        event = self.create_event(EventType.AGENT_ACTIVITY, payload=payload, step=step)
        await self.emit(event)

    async def emit_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tool_call_id: str = "",
        step: int = 0,
        **kwargs: Any,
    ) -> None:
        payload = {"tool_name": tool_name, "arguments": arguments, "tool_call_id": tool_call_id, **kwargs}
        event = self.create_event(EventType.TOOL_CALL, payload=payload, step=step)
        await self.emit(event)

    async def emit_observation(
        self,
        output: Any,
        tool_name: str = "",
        tool_call_id: str = "",
        exit_code: int = 0,
        is_error: bool = False,
        step: int = 0,
        **kwargs: Any,
    ) -> None:
        payload = {
            "output": output,
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "exit_code": exit_code,
            "is_error": is_error,
            **kwargs,
        }
        event = self.create_event(EventType.OBSERVATION, payload=payload, step=step)
        await self.emit(event)

    async def emit_step_end(self, step_number: int, elapsed_seconds: float = 0.0, **kwargs: Any) -> None:
        payload = {"step_number": step_number, "elapsed_seconds": elapsed_seconds, **kwargs}
        event = self.create_event(EventType.STEP_END, payload=payload, step=step_number)
        await self.emit(event)

    async def emit_ask_human(self, question: str, options: Optional[List[str]] = None, step: int = 0, **kwargs: Any) -> None:
        payload = {"question": question, "options": options or [], **kwargs}
        event = self.create_event(EventType.ASK_HUMAN, payload=payload, step=step)
        await self.emit(event)

    async def emit_error(self, error: str = "", step: int = 0, **kwargs: Any) -> None:
        payload = {"error": error, **kwargs}
        event = self.create_event(EventType.ERROR, payload=payload, step=step)
        await self.emit(event)

    async def emit_final(self, output: str = "", step: int = 0, **kwargs: Any) -> None:
        payload = {"output": output, **kwargs}
        event = self.create_event(EventType.FINAL, payload=payload, step=step)
        await self.emit(event)


__all__ = ["EventEmitter", "EventListener"]