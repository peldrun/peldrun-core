"""
PELDRUN Core Asynchronous Event Emitter.
Dispatches strictly typed lifecycle events to registered subscribers and SSE streams.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, AsyncIterator, Callable, Coroutine, Dict, List, Optional, Set, Union

from peldrun.events.schema import AgentEvent, BaseEvent, EventType

EventListener = Callable[[AgentEvent], Coroutine[Any, Any, None]]


class EventEmitter:
    """Thread-safe and asynchronous event broadcaster supporting typed listeners and SSE queues."""

    def __init__(self) -> None:
        self._listeners: Dict[str, List[EventListener]] = defaultdict(list)
        self._global_listeners: List[EventListener] = []
        self._sse_queues: Set[asyncio.Queue[AgentEvent]] = set()

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

    async def emit(self, event: AgentEvent) -> None:
        """Publish an event across matching listeners and active SSE stream queues."""
        key = event.type.value if hasattr(event.type, "value") else str(event.type)
        targets = list(self._listeners.get(key, [])) + list(self._global_listeners)

        for listener in targets:
            try:
                res = listener(event)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                # Listener resilience: isolate listener faults
                pass

        for q in list(self._sse_queues):
            await q.put(event)

    async def astream_sse(self) -> AsyncIterator[str]:
        """Asynchronously yield formatted SSE string frames as events occur."""
        q: asyncio.Queue[AgentEvent] = asyncio.Queue()
        self._sse_queues.add(q)
        try:
            while True:
                event = await q.get()
                yield event.to_sse()
                if (event.type.value if hasattr(event.type, "value") else str(event.type)) == EventType.FINAL.value:
                    break
        finally:
            self._sse_queues.discard(q)

    # Specialized convenience helpers
    async def emit_snapshot(self, snapshot: Dict[str, Any], **kwargs: Any) -> None:
        await self.emit(AgentEvent(type=EventType.SNAPSHOT, data={"snapshot": snapshot, **kwargs}))

    async def emit_step_start(self, step_number: int, **kwargs: Any) -> None:
        await self.emit(AgentEvent(type=EventType.STEP_START, data={"step_number": step_number, **kwargs}))

    async def emit_thought(self, thought: str = "", **kwargs: Any) -> None:
        data = {"thought": thought}
        data.update(kwargs)
        await self.emit(AgentEvent(type=EventType.THOUGHT, data=data))

    async def emit_tool_call(self, tool_name: str, arguments: Dict[str, Any], tool_call_id: str = "", **kwargs: Any) -> None:
        data = {"tool_name": tool_name, "arguments": arguments, "tool_call_id": tool_call_id}
        data.update(kwargs)
        await self.emit(AgentEvent(type=EventType.TOOL_CALL, data=data))

    async def emit_observation(self, output: Any, tool_name: str = "", tool_call_id: str = "", **kwargs: Any) -> None:
        data = {"output": output, "tool_name": tool_name, "tool_call_id": tool_call_id}
        data.update(kwargs)
        await self.emit(AgentEvent(type=EventType.OBSERVATION, data=data))

    async def emit_step_end(self, step_number: int, **kwargs: Any) -> None:
        await self.emit(AgentEvent(type=EventType.STEP_END, data={"step_number": step_number, **kwargs}))

    async def emit_ask_human(self, question: str, options: Optional[List[str]] = None, **kwargs: Any) -> None:
        data = {"question": question, "options": options or []}
        data.update(kwargs)
        await self.emit(AgentEvent(type=EventType.ASK_HUMAN, data=data))

    async def emit_error(self, error: str = "", **kwargs: Any) -> None:
        data = {"error": error}
        data.update(kwargs)
        await self.emit(AgentEvent(type=EventType.ERROR, data=data))

    async def emit_final(self, output: str = "", **kwargs: Any) -> None:
        data = {"output": output}
        data.update(kwargs)
        await self.emit(AgentEvent(type=EventType.FINAL, data=data))


__all__ = ["EventEmitter", "EventListener"]