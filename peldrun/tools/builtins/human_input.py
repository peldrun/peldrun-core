"""
PELDRUN Core Interactive Human Input Tool.
Enables Human-in-the-Loop workflows by pausing agent execution, emitting ask_human events,
and asynchronously awaiting operator confirmation or feedback.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, ClassVar, Dict, List, Optional, Type
from pydantic import BaseModel, Field

from peldrun.events.emitter import EventEmitter
from peldrun.tools.base import BaseTool, ToolResult

logger = logging.getLogger("peldrun.tools.builtins.human_input")


class HumanInputArgs(BaseModel):
    """Input arguments schema for soliciting human operator input."""
    question: str = Field(
        ...,
        description="The specific question, prompt, or clarification required from the human operator."
    )
    options: Optional[List[str]] = Field(
        default=None,
        description="Optional list of predefined choices presented to the operator."
    )
    timeout_seconds: float = Field(
        default=300.0,
        ge=5.0,
        le=3600.0,
        description="Maximum seconds to pause execution waiting for human input before timing out."
    )


class HumanInputTool(BaseTool):
    """
    Core tool for interactive human feedback and operator authorization.
    Registers asynchronous futures resolved externally via web APIs or interactive CLI.
    """

    name: str = "ask_human"
    description: str = (
        "Solicit direct input, approval, or clarification from the human operator. "
        "Pauses agent execution until a response is submitted or timeout expires."
    )
    args_schema: Optional[Type[BaseModel]] = HumanInputArgs

    # Global registry mapping request_id -> pending asyncio.Future
    _pending_futures: ClassVar[Dict[str, asyncio.Future[str]]] = {}
    _pending_metadata: ClassVar[Dict[str, Dict[str, Any]]] = {}

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        emitter: Optional[EventEmitter] = None,
    ) -> None:
        super().__init__(workspace_root=workspace_root)
        self.emitter = emitter

    def set_emitter(self, emitter: EventEmitter) -> None:
        """Attach or update the active event emitter."""
        self.emitter = emitter

    @classmethod
    def submit_human_response(cls, request_id: str, response: str) -> bool:
        """
        External entrypoint to fulfill a pending human interaction request.
        Returns True if the matching request was found and unblocked, False otherwise.
        """
        future = cls._pending_futures.pop(request_id, None)
        cls._pending_metadata.pop(request_id, None)

        if future and not future.done():
            future.set_result(response)
            logger.info("Human response provided for request: %s", request_id)
            return True

        logger.warning("No active pending human input future found for request: %s", request_id)
        return False

    @classmethod
    def get_pending_requests(cls) -> Dict[str, Dict[str, Any]]:
        """Return a snapshot of all actively pending human input prompts."""
        return {req_id: dict(meta) for req_id, meta in cls._pending_metadata.items()}

    async def _arun(
        self,
        question: str,
        options: Optional[List[str]] = None,
        timeout_seconds: float = 300.0,
        **kwargs: Any,
    ) -> ToolResult:
        """Pause execution, broadcast event, and await human response asynchronously."""
        request_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()

        # Register pending future and metadata
        self._pending_futures[request_id] = future
        self._pending_metadata[request_id] = {
            "request_id": request_id,
            "question": question,
            "options": options,
            "timeout_seconds": timeout_seconds,
        }

        logger.info("Awaiting human input for request %s: '%s'", request_id, question)

        # Broadcast ask_human event via emitter if available
        if self.emitter:
            try:
                await self.emitter.emit_ask_human(
                    question=question,
                    options=options,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as emit_err:
                logger.warning("Failed to emit ask_human event: %s", emit_err)

        try:
            user_response = await asyncio.wait_for(future, timeout=timeout_seconds)
            return ToolResult(
                output=user_response,
                exit_code=0,
                is_error=False,
                metadata={
                    "request_id": request_id,
                    "question": question,
                    "options": options,
                    "responded": True,
                },
            )
        except asyncio.TimeoutError:
            self._pending_futures.pop(request_id, None)
            self._pending_metadata.pop(request_id, None)
            logger.warning("Human input request %s timed out after %ds", request_id, timeout_seconds)
            return ToolResult(
                output=f"Interactive request timed out after {timeout_seconds} seconds with no operator input.",
                exit_code=1,
                is_error=True,
                metadata={
                    "request_id": request_id,
                    "error_type": "HumanInputTimeout",
                    "timeout_seconds": timeout_seconds,
                },
            )
        except asyncio.CancelledError:
            self._pending_futures.pop(request_id, None)
            self._pending_metadata.pop(request_id, None)
            logger.info("Human input request %s was cancelled.", request_id)
            raise