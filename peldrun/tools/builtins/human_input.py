"""
PELDRUN Core Human Input Tool.
Provides asynchronous suspension for human approval and feedback loops
compatible with CLI, Web API, and Event registry.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from peldrun.tools.base import BaseTool, ToolResult


class HumanInputRegistry:
    """Centralized pending requests manager for human-in-the-loop approvals."""
    _pending_requests: Dict[str, asyncio.Future[str]] = {}

    @classmethod
    def register_request(cls, request_id: str) -> asyncio.Future[str]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        cls._pending_requests[request_id] = future
        return future

    @classmethod
    def resolve_request(cls, request_id: str, response: str) -> bool:
        future = cls._pending_requests.pop(request_id, None)
        if future and not future.done():
            future.set_result(response)
            return True
        return False

    @classmethod
    def cancel_request(cls, request_id: str) -> None:
        future = cls._pending_requests.pop(request_id, None)
        if future and not future.done():
            future.cancel()


class HumanInputArgs(BaseModel):
    """Pydantic schema defining arguments for HumanInputTool."""
    query: str = Field(
        ...,
        description="The question or clarification needed from the human operator.",
    )
    timeout_seconds: Optional[int] = Field(
        default=600,
        description="Maximum wait time in seconds for the operator response.",
    )


# Backward-compatibility alias
HumanInputParameters = HumanInputArgs


class HumanInputTool(BaseTool):
    """Tool that suspends agent execution until a response is received from the user."""

    name: str = "human_input"
    description: str = "Ask the human operator for feedback, confirmation, or clarification when required."
    args_schema = HumanInputArgs

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        emitter: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(workspace_root=workspace_root)
        self.workspace_root = workspace_root
        self.emitter = emitter

    async def _arun(
        self,
        query: str,
        timeout_seconds: Optional[int] = 600,
        **kwargs: Any,
    ) -> ToolResult:
        """Internal asynchronous tool execution logic satisfying BaseTool contract."""
        request_id = str(uuid.uuid4())
        timeout_val = float(timeout_seconds) if timeout_seconds else 600.0

        future = HumanInputRegistry.register_request(request_id)

        try:
            response = await asyncio.wait_for(future, timeout=timeout_val)
            return ToolResult(
                output=f"Human Response: {response}",
                exit_code=0,
                is_error=False,
                metadata={"request_id": request_id, "approved": True},
            )
        except asyncio.TimeoutError:
            HumanInputRegistry.cancel_request(request_id)
            return ToolResult(
                output=f"Human response timed out after {timeout_val} seconds.",
                exit_code=1,
                is_error=True,
                metadata={"request_id": request_id, "timeout": True},
            )
        except asyncio.CancelledError:
            HumanInputRegistry.cancel_request(request_id)
            raise

    def _run(self, query: str, timeout_seconds: Optional[int] = 600, **kwargs: Any) -> ToolResult:
        """Fallback synchronous execution."""
        raise NotImplementedError("HumanInputTool requires asynchronous execution.")


__all__ = [
    "HumanInputArgs",
    "HumanInputParameters",
    "HumanInputRegistry",
    "HumanInputTool",
]