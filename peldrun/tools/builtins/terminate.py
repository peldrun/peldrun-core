from __future__ import annotations

from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, Field

from peldrun.tools.base import BaseTool, ToolResult


class TerminateParameters(BaseModel):
    status: str = Field(
        default="success",
        description="The final status of the task execution (e.g. 'success', 'failure').",
    )
    message: str = Field(
        default="",
        description="Summary or final message describing the result.",
    )


class TerminateTool(BaseTool):
    """Signals the agent runner that the task execution has been completed."""

    name: str = "terminate"
    description: str = "Call this tool to terminate the current execution when the task is accomplished or cannot proceed."
    args_schema: Optional[Type[BaseModel]] = TerminateParameters

    async def _arun(self, status: str = "success", message: str = "", **kwargs: Any) -> ToolResult:
        output_payload = f"Terminated with status '{status}': {message}".strip()
        return ToolResult(
            output=output_payload,
            exit_code=0,
            is_error=False,
            metadata={"status": status, "is_terminal": True},
        )


__all__ = ["TerminateParameters", "TerminateTool"]