"""
PELDRUN Core Tool Collection Architecture.
Aggregates and executes BaseTool instances with flexible invocation resilience.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from peldrun.tools.base import BaseTool, ToolResult


class ToolCollection:
    """Manages a collection of BaseTool instances for tool-calling agents."""

    def __init__(self, tools: Optional[List[BaseTool]] = None) -> None:
        self._tools: Dict[str, BaseTool] = {}
        if tools:
            for tool in tools:
                self.add_tool(tool)

    def add_tool(self, tool: BaseTool) -> None:
        """Register a tool in the collection."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Retrieve a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[BaseTool]:
        """Return all tools in the collection."""
        return list(self._tools.values())

    def to_openai_schemas(self) -> List[Dict[str, Any]]:
        """Export all tools as OpenAI-compatible function definitions."""
        schemas: List[Dict[str, Any]] = []
        for tool in self._tools.values():
            if hasattr(tool, "to_openai_schema"):
                schemas.append(tool.to_openai_schema())
            else:
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": getattr(tool, "parameters", {}),
                    },
                })
        return schemas

    async def execute(self, name: str, arguments: Any) -> ToolResult:
        """Execute a tool by name, handling argument parsing and calling aexecute or _arun."""
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(
                output=f"Tool '{name}' not found in ToolCollection.",
                exit_code=1,
                is_error=True,
            )

        parsed_args: Dict[str, Any]
        if isinstance(arguments, str):
            try:
                parsed_args = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                return ToolResult(
                    output=f"Failed to parse arguments as JSON: {exc}",
                    exit_code=1,
                    is_error=True,
                )
        elif isinstance(arguments, dict):
            parsed_args = arguments
        else:
            parsed_args = {}

        try:
            if hasattr(tool, "aexecute"):
                return await tool.aexecute(**parsed_args)
            elif hasattr(tool, "_arun"):
                return await tool._arun(**parsed_args)
            elif hasattr(tool, "execute"):
                res = tool.execute(**parsed_args)
                if hasattr(res, "__await__"):
                    return await res
                return res
            else:
                return ToolResult(
                    output=f"Tool '{name}' has no executable entrypoint.",
                    exit_code=1,
                    is_error=True,
                )
        except Exception as exc:
            return ToolResult(
                output=f"Tool '{name}' execution raised an exception: {exc}",
                exit_code=1,
                is_error=True,
            )


__all__ = ["ToolCollection"]