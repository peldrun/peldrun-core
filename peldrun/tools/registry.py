"""
PELDRUN Core Tool Registry.
Central catalog for registering, discovering, generating OpenAI schemas, and dispatching tool executions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional
from peldrun.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central repository storing and orchestrating tool invocations."""

    def __init__(self, workspace_root: Optional[str] = None) -> None:
        self.workspace_root = workspace_root
        self._tools: Dict[str, BaseTool] = {}

    def __contains__(self, name: str) -> bool:
        """Support 'in' operator to check tool existence."""
        return name in self._tools

    def __iter__(self) -> Iterator[str]:
        return iter(self._tools)

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        """Fetch tool instance by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """Return names of all registered tools."""
        return list(self._tools.keys())

    def get_openai_schemas(self) -> List[Dict[str, Any]]:
        """Return OpenAI tool/function definitions for all registered tools."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    async def aexecute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Asynchronously dispatch execution to the targeted tool."""
        tool = self.get(tool_name)
        if not tool:
            logger.error(f"Tool '{tool_name}' is not registered.")
            return ToolResult(
                output=f"Tool '{tool_name}' not found / is not registered.",
                exit_code=1,
                is_error=True,
                metadata={"error_type": "ToolNotFoundError"},
            )
        return await tool.aexecute(**kwargs)


__all__ = ["ToolRegistry"]