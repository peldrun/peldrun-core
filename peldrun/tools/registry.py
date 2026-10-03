"""
PELDRUN Core Central Tool Registry.
Manages tool lifecycle, runtime discovery, parameter schema compilation, and scoped execution.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type, Union

from peldrun.tools.base import BaseTool, ToolResult

logger = logging.getLogger("peldrun.tools.registry")


class ToolRegistry:
    """
    Central catalog for managing, inspecting, and dispatching agent tools.
    Propagates workspace roots and generates OpenAI function schemas.
    """

    def __init__(self, workspace_root: Optional[str] = None) -> None:
        self._tools: Dict[str, BaseTool] = {}
        self._workspace_root: Optional[str] = workspace_root

    @property
    def workspace_root(self) -> Optional[str]:
        """Return current bounded workspace root directory."""
        return self._workspace_root

    def set_workspace_root(self, workspace_root: str) -> None:
        """
        Update the workspace root and propagate it across all registered tools.
        """
        self._workspace_root = workspace_root
        for tool in self._tools.values():
            tool.set_workspace(workspace_root)

    def register(self, tool: Union[BaseTool, Type[BaseTool]]) -> BaseTool:
        """
        Register a tool instance or tool class.
        Instantiates classes automatically and injects current workspace root.
        """
        if isinstance(tool, type) and issubclass(tool, BaseTool):
            tool_instance = tool(workspace_root=self._workspace_root)
        elif isinstance(tool, BaseTool):
            tool_instance = tool
            if self._workspace_root and not tool_instance.workspace_root:
                tool_instance.set_workspace(self._workspace_root)
        else:
            raise TypeError(f"Object {tool} must be an instance or subclass of BaseTool.")

        if not tool_instance.name:
            raise ValueError("Cannot register a tool with an empty name identifier.")

        self._tools[tool_instance.name] = tool_instance
        logger.debug("Registered tool: '%s'", tool_instance.name)
        return tool_instance

    def unregister(self, tool_name: str) -> Optional[BaseTool]:
        """Remove a registered tool from the registry."""
        removed = self._tools.pop(tool_name, None)
        if removed:
            logger.debug("Unregistered tool: '%s'", tool_name)
        return removed

    def get(self, tool_name: str) -> Optional[BaseTool]:
        """Retrieve a registered tool instance by name."""
        return self._tools.get(tool_name)

    def has_tool(self, tool_name: str) -> bool:
        """Check whether a tool with the given name is registered."""
        return tool_name in self._tools

    def list_tools(self) -> List[BaseTool]:
        """Return all currently registered tool instances."""
        return list(self._tools.values())

    def get_tool_names(self) -> List[str]:
        """Return a list of registered tool identifier names."""
        return list(self._tools.keys())

    def get_openai_schemas(self, tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Compile OpenAI-compatible function calling schemas for specified tools (or all if None).
        """
        schemas: List[Dict[str, Any]] = []
        targets = tool_names if tool_names is not None else list(self._tools.keys())

        for name in targets:
            tool = self._tools.get(name)
            if tool:
                schemas.append(tool.to_openai_schema())
            else:
                logger.warning("Requested schema for unknown tool: '%s'", name)

        return schemas

    async def aexecute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """
        Asynchronously invoke a registered tool by its name.
        Returns a failed ToolResult if the tool is not registered.
        """
        tool = self.get(tool_name)
        if not tool:
            msg = f"Tool '{tool_name}' is not registered."
            logger.error(msg)
            return ToolResult(
                output=msg,
                exit_code=1,
                is_error=True,
                metadata={"error_type": "ToolNotFoundError"},
            )

        return await tool.aexecute(**kwargs)

    def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """
        Synchronously invoke a registered tool by its name.
        """
        tool = self.get(tool_name)
        if not tool:
            msg = f"Tool '{tool_name}' is not registered."
            logger.error(msg)
            return ToolResult(
                output=msg,
                exit_code=1,
                is_error=True,
                metadata={"error_type": "ToolNotFoundError"},
            )

        return tool.execute(**kwargs)