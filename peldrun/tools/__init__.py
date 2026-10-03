"""
PELDRUN Core Tools Subsystem.
Exports base tool classes, central registry, MCP client integration,
and standard built-in tools with a convenience registry factory.
"""

from typing import Optional

from peldrun.events.emitter import EventEmitter
from peldrun.tools.base import BaseTool, ToolResult
from peldrun.tools.builtins import (
    FileOpsArgs,
    FileOpsTool,
    HumanInputArgs,
    HumanInputTool,
    ShellExecArgs,
    ShellExecTool,
    WebSearchArgs,
    WebSearchTool,
    get_builtin_tools,
)
from peldrun.tools.mcp_client import DynamicMCPTool, MCPClient, MCPToolDefinition
from peldrun.tools.registry import ToolRegistry


def create_default_registry(
    workspace_root: Optional[str] = None,
    emitter: Optional[EventEmitter] = None,
) -> ToolRegistry:
    """
    Instantiate and return a central ToolRegistry pre-populated with standard built-in tools.
    Binds the provided workspace root and connects interactive tools to the event emitter.
    """
    registry = ToolRegistry(workspace_root=workspace_root)
    for tool in get_builtin_tools(workspace_root=workspace_root, emitter=emitter):
        registry.register(tool)
    return registry


__all__ = [
    # Base abstractions
    "BaseTool",
    "ToolResult",
    # Registry & factory
    "ToolRegistry",
    "create_default_registry",
    # MCP Integration
    "MCPClient",
    "DynamicMCPTool",
    "MCPToolDefinition",
    # Built-in tools and schemas
    "FileOpsTool",
    "FileOpsArgs",
    "ShellExecTool",
    "ShellExecArgs",
    "WebSearchTool",
    "WebSearchArgs",
    "HumanInputTool",
    "HumanInputArgs",
    "get_builtin_tools",
]