"""
PELDRUN Core Model Context Protocol (MCP) Client.
Provides asynchronous JSON-RPC client integration, dynamic MCP tool generation,
and seamless compatibility with BaseTool and ToolRegistry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel, Field

from peldrun.tools.base import BaseTool, ToolResult

logger = logging.getLogger("peldrun.tools.mcp_client")


class MCPToolDefinition(BaseModel):
    """Metadata describing an external tool exposed by an MCP server."""
    name: str = Field(..., description="Unique tool identifier")
    description: str = Field(default="", description="Functional description of tool")
    inputSchema: Dict[str, Any] = Field(default_factory=dict, description="JSON Schema of parameters")


class DynamicMCPTool(BaseTool):
    """
    Bridge tool dynamically wrapping a remote MCP server tool.
    Inherits from BaseTool to ensure 100% interoperability with ToolRegistry.
    """

    def __init__(
        self,
        name: str,
        description: str,
        mcp_client: MCPClient,
        input_schema: Optional[Dict[str, Any]] = None,
        workspace_root: Optional[str] = None,
    ) -> None:
        super().__init__(workspace_root=workspace_root)
        self.name = name
        self.description = description
        self.mcp_client = mcp_client
        self._raw_schema = input_schema or {"type": "object", "properties": {}}

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert MCP tool JSON schema to standard OpenAI function-calling format."""
        parameters = {
            "type": self._raw_schema.get("type", "object"),
            "properties": self._raw_schema.get("properties", {}),
            "required": self._raw_schema.get("required", []),
        }
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description.strip(),
                "parameters": parameters,
            },
        }

    async def _arun(self, **kwargs: Any) -> ToolResult:
        """Forward invocation to the remote MCP server client."""
        return await self.mcp_client.call_tool(self.name, kwargs)


class MCPClient:
    """
    Asynchronous client communicating with an external Model Context Protocol server.
    Supports tool discovery and invocation over JSON-RPC 2.0.
    """

    def __init__(
        self,
        server_url: str,
        auth_token: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.auth_token = auth_token
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-initialize HTTP client session."""
        async with self._lock:
            if self._client is None or self._client.is_closed:
                headers = {"Content-Type": "application/json"}
                if self.auth_token:
                    headers["Authorization"] = f"Bearer {self.auth_token}"
                self._client = httpx.AsyncClient(
                    base_url=self.server_url,
                    headers=headers,
                    timeout=httpx.Timeout(self.timeout),
                )
            return self._client

    async def close(self) -> None:
        """Release underlying network connections."""
        async with self._lock:
            if self._client is not None and not self._client.is_closed:
                await self._client.aclose()
                self._client = None

    async def _send_jsonrpc_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Dispatch a JSON-RPC 2.0 compliant request payload."""
        client = await self._get_client()
        request_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }

        try:
            response = await client.post("", json=payload)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                error_obj = data["error"]
                raise RuntimeError(
                    f"MCP RPC Error [{error_obj.get('code')}]: {error_obj.get('message')}"
                )

            return data.get("result", {})
        except httpx.HTTPError as http_err:
            logger.error("HTTP error connecting to MCP server '%s': %s", self.server_url, http_err)
            raise ConnectionError(f"Failed to communicate with MCP server: {http_err}") from http_err

    async def discover_tools(self, workspace_root: Optional[str] = None) -> List[DynamicMCPTool]:
        """
        Query the MCP server for available tools via 'tools/list'
        and return a list of initialized DynamicMCPTool instances.
        """
        try:
            result = await self._send_jsonrpc_request("tools/list")
            raw_tools = result.get("tools", [])
            tools: List[DynamicMCPTool] = []

            for tool_data in raw_tools:
                definition = MCPToolDefinition(**tool_data)
                tool_instance = DynamicMCPTool(
                    name=definition.name,
                    description=definition.description,
                    mcp_client=self,
                    input_schema=definition.inputSchema,
                    workspace_root=workspace_root,
                )
                tools.append(tool_instance)

            logger.info("Discovered %d tools from MCP server '%s'", len(tools), self.server_url)
            return tools
        except Exception as ex:
            logger.exception("Failed to discover tools from MCP server '%s': %s", self.server_url, ex)
            return []

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """
        Execute an MCP tool remotely via 'tools/call'.
        Returns a structured ToolResult.
        """
        try:
            params = {
                "name": tool_name,
                "arguments": arguments,
            }
            result = await self._send_jsonrpc_request("tools/call", params)
            content_blocks = result.get("content", [])
            is_error = result.get("isError", False)

            # Combine textual responses
            text_outputs: List[str] = []
            for block in content_blocks:
                if block.get("type") == "text":
                    text_outputs.append(block.get("text", ""))
                else:
                    text_outputs.append(json.dumps(block, ensure_ascii=False))

            combined_output = "\n".join(text_outputs) if text_outputs else str(result)

            return ToolResult(
                output=combined_output,
                exit_code=1 if is_error else 0,
                is_error=is_error,
                metadata={"mcp_server": self.server_url, "raw_result": result},
            )
        except Exception as ex:
            logger.exception("MCP call failed for tool '%s': %s", tool_name, ex)
            return ToolResult(
                output=f"MCP Invocation Error: {str(ex)}",
                exit_code=1,
                is_error=True,
                metadata={"mcp_server": self.server_url, "error_type": type(ex).__name__},
            )