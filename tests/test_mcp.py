"""
PELDRUN Core Model Context Protocol (MCP) Subsystem Test Suite.
Verifies MCPToolDefinition, DynamicMCPTool, MCPClient, StdioTransport,
and SSETransport with hermetic in-memory mock transports.
"""

from __future__ import annotations

import json
from typing import Any, Dict
import httpx
import pytest

from peldrun.tools.mcp_client import DynamicMCPTool, MCPClient, MCPToolDefinition
from peldrun.tools.transports.sse import SSETransport
from peldrun.tools.transports.stdio import StdioTransport


def test_mcp_tool_definition_model() -> None:
    """Verify MCPToolDefinition validation and default attributes."""
    tool_def = MCPToolDefinition(
        name="database_query",
        description="Query local sqlite database.",
        inputSchema={
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"],
        },
    )
    assert tool_def.name == "database_query"
    assert tool_def.description == "Query local sqlite database."
    assert "sql" in tool_def.inputSchema["properties"]


def test_dynamic_mcp_tool_schema_and_properties() -> None:
    """Verify DynamicMCPTool translates raw schema into valid OpenAI format."""
    dummy_client = MCPClient(server_url="http://localhost:8000")
    tool = DynamicMCPTool(
        name="calculate_sum",
        description="Add two numbers together.",
        mcp_client=dummy_client,
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    )

    schema = tool.to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "calculate_sum"
    assert schema["function"]["description"] == "Add two numbers together."
    assert "a" in schema["function"]["parameters"]["properties"]
    assert "b" in schema["function"]["parameters"]["required"]


@pytest.mark.asyncio
async def test_mcp_client_discover_tools_success() -> None:
    """Verify discover_tools queries 'tools/list' and constructs DynamicMCPTool instances."""
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["method"] == "tools/list"

        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {
                    "tools": [
                        {
                            "name": "remote_echo",
                            "description": "Echo input text",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"msg": {"type": "string"}},
                            },
                        }
                    ]
                },
            },
        )

    transport = httpx.MockTransport(handler)
    client = MCPClient(server_url="http://mcp-server:8000", http_transport=transport)

    try:
        tools = await client.discover_tools()
        assert len(tools) == 1
        assert tools[0].name == "remote_echo"
        assert tools[0].description == "Echo input text"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_client_call_tool_and_text_aggregation() -> None:
    """Verify call_tool aggregates textual blocks and returns valid ToolResult."""
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["method"] == "tools/call"
        assert body["params"]["name"] == "grep_files"

        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {
                    "content": [
                        {"type": "text", "text": "Match 1: app.py:10"},
                        {"type": "text", "text": "Match 2: runner.py:45"},
                    ],
                    "isError": False,
                },
            },
        )

    transport = httpx.MockTransport(handler)
    client = MCPClient(server_url="http://mcp-server:8000", http_transport=transport)

    try:
        result = await client.call_tool("grep_files", {"pattern": "run"})
        assert result.exit_code == 0
        assert result.is_error is False
        assert "Match 1: app.py:10" in result.output
        assert "Match 2: runner.py:45" in result.output
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_client_rpc_error_handling() -> None:
    """Verify MCPClient handles JSON-RPC error packets gracefully."""
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "error": {"code": -32601, "message": "Method not found"},
            },
        )

    transport = httpx.MockTransport(handler)
    client = MCPClient(server_url="http://mcp-server:8000", http_transport=transport)

    try:
        result = await client.call_tool("unknown_tool", {})
        assert result.exit_code == 1
        assert result.is_error is True
        assert "Method not found" in result.output
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_sse_transport_lifecycle() -> None:
    """Verify SSETransport connects, sends requests, and formats output."""
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["method"] == "ping"

        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": body["id"], "result": {"pong": True}},
        )

    transport = httpx.MockTransport(handler)
    sse = SSETransport(endpoint_url="http://localhost:9000/mcp", transport=transport)
    client = MCPClient(transport=sse)

    try:
        res = await client._send_jsonrpc_request("ping")
        assert res.get("pong") is True
    finally:
        await client.close()