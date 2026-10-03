"""
PELDRUN Core Tools Subsystem Test Suite.
Verifies ToolRegistry registration/dispatch, scoped FileOpsTool boundaries,
ShellExecTool subprocess confinement, HumanInputTool interactive unblocking,
WebSearchTool resilient parsing, and DynamicMCPTool schema conversion.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict
import httpx
import pytest

from peldrun.events.schema import EventType
from peldrun.tools.base import BaseTool, ToolResult
from peldrun.tools.builtins.file_ops import FileOpsTool
from peldrun.tools.builtins.human_input import HumanInputTool
from peldrun.tools.builtins.shell_exec import ShellExecTool
from peldrun.tools.builtins.web_search import WebSearchTool
from peldrun.tools.mcp_client import DynamicMCPTool, MCPClient
from peldrun.tools.registry import ToolRegistry
from tests.conftest import CapturedEventEmitter


class DummyTool(BaseTool):
    """Simple test tool implementation."""
    name: str = "dummy_tool"
    description: str = "A mock tool for testing registry mechanics."

    async def _arun(self, value: str = "default", **kwargs: Any) -> ToolResult:
        return ToolResult(output=f"Received: {value}", exit_code=0)


def test_tool_registry_registration_and_schemas(temp_workspace: Path) -> None:
    """Verify tool registration, schema synthesis, and dynamic lookups."""
    registry = ToolRegistry(workspace_root=str(temp_workspace))
    dummy = DummyTool(workspace_root=str(temp_workspace))

    registry.register(dummy)
    assert "dummy_tool" in registry
    assert registry.get("dummy_tool") is dummy

    schemas = registry.get_openai_schemas()
    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "dummy_tool"
    assert schemas[0]["function"]["description"] == dummy.description


@pytest.mark.asyncio
async def test_tool_registry_dispatch(temp_workspace: Path) -> None:
    """Verify registry dispatches execution to the corresponding registered tool."""
    registry = ToolRegistry(workspace_root=str(temp_workspace))
    registry.register(DummyTool(workspace_root=str(temp_workspace)))

    result = await registry.aexecute("dummy_tool", value="custom_val")
    assert result.exit_code == 0
    assert result.output == "Received: custom_val"

    missing_result = await registry.aexecute("non_existent_tool")
    assert missing_result.exit_code == 1
    assert missing_result.is_error is True
    assert "not found" in missing_result.output.lower()


@pytest.mark.asyncio
async def test_file_ops_tool_full_lifecycle(temp_workspace: Path) -> None:
    """Verify write, read, append, exists, list, and delete actions within workspace."""
    tool = FileOpsTool(workspace_root=str(temp_workspace))

    # 1. Write file
    write_res = await tool.aexecute(action="write", path="docs/readme.txt", content="Hello PELDRUN!")
    assert write_res.exit_code == 0
    assert not write_res.is_error
    assert "docs" in write_res.artifacts[0]

    # 2. Exists
    exists_res = await tool.aexecute(action="exists", path="docs/readme.txt")
    assert exists_res.output["exists"] is True
    assert exists_res.output["is_file"] is True

    # 3. Read
    read_res = await tool.aexecute(action="read", path="docs/readme.txt")
    assert read_res.output == "Hello PELDRUN!"

    # 4. Append
    append_res = await tool.aexecute(action="append", path="docs/readme.txt", content=" Appended text.")
    assert append_res.exit_code == 0

    read_updated = await tool.aexecute(action="read", path="docs/readme.txt")
    assert read_updated.output == "Hello PELDRUN! Appended text."

    # 5. List
    list_res = await tool.aexecute(action="list", path="docs")
    assert len(list_res.output) == 1
    assert list_res.output[0]["name"] == "readme.txt"

    # 6. Delete
    del_res = await tool.aexecute(action="delete", path="docs/readme.txt")
    assert del_res.exit_code == 0

    exists_after_del = await tool.aexecute(action="exists", path="docs/readme.txt")
    assert exists_after_del.output["exists"] is False


@pytest.mark.asyncio
async def test_file_ops_security_directory_traversal_blocked(temp_workspace: Path) -> None:
    """Verify FileOpsTool strictly blocks directory traversal attempts escaping workspace."""
    tool = FileOpsTool(workspace_root=str(temp_workspace))

    traversal_path = "../../outside_file.txt"
    res = await tool.aexecute(action="write", path=traversal_path, content="hacked")

    assert res.exit_code == 1
    assert res.is_error is True
    assert "access denied" in res.output.lower()


@pytest.mark.asyncio
async def test_shell_exec_tool_execution_and_cwd(temp_workspace: Path) -> None:
    """Verify command execution, stdout capture, and working directory isolation."""
    tool = ShellExecTool(workspace_root=str(temp_workspace))

    res = await tool.aexecute(command="python -c \"print('Peldrun Shell OK')\"")
    assert res.exit_code == 0
    assert not res.is_error
    assert "Peldrun Shell OK" in res.output

    sub_dir = temp_workspace / "sub_project"
    sub_dir.mkdir(parents=True, exist_ok=True)

    pwd_res = await tool.aexecute(
        command="python -c \"import os; print(os.path.basename(os.getcwd()))\"",
        working_subdir="sub_project",
    )
    assert pwd_res.exit_code == 0
    assert "sub_project" in pwd_res.output


@pytest.mark.asyncio
async def test_shell_exec_traversal_blocked(temp_workspace: Path) -> None:
    """Verify shell execution working directory escapes are blocked."""
    tool = ShellExecTool(workspace_root=str(temp_workspace))

    res = await tool.aexecute(command="dir", working_subdir="../../..")
    assert res.exit_code == 1
    assert res.is_error is True
    assert "access denied" in res.output.lower()


@pytest.mark.asyncio
async def test_web_search_tool_empty_query() -> None:
    """Verify WebSearchTool handles empty queries gracefully."""
    tool = WebSearchTool()
    res = await tool.aexecute(query="   ")
    assert res.exit_code == 1
    assert res.is_error is True
    assert "cannot be empty" in res.output.lower()


@pytest.mark.asyncio
async def test_web_search_tool_fallback_http_parsing() -> None:
    """Verify WebSearchTool parses HTML responses into structured results via mock transport."""
    mock_html = (
        '<html><body>'
        '<a class="result__url" href="/l/?uddg=https%3A%2F%2Fpeldrun.dev%2Fdocs"><b>PELDRUN Documentation</b></a>'
        '<a class="result__snippet" href="#">Autonomous agent framework documentation.</a>'
        '</body></html>'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=mock_html)

    transport = httpx.MockTransport(handler)
    tool = WebSearchTool(transport=transport)

    res = await tool.aexecute(query="peldrun docs", max_results=1)
    assert res.exit_code == 0
    assert res.is_error is False
    assert "PELDRUN Documentation" in res.output
    assert "https://peldrun.dev/docs" in res.output
    assert len(res.metadata.get("results", [])) == 1


@pytest.mark.asyncio
async def test_human_input_tool_interaction(event_collector: CapturedEventEmitter) -> None:
    """Verify HumanInputTool pauses, emits event, and unblocks upon operator response."""
    tool = HumanInputTool(emitter=event_collector)

    async def provide_answer_async() -> None:
        await asyncio.sleep(0.05)
        pending = HumanInputTool.get_pending_requests()
        assert len(pending) == 1
        request_id = list(pending.keys())[0]
        HumanInputTool.submit_human_response(request_id, "Operator approved.")

    answer_task = asyncio.create_task(provide_answer_async())

    res = await tool.aexecute(question="Proceed with database migration?", timeout_seconds=5.0)
    await answer_task

    assert res.exit_code == 0
    assert not res.is_error
    assert res.output == "Operator approved."

    events = event_collector.get_events_by_type(EventType.ASK_HUMAN.value)
    assert len(events) == 1
    assert events[0].data["question"] == "Proceed with database migration?"


def test_dynamic_mcp_tool_schema_conversion() -> None:
    """Verify DynamicMCPTool translates MCP tool schema into standard OpenAI function format."""
    dummy_client = MCPClient(server_url="http://localhost:8000")
    mcp_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search term"}
        },
        "required": ["query"],
    }

    tool = DynamicMCPTool(
        name="remote_mcp_search",
        description="Search tool provided by MCP server.",
        mcp_client=dummy_client,
        input_schema=mcp_schema,
    )

    openai_schema = tool.to_openai_schema()
    assert openai_schema["type"] == "function"
    assert openai_schema["function"]["name"] == "remote_mcp_search"
    assert openai_schema["function"]["description"] == "Search tool provided by MCP server."
    assert "query" in openai_schema["function"]["parameters"]["properties"]
    assert "query" in openai_schema["function"]["parameters"]["required"]