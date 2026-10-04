import asyncio
import os
import shutil
import tempfile
import uuid
from typing import Any, Dict

import pytest

from peldrun.agents.base import AgentConfig
from peldrun.agents.tool_call_agent import ToolCallAgent
from peldrun.events.emitter import EventEmitter
from peldrun.events.schema import EventType, PeldrunEvent
from peldrun.llm.client import LLMClient
from peldrun.llm.providers.openai_compat import OpenAICompatProvider
from peldrun.tools.builtins.str_replace_editor import StrReplaceEditor
from peldrun.tools.builtins.terminate import TerminateTool
from peldrun.tools.collection import ToolCollection


class MockLLMProvider:
    """Mock LLM provider simulating an autonomous agent creating a file and terminating."""

    def __init__(self):
        self.step = 0

    async def chat_completion(
        self,
        messages: list[Dict[str, Any]],
        tools: list[Dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self.step += 1

        if self.step == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Creating the verification file.",
                            "tool_calls": [
                                {
                                    "id": "call_create_file",
                                    "type": "function",
                                    "function": {
                                        "name": "str_replace_editor",
                                        "arguments": '{"command": "create", "path": "hello.txt", "file_text": "Hello PELDRUN"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

        elif self.step == 2:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "File verified. Terminating execution.",
                            "tool_calls": [
                                {
                                    "id": "call_terminate",
                                    "type": "function",
                                    "function": {
                                        "name": "terminate",
                                        "arguments": '{"status": "success", "message": "Task completed successfully."}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Nothing more to do.",
                    }
                }
            ]
        }


@pytest.mark.asyncio
async def test_core_standalone_execution_loop():
    temp_dir = tempfile.mkdtemp()
    try:
        run_id = uuid.uuid4()
        emitter = EventEmitter(run_id=run_id)
        captured_events: list[PeldrunEvent] = []

        async def capture_event(ev: PeldrunEvent):
            captured_events.append(ev)

        emitter.subscribe_all(capture_event)

        collection = ToolCollection()
        collection.add_tool(TerminateTool())
        collection.add_tool(StrReplaceEditor(workspace_root=temp_dir))

        mock_provider = MockLLMProvider()
        agent_config = AgentConfig(
            name="verification_agent",
            system_prompt="You are an autonomous agent.",
            max_steps=5,
        )

        agent = ToolCallAgent(
            config=agent_config,
            llm=mock_provider,  # type: ignore
            tool_collection=collection,
            emitter=emitter,
            workspace_dir=temp_dir,
        )

        result = await agent.run_task(
            prompt="Create hello.txt containing 'Hello PELDRUN'",
            max_steps=5,
        )

        target_file = os.path.join(temp_dir, "hello.txt")
        assert os.path.isfile(target_file), "hello.txt was not created on disk."

        with open(target_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert content == "Hello PELDRUN", f"Unexpected file content: {content}"
        assert any(
            ev.type == EventType.TOOL_CALLED
            and ev.payload.get("tool") == "str_replace_editor"
            for ev in captured_events
        ), "TOOL_CALLED event for str_replace_editor was not emitted."

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)