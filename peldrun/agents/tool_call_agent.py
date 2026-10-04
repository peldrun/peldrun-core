"""
PELDRUN Tool Call Agent.
Inherits from ReActAgent, implements concrete think/act cycles,
and safeguards against malformed tool generation loops.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from peldrun.agents.base import AgentConfig
from peldrun.agents.react_agent import ReActAgent
from peldrun.events.emitter import EventEmitter
from peldrun.events.schema import EventType, PeldrunEvent
from peldrun.llm.client import AsyncLLMClient
from peldrun.tools.base import ToolResult
from peldrun.tools.collection import ToolCollection

logger = logging.getLogger(__name__)


class ToolCallAgent(ReActAgent):
    """Autonomous agent executing actions through structured LLM tool calling."""

    def __init__(
        self,
        config: AgentConfig,
        llm: Optional[AsyncLLMClient] = None,
        tool_collection: Optional[ToolCollection] = None,
        emitter: Optional[EventEmitter] = None,
        workspace_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            config=config,
            llm=llm,
            tool_collection=tool_collection,
            emitter=emitter,
            workspace_dir=workspace_dir,
        )
        self._consecutive_empty_calls: int = 0
        self._final_answer: str = ""

    def _format_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Return standardized OpenAI function schemas for available tools."""
        tools_list: List[Dict[str, Any]] = []
        for name, tool in self.tool_collection.tools.items():
            parameters = getattr(tool, "parameters", {})
            if not parameters:
                parameters = {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }
            tools_list.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": getattr(tool, "description", "") or f"Tool {name}",
                        "parameters": parameters,
                    },
                }
            )
        return tools_list

    async def think(self) -> Optional[Any]:
        """Request model completion with available tool specifications."""
        tool_schemas = self._format_tools_for_llm()
        try:
            response = await self.llm.chat_complete(
                messages=self.messages,
                tools=tool_schemas if tool_schemas else None,
                temperature=0.2,
            )
        except Exception as err:
            error_msg = f"LLM Inference Failure: {str(err)}"
            logger.error(error_msg)
            await self.emitter.emit(
                PeldrunEvent(
                    type=EventType.ERROR,
                    step=self.current_step,
                    payload={"error": error_msg},
                )
            )
            self._final_answer = error_msg
            return None

        content = getattr(response, "content", "") or ""
        reasoning = getattr(response, "reasoning_content", "") or ""
        raw_tool_calls = getattr(response, "tool_calls", None) or []

        thought_text = reasoning.strip() or content.strip()
        if thought_text:
            await self.emitter.emit(
                PeldrunEvent(
                    type=EventType.THOUGHT,
                    step=self.current_step,
                    payload={"thought": thought_text},
                )
            )

        if not raw_tool_calls:
            self._final_answer = content or thought_text or "Task completed."
            self.messages.append({"role": "assistant", "content": self._final_answer})
            return None

        return {"response": response, "tool_call": raw_tool_calls[0]}

    async def act(self, decision: Any) -> bool:
        """Safely execute the tool call and register observation into context."""
        response = decision.get("response")
        selected_call = decision.get("tool_call")

        call_func = getattr(selected_call, "function", selected_call)
        tool_name = getattr(call_func, "name", "")
        raw_arguments = getattr(call_func, "arguments", "{}")

        parsed_args: Dict[str, Any] = {}
        if isinstance(raw_arguments, dict):
            parsed_args = raw_arguments
        elif isinstance(raw_arguments, str) and raw_arguments.strip():
            try:
                parsed_args = json.loads(raw_arguments)
            except Exception:
                parsed_args = {}

        if not parsed_args and tool_name not in ["terminate", "ask_human"]:
            self._consecutive_empty_calls += 1
            if self._consecutive_empty_calls >= 3:
                recovery_msg = (
                    f"Error: Tool '{tool_name}' was invoked repeatedly with empty arguments. "
                    "You must provide required parameters or call 'terminate' with the final answer."
                )
                self.messages.append({"role": "system", "content": recovery_msg})
                return True
        else:
            self._consecutive_empty_calls = 0

        await self.emitter.emit(
            PeldrunEvent(
                type=EventType.TOOL_CALL,
                step=self.current_step,
                payload={"tool_name": tool_name, "arguments": parsed_args},
            )
        )

        if tool_name == "terminate":
            status = parsed_args.get("status", "success")
            message = parsed_args.get("message", "") or "Task concluded."
            self._final_answer = message
            await self.emitter.emit(
                PeldrunEvent(
                    type=EventType.OBSERVATION,
                    step=self.current_step,
                    payload={"output": f"Terminated with status: {status}. {message}"},
                )
            )
            return False

        tool_instance = self.tool_collection.get_tool(tool_name)
        if not tool_instance:
            obs_text = f"Tool Execution Error: Tool '{tool_name}' is not registered."
        else:
            try:
                result: ToolResult = await tool_instance.arun(**parsed_args)
                obs_text = result.output if hasattr(result, "output") else str(result)
            except Exception as ex:
                obs_text = f"Tool Execution Error ({tool_name}): {str(ex)}"

        await self.emitter.emit(
            PeldrunEvent(
                type=EventType.OBSERVATION,
                step=self.current_step,
                payload={"output": obs_text},
            )
        )

        call_id = getattr(selected_call, "id", f"call_{self.current_step}")
        self.messages.append(
            {
                "role": "assistant",
                "content": getattr(response, "content", "") or "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(parsed_args, ensure_ascii=False),
                        },
                    }
                ],
            }
        )
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_name,
                "content": obs_text,
            }
        )
        return True

    async def run_task(self, prompt: str, max_steps: Optional[int] = None) -> str:
        """Direct driver method executing the iterative loop."""
        steps_limit = max_steps or self.config.max_steps or 30
        self.config.max_steps = steps_limit
        self.current_step = 0
        self.messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        self._consecutive_empty_calls = 0
        self._final_answer = ""

        while self.current_step < steps_limit:
            should_continue = await self._astep()
            if not should_continue:
                break

        return self._final_answer