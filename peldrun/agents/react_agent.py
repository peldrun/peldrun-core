"""
PELDRUN Core ReAct Autonomous Agent Implementation.
Implements the Reasoning + Acting loop with streaming thought extraction,
structured function calling, observation feedback, and graceful completion.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional
from pydantic import Field

from peldrun.agents.base import AgentConfig, BaseAgent
from peldrun.events.emitter import EventEmitter
from peldrun.llm.client import DeltaToolCall, StreamChunk
from peldrun.llm.providers.openai_compat import BaseLLMProvider
from peldrun.memory import MemoryManager
from peldrun.sandbox.base import BaseSandbox
from peldrun.tools.registry import ToolRegistry

logger = logging.getLogger("peldrun.agents.react_agent")


class ReActAgentConfig(AgentConfig):
    """Configuration options specific to the ReAct agent implementation."""
    name: str = Field(default="react_agent", description="ReAct agent identifier")
    description: str = Field(
        default="Autonomous ReAct agent performing iterative reasoning, tool calling, and observation synthesis.",
        description="Functional description"
    )
    enable_streaming: bool = Field(
        default=True,
        description="Whether to stream tokens and thoughts incrementally"
    )


class ReActAgent(BaseAgent):
    """
    Autonomous ReAct agent executing reasoning and action cycles.
    Interprets user intent, issues tool commands, incorporates observations,
    and returns verified deliverables.
    """

    def __init__(
        self,
        config: Optional[ReActAgentConfig] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        tool_registry: Optional[ToolRegistry] = None,
        memory_manager: Optional[MemoryManager] = None,
        sandbox: Optional[BaseSandbox] = None,
        emitter: Optional[EventEmitter] = None,
    ) -> None:
        super().__init__(
            config=config or ReActAgentConfig(),
            llm_provider=llm_provider,
            tool_registry=tool_registry,
            memory_manager=memory_manager,
            sandbox=sandbox,
            emitter=emitter,
        )

    def _extract_reasoning_and_content(self, text: str) -> tuple[Optional[str], str]:
        """
        Separate chain-of-thought traces wrapped in <think> tags from conversational output.
        """
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if think_match:
            thought = think_match.group(1).strip()
            clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return thought if thought else None, clean_text
        return None, text.strip()

    async def _astep(self) -> bool:
        """
        Execute one complete ReAct cognition-action cycle:
        1. Fit conversation history inside token budget.
        2. Query LLM provider with registered tool definitions.
        3. Parse thought traces, final content, and tool calls.
        4. Execute tool calls and feed observations back into state.
        5. Return True if additional steps are required, False if task reached conclusion.
        """
        if not self.llm:
            raise RuntimeError("LLM provider must be configured to execute ReActAgent steps.")

        # 1. Prepare message context and schemas
        budgeted_messages = self.memory.short_term.get_budgeted_messages()
        tool_schemas = self.tools.get_openai_schemas()

        content_buffer = ""
        thought_buffer = ""
        tool_calls_accumulator: Dict[int, Dict[str, Any]] = {}

        # 2. Dispatch LLM generation/streaming
        logger.debug("ReAct step %d: invoking LLM provider with %d tools", self.state.step, len(tool_schemas))

        if getattr(self.config, "enable_streaming", True):
            async for chunk in self.llm.stream(
                messages=budgeted_messages,
                tools=tool_schemas if tool_schemas else None,
            ):
                if chunk.content:
                    content_buffer += chunk.content

                # Accumulate partial tool calls
                if chunk.tool_calls:
                    for dtc in chunk.tool_calls:
                        idx = dtc.index
                        if idx not in tool_calls_accumulator:
                            tool_calls_accumulator[idx] = {
                                "id": dtc.id or f"call_{idx}",
                                "name": dtc.name or "",
                                "arguments": dtc.arguments or "",
                            }
                        else:
                            if dtc.id:
                                tool_calls_accumulator[idx]["id"] = dtc.id
                            if dtc.name:
                                tool_calls_accumulator[idx]["name"] += dtc.name
                            if dtc.arguments:
                                tool_calls_accumulator[idx]["arguments"] += dtc.arguments
        else:
            response = await self.llm.generate(
                messages=budgeted_messages,
                tools=tool_schemas if tool_schemas else None,
            )
            content_buffer = response.content or ""
            if response.tool_calls:
                for idx, tc in enumerate(response.tool_calls):
                    func = tc.get("function", {})
                    tool_calls_accumulator[idx] = {
                        "id": tc.get("id", f"call_{idx}"),
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", "{}"),
                    }

        # 3. Extract reasoning/thought if present
        thought, clean_content = self._extract_reasoning_and_content(content_buffer)
        if thought:
            await self.emitter.emit_thought(thought=thought)
            self.state.add_message("assistant", f"<think>{thought}</think>")

        # 4. Process tool invocations if generated
        if tool_calls_accumulator:
            tool_calls_payload: List[Dict[str, Any]] = []

            for idx in sorted(tool_calls_accumulator.keys()):
                item = tool_calls_accumulator[idx]
                t_id = item["id"]
                t_name = item["name"]
                t_args_raw = item["arguments"]

                # Parse JSON parameters
                try:
                    t_args = json.loads(t_args_raw) if t_args_raw.strip() else {}
                except json.JSONDecodeError:
                    t_args = {"raw_input": t_args_raw}

                tool_calls_payload.append({
                    "id": t_id,
                    "type": "function",
                    "function": {"name": t_name, "arguments": t_args_raw},
                })

                # Broadcast tool call event
                await self.emitter.emit_tool_call(
                    tool_name=t_name,
                    arguments=t_args,
                    tool_call_id=t_id,
                )

                # Execute tool safely
                result = await self.tools.aexecute(t_name, **t_args)

                # Record observation
                self.state.record_tool_execution(
                    tool_name=t_name,
                    arguments=t_args,
                    output=result.output,
                    exit_code=result.exit_code,
                    is_error=result.is_error,
                    artifacts=result.artifacts,
                    tool_call_id=t_id,
                )

                # Broadcast observation event
                await self.emitter.emit_observation(
                    output=result.output,
                    tool_name=t_name,
                    exit_code=result.exit_code,
                    is_error=result.is_error,
                    artifacts=result.artifacts,
                    tool_call_id=t_id,
                )

                # Synchronize observation with short-term memory
                obs_content = result.output if isinstance(result.output, str) else json.dumps(result.output, ensure_ascii=False)
                self.memory.short_term.add_message(
                    role="tool",
                    content=obs_content,
                    name=t_name,
                    tool_call_id=t_id,
                )

            # Record assistant turn with tool calls into short-term memory
            self.memory.short_term.add_message(
                role="assistant",
                content=clean_content if clean_content else None,
                tool_calls=tool_calls_payload,
            )

            # Continue ReAct loop
            return True

        # 5. No tool calls generated -> Model concluded with a final response
        if clean_content:
            self.state.add_message("assistant", clean_content)
            self.memory.short_term.add_message("assistant", clean_content)
            self.state.mark_completed(output=clean_content)
            return False

        # Edge case: No content and no tool calls -> Terminate to prevent infinite loops
        logger.warning("ReAct step %d received empty response from LLM.", self.state.step)
        self.state.mark_completed(output="No further actions required.")
        return False