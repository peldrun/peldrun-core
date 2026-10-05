"""
PELDRUN Core Human Input & Interactive Dialogue Tool (ask_human).
Provides suspension for human approval, feedback loops, and multi-choice questionnaires.
Compatible with CLI, Web API, and Event bus registries.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator

from peldrun.tools.base import BaseTool, ToolResult


class HumanInputRegistry:
    """Centralized pending requests manager for human-in-the-loop approvals."""
    _pending_requests: Dict[str, asyncio.Future[str]] = {}
    _request_payloads: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register_request(cls, request_id: str, payload: Dict[str, Any]) -> asyncio.Future[str]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        cls._pending_requests[request_id] = future
        cls._request_payloads[request_id] = payload
        return future

    @classmethod
    def get_pending_request(cls, request_id: str) -> Optional[Dict[str, Any]]:
        return cls._request_payloads.get(request_id)

    @classmethod
    def resolve_request(cls, request_id: str, response: str) -> bool:
        future = cls._pending_requests.pop(request_id, None)
        cls._request_payloads.pop(request_id, None)
        if future and not future.done():
            future.set_result(response)
            return True
        return False

    @classmethod
    def cancel_request(cls, request_id: str) -> None:
        future = cls._pending_requests.pop(request_id, None)
        cls._request_payloads.pop(request_id, None)
        if future and not future.done():
            future.cancel()


class HumanInputArgs(BaseModel):
    """
    Pydantic schema defining arguments for HumanInputTool / ask_human.
    Supports flexible aliases (prompt/query/question) and rich question types.
    """
    prompt: str = Field(
        ...,
        description="The question or clarification needed from the human operator."
    )
    input_type: str = Field(
        default="text",
        description="Interaction modality: 'text' (input field), 'confirm' (Yes/No), 'select' (single choice), or 'multiple_choice'."
    )
    options: List[str] = Field(
        default_factory=list,
        description="List of choices/buttons for the operator to select from (e.g. ['Yes', 'No'] or custom options)."
    )
    timeout_seconds: Optional[int] = Field(
        default=600,
        description="Maximum wait time in seconds for operator response (default: 600s)."
    )

    @model_validator(mode="before")
    @classmethod
    def reconcile_prompt_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Resolve interchangeable keys: prompt, query, question
            if "prompt" not in data or not data["prompt"]:
                for alias in ("query", "question", "text", "message"):
                    if alias in data and data[alias]:
                        data["prompt"] = str(data[alias])
                        break
            
            # Default options for confirm type
            itype = str(data.get("input_type", "text")).lower()
            if itype in ("confirm", "boolean") and not data.get("options"):
                data["options"] = ["Yes", "No"]
        return data


# Backward-compatibility alias
HumanInputParameters = HumanInputArgs


class HumanInputTool(BaseTool):
    """
    Tool that suspends agent execution until a response is received from the human user.
    Supports interactive questions, input forms, and single/multiple button choices.
    """

    name: str = "ask_human"
    description: str = (
        "Pauses agent autonomy and asks the human operator for feedback, confirmation, or clarification. "
        "Supports text input, confirmation (Yes/No), and selectable choices. The execution waits until user replies."
    )
    args_schema = HumanInputArgs

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        emitter: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(workspace_root=workspace_root)
        self.workspace_root = workspace_root
        self.emitter = emitter

    async def _arun(
        self,
        prompt: str,
        input_type: str = "text",
        options: Optional[List[str]] = None,
        timeout_seconds: Optional[int] = 600,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Asynchronously suspend execution, emit request event, and wait for human resolution.
        """
        request_id = str(uuid.uuid4())
        timeout_val = float(timeout_seconds) if timeout_seconds else 600.0
        opts = options or []

        payload = {
            "request_id": request_id,
            "prompt": prompt,
            "input_type": input_type,
            "options": opts,
            "timeout_seconds": timeout_val,
        }

        # 1. Register future in central registry
        future = HumanInputRegistry.register_request(request_id, payload)

        # 2. Emit human input event to SSE/WebSocket stream if emitter available
        if self.emitter is not None:
            try:
                if hasattr(self.emitter, "emit_event"):
                    await self.emitter.emit_event(
                        event_type="human_input_required",
                        data=payload
                    )
            except Exception as emit_err:
                pass

        # 3. Interactive CLI Fallback if running directly in a terminal
        if sys.stdin and sys.stdin.isatty():
            try:
                print(f"\n[HUMAN INPUT REQUIRED] {prompt}")
                if opts:
                    for idx, opt in enumerate(opts, 1):
                        print(f"  {idx}. {opt}")
                    print("Enter your choice or text: ", end="", flush=True)
                else:
                    print("Your response: ", end="", flush=True)

                cli_reply = await asyncio.to_thread(sys.stdin.readline)
                clean_reply = cli_reply.strip()
                HumanInputRegistry.resolve_request(request_id, clean_reply)
            except Exception:
                pass

        # 4. Await user resolution from Web UI or CLI
        try:
            response = await asyncio.wait_for(future, timeout=timeout_val)
            return ToolResult(
                output=f"Human Response: {response}",
                exit_code=0,
                is_error=False,
                metadata={
                    "request_id": request_id,
                    "approved": True,
                    "input_type": input_type,
                    "response": response
                },
            )
        except asyncio.TimeoutError:
            HumanInputRegistry.cancel_request(request_id)
            return ToolResult(
                output=f"Human response timed out after {timeout_val} seconds. Proceeding with default action.",
                exit_code=1,
                is_error=True,
                metadata={"request_id": request_id, "timeout": True},
            )
        except asyncio.CancelledError:
            HumanInputRegistry.cancel_request(request_id)
            raise

    def _run(
        self,
        prompt: str,
        input_type: str = "text",
        options: Optional[List[str]] = None,
        timeout_seconds: Optional[int] = 600,
        **kwargs: Any
    ) -> ToolResult:
        """
        Synchronous fallback bridging to _arun cleanly without NotImplementedError.
        """
        return super()._run(
            prompt=prompt,
            input_type=input_type,
            options=options,
            timeout_seconds=timeout_seconds,
            **kwargs
        )


__all__ = [
    "HumanInputArgs",
    "HumanInputParameters",
    "HumanInputRegistry",
    "HumanInputTool",
]