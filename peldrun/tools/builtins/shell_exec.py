"""
PELDRUN Core Shell Execution Tool.
Executes shell commands strictly through an isolated BaseSandbox instance with SecurityPolicy governance.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from peldrun.sandbox.base import BaseSandbox, SandboxConfig
from peldrun.sandbox.local_process import LocalProcessSandbox
from peldrun.security.policy import SecurityPolicy
from peldrun.tools.base import BaseTool, ToolResult


class ShellExecArgs(BaseModel):
    """Pydantic schema defining arguments for ShellExecTool."""
    command: str = Field(..., description="The shell command to execute inside the sandbox.")
    timeout: Optional[float] = Field(default=60.0, description="Command execution timeout in seconds.")


# Backward-compatibility alias
ShellExecParameters = ShellExecArgs


class ShellExecTool(BaseTool):
    """Executes shell commands strictly through an isolated sandbox environment."""

    name: str = "shell_exec"
    description: str = "Executes command-line instructions safely bounded within the project workspace."
    parameters: Dict[str, Any] = ShellExecArgs.model_json_schema()

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        security_policy: Optional[SecurityPolicy] = None,
        sandbox: Optional[BaseSandbox] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.workspace_root = workspace_root or os.getcwd()
        self.security_policy = security_policy or SecurityPolicy(workspace_root=self.workspace_root)
        self.sandbox = sandbox or LocalProcessSandbox(
            config=SandboxConfig(workspace_root=self.workspace_root),
            security_policy=self.security_policy,
        )

    async def execute(self, command: str, timeout: Optional[float] = 60.0, **kwargs: Any) -> ToolResult:
        try:
            res = await self.sandbox.aexecute(command=command, timeout=timeout)
            return ToolResult(
                output=res.stdout,
                error=res.stderr,
                exit_code=res.exit_code,
                metadata={
                    "is_timeout": res.is_timeout,
                    "duration_seconds": res.duration_seconds,
                    **(res.metadata or {}),
                },
            )
        except Exception as exc:
            return ToolResult(
                output="",
                error=f"Shell execution failure: {str(exc)}",
                exit_code=1,
            )


__all__ = [
    "ShellExecArgs",
    "ShellExecParameters",
    "ShellExecTool",
]