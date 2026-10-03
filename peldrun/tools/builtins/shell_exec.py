"""
PELDRUN Core Scoped Shell Execution Tool.
Executes terminal commands asynchronously with strict working-directory confinement,
timeout controls, and structured output capture.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, Field

from peldrun.tools.base import BaseTool, ToolResult

logger = logging.getLogger("peldrun.tools.builtins.shell_exec")

MAX_OUTPUT_CHARS = 30_000


class ShellExecArgs(BaseModel):
    """Input arguments schema for scoped shell command execution."""
    command: str = Field(
        ...,
        description="The shell command string to execute inside the workspace."
    )
    working_subdir: Optional[str] = Field(
        default=None,
        description="Optional relative subdirectory within the workspace root where execution occurs."
    )
    timeout_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=600.0,
        description="Timeout ceiling in seconds before terminating the subprocess."
    )


class ShellExecTool(BaseTool):
    """
    Core tool for running shell commands confined to the project workspace.
    Captures stdout, stderr, process return codes, and enforces strict execution boundaries.
    """

    name: str = "shell_exec"
    description: str = (
        "Execute shell commands securely within the bounded project workspace directory. "
        "Captures stdout, stderr, and process exit codes with timeout protection."
    )
    args_schema: Optional[Type[BaseModel]] = ShellExecArgs

    def _resolve_working_directory(self, subdir: Optional[str]) -> Path:
        """
        Validate and resolve target execution directory against workspace root.
        """
        if not self.workspace_root:
            raise PermissionError("Workspace root is not configured. Shell execution is blocked.")

        root = Path(self.workspace_root).resolve()
        if not root.exists():
            root.mkdir(parents=True, exist_ok=True)

        if not subdir:
            return root

        target = (root / subdir.strip().lstrip("/\\")).resolve()
        if root != target and root not in target.parents:
            raise PermissionError(
                f"Access denied: Working subdirectory '{subdir}' escapes workspace boundary '{root}'."
            )

        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)

        return target

    async def _arun(
        self,
        command: str,
        working_subdir: Optional[str] = None,
        timeout_seconds: float = 60.0,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute command asynchronously with timeout and output capture."""
        try:
            cwd_path = self._resolve_working_directory(working_subdir)
        except PermissionError as perm_err:
            return ToolResult(
                output=str(perm_err),
                exit_code=1,
                is_error=True,
                metadata={"error_type": "PermissionError"},
            )

        logger.debug("Executing shell command: '%s' in cwd: '%s'", command, cwd_path)

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(cwd_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                try:
                    process.kill()
                    await process.wait()
                except Exception as kill_err:
                    logger.debug("Error while terminating timed-out process: %s", kill_err)

                return ToolResult(
                    output=f"Command execution timed out after {timeout_seconds} seconds.",
                    exit_code=124,
                    is_error=True,
                    metadata={"error_type": "TimeoutError", "timeout_seconds": timeout_seconds},
                )

            stdout_text = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
            exit_code = process.returncode if process.returncode is not None else 0
            is_error = exit_code != 0

            # Format combined output
            output_parts = []
            if stdout_text:
                output_parts.append(stdout_text)
            if stderr_text:
                output_parts.append(f"[STDERR]\n{stderr_text}")

            combined_output = "\n".join(output_parts) if output_parts else "(Command completed with no output)"

            # Truncate output if it exceeds context safety limit
            if len(combined_output) > MAX_OUTPUT_CHARS:
                truncated_chars = len(combined_output) - MAX_OUTPUT_CHARS
                combined_output = (
                    combined_output[:MAX_OUTPUT_CHARS]
                    + f"\n... [Output truncated, {truncated_chars} characters omitted] ..."
                )

            return ToolResult(
                output=combined_output,
                exit_code=exit_code,
                is_error=is_error,
                metadata={
                    "cwd": str(cwd_path),
                    "exit_code": exit_code,
                    "stdout_len": len(stdout_text),
                    "stderr_len": len(stderr_text),
                },
            )

        except Exception as ex:
            logger.exception("Failed to execute shell command '%s': %s", command, ex)
            return ToolResult(
                output=f"Subprocess execution error: {str(ex)}",
                exit_code=1,
                is_error=True,
                metadata={"error_type": type(ex).__name__},
            )