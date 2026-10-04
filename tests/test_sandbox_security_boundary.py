"""
Automated Security Boundary Verification Test for Sandbox Execution.
Verifies command signature interception and path traversal blocking.
"""

import os
import shutil
import tempfile
import pytest

from peldrun.sandbox.base import SandboxConfig
from peldrun.sandbox.local_process import LocalProcessSandbox
from peldrun.security.policy import SecurityPolicy, SecurityViolationError
from peldrun.tools.builtins.shell_exec import ShellExecTool


@pytest.mark.asyncio
async def test_sandbox_blocks_destructive_commands():
    temp_dir = tempfile.mkdtemp()
    try:
        policy = SecurityPolicy(workspace_root=temp_dir)
        sandbox = LocalProcessSandbox(
            config=SandboxConfig(workspace_root=temp_dir),
            security_policy=policy,
        )
        tool = ShellExecTool(workspace_root=temp_dir, security_policy=policy, sandbox=sandbox)

        # Destructive command should be intercepted and blocked with code 126
        result = await tool.execute(command="rm -rf /")
        assert result.exit_code == 126
        assert "Security violation" in result.error
        assert result.metadata.get("security_blocked") is True

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_sandbox_blocks_path_traversal():
    temp_dir = tempfile.mkdtemp()
    try:
        policy = SecurityPolicy(workspace_root=temp_dir)
        sandbox = LocalProcessSandbox(
            config=SandboxConfig(workspace_root=temp_dir),
            security_policy=policy,
        )

        with pytest.raises(PermissionError):
            sandbox.resolve_safe_path("../../windows/system32/cmd.exe")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)