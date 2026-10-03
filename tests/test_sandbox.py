"""
PELDRUN Core Sandbox Subsystem Test Suite.
Verifies BaseSandbox contracts, LocalProcessSandbox command execution,
timeout kills, directory boundary containment, and sandbox factory routing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import pytest

from peldrun.sandbox import get_sandbox
from peldrun.sandbox.base import BaseSandbox, SandboxConfig, SandboxResult
from peldrun.sandbox.docker_sandbox import DockerSandbox, DockerSandboxConfig
from peldrun.sandbox.local_process import LocalProcessSandbox


def test_sandbox_result_contracts() -> None:
    """Verify success checks, output concatenation, and timeout properties on SandboxResult."""
    # Successful execution outcome
    success_res = SandboxResult(
        stdout="Operation completed.",
        stderr="",
        exit_code=0,
        duration_seconds=0.12,
        is_timeout=False,
    )
    assert success_res.is_success is True
    assert success_res.combined_output == "Operation completed."

    # Failed outcome with stderr
    err_res = SandboxResult(
        stdout="Partial progress",
        stderr="Error encountered",
        exit_code=1,
        is_timeout=False,
    )
    assert err_res.is_success is False
    assert "[STDERR]\nError encountered" in err_res.combined_output

    # Timeout outcome
    timeout_res = SandboxResult(
        stdout="",
        stderr="Execution timed out.",
        exit_code=124,
        is_timeout=True,
    )
    assert timeout_res.is_success is False
    assert timeout_res.is_timeout is True


@pytest.mark.asyncio
async def test_local_process_sandbox_execution(temp_workspace: Path) -> None:
    """Verify asynchronous command execution and output capture within workspace boundary."""
    sandbox = LocalProcessSandbox(config=SandboxConfig(workspace_root=str(temp_workspace)))
    await sandbox.ainitialize()

    result = await sandbox.aexecute('python -c "print(\'Sandbox Process OK\')"')
    assert result.is_success is True
    assert result.exit_code == 0
    assert "Sandbox Process OK" in result.stdout
    assert result.duration_seconds > 0.0


@pytest.mark.asyncio
async def test_local_process_sandbox_timeout_kill(temp_workspace: Path) -> None:
    """Verify process termination and exit code 124 upon reaching timeout threshold."""
    sandbox = LocalProcessSandbox(config=SandboxConfig(workspace_root=str(temp_workspace)))
    await sandbox.ainitialize()

    # Attempt a command that sleeps longer than timeout
    sleep_cmd = 'python -c "import time; time.sleep(3)"'
    result = await sandbox.aexecute(sleep_cmd, timeout=0.5)

    assert result.is_timeout is True
    assert result.exit_code == 124
    assert not result.is_success
    assert "timed out" in result.stderr.lower()


@pytest.mark.asyncio
async def test_local_process_sandbox_traversal_blocked(temp_workspace: Path) -> None:
    """Verify sandbox resolve_safe_path raises PermissionError for directory traversal attempts."""
    sandbox = LocalProcessSandbox(config=SandboxConfig(workspace_root=str(temp_workspace)))

    with pytest.raises(PermissionError) as exc_info:
        sandbox.resolve_safe_path("../../outside.txt")

    assert "escapes sandbox boundary" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_local_process_sandbox_file_io(temp_workspace: Path) -> None:
    """Verify sandboxed file write and read operations."""
    sandbox = LocalProcessSandbox(config=SandboxConfig(workspace_root=str(temp_workspace)))
    await sandbox.ainitialize()

    # Write file
    write_success = await sandbox.awrite_file("nested/test.txt", "Sandbox File Content")
    assert write_success is True

    # Read file
    content = await sandbox.aread_file("nested/test.txt")
    assert content == "Sandbox File Content"

    # Reading non-existent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        await sandbox.aread_file("non_existent.txt")


@pytest.mark.asyncio
async def test_local_process_sandbox_health_and_cleanup(temp_workspace: Path) -> None:
    """Verify healthcheck validation and cleanup execution."""
    sandbox = LocalProcessSandbox(config=SandboxConfig(workspace_root=str(temp_workspace)))
    await sandbox.ainitialize()

    # Health check
    is_healthy = await sandbox.acheck_health()
    assert is_healthy is True

    # Cleanup without errors
    await sandbox.acleanup()


def test_docker_sandbox_configuration_defaults() -> None:
    """Verify DockerSandboxConfig schema attributes and defaults."""
    cfg = DockerSandboxConfig(
        workspace_root="/tmp/sandbox_test",
        image="python:3.11-slim",
        memory_limit="1g",
        cpu_quota=0.5,
    )
    assert cfg.image == "python:3.11-slim"
    assert cfg.memory_limit == "1g"
    assert cfg.cpu_quota == 0.5
    assert cfg.network_disabled is False
    assert cfg.auto_remove is True


def test_sandbox_factory_routing(temp_workspace: Path) -> None:
    """Verify get_sandbox factory instantiates proper sandbox instances by identifier."""
    # Local process sandbox
    local_sb = get_sandbox("local", workspace_root=str(temp_workspace))
    assert isinstance(local_sb, LocalProcessSandbox)
    assert local_sb.workspace_root == str(temp_workspace)

    # Docker sandbox
    docker_sb = get_sandbox("docker", workspace_root=str(temp_workspace))
    assert isinstance(docker_sb, DockerSandbox)
    assert docker_sb.workspace_root == str(temp_workspace)

    # Unsupported identifier raises ValueError
    with pytest.raises(ValueError) as exc_info:
        get_sandbox("unsupported_runtime")

    assert "unsupported sandbox type" in str(exc_info.value).lower()