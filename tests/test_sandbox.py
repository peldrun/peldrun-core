"""
PELDRUN Core Sandbox Subsystem Test Suite.
Verifies BaseSandbox contracts, LocalProcessSandbox command execution,
timeout kills, directory boundary containment, DockerSandbox lifecycle & CLI delegation,
and sandbox factory routing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Optional, Tuple
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

    is_healthy = await sandbox.acheck_health()
    assert is_healthy is True

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


# =============================================================================
# DockerSandbox Unit & Lifecycle Tests
# =============================================================================

@pytest.mark.asyncio
async def test_docker_sandbox_initialization_daemon_unreachable(temp_workspace: Path) -> None:
    """Verify DockerSandbox raises RuntimeError when docker daemon is not running."""
    async def mock_runner(cmd: List[str], timeout: Optional[float] = None) -> Tuple[int, str, str]:
        if cmd == ["docker", "info"]:
            return 1, "", "Cannot connect to the Docker daemon at unix:///var/run/docker.sock"
        return 0, "", ""

    sandbox = DockerSandbox(
        config=DockerSandboxConfig(workspace_root=str(temp_workspace)),
        cmd_runner=mock_runner,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await sandbox.ainitialize()

    assert "Docker daemon is not running or unreachable" in str(exc_info.value)


@pytest.mark.asyncio
async def test_docker_sandbox_lifecycle_exec_and_health(temp_workspace: Path) -> None:
    """Verify full DockerSandbox lifecycle: bootstrap container, exec commands, check health, and cleanup."""
    executed_commands: List[List[str]] = []

    async def mock_runner(cmd: List[str], timeout: Optional[float] = None) -> Tuple[int, str, str]:
        executed_commands.append(cmd)
        if cmd == ["docker", "info"]:
            return 0, "Client: Docker Engine", ""
        elif cmd[0:3] == ["docker", "run", "-d"]:
            return 0, "mock_container_id_123456789", ""
        elif cmd[0:2] == ["docker", "exec"]:
            return 0, "Docker Execution Succeeded", ""
        elif cmd[0:2] == ["docker", "inspect"]:
            return 0, "true", ""
        elif cmd[0:2] == ["docker", "rm"]:
            return 0, "mock_container_id_123456789", ""
        return 0, "", ""

    sandbox = DockerSandbox(
        config=DockerSandboxConfig(
            workspace_root=str(temp_workspace),
            network_disabled=True,
            memory_limit="512m",
            cpu_quota=0.5,
        ),
        cmd_runner=mock_runner,
    )

    # 1. Initialize
    await sandbox.ainitialize()
    assert sandbox._initialized is True
    assert sandbox._container_id == "mock_container_id_123456789"

    # 2. Execute command
    res = await sandbox.aexecute(
        command="python -c 'print(1)'",
        workdir="subdir",
        env={"TEST_ENV": "active"},
    )
    assert res.exit_code == 0
    assert res.is_success is True
    assert res.stdout == "Docker Execution Succeeded"
    assert res.metadata["workdir"] == "/workspace/subdir"

    # 3. Check health
    healthy = await sandbox.acheck_health()
    assert healthy is True

    # 4. Cleanup
    await sandbox.acleanup()
    assert sandbox._initialized is False
    assert sandbox._container_id is None


@pytest.mark.asyncio
async def test_docker_sandbox_file_io(temp_workspace: Path) -> None:
    """Verify DockerSandbox file read and write operations via workspace bind-mount."""
    sandbox = DockerSandbox(config=DockerSandboxConfig(workspace_root=str(temp_workspace)))

    # Write file to mounted workspace
    written = await sandbox.awrite_file("artifacts/output.json", '{"status": "ok"}')
    assert written is True
    assert (temp_workspace / "artifacts" / "output.json").exists()

    # Read file back
    content = await sandbox.aread_file("artifacts/output.json")
    assert content == '{"status": "ok"}'

    # Missing file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        await sandbox.aread_file("artifacts/missing.json")


@pytest.mark.asyncio
async def test_docker_sandbox_exec_timeout(temp_workspace: Path) -> None:
    """Verify DockerSandbox flags timeout when command execution exceeds threshold."""
    async def mock_runner(cmd: List[str], timeout: Optional[float] = None) -> Tuple[int, str, str]:
        if cmd == ["docker", "info"]:
            return 0, "OK", ""
        elif cmd[0:3] == ["docker", "run", "-d"]:
            return 0, "container_timeout_id", ""
        elif cmd[0:2] == ["docker", "exec"]:
            # Emulate process timeout
            return 124, "", "Subprocess timed out."
        return 0, "", ""

    sandbox = DockerSandbox(
        config=DockerSandboxConfig(workspace_root=str(temp_workspace)),
        cmd_runner=mock_runner,
    )
    await sandbox.ainitialize()

    res = await sandbox.aexecute("sleep 100", timeout=1.0)
    assert res.exit_code == 124
    assert res.is_timeout is True
    assert not res.is_success