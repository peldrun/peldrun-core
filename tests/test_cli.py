"""
PELDRUN Core Standalone CLI Test Suite.
Verifies command line parsing, subcommands ('version', 'tools', 'check', 'run'),
and process exit codes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pytest

from peldrun.cli import main
from peldrun.engine.state import ExecutionState


def test_cli_no_args_displays_help(capsys: pytest.CaptureFixture) -> None:
    """Verify executing peldrun without arguments prints help and returns exit code 0."""
    exit_code = main([])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "usage:" in captured.out.lower()
    assert "peldrun" in captured.out.lower()


def test_cli_version_subcommand(capsys: pytest.CaptureFixture) -> None:
    """Verify 'peldrun version' outputs package version."""
    exit_code = main(["version"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "peldrun-core v0.1.0" in captured.out


def test_cli_tools_subcommand(capsys: pytest.CaptureFixture, temp_workspace: Path) -> None:
    """Verify 'peldrun tools' lists registered builtin tools."""
    exit_code = main(["tools", "--workspace", str(temp_workspace)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "file_ops" in captured.out
    assert "shell_exec" in captured.out
    assert "web_search" in captured.out
    assert "ask_human" in captured.out


def test_cli_check_connectivity_healthy(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """Verify 'peldrun check' returns 0 when health check succeeds."""
    async def mock_health(self: Any) -> bool:
        return True

    from peldrun.llm.providers.openai_compat import OpenAICompatProvider
    monkeypatch.setattr(OpenAICompatProvider, "check_health", mock_health)

    exit_code = main(["check", "--provider", "openai_compat"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "[OK]" in captured.out


def test_cli_check_connectivity_unhealthy(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """Verify 'peldrun check' returns 1 when health check fails."""
    async def mock_health(self: Any) -> bool:
        return False

    from peldrun.llm.providers.openai_compat import OpenAICompatProvider
    monkeypatch.setattr(OpenAICompatProvider, "check_health", mock_health)

    exit_code = main(["check", "--provider", "openai_compat"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "[FAIL]" in captured.out


def test_cli_run_command_success(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, temp_workspace: Path) -> None:
    """Verify 'peldrun run' executes agent task and outputs final summary."""
    async def mock_arun(self: Any, task: str) -> ExecutionState:
        self.state.mark_completed(output="CLI Task Succeeded.")
        self.state.deliverables.append("result.txt")
        return self.state

    from peldrun.agents.react_agent import ReActAgent
    monkeypatch.setattr(ReActAgent, "arun", mock_arun)

    exit_code = main([
        "run",
        "Generate automated tests",
        "--workspace", str(temp_workspace),
        "--agent", "react",
        "--verbose",
    ])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "CLI Task Succeeded." in captured.out
    assert "COMPLETED" in captured.out
    assert "result.txt" in captured.out


def test_cli_run_command_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, temp_workspace: Path) -> None:
    """Verify 'peldrun run' handles exceptions and returns exit code 1."""
    async def mock_arun_fail(self: Any, task: str) -> ExecutionState:
        raise RuntimeError("Simulated agent fatal failure")

    from peldrun.agents.react_agent import ReActAgent
    monkeypatch.setattr(ReActAgent, "arun", mock_arun_fail)

    exit_code = main([
        "run",
        "Failing task",
        "--workspace", str(temp_workspace),
    ])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Simulated agent fatal failure" in captured.out