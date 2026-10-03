"""
PELDRUN Core Security Policy Test Suite.
Verifies path traversal jailing, blocked commands, risk classification,
and human-in-the-loop requirement triggers.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from peldrun.security.policy import (
    ActionRiskLevel,
    SecurityPolicy,
    SecurityViolationError,
    validate_workspace_path,
)


def test_workspace_path_jailing_valid() -> None:
    """Verify that paths inside workspace root resolve safely."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()
        sub_file = root / "folder" / "test.txt"
        sub_file.parent.mkdir(parents=True, exist_ok=True)
        sub_file.touch()

        resolved = validate_workspace_path(sub_file, root)
        assert resolved == sub_file


def test_path_traversal_blocked() -> None:
    """Verify that relative escaping (../) is rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()
        escaping_path = root / ".." / "outside.txt"

        with pytest.raises(SecurityViolationError) as excinfo:
            validate_workspace_path(escaping_path, root)
        assert "Path traversal detected" in str(excinfo.value)


def test_sensitive_files_blocked() -> None:
    """Verify access to sensitive files (e.g. .env) is blocked even within workspace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()
        env_file = root / ".env"
        env_file.touch()

        with pytest.raises(SecurityViolationError) as excinfo:
            validate_workspace_path(env_file, root)
        assert "Access denied to sensitive file pattern" in str(excinfo.value)


def test_blocked_destructive_commands() -> None:
    """Verify that dangerous destructive terminal commands are intercepted."""
    policy = SecurityPolicy()

    # Must block rm -rf /
    with pytest.raises(SecurityViolationError) as excinfo:
        policy.check_command("rm -rf /")
    assert "destructive security signature" in str(excinfo.value)

    # Must block format C:
    with pytest.raises(SecurityViolationError):
        policy.check_command("format C: /y")

    # Safe command should pass
    policy.check_command("echo 'Hello PELDRUN'")
    policy.check_command("python --version")


def test_risk_classification_and_approval() -> None:
    """Verify risk classification and HITL approval logic."""
    policy = SecurityPolicy(require_human_approval_for_destructive=True)

    # Shell read
    risk_read = policy.classify_action("file_ops", {"action": "read"})
    assert risk_read == ActionRiskLevel.READ_ONLY
    assert not policy.is_approval_required(risk_read)

    # Workspace write
    risk_write = policy.classify_action("file_saver", {"path": "main.py"})
    assert risk_write == ActionRiskLevel.WORKSPACE_WRITE
    assert not policy.is_approval_required(risk_write)

    # Destructive shell
    risk_destr = policy.classify_action("shell_exec", {"command": "rm -f output.log"})
    assert risk_destr == ActionRiskLevel.CRITICAL_DESTRUCTIVE
    assert policy.is_approval_required(risk_destr)