# Changelog

All notable changes to the `peldrun-core` runtime and engine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-30

### Added
- **Unified Public Event Contract (Envelope Pattern)**:
  - Standardized immutable envelope schema: `version`, `event_id`, `sequence`, `run_id`, `timestamp`, `step`, `type`, `payload`, `metadata`.
  - Monotonic sequence numbering within execution runs for reliable event reordering and SSE reconnect semantics (`resume from sequence N`).
  - Strict top-level field validation via Pydantic v2 `extra="forbid"`.
  - Contract validation test suite (`tests/test_event_contracts.py`).
- **Deterministic Security & Governance Subsystem (`peldrun.security`)**:
  - Declarative `SecurityPolicy` model managing filesystem containment, outbound network controls, and command execution governance.
  - Strict path traversal detection (`validate_workspace_path`) preventing directory escapes and protecting sensitive configuration files (`.env`, SSH keys).
  - Malicious command signature interception for destructive terminal operations (`rm -rf /`, formatting, filesystem disruption).
  - Risk classification tiers (`ActionRiskLevel`) with Human-in-the-Loop elevation triggers for critical destructive actions.
  - Security test suite verifying policy boundaries (`tests/test_security_policy.py`).
- **Comprehensive E2E Runtime Lifecycle Suite (`tests/test_runtime_e2e.py`)**:
  - End-to-end execution loop verification across models, tool invocations, security validation, and event streaming.
  - Checkpoint persistence and state recovery validation via `CheckpointManager.asave_checkpoint` and `CheckpointManager.arestore_state`.
  - Dual-path execution verification distinguishing direct token streaming from autonomous ReAct agent loops.

### Changed
- Standardized `peldrun.agents.base.BaseAgent` to comply natively with `StepExecutableAgent` protocol (`name`, `step()`).
- Added transparent backward-compatible aliases to `ExecutionState` (`tool_calls` for `tool_history`, `output` for `final_output`).
- Modularized packaging extras in `pyproject.toml` into `docker`, `dev`, and `all`.
- Integrated `SecurityPolicy` directly into `FileOpsTool` and `ShellExecTool` builtins.

### Fixed
- Backward-compatible wire format framing in `PeldrunEvent.to_sse()` and `to_sse_payload()` providing `id` and `data` aliases for web and SSE consumers.
- Interception boundary regex refinements for multi-platform destructive command detection.