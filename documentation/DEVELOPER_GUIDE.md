# PELDRUN Core Runtime & Universal Bridge

## Developer & Technical Architecture Reference Manual

---

## 1. System Overview & Architectural Topology

PELDRUN is an autonomous agent ecosystem designed for sovereign, cross-platform execution. The architecture strictly separates core reasoning and execution logic from presentation and management interfaces:

1. **`peldrun-core` (Python Package / Runtime Engine):**
* Headless, standalone engine implementing agent loops, state machine graphs, sandboxed tool execution, event emitting, and memory management.
* Independent of any specific web framework or graphical interface.
* Distributed as a standard Python package (`peldrun`).


2. **`peldrun` (Platform & Orchestration Layer):**
* **Backend (`omweb`):** FastAPI-based gateway, server-sent events (SSE) broadcaster, project/chat manager, and dual-engine adapter bridge.
* **Frontend:** Next.js (TypeScript/React) web workspace with live thinking visualizers, real-time code/preview execution panels, artifact managers, and capability stores.



```
+-----------------------------------------------------------------------------------+
|                           Frontend (Next.js / React)                              |
|   [ChatThread] <---> [ChatTimeline] <---> [WorkspacePanel (Preview/Files/Editor)] |
+----------------------------------------^------------------------------------------+
                                         |
                                         | SSE Streams & REST APIs
                                         v
+-----------------------------------------------------------------------------------+
|                        Platform Backend (omweb / FastAPI)                         |
|   [routers/run.py] <---> [job_manager] <---> [project_manager]                    |
|                               |                                                   |
|                    [adapters/core_adapter.py]                                     |
|                               |                                                   |
|                     [agent_bridge.py (Router)]                                    |
|                      /                      \                                     |
|       (Dual-Engine) /                        \                                    |
|                    v                          v                                   |
|   [PELDRUN Core Runtime]            [OpenManus Legacy Runtime]                    |
|   - ToolCallAgent                    - Legacy Agent Base                          |
|   - EventEmitter & Schema            - Memory Instrumentation                     |
|   - BaseTool & Sandboxes             - Standard Toolset                           |
+-----------------------------------------------------------------------------------+

```

---

## 2. Core Agent Loop & Lifecycle Architecture

### 2.1 ToolCallAgent (`peldrun.agents.tool_call_agent`)

The primary driver for task execution in `peldrun-core` is `ToolCallAgent`. Its loop executes as an asynchronous state machine:

```
[Task Initialized]
       |
       v
+--> [Step Start] -------------------> Emits EventType.STEP_START
|      |
|      v
|    [Think / LLM Generation] -------> Emits EventType.THOUGHT
|      |
|      v
|    [Tool Call Resolution] ---------> Emits EventType.TOOL_CALL
|      |
|      v
|    [Tool Execution (Sandbox/IO)] --> Emits EventType.OBSERVATION
|      |
|      +---> Artifact Detected? -----> Emits EventType.OBSERVATION (artifact_created)
|      |
|      v
|    [Termination Check]
|      |-- Tool is 'terminate' ------> Finalize -> Complete Job
|      +-- Max Steps Exceeded -------> Abort / Complete with partial output
|      +-- Next Iteration -----------> Loops back to Step Start
+------+

```

### 2.2 Execution Invariants

* **Step Incrementing:** Steps strictly start at `1` and monotonically increase. Sub-actions (thoughts, calls, observations) are child records of the active step.
* **Isolated Directory Root (`workspace_root`):** All path operations, file inspections, and script executions are resolved relative to the active conversation directory (`storage/chats/{chat_id}/files`). Path traversal outside this root is rejected.
* **Deliverable Completion:** An execution turn is only successful when required deliverables are persisted to disk and verified via observation before calling `terminate`.

---

## 3. Event Bus & Real-Time Streaming Architecture

### 3.1 Event System Contracts (`peldrun.events.schema`)

The engine is instrumented with typed lifecycle contracts:

| Event Type (`EventType`) | Trigger | Payload Structure |
| --- | --- | --- |
| `STEP_START` | Beginning of execution cycle | `{"step": int, "model": str, "status": "running"}` |
| `THOUGHT` | Internal reasoning token output | `{"thought": str, "content": str, "model": str}` |
| `TOOL_CALL` | Agent produces function invocation | `{"name": str, "arguments": str/dict, "content": str}` |
| `OBSERVATION` | Execution outcome or error returned | `{"output": str, "content": str, "model": str}` |
| `ERROR` | Engine or tool validation failure | `{"message": str, "content": str, "model": str}` |
| `FINAL` | Task finished | `{"result": str, "produced_files": list}` |

### 3.2 Synchronous-to-Asynchronous Event Bridge

`EventEmitter` in `peldrun-core` dispatches callbacks synchronously. To prevent coroutine drops when pushing events to asynchronous SSE handlers:

```python
main_loop = asyncio.get_running_loop()

def _sync_event_handler(event: PeldrunEvent) -> None:
    """Synchronous wrapper ensuring async events execute correctly on the main loop."""
    try:
        main_loop.create_task(_async_on_core_event(event))
    except Exception as e:
        print(f"[BRIDGE ERROR] Failed to dispatch core event async: {e}")

emitter.subscribe_all(_sync_event_handler)

```

### 3.3 Payload Normalization for UI Components

The web interface (`ChatTimeline`, `ChatThread`, `StepTimeline`) requires flat attributes in event objects. The bridge normalizes payloads at the root level before JSON serialization:

```json
{
  "id": "evt_9b8a7c12",
  "type": "thought",
  "step": 1,
  "content": "Analyzing user requirements and checking workspace files.",
  "toolName": null,
  "data": {
    "thought": "Analyzing user requirements and checking workspace files.",
    "content": "Analyzing user requirements and checking workspace files.",
    "model": "nvidia/nemotron-3-nano-4b"
  }
}

```

### 3.4 Elimination of State Overwrite & Step Flickering

When the frontend streams events via SSE (`/api/run/jobs/{job_id}/stream`) and simultaneously polls `/api/run/jobs/{job_id}`:

* **Root Cause of Flickering:** Polling endpoints reading unfinished sessions from disk return `events: []`, overwriting client-side state.
* **Resolution:** An in-memory cache `ACTIVE_JOB_EVENTS` in `omweb/routers/run.py` maintains live event records during execution. Polling requests serve live in-memory events rather than empty session snapshots.

---

## 4. Tooling Architecture & Cross-Platform Execution

### 4.1 BaseTool Contract (`peldrun.tools.base`)

Every tool inherited from `BaseTool` must satisfy both asynchronous and synchronous entry points:

```python
class BaseTool(ABC):
    name: str
    description: str
    parameters: Dict[str, Any]

    @abstractmethod
    async def _arun(self, **kwargs: Any) -> ToolResult:
        """Asynchronous execution logic."""
        pass

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Entry point validating schemas and dispatching to _arun."""
        validated_kwargs = self._validate_arguments(**kwargs)
        return await self._arun(**validated_kwargs)

```

### 4.2 Canonical Tool Schemas

Local LLMs (e.g., `nemotron-3-nano-4b`, small quantized models) hallucinate infinite argument loops when tools are passed with empty parameter objects (`properties: {}`). Explicit parameter schemas are strictly enforced in `omweb/adapters/core_adapter.py`:

```python
CANONICAL_TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "bash": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command-line instruction to execute in the workspace."}
        },
        "required": ["command"]
    },
    "python_execute": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "The Python source code to execute inside the workspace."}
        },
        "required": ["code"]
    },
    "str_replace_editor": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "enum": ["view", "create", "str_replace", "insert", "undo_edit"]},
            "path": {"type": "string"},
            "file_text": {"type": "string"},
            "old_str": {"type": "string"},
            "new_str": {"type": "string"}
        },
        "required": ["command", "path"]
    },
    "terminate": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "default": "success"},
            "message": {"type": "string", "default": ""}
        },
        "required": []
    }
}

```

### 4.3 Windows Execution Sandbox & WSL Stub Avoidance

On Windows, `shutil.which("bash")` frequently resolves to `C:\Windows\System32\bash.exe`, which is an unconfigured WSL forwarding stub. Invoking it produces:
`This application requires the Windows Subsystem for Linux Optional Component...` in **UTF-16 LE**, which when read as UTF-8 generates diamond question-mark (``) corruption and fails to execute commands.

#### Resolution Engine:

1. **`find_safe_bash_executable()`:** Strictly ignores `System32\bash.exe` and `SysWOW64\bash.exe`. Searches exclusively for genuine Git Bash (`Git\bin\bash.exe`, `Git\usr\bin\bash.exe`) and MSYS2.
2. **`decode_terminal_bytes()`:** Inspects byte arrays for null bytes (`\x00`). If detected, decodes via `utf-16-le` / `utf-16`, strips residual nulls, and falls back to `utf-8` with error replacement.
3. **PowerShell Fallback:** When Git Bash is absent, commands are invoked through PowerShell with explicit UTF-8 encoding:
```powershell
powershell.exe -NoProfile -NonInteractive -Command "$OutputEncoding = [Console]::OutputEncoding = [Text.Encoding]::UTF8; <command>"

```



---

## 5. Artifact Lifecycle & Live Workspace Panel Triggering

### 5.1 Real-Time Artifact Detection

The execution workspace panel must automatically switch to `Preview` or `Artifacts` when the agent builds web apps or documents.

1. **Pre-Run Baselining:** Before executing steps, existing files in `project_dir` are recorded.
2. **Per-Observation Diffing:** After every tool call, directory contents are re-checked against baseline.
3. **Event Dispatching:** For every newly created file:
* File is added to `job_scoped_artifacts[job_id]`.
* Event `SSEEventType.OBSERVATION` is dispatched with payload:
```json
{
  "artifact": "index.html",
  "path": "index.html",
  "event": "artifact_created",
  "content": "Artifact created: index.html"
}

```




4. **Client-Side Trigger:** The Next.js client receives `artifact_created`, dispatches a DOM custom event `peldrun:artifact-created`, and automatically switches `WorkspacePanel` tabs.

---

## 6. Local LLM Integration & Production Resilience

### 6.1 LM Studio & Local Provider Validation

Local inference engines run in constrained VRAM environments. The bridge enforces pre-flight health checks:

* **Instance State Verification (`check_lmstudio_model_readiness`):**
Inspects `/api/v1/models` to confirm not only that the server is listening on port 1234, but that `loaded_instances` is non-empty (`is_loaded: True`).
* **Resilient Error Trapping:**
If an inference runner crashes (e.g. `openai.BadRequestError: Error code: 400 - {'error': 'Engine protocol predict request failed: fetch failed'}`), `format_smart_error` traps the crash, suppresses raw callstacks, and renders a recovery card instructing the operator to reload the model in LM Studio.

---

## 7. Historical Engineering Ledger (Resolved Issues)

| Incident / Bug Signature | Root Cause | Architectural Fix |
| --- | --- | --- |
| **Infinite Loop on Tool Invocation** | Tool parameters sent as `properties: {}`. Model invoked tools with `{}` repeatedly. | Implemented `CANONICAL_TOOL_SCHEMAS` enforcing required parameters (`code`, `command`, `path`). |
| **`TerminateTool` NotImplementedError** | Synchronous `_run()` was not implemented in subclass; base raised error. | Implemented `async def execute` and `async def _arun` returning standard `ToolResult`. |
| **`TerminateTool` TypeError on Init** | Missing abstract method `_arun` from `BaseTool` definition. | Added `_arun` implementation delegating to `execute`. |
| **WSL Prompt / Null Byte Corruption (``)** | Calling `C:\Windows\System32\bash.exe` without WSL enabled, outputting UTF-16 LE. | Implemented `find_safe_bash_executable()` and `decode_terminal_bytes()`. |
| **Missing `EventType.STEP_PROGRESS**` | Enum attribute accessed directly without verifying core definition. | Replaced with safe attribute check: `getattr(event.type, 'name', '')`. |
| **Un-Awaited Event Dropping** | Synchronous `EventEmitter` dropping async callback coroutines. | Created `_sync_event_handler` scheduling coroutines directly onto `main_loop.create_task`. |
| **Vanishing Step 1 upon Refresh (F5)** | Double increment on first step assigned real events to Step 2; Step 1 was discarded as empty. | Implemented `is_first_step_start` flag ensuring initial events anchor to Step 1. |
| **Live UI Step Flickering** | Background polling requests reading partial disk sessions with empty event arrays. | Added in-memory `ACTIVE_JOB_EVENTS` synchronized across job and chat identifiers. |
| **`ToolResult` Raw String Leakage** | `final_answer` contained stringified `ToolResult(output=...)`. | Added `sanitize_final_result_text()` extracting clean messages for the final turn card. |

---

## 8. Contributor Development & Validation Checklist

Before submitting PRs or modifying the core execution engine:

* [ ] **Schema Conformance:** New tools must define a Pydantic argument model and export a complete JSON schema via `model_json_schema()`.
* [ ] **Dual Async Contract:** Implement both `_arun(**kwargs)` and `execute(**kwargs)` on all tools inheriting from `BaseTool`.
* [ ] **No Shell Assumptions:** Never assume a POSIX shell exists. File operations must be reliable on native Windows, macOS, and Linux.
* [ ] **Event Thread Safety:** Never attach asynchronous functions directly to synchronous emitters without scheduling via `asyncio.get_running_loop().create_task()`.
* [ ] **Top-Level Event Keys:** All SSE payloads must expose `id`, `type`, `step`, and `content` at root level.
* [ ] **Isolated Path Resolution:** All filesystem reads and writes must be resolved using `Path(workspace_root) / path` and validated with `.is_relative_to(workspace_root)`.