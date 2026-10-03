# 📘 PELDRUN Core — Developer & Technical Architecture Manual

```markdown
# PELDRUN Core — Engineering Architecture & Developer Manual

> **Package:** `peldrun-core`  
> **Version:** `0.1.0`  
> **Runtime:** Python >= 3.10  
> **Repository:** `https://github.com/peldrun/peldrun-core`

---

## 1. Architectural Philosophy & Invariants

PELDRUN Core is an autonomous agent orchestration engine designed to execute complex, multi-turn, tool-assisted reasoning workflows. It is architected around six non-negotiable engineering invariants:

1. **Zero Monkey-Patching:**  
   Runtime behavior is deterministic. Classes, methods, and thread contexts are never dynamically overridden or patched at runtime.
2. **Strict Schema Contracts:**  
   All message exchanges, state containers, event payloads, configuration classes, and tool parameters are enforced using immutable or strongly validated Pydantic v2 schemas.
3. **Workspace Boundary Confinement:**  
   All filesystem and process operations are strictly confined within a designated `workspace_root`. Any attempt to escape the boundary via directory traversal (`../`) raises explicit permission exceptions.
4. **Asynchronous Non-Blocking Execution:**  
   All internal execution loops, network calls, tool executions, and event streams utilize native `asyncio` constructs.
5. **Listener Fault Isolation:**  
   The event dispatcher isolates listener exceptions. A failing telemetry listener or SSE sink will never interrupt or crash the primary agent execution loop.
6. **Decoupled User Interface:**  
   The core exposes standardized Server-Sent Events (SSE) and persistent JSON snapshots, allowing it to interface seamlessly with any web dashboard, CLI, or microservice.

---

## 2. System Topology & Directory Map

```text
peldrun-core/
├── .github/
│   └── workflows/
│       └── ci.yml               # Multi-platform CI pipeline (Ubuntu, Windows / Python 3.10-3.12)
├── peldrun/
│   ├── __init__.py              # Root package metadata (__version__ = "0.1.0")
│   ├── cli.py                   # Standalone CLI entrypoint (`peldrun`)
│   ├── engine/                  # State lifecycle, runner loop, graph, and checkpoints
│   │   ├── checkpoint.py        # Disk-based asynchronous state persistence
│   │   ├── graph.py             # Composable Directed Acyclic Graph (DAG) pipeline builder
│   │   ├── runner.py            # Step coordinator, cancellation, and snapshot broadcaster
│   │   └── state.py             # Canonical ExecutionState container & history models
│   ├── agents/                  # Autonomous agent implementations & factory
│   │   ├── base.py              # BaseAgent lifecycle, step ceilings, and pause/resume
│   │   ├── coding_agent.py      # Artifact-accumulating software engineering agent
│   │   ├── factory.py           # Centralized get_agent() factory router
│   │   ├── planning_agent.py    # Deconstructive milestone-driven planning agent
│   │   └── react_agent.py       # Autonomous thought-action-observation reasoning loop
│   ├── tools/                   # Tool registry, builtins, and MCP bridge
│   │   ├── base.py              # BaseTool contract & ToolResult definitions
│   │   ├── registry.py          # Unified ToolRegistry catalog & schema reflector
│   │   ├── mcp_client.py        # Model Context Protocol (MCP) JSON-RPC client
│   │   ├── builtins/            # Scoped native tools
│   │   │   ├── file_ops.py      # Boundary-checked filesystem operations
│   │   │   ├── human_input.py   # Operator-in-the-loop async pause tool (ask_human)
│   │   │   ├── shell_exec.py    # Confined subprocess command execution
│   │   │   └── web_search.py    # Non-blocking web search with resilient HTTP fallback
│   │   └── transports/          # Pluggable MCP transport mechanisms
│   │       ├── sse.py           # Remote HTTP Server-Sent Events transport
│   │       └── stdio.py         # Subprocess standard input/output transport
│   ├── sandbox/                 # Environment isolation runtimes
│   │   ├── base.py              # BaseSandbox contract & boundary containment checks
│   │   ├── docker_sandbox.py    # Isolated container execution with resource limits
│   │   ├── factory.py           # get_sandbox() runtime selector
│   │   └── local_process.py     # Boundary-confined host directory sandbox
│   ├── llm/                     # Unified inference gateway & token budgeting
│   │   ├── client.py            # Non-blocking AsyncLLMClient with connection pooling
│   │   ├── tokenizer.py         # Token counting & sliding-window context managers
│   │   └── providers/           # Provider adapters & factory resolution
│   │       ├── lmstudio.py      # Dedicated LM Studio adapter with <think> extraction
│   │       ├── ollama.py        # Dedicated Ollama adapter with local model discovery
│   │       └── openai_compat.py # Standard OpenAI-compatible provider adapter
│   ├── events/                  # Typed event subsystem & wire streaming
│   │   ├── emitter.py           # Asynchronous EventEmitter with fault isolation
│   │   └── schema.py            # Pydantic schemas for the 9 canonical event types
│   └── memory/                  # Multi-tier memory subsystem
│       ├── long_term.py         # Persistent local knowledge storage & search
│       ├── manager.py           # MemoryManager prompt context synthesis
│       └── short_term.py        # Dialogue window management & summarization
├── examples/                    # Developer guides & end-to-end usage scripts
├── tests/                       # Complete automated pytest test suite (79 passed)
├── pyproject.toml               # Hatchling packaging & script declarations
└── README.md                    # Public documentation and usage reference

```

---

## 3. Subsystem Specifications & Contracts

### 3.1 Execution Engine (`peldrun.engine`)

The engine subsystem governs state mutation, execution lifecycle, checkpointing, and structured execution graphs.

#### `ExecutionState` (`state.py`)

The mutable context container passed across all agent steps:

* `task_prompt: str`: Initial user directive (defaults to empty string).
* `current_step: int`: Monotonically increasing execution step counter. Accessible via `state.step` (with getter and setter).
* `status: ExecutionStatus`: Lifecycle status (`IDLE`, `RUNNING`, `PAUSED`, `COMPLETED`, `FAILED`).
* `messages: List[ChatMessage]`: Complete conversational turn history (`SYSTEM`, `USER`, `ASSISTANT`, `TOOL`).
* `tool_history: List[ToolExecutionRecord]`: Audit trail of every tool invocation containing `tool_call_id`, `tool_name`, `arguments`, `output`, `exit_code`, `is_error`, and `artifacts`.
* `deliverables: List[str]`: Deduplicated list of generated artifacts or target files produced during execution.
* `metadata: Dict[str, Any]`: Telemetry and phase metadata.
* `final_output: Optional[str]`: Definitive completion answer.

#### `AgentRunner` (`runner.py`)

Coordinates agents adhering to the `StepExecutableAgent` protocol (`async def step(self, state: ExecutionState, emitter: EventEmitter) -> bool`):

1. Broadcasts an initial `EventType.SNAPSHOT` with state metadata.
2. Loops through steps, emitting `EventType.STEP_START` (`step_number=state.current_step`).
3. Dispatches execution to `agent.step(state, emitter)`.
4. Emits `EventType.STEP_END` (`step_number=state.current_step`).
5. Evaluates termination condition: returns if `step()` returns `True` or if `state.current_step >= config.max_steps`.
6. Handles external interrupts: calling `runner.cancel()` halts execution gracefully and sets `state.status = ExecutionStatus.PAUSED`.

#### `CheckpointManager` (`checkpoint.py`)

Persists `ExecutionState` dumps into `{workspace_root}/.peldrun/checkpoints/` as immutable JSON files. Supports `asave_checkpoint()`, `alist_checkpoints()`, `aload_latest_checkpoint()`, and `arestore_state(checkpoint_id)`.

---

### 3.2 Autonomous Agents (`peldrun.agents`)

All agents inherit from `BaseAgent` (`agents/base.py`), which manages the lifecycle loop, step safety ceilings, and suspension controls:

* `pause()`: Sets internal pause flag. If called before `arun()`, execution does not reset it.
* `resume()`: Clears pause flag and unblocks execution.

#### Agent Archetypes:

* **`ReActAgent` (`react_agent.py`):**
Implements the standard Reasoning + Acting loop. Extracts reasoning tokens enclosed within `<think>...</think>` tags and broadcasts them as `EventType.THOUGHT`. Dispatches tool calls, receives observations, and terminates when the model produces final text without tool calls.
* **`PlanningAgent` (`planning_agent.py`):**
Deconstructs tasks into milestone plans (`Plan` containing ordered `PlanStep` items). Sequentially iterates through milestones, updating step statuses from `pending` to `in_progress` to `completed`.
* **`CodingAgent` (`coding_agent.py`):**
Specialized for software engineering tasks. Automatically inspects file creation and modification calls, accumulating file paths into `state.metadata["modified_artifacts"]` and `state.deliverables`.
* **`get_agent(agent_type, ...)` (`factory.py`):**
Dynamic factory returning initialized agents by key (`"react"`, `"planning"`, `"coding"`).

---

### 3.3 Tools & Protocol Integrations (`peldrun.tools`)

#### Tool Contracts (`tools/base.py`)

All tools extend `BaseTool` and define a Pydantic argument model:

```python
class BaseTool(ABC):
    name: str = ""
    description: str = ""
    args_schema: Optional[Type[BaseModel]] = None
    workspace_root: Optional[str] = None

    async def aexecute(self, **kwargs: Any) -> ToolResult: ...
    @abstractmethod
    async def _arun(self, **kwargs: Any) -> ToolResult: ...

```

`ToolResult` encapsulates execution outcomes:

* `output: Any`: Primary payload or error message.
* `exit_code: int`: Status indicator (`0` = success, non-zero = failure).
* `is_error: bool`: Boolean error flag.
* `artifacts: List[str]`: Relative paths to generated files.
* `metadata: Dict[str, Any]`: Execution telemetry.

#### Builtin Tools:

* `FileOpsTool`: Bounded filesystem operations (`read`, `write`, `append`, `list`, `exists`, `delete`).
* `ShellExecTool`: Subprocess shell execution strictly isolated to `workspace_root` or its subdirectories.
* `WebSearchTool`: Internet querying using `duckduckgo_search` with automatic fallback to DuckDuckGo HTML scraping.
* `HumanInputTool`: Interactive operator input tool with `name="ask_human"`. Suspends the loop via `asyncio.Event` until `HumanInputTool.submit_human_response()` is called.

#### Model Context Protocol (MCP) Bridge:

* `DynamicMCPTool`: Dynamically reflects remote MCP server tool schemas into OpenAI function-calling formats.
* `MCPClient`: Dispatches JSON-RPC 2.0 requests (`tools/list`, `tools/call`).
* Pluggable Transports:
* `StdioTransport`: Runs local MCP server processes communicating via newline-delimited JSON-RPC over `stdin`/`stdout`.
* `SSETransport`: Connects to remote HTTP Server-Sent Events endpoints.



---

### 3.4 Sandbox Isolation (`peldrun.sandbox`)

All sandbox environments implement `BaseSandbox` and guarantee path containment:

* `resolve_safe_path(relative_path: str) -> Path`: Strictly validates that target paths reside within `workspace_root`. Any attempt to escape the boundary raises `PermissionError`.

#### Implementations:

* **`LocalProcessSandbox` (`local_process.py`):**
Executes commands on the local host bounded to `workspace_root`. Enforces configurable timeouts (returning exit code `124` on timeout), truncates output exceeding `max_output_chars`, and cleans up active subprocesses.
* **`DockerSandbox` (`docker_sandbox.py`):**
Provisions an isolated container using Docker CLI. Mounts the workspace to `/workspace`, sets resource constraints (`--memory`, `--cpus`), isolates networking (`--network none`), and cleans up containers upon task termination. Supports custom `cmd_runner` injection for testing.

---

### 3.5 LLM Gateway & Token Budgeting (`peldrun.llm`)

#### `AsyncLLMClient` (`llm/client.py`)

High-performance asynchronous client communicating with OpenAI-compatible endpoints:

* Shared `httpx.AsyncClient` with connection pooling (`Limits(max_keepalive_connections=20, max_connections=50)`).
* Exponential backoff retry loop for transient network drops and status codes `429`, `500`, `502`, `503`, `504`.
* Real-time Server-Sent Events (SSE) decoding yielding `StreamChunk` instances.
* Stream reconstruction utility: `accumulate_stream_chunks(stream)` consolidates tokens and delta tool calls into a unified `LLMResponse`.

#### Tokenizer & Context Management (`llm/tokenizer.py`)

* `estimate_tokens_from_string(text, model)`: Fast token calculation using `tiktoken` (cl100k_base) with heuristic fallback (`len(text) // 4`).
* `count_message_tokens(messages, model)`: Comprehensive message envelope token calculation including function call structures.
* `truncate_messages_sliding_window(messages, max_context_tokens)`: Sliding-window trimmer that drops older dialogue turns while guaranteeing that the initial `system` directive and the latest `user` prompt are preserved.
* `ContextBudgetManager`: Tracks generation capacity and enforces token ceilings.

#### Provider Adapters:

* `OpenAICompatProvider`: Standard provider adapter coordinating `ContextBudgetManager` and `AsyncLLMClient`.
* `LMStudioProvider`: Tailored for local LM Studio instances on port `1234`. Automatically queries `/models` to resolve active models when `model="auto"`, and extracts `<think>...</think>` traces.
* `OllamaProvider`: Tailored for local Ollama servers on port `11434`. Queries `/models` or `/api/tags` to resolve installed model tags.

---

### 3.6 Event Protocol (`peldrun.events`)

PELDRUN Core defines nine canonical event types:

```python
class EventType(str, Enum):
    SNAPSHOT = "snapshot"
    STEP_START = "step_start"
    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    OBSERVATION = "observation"
    STEP_END = "step_end"
    FINAL = "final"
    ERROR = "error"
    ASK_HUMAN = "ask_human"

```

#### `EventEmitter` (`events/emitter.py`)

* Supports typed (`subscribe(event_type, listener)`) and global (`subscribe_all(listener)`) async listeners.
* Fault-isolated dispatching: exceptions raised by listeners are caught and logged without disrupting execution.
* Direct SSE Streaming: `emitter.astream_sse()` yields W3C-compliant SSE strings:
```http
event: thought
data: {"id": "...", "type": "thought", "timestamp": 1727985600.0, "data": {"thought": "..."}, "metadata": {}}

```



---

### 3.7 Memory Subsystem (`peldrun.memory`)

* **`ShortTermMemory` (`memory/short_term.py`):**
Maintains active conversational turns. Enforces sliding-window pruning (`max_messages >= 5`) while preserving the system directive. Supports turn summarization (`summarize_old_messages`).
* **`LongTermMemory` (`memory/long_term.py`):**
Persists domain knowledge, preferences, and facts in `{workspace_root}/.peldrun/memory.json`. Provides keyword and category-based semantic search.
* **`MemoryManager` (`memory/manager.py`):**
Coordinates short-term and long-term memory, querying relevant knowledge and compiling final system prompts for LLM turns.

---

## 4. Extension & Developer Guides

### 4.1 Authoring a Custom Tool

To add a new tool, inherit from `BaseTool` and define a Pydantic argument schema:

```python
from typing import Any, Optional, Type
from pydantic import BaseModel, Field
from peldrun.tools.base import BaseTool, ToolResult

class CodeLinterArgs(BaseModel):
    filepath: str = Field(..., description="Path to Python file relative to workspace root")

class CodeLinterTool(BaseTool):
    name: str = "code_linter"
    description: str = "Runs static syntax analysis on a Python source file."
    args_schema: Optional[Type[BaseModel]] = CodeLinterArgs

    async def _arun(self, filepath: str, **kwargs: Any) -> ToolResult:
        try:
            safe_path = Path(self.workspace_root or ".").resolve() / filepath.strip().lstrip("/\\")
            content = safe_path.read_text(encoding="utf-8")
            compile(content, filepath, "exec")
            return ToolResult(output="Syntax OK", exit_code=0, is_error=False)
        except SyntaxError as syn_err:
            return ToolResult(
                output=f"Syntax Error: {syn_err.msg} at line {syn_err.lineno}",
                exit_code=1,
                is_error=True,
            )

```

Register the tool with the agent registry:

```python
tool_registry.register(CodeLinterTool(workspace_root=str(workspace)))

```

---

### 4.2 Authoring a Custom Agent Archetype

To implement a new reasoning model, subclass `BaseAgent` and implement `_astep()`:

```python
from peldrun.agents.base import BaseAgent
from peldrun.engine.state import MessageRole

class ReflectiveAgent(BaseAgent):
    """An agent that generates a draft, critiques it, and finalizes the output."""

    async def _astep(self) -> bool:
        # Step 1: Draft
        if self.state.step == 0:
            prompt = f"Draft an answer for: {self.state.task_prompt}"
            self.state.add_message(MessageRole.USER, prompt)
            resp = await self.llm_provider.generate(self.state.get_llm_messages())
            self.state.add_message(MessageRole.ASSISTANT, resp.content)
            await self.emitter.emit_thought(thought=f"Draft generated: {resp.content[:100]}...")
            return False  # Continue to next step

        # Step 2: Critique & Finalize
        critique_prompt = "Review your previous draft and provide the final polished output."
        self.state.add_message(MessageRole.USER, critique_prompt)
        final_resp = await self.llm_provider.generate(self.state.get_llm_messages())
        self.state.mark_completed(output=final_resp.content or "")
        return True  # Terminate loop

```

---

### 4.3 Adding a Custom LLM Provider

Subclass `BaseLLMProvider` and implement the abstract contracts:

```python
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from peldrun.llm.client import LLMResponse, StreamChunk
from peldrun.llm.providers.openai_compat import BaseLLMProvider

class CustomCloudProvider(BaseLLMProvider):
    @property
    def name(self) -> str:
        return "custom_cloud"

    async def check_health(self) -> bool:
        return True

    async def close(self) -> None:
        pass

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        # Custom non-streaming implementation
        return LLMResponse(content="Response from custom provider", finish_reason="stop")

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        # Custom streaming generator
        yield StreamChunk(content="Response ")
        yield StreamChunk(content="chunk.")

```

---

## 5. Standalone CLI Reference

The CLI entrypoint is declared in `pyproject.toml` under `[project.scripts] peldrun = "peldrun.cli:main"`:

```bash
# General help
peldrun --help

# Query package version
peldrun version

# Inspect all registered tools with their schemas
peldrun tools --workspace ./workspace

# Verify connectivity to local LLM provider
peldrun check --provider lmstudio
peldrun check --provider ollama --model llama3.1

# Execute an autonomous task
peldrun run "Inspect the git status and write a summary" \
    --agent react \
    --provider lmstudio \
    --model "qwen2.5-coder-7b" \
    --workspace ./my_project \
    --max-steps 15 \
    --verbose

```

---

## 6. Testing & Quality Assurance

The test suite enforces 100% pass rates across all 79 tests without live external dependencies:

```bash
# Install package with development dependencies
pip install -e ".[dev]"

# Run full test suite with verbose output
python -m pytest tests/ -v

```

### Test Subsystem Distribution:

* `tests/test_agents.py`: 7 tests (BaseAgent step ceiling, pause/resume, ReAct loop, Planning progression, Coding artifacts, Factory routing)
* `tests/test_core.py`: 3 tests (Package metadata, end-to-end multi-tool workflow, error recovery workflow)
* `tests/test_engine.py`: 12 tests (ExecutionState lifecycle, messages export, checkpoints, CheckpointManager, ExecutionGraph, AgentRunner)
* `tests/test_events.py`: 7 tests (EventType completeness, SSE formatting, typed subscriptions, fault tolerance, SSE streaming)
* `tests/test_llm.py`: 11 tests (AsyncLLMClient completions/streaming, OpenAICompatProvider, LMStudioProvider, OllamaProvider, Tokenizer)
* `tests/test_mcp.py`: 6 tests (MCPToolDefinition, DynamicMCPTool schemas, MCPClient, StdioTransport, SSETransport)
* `tests/test_memory.py`: 4 tests (ShortTermMemory pruning/summarization, LongTermMemory persistence, MemoryManager coordination)
* `tests/test_sandbox.py`: 12 tests (LocalProcessSandbox execution/timeout/traversal, DockerSandbox lifecycle/exec/timeout/health)
* `tests/test_tools.py`: 10 tests (ToolRegistry dispatch, FileOpsTool lifecycle/traversal, ShellExecTool, WebSearchTool fallback, AskHuman)
* `tests/test_cli.py`: 7 tests (CLI argument parsing, subcommands: version, tools, check, run success/error)

**Total:** **79 passed in ~6.08s (100% pass rate).**
 


