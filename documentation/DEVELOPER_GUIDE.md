# 📖 PELDRUN Core — Developer & Architecture Manual

```markdown
# PELDRUN Core: Developer & Technical Architecture Specification

## 1. System Overview & Core Philosophy

PELDRUN Core is a sovereign, high-performance autonomous agent execution engine implemented in Python (>=3.10). 
It is engineered from first principles as an independent, embeddable package (`peldrun-core`) designed to run 
decoupled from any specific user interface, while exposing structured telemetry and lifecycle control via 
Server-Sent Events (SSE).

### Architectural Invariants:
1. Zero Monkey-Patching: No runtime class patching, dynamic method overrides, or unmanaged thread state.
2. Explicit Contracts: All states, configurations, tool arguments, and event payloads are enforced via Pydantic v2.
3. Hermetic Isolation: File access and process executions are strictly scoped to bounded workspace environments.
4. Asynchronous-First: All internal loops, client requests, tool invocations, and streams are non-blocking (`asyncio`).
5. Fault Isolation: Unhandled exceptions inside tool executions or event listeners never crash the core agent loop.

---

## 2. Directory Layout & Module Responsibilities

```text
peldrun-core/
├── peldrun/
│   ├── __init__.py           # Package entrypoint & version metadata (0.1.0)
│   ├── cli.py                # Standalone CLI interface (peldrun)
│   ├── engine/               # State, graph, checkpoints, and loop execution
│   │   ├── state.py          # ExecutionState and Message/Tool execution contracts
│   │   ├── runner.py         # Step loop coordinator & snapshot broadcaster
│   │   ├── checkpoint.py     # Disk persistence & state restoration manager
│   │   └── graph.py          # Directed acyclic graph pipeline builder
│   ├── agents/               # Autonomous agent archetypes & factory
│   │   ├── base.py           # BaseAgent lifecycle, step ceilings, and pause/resume
│   │   ├── react_agent.py    # Thought-Action-Observation reasoning loop
│   │   ├── planning_agent.py # Milestone deconstruction and progress tracking
│   │   ├── coding_agent.py   # Code generation & modified artifact accumulator
│   │   └── factory.py        # Dynamic agent resolution via get_agent()
│   ├── tools/                # Extensible tooling subsystem & MCP integration
│   │   ├── base.py           # BaseTool interface & ToolResult contracts
│   │   ├── registry.py       # Catalog for reflection, dispatch, and schema export
│   │   ├── mcp_client.py     # JSON-RPC 2.0 dynamic Model Context Protocol bridge
│   │   ├── builtins/         # Scoped native tools
│   │   │   ├── file_ops.py   # Boundary-checked filesystem CRUD operations
│   │   │   ├── shell_exec.py # Subprocess execution confined to workspace
│   │   │   ├── web_search.py # Resilient search with automated HTML fallback
│   │   │   └── human_input.py# Operator-in-the-loop async pause tool (ask_human)
│   │   └── transports/       # MCP transport layer implementations
│   │       ├── stdio.py      # Local subprocess stdin/stdout transport
│   │       └── sse.py        # Remote HTTP Server-Sent Events transport
│   ├── sandbox/              # Environment isolation runtimes
│   │   ├── base.py           # BaseSandbox contract & boundary containment checks
│   │   ├── local_process.py  # Host directory sandbox with directory traversal block
│   │   ├── docker_sandbox.py # OCI container sandbox with resource/network limits
│   │   └── factory.py        # get_sandbox() runtime resolution factory
│   ├── llm/                  # Inference client, providers, and token budgeting
│   │   ├── client.py         # AsyncLLMClient with connection pooling & retries
│   │   ├── tokenizer.py      # ContextBudgetManager & sliding-window context pruning
│   │   └── providers/        # Standardized LLM provider adapters
│   │       ├── openai_compat.py # Generic OpenAI-compatible HTTP adapter
│   │       ├── lmstudio.py   # LM Studio adapter with <think> trace extraction
│   │       └── ollama.py     # Ollama adapter with local tag discovery
│   ├── events/               # Typed event schemas and real-time streaming
│   │   ├── schema.py         # 9 canonical event schemas (Pydantic v2)
│   │   └── emitter.py        # Async event emitter with fault isolation & SSE stream
│   └── memory/               # Multi-tier memory subsystem
│       ├── short_term.py     # Message buffer with automatic summarization
│       ├── long_term.py      # Persistent JSON-based categorized knowledge store
│       └── manager.py        # Unified prompt compiler & memory coordinator
├── tests/                    # Hermetic unit and integration test suite (79 tests)
├── examples/                 # Reference developer scripts and workflows
├── pyproject.toml            # Build configuration, metadata, and dependencies
└── README.md                 # Public documentation and usage reference

```

---

## 3. Subsystem Deep-Dive & Data Contracts

### 3.1 The Engine Subsystem (`peldrun.engine`)

`ExecutionState` is the single source of truth for an execution turn:

* `task_prompt`: Original natural-language instruction (defaults to `""`).
* `current_step`: Current step number, exposed via the `step` property with setter.
* `messages`: Ordered sequence of conversation messages (`SYSTEM`, `USER`, `ASSISTANT`, `TOOL`).
* `tool_history`: Chronological audit log of every tool execution (`ToolExecutionRecord`).
* `deliverables`: Unique list of generated outputs (files, charts, reports).
* `status`: `IDLE`, `RUNNING`, `PAUSED`, `COMPLETED`, `FAILED`.

`AgentRunner` executes an agent implementing the `StepExecutableAgent` protocol:

1. Emits `EventType.SNAPSHOT` with full state metadata.
2. Emits `EventType.STEP_START` (`step_number=state.current_step`).
3. Calls `await agent.step(state, emitter)`.
4. Emits `EventType.STEP_END` (`step_number=state.current_step`).
5. Handles cancellation requests (`runner.cancel()`) gracefully by setting `state.status = ExecutionStatus.PAUSED`.

`CheckpointManager` persists complete snapshots to disk under `{workspace_root}/.peldrun/checkpoints/`
as immutable JSON files, ensuring full crash-recovery capabilities.

### 3.2 The Autonomous Agents Subsystem (`peldrun.agents`)

Agents inherit from `BaseAgent`, which encapsulates:

* Maximum step safety ceiling (`max_steps`).
* Execution controls: `pause()` and `resume()`, preserving pre-execution pause directives.
* The execution loop: calls internal abstract `_astep()` until completion or limit.

#### Built-in Agent Archetypes:

* `ReActAgent`: Prompts the LLM with tool schemas, parses chain-of-thought tokens wrapped in `<think>...</think>`,
broadcasts `thought` events, executes tool calls, records `observation` events, and terminates upon receiving final text.
* `PlanningAgent`: Issues an initial planning turn expecting a structured JSON milestone plan (`Plan`, `PlanStep`),
then advances through milestones sequentially, marking step completion.
* `CodingAgent`: Extends execution to inspect file system modifications, automatically registering authored files in
`state.metadata["modified_artifacts"]` and `state.deliverables`.

### 3.3 The Tooling & MCP Subsystem (`peldrun.tools`)

All tools derive from `BaseTool`:

* `name`: Unique identifier string matching regex `^[a-zA-Z0-9_-]+$`.
* `description`: Functional explanation injected into the LLM system prompt.
* `args_schema`: Pydantic `BaseModel` class defining parameter types and descriptions.
* Execution entrypoints:
* `aexecute(**kwargs) -> ToolResult`: Public async entrypoint that validates arguments, catches unhandled exceptions,
and returns a structured `ToolResult(output=..., exit_code=..., is_error=..., artifacts=..., metadata=...)`.
* `_arun(**kwargs) -> ToolResult`: Internal execution logic implemented by subclasses.



`ToolRegistry` registers `BaseTool` instances, exports canonical OpenAI function-calling schemas via
`get_openai_schemas()`, and dispatches execution asynchronously.

#### Model Context Protocol (MCP) Integration:

* `DynamicMCPTool`: Wraps external MCP servers, dynamically converting MCP JSON-RPC schemas into OpenAI function declarations.
* Transports:
* `StdioTransport`: Spawns a local process and communicates using newline-delimited JSON-RPC over `stdin`/`stdout`.
* `SSETransport`: Connects to remote HTTP endpoints utilizing Server-Sent Events for downstream JSON-RPC messaging.



### 3.4 The Sandbox Isolation Subsystem (`peldrun.sandbox`)

All sandboxes implement `BaseSandbox` to guarantee filesystem containment:

* `resolve_safe_path(relative_path) -> Path`: Strictly verifies that target paths reside inside `workspace_root`.
Any traversal attempt (`../../etc/passwd`) immediately raises `PermissionError`.
* `LocalProcessSandbox`: Executes commands on the host using `asyncio.create_subprocess_shell` bounded to the workspace directory.
Enforces process timeouts (`exit_code=124`), stream character truncation, and process tree termination (`acleanup()`).
* `DockerSandbox`: Provisions an isolated OCI container via Docker CLI (`docker run -d -v {workspace}:/workspace -w /workspace`).
Supports CPU limits (`--cpus`), memory ceilings (`--memory`), and network gating (`--network none`).

### 3.5 The LLM Gateway Subsystem (`peldrun.llm`)

Centralized gateway preventing vendor lock-in:

* `AsyncLLMClient`: Thin, non-blocking HTTP client using `httpx.AsyncClient` with connection pooling, configurable timeouts,
and exponential backoff retry policies for transient network drops (429, 5xx).
* Context Window Management:
* `estimate_tokens_from_string()`: Accurately estimates token consumption using `tiktoken` (cl100k_base) with heuristic fallback.
* `truncate_messages_sliding_window()`: Prunes intermediate dialogue turns while strictly guaranteeing the preservation
of the initial `system` directive and the latest `user` prompt.
* `ContextBudgetManager`: Dynamically calculates remaining generation headroom.


* Providers:
* `OpenAICompatProvider`: Standard OpenAI endpoints.
* `LMStudioProvider`: Local inference on port 1234; auto-discovers loaded models and extracts `<think>` traces.
* `OllamaProvider`: Local inference on port 11434; discovers installed tags via `/models` or `/api/tags`.



### 3.6 The Events Subsystem (`peldrun.events`)

Broadcasting mechanism implementing the 9 canonical PELDRUN event types:

1. `snapshot`: Complete state serializations emitted at lifecycle boundaries.
2. `step_start`: Signals the commencement of an agent reasoning step.
3. `thought`: Real-time chain-of-thought tokens extracted from model generation.
4. `tool_call`: Parameter payloads dispatched to a registered tool.
5. `observation`: Formatted tool execution output returned to the agent.
6. `step_end`: Signals the conclusion of an agent reasoning step.
7. `final`: Definitive textual answer delivered to the user.
8. `error`: Uncaught runtime exceptions with diagnostic metadata.
9. `ask_human`: Interactive prompt pausing the engine awaiting operator intervention.

`EventEmitter` features listener fault-isolation (one failing listener cannot disrupt others) and exposes
`astream_sse()`, which formats events directly according to the W3C Server-Sent Events specification:

```http
event: thought
data: {"id": "uuid", "type": "thought", "timestamp": 1727985600.0, "data": {"thought": "..."}, "metadata": {}}

```

---

## 4. Developer Workflows & How-To Guides

### 4.1 Creating a New Custom Tool

To add a new tool to PELDRUN, derive from `BaseTool` and define a Pydantic schema:

```python
from typing import Any, Optional, Type
from pydantic import BaseModel, Field
from peldrun.tools.base import BaseTool, ToolResult

class CurrencyConvertArgs(BaseModel):
    amount: float = Field(..., description="Numerical amount to convert")
    from_currency: str = Field(..., description="3-letter source currency code (e.g. USD)")
    to_currency: str = Field(..., description="3-letter target currency code (e.g. EUR)")

class CurrencyConvertTool(BaseTool):
    name: str = "currency_converter"
    description: str = "Convert monetary amounts between major world currencies."
    args_schema: Optional[Type[BaseModel]] = CurrencyConvertArgs

    async def _arun(self, amount: float, from_currency: str, to_currency: str, **kwargs: Any) -> ToolResult:
        # Business logic here
        converted = amount * 0.92  # Example static rate
        return ToolResult(
            output=f"{amount} {from_currency.upper()} = {converted:.2f} {to_currency.upper()}",
            exit_code=0,
            is_error=False,
            metadata={"rate": 0.92},
        )

```

Registration is immediate:

```python
registry.register(CurrencyConvertTool())

```

### 4.2 Adding a New Agent Archetype

To implement a novel reasoning architecture, subclass `BaseAgent` and implement `_astep()`:

```python
from peldrun.agents.base import BaseAgent
from peldrun.engine.state import MessageRole

class DebateAgent(BaseAgent):
    """Agent that alternates between proposer and critic personas."""

    async def _astep(self) -> bool:
        current_step = self.state.step
        if current_step >= self.config.max_steps:
            self.state.mark_completed(output="Debate concluded upon reaching step limit.")
            return False

        # Formulate argument
        response = await self.llm_provider.generate(self.state.get_llm_messages())
        self.state.add_message(MessageRole.ASSISTANT, response.content)
        await self.emitter.emit_thought(thought=f"Turn {current_step}: {response.content}")

        if "CONCEDE" in (response.content or ""):
            self.state.mark_completed(output=response.content)
            return False

        return True

```

### 4.3 Running the Automated Test Suite

PELDRUN Core enforces 100% test pass rates across all subsystems without flaky network dependencies:

```bash
# Install package in editable mode with development fixtures
pip install -e ".[dev]"

# Execute entire test matrix
python -m pytest tests/ -v

```

Expected result:

```text
============================== 79 passed in ~6.0s ==============================

```

### 4.4 Standalone CLI Commands

```bash
# Print version
peldrun version

# Enumerate registered tools with OpenAI-ready descriptions
peldrun tools

# Validate connectivity to local LM Studio
peldrun check --provider lmstudio

# Execute an autonomous task
peldrun run "Analyze codebase and output summary" --agent react --verbose

```

 