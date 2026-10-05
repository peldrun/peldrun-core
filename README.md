# ⚡ PELDRUN Core

**Modular runtime for autonomous AI agents.**

PELDRUN Core is the Python execution layer behind PELDRUN. It provides the pieces needed to run tool-using agents without tying the agent to one model provider, one tool implementation, or one user interface.

The core is built around a small set of explicit contracts:

* agents execute tasks step by step
* tools expose typed inputs and structured results
* LLM providers share one interface
* MCP tools can be discovered and invoked through the same tool layer
* execution state is kept separately from the UI
* runtime events are typed and streamable
* workspaces can be restricted and checked by a security policy
* graphs and checkpoints are available for structured execution

PELDRUN Core is currently **Beta (0.2.0)**. The API is usable, but some interfaces are still evolving as the PELDRUN Web integration is finalized.

---

## Why use it?

Most agent projects start as a loop:

```text
prompt -> LLM -> tool -> observation -> LLM -> ...
```

That loop works until the application needs multiple agents, real tools, local models, MCP, streaming, memory, workspace isolation, cancellation, or a second frontend.

PELDRUN Core keeps those concerns in separate modules so they can be replaced without rewriting the execution model.

```text
                    PELDRUN Core
                         │
        ┌────────────────┼────────────────┐
        │                │                │
      Agents           Tools            LLM
        │                │                │
   ToolCallAgent     Built-ins          OpenAI
   CodingAgent       Custom tools       LM Studio
   PlanningAgent     MCP tools          Ollama
        │                │                │
        └────────────────┼────────────────┘
                         │
              Execution / Events
                         │
             Web, CLI, or another app
```

The agent does not need to know where a tool came from or which UI is consuming its events.

---

## What is included

### Agents

The core includes:

* `BaseAgent` — common lifecycle, state, memory, tools, sandbox and event integration.
* `ToolCallAgent` — autonomous tool-calling loop.
* `ReActAgent` — base ReAct structure for custom think/act implementations.
* `PlanningAgent` — plan, execute and track structured milestones.
* `CodingAgent` — code-oriented agent with workspace changes and verification loops.
* `MultiAgentCoordinator` — delegate work from a parent agent to registered specialists.

A typical tool-calling run is:

```text
LLM
 ↓
tool_calls
 ↓
resolve tool
 ↓
execute
 ↓
observation
 ↓
LLM again
```

The loop ends when the model returns a final response, a termination tool is used, or the configured step limit is reached.

---

## LLM providers

The LLM layer is provider-agnostic.

Included adapters:

* `OpenAICompatProvider`
* `LMStudioProvider`
* `OllamaProvider`

Any service that exposes an OpenAI-compatible chat completion and tool-calling API can be used through `OpenAICompatProvider`.

The provider layer normalizes:

* text responses
* reasoning / thinking content
* tool calls
* streaming chunks
* finish reasons
* token usage

The client also normalizes common reasoning fields such as `reasoning`, `reasoning_content`, and `thought`.

### Example

```python
from peldrun.llm.client import LLMConfig
from peldrun.llm.providers import get_provider

provider = get_provider(
    "openai_compat",
    config=LLMConfig(
        api_base="http://localhost:1234/v1",
        model="your-model",
        temperature=0.2,
    ),
)
```

LM Studio:

```python
provider = get_provider(
    "lmstudio",
    api_base="http://localhost:1234/v1",
    model="auto",
)
```

Ollama:

```python
provider = get_provider(
    "ollama",
    api_base="http://localhost:11434/v1",
    model="your-model",
)
```

---

## Tools

Tools implement a small contract:

```python
class BaseTool:
    name: str
    description: str
    args_schema: Optional[Type[BaseModel]]

    async def _arun(self, **kwargs) -> ToolResult:
        ...
```

`BaseTool` handles argument validation and converts execution failures into `ToolResult` objects.

A result contains:

```text
output
exit_code
is_error
artifacts
metadata
```

This keeps tool execution independent from the LLM provider.

### Built-in tools

The repository includes tools for:

* file operations
* shell execution
* web search
* human input
* string/file editing
* termination
* agent delegation

Tools can be grouped in a `ToolCollection` and exported as OpenAI-compatible function schemas.

### Custom tools

A custom tool is a normal `BaseTool` implementation:

```python
from typing import Any, Optional, Type
from pydantic import BaseModel, Field

from peldrun.tools.base import BaseTool, ToolResult


class ReverseArgs(BaseModel):
    text: str = Field(..., description="Text to reverse")


class ReverseTool(BaseTool):
    name = "reverse_text"
    description = "Reverse a string."
    args_schema: Optional[Type[BaseModel]] = ReverseArgs

    async def _arun(self, text: str, **kwargs: Any) -> ToolResult:
        return ToolResult(output=text[::-1])
```

Register it and the agent can use it through the same function-calling interface as built-in tools.

See [`examples/03_custom_tool.py`](examples/03_custom_tool.py).

---

## MCP

PELDRUN Core can turn tools exposed by an MCP server into normal PELDRUN tools.

The MCP layer includes:

* JSON-RPC tool discovery
* `tools/list`
* `tools/call`
* dynamic `DynamicMCPTool` wrappers
* pluggable transports
* SSE transport
* stdio transport

Once discovered, an MCP tool follows the same `BaseTool` / `ToolResult` path as a native tool.

```text
MCP Server
    │
    ▼
 MCP Client
    │
    ▼
DynamicMCPTool
    │
    ▼
ToolCollection / Agent
```

This lets applications add external capabilities without adding provider-specific logic to the agent.

---

## Execution state

`ExecutionState` is the central runtime state for an agent run.

It tracks:

* `run_id`
* task prompt
* execution status
* current step
* maximum steps
* agent name
* workspace root
* conversation messages
* tool execution history
* deliverables
* metadata
* checkpoints
* final output

The state is separate from the web layer. A UI can consume events and snapshots without owning the actual execution state.

---

## Runner

`AgentRunner` provides a generic step-based execution controller around agents implementing the `StepExecutableAgent` protocol.

It handles:

* step limits
* per-step timeouts
* total timeouts
* pause / resume signals
* cancellation
* state snapshots
* checkpoint creation
* event emission

```text
AgentRunner
    │
    ├── ExecutionState
    ├── Agent
    ├── EventEmitter
    └── CheckpointManager
```

The runner does not need to know the concrete agent implementation.

---

## Graphs

The engine contains a lightweight directed execution graph in `peldrun.engine.graph`.

```text
START
  │
  ▼
Analyze
  │
  ▼
Execute
  │
  ├── success ──► Verify ──► END
  │
  └── retry ────► Execute
```

The current graph implementation supports:

* named nodes
* synchronous or asynchronous node actions
* fixed edges
* conditional routing
* explicit finish points
* graph compilation and validation
* iteration limits

Example:

```python
from peldrun.engine.graph import ExecutionGraph, END


async def analyze(state):
    state.metadata["phase"] = "analyze"


async def finish(state):
    state.metadata["phase"] = "done"


graph = (
    ExecutionGraph()
    .add_node("analyze", analyze)
    .add_node("finish", finish)
    .set_entry_point("analyze")
    .add_edge("analyze", "finish")
    .add_edge("finish", END)
)

compiled = graph.compile()
```

The graph layer is intentionally small. It is an execution primitive, not a separate workflow platform.

---

## Checkpoints

`CheckpointManager` provides filesystem-backed state snapshots.

```text
ExecutionState
      │
      ▼
CheckpointManager
      │
      ▼
.checkpoints/*.json
      │
      ▼
restore
      │
      ▼
ExecutionState
```

A checkpoint contains a serialized `ExecutionState` and can be restored later.

This is useful for long-running tasks, debugging, and basic recovery workflows.

The current implementation is deliberately simple: checkpoints are JSON files inside the workspace. More advanced durable execution semantics can be built on top of this layer as the runtime evolves.

---

## Events and streaming

PELDRUN Core exposes a typed event protocol through `PeldrunEvent` and `EventEmitter`.

The canonical event types are:

| Event            | Purpose                      |
| ---------------- | ---------------------------- |
| `snapshot`       | Current execution state      |
| `step_start`     | Step started                 |
| `thought`        | Reasoning / thinking payload |
| `agent_activity` | Execution status             |
| `tool_call`      | Tool invocation              |
| `observation`    | Tool result                  |
| `step_end`       | Step completed               |
| `final`          | Final task output            |
| `error`          | Execution error              |
| `ask_human`      | Agent requests user input    |

Every event uses a common envelope containing:

```text
version
event_id
sequence
run_id
timestamp
step
type
payload
metadata
```

The envelope is validated with Pydantic and can be serialized directly to SSE.

This makes the event stream usable by a web UI, CLI, logs, or another runtime consumer.

---

## Memory

The memory subsystem provides two levels.

### Short-term memory

Maintains the active conversation and context window used by the agent.

### Long-term memory

Stores persistent project knowledge and can retrieve relevant entries for a new task.

`MemoryManager` coordinates both layers and can build an effective system prompt with relevant stored context.

---

## Workspace and security

Agent tools should not have unrestricted access to the host filesystem.

PELDRUN provides workspace-aware execution and a `SecurityPolicy` for enforcing boundaries.

The security layer covers:

* workspace path validation
* path traversal prevention
* sensitive file protection
* shell execution policy
* blocked destructive command patterns
* network access policy
* action risk classification
* optional human approval for destructive actions
* execution time limits

Two sandbox implementations are available:

* `LocalProcessSandbox` — host execution with workspace restrictions.
* `DockerSandbox` — container execution with configurable CPU, memory and network settings.

Security boundaries are part of the runtime design, not a UI feature.

---

## Artifacts

`ArtifactManager` tracks files produced during a run.

It can:

* capture a workspace baseline
* detect newly created files
* build an artifact manifest
* calculate SHA-256 hashes
* produce a deliverables summary

This gives applications a structured way to report what an agent actually produced.

---

## Runtime contract

`peldrun.runtime.contract` defines the boundary between an application and the execution core.

### `WorkspaceContext`

Describes the workspace used by a run without changing the process working directory.

### `AgentSpec`

Describes the selected agent and its runtime configuration.

### `RunRequest`

Carries:

* run ID
* job ID
* prompt
* agent specification
* workspace
* LLM configuration
* active MCP servers
* application metadata

This contract is used by PELDRUN Web to communicate with the core runtime and can also serve as a boundary for another host application.

---

## Multi-agent execution

The core includes a parent/child delegation layer.

A parent can register specialist agents and delegate a task to one of them. Child execution receives its own run ID and its events can be bridged back to the parent stream with lineage metadata.

```text
Parent Agent
     │
     ├── delegate ──► Coding Agent
     ├── delegate ──► Research Agent
     └── delegate ──► Custom Agent
```

This is built on the normal agent and event contracts rather than a separate agent protocol.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/peldrun/peldrun-core.git
cd peldrun-core
```

Install the package and development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Optional integrations:

```bash
python -m pip install -e ".[mcp]"
python -m pip install -e ".[docker]"
```

Python 3.10 or newer is required.

### Current integration note

PELDRUN Core is being developed together with the PELDRUN Web runtime. Some compatibility code is still shared with the Web application, and the standalone package boundary is being cleaned up as the public API stabilizes.

For the current PELDRUN application, the recommended path is the PELDRUN Web integration.

---

## Development

Run the test suite:

```bash
python -m pytest tests/ -v
```

CI currently tests Python 3.10, 3.11 and 3.12 on Ubuntu and Windows, and also builds the package distribution.

Run local checks when needed:

```bash
flake8 peldrun tests
mypy peldrun
```

---

## Examples

The `examples/` directory contains small integration examples:

| Example                        | Shows                                  |
| ------------------------------ | -------------------------------------- |
| `01_quickstart_react.py`       | Basic autonomous execution             |
| `02_coding_agent_workspace.py` | Coding agent and workspace tools       |
| `03_custom_tool.py`            | Creating and registering a custom tool |
| `04_local_lmstudio.py`         | Local LM Studio integration            |

---

## Repository layout

```text
peldrun-core/
├── peldrun/
│   ├── agents/       # Agent implementations and multi-agent delegation
│   ├── artifacts/    # Artifact tracking and manifests
│   ├── engine/       # State, runner, graphs and checkpoints
│   ├── events/       # Event contracts, bus and emitter
│   ├── flows/        # Higher-level execution flows
│   ├── llm/          # LLM client and provider adapters
│   ├── memory/       # Short- and long-term memory
│   ├── runtime/      # Application/runtime contracts
│   ├── sandbox/      # Local and Docker execution
│   ├── security/     # Runtime security policy
│   └── tools/        # Tool contracts, built-ins and MCP
├── examples/         # Small examples
├── tests/            # Automated test suite
├── documentation/    # Developer and architecture documentation
├── pyproject.toml
├── CHANGELOG.md
└── LICENSE
```

---

## Design principles

1. **Keep the execution loop independent from the UI.**
2. **Treat tools as contracts, not hard-coded capabilities.**
3. **Do not make the agent depend on a specific model vendor.**
4. **Keep execution state explicit and serializable.**
5. **Stream structured events instead of UI-specific messages.**
6. **Keep workspace and security boundaries close to execution.**
7. **Prefer small runtime primitives that applications can compose.**

The goal is not to hide the runtime behind a large framework. The goal is to make the runtime understandable enough that another developer can change it.

---

## PELDRUN Web

PELDRUN Core is the execution layer. PELDRUN Web provides the application layer around it: workspaces, jobs, UI, tool configuration, MCP configuration, SSE consumption and engine selection.

The intended relationship is:

```text
PELDRUN Web
     │
     ▼
Engine Contract
     │
 ┌───┴──────────────┐
 ▼                  ▼
PELDRUN Core      OpenManus
Runtime           Compatibility
```

Repository: [https://github.com/peldrun/peldrun](https://github.com/peldrun/peldrun)

---

## Contributing

Issues and pull requests are welcome.

Before adding a new abstraction, check whether the existing agent, tool, event, state, provider, or runtime contracts already cover the use case.

For larger architectural changes, open an issue first and describe:

* the problem
* the proposed interface
* why the existing contracts are insufficient
* how the change affects agents, tools, events and the Web integration

---

## License

Apache License 2.0. See [`LICENSE`](LICENSE).

---

## Status

PELDRUN Core is an active open-source project under development.

If you are building an agent runtime, tool system, local AI application, MCP integration, or developer-facing autonomous agent, the project is intended to be used as a base you can inspect, modify and extend.

**Star the repository if you want to follow the project. Fork it if you want to build on the runtime.**