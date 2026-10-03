# ⚡ PELDRUN Core

**PELDRUN Core** is an enterprise-grade, sovereign autonomous agent orchestration engine authored in Python. Designed and implemented from first principles with zero legacy monkey-patching, it provides native support for self-directed reasoning loops, isolated filesystem and container execution boundaries, dynamic tool registries, local-first inference gateways, and typed Server-Sent Events (SSE) streaming.

---

## 🏛️ Repository Architecture

```
peldrun-core/
├── .github/
│   └── workflows/
│       └── ci.yml               # Multi-OS & multi-version CI test matrix
├── peldrun/
│   ├── __init__.py              # Package entrypoint & version metadata
│   ├── cli.py                   # Standalone CLI entrypoint (peldrun)
│   ├── engine/                  # State lifecycle, runner loop, graph, and checkpoints
│   │   ├── checkpoint.py        # Asynchronous disk state snapshot management
│   │   ├── graph.py             # Composable execution graph pipelines
│   │   ├── runner.py            # Execution loop orchestrator & event coordinator
│   │   └── state.py             # Canonical ExecutionState container & history models
│   ├── agents/                  # Autonomous agent archetypes & dynamic resolver
│   │   ├── base.py              # Abstract BaseAgent execution lifecycle
│   │   ├── coding_agent.py      # Artifact-accumulating software engineering agent
│   │   ├── factory.py           # Centralized get_agent() factory router
│   │   ├── planning_agent.py    # Deconstructive milestone-driven planning agent
│   │   └── react_agent.py       # Autonomous thought-action-observation reasoning loop
│   ├── tools/                   # Extensible tooling subsystem & protocol bridges
│   │   ├── base.py              # BaseTool contract & ToolResult definitions
│   │   ├── registry.py          # Unified ToolRegistry catalog & schema reflector
│   │   ├── mcp_client.py        # Model Context Protocol (MCP) dynamic bridge
│   │   ├── builtins/            # Scoped native tool implementations
│   │   │   ├── file_ops.py      # Boundary-checked filesystem operations
│   │   │   ├── human_input.py   # Operator-in-the-loop async suspension (ask_human)
│   │   │   ├── shell_exec.py    # Confined subprocess command execution
│   │   │   └── web_search.py    # Non-blocking web search with resilient HTTP fallback
│   │   └── transports/          # Pluggable MCP transport mechanisms
│   │       ├── sse.py           # Remote HTTP Server-Sent Events transport
│   │       └── stdio.py         # Subprocess standard input/output transport
│   ├── sandbox/                 # Isolation runtimes for safe code evaluation
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
│   ├── 01_quickstart_react.py
│   ├── 02_coding_agent_workspace.py
│   ├── 03_custom_tool.py
│   └── 04_local_lmstudio.py
├── tests/                       # Complete automated pytest test suite (79 passed)
├── pyproject.toml               # Hatchling packaging & script declarations
└── README.md

```

---

## ⚡ Core Subsystems

### 1. Engine & State (`peldrun.engine`)

* **`ExecutionState`**: The centralized mutable execution context tracking message exchanges, completed step counts, accumulated deliverables, runtime artifacts, and structured tool invocation histories.
* **`AgentRunner`**: Orchestrates sequential step execution, periodic telemetry snapshot broadcasts, external cancellation triggers, and graceful termination.
* **`ExecutionGraph`**: Directed execution graphs enabling linear pipelines and conditional step routing.
* **`CheckpointManager`**: Asynchronous disk persistence supporting immutable state serialization, listing, and state restoration.

### 2. Autonomous Agents (`peldrun.agents`)

* **`ReActAgent`**: Implements the classic autonomous reasoning loop. Parses reasoning tags (`<think>...</think>`), emits real-time thoughts, executes function calls, and synthesizes observations into final answers.
* **`PlanningAgent`**: Deconstructs complex objectives into formal milestone plans, tracking individual step statuses (`pending`, `in_progress`, `completed`).
* **`CodingAgent`**: Authoring agent specialized in software engineering tasks, automatically tracking and registering modified source files as deliverables.
* **`get_agent(agent_type, ...)`**: Factory routing function resolving configured agents instantly.

### 3. Tooling & MCP Subsystem (`peldrun.tools`)

* **`ToolRegistry`**: Unified catalog generating OpenAI-compatible function-calling JSON schemas from Pydantic arguments and dispatching async calls via `aexecute(tool_name, **kwargs)`.
* **Builtin Tools**:
* `file_ops`: Boundary-verified file operations (`read`, `write`, `append`, `list`, `exists`, `delete`).
* `shell_exec`: Command execution securely confined to the designated workspace directory.
* `web_search`: Non-blocking internet querying via `duckduckgo_search` with automated HTML fallback.
* `ask_human`: Interactive operator-in-the-loop tool pausing the agent loop until response submission.


* **Model Context Protocol (MCP)**:
* `DynamicMCPTool`: Dynamically wraps remote MCP tools into standard `BaseTool` instances.
* `StdioTransport`: Interacts with local MCP servers over standard input/output streams.
* `SSETransport`: Connects to remote HTTP Server-Sent Events endpoints.



### 4. Sandbox Isolation (`peldrun.sandbox`)

* **`LocalProcessSandbox`**: Confinement restricting script evaluation and filesystem access to host workspaces with directory traversal prevention (`resolve_safe_path`).
* **`DockerSandbox`**: Full container isolation with configurable memory ceilings (`memory_limit`), CPU quotas (`cpu_quota`), network gating (`network_disabled`), and bind-mounted workspaces.

### 5. Unified LLM Gateway (`peldrun.llm`)

* **`AsyncLLMClient`**: Resilient HTTP client using `httpx.AsyncClient` with connection pooling, exponential backoff retries, and SSE stream decoding.
* **Provider Adapters**:
* `OpenAICompatProvider`: Standard cloud and local compatible servers.
* `LMStudioProvider`: Zero-config connection to port `1234` with auto-discovery of loaded models and `<think>` token separation.
* `OllamaProvider`: Optimized connection to port `11434` with automatic model inventory polling.


* **Context Budgeting**: `ContextBudgetManager` and `truncate_messages_sliding_window` enforcing token limits while preserving system instructions and recent conversation context.

### 6. Event Stream Protocol (`peldrun.events`)

Emits nine canonical event types with fault-isolated listeners:

| Event Type | Description |
| --- | --- |
| `snapshot` | Periodic complete state dumps for user interfaces |
| `step_start` | Signals the initiation of an agent reasoning step |
| `thought` | Chain-of-thought reasoning tokens extracted from model output |
| `tool_call` | Parameters and tool identifier scheduled for execution |
| `observation` | Formatted output and status returned by an executed tool |
| `step_end` | Concludes an agent reasoning step |
| `final` | Terminal response payload answering the original prompt |
| `error` | Diagnostic details concerning runtime execution exceptions |
| `ask_human` | Human operator clarification request and option prompts |

Wire-ready Server-Sent Events generation is accessible via `emitter.astream_sse()`.

---

## 📦 Installation

Install `peldrun-core` as an editable package with development dependencies:

```bash
# Clone the repository
git clone https://github.com/peldrun/peldrun-core.git
cd peldrun-core

# Install in editable mode
pip install -e ".[dev]"

```

---

## 💻 Standalone CLI Usage

The package registers the executable `peldrun` command:

```bash
# Display installed version
peldrun version

# Inspect all registered tools and their functional descriptions
peldrun tools

# Validate connectivity to local LM Studio
peldrun check --provider lmstudio

# Validate connectivity to local Ollama
peldrun check --provider ollama --model llama3.1

# Execute an autonomous task using the ReAct agent
peldrun run "Inspect the git status and create a release summary" --agent react --verbose

# Run a coding agent with a specific step limit and workspace
peldrun run "Author a FastAPI healthcheck endpoint in app.py" --agent coding --workspace ./project --max-steps 10

```

---

## 🐍 Python SDK Quickstart

### 1. Basic ReAct Agent Execution

```python
import asyncio
from pathlib import Path

from peldrun.agents import get_agent
from peldrun.llm.client import LLMConfig
from peldrun.llm.providers import get_provider
from peldrun.memory import MemoryManager
from peldrun.tools.builtins import FileOpsTool
from peldrun.tools.registry import ToolRegistry


async def main() -> None:
    workspace = Path("./workspace").resolve()
    workspace.mkdir(exist_ok=True)

    # Configure local or cloud provider
    provider = get_provider(
        "openai_compat",
        config=LLMConfig(api_base="http://localhost:1234/v1", model="local-model"),
    )

    # Register tools
    tool_registry = ToolRegistry(workspace_root=str(workspace))
    tool_registry.register(FileOpsTool(workspace_root=str(workspace)))

    # Initialize memory
    memory = MemoryManager(
        workspace_root=str(workspace),
        system_prompt="You are a helpful software engineer assistant.",
    )
    await memory.ainitialize()

    # Instantiate agent
    agent = get_agent(
        agent_type="react",
        llm_provider=provider,
        tool_registry=tool_registry,
        memory_manager=memory,
    )

    # Run task
    state = await agent.arun("Write a configuration file named config.json with env=production.")

    print(f"Status: {state.status.value}")
    print(f"Deliverables: {state.deliverables}")
    print(f"Output:\n{state.final_output}")

    await provider.close()


if __name__ == "__main__":
    asyncio.run(main())

```

### 2. Defining a Custom Tool

```python
from typing import Any, Optional, Type
from pydantic import BaseModel, Field
from peldrun.tools.base import BaseTool, ToolResult
from peldrun.tools.registry import ToolRegistry


class TextReverseArgs(BaseModel):
    text: str = Field(..., description="String to reverse")


class TextReverseTool(BaseTool):
    name: str = "text_reverser"
    description: str = "Reverses the input text string."
    args_schema: Optional[Type[BaseModel]] = TextReverseArgs

    async def _arun(self, text: str, **kwargs: Any) -> ToolResult:
        return ToolResult(output=text[::-1], exit_code=0)


# Registration
registry = ToolRegistry()
registry.register(TextReverseTool())

```

---

## 🧪 Verification & Testing

Execute the comprehensive automated test suite across all subsystems:

```bash
python -m pytest tests/ -v

```

```text
============================== 79 passed in 6.08s ==============================

```

---

## 📄 License

Licensed under the Apache License, Version 2.0 (the "License"). You may obtain a copy of the License in the [LICENSE](https://www.google.com/search?q=LICENSE) file.