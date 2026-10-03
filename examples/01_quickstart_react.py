"""
PELDRUN Core Quickstart Example: ReAct Agent.
Demonstrates basic autonomous agent execution using the ReAct reasoning loop,
in-memory memory management, and scoped filesystem operations.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from peldrun.agents import get_agent
from peldrun.events.emitter import EventEmitter
from peldrun.events.schema import AgentEvent
from peldrun.llm.client import LLMConfig
from peldrun.llm.providers import get_provider
from peldrun.memory import MemoryManager
from peldrun.tools.builtins import FileOpsTool
from peldrun.tools.registry import ToolRegistry


async def main() -> None:
    # 1. Prepare sandboxed workspace
    workspace = Path("./workspace_quickstart").resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    # 2. Configure LLM Provider (Points to local LM Studio or Ollama by default)
    llm_provider = get_provider(
        "openai_compat",
        config=LLMConfig(
            api_base="http://localhost:1234/v1",
            model="local-model",
            temperature=0.7,
        ),
    )

    # 3. Setup Tool Registry with safe filesystem access
    tool_registry = ToolRegistry(workspace_root=str(workspace))
    tool_registry.register(FileOpsTool(workspace_root=str(workspace)))

    # 4. Initialize Short/Long Term Memory Coordinator
    memory_manager = MemoryManager(
        workspace_root=str(workspace),
        system_prompt="You are a helpful and meticulous software engineer assistant.",
    )
    await memory_manager.ainitialize()

    # 5. Setup Event Stream Emitter
    emitter = EventEmitter()

    async def log_event(event: AgentEvent) -> None:
        etype = event.type.value if hasattr(event.type, "value") else str(event.type)
        if etype == "thought":
            print(f"🤔 Thought: {event.data.get('thought', '')}")
        elif etype == "tool_call":
            print(f"🔧 Tool: {event.data.get('tool_name')} ({event.data.get('arguments')})")
        elif etype == "observation":
            output = str(event.data.get("output", ""))[:120]
            print(f"👁️ Observation: {output}...")

    emitter.subscribe_all(log_event)

    # 6. Instantiate Autonomous ReAct Agent via Factory
    agent = get_agent(
        agent_type="react",
        llm_provider=llm_provider,
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        emitter=emitter,
    )

    print("=== Launching PELDRUN ReAct Agent ===")
    task = "Create a file named hello_peldrun.txt with a welcome message and verify it exists."
    state = await agent.arun(task)

    print("\n=== Execution Summary ===")
    print(f"Status:       {'COMPLETED' if state.is_completed else 'FAILED'}")
    print(f"Total Steps:  {state.step}")
    print(f"Deliverables: {state.deliverables}")
    print(f"Output:\n{state.final_output}")

    await llm_provider.close()


if __name__ == "__main__":
    asyncio.run(main())