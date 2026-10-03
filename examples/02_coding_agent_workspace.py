"""
PELDRUN Core Coding Agent Example.
Demonstrates automated code generation, artifact accumulation, and workspace containment
using CodingAgent and ShellExecTool.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from peldrun.agents import get_agent
from peldrun.events.emitter import EventEmitter
from peldrun.llm.client import LLMConfig
from peldrun.llm.providers import get_provider
from peldrun.memory import MemoryManager
from peldrun.tools.builtins import FileOpsTool, ShellExecTool
from peldrun.tools.registry import ToolRegistry


async def main() -> None:
    workspace = Path("./workspace_coding").resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    llm_provider = get_provider(
        "openai_compat",
        config=LLMConfig(
            api_base="http://localhost:1234/v1",
            model="local-model",
            temperature=0.2,
        ),
    )

    tool_registry = ToolRegistry(workspace_root=str(workspace))
    tool_registry.register(FileOpsTool(workspace_root=str(workspace)))
    tool_registry.register(ShellExecTool(workspace_root=str(workspace)))

    memory_manager = MemoryManager(
        workspace_root=str(workspace),
        system_prompt="You are an autonomous senior Python engineer writing reliable programs.",
    )
    await memory_manager.ainitialize()

    emitter = EventEmitter()

    agent = get_agent(
        agent_type="coding",
        llm_provider=llm_provider,
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        emitter=emitter,
    )

    print("=== Launching PELDRUN Coding Agent ===")
    task = (
        "Write a Python script named fibonacci.py that prints the first 10 Fibonacci numbers, "
        "then execute it via shell and verify the output."
    )
    state = await agent.arun(task)

    print("\n=== Coding Summary ===")
    print(f"Status:             {'COMPLETED' if state.is_completed else 'FAILED'}")
    print(f"Modified Artifacts: {state.metadata.get('modified_artifacts', [])}")
    print(f"Deliverables:       {state.deliverables}")
    print(f"Result:\n{state.final_output}")

    await llm_provider.close()


if __name__ == "__main__":
    asyncio.run(main())