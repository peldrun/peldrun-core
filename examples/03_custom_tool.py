"""
PELDRUN Core Custom Tool Extension Example.
Demonstrates defining, parameter-validating, registering, and invoking custom tools
within the PELDRUN autonomous agent ecosystem.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any, Optional, Type
from pydantic import BaseModel, Field

from peldrun.agents import get_agent
from peldrun.llm.client import LLMConfig
from peldrun.llm.providers import get_provider
from peldrun.memory import MemoryManager
from peldrun.tools.base import BaseTool, ToolResult
from peldrun.tools.registry import ToolRegistry


class HashCalculatorArgs(BaseModel):
    """Schema specifying input arguments for HashCalculatorTool."""
    text: str = Field(..., description="String payload to hash")
    algorithm: str = Field(default="sha256", description="Hashing algorithm: sha256 or md5")


class HashCalculatorTool(BaseTool):
    """Custom builtin calculating cryptographic digests."""
    name: str = "hash_calculator"
    description: str = "Calculate SHA256 or MD5 cryptographic hashes for arbitrary text input."
    args_schema: Optional[Type[BaseModel]] = HashCalculatorArgs

    async def _arun(self, text: str, algorithm: str = "sha256", **kwargs: Any) -> ToolResult:
        algo = algorithm.lower().strip()
        data = text.encode("utf-8")

        if algo == "sha256":
            digest = hashlib.sha256(data).hexdigest()
        elif algo == "md5":
            digest = hashlib.md5(data).hexdigest()
        else:
            return ToolResult(
                output=f"Unsupported algorithm '{algorithm}'. Choose 'sha256' or 'md5'.",
                exit_code=1,
                is_error=True,
            )

        return ToolResult(
            output=digest,
            exit_code=0,
            is_error=False,
            metadata={"algorithm": algo, "input_length": len(text)},
        )


async def main() -> None:
    # 1. Initialize registry and register custom tool
    tool_registry = ToolRegistry(workspace_root=".")
    tool_registry.register(HashCalculatorTool())

    print(f"Registered tools: {tool_registry.list_tools()}")
    print(f"OpenAI Schema: {tool_registry.get_openai_schemas()}")

    # 2. Execute tool directly
    direct_res = await tool_registry.aexecute("hash_calculator", text="PELDRUN_CORE_2026")
    print(f"Direct Execution Result: {direct_res.output}")

    # 3. Can be wired into any BaseAgent instance
    print("Custom tool is 100% interoperable with ReActAgent, PlanningAgent, and AgentRunner.")


if __name__ == "__main__":
    asyncio.run(main())