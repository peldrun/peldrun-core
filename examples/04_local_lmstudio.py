"""
PELDRUN Core Local LM Studio Integration Example.
Demonstrates zero-config connection to local LM Studio inference engine (port 1234),
active model discovery, and reasoning token extraction (<think> tags).
"""

from __future__ import annotations

import asyncio
from peldrun.llm.providers import get_provider
from peldrun.llm.providers.lmstudio import LMStudioProvider


async def main() -> None:
    print("[*] Connecting to local LM Studio (http://localhost:1234/v1)...")

    # Connect with auto-model resolution
    provider = get_provider("lmstudio", api_base="http://localhost:1234/v1", model="auto")

    # 1. Health check & model discovery
    is_healthy = await provider.check_health()
    if not is_healthy:
        print("[!] LM Studio is not running or has no model loaded on port 1234.")
        print("    Please start LM Studio, load a model, and start the local server.")
        return

    if isinstance(provider, LMStudioProvider):
        active_model = await provider.discover_active_model()
        print(f"[OK] Detected Active LM Studio Model: {active_model}")

    # 2. Streaming completion with reasoning extraction
    prompt = [{"role": "user", "content": "Explain in two sentences what an autonomous agent is."}]
    print("\n--- Streaming Response ---")

    full_text = []
    async for chunk in provider.stream(messages=prompt):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            full_text.append(chunk.content)

    print("\n--------------------------")

    # 3. Extract <think> reasoning trace if produced by DeepSeek-R1 / Qwen models
    combined = "".join(full_text)
    if isinstance(provider, LMStudioProvider):
        clean_content, thought = LMStudioProvider.extract_thinking_trace(combined)
        if thought:
            print(f"\n[Extracted Thought]\n{thought}")
            print(f"\n[Final Answer]\n{clean_content}")

    await provider.close()


if __name__ == "__main__":
    asyncio.run(main())