"""
PELDRUN Core Memory Subsystem Test Suite.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from peldrun.memory import (
    LongTermMemory,
    MemoryManager,
    ShortTermMemory,
    ShortTermMemoryConfig,
)


def test_short_term_memory_messages_and_pruning() -> None:
    """تحقق من إضافة الرسائل، الحفاظ على رسالة النظام، وتقليم النافذة المنزلقة.

    ملاحظة: ShortTermMemoryConfig يفرض max_messages >= 5.
    """
    config = ShortTermMemoryConfig(max_messages=5, auto_prune=True)
    memory = ShortTermMemory(config=config, system_prompt="Base system directive.")

    assert len(memory) == 1
    sys_msg = memory.get_system_message()
    assert sys_msg is not None
    assert sys_msg["content"] == "Base system directive."

    # املأ إلى السعة بالضبط: system + 4 turns = 5
    memory.add_message("user", "Hello turn 1")
    memory.add_message("assistant", "Response turn 1")
    memory.add_message("user", "Hello turn 2")
    memory.add_message("assistant", "Response turn 2")
    assert len(memory) == 5

    # إضافة رسالة سادسة يجب أن تـ prune الأقدم
    memory.add_message("user", "Hello turn 3")
    assert len(memory) == 5

    assert memory.messages[0]["role"] == "system"
    assert memory.messages[0]["content"] == "Base system directive."
    assert memory.messages[1]["content"] == "Response turn 1"
    assert memory.messages[-1]["content"] == "Hello turn 3"


def test_short_term_memory_summarization_and_snapshot() -> None:
    memory = ShortTermMemory(system_prompt="Initial prompt")
    memory.add_message("user", "Detail 1")
    memory.add_message("assistant", "Detail 2")
    memory.add_message("user", "Detail 3")
    memory.add_message("assistant", "Detail 4")

    memory.summarize_old_messages(summary_text="Summarized details 1 and 2.", keep_last_n=2)

    messages = memory.messages
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "system"
    assert "Summarized details 1 and 2." in messages[1]["content"]
    assert messages[-2]["content"] == "Detail 3"
    assert messages[-1]["content"] == "Detail 4"

    snapshot = memory.export_snapshot()
    assert snapshot["token_count"] > 0
    assert len(snapshot["messages"]) == len(messages)

    new_memory = ShortTermMemory()
    new_memory.load_snapshot(snapshot)
    assert len(new_memory) == len(memory)
    assert new_memory.messages[1]["content"] == messages[1]["content"]


@pytest.mark.asyncio
async def test_long_term_memory_persistence_and_search(temp_workspace: Path) -> None:
    memory = LongTermMemory(workspace_root=str(temp_workspace))
    await memory.ainitialize()

    mem1 = await memory.aadd_memory(
        content="PostgreSQL database runs on port 5432.",
        category="architecture",
        tags=["database", "postgres", "ports"],
    )
    mem2 = await memory.aadd_memory(
        content="Always use tabs for indentation in python files.",
        category="preference",
        tags=["python", "style"],
    )
    await memory.aadd_memory(
        content="FastAPI backend runs on port 8088.",
        category="architecture",
        tags=["api", "fastapi", "ports"],
    )

    assert len(memory) == 3
    assert memory.file_path is not None
    assert memory.file_path.exists()

    search_results = await memory.asearch(query="postgres database port", limit=2)
    assert len(search_results) > 0
    assert search_results[0].id == mem1.id

    pref_results = await memory.asearch(query="style", category="preference")
    assert len(pref_results) == 1
    assert pref_results[0].id == mem2.id

    prompt_context = await memory.format_context_for_prompt(query="port", limit=2)
    assert "Long-Term Memory & Project Knowledge" in prompt_context
    assert "port" in prompt_context.lower()

    assert await memory.adelete_memory(mem2.id) is True
    assert len(memory) == 2


@pytest.mark.asyncio
async def test_memory_manager_coordination(temp_workspace: Path) -> None:
    manager = MemoryManager(
        workspace_root=str(temp_workspace),
        system_prompt="You are PELDRUN Core autonomous agent.",
    )
    await manager.ainitialize()

    await manager.long_term.aadd_memory(
        content="Redis cache is disabled for testing environments.",
        category="fact",
        tags=["redis", "cache"],
    )

    compiled_prompt = await manager.compile_system_prompt(query="How to configure redis cache?")
    assert "You are PELDRUN Core autonomous agent." in compiled_prompt
    assert "Redis cache is disabled for testing environments." in compiled_prompt

    head_sys_msg = manager.short_term.get_system_message()
    assert head_sys_msg is not None
    assert "Redis cache is disabled" in head_sys_msg["content"]

    snapshot = manager.export_snapshot()
    assert snapshot["workspace_root"] == str(temp_workspace)
    assert snapshot["long_term_count"] == 1

    restored = MemoryManager()
    restored.load_snapshot(snapshot)
    assert restored.workspace_root == str(temp_workspace)
    assert "PELDRUN Core" in restored._raw_system_prompt