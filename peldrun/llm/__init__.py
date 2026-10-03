"""
PELDRUN Core LLM Subsystem.
Exports unified client primitives, token counting utilities, provider adapters, and dynamic provider factory.
"""

from peldrun.llm.client import (
    AsyncLLMClient,
    DeltaToolCall,
    LLMConfig,
    LLMResponse,
    StreamChunk,
    accumulate_stream_chunks,
)
from peldrun.llm.providers import (
    BaseLLMProvider,
    LMStudioProvider,
    OllamaProvider,
    OpenAICompatProvider,
    get_provider,
)
from peldrun.llm.tokenizer import (
    ContextBudgetManager,
    count_message_tokens,
    estimate_tokens_from_string,
    truncate_messages_sliding_window,
)

__all__ = [
    # Client primitives
    "AsyncLLMClient",
    "LLMConfig",
    "LLMResponse",
    "StreamChunk",
    "DeltaToolCall",
    "accumulate_stream_chunks",
    # Tokenizer & context budgeting
    "ContextBudgetManager",
    "count_message_tokens",
    "estimate_tokens_from_string",
    "truncate_messages_sliding_window",
    # Providers & factory
    "BaseLLMProvider",
    "OpenAICompatProvider",
    "LMStudioProvider",
    "OllamaProvider",
    "get_provider",
]