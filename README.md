# PELDRUN Core

peldrun-core is the native, modular autonomous agent runtime designed for high-throughput reasoning, structured event streaming, and local/cloud LLM orchestration.

## Key Features

- Native Event Streaming: Built-in asynchronous event emitter compatible with SSE and WebSockets.
- Provider Agnostic: Out-of-the-box support for Local LLMs (LM Studio, Ollama) and cloud providers.
- Decoupled Architecture: Clean execution boundaries requiring zero monkey-patching for external web dashboards.
- Extensible Tool Contracts: Type-safe tool schema validation powered by Pydantic.

## Project Structure

    peldrun-core/
      peldrun/
        engine/
        agents/
        tools/
        llm/
      tests/
      pyproject.toml
      LICENSE
      README.md

## Development Setup

Install in editable development mode:

    pip install -e ".[dev]"

Run the test suite:

    pytest

## License

Licensed under the Apache License, Version 2.0. See LICENSE for details.
