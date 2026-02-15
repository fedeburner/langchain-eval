# Spec: Barebones CLI Chat Agent with LangChain Deep Agents

## Context

Starting a brand new repo from scratch. The goal is a minimal, extensible CLI chat interface built on LangChain's `deepagents` library (v0.4.1) using `uv` for package management. It should work with all major LLM providers (OpenAI, Anthropic, Google) out of the box via `init_chat_model`. Includes a simple eval harness to test the agent in isolation.

## Project Structure

```
take-home/
├── pyproject.toml          # uv project config, dependencies
├── .python-version         # pin Python 3.13
├── .env.example            # template for API keys
├── .gitignore              # Python + env ignores
├── README.md               # human-readable setup & usage guide
├── src/
│   └── agent/
│       ├── __init__.py
│       ├── core.py         # create_deep_agent wrapper, model init
│       └── cli.py          # interactive chat loop (entry point)
└── evals/
    ├── __init__.py
    └── test_agent.py       # barebones eval: invoke agent, assert on output
```

## Step 1: Initialize uv project & dependencies

- Run `uv init` in the repo root
- Set Python 3.13 via `.python-version`
- Configure `pyproject.toml` with:
  - **Core deps**: `deepagents`, `langchain`, `langchain-openai`, `langchain-anthropic`, `langchain-google-genai`, `python-dotenv`
  - **Dev deps**: `pytest`, `pytest-asyncio`
  - **CLI entry point**: `[project.scripts] chat = "agent.cli:main"`
- Run `uv sync` to install everything

## Step 2: Create `.env.example` and `.gitignore`

**.env.example** — template showing all supported API keys:
```
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
```

**.gitignore** — standard Python ignores + `.env`, `.venv`, `__pycache__`, etc.

## Step 3: Implement `src/agent/core.py` — Agent factory

```python
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

def make_agent(model_str: str = "anthropic:claude-sonnet-4-5-20250929", system_prompt: str | None = None):
    """Create a deep agent with the specified model provider."""
    model = init_chat_model(model_str)
    kwargs = {}
    if system_prompt:
        kwargs["system_prompt"] = system_prompt
    return create_deep_agent(model=model, **kwargs)
```

Key points:
- Uses `init_chat_model("provider:model")` for universal provider support
- Default model is Anthropic Claude Sonnet but trivially swappable via CLI flag
- Returns a compiled LangGraph graph supporting `.invoke()`, `.stream()`, `.astream()`

## Step 4: Implement `src/agent/cli.py` — Interactive chat loop

```python
import argparse, sys
from dotenv import load_dotenv
from agent.core import make_agent

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="CLI Chat Agent")
    parser.add_argument("--model", default="anthropic:claude-sonnet-4-5-20250929",
                        help="Model string, e.g. openai:gpt-4o, anthropic:claude-sonnet-4-5-20250929, google_genai:gemini-2.5-flash")
    parser.add_argument("--system", default=None, help="Custom system prompt")
    args = parser.parse_args()

    agent = make_agent(args.model, args.system)
    messages = []
    print("Chat started. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        messages.append({"role": "user", "content": user_input})
        result = agent.invoke({"messages": messages})
        ai_msg = result["messages"][-1]
        print(f"\nAssistant: {ai_msg.content}\n")
        messages = result["messages"]
```

Key points:
- Uses `argparse` for `--model` and `--system` flags
- Loads `.env` for API keys via `python-dotenv`
- Simple REPL loop with message history passed back each turn
- Uses `invoke()` (keeping it simple; streaming can be added later)

## Step 5: Implement `evals/test_agent.py` — Barebones eval

```python
import pytest
from dotenv import load_dotenv
from agent.core import make_agent

load_dotenv()

@pytest.fixture
def agent():
    return make_agent()

def test_agent_responds(agent):
    """Agent should return a non-empty response to a simple question."""
    result = agent.invoke({"messages": [{"role": "user", "content": "What is 2 + 2?"}]})
    assert len(result["messages"]) > 1
    ai_msg = result["messages"][-1]
    assert ai_msg.content  # non-empty
    assert "4" in ai_msg.content

def test_agent_multi_turn(agent):
    """Agent should handle multi-turn conversation."""
    r1 = agent.invoke({"messages": [{"role": "user", "content": "My name is Alice."}]})
    msgs = r1["messages"]
    msgs.append({"role": "user", "content": "What is my name?"})
    r2 = agent.invoke({"messages": msgs})
    ai_msg = r2["messages"][-1]
    assert "Alice" in ai_msg.content
```

Key points:
- Real LLM calls (not mocked) — tests actual provider integration
- Two tests: basic response check and multi-turn memory
- Run via `uv run pytest evals/ -v`

## Step 6: Write `README.md`

Human-readable docs covering:
- What this project is (one paragraph)
- Prerequisites (Python 3.13+, uv, at least one API key)
- Quick start (clone, `uv sync`, copy `.env.example` to `.env`, fill keys)
- Usage: `uv run chat --model openai:gpt-4o`
- Running evals: `uv run pytest evals/ -v`
- Supported providers table (OpenAI, Anthropic, Google + model string examples)
- Project structure overview

## Verification

1. `uv sync` — installs all deps successfully
2. `uv run chat` — starts interactive CLI, can send messages and get responses
3. `uv run chat --model openai:gpt-4o` — works with different providers
4. `uv run pytest evals/ -v` — both eval tests pass

## Sources

- [deepagents on PyPI](https://pypi.org/project/deepagents/) — v0.4.1, MIT licensed
- [langchain-ai/deepagents on GitHub](https://github.com/langchain-ai/deepagents)
- [Deep Agents overview docs](https://docs.langchain.com/oss/python/deepagents/overview)
- [LangChain init_chat_model](https://docs.langchain.com/oss/python/langchain/models) — universal provider format `"provider:model"`
- [LangChain eval patterns for Deep Agents](https://blog.langchain.com/evaluating-deep-agents-our-learnings/)
