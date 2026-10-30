---
title: "Quickstart: library"
sidebar_label: "Quickstart: library"
description: "Build an agent with a tool in Python, run it, and read the events it produces"
---

# Quickstart: library

This guide builds a weather agent with a single tool, in Python, and runs it.

## 1. Install the library

In a Python project, add Domyn Agents as a dependency:

```bash
uv add domyn
```

## 2. Write the agent

Save this as `weather.py`:

```python
import asyncio
import os

from domyn.agents import Agent
from domyn.agents import BaseEvent, ExecutionEventType, Part
from domyn.agents import OpenAIProvider
from domyn.agents import Runner
from domyn.agents import FunctionTool


async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny, 22°C"


agent = Agent(
    name="WeatherAgent",
    description="Reports the weather",
    instruction="You are a weather assistant. Use the weather tool to answer.",
    llm_provider=OpenAIProvider(model_name="gpt-4o", api_key=os.environ["OPENAI_API_KEY"]),
    tools=[FunctionTool(func=get_weather)],
)


async def main() -> None:
    user_input = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="What's the weather in Rome?")],
    )
    async for event in Runner().run(
        application_name="weather-app",
        user_id="user-1",
        conversation_id="conv-1",
        root=agent,
        user_input=user_input,
    ):
        if event.event_type == ExecutionEventType.RESPONSE:
            print("".join(part.text or "" for part in event.content or []))


if __name__ == "__main__":
    asyncio.run(main())
```

What each part does:

- **The tool.** Any Python function can be a tool. Its docstring is the description the model reads to decide when to call it, and its type annotations become the schema of its arguments, so write the first clearly and always add the second.
- **The provider.** `OpenAIProvider` connects the agent to an OpenAI model. Other providers work the same way. See [LLM providers](../guides/llm-providers.md).
- **The agent.** `name` identifies it, `description` says what it does (other agents read it when they decide whether to hand work to this one), `instruction` is its system prompt, and `tools` lists what it may call.
- **The runner.** `Runner().run(...)` executes the agent and yields an **event** for every step: each model call, tool call and response. The loop prints only the final answer, the `RESPONSE` event.

## 3. Run it

From Python, for a single question:

```bash
python weather.py
```

Or, with the [CLI installed](quickstart-cli.md), chat with it from the terminal:

```bash
domyn run weather.py
```

`domyn run` finds the agent defined in the file. The session has only the tools you gave the agent, not the file and command tools of the built-in agent.

## Next steps

- [Tools](../guides/tools.md): streaming tools, MCP servers and tool hooks.
- [Orchestration](../guides/orchestration.md): agents that hand off work to each other.
- [YAML configuration](../guides/configuration.md): the same agent defined in a YAML file.
- [Events and streaming](../guides/events-and-streaming.md): every event the runner yields.
