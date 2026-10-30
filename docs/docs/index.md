---
title: Introduction
sidebar_label: Introduction
slug: /
description: "Domyn Agents: a Python library for building agents and teams of agents, and a terminal assistant built with it"
---

# Domyn Agents

Domyn Agents is a Python library for building AI agents: agents that call tools, hand work to each other, and work with any LLM provider. It also ships a command-line assistant built with the library, so you can work with an agent in your terminal without writing any code.

## Two ways to use it

**From the terminal.** `domyn run` opens a chat with an agent that reads and edits the files of a directory and runs commands in it, asking before it changes anything. You configure it with a setup wizard and extend it with skills and standing instructions. Start with the [CLI quickstart](getting-started/quickstart-cli.md).

**As a library.** Build your own agents in Python or YAML, give them your tools, put them in teams, and run them from your application. Start with the [library quickstart](getting-started/quickstart-library.md).

The two meet: an agent you build can be opened in the terminal with `domyn run agent.py`, and the terminal assistant is an ordinary agent configuration you can change.

## Core concepts

**Agents.** An agent has a name, a description, an instruction (its system prompt), an LLM provider and tools. Agents can form a team: a main agent hands work to the members whose description fits, and continues with what they return. See [Orchestration](guides/orchestration.md).

**Tools.** Tools are how an agent acts. Any Python function can be one; tools can also stream their output, or come from a remote MCP server. See [Tools](guides/tools.md).

**LLM providers.** A provider connects an agent to a model: the models of your Domyn platform space, OpenAI, Anthropic, vLLM, Cerebras, OpenRouter or any OpenAI-compatible endpoint, or one you write. See [LLM providers](guides/llm-providers.md).

**Planner strategies.** A planner runs an agent's turn as a loop: reason, call a tool or another agent, read the result, repeat until it can answer. See [Planner strategies](guides/planner-strategies.md).

**Runner and events.** The `Runner` executes an agent and yields an event for every step: each model call, tool call, hand-off and reply. An application shows an agent to its users by reading these events. See [Events and streaming](guides/events-and-streaming.md).

## Where to go next

| You want to | Read |
|---|---|
| Chat with an agent in your terminal | [Quickstart: CLI](getting-started/quickstart-cli.md), then [Sessions and commands](cli/sessions.md) |
| Build an agent | [Quickstart: library](getting-started/quickstart-library.md), then [Tools](guides/tools.md) |
| Build a team of agents | [Orchestration](guides/orchestration.md) |
| Define agents without Python | [YAML configuration](guides/configuration.md) |
| Give an agent knowledge it loads when needed | [Skills](guides/skills.md) |
| Put an agent in your application | [Events and streaming](guides/events-and-streaming.md) |
