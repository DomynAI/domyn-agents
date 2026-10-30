# YAML and MCP, on vLLM

This example runs on a model you serve yourself with vLLM, or on any OpenAI-compatible
endpoint that supports native tool calling. Set `VLLM_URL`, `VLLM_MODEL` and `VLLM_API_KEY`
in `.env` (see the [setup](../README.md#setup)). The other examples run on OpenRouter.

It shows three things:

- how to write the configuration of an agentic system in YAML, without any Python code;
- how to give an agent the built-in file tools;
- how to add the tools of a remote MCP server.

The configuration defines a single agent, a code assistant. It can read and write the files
in the folder it runs in, and it can answer questions about public GitHub repositories.

## Features

- **YAML configuration.** `config.yaml` declares the model, the tools and the agent, and
  `domyn run` loads it directly.
- **Environment variables.** `${VLLM_API_KEY}` and the other `${...}` values are read from
  the environment when the file is loaded, so no secret is written in the file.
- **Built-in file tools.** `tool_bundles` gives the agent the `files` bundle, which can only
  reach files inside the folder the agent runs in. Reading files runs freely, while writing
  them asks you first.
- **MCP server.** When a server is listed under `mcp_servers`, the agent gets every tool
  that server offers. This example uses [DeepWiki](https://deepwiki.com), a public server
  that answers questions about public GitHub repositories. The comments in `config.yaml`
  show how to declare a server that needs OAuth or a token instead.

## Files

- `config.yaml`: the configuration of the system, made of the model, the file tools, the
  MCP server and the agent that uses them.

## Run it

Copy `.env.example` to `.env` in this folder and fill it in (see [Setup](../README.md#setup)).

```bash
cd examples/yaml-and-mcp-vllm
uv run domyn run config.yaml
```

## Try it

- "Read config.yaml and tell me what you can do": the agent reads the file without asking
  you.
- "How does the httpx repository implement retries?": the agent asks DeepWiki about the
  `encode/httpx` repository.
- "Write a short summary of both answers to notes.md": the agent asks you before it writes
  the file.

## From your own code

You do not have to use `domyn run` to run a YAML configuration. You can also load it in your
own application and hand the agent to a `Runner`, as the
[`runner-in-your-app`](../runner-in-your-app/) example does:

```python
from domyn.agents import load_application_from_file

agent = await load_application_from_file("config.yaml")
```

## Learn more

- [Configuration](../../docs/docs/guides/configuration.md)
- [YAML schema](../../docs/docs/reference/yaml-schema.md)
- [Tools](../../docs/docs/guides/tools.md)
