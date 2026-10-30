# Examples

Each folder here is a self-contained example. It holds the configuration of an agentic
system (`config.py` or `config.yaml`), any other files that system needs, and a README. The
README says which features the example shows and where in the files to find them, how to run
it, and what to ask so that you can see each feature at work.

The examples are listed from the simplest to the most involved:

| Example | Features it shows |
|---|---|
| [`tools-and-approvals`](tools-and-approvals/) | Python functions as tools, tools that wait for your approval, data kept in the conversation state |
| [`yaml-and-mcp-vllm`](yaml-and-mcp-vllm/) | A configuration written in YAML only, the built-in file tools, the tools of an MCP server, on a model you serve with vLLM |
| [`handoffs-and-hooks`](handoffs-and-hooks/) | Several agents handing work to one another, what each one sees and whom it can hand off to, hooks that run your code around tools |
| [`skills-and-soul`](skills-and-soul/) | Skills (know-how an agent reads only when a task needs it), a `SOUL.md` that sets the tone of every reply |
| [`rest-guardrail`](rest-guardrail/) | Hooks served over HTTP, used as a guardrail on what reaches the model and what reaches you |
| [`runner-in-your-app`](runner-in-your-app/) | Running a system from your own code with the `Runner`, approvals answered in code, saved conversations, long-term memory |
| [`remote-agent`](remote-agent/) | An agent that runs in another process, called as a sub-agent through a relay |

## Setup

From the repository root, install the package:

```bash
uv sync --extra cli
```

Each example folder has its own `.env.example`. In the folder of the example you want to run,
copy it and fill it in:

```bash
cd examples/tools-and-approvals
cp .env.example .env    # then fill it in
```

The examples run on [OpenRouter](https://openrouter.ai) with
`nvidia/nemotron-3-ultra-550b-a55b:free`. Set your key in `.env`:

| Variable | Value |
|---|---|
| `OPENROUTER_API_KEY` | your OpenRouter API key |

[`yaml-and-mcp-vllm`](yaml-and-mcp-vllm/) runs instead on a model you serve yourself, with
vLLM or any OpenAI-compatible endpoint that supports native tool calling. It reads three more
variables:

| Variable | Value |
|---|---|
| `VLLM_URL` | the full chat completions URL, e.g. `http://localhost:8000/v1/chat/completions` |
| `VLLM_MODEL` | the model the endpoint serves |
| `VLLM_API_KEY` | the API key, or empty if the endpoint needs none |

Run each example from its own folder, because the paths in its files are relative to that
folder. When you run an example with `domyn`, it finds the `.env` in that folder on its own.
When you run a script with Python, pass the file explicitly with `--env-file .env`.

## Adding an example

Give the new example its own folder, with its configuration in `config.py` or
`config.yaml` and a README laid out like the others. `test_examples.py` picks up the new
folder automatically. Without a model, it checks that the configuration builds, that its
skills and instructions load, and that its Python files import.

## Going further

The [documentation](https://domynai.github.io/domyn-agents/) covers each topic in depth:
[tools](../docs/docs/guides/tools.md), [approvals](../docs/docs/guides/approvals.md),
[orchestration](../docs/docs/guides/orchestration.md), [hooks](../docs/docs/guides/hooks.md),
[skills](../docs/docs/guides/skills.md),
[standing instructions](../docs/docs/guides/standing-instructions.md),
[events and streaming](../docs/docs/guides/events-and-streaming.md),
[conversations and memory](../docs/docs/guides/conversations-and-memory.md),
[remote agents](../docs/docs/guides/remote-agents.md) and the
[YAML schema](../docs/docs/reference/yaml-schema.md).
