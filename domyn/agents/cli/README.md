# Domyn CLI

The `domyn` CLI is the main interface for creating, running, and serving agents built with the Domyn Agents framework.

## Installation

The CLI is included when you install `domyn`. The `domyn` entry point is available after installation:

```bash
uv add domyn
```

## Global Options

```
domyn [OPTIONS] COMMAND
```

| Option | Default | Description |
|--------|---------|-------------|
| `--log-level TEXT` | `WARNING` for interactive commands, `INFO` for servers | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |
| `--version` | — | Print installed version and exit |

---

## Commands

### `domyn init`

Scaffold a new Domyn Agents project using `uv`.

```bash
domyn init [PROJECT_NAME] [OPTIONS]
```

| Argument / Option | Default | Description |
|-------------------|---------|-------------|
| `PROJECT_NAME` | prompted | Name of the new project |
| `--directory / -d TEXT` | `./<project_name>` | Target directory |
| `--force / -f` | `false` | Overwrite an existing directory |

**What it creates:**

```
<project>/
├── agent.py          # Pre-wired Agent with example tools
├── pyproject.toml    # uv project with domyn[cli] dependency
├── Dockerfile        # Ready-to-build production image
├── README.md         # Project-level instructions
└── .gitignore
```

**Example:**

```bash
domyn init my-assistant
cd my-assistant
# Edit agent.py to configure your LLM provider and tools
uv run domyn run agent.py "Hello!"
```

---

### `domyn run`

Chat with an agent, interactively or with a single prompt.

```bash
domyn run [FILE] [PROMPT] [OPTIONS]
```

| Argument / Option | Default | Description |
|-------------------|---------|-------------|
| `FILE` | saved configuration | Path to a `.py` module or `.yaml` config file. Omitted: the agent from `~/.domyn/config.yaml` (the setup wizard runs first if there is none) |
| `PROMPT` | — | Send one message and exit. Omitted: open an interactive session |
| `--agent-name / -a TEXT` | auto-detected | Variable name of the agent to run |

**Agent auto-detection** (`.py` files):

1. Exact match on `--agent-name`
2. Root agent — the one that is not a `sub_agent` of any other agent
3. Name-based fallback (`main`, `service`, `customer`)
4. First `Agent` instance found

**Terminal output:**

- Streaming tokens printed as they arrive
- Tool calls shown in yellow (`→ tool_name {params}`)
- Tool results shown in dim (`← observation`)
- Agent handoffs shown in magenta (`↪ / ↩`)

In an interactive session the CLI prompts for the next input after each response. Type `/exit` (`/quit`, `/q`) or press Ctrl-D to end the session.

**Examples:**

```bash
# Interactive session with the configured agent
domyn run

# Run from a Python file (auto-detects root agent)
domyn run agent.py "What is the weather in Milan?"

# Run a specific agent by variable name
domyn run agent.py "Help me" --agent-name customer_support_agent

# Run from a YAML config
domyn run config.yaml "Summarise the last report"
```

---

### `domyn serve`

Expose an agent as a local HTTP server (REST by default).

```bash
domyn serve FILE [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--channel / -c TEXT` | `rest` | Transport channel (`rest`, `relay` or `multi`) |
| `--host TEXT` | `0.0.0.0` | Bind address |
| `--port INTEGER` | `8000` | Bind port |
| `--agent-name TEXT` | auto-detected | Agent to serve |

**REST API endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check — returns `{"status": "healthy", "agent": "<name>"}` |
| `GET` | `/agents` | List all agents with name and description |
| `GET` | `/agents/{name}` | Agent detail — name, description, tool list |
| `POST` | `/run` | Run agent synchronously, returns final response text |
| `POST` | `/execute` | Run agent, stream all events via Server-Sent Events (SSE) |

**`POST /run` request / response:**

```jsonc
// Request
{ "text_input": "Hello", "conversation_id": "conv_abc123", "user_id": "user_1" }

// Response
{ "response": "Hi there!", "conversation_id": "conv_abc123", "agent_name": "MyAgent" }
```

**`POST /execute` SSE stream:**

Each line is `data: <BaseEvent JSON>`. The stream ends with `event: end\ndata: {}`.

**Examples:**

```bash
domyn serve agent.py
domyn serve agent.py --port 9000 --host 127.0.0.1
domyn serve config.yaml --channel rest

# Query the running server
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"text_input": "Hello!"}'
```

---

## More Commands

Run `domyn --help` for the full list of commands, and `domyn COMMAND --help` for the options of each.

The full command reference is on the documentation site:
<https://domynai.github.io/domyn-agents/reference/cli>.
