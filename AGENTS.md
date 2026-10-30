# AGENTS.md

Guidance for coding agents (and the people running them) working in this repository. Contributors using an agent must have it follow this file: see [CONTRIBUTING.md](CONTRIBUTING.md). Claude Code reads `CLAUDE.md` rather than this file: a local one containing `@AGENTS.md` imports it (`CLAUDE.md` is git-ignored).

## Project Overview

Domyn Agents is a modern Python agentic framework built on the ReAct (Reasoning and Acting) pattern. It provides a flexible, registry-based IoC (Inversion of Control) system for building AI agents with tool execution capabilities, agent handoffs, and pluggable LLM providers.

## Development Setup

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync

# Install with dev dependencies
uv sync --dev

# Setup development environment (installs dev dependencies)
make dev-setup
```

Every `domyn` invocation loads a `.env` file from the working directory (or a parent of it) before doing anything else (`_load_dotenv()`, `domyn/agents/cli/main.py`) — process-exported env vars always win over a same-named `.env` entry. This is what backs the `${VAR_NAME}` interpolation in standalone YAML configs (see YAML Configuration below) and `api_key_env` in `~/.domyn/config.yaml`: secrets can live in a git-ignored `.env` instead of the shell profile.

## Common Commands

### Testing
```bash
# Whole suite with coverage (~50s)
make test

# Whole suite, no coverage — what you want while iterating. Runs across cores
# by default (`addopts = "-n auto"` in pyproject): 26s instead of 190s, because
# a third of the wall time is tests sleeping while real background processes
# work. `-n0` turns it off, for a debugger or to read output in order.
uv run pytest tests/ examples/ -q --no-cov

# One file, one class, one test
uv run pytest tests/cli/test_repl_session.py -q --no-cov
uv run pytest tests/planner_strategy/test_compaction.py::TestTheDefaultStrategy -q --no-cov
uv run pytest "tests/cli/test_context_view.py::TestTheIndicatorStaysQuiet::test_before_any_request" -q --no-cov

# By name across the suite
uv run pytest tests/ -q --no-cov -k "compaction and not cli"

# Stop at the first failure, with the full assertion diff
uv run pytest tests/ -q --no-cov -x
```

`tests/` mirrors the package layout, and `pytest-asyncio` is used with an explicit
`@pytest.mark.asyncio` on every async test.

Three things to know before adding tests:

- **`tests/cli/conftest.py` has an autouse fixture that redirects `~/.domyn`.** Building a `Repl` without an explicit runner wires the disk-backed approval store — that is the product behaviour — so without it a test run would read, and on approval *write*, the real user's standing permissions. Do not remove it, and do not build a `Repl` in a test outside `tests/cli/` without the same protection.
- **Background commands a test starts are stopped after it** (`tests/conftest.py`). They outlive the call that started them by design, for up to a day, and a run of `tests/tools` used to leave about two hundred behind, slowing the timing tests of the next run. The fixture kills, after every test, the group of each command still running in a workspace up to two levels under the session's temporary directory, and the session end sweeps the rest; a workspace deeper than that is caught only at the end.
- **A `MagicMock` provider is a trap for anything reading numbers off it.** `int(MagicMock())` is `1`, so a mocked `context_window` silently becomes a 1-token window and every context policy fires. Two tests were broken this way; the fix belonged in the code (validate the value), not in the tests.

### Code Quality
```bash
# Run all checks (lint, format, type checking, tests)
make check-all

# Lint code
make lint

# Lint and auto-fix issues
make lint-fix

# Format code
make format

# Check formatting without changes
make format-check

# Type checking with mypy
make mypy
```

### Running Agents

```bash
# Interactive session with the configured default agent — the main path.
# Runs the setup wizard first if ~/.domyn/config.yaml does not exist yet.
domyn run

# One prompt, no session
domyn run "<prompt>"

# An agent from a file, interactively or with one prompt
domyn run <file.py>
domyn run <file.py> "<prompt>"
domyn run <file.py> "<prompt>" --agent-name <agent_variable_name>

# Continue a past conversation: by id, or pick one from a list
domyn run --resume <conversation_id>
domyn run --resume
```

Conversations are persisted to SQLite beside the config (`conversations.db`), so
an id printed on the way out of a session is an address the next one can open.
`/resume` does the same from inside a session. The id is the *only* handle, which
is why it is printed on exit.

With a file, `domyn run` finds the root agent in the dependency tree (the one that is not a `sub_agent` of another). With no file it builds the agent described by `~/.domyn/config.yaml` — see **The interactive CLI** below.

### CLI Commands

Verify against `domyn --help` before relying on this list.

| command | what it does |
|---|---|
| `domyn run` | Chat with an agent, interactively or with a single prompt |
| `domyn setup` | Configure provider, model and default agent (also runs automatically on first `run`); `--project <name>` creates or edits `~/.domyn/projects/<name>`; `--import <dir>` creates that project from an export (`~/.domyn/projects/<the folder's name>` without `--project`) |
| `domyn export <dir>` | Copy the configuration, its skills and instruction files into a new folder laid out as a project |
| `domyn init` | Initialize a new project |
| `domyn skills` | List available skills |
| `domyn skill show\|remove <name>` | Show a configured skill (`<name>#<section>` for one section), or unlist it from the config |
| `domyn serve` | Serve an agent from a file over a chosen protocol (hidden from `--help`) |
| `domyn connect` | Connect a runnable to the Domyn platform as a relay worker (`domyn expose` still works, hidden) |

## Architecture

### Core Concepts

**Registry System**: The framework uses a centralized registry pattern for managing components:
- `AgentRegistry` - Manages agent instances
- `ToolRegistry` - Manages tool definitions
- `LLMProviderRegistry` - Manages LLM provider configurations
- `PlannerStrategyRegistry` - Manages planning strategies
- `HookRegistry` - Manages lifecycle hooks, agent and tool alike

Registries are accessed via factory functions: `get_agent_registry()`, `get_tool_registry()`, `get_llm_registry()`, etc.

**Agent Types**:
- `Agent` - Main agent class with support for tools, sub-agents, and planning strategies
- `WrappedAgent` - Wraps arbitrary Python classes/functions as agents
- `BaseAgent` - Interface for all agent types

**Visibility — two orthogonal axes on the agent** (`domyn/agents/agents/base.py`):
- `ContextVisibility` (READ scope: what an agent sees of context/history) - `GLOBAL` / `PEERS` / `STANDARD`
- `HandoffScope` (EXECUTE scope: which agents it may hand off to) - `EXPLICIT` / `DIRECT_PEERS` / `GLOBAL`

The legacy single `handoff_visibility` config key is still accepted and mapped onto these two by `normalize_legacy_visibility()`, with a deprecation warning.

**Separate, unrelated axis** — `VisibilityScope` (`domyn/agents/core/visibility.py`): `ISOLATED` / `SCOPED` / `TRUSTED` / `GLOBAL`. It governs trust/PII propagation (`run_config` access) toward MCP tools and delegate agents, *not* handoffs or context.

**Planner Strategies** (`domyn/agents/planner_strategy/`, registered in `PlannerStrategyFactory`):
- `ReactPlannerStrategy` (`react.py`) - ReAct pattern; the tool protocol is carried in the prompt text, so it is the one for an endpoint without native tool calls. Supports `use_stop`. Its name is enough: `ReactPlannerStrategy()`, `{"type": "react"}` or a YAML entry without `context_manager` get full messages in structural tags (`_default_context_manager`).
- `ToolUsePlannerStrategy` (`tool_use.py`) - native tool-calling APIs (`tool_calls` on the wire) instead of a prompt protocol.
- `ToolUseWithThoughtPlannerStrategy` (`tool_use_thought.py`) - as above, plus a reasoning field surfaced before each call. **The default**: an agent that names no planner gets it, built in code (`Agent(...)`) or from a config without `planner_strategy` (`_default_planner_factory`, `agents/agent.py`). `react` has to be asked for by name.

The interactive CLI is **fixed** to `tool_use_thought` and offers no way to change it — see `CLI_PLANNER` (`interfaces/cli/config.py`).

**Two layers, often confused** — a planner composes both:

1. **Context management** (`planner_strategy/context_management/`) decides *which* history becomes messages. `ContextManager.get_context(context, agent, ...) -> list[LLMMessage]` is the single seam every planner goes through, so it is also where any context-window policy belongs.

   | `type` | Class | Behaviour |
   |---|---|---|
   | `full_messages` | `FullMessageContext` | whole history verbatim |
   | `tool_use` | `ToolUseContextManager` | native tool-call format; caps each tool result at `max_tool_result_chars` (20 000), keeping its start and its end, with the `read_tool_result` call that reads the rest |
   | `thought_tool_use` | `ToolUseWithThoughtContextManager` | as above, with the thought field — **the CLI default** |
   | `agents_summary` | `PreviousAgentsSummaryContext` | summarizes past interactions once there are more than `min_events_for_summary` (5) |
   | `old_summary` | `OldEventsSummaryContext` | keeps the last `message_history_length` (8) events, summarizes what precedes |
   | `interactions_summary` | `PreviousInteractionsSummaryContext` | summarizes each agent-execution block between user turns |

   The three summarizing managers derive from `EventsSummarizer` (`context_management/summary.py`), which calls an LLM and caches by event-id tuple under `context.state["summaries"]`. **They trigger on event counts, not on token pressure** — nothing in the package counts tokens or knows a model's context window, so a long session fails on a rejected request rather than degrading. They also sit only on the `react` branch: the `tool_use*` managers the CLI uses do not summarize at all.

   **Compaction** (`context_management/budget.py`, `context_management/compaction.py`) is the token-pressure answer the summarizing managers above are not. It sits on the `tool_use*` branch, i.e. the one the CLI and the default planner use.

   - `HeuristicEstimator` prices messages by characters (`chars_per_token=3.0`, rounding up). No `tiktoken`: it is OpenAI's tokenizer and would be the wrong segmentation for a vLLM-served model. On its own the estimate is off in both directions — it overcounts prose (0.62–0.77 real tokens per estimated one on DeepSeek-V4-Flash, ~0.56 on Qwen3-32B) and undercounts token-dense text (UUIDs, hashes, hex: 12,000 characters cost 9,421 real tokens on Qwen3-32B, 2.4× the estimate) — and providers disagree among themselves: the same prompt counted 6,630 (OpenAI), 8,561 (Anthropic), 9,704 (Qwen3-32B).
   - So the size of a request is **anchored to the provider's own count of the previous one** (`Compactor.observe`, called by both planner loops after every response): that count plus an estimate of only what was added since, as pi and hermes do. Arithmetically it is the estimate plus a fixed `correction_tokens` (reported − estimated), kept under `context.state["context_anchors"]` per agent. The count is normalized by `reported_prompt_tokens` (`llm/base.py`): `prompt_tokens`, or Anthropic's `input_tokens` **plus its two cache counts** — with a cache in play `input_tokens` alone is the uncached tail (4 tokens of 8,561, measured). The correction belongs to the text it was measured on: a compaction carries the anchor over to the rendering it produced (`_carry_anchor`, under the record's new `generation`) with the share of the correction that belongs to the text kept (`_rescale`: drop part of the text, drop the same share). Kept whole, it bought summaries and reported "still over the limit" for requests the provider then counted under the threshold; dropped whole, a pruning that kept the dense text was judged sufficient and went out 6k over it (hex tool output, Qwen3-VL-32B). Carried, the decision, the result reported, the verbatim tail (`Budget.tail_tokens`, sized at the measured rate), the request that goes out and a recovery from its rejection all read one number, and `/compact` includes that share in `tokens_saved`. The share is still an estimate, and an open one: it undercounts when the text kept is denser than the text dropped — after a summary that kept fresh hex output verbatim, a request put at 15,160 was counted 22,609, about 5.4k of it tool output no count had seen yet; with `max_tokens` 16000 on vLLM before 0.32 such a request is rejected, and goes through the overflow recovery. A record changed any other way retires the anchor; so does a count from another deployment (`deployment_key`: provider class, URL and model — a `/model`, a conversation resumed on another model, a model the gateway replaced), and a count more than 4× off the estimate either way is ignored as counted on another basis. Not the calibration it replaces: that multiplied every estimate by a ratio learned on other text, so prose made the next dense output look small. `/context` shows the correction as its own row.
   - The context window comes from `context_window` (or `max_model_len`) on the provider. When none is declared the runner asks, once per turn and before any request (`_resolve_turn_context_windows` → `resolve_context_window`, `llm/context_window.py`): first the provider's own API (`fetch_context_window`: vLLM's `/v1/models` `max_model_len`, a gateway's `max_input_tokens`, OpenRouter's `top_provider.context_length`, Anthropic's `/v1/models/{id}`), then, for hosted APIs that report nothing (OpenAI), the models.dev catalogue `llm/context_windows.json` by exact id (`context_window_catalog`). The catalogue is a snapshot, never fetched at run time: `make context-windows` (`scripts/update_context_windows.py`) regenerates it, and the file is committed like any change (a release is a tag and commits nothing, so it ships the catalogue main has); a refresh may change any number, so tests must not pin them (they use a catalogue of their own). Answers, "none" included, are cached for 5 minutes per provider class, URL and model (`CACHE_SECONDS`), so a host that builds its providers per request asks once; a question that failed — refused, timed out, unreachable — says nothing about the window and is asked again after a minute (`RETRY_SECONDS`) by a provider that has no window yet (the answer is written on the provider like a declaration, so one kept for a whole session, the CLI's, asks once and keeps what it got, its 150k fallback included, until `/model` or a restart), and a provider that cannot answer stays unknown and never fails the turn. The CLI assumes `DEFAULT_CONTEXT_WINDOW` (150k) only after that (`apply_context_window`) and writes no window for an empty answer (`auto`); any number typed, 150000 included, is a declaration. What is still unknown (behind a gateway or router that does not publish it) has no threshold until a provider **rejects** a request as too long: the window that rejection states (`context_limit_from_error`: vLLM/OpenAI/OpenRouter "maximum context length is N tokens", vLLM's later check "Input length (M) exceeds model's maximum context length (N)" (the only one a gpt-oss prompt meets: vLLM renders Harmony without the first), vLLM's engine "The decoder prompt (length M) … maximum model length of N", Anthropic "… > N maximum", the LiteLLM gateway's pre-call check "Max Input Tokens=N" — the widest, when it lists several deployments) is adopted on the provider, and lowers a declared or reported one that was too large. It is also **remembered** per provider class, URL and model until the process exits (`remember_stated_window`) and caps every provider built for them afterwards, declared, reported or catalogued: a host that builds its providers for every request would otherwise lose it with each instance and see the next long request refused again. Only ever lowered: a server later given a larger window keeps the old one until the process restarts, since nothing is written to disk; after a restart the endpoint is asked again, and a window nobody reports is learned again from the first rejection. Nothing is inferred from a model name on a self-hosted endpoint — the same weights served with a different `--max-model-len` have a different limit — which is why vLLM and custom endpoints have no catalogue.
   - `Compactor.assess()` is synchronous and cheap (threshold = `window − reserve_tokens`, default reserve 16k); `apply()` may call a model. The split exists so the start can be announced before a multi-second pause. It checks the **whole request** — conversation, the turn's injected context (`turn_context_messages`: skills, memory) and tool definitions — and `usage_metadata` reports that same total in `LLM_START` metadata, which is what the CLI's indicator and `/context` show: the number on screen is the number the threshold is checked against. Only the conversation is compacted; the rest is `Budget.fixed_tokens`, and the measured correction goes with the text it was measured on (above). `CompactionStrategy` is the replaceable seam, selected by `compaction.strategy` in `~/.domyn/config.yaml`:

     | name | behaviour |
     |---|---|
     | `two_phase` | **default.** Prunes; if the result is still over the threshold, summarizes the middle with a model call |
     | `prune_only` | deterministic only: replaces old tool-result bodies with a line saying what they were, no model call ever |
     | `noop` | does nothing, and says so rather than announcing a threshold that will not fire |

     `two_phase` is the default because it *is* `prune_only` plus a fallback that runs only once the free phase has demonstrably failed — at which point the alternative is a rejected request. A failed summary falls back to the pruning rather than discarding it.
   - The summarized span is bounded by `_clean_boundary()`: never immediately after a message carrying `tool_calls`, never immediately before a `tool` result. That is the one rule the wire protocol enforces by *rejecting* a request, and placing both ends on it makes the rest structural — the span is whole blocks (a user turn or an assistant message through to a finished reply), so a single long exchange is still summarizable and the roles around the replacement read as before. `compact()` builds its result with the same function `replay()` uses, so a request and every later re-render of it agree by construction. The span also ends before the last request (`CompactionPolicy.keep_last_user_messages`): that, not a tail stretched back to it, is what keeps the request verbatim — the tail is cut by budget alone, so the older tool outputs of one long turn can still be pruned.
   - `CONTEXT_COMPACTED` metadata carries `changed` (something was done) **and** `sufficient` (the result is under the threshold). They are different claims: past a point pruning always finds one fresh tool output to drop while the total keeps climbing, and reporting that as success is how a session dies on a rejected request. `sufficient=False` is also what tells `two_phase` to summarize.
   - `InteractionContext` is **never** rewritten. What persists is a `CompactionRecord` per agent under `context.state["compaction_records"]` (per agent because `context_visibility` gives agents different histories): the dropped event ids, plus one summary and the events it stands for. `Compactor.replay()` re-applies it on every render — without that a compaction below the threshold would be undone by the next re-render, and the summary would be re-bought every turn. Everything is keyed by `source_event_id`, which `context_management/tool_use.py` stamps on **every** message it builds (attribution recorded at the top of the event loop, so an early `continue` cannot skip it).
   - Events: `CONTEXT_COMPACTION_START` / `CONTEXT_COMPACTED` are emitted; `COMPACTION_REQUESTED` is an *input* event handled by the runner (`_handle_compaction_request`) for the manual path, and never enters the conversation history. The CLI sends it from `/compact`; `/context` shows the breakdown. The manual path only drops old tool outputs (never summarizes); its `CONTEXT_COMPACTED` carries the note saying what it did or why it did nothing, and `tokens_saved`, measured by replaying the record onto the conversation as the next request renders it. The CLI prints both and takes the saving off its last measurement, so the indicator and `/context` do not keep showing the context from before.
   - **A request the provider rejects as too long** (`is_context_overflow`, `llm/base.py`: a 400/413 in vLLM's, the LiteLLM gateway's, OpenAI's or Anthropic's wording) means the estimate under-counted: the planner compacts toward half the estimate (`Compactor.overflow_plan`) and retries, **once per turn**, in both planner loops. A second rejection ends the turn with the error. No declared window is needed — the rejection is the signal — so this also covers agents that declare none, and the window the rejection states is adopted for the requests that follow (above). On vLLM the rejection only arrives because `VLLMProvider` does not ask for prompt truncation (see LLM Providers).
   - **From vLLM 0.32 that rejection comes later** (vllm-project/vllm#57035): `max_tokens` is a ceiling on the output, not room reserved in the window, so a prompt that fits but leaves less than `max_tokens` is answered with the output clamped to what is left (`get_max_tokens`: HTTP 200, and an answer longer than that ends with `finish_reason: length`); only a prompt over the window on its own is rejected. Without a known window an agent therefore gets cut answers before any rejection; with one, the threshold (`window − reserve_tokens`) compacts first.
   - **A cut or dropped result can be read back** with `read_tool_result` (`domyn/agents/tools/read_tool_result.py`): the line where the context manager cuts a result says which characters it leaves out and gives the call, with the event id as `result_id` and the `offset` of the first character not shown; the placeholder that replaces a pruned tool result gives the call too (`LLMMessage.read_back`, set by the context manager on a tool's result only: the other tool-role messages, a sub-agent's task and its STOP report, offer none, and a pruned read offers that same read again). Nothing is stored for it: the whole result is the `TOOL_END`/`TOOL_ERROR` event, and the tool looks it up among the events the calling agent sees, so `context_visibility` applies unchanged. Offsets count the result as the model is shown it, what a streamable tool showed the user included: `ShownTexts` builds that text for the context manager and for the tool, on the same events in the same order (`_rendered_events`). A read returns as much as reaches the reading agent's model whole, its note included (`max_tool_result_chars`, less the event-id header when there is one), with the call that reads on; `find` lists the offset of each place a text occurs instead, matched as written — no regular expressions, which a model could write to backtrack for minutes while holding the GIL. The tool is in **no** agent's `tools`: `ToolUseContextManager.get_tool_definitions` offers it to every agent with tools of its own or agents to hand off to, `Agent.get_tool_by_name` resolves the name, and the runner passes a `ToolResults` as the reserved argument `tool_results`, made by the agent's context manager (`ContextManager.tool_results()`, `None` from a manager that gives no call), which knows both the events and the budget. It reaches the tool alone, merged into its arguments after its pre-hooks, as `loaded_skills` does: a REST pre-hook puts `tool_args` in its request body, where a live object cannot be serialized, so the request was never sent and the tool ran as if the hook did not exist. `ToolResults` deep-copies to itself, since a function tool deep-copies its arguments on every call. An agent with a tool of its own named `read_tool_result` keeps it: ours is not offered and its notes give no call. Neither is it offered to an agent with no tools and no one to hand off to, whose notes give no call either: such an agent has hardly a result to read back, and the definition (~970 characters with `reasoning`) would be the only reason for it to send tools at all, which a vLLM server started without `--enable-auto-tool-choice` rejects.
   - **The cut line says that the part not shown cannot be told from the rest** (`_keep_start_and_end`), not only which characters are missing. Told only the count, models took the visible start and end for the whole result and answered from their pattern. Measured with gpt-oss-120b on a listing whose exceptions were in the part not shown: 5 right answers in 15 with the count alone, 23 in 31 with this line (two-sided Fisher p = 0.011). Where the line sits matters too, and differently from what one might expect: the same sentence at the top of the result, with a short marker in the middle, got 0 right in 12 runs over two models. Of the 56 runs in which a model called `read_tool_result`, 55 answered right; the other searched for a file name escaped twice, which matched nothing.

2. **Prompting** (`planner_strategy/prompting/`) decides *how* a message is rendered. `BasePrompting` + `MessageBuilder` are the two interfaces (Python has no intersection types, so `MessageContextManager` validates both at construction); concrete classes are `JSONPrompting` / `JSONMessagePrompting` and `StructuralTagsPrompting` / `StructuralTagMessagePrompting`, selected by `PromptingType` (`json_schema`, `structural_tags`). This layer is **not used by the `tool_use*` managers**, which emit native tool calls directly.

### The interactive CLI

Two packages, and the split is not obvious from the names:

- `domyn/agents/cli/` — the Typer entrypoint and one subpackage per command (`cli/commands/run/`, …). Argument parsing and wiring.
- `domyn/agents/interfaces/cli/` — everything the session itself is made of: the REPL, the renderer, the theme, the config model, the views. **This is where almost all CLI behaviour lives**, so look here first despite the name.

`~/.domyn/` holds the state: `config.yaml` (provider, model, agent, tools, workspace, compaction), `credentials` (0600, API keys never in `config.yaml`), `approvals.yaml` and `projects/*/approvals.yaml`, `builtin-skills/` (the builtin skills chosen in setup, rewritten by every setup — see Skills), and `conversations.db` (the history `--resume` and `/resume` read). It is **not** an instructions folder: `SOUL.md` / `AGENTS.md` there are not read (see below).

**`domyn setup` over an existing config** first shows each section — model, instructions, tools, skills — and asks whether to keep it (`_keep`, `interfaces/cli/setup.py`, yes by default); a yes skips that section's questions. Not asked on a first setup, by `/model` (`provider_only`), or for a model the config leaves incomplete; asked on `--import` too, each section showing what the export brought. Kept skills are not reinstalled, so built-in skills changed by an upgrade arrive only on a setup that does not keep them. `tests/cli/conftest.py` answers it no, so scripted re-runs go through every question; `tests/cli/test_keep_sections_setup.py` puts it back.

**A text field opens with its current value, and an erased one is its default** (`_ask(label, current, default=…)`, `interfaces/cli/setup.py`, over `read_line(prefill=…)`, `interfaces/cli/prompt.py`). Enter keeps the value; erased, the field returns `default`, which the label names (`empty = provider default`, `empty = 16000`, `empty = use DOMYN_SPACE_ID`). When Enter kept the bracketed value there was no way back to a default once one had been saved: an Anthropic URL pointing at a gateway, a pinned Domyn configuration id. A pipe cannot pre-fill, so there the value is shown in brackets and Enter keeps it, as before, which is what the scripted tests in `tests/cli` see.

**Ctrl-B goes back one question** (`run_setup`, `interfaces/cli/setup.py`). Each question is a `_Step` reading and writing one `_Draft`, with a condition saying when it is asked, so going back follows the branches (the Domyn provider asks URL, space, configuration, key, then model). Before a step runs the draft is copied; Ctrl-B pops back to the previous step that asked something and **restores its draft**, so a step asked again never sees its own earlier result (the custom skills folder would be listed twice, an empty instructions path would remove what it had found). What was typed is not lost: `_Recall` opens each question on the answer given there in an earlier run of the step (`_ask`, and the wizard's `_select` / `_multiselect` funnels), never a password, and picking another provider forgets the model section's answers. A step that asked nothing (a key already set) is passed over, or Ctrl-B would land on it and come straight forward again. **On a terminal the question gone back to is asked in its place**, not below: everything after where it started is erased (`_Screen`, `setup.py`), the rows counted as newlines written through the wizard's console, which works because rich wraps lines itself and, inside `back_key`, a question erases itself when it ends (`erase_when_done`): an answered text field or pick is printed back as one line (`read_line`, `select._echo`), a tick list prints nothing, its caller saying what was ticked (`Tools: …`, `Installed 11 skills`). `run_setup` counts on a console of its own over the caller's file, so the caller's is left as it was. A row that has scrolled off the top cannot be erased (the cursor stops at the top and the rest stays in the scrollback), a window resized mid-wizard throws the count off, and anything printed past the console (prompt_toolkit's CPR warning on a terminal that does not answer it) does too. Off a terminal it prints `← back` instead. Checked by driving `domyn setup` and `/model` in a pseudo-terminal (pyte) as well as by `tests/cli/test_setup_back.py`. The key is opt-in: `back_key()` (`interfaces/cli/prompt.py`) turns it on for `read_line`, `select` and `multiselect`, which raise `GoBack`; outside it Ctrl-B keeps its meaning, Esc cancels a picker everywhere, and Ctrl-C always aborts. It is not Esc: a lone Esc can only be told from the start of Option+← (`ESC b`) by waiting for the key after it, so going back came half a second late. The cost is that in a wizard text field Ctrl-B no longer moves the cursor a character to the left (← does). A pipe has no key to press. The deletion an import makes of the instructions it does not keep waits until the config is saved (`_Draft.discard`): done in the step, going back to keep them would keep paths whose files were gone. It then skips every file the saved config lists, and every folder holding one: the instructions question asked after the unlisting can list them again. Tests script Ctrl-B as a `GoBack()` among the answers (`tests/cli/test_setup_back.py`); a step replaced by a stub asks nothing through the prompts, so it is passed over when going back.

**Projects: `--project <name>`** (`setup`, `run`, `export`, `skills`, `skill`; `cli/project.py` for the option, `use_project` in `interfaces/cli/config.py` for the rest) selects `~/.domyn/projects/<name>/`, or a project folder anywhere when the value is a **path** (`is_project_path`: it has a `/` or starts with `.` or `~`, none of which a name may have, so `alpha` is always the one under `projects/` and never `./alpha`; a relative path is read against the working directory). The folder stands in for `~/.domyn` for the whole process, and `domyn run` names it under the banner as `project <folder> (<location>)` (`describe_project`, home written `~`): `config_dir()`/`home_dir()` return it ahead of `./.domyn`, so its `config.yaml`, the skills setup installs (`skills_install_dir()` = `<project>/builtin-skills`), `conversations.db`, `credentials` and approvals (`approvals.yaml`, and the per-folder ones under the project's own `projects/`) are its own, with **no fallback** to `~/.domyn` at run time — **except API keys**: a key the project's `credentials` does not set is read from `~/.domyn/credentials` (`credential_files`, project first), so a key kept at home serves every project and a project overrides it by setting its own. Instructions are no exception to the rule that nothing is read without being declared: a project's `SOUL.md`/`AGENTS.md` are the files its config lists (as `${DOMYN_PROJECT_DIR}/…` when they are inside it), and a file at the project's top that the config does not list is not read. Setup looks for the API key (`find_api_key`) in the same order a session reads it — the environment (a `.env` in the folder it runs in included, named as such), then the project's `credentials`, then `~/.domyn/credentials` — and reports where it found it, **copying nothing**: a key found at home stays there. A key in a credentials file (`FoundKey.path`), pasted earlier in the same run included, is then offered as a masked field (`_ensure_api_key`): Enter keeps it, a key typed replaces it in the active `credentials` (the project's, in a project, overriding the global one there only) and goes into the environment through `load_credentials_into_environment`, so going back to the question offers it again; a key from the environment or a `.env` is only reported, since setup could not replace it. A pipe shows a pre-filled password as asterisks (`read_line`). Credentials are put in the environment at startup, before `--project` is parsed, so `select_project` calls `load_credentials_into_environment` again, which first takes back what the previous call set (only while it still holds that value; a real environment variable or `.env` entry always wins) and loads the project's keys over the global ones. **Export never copies them**: it copies only what the config names (below), and refuses a file named `credentials`, `approvals.yaml` or `conversations.db*` even when a skill's `external_files` declares it (`is_private`, `interfaces/cli/export.py`, checked in `_copy`, the one place every file goes through). Only `domyn setup` creates one (`must_exist=False`); every other command refuses a project with no `config.yaml` and names the existing ones, so a typo never opens the wizard. Names are one folder, and never `<x>-<12 hex>`, the shape of the per-directory approvals folders that share `projects/`. **A relative path means the folder domyn runs in, in a project as everywhere else; a project names its own files with `${DOMYN_PROJECT_DIR}`** (`PROJECT_DIR_VAR`). `use_project` sets that variable to the selected folder, **over** any value already in the environment (a stale one must not point the config at another project's skills), and **removes** it when no project is selected, so the variable means "this session runs a project" and nothing else (a config file run by name that uses it is run with `--project <its folder>`), and the loader's ordinary `${VAR}` interpolation fills it in — no path is rewritten on read, and `run`/`/model` load through the plain `load_application_from_file`. Setup works on the paths with the variable expanded (`expand_project_dir`, that variable only, so its "installed here?" checks compare real folders) and saves paths inside the project as `${DOMYN_PROJECT_DIR}/…` (`in_project_terms`), so a project folder works moved, renamed or exported. The `.domyn/skills` and `.agents/skills` autodiscovery entries are just relative paths, so they need no exception. Outside the CLI, whoever loads a project's config sets the variable; unset, the loader names it. It is also the one CLI variable `run_command` passes through its isolated environment (`_SAFE_ENV_NAMES`, `tools/builtin/session.py`): a folder path, not a secret, and the only way the agent can tell which configuration the session runs — the `domyn-cli` skill has it check `$DOMYN_PROJECT_DIR` first and pass `--project "$DOMYN_PROJECT_DIR"` to any `domyn` command it runs, which otherwise changes `~/.domyn` instead. This is what makes an export a project: used in place with `--project <its path>`, or moved to `~/.domyn/projects/<name>` and run with `--project <name>`. **A project given by path is checked every time it is used; one given by name is trusted** (`select_project`, `cli/project.py`; the checks in `interfaces/cli/project_trust.py`). The folders under `~/.domyn/projects` are the ones `domyn setup` made or imported, so `--project <name>` uses them as they are; a path can be any folder, an export someone sent included, and nothing about it is remembered, as a record of accepted folders would be one more thing to keep honest. For a path, two things are shown and asked about on a terminal (default no), every time, and refused without one: **files outside the folder** (`project_problems`: every path the config names, `config_paths`, judged by where it really leads — a link, `..`, an absolute path into the project — and every skill's `external_files` entry leading out of it), and, for `domyn run` only (`runs_code`), **the code it runs** (`code_files`: the file of every `tools`/`hooks` entry, executed when the agent is built, before any approval), each with what it imports (`import_warnings`). A project setup made, used by its path, can name files outside itself (a skills folder or an instruction file elsewhere is listed where it is), so it is asked about too. Approvals, credentials and history in the folder are not checked: every project used has them.

**`domyn export <dir>`** (`interfaces/cli/export.py`) writes a folder laid out like `~/.domyn` — `config.yaml`, `builtin-skills/`, `skills/`, `instructions/` — holding every file the configuration reads, with every path to a copied file rewritten as `${DOMYN_PROJECT_DIR}/…`: a project (above), usable in place or under `~/.domyn/projects/`, and `DOMYN_PROJECT_DIR` is never listed among the variables to set. `--project` exports a project, with its own `builtin-skills/` as the install directory. **What is copied is opt-in, never a folder's contents**: the config, each listed skill's `SKILL.md` (or loose `.md`) plus exactly what its `external_files` declares — the list setup installs built-ins from, so their exports are complete by construction — and the `SOUL.md`/`AGENTS.md` the instruction sources contribute. Anything else in a skill's folder stays behind and is named in the report's warnings (the fix is to declare it); a declared folder entry (`scripts/`, `.`) copies what it holds except hidden files, and no entry copies a file with a hidden part in its path relative to the skill (`../../.ssh/id_rsa`, `.env` named outright): it is named in the report's dropped list (`_hidden_below`), since a skill listed by path may be someone else's. A skills path is read as the loader reads it (group folders walked, symlinks not followed) and copied with its *anchor*, the nearest folder holding the path and every `external_files` entry of its skills, so `../shared.py` still resolves in the copy. Setup's `~/.domyn/builtin-skills` is `builtin-skills/` (its scripts find shared files by position, which the anchor keeps; an install older than the `external_files` name lists none and needs `domyn setup` again), so the export used as a project has its built-ins where its own setup installs them; any other anchor is `skills/<folder name>/` (a folder called `skills` is named after its parent; a name already taken gets a suffix). The `.domyn/skills` and `.agents/skills` autodiscovery entries are **snapshotted and merged** (`_merged_skills`, `_copy_merged`): the skills in those folders of the folder the export runs in are copied straight into `skills/` with their categories (`.domyn/skills/<category>/<name>` → `skills/<category>/<name>`) and listed **one by one**, as `${DOMYN_PROJECT_DIR}/skills/<category>/<name>`, where the first entry was — `skills/` itself cannot be listed, as it also holds the other copied folders, which would load a second time. A clash is settled before copying, as the loader settles it — by skill name (or by `category/name`) the earlier folder wins, `.domyn` in setup's order — and the skill left out is named in the report. A skill another listed path also reaches (the same folder named on its own, or inside a listed folder) is the same files: it is copied once and listed by that copy (`_place_in_anchor`); so is one whose place already holds an identical copy from another source (`_identical_copy_at`: every file at its position with the same bytes, shared ones above the skill included, and nothing else in its folder). A place already taken in `skills/` by anything else moves the skill instead, still listed first so it still wins its name: another skill at the same place makes it `<name>-2`, a category that is another entry's copied folder (walked when loading) makes it `<category>-2`. A copy of a built-in is not one: the built-ins are in `builtin-skills/`, so the copy is exported as the user's own and still wins the name. A file a skill shares from beside or above it (`../style.md`, `../../shared.py`) keeps its position relative to the skill anywhere in the export (`_shared_place_problem`), except outside it, at `config.yaml`, in `instructions/` or in `builtin-skills/`; a place another file already has keeps the first (the folders in order); either refusal is named, and that skill is unavailable in the export. A folder that is missing there is left out quietly. Instructions: only what an agent's `standing_instructions` lists, as nothing else is read — a folder, a file, and `.domyn`/`.agents` of the folder exported from (`LOCAL_INSTRUCTION_DIRS`), which are **snapshotted** like the skills autodiscovery: walked as the loader walks them (`find_instruction_files`, `STATE_DIRS` skipped; one with no file is left out quietly, and `./.domyn` is when it is `~/.domyn`, exported from `$HOME`), every file found listed **one by one** — goes in `instructions/` with **the least nesting that keeps them all** (`_place_instruction_sources`): straight in when no two files share a name (`AGENTS.md` and `AGENT.md` counting as one, as a folder contributes only the first), otherwise a folder per source named by the fewest trailing parts of its path that tell them apart (`dir/a`, `dir/b` → `instructions/a`, `instructions/b`; `x/a`, `y/a` → `instructions/x/a`, `instructions/y/a`; the souls of `.domyn` and `.agents` → `instructions/domyn/SOUL.md`, `instructions/agents/SOUL.md`), never with a leading dot. Nothing is merged away: every `SOUL.md` and `AGENTS.md` is read, in the config's order. The file each `tools`/`hooks` entry runs is copied into `code/` (the file alone; `_export_code`), and its imports are read, never run (`import_warnings`: `ast`, then `find_spec` on each top-level name): the report warns about a module beside it or a relative import (not copied), a package that is neither the standard library nor domyn or one of its requirements (named with its distribution, to install where it runs), and a module not found even here. **The exported folder names nothing outside itself**: a path left in the config that is not `${DOMYN_PROJECT_DIR}/…` (skills, instructions, tool or hook code, a bundle's `workspace`, a browser's `executable_path` or a `binary` with a `/`, an OAuth `storage_dir` — `paths_outside_project`), or an `external_files` entry leading out of the export from where its skill was copied to, because a file was not copied (`external_files_outside`), refuses the export, every one named, and nothing is written. The destination must not exist or be empty, and is built beside itself and moved into place. **`/export <dir>`** does the same from inside a session, for the session's config file (any YAML one, not only the default), and prints the same report (`print_export_report`).

**`domyn setup --import <dir>`** (`interfaces/cli/project_import.py`) is the way back: a new project started from an export: the `--project` one (a name or a path, as everywhere), else `~/.domyn/projects/<the name of dir>` (a folder name that is not a valid project name is then refused, naming both fixes; `--show` alongside it is refused). That folder must not exist or be empty. The export's files are copied in first, then the wizard runs with its config as every question's default (the title names the export); an aborted wizard removes the project again, a completed one ends by naming the project and `domyn run --project <it>` (a path project spelled absolute, as a relative one names it only from this folder). Every entry at the export's top keeps its name: `builtin-skills/` becomes the project's install directory, `skills/` and `instructions/` stay where they are, so the config's `${DOMYN_PROJECT_DIR}/…` paths need no rewrite; hidden entries, links and `is_private` files are not copied. A later `domyn setup --project <it>` offers the imported built-ins in its catalog, ticked, and reinstalls them from the package. The export's `code/` is not copied, and the `tools`/`hooks` entries that run it are named in a warning (`ProjectImport.code_left_out`): setup writes `config.yaml` from its own questions and keeps no tools or hooks, so the code would sit there with nothing naming it. **The version is compared, not enforced**: `domyn export` writes a hidden `.info` beside `config.yaml` (`INFO_FILENAME`, `export_info`, `interfaces/cli/export.py`): `key: value` lines with the `export_format_version` (`EXPORT_FORMAT_VERSION`: the export's folder layout, how its config names those files, and these keys; raised when one changes in a way an import has to know), the `domyn_agents_version` that made it, `exported_at` (UTC), `python_version`, `platform` (OS and architecture) and `domyn_platform_sdk_version` when the SDK is installed — nothing naming the machine or its user (no path, host or user name), as an export is handed on; the import reads it (`ProjectImport.info`, `exported_with`), warns before the wizard when it differs from the running version or is missing (`ProjectImport.version_warning`), and does not copy it into the project. A skill's shared file may not land at that name (`_shared_place_problem`). **The import refuses an export that names anything outside itself**, by the same two checks, before copying anything: an export is a folder anyone can edit after it is made, so the check that protects the machine importing it is the one run there. Every section is first offered for keeping, as on a re-run (`_keep`), so Enter on each keeps the export as it came. Two steps differ from a plain setup when a section is not kept, and only with `--import`: **the built-in picker is replaced by one over the project's own skills** (`_choose_project_skills`, `interfaces/cli/setup.py`): every skill the config lists inside the project, `builtin-skills/` and `skills/` alike (one walk of the project, so labels stay unique), offered as exported and grouped by folder, all ticked, and none installed again; a listed folder whose skills all stay ticked is kept as written, otherwise its ticked skills are listed one by one in its place. The catalog's built-ins the export does not include (recognised by their place in the install directory, never by name) are offered beside them, unticked, in their category's group; those ticked are added to the install directory by `add_skills` (`skills/catalog.py`), which, unlike `install_skills`, replaces nothing already there, so the exported built-ins and the shared files they came with stay as exported. **Instructions not kept are erased** (`_choose_imported_instructions`): the declared paths are unlisted and the copies inside the project deleted once the config is saved, as nothing would read them again, then the usual question is asked and may bring a soul of its own. A copy the saved config lists again (that question answered with the project's own `instructions/`) is not deleted. The `./.domyn` / `./.agents` instructions autodiscovery question follows, as in any setup (`_choose_local_instructions`, never as a first setup: an export lists what it found, so it defaults to off).

**Slash commands** are declared once in `interfaces/cli/commands.py` (`COMMANDS`), which is the single source for both `/help` and the autocompleter, and dispatched by name in `repl.py`: `/help /exit /clear /agent /skills /skill /skills-autodiscover /auto /approvals /verbose /voice /model /config /export /context /compact /ps /stop /resume`.

The interactive session is **fixed** to the `tool_use_thought` planner — see `CLI_PLANNER` (`interfaces/cli/config.py`). Note `ToolUseWithThoughtPlannerStrategy.execute` is a full copy of its parent's loop rather than an extension: anything added to one loop is silently missing from the other, which has already shipped a broken feature once. Test both.

### Standing instructions: SOUL.md and AGENTS.md

`domyn/agents/instructions/` — two instructions merged into the system prompt, loaded **like skills**: a source, a runtime, and the runner loading it every turn. The distinction between them is the design, not a naming preference:

| | `SOUL.md` | `AGENTS.md` |
|---|---|---|
| what | identity, tone, standing preferences | what the agent is working on |
| follows | the **user** | the task |
| where | the user's folder, else `./.domyn/SOUL.md`, else `./.agents/SOUL.md` — **one**, the first in path order | the user's folder, then `./.domyn/AGENTS.md`, then `./.agents/AGENTS.md` — **all** (`AGENT.md` accepted) |

The pieces, mirroring skills:

| skills | instructions |
|---|---|
| `Agent.skills: SkillsRuntime` | `Agent.standing_instructions: InstructionRuntime` (`runtime.py`, owns the 32 KiB budget) |
| `SkillSource` | `InstructionsSource` (`sources.py`): `LocalDirInstructionsSource` (`local_dir`), `InMemoryInstructionsSource` (`memory`), resolved by `type` tag so it can be declared in YAML |
| `LoadedSkills` | `AgentInstructions` (`models.py`): one tuple of `Instruction`s per `InstructionKind` — `soul`, `agent`, `task` — plus `dropped` (cut by the budget) and `superseded` (souls after the first, not read); `from_parts` sorts a flat list into them |
| `Runner._load_turn_skills` → `RunnerState.skills` | `Runner._load_turn_instructions` → `RunnerState.instructions`, both handed to `Agent.execute` by `Runner._execute_kwargs` |
| `attach_skills` | `attach_instructions` |

There is no command-line flag and nothing walks up to a git root. A set of instructions is selected by declaring the paths that hold them in the agent's config — a directory, or a single file (the kind follows the name: `SOUL.md` is `SOUL`, `AGENTS.md`/`AGENT.md` is `AGENT`, any other name is `TASK`, a one-off brief — a directory only ever contributes the first two — directly in it, except the two autodiscovery entries, which are walked; below). A relative path is resolved against the working directory on **every load**, which is what the project layer is made of (below). Every `AGENTS.md` and `TASK` file found is included, each once even when reached twice; **souls are not**: an agent has one identity, so the first `SOUL.md` in path order is the soul, and any later one is not read but recorded in `AgentInstructions.superseded` (logged at debug). An empty `SOUL.md` is no soul, and gives way to the next. They are ordered **by kind, never by the order the paths were given** (`SOUL`, then `AGENT`, then `TASK` — the declaration order of `InstructionKind` — with path order kept within a kind), so the instructions closest to the task land last whichever way round the paths were listed. **No link is followed** (`LocalDirInstructionsSource._candidates`): the listed path, every folder on the way to a relative one (`listed_path_is_linked`, the skills check — `.agents -> ~` would read the home folder) and the files in a directory (`is_regular_file`, shared with the setup walk), each a warning; a relative path is chosen by whatever project the CLI starts in, and `AGENTS.md -> ~/.ssh/id_rsa` would send that file to the provider on every turn. Every agent in a YAML config can declare its own, beside `skills:` — `standing_instructions: {source: {type: local_dir, paths: [...]}}` (`source`, the same key `skills:` uses) — and reads only those. The CLI touches the root alone (`apply_session_instructions`, `cli/commands/run/command.py`, over `resolve_instruction_dirs`, `interfaces/cli/config.py`): a declared `local_dir` source keeps its paths, duplicates removed; a root that declares none gets none; a non-`local_dir` source is kept as it is. **`~/.domyn` is not read**: it used to be prepended to every root, and no longer is, with no migration.

**The project layer: `./.domyn` and `./.agents` autodiscovery, opt-in at setup.** `AUTODISCOVERY_DIRS` (`instructions/discovery.py`, imported by setup and the CLI config) names the two folders. They are the only declared directories the loader **walks**, and it does so on every load (`_candidates` → `find_instruction_files(source, skip=STATE_DIRS)`), recognising them as written (`.domyn`, `./.domyn/` alike): breadth first, sorted by name within a level, hidden and dependency folders and links skipped as in the setup walk, so a file added in any subfolder is read on the next message. Within one folder the shallowest `SOUL.md` comes first (deeper ones, and any after it, are `superseded`) and the `AGENTS.md` files go from the most general to the most specific. **domyn's own state folders are skipped**: `STATE_DIRS` (`conversations`, `browser`, `processes`, `skills`), among the root's direct children only, so a user's `docs/skills/` deeper down is still walked; what is in them is the agent's scratch or skill folders, and an `AGENTS.md` there must never reach the system prompt. `tests/instructions/test_state_dirs.py` keeps the list honest: it fails when a built-in writes under `.domyn` to a folder not in it, so a new state folder has to be added there. Setup writes them as written, `.domyn` then `.agents`, **after** the user's own paths — the reverse of skills, where they go first, because the order means the opposite: skills settle a name clash by the earlier directory, while here the first soul wins and the last `AGENTS.md` is the one a model weighs most. So the user's soul beats a project's (and `.domyn`'s beats `.agents`'s), and the project's brief lands after the user's (`.agents`'s after `.domyn`'s — one list cannot make `.domyn` win both). `is_instruction_autodiscovery` recognises the entries by **exact** match, not containment: a file of the user's folder that sits under `./.domyn` is the user's entry. Launched from `$HOME`, `./.domyn` *is* `~/.domyn`, so `resolve_instruction_dirs` drops an autodiscovery entry that resolves to `GLOBAL_DIR`. This puts a cloned repository's text into the **system prompt**, every turn and in full — more than skills autodiscovery does, which is why it is opt-in. Sub-agents never get it from setup.

**`domyn setup` asks for the path, then whether to autodiscover** (`_choose_instructions` then `_choose_local_instructions`, `interfaces/cli/setup.py`), between the agent's name and its own `instruction`, in this order: name, path, autodiscovery, `instruction` (the instructions section's keep question shows the paths, the name and the `instruction` first). The `instruction` is asked last: Enter keeps the current one, an erased field is none (`AgentConfig`'s default, empty). The agent's name (the config's `entrypoint`) is asked first: Enter keeps the current one, an erased field is `AgentConfig`'s default (`Assistant`), and a name another agent of the file has, or one longer than `Agent` accepts (`MAX_AGENT_NAME_LENGTH`, 30), is refused and asked again; a rename keeps the agent's hand-written keys, since `_keep_unowned` finds the previous entrypoint as the wizard read it rather than by name. The path: a file is saved as given; a folder is walked once, at setup (`find_instruction_files`, `instructions/discovery.py`): breadth first, the shallowest `SOUL.md` and every `AGENTS.md` (`AGENT.md` where a folder has none), hidden and dependency folders skipped and no link followed, as in the skills walk. Each file found is written by absolute path into the root's `standing_instructions`, so an edit is read on the next message but a file added later needs setup again — the loader never walks a declared directory other than the two autodiscovery entries. Nothing found, a missing path or a link is a warning and saves nothing. Unlike every other prompt, an empty answer **removes** the current paths rather than keeping them, so they are printed first. The path step owns every entry but the two autodiscovery ones, which it keeps after the user's whatever the answer; the autodiscovery step (yes/no, yes on a first setup, the current state on a re-run, asked every time since it applies to every folder) owns those two and appends or removes them. A non-`local_dir` source is kept by both without asking. `AgentConfig.standing_instructions` holds the raw mapping, like `skills`, so the rewrite keeps it. `tests/cli/conftest.py` replaces both steps, and answers the system instruction with the current one, like the skills and tools steps, and points `AUTODISCOVERY_DIRS` (in the CLI config and setup modules) at folders that do not exist (this repository has a `.domyn` of its own); `tests/cli/test_instruction_setup.py`, `tests/cli/test_instructions_setup.py` and `tests/cli/test_local_instructions_setup.py` put them back.

Things that are decided rather than incidental:

- **Loaded every turn, so edits and `/clear` need nothing special.** Nothing is cached between loads: an edited `SOUL.md` reaches the next message, and a new conversation is just the next turn. A source that fails gives that turn no instructions, with a warning; unlike a `required` skill it never refuses a turn.
- **Merged at render time, never written into `Agent.instruction`.** `Agent._execute` merges what the runner loaded (`InstructionRuntime.merge`: appended after the author's instruction, never substituted) into a local and passes it to the planner as `instruction=`, which goes to `get_context(..., instruction=)` and from there to the two system-prompt builders (`ToolUseContextManager._build_system_prompt`, `BasePrompting.get_agent_instructions_text`) in place of `agent.instruction`. One agent serves concurrent conversations, and a remote source may serve each user a different soul, so nothing per-turn is kept on the agent. This is also what makes `attach_instructions` idempotent. Outside a runner, `Agent._execute` loads on demand, as for skills. An `instruction` that is an `Awaitable[str]` is left alone with a warning.
- **`Agent.instruction` is deliberately uncapped.** It carried `max_length=2000`; the limit validated nothing — the value is never serialized, never sent to a peer, never reaches a delegate — and had already bent two designs around itself.
- **The system prompt, not an injected message.** Skills and memory are injected messages (`turn_context_messages`); these are not. Compaction never touches the system prompt, and its content is constant between edits, so it is the part providers can cache.
- **The root agent only, unless declared.** The CLI attaches a runtime to the root, never to `sub_agents`: a standing preamble repeated down a tree is paid again at every handoff. A sub-agent that declares its own is loaded by the runner like any other (`_load_turn_instructions` covers the whole tree) and handed only its own. `/model` rebuilds the agent from disk, so `Repl._change_model` carries the runtime across, as it does the skills runtime. Agents that do not declare the field (`WrappedAgent`, `DelegateAgent`, adapters) are skipped.
- **32 KiB total, and nothing is cut silently.** Over budget (`InstructionRuntime.max_bytes`), the instruction that crosses the line (in that same order) is truncated on a character boundary and the rest are recorded in `AgentInstructions.dropped`; both show up in the log and in the rendered block itself.

**`config.yaml` is reread every turn.** The REPL keeps the content of the config file the session was built from, and before every turn, after every turn and before `/skills`, `/skill`, `/config`, `/agent` and `/model` compares it with the file (`Repl._reload_if_changed`). When it changed, the agent is rebuilt the way startup builds it — `load_session_agent` / `configure_session_agent` (`cli/commands/run/command.py`), with the session's command-line options kept in `SessionOptions` (`--expand-skills`, `--skill-sections`) — and the session prints `Reloaded …`: any change, model, tools, skills, instructions, MCP servers, is in use from the next message. Only a changed file is rebuilt (an MCP server reconnects on a rebuild). A file that no longer loads keeps the agent the session has and is reported once, with the YAML line where it broke; its content is remembered either way, so it is not retried every turn. The conversation is the runner's, so it carries over, as for `/model` (which reloads first, so it never overwrites a hand edit, and records its own write so it is not reloaded twice). Like `/model`, the old agent is not closed: its shell backend and background processes outlive the swap. Command-line options and credentials still need a restart.

**The line under the banner is said again when it changes.** It names the skills and instruction files once, at startup, and is scrollback; both are read from disk every turn, so at the end of each turn the REPL takes an inventory (`interfaces/cli/inventory.py`: `take_inventory`, `describe_change`) and, only if a skill or an instruction file appeared or disappeared since the last one — added by the user, or by the agent during the turn — prints one `Now: …` line naming what changed. An edit inside an existing file is not a change there. `/skills`, `/skill` and `/config` need nothing of the kind: they read from disk when used.

Files outside the workspace (a user's folder anywhere on disk) are **not** reachable by the file tools — instruction paths are not added to the workspace's extra roots, unlike skill directories; the project's own under `./.domyn` / `./.agents` are, being in the workspace. That is why every rendered block carries its own absolute path: the path in the text is the route from "why does it keep doing that" back to the file to edit. `/config` and `domyn setup --show` print what the file declares (`describe_configuration_v1`: the path, then one section per kind of element, the agent's `standing_instructions` source among them) and not which files were found, so they cannot say which `SOUL.md` is in use; the session names those once in the startup status line, and again whenever the set changes.

### Built-in Tools and the Workspace

`domyn/agents/tools/builtin/` — enabled per agent via `agent.tools` in `config.yaml`, in bundles (`BUNDLES = ("files", "shell", "browser")`; a `tool_bundles` entry that lists none gets `DEFAULT_BUNDLES = ("files", "shell")`, so adding a bundle never switches it on for existing configs):

- **files**: `read_file`, `list_dir`, `glob_files`, `grep` (ungated — they cannot damage anything) and `write_file`, `append_file`, `edit_file` (gated by `USER_CONFIRM`, with the *path* as approval subject, so "keep editing this file" is expressible and "write anywhere" is not). They propose no pattern: a directory glob needs normalized paths first, or `src/*` would match `src/../.domyn/approvals.yaml`.
  - `grep(pattern, path, glob)` reads `path` and `glob` as ripgrep and the other harnesses do, because that is how models write them: `path` is a directory or one file; a `glob` without a `/` matches the file name at any depth under `path` (`*.py`), one with a `/` is a path from the root that holds `path` — the workspace, like `path` itself (`src/*.py` with `path="src"`), or the extra root `path` is in — and `path` still bounds the search. Only `**/` followed by a name alone (`**/*.py`) is read from `path`, where it names the same files and walks less: a directory after `**/` may be `path` itself (`**/src/*.py` under `src` names `src/lib.py`). Read from `path`, as before, `src/*.py` named `src/src/*.py`, and a file `path` was searched as an empty directory: in 85 sessions on four models, 56 of 133 `grep` calls came back empty for one of the two. An empty search says which kind it is (`_files_to_search`): `No matches for … in N files under …` after reading files, `Nothing was searched: …` when the arguments name no text file, since a model reads a bare "No matches" as "the text is not there".
  - Every file a glob finds is resolved and checked against the workspace, in `grep` (`_contained`) and `glob_files` (`_matches`) alike, as `Workspace.resolve` checks every path a tool is given. Both are ungated, and a glob climbing out with `..` (`../*`) or a symlink in the workspace pointing out used to let them read and list any file the process could, where `read_file` and `list_dir` refuse the same path.
  - `read_file` returns windows of at most `MAX_READ_CHARS` (12 000), each ending with the `start_line` to continue from. The window has to fit under the context manager's `max_tool_result_chars` (20 000), which keeps the start and the end of a longer result: a longer window would lose its middle, and the note's `start_line` would then read on past lines the model never saw. A single line longer than a window is cut there and continued inside it with `start_char` (the note gives both values): minified JSON or a base64 blob was otherwise unreadable past its first 12 000 characters.
  - Documents (PDF, Word, Excel, PowerPoint, OpenDocument, RTF, EPUB) are converted to markdown on this machine only, never by a hosted OCR: a PDF by `pdf-inspector`, everything else by `firecrawl-anydoc` with `ocr="reject"`. Which files are documents and what a failed conversion means is **`domyn/agents/core/documents.py`**, a public module so that every reader of files — this tool, and hosts that embed the library and read files from object stores or mounted volumes — reads a file the same way; `read_file` keeps only what is its own (line windows, size limit, the words shown to the model). The name decides what is tried first (anydoc is asked instead of keeping a list) and the content settles it: text saved under a document's name (a TSV as `.xls`) is read as text when it is plainly text (UTF-8, no NUL byte, no document's signature, so a damaged document is still reported as one), a document under another name (no extension, a `.dotx` template) is converted. A CSV is read as written, by every reader; one that is not UTF-8 (an Excel export in Windows-1252) is converted to a table, and that is decided on the whole file, so a CSV that stops decoding late is a table from its first line rather than two numberings. Line numbers count lines of the converted text. A PDF is read **page by page**, each page under a `<!-- Page N -->` line (the physical page, from 1), because anydoc returns a PDF only whole and judges from eight sampled pages whether it has text: a page without text outside the sample vanished without a word, one inside it made the whole PDF unreadable. What a page holds that is not text is marked on it, never dropped in silence (`[Page not included: <why>.]`, large images, undecodable characters; a scan is read from the text its scanner left on it), `Text.pages_not_read`, `images_not_read` and `page_count` say the same, and `read_file`'s first window names what was left out, since the marks may be windows away. Only a PDF with no text on any page fails (`needs_ocr`). The rules per page, and what is deliberately not read (invisible text on a page of text, annotations), are in the module's docstring. **pdf-inspector runs in a child process** (`domyn/agents/scripts/pdf_reader.py`, started with `sys.executable -P`, answering in JSON), because it holds the GIL through each of its calls, seconds on a long PDF with every other thread stopped. The child is stopped after `_READ_SECONDS`, and on Linux at `_MEMORY_BYTES` of memory, both `resource_limit`; at most two run at once (`_READERS`). Under Kubernetes a container that runs out of memory loses every process in it, the caller included, so a host leaves room for twice `_MEMORY_BYTES`. A process that cannot start the child — a frozen or compiled build (PyInstaller, Nuitka), whose `sys.executable` is the application itself or no file at all, or one where starting it fails — reads the PDF in its own process instead (`_read_here`, logged once): the same pages, without the isolation, the time limit or the memory cap. A PDF pdf-inspector cannot open goes to anydoc in this process, for the kind of failure (`encrypted`…) or a whole read that says its pages are not marked. There is no OCR. Over `MAX_DOCUMENT_BYTES` (25 MB) a document is refused, not windowed. `grep` does not search inside documents.
  - `write_file`, `append_file` and `edit_file` refuse to write over a document `read_file` converts (`_refuse_text_over_document`, which asks `documents.document_format`): what the model read is a markdown conversion, and writing it back would replace the document with text that a later read shows as if nothing had happened. Decided as reading decides, so a document under another name is protected, text under a document's name stays writable, and a CSV is protected only when it is not UTF-8; a file over `MAX_DOCUMENT_BYTES` is left to the approval gate. The check runs inside the tool, so a user can approve a write that the tool then refuses.
- **shell**: `run_command`, always gated.
- **browser** (`tools/builtin/browser.py` for the tools and their approval subjects, `tools/builtin/browser_sessions.py` for the runtime — `BrowserSessions`, `BrowserOptions`, the URL policy in `UrlPolicy`; a row of `domyn setup`'s tool checklist, ticked on a first setup like the others; a `tool_bundles` entry that lists no bundles still does not get it): `browser_open`, `browser_snapshot`, `browser_read`, `browser_scroll`, `browser_back` ungated; `browser_click`, `browser_type`, `browser_press` gated **per site**. It drives the [`agent-browser`](https://github.com/vercel-labs/agent-browser) CLI (shipped in the standalone build at `<runtime>/bin`, found by `browser_install.resolve_binary` when it is not on `PATH`, pinned with its sha256 in `scripts/agent_browser.lock` and checked by `build_standalone.sh`; refresh with `scripts/update_agent_browser.py`) and **Chrome is downloaded on the first browser call** that finds none (`browser_install.py`: Chrome for Testing, current stable, into `~/.domyn/browsers/`; not when `executable_path`, `cdp_url`, another engine or a non-local backend is in play; a Chrome the user has is used as is; announced through `BrowserSessions.on_notice`, which the REPL prints), and every command is a shell line through the **same `CommandBackend` as the shell** — so the browser runs wherever commands run, and a sandbox backend moves both. Options live under `browser:` on the bundle entry (`BrowserOptions`: `engine` chrome/lightpanda, `executable_path`, `cdp_url`, `allowed_domains`, `allow_private_urls`, `idle_timeout_s`, `command_timeout_s`, `binary`; unknown keys are refused). The module docstring records what was *measured* about agent-browser, and the design follows from it:
  - **URL policy, before anything runs**: only `http`/`https` (agent-browser itself opens `file:` and `data:`), no loopback/private/link-local unless `allow_private_urls`, cloud metadata never (`UrlPolicy._ALWAYS_BLOCKED_IPS`, one comment per provider; compared after unmapping `::ffff:`, and not all private — Azure's WireServer `168.63.129.16` is public and Alibaba's `100.100.100.200` is in 100.64/10, so neither is caught by the private check). A URL is normalised the way agent-browser reads it first (`localhost:3000` → `https://localhost:3000`; `urlparse` calls `localhost` the scheme). A host Chrome and `urlsplit` would read differently is refused rather than interpreted: a backslash anywhere in the authority (Chrome ends the host there, so `127.0.0.1\@example.com` is loopback to it and `example.com` to Python), a `%` in the host (decoded by Chrome: `127%2e0%2e0%2e1`) and a non-ASCII host (mapped with a newer Unicode table than Python's IDNA codec: `lᵒcalhost`). All three were measured reaching a loopback server past the check; an internationalised name is given in its `xn--` form. It is re-checked where the page *landed* after every action, and at the start of every call, because a page script can navigate after any check. Requests a page makes itself are only fenced by `allowed_domains`.
  - **Every command passes `--config /dev/stdin` with `{}` piped in**: agent-browser applies `./agent-browser.json` from the workspace root and `~/.agent-browser/config.json`, either of which can name an executable — code execution on an ungated `browser_open`, from a cloned repo. Piped, not written: a config file in the workspace is one that repo can commit as a symlink, which `>` writes through, and one that every conversation shares, which `>` empties before refilling (a concurrent command then fails on an empty config). `/dev/null` is no substitute — an empty file is an invalid config.
  - **Gated actions name their target, and the target is checked.** A wrong ref *succeeds silently* in agent-browser (a ref from the previous page, or `e5` meant for `e6`), so `browser_click` / `browser_type` take `url`, `ref` and `name`, and `browser_press` takes `url`, all copied by the model from the latest snapshot. The approval prompt is built from those arguments alone — `https://github.com :: click "Sign in" [@e5]`, site first, proposed grant `https://github.com :: *` — and a URL that is not a plain `http(s)` host renders as `(invalid url)` with no grant offered, so a crafted one cannot borrow a granted site's prefix. Approving claims is safe because `verify()` checks them before anything runs: a fresh snapshot, then the **same page** (fragment ignored, empty path = `/`, path and query exact), the ref still present, and its accessible name **exactly** equal to `name`, except that every whitespace character counts as a plain space: the snapshot text the model copies prints a non-breaking space as a plain one, the JSON `refs` keep it. Any mismatch is a `BrowserError` saying nothing ran (and, for a name, which ref carries it now). Refs are stable across snapshots of one document (measured), which is why a ref from an earlier snapshot still verifies.
  - **No page cache.** `BrowserSessions` keeps only which sessions it started, so `close()` can end them. Every tool asks the browser where it is and what is on it at call time, so a page a script moved, or another call changed, is never described from memory. For the same reason `browser_read(ref=…)` takes `url` too, and is refused on any other page.
  - **Failures raise `BrowserError`** (a `TOOL_ERROR`), never return a string, like `WorkspaceError` for the file tools. The model reads only `str(e)`, with no error flag, so a failure *after* an action that ran names that action (`Clicked …, but: …`), or the model would repeat it.
  - Output is fenced as untrusted (a nonce-bearing marker) and capped at `BrowserSessions.INLINE_PAGE_CHARS` (12 000); the rest is written beside the browser *through the backend* and read back with a shell command (`sed -n`/`grep`, path given in the note) — not `read_file`, because a sandbox may keep that file outside the folders the file tools see. One session per conversation (`d-<hash>`, short because the socket path has a 103-byte budget), closed by `BuiltinTools.close()` — browser before backend — with the last-URL file agent-browser leaves removed. A Chrome session is ~1 GB; Lightpanda ~58 MB.

Which bundles the CLI agent mounts is asked by `domyn setup` (`_choose_tools`, `interfaces/cli/setup.py`), a checklist over `BUNDLES` like the built-in skills' one: every bundle ticked on a first setup, `browser` included (mounted without `agent-browser` on PATH, it prints the install hint), the current choice on a re-run, and `/model` does not ask. Browsing is a row of this checklist, not a question of its own. A new bundle needs an entry in `BUNDLES`, `BUNDLE_TOOL_NAMES` and `BUNDLE_DESCRIPTIONS` (the line the picker shows), and `TestBundles` fails if they disagree. A name in the config that is not a built-in bundle is kept as it is. `AgentConfig.tools` defaults to **empty**: the "all bundles on" default lives in the picker's first-setup ticks alone (a hand-written `tool_bundles` entry without `bundles` reads as `DEFAULT_BUNDLES`, files and shell, as the loader mounts it — the picker's ticks are a separate choice), so unticking everything (which writes no `tool_bundles`) reads back as no tools rather than ticking them all again on the next setup. `tests/cli/conftest.py` replaces the step with one that keeps the config as it is, for the same reason it replaces the project skills question, and `tests/cli/test_tool_bundles_setup.py` puts it back.

Every built-in resolves paths through `Workspace` (`tools/builtin/workspace.py`), which is the only *structural* protection they have — the approval gate stops the unexpected, this stops the expected-but-wrong (`~/.ssh/id_rsa` looking relevant). Containment resolves both paths and compares, never a string prefix: `/home/me/work` is a prefix of `/home/me/work-secrets`, and a symlink inside the workspace can point anywhere. That holds for the search tools too: `glob_files` and `grep` hand their pattern to `Path.glob`, which does not know the workspace, so every match is resolved and dropped when it is in no root (`../*` listed the parent, and `grep`'s `glob` then *read* what it matched, ungated). `glob_files(pattern, path)` takes the folder as `path` so an extra root can be searched, and its `_matches` (`tools/builtin/files.py`) also refuses, with a message saying what to do instead, a pattern that is absolute, starts with `~` or has `..`; `grep` reads its `glob` as ripgrep does (`_files_to_search`, above), from the root that holds `path`. `Workspace.extra_roots` adds reachable directories beyond the root — used for skill directories, and nothing should be added to it casually.

Shell commands run through a `CommandBackend` (`tools/builtin/backends.py`). Only `local` exists; an unknown name **raises** rather than falling back to running on this machine.

### Approvals

`domyn/agents/approvals/` — whether a tool call needs a human first.

- `ApprovalType` (`core/__init__.py`): `USER_CONFIRM`, `PARAMS_UPDATE`. Set per tool.
- `ApprovalDecision`: `ALLOW` / `DENY` / `ASK`.
- `ApprovalScope`: `ONCE`, `CONVERSATION`, `USER`, `PROJECT`, `GLOBAL` — how long a grant lasts.
- Stores: `InMemoryApprovalStore`, `ConversationApprovalStore`, `FileApprovalStore` (written atomically). The CLI keeps conversation grants on the interaction state, project grants in `~/.domyn/projects/<name>-<hash of the path>/approvals.yaml`, user grants in `~/.domyn/approvals.yaml`. Project grants are never read from the repo, and launching from `$HOME` offers no project scope.
- `ApprovalPolicy` (`approvals/policy.py`) holds the wiring; the hooks hold the judgement.

Grants are keyed by `user_id` + subject, also within a shared conversation, and `/approvals` lists and revokes them. `/auto` approves everything for the rest of the session; Shift+Tab toggles the same flag from the editor, mid-turn and on an open approval question too (`Repl._toggle_auto_from_key`), and is not bound while the completion menu is open, where it means "previous entry".

**Several calls in one response** (`Runner._run_tool_calls`). The tool-use planners stamp `METADATA_TOOL_BATCH` (`id`, `index`, `size`) on each `TOOL_START` of a run of two or more consecutive calls to the agent's own tools (`tool_batches`, `planner_strategy/tool_use.py`, used by both planner copies); the runner holds them until the last and runs the batch. Consecutive calls whose tool has no `approval_type` (and is not an MCP tool behind OAuth) run concurrently, at most `Runner(max_parallel_tool_calls=...)` (8) at once (`_run_tool_group`); a gated call runs alone and in order, after everything before it and before everything after it. `parallel_tool_calls` on the tool-use strategies (default `True`, also in YAML) turns batching off: no metadata is stamped, so each call runs on arrival, in order. A pause stores the waiting call id and the calls after it under `context.state["held_tool_calls"]`: the TOOL_FEEDBACK resume answers the call recorded as waiting (by `call_id`, not the latest TOOL_START, which is the last of the batch), runs the held calls, then asks the model; a USER_INPUT instead closes them (`_drop_held_tool_calls`), and a hook stopping the turn mid-batch closes the rest (`_close_tool_calls`), each with a `TOOL_END` saying it was not executed. Every TOOL_START of a batch is stored the moment the planner yields it, so every one needs a result, or the next request carries a call without one and the provider refuses it. A TOOL_START without the metadata (the react planner, a delegate, a custom planner) runs on arrival, as before.

**The reply answers the waiting call** (`_handle_tool_feedback`): tool and arguments come from the waiting `TOOL_START`, never from the reply. A reply naming another tool, or the `call_id` of an earlier call, is ignored and the call keeps waiting (`_answers_another_call`); a `call_id` the conversation has never seen is accepted, so a client that does not echo it gets tool-name matching only. Edited `PARAMS_UPDATE` arguments go through hardline, deny rules and approval hooks again (`_refuse_edited_call`).

A grant is the exact subject or an `fnmatch` pattern proposed by the tool's `approval_pattern` (`run_command`: the line up to its program, `$ PYTHONPATH=src python *`). The subject is never shortened — it is the grant key; one that does not fit the prompt's heading gives way to the tool's name, with the arguments below it. A pattern that does not cover the call it is offered on is dropped (`_proposed_pattern`) or asked again (`_edit_pattern`). In the CLI a pattern lasts for the conversation only.

**A call is shown once, as written** (`interfaces/cli/calls.py`). The trace line is one line per call, cropped at the terminal's edge: the argument that names the call first (`_SHAPES`, built-in tools only — any other tool reads `key=value`), a value on several lines counted (`content=(3 lines)`), `reasoning` left out. The approval heading is the subject when it fits on one line and does not flatten a multi-line argument (`$ a b` reads as one command where two run), else the tool's name; the body has every argument the heading does not show, in full: a command as its block, `edit_file` as a `-`/`+` diff, anything else as `name: value`. Whether the heading already shows an argument is checked on its text, never assumed from how a tool builds its subject, so a mismatch shows an argument twice rather than hides it. `PARAMS_UPDATE` keeps the editable form (name, type, required). Values go through `as_written`, never `repr`: it doubled every backslash, so `grep "a\|b"` read as `'grep "a\\|b"'`. Only what a terminal would act on rather than print — ESC, a carriage return, NUL, bidi overrides, a non-breaking space — is escaped, so an argument cannot redraw the question asking about it; the call is printed as `rich.Text`, never a markup string, or `[/]` crashes a markup console and `sed 's:a:b:'` reads as `sed 's🅰b:'`.

**Patterns cover a call part by part** (`is_already_approved`, `approvals/models.py`, shared by stores, runner and CLI). A tool lists what it runs with `approval_parts` (`ApprovalRequest.parts`), and each part must match a pattern: `$ git *` does not cover `git status && rm x`; `$ git *` plus `$ make *` cover `git pull && make`. `run_command` splits on `&&` `||` `;` `|` `&` and newlines, quotes respected; drops comments and redirects to nowhere (`2>&1`, `2>/dev/null`); and skips glue: `cd`, `echo`, `VAR=…` (unless inherited or read by the shell, like `PATH`), `| tail -30`, `| wc -l`, `| grep -i <literal>`. A line it cannot read with certainty (`$(…)`, backticks, `(…)`, unquoted heredocs, unterminated quotes, `$[…]`, a `${…}` beyond a plain name, a program named by an expansion like `$CMD`) has no parts: only an exact grant covers it. Hardline and deny rules check the subject and every part. `X *` covers a bare `X`, never `Xfoo`. No proposal for programs that run their arguments (`bash`, `sudo`, `env`, `xargs`…, by name or by path) or for lines needing two patterns.

`approval_subject` / `approval_pattern` callables see the call's arguments and nothing else: what is approved is what the model asked for. A tool whose risk depends on where it acts takes that place as an argument and checks it when it runs (the browser tools' `url`). A pattern callable returning `None` declines to offer one.

### Decision models

`domyn/agents/decisions/` — a model that answers **typed questions with probabilities** instead of writing text (Jev from TypeSafe, Clef from Cloudflare, which speaks Jev's API on purpose). Three question types: `noul` (true/false), `choice` (2–255 named options), `score` (2–10 ordered levels, numbered **from zero**). Every question in one request is answered in a single forward pass, in ~150 ms.

**It is off unless configured, and off means today's behaviour exactly.** `Agent.decisions` is `None` by default and `NoDecisions` declares `acts = False`, so nothing is asked, nothing is logged per turn, and no code path exists that only runs when a model is present.

| file | what it is |
|---|---|
| `models.py` | `Question` / `Answer` / `Decision`, and `Provenance` — see below |
| `base.py` | `DecisionProvider`: subclasses implement `_decide`, which may raise; `decide` never does |
| `system_one.py` | `SystemOneProvider` (Jev/Clef) and `WorkersAIProvider` (same protocol, `result` envelope) |
| `llm.py` | an ordinary LLM standing in, `logprobs` or `stated` |
| `none.py` | the default |
| `runtime.py` | `DecisionRuntime`, `USE_CASES`, and the rules about when an answer may matter |
| `approvals.py` | the first use case |
| `config.py` | `decision_providers:` and the per-agent `decisions:` block |

**`Provenance` is the type that carries the whole design.** Every threshold is only as good as the number it reads, and that was measured rather than assumed (13 approval commands): the decision model's one wrong answer carried **0.53** against 0.83–0.98 for its right ones, so a 0.8 threshold rejected exactly the wrong one. The same eval against an LLM asked for its own confidence put its wrong answers at **0.78–0.85** — above the same threshold, at the same accuracy. So the provenance travels with every answer (`native` → `logprobs` → `stated` → `none`) and a use case declares the minimum it will act on.

- **`logprobs` on a reasoning model is a trap.** The logprobs array covers the *thinking*, so scanning forward for the first token that looks like an answer finds it inside "I should respond with one word: allow" — measured at 7/13 with confidence 1.00 everywhere, which looks excellent and is noise. The provider scans backwards, and when the alternatives are not observable it **downgrades itself to `stated`** rather than reporting a fabricated 1.0.
- **One question per call when the answer has consequences.** Questions in a batch are not independent: measured, adding four supporting `noul`s beside the approval verdict flipped `rm -rf build/` from `ask` (0.59) to `allow` (0.53), reproducibly, and pulled every other confidence toward the middle. Batching is still right for homogeneous, independent questions.
- **The client counts the tokens, because the server does not say.** A 200k and a 400k character state both come back `200 OK` with `input_tokens: 73728` — truncated to the window, silently. `max_state_tokens` makes an oversized state a refusal rather than a confident answer about half a document.
- **`model` is the name the *server* exposes** (`Cloudflare/clef`, not `clef`). It is configuration, never derived.

**Use cases** (`USE_CASES` in `runtime.py`) are a closed registry: the list of places a model is allowed to have an opinion. Each declares the provenance it needs and why. Approvals declare **two** bars, because they have two directions — allowing skips a human confirmation and needs `logprobs`, denying only adds friction the user can override and runs on `stated`. A name that is not in the registry is OFF with an unreachable threshold, so a typo in a config file cannot quietly run an unreviewed use case.

**Modes, per use case, all `off` by default**: `off` (not asked), `shadow` (asked, recorded, ignored — nothing about the turn changes) and `active`. Configuring a provider switches nothing on. An `active` use case whose backend cannot meet its bar stays in `shadow` and says so once, not once per turn.

"Nothing about the turn changes" is measured against a control rather than asserted: the same four commands through a real runner, a real shell tool and a served Clef, run at `shadow` and with no decision model at all, produce the same outcome for every one — `git status`, `rm -rf /tmp/x` and `cat /etc/hosts` all still ask, `rm -rf /` is still refused by the hardline list, which is what stops it in *both* arms. `active` then differs in exactly the two places it should: the first and last run without asking (0.99, 0.97), while `rm -rf /tmp/x` comes back `allow` at **0.79** and is therefore still asked about — the threshold doing the only job it has.

**The use cases wired up**, each with its own module and its own direction rule:

| use case | module | what it does | may only |
|---|---|---|---|
| `approvals` | `decisions/approvals.py` | judge a gated tool call | allow above `logprobs`, deny above `stated` |
| `skills` | `skills/selectors.py` (`DecisionSelector`) | pick which skills to expand up front | add |
| `instructions` | `instructions/conditional.py` | keep the `##` sections that apply | remove, on a confident no |
| `tools` | `decisions/tools.py` | hide tools the turn does not need | hide |
| `context_ladder` | `decisions/context.py` | mark old tool outputs compaction should drop first | mark for dropping |
| `sensitivity` | `decisions/sensitivity.py` | rate the data a turn handles | refuse a turn |

Three of these turn on **`Answer.is_false`, not `not is_true`**, and the distinction is the whole safety argument: between "confidently yes" and "confidently no" sits *it does not know*, and that case has to keep the user's instruction, keep the tool, keep the output. Reading a 0.45 as "not yes" would delete a line someone wrote on the strength of a coin flip.

**Skills** (`DecisionSelector`, `type: decision`) ask one `noul` per catalogue entry in a single request. This is the case where batching is right — homogeneous, independent questions with nothing to contaminate — unlike the approval verdict, which has to be asked alone. Measured 4/4 at 119-214 ms on a six-skill catalogue with no false positives. The index is never trimmed, so a skill it did not pick is still one `skill_view` away.

**Conditional instructions** let a `##` section declare when it applies, as the first line under its heading:

```markdown
## SQL migrations
when: the task involves a database migration or a schema change

Always check the rollback path before the forward one.
```

A `when:` further down is prose, not a directive. Sections without one are unconditional, which is every section written so far. Resolved **once per user turn** in `Runner._load_turn_instructions` — this text is the cached part of the prompt — and what was left out is recorded on `AgentInstructions.skipped_sections` rather than vanishing. It is conservative in practice: on a live catalogue it kept instructions it could not confidently exclude, which is the direction it was built to err in.

**Tools and the context ladder** are both decided once per turn in the runner and stashed on `context.state`, because both of their consumers are synchronous and sit at the front of the prompt. `get_tool_definitions` filters by the stashed set, and `_prune_irrelevant_first` hands the compaction pruner the event ids to drop before size is even considered — reusing `prune_recorded`, which already existed to replay a past decision. The ladder judges a 400-character **preview** of each output, not the whole thing: the full outputs are what compaction is trying to get rid of. Live: it dropped a `node_modules` listing and kept the failing-test output for a task about that test. Hiding a tool never touches `skill_view`, `stop` or `forward_verbatim`, and a catalogue under twelve tools is left alone.

**Sensitivity** is the odd one out and knowingly so. A provider declares `max_sensitivity: internal`; a turn rated above it is refused before the agent produces anything, like a missing required skill. Live: an AWS key rated `regulated` at 0.97 and a health-data CSV at 0.91, both blocked for a cloud model capped at `internal`. **It is the one use case that does not degrade to today's behaviour**, because here today's behaviour is "send it anyway" — a guardrail that fails open is not a guardrail. An unobtainable rating blocks by default; `on_unknown: allow` is available and visible in configuration. A fractional score rounds **up**: a turn between internal and confidential is confidential. A provider that declares no ceiling is never checked, never asked, and costs nothing.

**Approvals** (`decisions/approvals.py`) are the use case the measurements were taken on. The verdict joins the gate's **hook verdicts**, so it is consulted in the position `ApprovalPolicy` defines and nowhere else: after hardline rules, after deny rules, after a stored grant. A grant returns before hooks are read, so a model can never revoke a permission the user gave, and the only case it ever speaks in is the one whose answer would have been *ask the user*. `Runner._resolve_approval` emits `DECISION_MADE` with the confidence, provenance and **latency** — and never the state, which is customer content.

```yaml
decision_providers:
  - name: local_clef
    type: system_one
    base_url: http://localhost:28080
    model: Cloudflare/clef        # the name the SERVER exposes
    max_state_tokens: 73728

agents:
  - name: Assistant
    decisions:
      provider: local_clef
      approvals: {mode: shadow, min_confidence: 0.8}
      skills: {mode: active}
      instructions: {mode: active}
      tools: {mode: off}
      context_ladder: {mode: off}
      sensitivity: {mode: active, on_unknown: block}
```

A use-case name that is not in the registry is refused at load time, and a
runtime built in code with one is OFF with an unreachable threshold — a typo
must not quietly run an unreviewed use case on a number nobody chose.

**`domyn setup` asks for one**, so the block above does not have to be written by hand. It offers four levels (`DECISION_LEVELS`, `interfaces/cli/config.py`), in order of consequence: `no`, `shadow` (ask and record, change nothing), `active` (skills and instructions act; the gate still only records) and `active, approvals too`. The approval gate is a level of its own because everywhere else a wrong answer changes what the model is *shown* and the turn carries on, while at the gate allowing removes a human confirmation — that is chosen on its own line, not bundled. The wizard writes `system_one` only, and never switches on `tools`, `context_ladder` or `sensitivity`: those stay a deliberate edit. The endpoint is probed with a real `POST /v1/systemone` rather than a socket connect, because the two silent mistakes here — an endpoint that is not up and a model name the server does not expose — look identical to a connect and different to a request; a failure warns and saves anyway.

`/model` runs the wizard `provider_only`, so it never asks and passes `decisions=None`, which `_apply_decisions` treats as "leave it". That is load-bearing: `decision_providers` is otherwise carried across by `_keep_unowned` *because* the wizard does not own it, and owning it unconditionally would hand `/model` back the silent deletion that function exists to prevent.

A configuration mistake raises at load time (unknown type, unknown use case, a `provider:` nothing declares), the same rule `LLMProviderFactory` and `SkillsRuntime` follow. A runtime failure degrades: a timeout, a dead endpoint or a malformed answer all arrive as "I don't know", and after `max_failures` in a row the provider stops asking. The `PRE_TOOL_APPROVAL` phase has **no timeout of its own**, so the provider's `timeout_s` is the only thing between a hung endpoint and a hung turn.

`tests/decisions/test_live_contract.py` runs against a real endpoint and is **skipped unless `DOMYN_DECISIONS_URL` is set** — it needs a live endpoint that is not always available, and a suite that fails when that endpoint goes away is one people learn to ignore.

### Skills

Skills are `SKILL.md` files read from directories: the paths the agent declares under `skills.source.paths` in `~/.domyn/config.yaml` — there is no command-line flag adding one. A skill loads because the config names it — the folders read on their own are `./.domyn/skills` and `./.agents/skills`, and only when setup turned that on (autodiscovery, below). A listed directory is walked by `skills/layout.py:walk_skill_tree`, the same walk setup uses: a folder with a `SKILL.md` is a skill and is not entered, a loose `<name>.md` directly in the directory is one, any other folder is a **group** and is walked into, at any depth — so a skill's path is **not** derivable from its name. Hidden and dependency folders are skipped (inside a listed path; the listed path itself may be hidden) and **no symlink is followed, the listed path included** (`listed_path_is_linked`, `skills/layout.py`): for a relative path every folder on the way to it from the working directory is checked too, for an absolute one only its last component, so a symlinked parent like macOS's `/tmp` is fine. The relative part comes from whatever folder the CLI starts in, possibly a fresh clone, and `.agents -> ~` would otherwise make `.agents/skills` load from, and open the ungated read tools to, `~/skills`. A path may also name a skill's own folder or one loose `.md` file, which loads just that skill (`_iter_skill_files`).

**The builtin skills are installed, not loaded from the package.** `domyn setup` asks which to include (`multiselect`, `interfaces/cli/select.py`; each category folder is a **group** row with its skills indented below, `catalog.skill_groups()`: ticking a group ticks its skills, each skill still toggles alone, and a group's box is derived from them — `_TickList` holds that logic with no terminal, and only skills are ever returned or written to the config) and `skills/catalog.py` copies them into `~/.domyn/builtin-skills` (`skills_install_dir()`, `BUILTIN_SKILLS_DIRNAME`), writing each one's absolute folder path to `agents[0].skills.source.paths`. Decisions behind it:

- **The directory mirrors the package layout** (`<category>/<skill>`, shared files at the category level, `resources/` at the top), so the scripts' `parents[2]` / `parents[3]` lookups and the SKILL.md text hold unchanged. Which files are copied is each skill's own **`external_files:` list** (see the skill format below), not a table in `catalog.py`: `catalog.install_plan(skill)` is the `SKILL.md` plus whatever its entries expand to, and `install_skills` copies the union of the chosen skills' plans, once per destination, so a skill's folder is **not** copied wholesale and a script it does not list is not installed. A list that is absolute, missing a file or written to lead outside `skills/builtin/` raises `SkillInstallError` before anything is touched (judged on the path as written, so the package's own link `domyn-docs/pages` is copied through; a link inside a listed folder is still skipped), and `unrecognized_entries` (what the wizard names before deleting) is built from the same plan so the two cannot disagree. `tests/skills/test_catalog.py` keeps the lists honest: every entry resolves, **no file in a built-in skill's folder is left out of its list** (nor a linked folder, which `rglob` never enters), and a skill whose scripts use `data_utils` or `seaborn_style` lists it.
- **Every setup replaces the directory from scratch** (`install_skills`, built beside it and swapped in, so a failed copy leaves the old one). A deselected skill disappears and a changed one arrives in its new version. That is why `~/.domyn/builtin-skills` is **not scanned and holds nothing of the user's**: user skills belong in `./.domyn/skills`, `./.agents/skills` or a folder given to `domyn setup`. It is not called `skills` so that, launched from `$HOME`, `./.domyn/skills` is never the folder setup rewrites. It refuses a symlinked directory, and the wizard names anything in it that is not a catalog skill (`unrecognized_entries`, by location, never by content, or every upgrade would look like an edit) and asks before deleting.
- **The package is not a fallback, and neither are the project folders.** What loads is the config's paths and nothing else (duplicates dropped by `unique_paths`); a config with no `skills` entry has no builtin skills until `domyn setup` runs again, and `start_session` says so (`_lacks_chosen_skills`, saved config only).
- **The project's own skills: `./.domyn/skills` and `./.agents/skills` autodiscovery, opt-in at setup.** After the builtin step, `domyn setup` asks one question — load skills from `./.domyn/skills` and `./.agents/skills` in the folder domyn runs in? — every time, whatever the current folder holds, since the answer applies to every folder (yes by default on a first setup, the current state on a re-run). One answer covers both. Yes writes the literal paths `.domyn/skills` then `.agents/skills` into `config.yaml`, **before** the builtins so a project skill beats a builtin of the same name, and domyn's own folder beats the cross-tool one; no leaves them out. Nothing is walked or copied by setup. Being relative, the entries are resolved against the working directory **every turn**, so they work from any folder and a skill added to either is loaded on the next turn without running setup again. The step owns every listed path that points into either folder (`_is_local`). `LOCAL_SKILLS_DIRS` (`interfaces/cli/config.py`) names the folders in precedence order and `is_skill_autodiscovery` recognises the entries however they are spelled (`./.agents/skills/`); a plain `skills` entry is an ordinary path. The switch itself is `with_skill_autodiscovery` (`interfaces/cli/setup.py`), shared with **`/skills-autodiscover [on|off]`**, which flips it from inside a session: it rewrites the default config (only that one, like `/model`: a file run by name keeps its comments) and lets `_reload_if_changed` rebuild the agent, so it applies from the next message. This puts a cloned repository's skills into the prompt — that is why it is opt-in; tool calls stay gated as always. `collect_skill_workspace_roots` drops a linked path by the same `listed_path_is_linked` as the loader, so the file tools never reach a folder the loader refused. `discover_local_skills` (`skills/local.py`) remains as the parsed, labelled view of a folder over the same walk.
- **A folder of the user's own skills, asked between the two.** `_choose_custom_skills` asks for one folder (empty = none), walks it once with `discover_local_skills` (the loader's walk: group folders entered, links never followed) and lists **every skill in it on its own**, by **absolute** path (its folder, or its file for a loose `.md`; `~` expanded, `..` collapsed, a link not resolved), first in the list, so the order is `.domyn/skills`, `.agents/skills`, the user's skills, the builtins, anything else. The folder itself is not listed, so a skill added to it later loads only once setup is run again and the folder given again (only the new ones are added). Nothing is copied: each skill is read from where it is every turn. Refused and asked again (three tries, then skipped): a folder that does not exist, a file, a link (the loader would not follow it), one inside `skills_install_dir()`, the current `./.domyn/skills` or `./.agents/skills` (the next question owns them), one the config **already lists**, one with no skills, and one whose skills are all listed already, all compared by resolved path, so another spelling or a listed link is caught; a skill already listed is not listed twice. Entries added on an earlier setup are kept by the other two steps like any other path, so an empty answer changes nothing; removing one is still a hand edit. Each skill's folder becomes a workspace extra root, not the folder above it, so files a skill shares with its siblings there are outside the file tools' reach.
- **Relative means relative to where `domyn run` starts** (the loader resolves it against the working directory). `domyn run` checks the *listed* paths at startup (`_missing_skills_warning`, root agent only, printed even with `--no-banner`) — a check of what the config names, not a scan: a path that is a symlink is named as not loaded ("links are not followed"; a broken link once, as a link), then a path that does not exist as "Skill not found: …" — except the `skills` autodiscovery entry, whose absence is normal in most folders.
- **A skill's `external_files` are checked on every load, and once more for the human at startup.** `evaluate_requirements` marks a skill whose entries are missing or unusable `[unavailable: file '<entry>' (does not exist)]`, as it does for a missing binary, so the model does not pick a skill whose scripts would crash and an `enforcement: required` one refuses the turn; a file restored mid-session counts from the next turn. `discover_local_skills` records the missing ones on each `LocalSkill` for the setup picker, and `domyn run` names them at startup (`_missing_skills_warning`, from the one skill index `_startup_index` loads for the status line and the warnings, so each file is parsed once). It covers every listed skill, a built-in whose installed file was deleted included.
- `domyn skills` and `domyn skill` do not build the agent, so they read the config's paths themselves (`_configured_skill_dirs`, `collect_skill_workspace_roots`). `domyn skill show|remove <name>` (`show <name>#<section>` prints one `##` section, as `/skill` does: both go through `skills_view._show_one`, which raises `SkillNotFoundError` with a message written for either place when the skill or section does not exist or the file cannot be read — `/skill` prints it, `domyn skill show` prints it and exits 1) looks the skill up among those the config loads: the `--project` one when given, else the **global** one (`config_path(prefer_project=False)`, even in a folder with a `./.domyn/config.yaml`, which `domyn run` would use there instead). `remove` unlists the config entry that names the skill — its folder or loose `.md` — and never deletes files; a skill loaded only through a listed folder (`./.domyn/skills` / `./.agents/skills` autodiscovery, or a hand-listed folder of skills) is left alone with a warning saying how to stop loading it. There is no `domyn skill new`.
- `tests/cli/conftest.py` redirects `skills_install_dir` for every test (setup deletes, so a test that reaches the wizard unredirected would delete the developer's real directory) replaces `setup._choose_local_skills` and `setup._choose_custom_skills` with steps that keep the config as it is (both questions are asked every time, and would add prompts to every scripted wizard; `tests/cli/test_custom_skills_setup.py` puts the second back), and points `LOCAL_SKILLS_DIRS` at folders that do not exist (so the autodiscovery entries never load the repository's own `.domyn/skills` or `.agents/skills`); `tests/cli/test_local_skills_setup.py` puts both back.

Two things follow from that, and both were missing until an agent was seen looking for its skills in the current directory:

- Every body handed to the model — by `skill_view` or by expansion — starts with `SkillContent.header()`: the skill's `[file: <path>]`, saying the file can be read and edited. Not the index: there the path was half of every line, paid on every request for skills never opened. The builtin skills name their files relative to it ("the `[file: …]` line above this text"). `/skills` shows the same paths, because "where is this coming from" is the first question when a skill misbehaves.
- The skill directories are passed to `build_agent(config, skill_dirs=...)` and become `Workspace.extra_roots`, so the built-in file tools can actually open them — a directory outside the project workspace (the installed skills in `~/.domyn/builtin-skills`, say) was refused before. The skills `domyn setup` installs also share files (`resources/`, `data/*.py`) that sit in the folders above them, so when the config lists any skill under `~/.domyn/builtin-skills`, `skill_workspace_roots` (`interfaces/cli/config.py`) adds that whole directory — the file tools use it, loading does not (`collect_skill_workspace_roots` stays the list of what is *loaded*; the install directory is never scanned). `Workspace.at` drops an extra root already inside another. Reach was widened, **not** the approval gate: `write_file` / `append_file` / `edit_file` still require `USER_CONFIRM`, keyed on the path, with the directory as the grant unit.

The config's paths, through `unique_paths`, are the single answer to "where do skills live", used by both the source that indexes them and the workspace that decides whether their files can be opened. Those two disagreeing is exactly the bug above.

**Writing one.** YAML frontmatter, then markdown. Only `name` and `description` matter for most skills:

```markdown
---
name: user-journeys
description: How to write a user journey document. Used when asked to draft or review one.
version: "1.0"              # optional, free-form
enforcement: advisory       # or `required` — see below
metadata:
  requires:
    bins: [git, jq]         # binaries that must be on PATH
    env: [GITHUB_TOKEN]     # environment variables that must be set
    os: [darwin, linux]     # platforms it works on
    tools: [read_file]      # agent tools it needs
    external_files:         # files it needs besides this one, relative to THIS file — see below
      - scripts/
      - ../shared_helper.py
---

Body in markdown. `##` headings become **sections**, and a section can be
fetched on its own — the model calls `skill_view(name, section)` instead of
pulling the whole file into context.

## Structure
...

## Review checklist
...
```

What the parser actually does with it (`skills/parser.py`), because several of these are forgiving in ways worth knowing:

- **`description` is what selection runs on.** Omit it and one is derived from the first non-empty line of the body with any leading `#` stripped — so a file starting with `## Structure` gets the description "Structure" — the skill still appears, but it is matched against tasks by a line that was not written for the purpose. Write it, and say *when* the skill applies, not only what it is.
- **`name` falls back to the filename.** So `<dir>/<name>/SKILL.md` and a flat `<dir>/<name>.md` both work without frontmatter — and the path is therefore **not** derivable from the name, which is why the body's header carries it.
- **`requires` is accepted both nested under `metadata:` and as flat top-level keys** (`bins`, `env`, `os`, `tools`, `external_files`; the nested one wins when both are given), because skill files in the wild use both. The built-in skills write `external_files:` flat. Unmet requirements never hide a skill: it stays in the index marked `[unavailable: …]`, so the model knows it exists and why it cannot be used.
- **`external_files` is the `requires` key listing the files the skill needs besides its `SKILL.md`** (`SkillRequirements.external_files`), the only one checked against the skill's own path rather than the machine. External to the file, not to the folder: `scripts/` is an entry like `../shared.py`. Entries are paths **relative to the `SKILL.md`**, forward slashes, `..` allowed; a file, or a folder meaning everything under it (no `__pycache__`), or `.` for the skill's own folder; absolute paths are refused. `skills/external_files.py` (`resolve_external_files`, `expand_files`, `missing_external_files`) is the one place that decides what an entry means. **For a built-in skill the list is what `domyn setup` copies** (below). For any other listed skill it is only **checked**: nothing is copied, a discovered skill stays where it is, and the files must already be there. **A missing or unusable entry makes the skill unavailable**, through the same `evaluate_requirements` as the other four, which for an `enforcement: required` skill refuses the turn; a skill with no `path` (in memory, remote) has nothing to check. Its entries are read more strictly than the other keys (`_paths`, `skills/parser.py`): an empty entry would resolve to the skill's own folder and pass silently, so it is dropped. The model sees the entries only in that unavailable reason. The key was called `required:` before it reached `main`; that name is no longer read.
- **`enforcement: required` is a guardrail, not a preference.** Two things refuse, both before the agent produces anything: a skill whose `requires:` are unmet, checked on the **index** by `raise_if_required_unavailable` (`skills/runtime.py`) at both entry points — `Runner._load_turn_skills` and `SkillsRuntime.select` (`attach_skills` only wires the runtime and checks nothing); and a required skill whose **body** cannot be fetched during expansion. The index check blocks the **whole turn** when **any** agent in the tree trips it, including one the turn would never have reached — which agents a turn reaches is not knowable up front, so a guardrail checked on entry is one the turn can route around by taking another path. The one case that does *not* raise is a skill file that cannot be read at all: `enforcement` is declared *inside* the file, so an unreadable file cannot be known to have declared `required` — it is dropped with a warning, indistinguishable from a skill that was never written. Anything else is `advisory`.
- **Sections are `##` headings only**, deduplicated, in document order.

**How a skill reaches the model**, since the work is split across three layers by how often each has to happen:

| layer | what it does | how often |
|---|---|---|
| `Runner.run` → `_load_turn_skills` | loads every declaring agent's catalogue into `RunnerState.skills[name]`, and refuses the turn with a `RequiredSkillUnavailable` error event if any of them advertises an unavailable `required` skill | **once per turn**, for every agent in the tree |
| `Agent._execute` → `skills/turn.py:build_skills_payload` | runs the selector, expands, renders the block | **once per agent entry** (A → B → A selects twice) |
| `planner.execute(..., skills_payload=...)` | injects it as an assistant message at index 1 | once per LLM round-trip |

**The index budget shortens, never cuts** (`SkillsRuntime.layout`, `skills/runtime.py`). `SkillBudget.max_index_chars` is applied where the index is rendered (`build_payload`), **not** at load: `load()`/`index()` return the whole catalogue, so the selector, `skill_view` and `/skills` see every skill — applied at load, it hid skills from an `LLMSelector` that could never pick them. Over budget, every advisory skill takes the first tier that fits — whole descriptions, descriptions cut to `max_description_chars` (`clip_text`), names only on one line — one tier for all, so which skill keeps its description never depends on its filename. `required` skills are always whole and charged first, as before. Past the names tier, names are listed in order while they fit and the rest counted in the index itself (`+N more skills`), with a warning in the log; the shorter tiers log at info, since the index renders on every agent entry. `IndexLayout` is what the status line of `domyn run` and `/skills` report from.

**`skill_search` reaches what the index shortened** (`skills/search.py`, tool in `skills/integration.py`, attached with `skill_view` by `Agent._attach_skills` and `attach_skills`). A query is required and the answer capped at `MAX_SEARCH_RESULTS` (10) with descriptions clipped, so its cost does not grow with the catalogue — a list-everything mode would just move the index's problem into a tool result, which `max_tool_result_chars` then cuts silently. `SkillsRuntime.find` tries, in order: `SkillSource.search()` (every in-tree source returns `None`; it is for a remote catalogue searched where it lives), a model (`SkillSearch`, the agent's provider unless `search.provider` names one, catalogue **before** the query so repeated searches share a cacheable prefix, `limit + 1` asked to know whether there are more), then words (`rank_by_words`: accents folded, three-letter pieces, name > description > section headings > exact body words). The model is there for languages: a word match cannot meet an Italian skill from an English query, and the result says when it came from words. A reasoning model thinks through the timeout; `search.generation_params` is where its thinking is turned off, since providers spell that differently. The runner hands the turn's `LoadedSkills` to it as to `skill_view`, so a search reads no file.

`SkillSource.load()` is where every in-tree source does its I/O, with `index()`/`fetch()` expressed in terms of it — `LocalDirSkillSource` and `InMemorySkillSource` parse both halves in one directory walk. Expansion falls back to `fetch()` for anything a source did not preload, and the index-only default on the base class remains for third-party sources that cannot do better: it invents no bodies, so every expanded skill is paid for one `fetch()` at a time. `RemoteSkillSource` (`skills/remote.py`) is a **placeholder**: the `type: remote` tag still resolves, but both methods raise `NotImplementedError` — there is no HTTP skill client on this branch. Consequences worth knowing: the load-time `SkillQuery.task` is the *user's* text (a sub-agent's delegated sub-task does not exist yet), while the selector sees the precise per-entry task; and nothing derived is cached on the `Agent`, the planner or `context.state` — all three are shared in ways that would leak one user's skills to another.

`domyn skill show|remove <name>` shows a configured skill or unlists it from the config; `/skills` and `/skill <name>` read them from inside a session.

**The documentation, as a builtin skill.** `skills/builtin/cli/domyn-docs/` holds no copy of the pages: in the repository its `pages` is a symlink to `docs/docs`, which a wheel built from the tree follows, and the sdist force-includes `docs/docs` at that path (an sdist would keep the symlink pointing at nothing), so an installed package has them as files. `domyn setup` copies the pages into the installed skill as files (its `external_files` is `pages/`, and `install_plan` follows the package's own link), so the copy in `~/.domyn/builtin-skills` holds no link. The agent reads the index in `SKILL.md`, then the page, with `read_file` — the installed skill's folder is a workspace extra root. The index is written by hand — title and description of each page's front matter, in `sidebars.js` order — and `tests/skills/test_agent_docs.py` fails, printing the expected lines, when it does not match. The directory is `pages`, not `docs`: `[tool.hatch.build] exclude` drops every directory named `docs`, at any depth. `domyn-cli` beside it keeps only what the pages do not say — how to act on the user's own setup (where files are, editing the config safely, when a change takes effect, troubleshooting) — and sends how-to questions to `domyn-docs`, so nothing is described twice.

**Brand resources** live in `skills/builtin/resources/`, shared by every builtin skill: `palette.json` (every colour, extracted from the Domyn master slides deck — the theme's six accents, text, surfaces, lines, and the slide and page `backgrounds` — and a `charts` section that only arranges those colours: categorical order, scales, grid; a test fails on a chart colour the rest of the palette does not have), `typography.json` (the same for type: the five styles as face + bold flag + file, the type scale, and the style and size of every text *role* both slides and documents have — `title`, `body`, `caption`, `table_cell`…; the slides pass a role name where a size goes, `pdf_layout.json` names a role instead of a font and size, and only what one medium alone has keeps its own size), `assets/fonts/` (DM Sans, OFL), `seaborn_style.py` and `plotly_style.py`. There are no logos and no background images, on purpose: the builtin skills ship to whoever installs the package, so they carry no company's brand or name (`tests/skills/test_builtin_skills_are_unbranded.py`); backgrounds are palette entries, white or a gradient between palette colours, drawn natively by PowerPoint and by reportlab. It is a directory under `builtin/`, copied to `~/.domyn/builtin-skills/resources/` by the skills that list its files, and reachable by the file tools there because `skill_workspace_roots` adds the whole install directory. It holds no skill, so the setup picker shows no group for it: never put a `<name>/SKILL.md` in it (the catalog would offer it as a skill) or a flat `.md` (anything reading `builtin_skill_dirs()` would index it as one). The presentations skill builds decks through a module, `presentations/scripts/slides.py` (imported, never copied; `Deck`, one method per layout, `save()`), on a PowerPoint template, `presentations/template/layouts.pptx`: one slide layout per layout and background variant, so slides are real slides of their layouts (placeholders, outline, "Reset slide"). The template holds geometry only. Every shape names what it takes in its name, `label [key=value; …]` (palette keys, roles, files under `assets/`, `slot=` for placeholders), and `slides.inject()` writes the resources in when a `Deck` opens it, theme colours and fonts included, refusing an untagged shape or an unknown key or file. Every layout tags its own name too, `Content · navy [background=navy]`: `inject()` writes that background in as the layout's real background (`p:bg`, what Format Background sets) and strips the tag from the name, so `Deck.layouts` and a delivered deck's layout picker see the label only; the template holds no picture. Two things it does on purpose: every placeholder paragraph the module writes states its own indent, spacing and bullet (`_own_paragraph`), because the master body style carries bullets and 90% spacing and the collision check reads only what a slide itself says; and `save()` writes `<name>.unchecked.pptx`, checks it, and only on a pass renames it to the name asked for, removing a stale one first, so a deck that failed is never at the deliverable name. `scripts/check_pptx.py` walks each slide's layout shapes (cards, rules) under the slide's own. So does the documents skill (`productivity/documents/`, PDF only so far), through its module `scripts/pdf_layout.py` (reportlab, imported from the skill by absolute path, never copied out: it finds `resources/` at `parents[3]`): cover, content pages and closing in the deck's style, DM Sans **embedded** (unlike a deck, which only names its fonts), every colour from `PALETTE` by key. One thing it does on purpose: a heading is kept off the bottom of a page by a `CondPageBreak` that `build()` inserts before it, not by `keepWithNext`, which tied it to a whole long table and emptied the page. Its backgrounds are the same palette gradients drawn on the canvas, so a document embeds no picture of its own. `scripts/check_pdf.py` (pymupdf) is the required last step: it reports text on text, text on a non-background image, text crossed by a stroked line and text past the layout's margins, measured on glyphs (cap height to descender, not line boxes). Two pymupdf behaviours it works around: a span joins whatever shares a baseline and font ("10" printed into "Step ten" is one span, so spans are cut where a glyph starts before the previous ends), and characters wholly off the page are dropped even with `TEXT_MEDIABOX_CLIP` off (so the mediabox is widened in memory before reading).

**Charts in the builtin skills** come in two kinds, one module each in `skills/builtin/resources/`, and neither needs a browser. **Static charts** (every PNG: reports, documents, slides) are drawn with seaborn on matplotlib through `seaborn_style.py`, which calls `seaborn.set_theme` with the house rcParams **on import** and adds what rcParams cannot hold: `figure()` (size in CSS pixels at 96 per inch, written at 2×, constrained layout), `title()` (the chart's title is the figure's suptitle, left-aligned; an axes title is a panel's), `legend()` (attached to the axes above the plot, so the layout makes room for it under the title — a figure legend in the top margin collides with the suptitle; on a titled panel it goes right) and `save()` (PNG only). Text sizes are `typography.json`'s roles in points and the font is DM Sans **registered from the shipped TTFs** (`font_manager.addfont`), so a PNG is in DM Sans whether or not it is installed — and a chart drawn at the size it is placed at (642 px for an A4 document's text width, 901 × 298 for a slide's content area) has the slides' and documents' type sizes exactly. `text.parse_math` is off, or `$1M to $3M` renders as a formula. **Interactive charts** are `plotly_style.py`: the Plotly `house` template, made the default on import, and `save()` writing a self-contained HTML page (plotly.js inline, ~4.8 MB) — never `write_image`, which is what needed kaleido and Chrome. The `charts` skill teaches the first, `interactive-charts` the second, and the data skills draw PNGs only: an interactive version is asked for and drawn afterwards with `interactive-charts` (the scripts' former `--interactive-chart` option is gone, since one figure can no longer give both). Neither module has a colour of its own: every one is read from `palette.json` beside it (categorical order: the template theme's six accents, the four that stay distinct under colour blindness first — `#1A57FF` and `#6038FF` are nearly identical to deuteranopes — hence at most four series per bar/line chart), so charts and slides cannot drift apart — and each refuses to import without its files next to it. Scripts find `seaborn_style` by their own place in the tree (an installed copy keeps that place) — each skill's `common.py` inserts `parents[3] / "resources"` at `sys.path[1]`, so a copy next to the script (a run folder) still wins but nothing on `PYTHONPATH` can stand in for it — and chart scripts must `import common` before `import seaborn_style` (isort keeps that only because of the names; `tests/skills/test_seaborn_style.py` guards it, and that no file outside `interactive-charts` imports plotly). `data_utils.record_run` copies `seaborn_style.py`, `palette.json`, `typography.json` and the fonts (into `fonts/`, where the style looks after `assets/fonts/`) into every run folder. Seaborn's own defaults that break the palette are overridden per call, not globally: `saturation=1` on bar and box plots (they desaturate otherwise), `errorbar=None` on bar plots. The CLI's own accent (`#1f6dff`, `interfaces/cli/theme.py`) is separate, picked for terminal legibility.

### The Domyn platform is optional

domyn-agents runs standalone, and **no platform logic, route or endpoint lives in this repo**. What a delegate agent needs from the platform — the tools it can call (`list_delegate_tools_for_channel`), the stored configuration (the approval toggle), the relay's WebSocket URL and headers, and the language models of a space (`DomynProvider`) — comes from `domyn-platform-sdk` (`api.agents`, `api.relay`, `api.llm`), which owns the paths, headers, host rules and URL overrides. The SDK is **not a dependency or an extra** of this package: it is distributed with the platform, so `uv.lock` cannot resolve it and a declared extra would break every plain `uv run`. It is installed next to domyn-agents (`uv pip install "domyn-platform-sdk>=0.37.0"`, see CONTRIBUTING.md); a later `uv sync` removes it, `uv sync --inexact` and `uv run` keep it. Once it is published, it can become the `platform` extra again.

`domyn/agents/integrations/domyn_platform_sdk.py` is **the only module that imports it**, lazily; a call without the SDK, or with one older than `MIN_SDK_VERSION` (the one place the version is written, checked with `importlib.metadata`), raises `PlatformSDKNotInstalledError` naming the requirement and the install command. Everything else depends on the neutral seam in that module: `ToolCatalog` (a protocol: `list_tools(agent_name=)`), `PlatformSettings` (api key, space, channel, base URL, configuration id, read from the `DOMYN_*` variables) and `relay_connection()`. The SDK is async-only; `run_sync()` is the bridge for the two synchronous callers (`get_platform_tools`, the hermes plugin), safe inside a running event loop. `fetch_delegate_tools`, `populate_delegate_tool_schemas` and `get_platform_tools` take a `catalog=` to inject another source. `domyn serve`/`domyn connect` attach every undeclared platform tool (`--platform-tools`) **by default when the SDK is installed** (`platform_tools_enabled`, which only looks for the module); `--no-platform-tools` turns it off, and an explicit `--platform-tools` without the SDK reports the missing SDK instead of skipping silently.

Tests use the `platform` fixture (`tests/fake_platform.py`), which replaces the catalog, so none needs the SDK installed. When the platform changes, the change is in the SDK, not here.

### Decorator-Based Registration

Components can be registered using decorators:

```python
from domyn.agents.core.decorators import tool, agent, llm_provider

@tool(name="my_tool", description="Tool description")
async def my_function(arg: str) -> str:
    return f"Result: {arg}"

@agent(name="MyAgent", description="Agent description")
class MyCustomAgent(Agent):
    pass

@llm_provider(name="my_llm", model_name="model-name")
class MyLLMProvider(BaseLLMProvider):
    async def generate(self, messages, **kwargs):
        # Implementation
        pass
```

### YAML Configuration

Agents can be defined via YAML configuration:

```yaml
version: "1.0"
entrypoint: AgentName

llm_providers:
  - name: provider_name
    type: vllm
    model_name: model/name
    url: http://llm-url/v1/chat/completions

planner_strategies:            # optional: an agent naming none gets tool_use_thought
  - name: tool_use_thought
    type: tool_use_thought
    max_iterations: 10

hooks:
  - name: log_agent_start
    path: path/to/hooks.py
    type: listener
    trigger: pre_agent_call
  - name: validate_args
    path: path/to/hooks.py
    type: updater
    trigger: pre_tool_call

tools:
  - name: tool_name
    type: function
    path: path/to/file.py
    hooks:
      - validate_args

agents:
  - name: AgentName
    description: Agent description
    instruction: System instruction
    llm_provider: provider_name
    planner_strategy: tool_use_thought
    tools:
      - tool_name
    hooks:
      - log_agent_start
```

Any string value in the file may reference an environment variable with `${VAR_NAME}` syntax (e.g. `api_key: ${VLLM_API_KEY}`), resolved once against `os.environ` when the file is loaded (`load_from_file`, `domyn/agents/loader/__init__.py`) — so secrets never need to be typed literally into a YAML file. This is unrelated to, and simpler than, the interactive CLI's `api_key_env` field (`interfaces/cli/config.py`), which names an env var to resolve at config-build time rather than substituting into arbitrary text; both end up doing an explicit `os.environ` lookup rather than storing a secret. An unset referenced variable raises immediately rather than silently loading the literal `${VAR_NAME}` string as a credential.

All hooks live under the single top-level `hooks` key. Each entry **must** specify a `trigger` field:

| `trigger` value   | Phase                                                    |
|-------------------|----------------------------------------------------------|
| `pre_agent_call`  | Before an agent executes                                 |
| `post_agent_call` | After an agent completes                                 |
| `pre_tool_approval` | Before the approval gate — the only phase that can influence it |
| `pre_tool_call`   | Before a tool is invoked (already past the approval decision) |
| `post_tool_call`  | After a tool returns                                     |
| `on_agent_event`  | On every event emitted or received by a specific agent   |

> **Note on `on_agent_event` hooks**: these hooks are attached to individual agents (via the agent's `hooks` list). They fire on the initial user input directed at that agent, on every event yielded by the agent's iterator, and on TOOL_FEEDBACK returned to that agent. When the active agent changes during a handoff, the new agent's `on_agent_event` hooks take over. Updater hooks receive a `BaseEvent` and return a new `BaseEvent` (returning `None` passes the event through unchanged); listener hooks run concurrently and cannot modify the event.

Hooks loaded by `reference` from the registry still require a `trigger` so the loader knows which registry bucket to look in:

```yaml
hooks:
  - name: my_hook
    reference: my_hook   # loaded from the HookRegistry
    trigger: pre_agent_call
```

Load YAML config with:
```python
from domyn.agents import load_application_from_file
agent = await load_application_from_file("config.yaml")
```

### Hooks System

**One decorator**, `@hook(trigger=...)`, for every phase. There is no
`@agent_hook` / `@tool_pre_hook` / `@tool_post_hook` — the phase is an argument,
not a different decorator.

```python
from domyn.agents.core.decorators import hook
from domyn.agents.hooks import HookPhase

@hook(trigger=HookPhase.PRE_AGENT_CALL, is_listener=True)
async def log_agent_start(context, agent):
    print(f"Agent {agent.name} starting")

@hook(trigger=HookPhase.PRE_TOOL_CALL)          # updater by default
def validate_tool_args(tool, tool_args):
    tool_args["query"] = tool_args["query"].strip()
    return {"new_tool_args": tool_args}          # replaces the call's arguments

@hook(trigger=HookPhase.POST_TOOL_CALL)
def transform_result(tool, result):              # result is a ToolResult
    result.data = result.data.upper()
    return result                                # or {"new_result": result}
```

The parameter names are fixed per phase (`FunctionHook.phase_params`) and checked
when the hook is defined: `(tool, **kwargs)` is refused for missing `tool_args`.
Return values are read by `FunctionHook.process_result`: a dict changes something
only through `HookResult` field names (`new_tool_args`, `new_result`, …); any other
dict, such as `{"result": ...}`, is taken as content to emit and the arguments or
result pass through unchanged.

Phases (`HookPhase`, `hooks/base.py`):

| phase | when |
|---|---|
| `PRE_AGENT_CALL` / `POST_AGENT_CALL` | around an agent's execution |
| `PRE_TOOL_APPROVAL` | **before the approval gate** — the only phase that can influence it |
| `PRE_TOOL_CALL` / `POST_TOOL_CALL` | around a tool call |
| `ON_AGENT_EVENT` | on every event emitted by or sent to one agent |

`PRE_TOOL_APPROVAL` exists because `PRE_TOOL_CALL` is too late: it runs inside
tool execution, by which point the decision to execute has already been taken.
A hook that needs to allow or deny a call has to be on the earlier phase. It
returns an `ApprovalVerdict`, bare or as `{"approval": verdict}`, or nothing to
leave the decision to the next hook and then to the human.

Two kinds, chosen with `is_listener`:

- **Updater** (the default) — returns a value that replaces what it was given.
- **Listener** — observes only, runs concurrently, cannot modify anything.

Other `@hook` arguments worth knowing: `on_event_triggers` narrows an
`ON_AGENT_EVENT` hook to specific event types, and `break_on_error` (default
`True`) decides whether a failing hook stops the run.

### Event-Driven Execution

The `Runner` executes agents and emits events:

```python
from domyn.agents import Runner
from domyn.agents import BaseEvent, ExecutionEventType, Part

runner = Runner()
async for event in runner.run(
    application_name="app_name",
    user_id="user_id",
    root=agent,
    user_input=input_event,
    conversation_id="conv_id",
):
    if event.event_type == ExecutionEventType.RESPONSE:
        # Handle agent response
    elif event.event_type == ExecutionEventType.TOOL_START:
        # Handle tool invocation
    elif event.event_type == ExecutionEventType.TOOL_END:
        # Handle tool result
```

Event types:
- `USER_INPUT` - User input received
- `RESPONSE` - Agent generates response
- `RESPONSE_TOKEN` - One chunk of a streamed response
- `TOOL_START` / `TOOL_END` - Tool execution lifecycle
- `AGENT_START` / `AGENT_END` - Agent handoff lifecycle
- `LLM_START` / `LLM_END` - LLM invocation lifecycle
- `DECISION_MADE` - A decision model was asked something (see Decision models)
- `CONTEXT_COMPACTION_START` / `CONTEXT_COMPACTED` - The conversation is being shortened
- `COMPACTION_REQUESTED` - An *input* event asking for a compaction now (see Compaction)
- `ERROR_OCCURRED` / `TOOL_ERROR` - Error events

### Streaming

Controlled per agent by `Agent.stream` (the CLI sets it from `agent.stream` in `config.yaml`, default on). Providers implement `generate_streaming` alongside `generate`.

**What the consumer sees, and the trap in it.** With streaming on, the answer arrives as a sequence of `RESPONSE_TOKEN` events and the runner then **suppresses the final `RESPONSE`** — it would carry the same text again, in full. So a consumer that only handles `RESPONSE` renders nothing at all under `stream=True`. Three exceptions where `RESPONSE` is emitted even while streaming, all because no token sequence preceded it:

- the agent is not streaming;
- the event carries `FORWARD_VERBATIM` metadata from the user-facing agent;
- the event needs feedback — a question to the user is a single direct yield, and dropping it would lose the question.

`BaseEvent.is_partial` marks an event that will be followed by more.

The CLI renderer (`interfaces/cli/renderer.py`) prints tokens as they arrive and must not print the trailing `RESPONSE` again. Its own documented ordering trap: closing the stream is what clears the "a stream was open" flag, so the flag has to be read **before** closing, or the answer appears twice.

**Markdown in the CLI.** On a terminal that takes colour, the renderer turns the answer and the reasoning into rendered markdown through `interfaces/cli/markdown.py` (`MarkdownStream`), built on `markdown-it-py` — already a dependency of `rich`, so nothing new is installed. Anywhere else (a pipe, a file, `NO_COLOR`) the source is printed as it came, so `domyn run … > out.md` keeps what the model wrote. Decisions worth knowing before changing it:

- **A line at a time into the scrollback, never repainted there.** A full-screen interface can re-render the unfinished tail on every token because it owns the screen; this CLI prints into the scrollback, and under the steering editor a write without a trailing newline is erased. So a line is printed when its newline arrives; a long paragraph goes out a visual line at a time as it overflows, but only up to its first delimiter that is still open (`**bold` stays held until the closing `**`). A delimiter still open when its line ends is printed as written rather than holding the line for the next one. A table is held until its last row, since its columns cannot be sized earlier. The invariant `tests/cli/test_markdown.py` pins: fed one character at a time or all at once, the screen is identical. The tokens go through the same `render()` path as plain text, so the spinner goes quiet once the answer starts (`Spinner.streaming()`), as it does without markdown: no clock under an answer being written, even while a table or a long line is held.
- **The unfinished end is drawn by the editor.** While a turn runs, the steering editor repaints its own region above the prompt anyway, so the part not final yet — the line being written, a table being collected — is drawn there and replaced on every repaint, the way Claude Code or aider stream: the scrollback gets final lines only, the region gets the rest. `MarkdownStream.preview()` is what `finish()` would print if the answer ended now, worked out on a copy (it never changes what the stream prints; an open code block shows no closing fence). `EventRenderer.preview()` turns it into ANSI for the editor (`InputSession.ask(preview=…)`, drawn first in `_live_prompt`), capped at half the window (at least `PREVIEW_ROWS`), hyperlinks stripped (prompt_toolkit's ANSI parser would print OSC 8 as text). A token asks the editor to repaint (`on_preview` → `InputSession.refresh`, at most 30 times a second); the editor's own timer is 10. **The handover is the subtle part:** printed rows go through prompt_toolkit's stdout proxy, which writes from a thread up to `sleep_between_writes` later, while the editor keeps repainting. Dropped from the preview at print time, a line was on screen nowhere for up to ~20 ms (measured in a pty with pyte, about one line in four in a fast list); dropped per call, the first of two rows was drawn twice. So the proxy counts characters in and characters written (`OutputProgress`, `shared_stdout(progress)`, the count raised *inside* the write, before the redraw that follows it), and each printed row stays in the preview, with its own mark, until the proxy has written it. Without an editor (a pipe, `DOMYN_NO_STEERING`) there is no preview and the stream is line by line as above.
- **The look.** Headings lose their `#` up to level two, bullets and quote bars are dimmed, code keeps its ``` fences (dimmed) with the body indented two columns and syntax-coloured, tables get drawn borders, and a link shows its address in brackets when the text differs from it. A code line wider than the window is not broken: the terminal folds it, so a pasted command stays one line.
- **Colour stays semantic** (`theme.py`): structure by attributes only, one accent (`md_accent`, `#00875f`, measured ≥4.5:1 on both black and white) for inline code and links, and a named-ANSI syntax palette (`Styles.syntax`) so the terminal's own palette keeps code legible on light and dark backgrounds.
- **Reasoning arrives whole, not streamed.** The providers that stream reasoning — vLLM, custom, Cerebras and OpenRouter — accumulate its deltas and hand over the complete text once, in a chunk of its own when the answer starts (`reasoning_sent`), so the renderer formats it in one go. Streaming reasoning token by token or line by line is an open item; it would reuse `MarkdownStream` the way the answer does.

### Context and Memory

`InteractionContext` (`core/context.py`) is what an agent is given for one turn.
Its fields, since a stale list here sends people looking for attributes that do
not exist:

| field | what it holds |
|---|---|
| `conversation_id`, `user_id` | who and which conversation |
| `interaction_id`, `turn_id` | this interaction, this turn |
| `agent_input` | the event that started the turn |
| `conversation_history` | prior events **visible to this agent** (filtered by `context_visibility`) |
| `interaction_history` | events produced within the current interaction |
| `handoff_targets_description` | agents this one may hand off to |
| `visible_agent_names` | agents visible in its history |
| `state` | free-form dict, **persisted per conversation** by the runner |
| `run_config` | read-only config, gated by `VisibilityScope` |
| `is_subagent` | whether this agent was called by another |

**`state` is the durable scratchpad**, and the one to reach for. The runner writes
it back after every event, so anything left there survives into the next turn's
context — which is how summary caches and compaction records outlive the turn
that produced them. Keys in use: `summaries`, `compaction_records`,
`context_usage`.

```python
context.state["my_key"] = "value"          # survives to the next turn
value = context.state.get("my_key")
```

**Long-term memory** is a separate thing, attached to the *agent* rather than the
context: `Agent.memory_manager`, a `BaseMemoryManager`
(`memory/managers/`). Two implementations, `FilesystemMemoryManager` and
`RemoteMemoryManager`, and two modes (`MemoryExecutionMode`):

- `AUTOMATIC` — the manager writes and recalls memories around the turn by itself.
- `AGENTIC` — `remember` is exposed to the model as a tool, so the agent decides
  what is worth keeping.

### Tool Types

**Function Tools** — a plain callable; the docstring becomes the description and
the annotations become the schema.

```python
from domyn.agents import FunctionTool

def my_tool(param: str) -> str:
    """Tool description from docstring."""
    return f"Result: {param}"

tool = FunctionTool(func=my_tool)
```

**Streamable Function Tools** — an async generator, for output that arrives in
pieces. Same registration, `type=ToolType.STREAMABLE_FUNCTION` via the decorator.

```python
from domyn.agents import StreamableFunctionTool

async def streaming_tool(param: str):
    """Async generator tool."""
    for chunk in process_stream(param):
        yield chunk

tool = StreamableFunctionTool(func=streaming_tool)
```

**Built-in tools** — the file and shell bundles, bound to a `Workspace`. See
**Built-in Tools and the Workspace** above.

**MCP Tools** (Model Context Protocol) — an agent connects to a remote MCP server
and its tools are discovered automatically, powered by `fastmcp`. A tool's text
comes back whole: it used to be cut at 10 000 characters before the result became
an event, so a Jira listing reached the model, the history and storage as its
first two issues, with nothing left to read the rest from. Fitting a result to
the model is the context manager's job (`max_tool_result_chars`), and the part it
leaves out is read back with `read_tool_result`.

```python
from domyn.agents import MCPTool

mcp_tool = MCPTool(
    server_url="https://api.example.com/mcp",
    tool_name="search",
    use_oauth=False,                              # or OAuth
    headers={"Authorization": "Bearer token"},    # or header auth
)
agent.tools.append(mcp_tool)
```

The usual route is YAML: declare servers under `mcp_servers` and reference them by
name from an agent. Options are `name`, `url`, `timeout` (30.0), `use_oauth`,
`oauth_provider`, `headers` (OAuth and headers being the two alternative ways to
authenticate) and `tool_autodiscovery` (default `true`; set it `false` to keep a
server's tools out of the fetch). See `MCPServerStreamableHTTPConfig` for the full
set. The same `use_oauth` / `oauth_provider` / `headers` options exist on `type:
mcp` entries under a top-level `tools:` list too, for declaring individual tools
instead of a whole server (`MCPTool.from_config`, `tools/mcp/mcp_tool.py`).

**OAuth configuration itself lives in a separate, named `oauth_providers:` section**,
not inline on `mcp_servers`/`tools` entries — referenced by name via `oauth_provider:
<name>`; omit it and a shared default provider is used. Each entry
(`OAuthProviderConfig`, `tools/mcp/oauth_providers.py`) has `name`, `type` (`local`,
the default: a `FileTreeStore`-backed token cache under `~/.domyn/mcp_oauth/`; or
`in_memory`: in-process only, nothing written to disk, tokens gone on
restart), `scopes`, `client_name`, `callback_port` (the local OAuth callback
port — see below), and optionally a pre-registered `client_id`/`client_secret`
(skips Dynamic Client Registration when set). A named entry is a *template*,
not a shared live instance — an `OAuth` client is bound to one server URL at
construction, so the loader builds one instance per `(provider_name,
server_url)` pair actually referenced, letting tools/servers that name the
same provider *and* point at the same server genuinely share one OAuth
session instead of each authenticating independently.

`callback_port` defaults to `DEFAULT_OAUTH_CALLBACK_PORT`, one fixed constant
(not derived from `server_url`), so `http://localhost:<port>/callback` is
knowable and registrable with an OAuth app before any config exists. Every
provider therefore shares one port unless a config explicitly overrides it,
and MCP tool/server initialization runs concurrently (`asyncio.gather`, both
within `Agent.initialize_mcp_tools()` and across sub-agents in the loader),
so a whole authorization round trip — opening the browser tab and waiting
for *its own* redirect — is serialized per port by patching
`OAuthClientProvider._perform_authorization_code_grant` (keyed by the port
in the flow's own registered redirect_uri, `_lock_for_port`). Locking only
the local server's bind was tried first and rejected: that method opens the
browser tab before waiting for the callback, so with only the bind locked,
multiple tabs could still be open at once, and whichever one the user
completed first could land on a different flow's listener — a genuine
`state` mismatch, not merely the bind-collision crash (uvicorn turns a bind
collision into an uncaught `SystemExit`, since `Server.startup()` calls
`sys.exit(1)` on failure, capable of aborting the whole process). Locking
the whole round trip means: first-time (or refresh-failed) auth across
servers sharing one port happens strictly one at a time, not concurrently —
the necessary trade-off for a fixed, pre-registrable, shared-by-default
redirect_uri.

**Who calls `initialize_mcp_tools()` depends on how you loaded the agent**, and
this is easy to get backwards:

```python
# Loaded through the loader: already done for you (mcp_init=True by default).
agent = await load_application_from_file("config.yaml")

# Pass mcp_init=False to skip it — e.g. to avoid network at construction time.
agent = await load_application_from_config_new(config, mcp_init=False)

# Built by hand from a parsed config: nobody has called it, so you must.
config = ConfigurationV1.from_yaml("config.yaml")
agent = config.agents[0]
await agent.initialize_mcp_tools()
```

Calling it twice is not a problem, but assuming the wrong one leaves an agent
with no tools and no error to explain why.

A working example, kept in the repo so it cannot go stale the way a snippet
here would: `examples/yaml-and-mcp-vllm/config.yaml` (a whole server under `mcp_servers`, its
tools fetched by the loader, run with `domyn run`).

### LLM Providers

Built-in provider types (registered in `LLMProviderFactory`, `domyn/agents/llm/__init__.py`):
- `VLLMProvider` - vLLM backend. It sends no `truncate_prompt_tokens`, so vLLM's own default applies: a prompt that does not fit is rejected, and the tool-use planners recover from that (see Compaction). `-1`, which it used to send by default, makes vLLM cut the prompt's **beginning** — system prompt, tool definitions — and answer as if nothing had happened. Opt in explicitly only where that beats an error. It posts to its `url` verbatim; only `domyn setup` completes a server's base URL (`http://host:8000`, `…/v1`) to `/v1/chat/completions` before saving it (`vllm_chat_url`, `interfaces/cli/config.py`), so services building the provider in code are unaffected.
- `DomynProvider` (`domyn`) - the models of a Domyn platform space, reached through `domyn-platform-sdk` (`api.llm`) with the user's own platform API key, space and configuration (`DOMYN_API_KEY`, `DOMYN_SPACE_ID`, `DOMYN_BASE_URL`, `DOMYN_CONFIGURATION_ID` fill what the config leaves out; `url` is read as `base_url`). It goes through the seam in `integrations/domyn_platform_sdk.py` (`chat_completion`, `stream_chat_completion`) like every other platform call, so it needs the SDK installed. **In the interactive CLI it is the first provider `domyn setup` offers and the default of a fresh `ProviderConfig`** (`PROVIDER_HINTS` order is the menu order); the wizard asks its platform url, space id and configuration id first, settles the key, then lists the space's models (`api.llm.list_models`, through the seam's `list_models`) in the arrow-key picker with an "other" row to type a name; with no key, url, space or SDK, or when the lookup fails, it says so and asks for the name.
- `OpenAIProvider` - OpenAI API
- `AnthropicProvider` - Anthropic API
- `CerebrasProvider` - Cerebras API
- `OpenRouterProvider` - OpenRouter API
- `CustomLLMProvider` - generic OpenAI-compatible endpoint
- `DelegateLLMProvider` - marker for runner-to-runner delegation (not called directly)

All providers talk raw HTTP via `httpx` — no vendor SDKs are declared as dependencies.

There is **no** mock/fake provider in the package: tests define local mock providers as fixtures.

Providers implement `BaseLLMProvider` interface with `async def generate(messages, **kwargs) -> LLMResponse`.

### Transcription

`domyn/agents/audio/` — speech to text, the base of the voice features. A `Transcriber` (`audio/base.py`) registers by its `type` tag like skill and instruction sources, and `resolve_transcriber({"type": ...})` builds one from configuration, raising on an unknown type. `transcribe(audio, filename=None) -> Transcript` (`text`, `segments` with times, `language`, `duration`); every failure is a `TranscriptionError` with a message for the person who has to fix it, the audio and response bodies at debug only.

- `LocalTranscriber` (`local`): onnx-asr with Parakeet TDT 0.6B v3 int8 and Silero VAD, from the optional `audio` extra (not installable on Python 3.15: no onnxruntime wheel). **Nothing is downloaded or loaded before the first `transcribe` or `prepare()`**, which is decided, not incidental: installing the extra or building the object costs nothing. The model goes to `~/.domyn/models/<name>-<quantization>`, downloaded into a `.partial-*` sibling and renamed when complete (onnx-asr downloads into a directory that does not exist and reads one that does, so an existing directory must always be a whole model). Runs on `CPUExecutionProvider` only: CoreML was slower in every measure on Apple Silicon and noisy on stderr. Non-WAV audio goes through `ffmpeg` (`audio/convert.py`), looked up only when needed.
- `OpenAICompatibleTranscriber` (`openai`): multipart `POST` to `/v1/audio/transcriptions` (OpenAI, vLLM, gateways), audio sent as it came, `verbose_json` by default. Written against the contract and tested with `respx`.

**Push-to-talk** (`interfaces/cli/voice.py`, `/voice` or `domyn run --voice`): the editor binds the space bar on an empty line (and every space while voice is busy, never while an approval question is open) to `VoiceInput.on_space`. A terminal reports presses, never releases, so a hold is read from the key repeat: spaces closer than `REPEAT_GAP` are repeats, the recording ends `release_delay` after the last one; no repeat within `TAP_WINDOW` makes it a tap (press again to send), and a later press is confirmed as a press by no repeat following it. The kitty keyboard protocol would report releases, but only with every key sent as an escape sequence, which prompt_toolkit cannot parse. The transcription is put in the editor running at that moment and submitted (`buffer.validate_and_handle()`), so the REPL sees a typed line and nothing downstream knows it was spoken; a read that ended before the transcription did (an interrupted turn) hands it to the next one through `pre_run` → `attach`. A space during a turn exits the alongside read with `KeyboardInterrupt`, the existing "stop the turn" path, without arming the double Ctrl-C. The microphone is first opened by `Recorder.probe` in a thread at `/voice`, because that is when macOS asks for permission and it blocks: in a key handler it would freeze the editor. `voice.transcriber` in `~/.domyn/config.yaml`, default `{"type": "local"}`. `tests/conftest.py` points the default models directory at a temporary one and sets `HF_HUB_OFFLINE`, because a REPL test reaching `/voice` once downloaded the real model into `~/.domyn/models`.

The chat channels (Telegram, Slack, WhatsApp) take a `transcriber` and put a text `Part` after every audio `Part` (`channels/_voice.py`): the transcription, prefixed `[Voice message, transcribed]`, or a line saying why there is none (no transcriber, failure, silence), so the agent can say so instead of answering nothing. No context manager renders a file `Part` to a model, which is why the text is needed at all. Each channel passes the headers its downloads need (Telegram's URL carries the token; WhatsApp and Slack take a Bearer header); downloads are capped at 25 MB. `domyn serve --channel multi` reads `serve.transcriber` and gives it to every entry without its own. Channel attachments are `File(file_id=<url>)`: the field was renamed from `uri` in #486 and the channels kept building `File(uri=...)`, which failed validation, until this was fixed alongside.

`import domyn.agents.audio` must not import `onnx_asr`/`onnxruntime` (`tests/audio/test_resolve.py` checks it); the engine is imported inside `LocalTranscriber._load`.

## Key Directory Structure

The distribution is `domyn` (`pip install domyn`; the repository stays `domyn-agents`), and the package is `domyn`, with this framework as its first subpackage, `domyn.agents`. Further modules (an SDK, say) are added beside it as `domyn/<name>/`.

`domyn/__init__.py` imports nothing and holds `__version__`. The tree below is `domyn/agents/`:

```
domyn/agents/
├── agents/              # Agent, WrappedAgent, DelegateAgent
├── core/                # Types, events, decorators, context, ApprovalType; documents.py reads any file as text
├── scripts/             # Programs the package runs in a process of their own (the PDF reader)
├── runner/              # Execution loop and event emission
├── planner_strategy/    # Planners; context_management/ holds compaction and budget
├── llm/                 # Providers, all raw httpx
├── audio/               # Speech to text: Transcriber, local (onnx-asr) and OpenAI-compatible
├── tools/               # Tool types; builtin/ holds the file and shell tools
├── skills/              # SKILL.md discovery, selection, injection
├── instructions/        # SOUL.md / AGENTS.md discovery, into the system prompt
├── approvals/           # Whether a tool call needs a human first
├── decisions/           # Typed questions to a decision model (Jev/Clef), off by default
├── hooks/               # Lifecycle hooks
├── registry/            # The registries behind the decorators
├── loader/              # YAML/config loading
├── memory/              # Memory providers
├── conversation_manager/# Conversation persistence (base, in_memory, sqlite)
├── interfaces/          # cli/ (the REPL and everything it draws), plus channel
│                        # and adapter plumbing
├── cli/                 # Typer entrypoint and one subpackage per command
├── channels/            # Slack, Telegram, WhatsApp, relay
├── adapters/            # Foreign-framework agents (langgraph, domyn_agent)
├── plugins/             # External plugins: placeholder, support still to be added (TODO)
├── integrations/        # Host integrations (hermes gateway)
├── tracing/             # Tracing hooks
└── logger/              # Logger injection (`get_logger` / `set_logger`)

examples/                # Example agents
tests/                   # Test suites, mirroring the package layout
```

Two of these are easy to trip over. **`cli/` versus `interfaces/cli/`**: the first is argument parsing, the second is the session itself, and most CLI behaviour is in the second. And **`planner_strategy/context_management/`** carries two unrelated jobs — choosing which history becomes messages, and keeping it inside the model's window.

## Type Checking

The project uses strict mypy type checking:
- All code must have type annotations
- `disallow_untyped_defs = true`
- Tests and examples are exempt from strict typing

## Important Patterns

1. **Avoid circular imports**: Use `TYPE_CHECKING` blocks for type-only imports. The codebase carefully manages imports between `agents`, `planner_strategy`, and `core` modules.

2. **Registry pattern**: Always register components through registries or decorators, not direct instantiation.

3. **Event-driven**: Agent execution is asynchronous and event-driven. Always handle events in async generators.

4. **Hooks are optional**: Hooks extend behavior but aren't required for basic functionality.

5. **Agent discovery**: The CLI automatically finds root agents (not sub-agents of others) when using `domyn run`.

6. **Dependency management**: Use `uv` for all package operations, not `pip`.

7. **The package init stays empty**: `domyn/agents/__init__.py` imports none of its subpackages, so a process that needs one light module does not pay for the agent machinery. `decorators`, `register_transport` and the public names (`Agent`, `Runner`, `FunctionTool`, `VLLMProvider`, `load_application_from_file`, … — `_LAZY_EXPORTS`) load on first use (a PEP 562 `__getattr__`), so `from domyn.agents import Agent` works. The subpackage `__init__`s are the shorter import path (`from domyn.agents.tools import FunctionTool`, not `…tools.function_tool`); a module inside a package still imports its siblings by full path, and a test that patches a name patches it where it is defined (`domyn.agents.channels.telegram.TelegramChannel`), and `domyn/agents/decorators.py` keeps `import domyn.agents.decorators` working. `tests/test_package_import.py` fails if an eager import comes back.

## Logging Conventions

Customer data must never reach logs at default level (hosts embedding the library typically run at INFO).

- **Levels**: `error` = unrecovered failure; `warning` = recovered anomaly; `info` = flow milestones with ids/counts only; `debug` = payloads, exception detail, diagnostics. User content (messages, tool outputs/args, observations, LLM payloads, raw events) goes to **debug only**.
- **Exceptions — two-line idiom**: context + `type(e).__name__` at error/warning, full detail on an adjacent debug line ending with `detail: {e}`. Never interpolate raw `{e}` at error/warning (pydantic `ValidationError` embeds input values, httpx errors embed response bodies).
- **Format — eager f-strings only**: `get_logger()` returns a wrapper around whatever logger the host injects via `set_logger()`. With a host logger taking `(message, payload=None, **kwargs)`, lazy `%`-style args are lost or left as literal `%s`. Exception: `integrations/hermes_platform_gateway/*` uses a plain stdlib logger — lazy `%`-style is correct there.
- **LLM providers**: log requests/responses only through `log_inference_request` / `log_inference_response` (`llm/base.py`) — metadata at info, full payload at debug, `[Inference *]` markers stable for observability queries.
- **Hermes gateway plugin**: logging is injected by the host that launches the plugin, before it loads; its level follows that host's `LOG_LEVEL`, with no plugin-specific override.

## Testing Notes

- Tests use `pytest` with `pytest-asyncio` for async tests
- Coverage reporting with `pytest-cov`. On a pull request the `test` job (a required check) also fails when under 80% of the changed lines in `domyn/agents/` are covered (`diff-cover` against the base branch) or when the total drops more than 0.1% below `main`'s, which the badge job stores as `coverage-total.txt` on the `coverage-badge` branch (`scripts/coverage_total.py`). The `coverage-exempt` label skips both; it is read from the API when the job runs, so adding it takes a re-run of the job. A pull request that changes no code the tests read skips the job (the `changes` job decides, in place of a path filter, since a required check a filter never starts would wait forever), which counts as passed.
- Mock HTTP with `respx`
- Tests in `tests/` do not import from `examples/`. `examples/test_examples.py` is the one test of the examples: every example's agent builds and its files import, with placeholder settings, no model and no network

## Version Management

```bash
# Tag the next release on origin/main and push the tag (asks first)
make release-patch              # v1.6.2 -> v1.6.3
make release-minor              # v1.6.2 -> v1.7.0
make release-major              # v1.6.2 -> v2.0.0
make release VERSION=1.7.0rc1   # a pre-release, in the PEP 440 spelling
```

`make version-patch`, `version-minor` and `version-major` are aliases of the three, under the names releases were made with before.

Every night at 01:00 UTC (02:00 in Italy in winter, 03:00 in summer) `.github/workflows/nightly.yml` tags an alpha of `main` (`scripts/tag_release.py alpha`: the next patch after the latest stable tag, `v1.6.3a1`, `a2`…), unless `main` carries a tag already, and dispatches `release.yml` on it (a tag pushed with the workflow token starts no workflow). An alpha is published like a release, marked pre-release, except the documentation (no pre-release, alpha or release candidate, builds the docs: only a stable tag does); `scripts/prune_alpha_releases.py` keeps the Release pages of the last five alphas and drops those before a stable release (tags and packages stay).

A release is a tag and nothing else. No file holds the version: hatch-vcs reads it from the git tag when the package is built (`[tool.hatch.version]` in `pyproject.toml`), so a build at `v1.7.0` is `1.7.0` and one a few commits later `1.7.1.dev3+g<hash>` — which is what an editable install reports. No commit bumps it, so nothing is pushed to `main`. `scripts/tag_release.py` puts an annotated tag on `origin/main` as fetched, whatever is checked out. The tag starts `.github/workflows/release.yml`, which runs the tests on every supported Python version and only then publishes: the standalone archives to the GitHub Release (`standalone.yml`) and the documentation to GitHub Pages (`docs.yml`). A failing test publishes nothing; fix it and tag a new version. The release notes are written by `scripts/release_notes.py`: the pull requests merged since the previous stable tag grouped by type (New Features, Bug Fixes…, breaking changes first), then all of them by title and author (GitHub draws the contributors' avatars under the release from those mentions; bots are written without `@`). Titles are therefore Conventional Commits (`feat:`, `fix:`, `docs:`…, plus `add:` as `feat`, `change:` as `chore` and `remove:` for a feature taken out): `.github/workflows/pr-title.yml` checks it, as a required check of the `main: pull requests only` ruleset, so it runs on every push too (a required check is looked up on the latest commit). A ruleset's commit message rule cannot enforce it: tried, it checks the commits of the pull request, not the title the squash commit takes.
