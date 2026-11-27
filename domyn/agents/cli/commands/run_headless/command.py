"""``domyn run-headless``: one task, carried out to the end with nobody to answer and no
follow-up, then exit.

For running domyn as a harness: in a container, in CI, on a benchmark. It needs no
``domyn setup`` and reads nothing a setup left: not ``~/.domyn/config.yaml``, its
credentials, approvals, remembered settings or history. Its whole configuration is
built from the command line into a folder of its own
(``--state-dir``, a new temporary folder by default), which is used as a project
(`use_project`), so every file the session reads or writes is there.

The endpoint is named on the command line, never guessed from the environment: an
OpenAI-compatible ``--model_url``, the ``--model_name`` it serves, and ``--api-key-env``, the
variable holding its key (none is sent without it). No ``OPENAI_*`` variable is read, so
one left in the environment for some other tool cannot point the session elsewhere. The
model's limits are options of their own (``--context-window``, ``--max-tokens``), and so
is its reasoning effort (``--reasoning-effort``). Its default, ``default``, sends none:
the endpoint's own applies, so an endpoint that refuses the field still works.

What differs from an interactive session, because nobody answers:

* **Auto-approve is on** for every tool call. The hardline rules still apply.
* **One turn.** The task is the only message; the session ends when the agent ends
  its turn, without reading the input.
* **Standing instructions** say so (`DEFAULT_INSTRUCTIONS`): work alone, do the work,
  do not ask. Given inline, so no file is needed; ``--instructions`` replaces them.
* **No step cap** to speak of (``--max-iterations``, 1000 by default against the
  planner's 25), and no temperature unless one is given: the server's default.

``--config <file>`` runs a configuration file instead (the format of
``~/.domyn/config.yaml``), as written: it is copied unchanged into the state folder as
its ``config.yaml``, the options that build the configuration (`BUILD_OPTIONS`) are
refused with it, and the session still behaves as above, the standing instructions
aside, which are the file's. ``${DOMYN_PROJECT_DIR}`` in it is the state folder. Its
``${VAR}`` references are read from the environment, a ``.env`` and
``~/.domyn/credentials``: the file names the variables it uses, so a key loaded for some
other tool cannot reach it unasked.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any

import typer
import yaml

from domyn.agents.cli.commands.run.command import (
    SessionOptions,
    configure_session_agent,
    console,
    resolve_agent_from_file,
    start_session,
)
from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli.config import (
    CLI_DEFAULT_PROVIDER_NAME,
    CLI_DEFAULT_TOOL_BUNDLE_NAME,
    CLI_PLANNER,
    CONFIG_FILENAME,
    config_path,
    drop_credentials_from_environment,
    use_project,
    vllm_chat_url,
)

DEFAULT_INSTRUCTIONS = """\
# Working on your own

You are working alone in a Linux environment, with a shell and file tools. The message you were
given is a task to carry out, not a question to answer: nobody will read a plan or reply to a
question, and the session ends when you stop.

- **Do the work yourself.** Run the commands, write and edit the files, start the services. Do not
  describe steps for someone else to follow, and do not offer to do something: do it.
- **Do not ask questions.** When something is unclear, choose the most reasonable reading of the
  request, act on it, and say in one line what you assumed.
- **Get what you need.** Install missing tools and packages, create missing folders, and fix errors
  you run into on the way rather than stopping at them.
- **Follow the request exactly.** Use the file names, paths, function and command names, ports,
  formats and output wording it asks for, as written.
- **Check your work.** Before you finish, run what you built and confirm it does what was asked: the
  program runs, the service answers, the output matches. If it does not, fix it and check again.
- **Leave it working.** A server that has to keep running stays running when you finish; files go
  where the request says they go.
- **Finish with a short summary** of what you did and how you checked it.
"""
"""The standing instructions of a `domyn run-headless` session, unless ``--instructions`` gives others.

What the built-in prompts of other coding agents run without a user say, and what a session with
nobody to answer needs: without them a model tends to describe the steps and offer to
carry them out, and the session ends on the offer."""

DEFAULT_MAX_ITERATIONS = 1000
BUILD_OPTIONS = {
    "base_url": "--model_url",
    "model": "--model_name",
    "api_key_env": "--api-key-env",
    "context_window": "--context-window",
    "max_tokens": "--max-tokens",
    "temperature": "--temperature",
    "reasoning_effort": "--reasoning-effort",
    "max_iterations": "--max-iterations",
    "instructions": "--instructions",
    "no_instructions": "--no-instructions",
    "workspace": "--workspace",
}
"""The options that build the configuration, by parameter name: refused with ``--config``,
whose file is run as written."""
NO_REASONING_EFFORT = "default"
"""The ``--reasoning-effort`` that sends no ``reasoning_effort`` at all, and its default:
the endpoint's own applies."""
AGENT_NAME = "Assistant"


def run_headless_config(
    *,
    model: str,
    url: str,
    api_key_env: str | None,
    context_window: int | None,
    max_tokens: int | None,
    temperature: float | None,
    reasoning_effort: str | None,
    max_iterations: int,
    instructions: str | None,
    workspace: str,
) -> dict[str, Any]:
    """The ``config.yaml`` of a `domyn run-headless` session.

    ``None`` for ``context_window`` leaves it to the server (asked at session start,
    `apply_context_window`). ``max_tokens`` and ``temperature`` are always passed, so
    the provider's own defaults (0.7 and 2,000) are never merged in: ``None`` goes out
    as JSON ``null``, OpenAI's "unset", and the server's default applies. A
    ``reasoning_effort`` of ``None`` is not sent at all.

    The compactor is given no ``reserve_tokens``: it keeps room for the provider's
    reply ceiling (``max_reply_tokens``), which is ``max_tokens`` after
    ``max_tokens_share`` has lowered it, so a ``max_tokens`` as large as the window
    leaves the prompt half of it rather than the tenth floor; with no ``max_tokens``,
    its own default.
    """
    generation_params: dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature}
    if reasoning_effort is not None:
        generation_params["reasoning_effort"] = reasoning_effort
    provider: dict[str, Any] = {
        "name": CLI_DEFAULT_PROVIDER_NAME,
        "type": "vllm",
        "url": vllm_chat_url(url),
        "model_name": model,
        "generation_params": generation_params,
    }
    if api_key_env is not None:
        # The variable's name, not its value: the key is never written to disk, and the
        # loader reads it from the environment (`interpolate_env_vars`).
        provider["api_key"] = f"${{{api_key_env}}}"
    if context_window is not None:
        provider["context_window"] = context_window

    compactor: dict[str, Any] = {"strategy": "two_phase"}

    agent: dict[str, Any] = {
        "name": AGENT_NAME,
        "description": "A general-purpose assistant",
        "instruction": "",
        "llm_provider": CLI_DEFAULT_PROVIDER_NAME,
        "planner_strategy": CLI_PLANNER,
        "stream": False,
        "tool_bundles": [CLI_DEFAULT_TOOL_BUNDLE_NAME],
    }
    if instructions:
        agent["standing_instructions"] = {"source": {"type": "memory", "agent": instructions}}

    return {
        "version": "1.0",
        "entrypoint": AGENT_NAME,
        "llm_providers": [provider],
        "planner_strategies": [
            {
                "name": CLI_PLANNER,
                "type": CLI_PLANNER,
                "max_iterations": max_iterations,
                "compactor": compactor,
            }
        ],
        "agents": [agent],
        "tool_bundles": [
            {
                "name": CLI_DEFAULT_TOOL_BUNDLE_NAME,
                "bundles": ["files", "shell"],
                "workspace": workspace,
                # Commands run where domyn runs: a harness's sandbox is the machine
                # or container it was started in.
                "backend": "local",
                # Commands see the environment domyn was started with, the task's.
                "isolate_environment": False,
                "allowed_env": [],
            }
        ],
    }


def _fail(message: str) -> typer.Exit:
    console.print(f"  {message}", highlight=False, markup=False, style="red")
    return typer.Exit(2)


def _read_task(prompt: str | None, prompt_file: Path | None) -> str:
    if prompt is not None and prompt_file is not None:
        raise _fail("Give the task as an argument or with --prompt-file, not both.")
    if prompt_file is not None:
        try:
            text = sys.stdin.read() if str(prompt_file) == "-" else prompt_file.read_text("utf-8")
        except OSError as exc:
            raise _fail(f"Cannot read the task from {prompt_file}: {exc}") from exc
    else:
        text = prompt or ""
    if not text.strip():
        raise _fail("No task: give it as an argument or with --prompt-file.")
    return text


def _read_instructions(instructions: Path | None, no_instructions: bool) -> str | None:
    if no_instructions:
        if instructions is not None:
            raise _fail("Give --instructions or --no-instructions, not both.")
        return None
    if instructions is None:
        return DEFAULT_INSTRUCTIONS
    try:
        return instructions.read_text("utf-8")
    except OSError as exc:
        raise _fail(f"Cannot read the instructions in {instructions}: {exc}") from exc


def _given(ctx: typer.Context, names: dict[str, str]) -> list[str]:
    """The options of ``names`` given on the command line, a default left alone excluded.

    By the source's name: Typer ships its own copy of Click, whose ``ParameterSource`` is
    not the ``click`` package's."""
    given = []
    for name, flag in names.items():
        source = ctx.get_parameter_source(name)
        if source is not None and source.name == "COMMANDLINE":
            given.append(flag)
    return given


def run_headless_command(
    ctx: typer.Context,
    prompt: str | None = typer.Argument(
        None, help="The task. Or --prompt-file, to keep it off the command line."
    ),
    prompt_file: Path | None = typer.Option(
        None,
        "--prompt-file",
        help=(
            "Read the task from this file ('-' for the standard input). Keeps it off the "
            "command line, where `pkill -f <word>` run by the agent would match domyn itself."
        ),
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help=(
            "A configuration file to run as written (the format of ~/.domyn/config.yaml) "
            "instead of the built-in one. The options that build that one cannot be given "
            "with it."
        ),
    ),
    model: str | None = typer.Option(
        None,
        "--model_name",
        "-m",
        help="The model's name on the endpoint (needed without --config)",
    ),
    base_url: str | None = typer.Option(
        None,
        "--model_url",
        help="The OpenAI-compatible endpoint, e.g. http://host:8000/v1 (needed without --config)",
    ),
    api_key_env: str | None = typer.Option(
        None,
        "--api-key-env",
        help="The environment variable holding the endpoint's key (default: no key sent)",
    ),
    context_window: int | None = typer.Option(
        None,
        "--context-window",
        min=1,
        help="The model's context window in tokens (default: asked of the endpoint)",
    ),
    max_tokens: int | None = typer.Option(
        None,
        "--max-tokens",
        min=1,
        help="The longest reply to ask for (default: null, the endpoint decides)",
    ),
    temperature: float | None = typer.Option(
        None, "--temperature", help="Sampling temperature (default: null, the endpoint's)"
    ),
    reasoning_effort: str = typer.Option(
        NO_REASONING_EFFORT,
        "--reasoning-effort",
        help=(
            "Sent as `reasoning_effort` on every call, e.g. low, medium, high "
            f"(default: '{NO_REASONING_EFFORT}', none sent, the endpoint decides)"
        ),
    ),
    max_iterations: int = typer.Option(
        DEFAULT_MAX_ITERATIONS, "--max-iterations", min=1, help="Most model calls in the turn"
    ),
    instructions: Path | None = typer.Option(
        None,
        "--instructions",
        help="A file whose text replaces the built-in standing instructions",
    ),
    no_instructions: bool = typer.Option(
        False, "--no-instructions", help="Run with no standing instructions"
    ),
    workspace: str = typer.Option(
        "/",
        "--workspace",
        help="The folder the file tools may read and write under (default: the whole machine)",
    ),
    state_dir: Path | None = typer.Option(
        None,
        "--state-dir",
        help=(
            "Folder for the session's configuration and conversation history "
            "(default: a new temporary folder). Never ~/.domyn."
        ),
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show internal events (LLM, hooks, agent lifecycle)"
    ),
) -> None:
    """Carry out one task on your own: no questions, no follow-up, then exit.

    Needs no `domyn setup` and ignores any: the endpoint is --model_url, the model --model_name
    and the key the variable --api-key-env names, or --config names a configuration file
    to run. Every tool call is approved automatically.
    """
    task = _read_task(prompt, prompt_file)
    given_config: str | None = None
    standing: str | None = None
    if config is not None:
        if given := _given(ctx, BUILD_OPTIONS):
            raise _fail(
                f"{', '.join(given)} build the configuration, which --config gives: "
                f"set it in {config} instead."
            )
        try:
            given_config = config.expanduser().read_text("utf-8")
        except OSError as exc:
            raise _fail(f"Cannot read the configuration in {config}: {exc}") from exc
    else:
        if model is None or base_url is None:
            raise _fail(
                "Give --model_url and --model_name, or --config to run a configuration file."
            )
        standing = _read_instructions(instructions, no_instructions)
        if api_key_env is not None and not os.environ.get(api_key_env):
            raise _fail(f"--api-key-env names {api_key_env}, which is not set.")
        # Startup put ~/.domyn/credentials in the environment, before any option was
        # read: the configuration built here names no variable but --api-key-env, so
        # the environment's own is the only key it may use.
        drop_credentials_from_environment()

    folder = (
        state_dir.expanduser().resolve()
        if state_dir is not None
        else Path(tempfile.mkdtemp(prefix="domyn-run-headless-"))
    )
    folder.mkdir(parents=True, exist_ok=True)
    if given_config is not None:
        # Copied as it is, so the folder is a project like any other (the resume command
        # printed at the end opens it) and keeps the configuration the run used.
        (folder / CONFIG_FILENAME).write_text(given_config, "utf-8")
    else:
        assert model is not None and base_url is not None
        built = run_headless_config(
            model=model,
            url=base_url,
            api_key_env=api_key_env,
            context_window=context_window,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=None if reasoning_effort == NO_REASONING_EFFORT else reasoning_effort,
            max_iterations=max_iterations,
            instructions=standing,
            workspace=workspace,
        )
        (folder / CONFIG_FILENAME).write_text(yaml.safe_dump(built, sort_keys=False), "utf-8")
    # The folder plays ~/.domyn from here on: approvals, state.yaml, conversations.db and
    # the tools' working files are all its own.
    use_project(str(folder))

    # Every session of this command starts without a terminal to talk to.
    theme = Theme.build(allow_animation=False)
    resolved = asyncio.run(resolve_agent_from_file(config_path(), theme, allow_setup=False))
    if resolved is None:
        raise typer.Exit(1)
    agent, raw = resolved
    agent = configure_session_agent(agent, SessionOptions())
    exit_code = asyncio.run(
        start_session(
            agent,
            theme,
            initial_prompt=task,
            verbose=verbose,
            show_status=False,
            config=raw,
            config_file=config_path(),
            run_command=f"domyn run --project {shlex.quote(str(folder))}",
            auto_approve=True,
            single_turn=True,
            steering=False,
        )
    )
    raise typer.Exit(exit_code)
