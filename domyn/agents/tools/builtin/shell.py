"""Running commands.

**This runs through a real shell, and that is a decision.** An explicit argument
vector is safer in the abstract and useless in practice — no pipes, no ``&&``,
no redirection — and the model would spend its turns working around the tool.

What makes it honest is that **the approved subject is the entire command line
the model supplied**: approval theatre is showing ``git`` and running
``git status; rm -rf ~``, and nothing the model wrote is hidden from the person
approving it. An exact grant matches the whole line; a pattern must match each
command in it (:func:`_approval_parts`), and a line running something it does
not show (``$(…)``, backticks, an unquoted heredoc, ``${x:1}``, ``$CMD`` as the
program) takes only an exact grant.

**This used to say "byte for byte", and no longer can.** What the shell receives
is a wrapper — see :mod:`~domyn.agents.tools.builtin.processes` — so the bytes
differ and the claim has to be what it actually guarantees:

* the model's text appears **exactly once**, ``shlex.quote``-d into a file the
  wrapper runs, and nowhere else; same for any ``stdin`` it supplies;
* everything else in that wrapper is written here and is fixed.

So a person still cannot approve one thing and have another run. The weaker
property, that the panel shows the literal argv, is gone — and pretending
otherwise in a docstring is worse than losing it.

*Where* a command runs belongs to :mod:`~domyn.agents.tools.builtin.backends`.
This module owns what the model sees, including truncation that **keeps the tail
as well as the head**: errors are at the end of a build log.

None of this makes a shell safe. With the local backend the boundary is the
machine the agent runs on; if that matters the answer is an isolated backend,
not a longer blocklist.
"""

# No `from __future__ import annotations` here: FunctionTool reads the signature
# of each tool at construction time and cannot resolve string annotations. The
# tools are defined in this module, so the import has to stay off.

import os
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from domyn.agents.core import ApprovalType
from domyn.agents.tools.base import UpdateToolResult
from domyn.agents.tools.builtin import processes
from domyn.agents.tools.builtin.backends import CommandBackend, CommandResult, LocalBackend
from domyn.agents.tools.builtin.workspace import Workspace

# Long enough for an install or a test suite, short enough that a hung command
# does not look like a hung agent.
DEFAULT_TIMEOUT_S = 120.0

# How long `background=True` waits before handing back a handle: long enough to
# catch a command that fails on its first line, short enough not to be a wait.
BACKGROUND_YIELD_S = 1.0

# Per stream. Command output is for the model to read, and a build log will
# exhaust its context long before it becomes informative.
MAX_OUTPUT_CHARS = 20_000


def _clip(text: str, label: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    head = text[: MAX_OUTPUT_CHARS // 2]
    tail = text[-MAX_OUTPUT_CHARS // 2 :]
    dropped = len(text) - len(head) - len(tail)
    return f"{head}\n\n... [{dropped} characters of {label} omitted] ...\n\n{tail}"


def render(result: CommandResult, *, timeout: float) -> str:
    """Turn a result into the text the model reads.

    Every branch says what to do next, not only what went wrong: "it failed"
    does not tell the model whether to retry, wait longer or give up.
    """
    if result.backend_error:
        return f"The command could not be run: {result.backend_error}"
    if result.timed_out:
        return (
            f"The command was killed after {timeout:.0f}s without finishing. "
            "If it is expected to take longer, pass a larger timeout. If it "
            "stopped to ask something, nothing was there to answer: re-run it "
            "with the answer in `stdin`, piped (`yes | ...`), or with whatever "
            "flag makes it stop asking."
        )

    parts = [f"exit code: {result.exit_code}"]
    stdout = _clip(result.stdout, "output")
    stderr = _clip(result.stderr, "errors")
    if stdout.strip():
        parts.append(f"stdout:\n{stdout.rstrip()}")
    if stderr.strip():
        parts.append(f"stderr:\n{stderr.rstrip()}")
    if len(parts) == 1:
        parts.append("(no output)")
    return "\n\n".join(parts)


def _seconds(value: object, fallback: float) -> float:
    """Read a timeout the model supplied, whatever shape it arrived in.

    The schema says integer and a model may still send ``"30"``. This used to be
    ``timeout > 0``, which raised a ``TypeError`` before the command ever ran — a
    tool refusing to work because an *optional* argument was quoted.

    Anything unusable falls back to the default: the caller asked to run a
    command, not to negotiate a timeout.
    """
    try:
        seconds = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    return seconds if seconds > 0 else fallback


def flag(value: object) -> bool:
    """Read a boolean the model supplied, ``"true"`` included.

    Every tool's boolean goes through this, not ``bool()`` or a bare ``if``: the
    schema says boolean and a model may still send ``"false"``, which they read
    as true. The browser tools use it too, where that would submit a form."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


# Said after the fact, because the command line is the approved subject and is
# never rewritten: stripping the `&` would run something other than what was
# approved.
DETACHED_NOTE = (
    "\n\nNote: this line detached a command (an `&`, `disown` or `setsid -f`), so it is "
    "not tracked: no handle, no log, and the person cannot see or stop it with "
    "/ps and /stop. If it should keep running, stop the detached process and run "
    "the command again without `&`, with background=true."
)


def _detaches(command: str) -> bool:
    """Whether a line leaves a command running past the shell that started it.

    Read with the approval tokenizer, so an `&` is found wherever it is and told
    apart from `&&`, `2>&1` and `&>file`, and `disown` counts only as a program,
    never as an argument or quoted text. A bare `wait` joins what was started
    before it (`a & b & wait`), so that is not a detach.
    """
    commands = _commands(command)
    if commands is None:  # a subshell or a substitution: only the line's end is certain
        line = command.rstrip()
        return line.endswith("&") and not line.endswith("&&")
    pending = False
    for c in commands:
        i = _program_index(c.words)
        program, args = (c.words[i], c.words[i + 1 :]) if i is not None else ("", [])
        if program == "disown" or (program == "setsid" and _forks(args)):
            return True
        if program == "wait" and not args and not c.background:
            pending = False
        elif c.background:
            pending = True
    return pending


def _forks(setsid_args: list[str]) -> bool:
    """``setsid -f`` / ``--fork`` without ``-w`` / ``--wait``: the shell moves on
    while the command runs in a session of its own."""
    flags = set()
    for arg in setsid_args:
        if not arg.startswith("-") or arg == "--":
            break
        flags |= {arg[2:]} if arg.startswith("--") else set(arg[1:])
    return bool(flags & {"f", "fork"}) and not flags & {"w", "wait"}


def build_run_command(
    backend: CommandBackend,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_lifetime_s: float = processes.DEFAULT_MAX_LIFETIME_S,
) -> Callable[..., Awaitable[Any]]:
    """Build the ``run_command`` callable bound to one backend."""

    async def run_command(
        command: str,
        timeout: int = 0,
        stdin: str = "",
        background: bool = False,
        conversation_id: str = "",
        state: dict[str, Any] | None = None,
    ) -> Any:
        r"""Run a shell command in the workspace and return its output.

        Use this to install packages, run tests, inspect the system, use git —
        anything a terminal can do. Pipes, redirection and `&&` all work.

        **A command that takes longer than `timeout` is not killed.** You get
        back a handle and whatever it has printed so far, and it carries on
        running while you do other things. Check on it with `check_command`, and
        `stop_command` it if it is going nowhere. So there is no need to guess a
        large timeout for something that might be slow: start it and look again.

        **To run something in the background** — a server, a watcher, a long
        build, or whenever you are asked to — pass `background=true`: you get
        the handle straight away. Never use `&`, `nohup`, `setsid` or `disown`
        for it: a detached process cannot be followed, stopped, or seen in the
        person's `/ps`.

        There is no terminal, so a command that stops to ask a question gets no
        answer and waits forever. When you expect one, either make it not ask
        (`-y`, `--yes`, `DEBIAN_FRONTEND=noninteractive`) or answer it in
        advance: pass `stdin`, or pipe it — `yes | some-command`. Several
        answers work the same way, one per line, in the order it asks.

        Args:
            command: The command line to run, exactly as you would type it.
            timeout: Seconds to wait before handing back a handle and letting it
                continue. 0 uses the default.
            stdin: Input handed to the command, all of it, before it starts.
                Use it to answer prompts you know are coming — `"y\n"`, or
                `"y\n2\n"` for two of them.
            background: Hand back a handle at once and leave the command
                running, instead of waiting `timeout` for it.
        """
        asked = BACKGROUND_YIELD_S if flag(background) else _seconds(timeout, timeout_s)
        limit = processes.block_seconds(asked, timeout_s)
        # Idempotent, and here rather than at construction so a session that
        # never runs a command never pays to prepare the backend.
        await backend.start()

        live = [r for r in processes.registry(state).values() if not r.finished]
        if len(live) >= processes.MAX_PROCESSES:
            # Look before refusing: nothing marks a command finished on its
            # own, so a conversation that never looked would believe all of
            # them were running and refuse forever. One sweep, only here.
            sweep = await backend.run(
                processes.build_sweep(live), timeout=60, conversation_id=conversation_id
            )
            if not sweep.backend_error:
                live = processes.refresh(state, processes.parse_sweep(sweep.stdout))

        if len(live) >= processes.MAX_PROCESSES:
            listed = "\n".join(f"  `{r.handle}` — $ {r.command}" for r in live)
            return _result(
                f"{len(live)} commands are still running, which is the limit. Stop one "
                f"with stop_command, or wait for one to finish:\n{listed}",
                state,
            )

        record = processes.new_record(command, conversation_id)
        result = await backend.run(
            processes.build_launch(
                record, yield_s=limit, max_lifetime=max_lifetime_s, stdin=stdin or None
            ),
            # The call returns on its own after `limit`; this only bounds a
            # backend that has stopped answering, so it is deliberately longer.
            timeout=limit + 60,
            conversation_id=conversation_id,
        )
        if result.backend_error:
            return _result(f"The command could not be run: {result.backend_error}", state)

        observed = processes.parse_report(result.stdout, record, from_cursor=False)

        if not observed.running:
            # Finished inside the window, which is almost every command. It must
            # read exactly as it did before any of this existed: no handle, no
            # mention of background, nothing new to learn for the common case.
            text = render(
                CommandResult(
                    exit_code=observed.exit_code,
                    stdout=observed.out,
                    stderr=observed.err,
                ),
                timeout=limit,
            )
            if _detaches(command):
                text += DETACHED_NOTE
            return _result(text, state)

        record.cursor = observed.out_bytes
        record.err_cursor = observed.err_bytes
        processes.remember(state, record)
        return _result(processes.describe(observed, following=False, command=command), state)

    return run_command


# How often a deliberate wait reports what it has seen. A single blocking call
# cannot say anything while it blocks, so a wait is chunked into probes — paid
# for in round trips, which is why only a wait the model asked for is chunked.
# The first yield of `run_command` stays one call: making `ls` pay for that
# would be a bad trade taken on every command.
PROGRESS_STEP_S = 10


def build_check_command(backend: CommandBackend) -> Callable[..., Any]:
    """Build the ``check_command`` callable bound to one backend."""

    async def check_command(
        handle: str,
        wait: int = 0,
        conversation_id: str = "",
        state: dict[str, Any] | None = None,
    ) -> Any:
        """Look at a command that is still running, and see what it has printed.

        Returns whether it is alive, how long it has been going, **how long since
        it last printed anything** — a run that has said nothing for ten minutes
        is usually stuck — and whatever is new since you last looked.

        Only what is new: to see the whole log, or to search it, use the file
        tools on the path given when the command started.

        Use `wait` when something should finish shortly and you would rather
        carry on afterwards than answer now — it comes back the moment the
        command ends, so waiting 60 seconds for something that takes 5 costs 5.
        A conversation has a limited amount of waiting in total, so spend it on
        something about to finish rather than on watching a long run go by.
        While waiting, what the command prints is shown to the person watching.
        Do not use it to sit through a long run: say what you have seen, let them
        read it, and look again later.

        Args:
            handle: The handle you were given when the command did not finish.
            wait: Seconds to wait for it to finish before answering. 0 looks and
                returns at once.
        """
        await backend.start()
        known = processes.registry(state)
        record = known.get(handle)
        if record is None:
            yield _result(
                f"No command with handle `{handle}`. Running ones: "
                + (", ".join(f"`{h}`" for h in known) or "none")
                + ".",
                state,
            )
            return

        allowed, remaining = processes.wait_allowance(state, int(wait or 0))
        began = time.monotonic()
        observed = None
        left = allowed

        while True:
            step = min(PROGRESS_STEP_S, left) if left > 0 else 0
            result = await backend.run(
                processes.build_probe(record, wait_s=int(step)),
                timeout=step + 60,
                conversation_id=conversation_id,
            )
            if result.backend_error:
                yield _result(f"Could not check `{handle}`: {result.backend_error}", state)
                return

            observed = processes.parse_report(result.stdout, record, from_cursor=True)
            record.cursor = observed.out_bytes
            record.err_cursor = observed.err_bytes

            left -= step
            if not observed.running or left <= 0:
                break
            if observed.out.strip():
                # Shown while it happens, to whoever is watching. Not a partial
                # RESPONSE: that is the agent's own content, and a progress line
                # is not the agent speaking — it is the command doing so.
                yield UpdateToolResult(
                    data={
                        "progress": processes.tail_line(observed.out),
                        "handle": handle,
                        "elapsed_s": observed.elapsed_s,
                    }
                )

        # What it cost, not what it was allowed: a wait ends when the command
        # does, and charging the allowance would let a few quick checks exhaust a
        # budget they never used.
        processes.spend_wait(state, int(time.monotonic() - began))
        record.finished = not observed.running
        processes.remember(state, record)

        text = processes.describe(observed, following=True, command=record.command)
        if wait and not allowed:
            # **First**, not appended. It used to sit at the bottom, under the
            # command's output, where it read as a footnote — so what this looked
            # like from outside was a model asking to wait and being answered
            # instantly, again and again, for no reason it could see. A refusal
            # has to arrive before the thing it is refusing.
            text = (
                "NOTE: you asked to wait, and no waiting was done — this "
                "conversation has used up its waiting time. Asking again will "
                "not wait either. Tell the person what you have seen and let "
                "them decide; look again later, or leave it running.\n\n"
            ) + text
        elif allowed and observed.running and remaining - allowed < processes.MAX_WAIT_S:
            text += f"\n\n(About {remaining - allowed}s of waiting left in this conversation.)"
        yield _result(text, state)

    return check_command


def build_list_commands(backend: CommandBackend) -> Callable[..., Awaitable[Any]]:
    """Build the ``list_commands`` callable bound to one backend."""

    async def list_commands(conversation_id: str = "", state: dict[str, Any] | None = None) -> Any:
        """List the commands you started that are still around, with their handles.

        Use it when you have lost track of a handle, or to check whether anything
        is still going before you finish. It costs nothing to run.

        Handles arrive in the reply of the command that yielded them, and replies
        are not kept forever — so if one is no longer in front of you, this is how
        to get it back rather than starting the work again.

        When you tell someone about these, name them by what they run. A handle
        is an internal identifier and means nothing to the person reading; they
        only need one if they are going to type `/stop` themselves.
        """
        await backend.start()
        known = processes.registry(state)

        # One call, and it asks the directory rather than the registry — so it
        # also finds anything this conversation started whose record was lost.
        seen: dict[str, Any] = {}
        result = await backend.run(
            processes.build_discover(conversation_id), timeout=60, conversation_id=conversation_id
        )
        if not result.backend_error:
            seen = processes.parse_discover(result.stdout)

        if not seen and not known:
            return _result("Nothing running, and nothing finished recently.", state)

        lines = []
        for handle in dict.fromkeys(list(seen) + list(known)):
            record = known.get(handle) or processes.ProcessRecord(
                handle=handle,
                command=seen.get(handle, {}).get("command", ""),
                conversation=conversation_id,
            )
            observed = seen.get(handle)
            if observed is None:
                lines.append(f"`$ {record.command}` — cannot be checked — handle `{handle}`")
                continue
            record.finished = not observed["running"]
            processes.remember(state, record)
            if observed["running"]:
                where = f"running, {observed['elapsed_s']}s in, quiet for {observed['idle_s']}s"
            elif observed["killed"]:
                where = "stopped"
            else:
                where = f"finished, exit {observed['exit_code']}"
            lines.append(f"`$ {record.command}` — {where} — handle `{handle}`")
        return _result("\n".join(lines), state)

    return list_commands


def build_stop_command(
    backend: CommandBackend, grace_s: int | None = None
) -> Callable[..., Awaitable[Any]]:
    """Build the ``stop_command`` callable bound to one backend."""

    async def stop_command(
        handle: str, conversation_id: str = "", state: dict[str, Any] | None = None
    ) -> Any:
        """Stop a command you started earlier, and everything it started.

        Use it when a run is going nowhere — stuck, diverging, or no longer
        needed. What it produced stays readable in its log afterwards.

        Args:
            handle: The handle of the command to stop.
        """
        await backend.start()
        known = processes.registry(state)
        record = known.get(handle)
        if record is None:
            return _result(f"No command with handle `{handle}`.", state)

        result = await backend.run(
            processes.build_kill(record, grace=grace_s or processes.DEFAULT_GRACE_S),
            # It sends SIGTERM, waits out the grace period, then SIGKILL.
            timeout=(grace_s or processes.DEFAULT_GRACE_S) + 60,
            conversation_id=conversation_id,
        )
        record.finished = True
        processes.remember(state, record)
        if result.backend_error:
            return _result(f"Could not stop `{handle}`: {result.backend_error}", state)
        if "not running" in result.stdout:
            return _result(f"`{handle}` had already finished; nothing to stop.", state)
        return _result(f"Stopped `{handle}` and everything it started.", state)

    return stop_command


def _result(text: str, state: dict[str, Any] | None) -> Any:
    """Carry the text and the updated state back together.

    `state` has to be *returned*, not mutated: ``FunctionTool.execute`` deep
    copies what it injects precisely so a tool cannot change it in place without
    saying so.
    """
    from domyn.agents.tools.base import ToolResult

    return ToolResult(success=True, data=text, state=state)


def shell_tools(
    workspace: Workspace,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_lifetime_s: float = processes.DEFAULT_MAX_LIFETIME_S,
    backend: CommandBackend | None = None,
) -> list[Any]:
    """Running commands, and following the ones that outlive the call.

    Only ``run_command`` is gated. Watching reads a file, and stopping can only
    reach something this conversation started (``processes.build_kill``) —
    asking permission for either is asking permission to notice.
    """
    resolved = backend or LocalBackend(workspace)
    return [
        run_command_tool(
            workspace, timeout_s=timeout_s, max_lifetime_s=max_lifetime_s, backend=resolved
        ),
        check_command_tool(resolved),
        list_commands_tool(resolved),
        stop_command_tool(resolved),
    ]


def check_command_tool(backend: CommandBackend) -> Any:
    """Ungated: it reads a log file.

    Streamable, so a wait can report what the command is printing while it is
    still printing it. Without that the person watching gets a spinner and
    nothing else for however long the wait lasts.
    """
    from domyn.agents.tools.streamable_function_tool import StreamableFunctionTool

    return StreamableFunctionTool(func=build_check_command(backend))


def list_commands_tool(backend: CommandBackend) -> Any:
    """Ungated: it answers a question the model had the answer to and lost.

    Handles arrive in a tool reply, and tool replies are pruned when the
    conversation is compacted — so a running process becomes unreachable while
    its record sits in state, intact and unread. This is the way back in.
    """
    from domyn.agents.tools.function_tool import FunctionTool

    return FunctionTool(func=build_list_commands(backend))


def stop_command_tool(backend: CommandBackend, grace_s: int | None = None) -> Any:
    """Ungated, and that is a decision — see ``shell_tools``."""
    from domyn.agents.tools.function_tool import FunctionTool

    return FunctionTool(
        func=build_stop_command(backend, grace_s=grace_s or processes.DEFAULT_GRACE_S)
    )


def run_command_tool(
    workspace: Workspace,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_lifetime_s: float = processes.DEFAULT_MAX_LIFETIME_S,
    backend: CommandBackend | None = None,
) -> Any:
    """The gated ``run_command`` tool."""
    from domyn.agents.tools.function_tool import FunctionTool

    return FunctionTool(
        func=build_run_command(
            backend or LocalBackend(workspace),
            timeout_s=timeout_s,
            max_lifetime_s=max_lifetime_s,
        ),
        approval_type=ApprovalType.USER_CONFIRM,
        # The whole command line, so what is shown is what is run and a grant
        # covers that exact line rather than "the shell".
        approval_subject=lambda command, **_: f"$ {command}",
        # Each command the line runs: a pattern must cover them all.
        approval_parts=lambda command, **_: _approval_parts(command),
        # The program, not the whole line: "let it run curl" is a permission a
        # person can reason about, "let it run this exact URL" is not one they
        # will ever hit twice.
        approval_pattern=lambda command, **_: _pattern_for(command),
    )


def _program(words: list[str]) -> tuple[str, str]:
    """The words up to the program, ``FOO=bar`` included, and the program's name.

    From the parser's words, so ``FOO='a b'`` stays one word.
    """
    i = _program_index(words)
    if i is None:
        return " ".join(words), ""
    # By name or by path (`/bin/bash`).
    return " ".join(words[: i + 1]), words[i].strip("'\"").rsplit("/", 1)[-1]


def _pattern_for(command: str) -> str | None:
    """``$ <program> *`` when every part runs the same program.

    None for programs that run their arguments (``bash``, ``sudo``…): the pattern
    would cover any command.
    """
    programs = {_program(c.words) for c in _needed(command) or []}
    if len(programs) != 1:
        return None
    prefix, name = programs.pop()
    return None if name in _RUNS_ANYTHING else f"$ {prefix} *"


# --- the commands a line runs ------------------------------------------------
#
# Not a full parser: anything uncertain makes the line unsplittable, and only an
# exact grant covers an unsplittable line.

_SEPARATORS = ("&&", "||", "|&", ";", "|", "&", "\n")
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
# Only the quoted forms: an unquoted delimiter runs `$(…)` inside the body.
_HEREDOC = re.compile(r"<<(-?)[ \t]*(?:'([\w.-]+)'|\"([\w.-]+)\"|\\([\w.-]+))(?=[\s;&|<>]|$)")
# Builtins that act on nothing but the shell's own state or its output.
_GLUE = frozenset({"cd", "echo", "pwd", "true", "false"})
# Read by the shell, or exported by the session: assigning one changes how later
# commands run.
_SHELL_VARIABLES = frozenset(
    {"PATH", "HOME", "IFS", "CDPATH", "GLOBIGNORE", "PS4", "ENV", "BASH_ENV", "VIRTUAL_ENV"}
)
_RUNS_ANYTHING = frozenset(
    {
        *("sh", "bash", "zsh", "dash", "ksh", "fish", "source", "."),
        *("sudo", "doas", "su", "env", "xargs", "exec", "eval", "command", "builtin"),
        *("nohup", "nice", "timeout", "time", "watch"),
    }
)
_LINE_COUNT = re.compile(r"-\d+|-[nc]\d+")
_WC_FLAGS = re.compile(r"-[lcwmL]+")
_GREP_FLAGS = re.compile(r"-[ivnwxocEFGHhsq]+")
# A word with nothing to expand, glob or split.
_LITERAL = re.compile(r"""'[^']*'|"[^"$`]*"|[\w.,:@%+=/^-]+""")
# Runs a variable's value as code: `$[x]`, and any `${…}` beyond a plain name.
_EVALUATING = re.compile(r"\$\[|\$\{(?![A-Za-z_][A-Za-z0-9_]*\})")
# A program named by an expansion or a glob: known only when the line runs.
_DYNAMIC = re.compile(r"[$*?]")
# Writes nowhere: `2>/dev/null`, `&>/dev/null`, `2>&1`, `>&2`.
_HARMLESS_REDIRECT = re.compile(
    r"&?>>?[ \t]*/dev/null(?=[\s;&|<>]|$)|>&(?:[0-9]+|-)(?=[\s;&|<>]|$)"
)


@dataclass
class _Command:
    words: list[str]
    after_pipe: bool
    redirects: bool
    background: bool = False  # ended by a bare `&`


def _approval_parts(command: str, environment: Mapping[str, str] = os.environ) -> list[str]:
    """The commands of a line a pattern must cover, as subjects.

    Glue (``cd``, ``echo``, ``S=…``, ``| tail -30``) is left out unless it is all
    there is, so ``$ python *`` covers ``S=… python "$S/a.py" | tail -30``.

    ``environment`` is what the commands inherit; ``os.environ`` is a superset,
    so it errs towards asking.
    """
    return [f"$ {' '.join(c.words)}" for c in _needed(command, environment) or []]


def _needed(command: str, environment: Mapping[str, str] = os.environ) -> list[_Command] | None:
    """The commands a pattern must cover; None when the line cannot be split."""
    commands = _commands(command)
    if commands is None or any(_EVALUATING.search(w) for c in commands for w in c.words):
        return None
    needed = [c for c in commands if not _is_glue(c, environment)] or commands
    for c in needed:
        i = _program_index(c.words)
        if i is not None and _DYNAMIC.search(c.words[i]):
            return None
    return needed


def _program_index(words: list[str]) -> int | None:
    """Where the program is, past any leading ``FOO=bar``."""
    return next((i for i, word in enumerate(words) if not _ASSIGNMENT.match(word)), None)


def _is_glue(command: _Command, environment: Mapping[str, str]) -> bool:
    """Whether a command can do nothing a grant is needed for."""
    if command.redirects:  # `cd x > ~/.bashrc` truncates a file
        return False
    first, *rest = command.words
    if first in _GLUE:
        return True
    if all(_ASSIGNMENT.match(word) for word in command.words):
        # A new variable reaches later commands only via a visible `$NAME` or an
        # `export` (not glue); an inherited one changes them unseen.
        names = {word.split("=", 1)[0] for word in command.words}
        return not any(name in _SHELL_VARIABLES or name in environment for name in names)
    return command.after_pipe and _filters_the_pipe(first, rest)


def _filters_the_pipe(program: str, args: list[str]) -> bool:
    """A pipe filter that opens no file: ``| tail -30``, ``| wc -l``, ``| grep -i x``."""
    if program in ("head", "tail"):
        return _counts_only(args)
    if program == "wc":
        return all(_WC_FLAGS.fullmatch(arg) for arg in args)
    if program == "grep":
        operands = [arg for arg in args if not _GREP_FLAGS.fullmatch(arg)]
        return (
            len(operands) == 1
            and not operands[0].startswith("-")
            and _LITERAL.fullmatch(operands[0]) is not None
        )
    return False


def _counts_only(args: list[str]) -> bool:
    """``-30``, ``-n 30``, ``-n30``."""
    i = 0
    while i < len(args):
        if _LINE_COUNT.fullmatch(args[i]):
            i += 1
        elif args[i] in ("-n", "-c") and i + 1 < len(args) and args[i + 1].isdigit():
            i += 2
        else:
            return False
    return True


def _commands(line: str) -> list[_Command] | None:
    """Split a shell line into its simple commands, or ``None`` when it cannot be
    read with certainty: command or process substitution, subshells, unquoted
    heredocs, unterminated quotes."""
    commands: list[_Command] = []
    words: list[str] = []
    word: list[str] = []
    after_pipe = redirects = False
    heredoc: tuple[str, bool] | None = None
    i, n = 0, len(line)

    def end_word() -> None:
        if word:
            words.append("".join(word))
            word.clear()

    def end_command(next_after_pipe: bool, background: bool = False) -> None:
        nonlocal after_pipe, redirects
        end_word()
        if words:
            commands.append(_Command(list(words), after_pipe, redirects, background))
            words.clear()
        after_pipe, redirects = next_after_pipe, False

    while i < n:
        c = line[i]
        if c == "\n" and heredoc is not None:
            end_command(False)
            i = _after_heredoc(line, i + 1, *heredoc)
            heredoc = None
            continue
        if c == "\\":
            if not line.startswith("\\\n", i):  # a continuation joins the lines
                word.append(line[i : i + 2])
            i += 2
            continue
        if c in " \t":
            end_word()
            i += 1
            continue
        if c == "'":
            close = line.find("'", i + 1)
            if close < 0:
                return None
            word.append(line[i : close + 1])
            i = close + 1
            continue
        if line.startswith("$'", i) or c == '"':
            j = i + (2 if c == "$" else 1)
            quote = "'" if c == "$" else '"'
            while j < n and line[j] != quote:
                if quote == '"' and (line[j] == "`" or line.startswith("$(", j)):
                    return None
                j += 2 if line[j] == "\\" else 1
            if j >= n:
                return None
            word.append(line[i : j + 1])
            i = j + 1
            continue
        if c in "`()" or line.startswith(("$(", "<(", ">("), i):
            return None
        if c == "#" and not word:  # a comment: the rest of the line does not run
            end = line.find("\n", i)
            i = n if end < 0 else end
            continue
        if line.startswith("<<<", i):
            word.append("<<<")
            i += 3
            continue
        if line.startswith("<<", i):
            match = _HEREDOC.match(line, i)
            if match is None or heredoc is not None:
                return None
            heredoc = (match.group(2) or match.group(3) or match.group(4), bool(match.group(1)))
            end_word()
            words.append(match.group(0))
            i = match.end()
            continue
        if c == ">" or line.startswith("&>", i):
            harmless = _HARMLESS_REDIRECT.match(line, i)
            if harmless is not None:
                # Not part of the command: `make 2>&1` is `make`.
                if "".join(word).isdigit():
                    word.clear()  # the 2 of `2>/dev/null`
                end_word()
                i = harmless.end()
                continue
            redirects = True
        separator = next((s for s in _SEPARATORS if line.startswith(s, i)), None)
        if separator == "&" and (line.startswith("&>", i) or (word and word[-1][-1] in "<>")):
            separator = None  # `&>file`, `>&file`, `<&3`
        if separator == "|" and word and word[-1][-1] == ">":
            separator = None  # `>|file`
        if separator is not None:
            end_command(separator in ("|", "|&"), background=separator == "&")
            i += len(separator)
            continue
        word.append(c)
        i += 1
    end_command(False)
    return commands


def _after_heredoc(line: str, start: int, delimiter: str, strip_tabs: bool) -> int:
    """Where commands resume after a heredoc body: the end, if it is unterminated."""
    position = start
    while position < len(line):
        end = line.find("\n", position)
        text = line[position:] if end < 0 else line[position:end]
        if (text.lstrip("\t") if strip_tabs else text) == delimiter:
            return len(line) if end < 0 else end + 1
        if end < 0:
            break
        position = end + 1
    return len(line)
