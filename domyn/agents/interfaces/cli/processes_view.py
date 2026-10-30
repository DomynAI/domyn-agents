"""Showing the commands that are still running.

A process the agent started and nobody can see is a process the user finds out
about from the fan. The point of letting a command outlive its call is that the
session stays usable while it runs — which only works if "what is running" is a
question with an answer, and if stopping something does not require asking the
agent to do it.

Read-only except for ``/stop``, and deliberately outside the agent's turn: this
asks the machine directly rather than going through the model, so it works while
the agent is busy and costs no tokens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from domyn.agents.tools.builtin import processes

if TYPE_CHECKING:
    from rich.console import Console

    from domyn.agents.interfaces.cli.theme import Theme


async def render(
    console: Console,
    theme: Theme,
    state: dict[str, Any] | None,
    backend: Any,
    conversation_id: str = "",
) -> dict[str, Any] | None:
    """List what this conversation still has running, asking the machine.

    One call, and it enumerates the process directory rather than the registry —
    so it also finds anything started before a crash, which are precisely the
    ones nothing else can reach.

    **Only what is still going.** A list that keeps yesterday's commands is one
    nobody reads to the end, and "what is running" is the question being asked.
    The ones that have ended are still *noticed*, though, and written back:
    knowing a command is over is what stops the prompt claiming otherwise, and
    what keeps it from counting against the cap.

    Returns the state to persist when something turned out to have finished.
    """
    known = processes.registry(state)
    seen: dict[str, dict[str, Any]] = {}
    if backend is not None:
        result = await backend.run(processes.build_discover(conversation_id), timeout=60)
        if not result.backend_error:
            seen = processes.parse_discover(result.stdout)

    changed = False
    running: list[tuple[str, dict[str, Any], Any]] = []
    unreachable: list[str] = []

    for handle in dict.fromkeys(list(seen) + list(known)):
        observed = seen.get(handle)
        record = known.get(handle)
        if observed is None:
            unreachable.append(handle)
            continue
        if not observed["running"]:
            if record is not None and not record.finished:
                record.finished = True
                processes.remember(state, record)
                changed = True
            continue
        running.append((handle, observed, record))

    if not running and not unreachable:
        console.print("  No commands running.", style=theme.style("meta"), highlight=False)
        return state if changed else None

    console.print()
    for handle in unreachable:
        console.print(f"  {handle}  (could not be checked)", style=theme.style("warning"))
    for handle, observed, record in running:
        console.print(
            f"  {handle}  running {_short(observed['elapsed_s'])}, "
            f"quiet for {_short(observed['idle_s'])}",
            style=theme.style("accent"),
            highlight=False,
        )
        # The command on its own line: it is what a person recognises the
        # process by, and it is usually too long to share one with the status.
        command = (record.command if record else "") or observed.get("command", "")
        console.print(f"      $ {_trim(command)}", style=theme.style("meta"), highlight=False)
        # The log, so it can be followed from another terminal without going
        # through the agent at all.
        where = record.dir if record else f"{processes.HOME}/{handle}"
        console.print(f"      {where}/out.log", style=theme.style("meta"), highlight=False)
        if record is None:
            console.print(
                "      (not in this session's records — started before it)",
                style=theme.style("meta"),
                highlight=False,
            )
    console.print()
    return state if changed else None


async def stop(
    console: Console,
    theme: Theme,
    state: dict[str, Any] | None,
    backend: Any,
    handle: str,
) -> dict[str, Any] | None:
    """Stop one command. Returns the state to persist, or ``None`` if unchanged."""
    known = processes.registry(state)
    if not handle:
        console.print(
            "  Which one? /stop <handle> — /ps lists them.",
            style=theme.style("warning"),
            highlight=False,
        )
        return None
    record = known.get(handle)
    if record is None:
        console.print(f"  No command with handle {handle}.", style=theme.style("warning"))
        return None
    if backend is None:
        console.print("  No shell backend in this session.", style=theme.style("warning"))
        return None

    await backend.run(processes.build_kill(record), timeout=60)
    record.finished = True
    processes.remember(state, record)
    console.print(f"  Stopped {handle}.", style=theme.style("warning"), highlight=False)
    return state


def summary(state: dict[str, Any] | None) -> str:
    """One phrase for the prompt indicator, or empty when there is nothing.

    Counts records rather than probing: this is drawn next to the prompt, and it
    must not run a shell command to do it. It can therefore be one turn stale,
    which is the right trade for something whose job is only to remind.
    """
    live = [r for r in processes.registry(state).values() if not r.finished]
    if not live:
        return ""
    if len(live) == 1:
        return "1 command running"
    return f"{len(live)} commands running"


def _short(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}"


def _trim(command: str, limit: int = 70) -> str:
    flat = " ".join(command.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
