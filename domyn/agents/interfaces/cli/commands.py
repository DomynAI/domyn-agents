"""Slash-command specifications — the single source of truth.

``/help`` and the autocompleter are both generated from ``COMMANDS``. Keeping one
list avoids the failure mode where the help text and the completion list drift
apart, which is exactly what happens when they are maintained separately.

This module holds only *specifications*: no handlers, no session state. The REPL
owns the behaviour and looks commands up by canonical name, so nothing here has
to import the runner or the agent.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SlashCommand:
    name: str
    summary: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    usage: str | None = None
    mid_turn: bool = False
    """Whether this command may run while a turn is still in flight.

    True only for commands that print, or flip a flag this process reads later.
    Anything that swaps the conversation id, writes conversation state, re-enters
    the runner, or reads the terminal itself is False and gets held until the turn
    ends — running it alongside a live ``run()`` would change state the runner is
    still using. Default False, so a command added later is held rather than
    silently allowed.
    """


COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand("/help", "Show this help", aliases=("/?",), mid_turn=True),
    SlashCommand("/exit", "End the session", aliases=("/quit", "/q")),
    SlashCommand("/clear", "Start a fresh conversation and clear the screen"),
    SlashCommand("/agent", "Show the active agent, its tools and sub-agents", mid_turn=True),
    SlashCommand("/skills", "List the skills available to the agent", mid_turn=True),
    SlashCommand("/skill", "Show one skill in full", usage="/skill <name>", mid_turn=True),
    SlashCommand(
        "/skills-autodiscover",
        "Show, or turn on or off, loading skills from ./.domyn/skills and ./.agents/skills",
        usage="/skills-autodiscover [on|off]",
    ),
    SlashCommand("/auto", "Approve every tool call for the rest of the session", mid_turn=True),
    SlashCommand(
        "/approvals",
        "Review and remove standing approvals",
        usage="/approvals [revoke <n> | clear]",
    ),
    SlashCommand("/verbose", "Toggle internal events and library logs"),
    SlashCommand("/voice", "Talk instead of typing: hold space to speak, let go to send"),
    SlashCommand("/model", "Change provider and model"),
    SlashCommand("/config", "Show the config file backing this session", mid_turn=True),
    SlashCommand(
        "/export",
        "Save this configuration, its skills and instructions into a new folder to share",
        usage="/export <folder>",
    ),
    SlashCommand(
        "/context", "Show how much of the model's context window is in use", mid_turn=True
    ),
    SlashCommand("/compact", "Shorten the conversation now, without waiting for the limit"),
    SlashCommand("/ps", "Show commands still running in the background", mid_turn=True),
    SlashCommand("/stop", "Stop a running command", usage="/stop <handle>"),
    SlashCommand(
        "/resume",
        "Continue a past conversation",
        usage="/resume [conversation_id]",
    ),
)

# Why the ones that look harmless are not mid_turn:
#   /clear     swaps conversation_id while the runner is still writing events under
#              the old one, and sends the next steering message to a conversation
#              nobody is running.
#   /resume    swaps it for the same reason, and reads the terminal to offer a choice.
#   /compact   re-enters Runner.run on the same conversation, which collides in
#              injection_queue — keyed by (application, conversation).
#   /model     rebuilds the agent, and opens a second reader of the terminal.
#   /skills-autodiscover  rewrites config.yaml and rebuilds the agent.
#   /export    writes a folder, and may first rebuild the agent from a changed config.
#   /voice     may download and load a speech model, and changes what the space
#              bar does in the editor that is reading.
#   /verbose   replaces the renderer mid-stream, handing the next finish() one whose
#              "a stream is open" flag is False while the old one still has tokens
#              arriving.
# /approvals stays non-mid_turn even in its read-only bare form, so the answer is a
# per-command property rather than a per-argument special case.

_BY_NAME: dict[str, SlashCommand] = {}
for _command in COMMANDS:
    _BY_NAME[_command.name] = _command
    for _alias in _command.aliases:
        _BY_NAME[_alias] = _command


def completion_names() -> list[str]:
    """Names offered by the autocompleter: canonical first, then aliases."""
    return [c.name for c in COMMANDS] + [a for c in COMMANDS for a in c.aliases]


def resolve(text: str) -> tuple[SlashCommand, str] | None:
    """Map raw input to ``(command, argument)``.

    Returns ``None`` when the input is not a slash command, so the caller can
    treat it as a prompt for the agent. An unknown ``/word`` also returns
    ``None``-like behaviour via a missing entry, letting the caller report it
    rather than silently sending a typo'd command to the model.
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    head, _, rest = stripped.partition(" ")
    command = _BY_NAME.get(head.lower())
    if command is None:
        return None
    return command, rest.strip()


def is_slash(text: str) -> bool:
    return text.strip().startswith("/")


def help_lines() -> list[tuple[str, str]]:
    """``(invocation, summary)`` pairs for rendering help."""
    rows: list[tuple[str, str]] = []
    for command in COMMANDS:
        invocation = command.usage or command.name
        if command.aliases:
            invocation = f"{invocation}  ({', '.join(command.aliases)})"
        rows.append((invocation, command.summary))
    return rows
