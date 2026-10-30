"""Interactive approval prompt for tools gated by ``approval_type``.

The framework already implements the whole gate: the runner emits
``TOOL_NEEDS_FEEDBACK`` before executing a gated tool and suspends, and resuming
with a ``TOOL_FEEDBACK`` event either runs the tool or feeds a refusal back to
the model. What was missing was a consumer — this module.

Three details are deliberate:

* An edited parameter is coerced against its *declared* type, falling back to
  the raw string rather than crashing a live session. ``type(current_value)(new_value)``
  would raise when the current value is ``None`` (``NoneType("42")``).
* Ctrl-C and Ctrl-D mean "deny", which is the safe interpretation, so there is
  always a way out of the prompt.
* Required parameters are told apart from optional ones while editing.

Approving is not a yes/no question but a *scope* question. Asking again for
every identical call is what trains a user to stop reading the prompt, which is
worse than not asking at all, so the choices, picked with the arrow keys, are
"just this once / for the rest of this conversation / always / no". Only the scopes the runner can actually
store are offered — a choice that silently does nothing is worse than no choice.

What gets remembered is a **subject**, not a tool: the runner computes it and
sends it along with the request, and this module shows that exact string as the
heading whenever it fits on one line. If the prompt described the call
differently from what the grant matches on, the user would be approving
something other than what they read. A subject too long for one line (a long or
multi-line command) gives way to the tool's name, with the arguments it is built
from shown in full underneath — once, never as a flattened copy (``calls.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import indent
from typing import TYPE_CHECKING, Any

from rich.cells import cell_len
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from domyn.agents.approvals import ApprovalGrant, ApprovalRequest, is_already_approved
from domyn.agents.core import (
    ApprovalType,
    BaseEvent,
    ExecutionEventType,
    FeedbackAction,
    FeedbackParameters,
)
from domyn.agents.interfaces.cli import select as picker
from domyn.agents.interfaces.cli.calls import Diff, Field, approval_view, as_written
from domyn.agents.interfaces.cli.prompt import read_line

if TYPE_CHECKING:
    from rich.console import Console

    from domyn.agents.interfaces.cli.theme import Theme


def _coerce(raw: str, declared_type: str | None, current: Any) -> Any:
    """Best-effort conversion of typed console input.

    Never raises: a value we cannot convert is passed through as a string, and
    the tool's own validation decides. Crashing an interactive session over a
    typo would be a worse outcome than a rejected argument.
    """
    name = (declared_type or "").lower() or type(current).__name__.lower()

    try:
        if name in ("int",):
            return int(raw)
        if name in ("float",):
            return float(raw)
        if name in ("bool",):
            return raw.strip().lower() in ("1", "true", "yes", "y", "on")
    except ValueError:
        return raw
    return raw


def _render_params(console: Console, theme: Theme, action: FeedbackAction) -> None:
    """Every parameter as a form — name, type, required — for a call about to be edited."""
    if not action.parameters:
        console.print("  (no parameters)", style=theme.style("meta"), highlight=False)
        return
    for param in action.parameters:
        flag = "*" if param.is_required else " "
        label = f"  {flag} {param.name} ({param.type}) ="
        value = as_written(param.value, multiline=True)
        # Over several lines it starts below the label, so code keeps its indentation.
        text = f"{label}\n{indent(value, ' ' * 6)}" if "\n" in value else f"{label} {value}"
        # As text, whatever console was handed in: `[/]` is not markup, `:a:` no emoji.
        console.print(text, style=theme.style("meta"), highlight=False, markup=False, emoji=False)


# Only the heading is shortened; the subject is the grant key.
_HEADING_CHARS = 200


def _shown(subject: str) -> str:
    shown = as_written(subject)
    if len(shown) <= _HEADING_CHARS:
        return shown
    return f"{shown[: _HEADING_CHARS - 1]}… (+{len(shown) - _HEADING_CHARS + 1} chars)"


# Under the heading, so the body reads as part of the question.
_BODY = 4


def _width(console: Console) -> int:
    width = getattr(console, "width", None)
    return width if isinstance(width, int) else 80


def _render_body(console: Console, theme: Theme, body: list[Field | Diff]) -> None:
    """The arguments the heading does not show, in full: text, never markup."""
    meta = theme.style("meta")
    for item in body:
        if isinstance(item, Diff):
            marks = {"-": "diff_removed", "+": "diff_added"}
            # Two columns, so a wrapped line continues under its text, not its mark.
            grid = Table.grid()
            grid.add_column(width=2, no_wrap=True)
            grid.add_column()
            for line in item.lines:
                style = theme.style(marks.get(line.mark, "diff_kept"))
                grid.add_row(Text(f"{line.mark} ", style=style), Text(line.text, style=style))
            console.print(Padding(grid, (0, 0, 0, _BODY), expand=False))
        elif item.name is None:
            console.print(Padding(Text(item.text), (0, 0, 0, _BODY), expand=False))
        elif "\n" in item.text:
            console.print(
                Padding(Text(f"{item.name}:", style=meta), (0, 0, 0, _BODY), expand=False)
            )
            console.print(Padding(Text(item.text), (0, 0, 0, _BODY + 2), expand=False))
        else:
            field = Text.assemble((f"{item.name}: ", meta), item.text)
            console.print(Padding(field, (0, 0, 0, _BODY), expand=False))


# How long each scope the runner may offer lasts, narrowest first. Anything the
# runner reports that is not here is skipped rather than shown untranslated.
_SCOPE_WHEN: dict[str, str] = {
    "conversation": "for the rest of this conversation",
    "project": "in this project",
    "user": "always",
    "global": "for everyone",
}
_SCOPE_ORDER = ("conversation", "project", "user", "global")

# Typed answers (the numbered fallback, the pattern) are re-asked on invalid
# input, and need a floor: a caller whose stdin keeps returning the same unusable
# answer — a script, a stuck pipe, a test — would otherwise spin forever holding
# the turn open. After this many tries the safest answer is taken: deny for the
# gate, "just once" for the pattern. Never the permissive one.
_MAX_RETRIES = 3


@dataclass
class ApprovalMenu:
    """What to ask about one gated call, independent of who draws the question.

    Built by :func:`begin_approval` and answered by :func:`settle_approval`, so
    the blocking prompt below and the REPL's live editor ask exactly the same
    question and read the answer the same way.

    "no" is last and every "yes" says what it remembers, so nothing reads as
    "remember that I refused". The pattern option comes after the exact ones,
    with the glob spelled out, because it is the only choice that grants more
    than the line above it — an exact approval that repeats is a nuisance, a
    pattern approved without reading is a permission nobody gave.
    """

    question: str
    options: list[str]
    # Per option: (approved, scope to remember, whether it is the pattern).
    outcomes: list[tuple[bool, str | None, bool]]
    pattern: str | None
    request: ApprovalRequest

    def edits_pattern(self, picked: int | None) -> bool:
        """Whether settling this answer reads a line: the pattern is editable."""
        return picked is not None and self.outcomes[picked][2]


def _menu(
    question: str, available: list[str], pattern: str | None, request: ApprovalRequest
) -> ApprovalMenu:
    offered = [s for s in _SCOPE_ORDER if s in available]
    options = ["yes, just this once"]
    outcomes: list[tuple[bool, str | None, bool]] = [(True, None, False)]
    for scope in offered:
        options.append(f"yes, this exact call, {_SCOPE_WHEN[scope]}")
        outcomes.append((True, scope, False))
    if pattern and offered:
        # A pattern is stored in the narrowest scope offered.
        options.append(f"yes, anything matching  {pattern}  {_SCOPE_WHEN[offered[0]]}")
        outcomes.append((True, offered[0], True))
    options.append("no")
    outcomes.append((False, None, False))
    return ApprovalMenu(question, options, outcomes, pattern, request)


def _edit_pattern(
    console: Console, theme: Theme, proposed: str, request: ApprovalRequest
) -> str | None:
    """Confirm or edit the glob before it becomes a rule.

    Asked again while it misses the call (``python *`` for ``$ python a.py``):
    such a rule would never apply.
    """
    for _ in range(_MAX_RETRIES):
        try:
            raw = read_line(
                console, f"  pattern [{proposed}]: ", style=theme.style("approval")
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return None
        edited = raw or proposed
        # The policy's own check, so the two cannot disagree.
        if is_already_approved(
            request, [ApprovalGrant(value=request.pattern_key(edited), kind="pattern")]
        ):
            return edited
        console.print(
            f"  {edited} does not cover {_shown(request.subject)}, so it would never apply",
            style=theme.style("warning"),
            highlight=False,
            markup=False,
            emoji=False,
        )
    return None


def _edit_params(
    console: Console, theme: Theme, action: FeedbackAction
) -> list[FeedbackParameters]:
    console.print(
        "  Enter new values, empty keeps the current one:",
        style=theme.style("meta"),
        highlight=False,
    )
    updated: list[FeedbackParameters] = []
    for param in action.parameters:
        try:
            raw = read_line(console, f"  {param.name} [{as_written(param.value)}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            # Keep every remaining parameter untouched rather than losing the
            # edits already made.
            raw = ""
        value = _coerce(raw, param.type, param.value) if raw else param.value
        updated.append(
            FeedbackParameters(
                name=param.name,
                value=value,
                type=param.type,
                is_required=param.is_required,
            )
        )
    return updated


def _feedback_action(event: BaseEvent) -> Any:
    action: Any = event.action
    if not isinstance(action, FeedbackAction):  # pragma: no cover - defensive
        raise TypeError(
            f"expected a FeedbackAction on TOOL_NEEDS_FEEDBACK, got {type(action).__name__}"
        )
    return action


def begin_approval(console: Console, theme: Theme, event: BaseEvent) -> ApprovalMenu:
    """Show what is being asked, and return the question to put to the user.

    Reads lines itself for a ``PARAMS_UPDATE`` call — the new values come before
    the question — so a caller with its own editor on screen has to stand it down
    first for those.
    """
    action = _feedback_action(event)
    subject = str(event.metadata.get("approval_subject") or action.name)
    scopes = [str(s) for s in (event.metadata.get("approval_scopes") or [])]
    parts = event.metadata.get("approval_parts")
    request = ApprovalRequest(
        tool_name=str(action.name),
        agent_name=event.author,
        subject=subject,
        parts=[str(p) for p in parts] if isinstance(parts, list) else None,
    )

    lead = f"{theme.symbols.bullet} approval required: "
    editing = action.approval_type == ApprovalType.PARAMS_UPDATE
    body: list[Field | Diff] = []
    if editing:
        # About to be edited: every parameter, as a form.
        heading = _shown(subject)
    else:
        params = {param.name: param.value for param in action.parameters}
        heading, body = approval_view(
            str(action.name), params, subject, room=_width(console) - cell_len(lead)
        )

    console.print()
    # A Text: `[/]` in a command is not markup, `:a:` no emoji.
    console.print(Text(lead + heading, style=theme.style("approval")))
    if action.thought:
        console.print(f"  {action.thought}", style=theme.style("thought"), highlight=False)
    if editing:
        _render_params(console, theme, action)
        action.parameters = _edit_params(console, theme, action)
        question = "run with these parameters?"
    else:
        _render_body(console, theme, body)
        question = "run it?"

    return _menu(
        question, scopes, str(event.metadata.get("approval_pattern") or "") or None, request
    )


def settle_approval(
    console: Console, theme: Theme, event: BaseEvent, menu: ApprovalMenu, picked: int | None
) -> BaseEvent:
    """Turn the option picked (``None``: aborted) into the runner's feedback.

    Aborting denies, as everywhere else in this prompt: an interrupted choice
    must never grant anything. The pattern option reads one more line, the glob
    itself — editable, because the proposal is only a guess; giving up on it
    keeps the yes, narrowed to this call.
    """
    action = _feedback_action(event)
    approved, scope, is_pattern = (
        menu.outcomes[picked] if picked is not None else (False, None, False)
    )
    pattern: str | None = None
    if is_pattern and menu.pattern is not None:
        pattern = _edit_pattern(console, theme, menu.pattern, menu.request)
        if pattern is None:
            scope = None
    action.approve = approved
    action.remember_scope = scope
    action.remember_pattern = pattern

    sym = theme.symbols
    if approved and pattern:
        outcome = f"  {sym.approved} approved, remembering {pattern}"
    elif approved and scope:
        outcome = f"  {sym.approved} approved, remembered ({scope})"
    elif approved:
        outcome = f"  {sym.approved} approved"
    else:
        outcome = f"  {sym.denied} denied"
    console.print(
        outcome,
        style=theme.style("approval") if approved else theme.style("warning"),
        highlight=False,
    )
    console.print()

    return BaseEvent(
        event_type=ExecutionEventType.TOOL_FEEDBACK,
        author="user",
        action=action,
    )


def prompt_for_approval(
    console: Console,
    theme: Theme,
    event: BaseEvent,
) -> BaseEvent:
    """Ask the human to approve, edit or deny a gated tool call, blocking.

    The arrow-key picker on a terminal, a numbered list on a pipe. The REPL uses
    this only when its own editor is not running alongside the turn; otherwise
    it puts the same :class:`ApprovalMenu` inside that editor.

    Args:
        console: rich console to draw on.
        theme: resolved theme.
        event: the ``TOOL_NEEDS_FEEDBACK`` event emitted by the runner.

    Returns:
        A ``TOOL_FEEDBACK`` event to resume the runner with. Denial is expressed
        as ``approve=False``; the runner turns that into a refusal observation
        the model can read, so the conversation continues coherently instead of
        dead-ending.
    """
    menu = begin_approval(console, theme, event)
    once, no = menu.options[0], menu.options[-1]
    chosen = picker.select(
        console,
        theme,
        menu.question,
        menu.options,
        aliases={"y": once, "yes": once, "n": no, "no": no},
        max_tries=_MAX_RETRIES,
    )
    picked = menu.options.index(chosen) if chosen is not None else None
    return settle_approval(console, theme, event, menu, picked)
