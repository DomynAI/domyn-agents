"""Standing instructions that only apply sometimes.

A ``SOUL.md`` grows: a line about SQL migrations, a line about how to write a
commit message, a line about which dashboard to check. Each one is right in its
moment and noise in every other turn, and all of them are paid for on every
request, because standing instructions live in the system prompt.

A section can therefore say when it applies:

.. code-block:: markdown

    ## SQL migrations
    when: the task involves a database migration or a schema change

    Always check the rollback path before the forward one.

The ``when:`` line is the first non-empty line under a ``##`` heading. It is
read, removed from the text, and turned into one ``noul`` question. Sections
without it are unconditional, which is every section anyone has written so far —
**a file that uses none of this behaves exactly as it does today**, which is
also what happens with no decision model, with the use case off, or when the
model cannot answer.

Two deliberate restrictions:

* **Decided once per user turn, never inside the loop.** This text is in the
  system prompt, the part providers cache; changing it mid-turn would throw the
  prefix away on every iteration to save a few hundred tokens.
* **Nothing is cut silently.** What was left out is recorded on the result, as
  the 32 KiB budget already does with what it drops, so ``/config`` can show it.
  An instruction that vanished with no way to find out is indistinguishable
  from one the model decided to ignore.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, NamedTuple

from domyn.agents.instructions.models import AgentInstructions, Instruction, InstructionKind
from domyn.agents.logger import get_logger

if TYPE_CHECKING:
    from domyn.agents.decisions import DecisionRuntime

USE_CASE = "instructions"

_HEADING = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_WHEN = re.compile(r"^\s*when:\s*(.+?)\s*$", re.IGNORECASE)


class Section(NamedTuple):
    """One ``##`` section of an instruction, and the condition it declares."""

    heading: str
    body: str
    condition: str | None
    """``None`` for a section that always applies."""

    @property
    def conditional(self) -> bool:
        return self.condition is not None

    def render(self) -> str:
        return f"## {self.heading}\n{self.body}".rstrip()


class Resolved(NamedTuple):
    """An instruction's text after the conditions were applied."""

    text: str
    kept: tuple[str, ...]
    """Headings of the conditional sections that were kept."""
    skipped: tuple[str, ...]
    """Headings left out, so a caller can say so rather than lose them quietly."""

    @property
    def changed(self) -> bool:
        return bool(self.skipped)


def split_sections(text: str) -> tuple[str, list[Section]]:
    """The preamble before the first ``##``, and the sections after it."""
    matches = list(_HEADING.finditer(text))
    if not matches:
        return text, []

    preamble = text[: matches[0].start()].rstrip()
    sections: list[Section] = []
    for position, match in enumerate(matches):
        start = match.end()
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        body = text[start:end].strip("\n")
        condition, body = _take_condition(body)
        sections.append(Section(heading=match.group(1).strip(), body=body, condition=condition))
    return preamble, sections


def _take_condition(body: str) -> tuple[str | None, str]:
    """Pull a leading ``when:`` line off a section body.

    Only the first non-empty line is considered: a ``when:`` further down is
    prose about something else, and treating it as a directive would silently
    change what a file means.
    """
    lines = body.split("\n")
    for position, line in enumerate(lines):
        if not line.strip():
            continue
        match = _WHEN.match(line)
        if match is None:
            return None, body
        remainder = "\n".join(lines[position + 1 :]).strip("\n")
        return match.group(1), remainder
    return None, body


async def resolve(text: str, task: str, runtime: DecisionRuntime | None) -> Resolved:
    """Keep the sections that apply to ``task``.

    Never raises, and never keeps less than today when it cannot tell: no
    runtime, the use case off, a model that will not answer, or an answer below
    the threshold all keep the section. **The direction is the safety property**
    — dropping an instruction the user wrote is a change they did not ask for,
    so it takes a confident yes, while keeping one costs only tokens.
    """
    preamble, sections = split_sections(text)
    conditional = [section for section in sections if section.conditional]
    if not conditional or runtime is None or not runtime.enabled(USE_CASE):
        return Resolved(text=text, kept=(), skipped=())

    from domyn.agents.decisions import noul

    questions = {
        f"c{position}": noul(
            "An agent is about to work on a task. Does this instruction apply to it? "
            f"The instruction applies when: {section.condition}"
        )
        for position, section in enumerate(conditional)
        if position < 64
    }
    decision = await runtime.ask(USE_CASE, {"task": task}, questions)

    threshold = runtime.threshold(USE_CASE)
    skipped: set[str] = set()
    kept: list[str] = []
    for position, section in enumerate(conditional):
        answer = decision.get(f"c{position}")
        if not runtime.may_act(USE_CASE, answer):
            kept.append(section.heading)  # unknown keeps it: today's behaviour
            continue
        # `is_false`, not `not is_true`: an answer that knows nothing either
        # way keeps the section, which is the behaviour without any of this.
        if answer.is_false(threshold):
            skipped.add(section.heading)
        else:
            kept.append(section.heading)

    if not skipped:
        return Resolved(text=text, kept=tuple(kept), skipped=())

    body = "\n\n".join(section.render() for section in sections if section.heading not in skipped)
    rebuilt = f"{preamble}\n\n{body}".strip() if preamble else body
    get_logger().info(
        f"instructions: {len(skipped)} conditional section(s) did not apply this turn"
    )
    return Resolved(text=rebuilt, kept=tuple(kept), skipped=tuple(sorted(skipped)))


async def apply(
    loaded: AgentInstructions, task: str, runtime: DecisionRuntime | None
) -> AgentInstructions:
    """Resolve every instruction's conditional sections for this turn.

    Returns the original object untouched when there is nothing to decide, so
    the common case — no decision model, or no ``when:`` anywhere — costs one
    attribute check and allocates nothing.
    """
    if runtime is None or not runtime.enabled(USE_CASE) or not task:
        return loaded

    skipped: list[str] = []
    # `Any` because the update below carries both the per-kind tuples of
    # `Instruction` and `skipped_sections`, a tuple of `str`. Typed to the
    # narrower one, the merge is what mypy reports rather than the field.
    rebuilt: dict[str, Any] = {}
    changed = False
    for kind in InstructionKind:
        parts = getattr(loaded, kind.value)
        new_parts: list[Instruction] = []
        for one in parts:
            resolved = await resolve(one.text, task, runtime)
            if resolved.changed:
                changed = True
                skipped.extend(f"{one.name}: {heading}" for heading in resolved.skipped)
                new_parts.append(one.model_copy(update={"text": resolved.text}))
            else:
                new_parts.append(one)
        rebuilt[kind.value] = tuple(new_parts)

    if not changed:
        return loaded
    return loaded.model_copy(update={**rebuilt, "skipped_sections": tuple(skipped)})
