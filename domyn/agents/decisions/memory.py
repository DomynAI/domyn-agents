"""Whether a turn holds something worth remembering about the user.

In ``AUTOMATIC`` mode a memory manager runs after every turn
(:meth:`~domyn.agents.memory.managers.base.BaseMemoryManager.update_memory`, in the
background), and without a decision model it asks the agent's LLM two things in
one call: whether to store anything, and what. Most turns hold nothing worth keeping, so most of
those calls end in "no".

With a decision model the two halves are split. The model answers the yes/no — a
single ``noul``, asked alone, on the turn's own exchange — and the LLM is called
only after a yes, to write the fact. A confident no skips the LLM altogether.

**Unknown is today's behaviour.** No runtime, the use case ``off`` or in
``shadow``, a timeout, an answer that is not confident either way: all of them
return :attr:`Verdict.UNDECIDED`, and the caller takes the path it took before
this existed, the LLM deciding and writing together. A decision model can only
replace that path, never break it.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from domyn.agents.core import ExecutionEventType
from domyn.agents.decisions.models import noul
from domyn.agents.decisions.runtime import USE_CASES, DecisionMode
from domyn.agents.logger import get_logger

if TYPE_CHECKING:
    from domyn.agents.decisions.runtime import DecisionRuntime

USE_CASE = "memory"

#: The longest exchange sent to be judged, each side. A fact about the user is
#: in what they wrote, rarely past its first few thousand characters, and a turn
#: that pasted a file must not be refused for being over the model's window.
MAX_SIDE_CHARS = 4_000

QUESTION = (
    "Below is one exchange between a user and an assistant, and what is already "
    "remembered about the user. Does the user reveal something new about "
    "themselves worth remembering in future conversations: who they are, a lasting "
    "preference, their situation or an ongoing goal? Answer no for questions, small "
    "talk, one-off requests and anything only relevant to this exchange, and answer "
    "no when everything worth remembering is already remembered."
)


class Verdict(StrEnum):
    """What the decision model said about the turn."""

    SAVE = "save"
    """Confidently yes: have the LLM write the memory."""

    SKIP = "skip"
    """Confidently no: store nothing, call nothing."""

    UNDECIDED = "undecided"
    """No usable answer: take the path without a decision model."""


async def decide(
    runtime: DecisionRuntime | None,
    *,
    user: str,
    assistant: str,
    known: Sequence[str] = (),
) -> Verdict:
    """Ask whether the exchange holds something new to remember. Never raises.

    ``known`` is what is already remembered (the caller's best matches): a fact
    among them is not new, which is what keeps a fact from being stored again on
    every turn that mentions it.
    """
    if runtime is None or not runtime.enabled(USE_CASE) or not user.strip():
        return Verdict.UNDECIDED

    state = {
        "user": user[:MAX_SIDE_CHARS],
        "assistant": assistant[:MAX_SIDE_CHARS],
        "already_remembered": list(known) or "nothing yet",
    }
    decision = await runtime.ask(USE_CASE, state, {"store": noul(QUESTION)})
    answer = decision.get("store")

    verdict = Verdict.UNDECIDED
    if runtime.may_act(USE_CASE, answer):
        threshold = runtime.threshold(USE_CASE)
        minimum = USE_CASES[USE_CASE].requires
        if answer.is_true(threshold, minimum=minimum):
            verdict = Verdict.SAVE
        elif answer.is_false(threshold, minimum=minimum):
            verdict = Verdict.SKIP

    mode = runtime.mode(USE_CASE)
    get_logger().info(
        f"decisions[memory]: mode={mode.value} value={answer.value} "
        f"confidence={answer.confidence:.2f} verdict={verdict.value}"
        + (" (shadow: not acted on)" if mode is DecisionMode.SHADOW else "")
    )
    return verdict


def last_exchange(context: Any) -> tuple[str, str]:
    """The text of the turn's last user message and of the replies after it.

    Read from the conversation and the current interaction together, since which
    of the two holds the turn when the hook runs is the runner's business. A
    response converted to a hand-off is still text the user was shown, so it
    counts; tokens are not stored, the whole ``RESPONSE`` is.
    """
    events = [
        *(getattr(context, "conversation_history", None) or []),
        *(getattr(context, "interaction_history", None) or []),
    ]
    start = None
    for position in range(len(events) - 1, -1, -1):
        if getattr(events[position], "event_type", None) is ExecutionEventType.USER_INPUT:
            start = position
            break
    if start is None:
        agent_input = getattr(context, "agent_input", None)
        return _text(agent_input), ""

    seen: set[str] = set()
    replies: list[str] = []
    for event in events[start + 1 :]:
        if getattr(event, "event_type", None) is not ExecutionEventType.RESPONSE:
            continue
        event_id = getattr(event, "event_id", None)
        if event_id is not None:
            if event_id in seen:
                continue
            seen.add(event_id)
        text = _text(event)
        if text:
            replies.append(text)
    return _text(events[start]), "\n\n".join(replies)


def _text(event: Any) -> str:
    parts = getattr(event, "content", None) or []
    return "\n".join(part.text for part in parts if getattr(part, "text", None)).strip()
