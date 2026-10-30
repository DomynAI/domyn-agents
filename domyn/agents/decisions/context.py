"""Which old tool outputs this turn still needs.

Compaction today drops tool results by **age and size**: the oldest, biggest
ones go first. That is the right default — it is free and it cannot be wrong
about anything — but it is blind to the obvious: the 40 KB directory listing
from four turns ago is worth less than the test output from two, whatever their
sizes, if the task is now about that test.

So before compaction runs, a decision model is asked, once per turn, which of
the prunable tool outputs are **not needed for the task at hand**. The ids it
is confident about are stashed on ``context.state``, and the pruner — which
already knows how to drop a set of event ids, because that is how it replays a
past decision — drops those first. Everything else falls through to the size
rule exactly as before.

Three things that make this safe rather than clever:

* **It only ever prunes, never restores**, and a pruned output is replaced by a
  line saying what it was, so the model can re-run the command if it was wrong.
* **It judges a preview, and says so.** The full outputs are what compaction is
  trying to get rid of; sending all of them to be judged would cost more than
  it saves. Each question carries the tool name and the first few hundred
  characters, which is enough to tell a directory listing from a stack trace.
* **Only a confident no counts**, at the ladder's own high bar. Measured on
  this shape of question, the model is sure when it says drop (0.96-0.98) and
  unsure when it says keep (0.57-0.72) — so it is used only in the direction it
  is good at, which is more conservative than the usual proposal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from domyn.agents.decisions.models import noul
from domyn.agents.logger import get_logger

if TYPE_CHECKING:
    from domyn.agents.decisions.runtime import DecisionRuntime

USE_CASE = "context_ladder"
STATE_KEY = "context_irrelevant"

#: How much of an output the model is shown. Enough to tell what it is, far
#: short of what it would cost to send the whole thing.
PREVIEW_CHARS = 400

#: Outputs smaller than this are not worth a question: dropping one saves
#: nothing the size rule would not have left alone anyway.
MIN_RESULT_CHARS = 2_000


def irrelevant_ids(state: dict[str, Any] | None) -> set[str]:
    """Event ids a decision model judged irrelevant. Empty for anything unexpected."""
    if not isinstance(state, dict):
        return set()
    stored = state.get(STATE_KEY)
    if not isinstance(stored, (list, tuple, set)):
        return set()
    return {str(item) for item in stored}


async def decide(
    runtime: DecisionRuntime | None,
    *,
    history: list[Any],
    task: str,
    state: dict[str, Any] | None,
) -> set[str]:
    """Judge the prunable tool outputs in ``history``. Returns what to drop first.

    Never raises. With no model, no task, nothing big enough to judge, or an
    answer below the bar, it returns an empty set and compaction behaves exactly
    as it does today.
    """
    if runtime is None or not runtime.enabled(USE_CASE) or not task:
        return set()

    candidates = _candidates(history)
    if not candidates:
        return set()

    asked = candidates[:64]
    questions = {
        f"r{position}": noul(
            "An agent is working on a task. Below is the output of a tool it ran "
            f"earlier, from '{name}'. Is this output still needed to finish the "
            "task? Answer yes if there is any reasonable chance it is."
        )
        for position, (_, name, _) in enumerate(asked)
    }
    previews = {
        f"r{position}": {"tool": name, "output": preview}
        for position, (_, name, preview) in enumerate(asked)
    }
    decision = await runtime.ask(USE_CASE, {"task": task, "outputs": previews}, questions)

    threshold = runtime.threshold(USE_CASE)
    dropped: set[str] = set()
    for position, (event_id, _, _) in enumerate(asked):
        answer = decision.get(f"r{position}")
        if not runtime.may_act(USE_CASE, answer):
            continue
        if answer.is_false(threshold):
            dropped.add(event_id)

    if dropped and isinstance(state, dict):
        state[STATE_KEY] = sorted(dropped)
    if dropped:
        get_logger().info(
            f"decisions[context_ladder]: {len(dropped)}/{len(asked)} old tool "
            "outputs judged no longer needed"
        )
    return dropped


def _candidates(history: list[Any]) -> list[tuple[str, str, str]]:
    """``(event_id, tool name, preview)`` for every tool output worth judging."""
    from domyn.agents.core import ExecutionEventType

    found: list[tuple[str, str, str]] = []
    for event in history:
        if getattr(event, "event_type", None) is not ExecutionEventType.TOOL_END:
            continue
        event_id = getattr(event, "event_id", None)
        action = getattr(event, "action", None)
        observation = getattr(action, "observation", None)
        if not event_id or not isinstance(observation, str):
            continue
        if len(observation) < MIN_RESULT_CHARS:
            continue
        found.append(
            (str(event_id), str(getattr(action, "name", "a tool")), observation[:PREVIEW_CHARS])
        )
    return found
