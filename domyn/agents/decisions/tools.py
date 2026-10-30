"""Deciding which tools the model is shown this turn.

Three different things get called "choosing a tool", and only two of them belong
to a decision model:

======================================  ====================================
which tools the model **sees**          this module
whether **this call** may run           ``decisions/approvals.py``
which tool to call                      the LLM's job, not ours
======================================  ====================================

The third is deliberately absent. Picking the tool is the decision the agent's
own model has the whole conversation for, and a decision model would see a
summary we prepared. Worse, it has no cautious direction: there is no "when in
doubt, ask" as there is at the approval gate, only the right tool and the wrong
one — no threshold, no safe abstention.

**This one may only hide, and only when sure.** Hiding is irreversible within
the turn, and a tool the agent needed and cannot see produces "I can't read
files" instead of a question. So the rule is the mirror image of the approval
gate's: there the base is ASK and *allowing* needs a real number; here the base
is "the tool is there" and *hiding* does.

**Decided once per user turn**, in the runner, and stashed on
``context.state``. ``get_tool_definitions`` is synchronous and runs on every
request of every iteration; worse, tool definitions sit at the front of the
prompt, so changing them mid-turn throws away the prefix cache the whole turn
is paying for. One decision, held for the turn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from domyn.agents.decisions.models import noul
from domyn.agents.logger import get_logger

if TYPE_CHECKING:
    from domyn.agents.decisions.runtime import DecisionRuntime

USE_CASE = "tools"
STATE_KEY = "tool_visibility"

#: Tools that are never hidden, whatever anyone decides. The gate cannot be
#: routed around by hiding what it protects, and the agent's own protocol tools
#: are not optional.
ALWAYS_VISIBLE: frozenset[str] = frozenset({"skill_view", "stop", "forward_verbatim"})

#: Below this many tools, hiding saves tokens nobody was short of and costs a
#: round trip. The case for it starts at a large catalogue.
MIN_TOOLS = 12


class Visibility(NamedTuple):
    """What this turn decided, cached on ``context.state``."""

    hidden: tuple[str, ...]
    turn_id: str | None

    def applies_to(self, turn_id: str | None) -> bool:
        return self.turn_id == turn_id


def hidden_for(context: Any) -> frozenset[str]:
    """The tool names hidden for the turn this context is on.

    Reads only what the current turn stored: a decision taken two turns ago is
    about a task that is over. Returns an empty set for everything unexpected,
    which is the behaviour with none of this.
    """
    state = getattr(context, "state", None)
    if not isinstance(state, dict):
        return frozenset()
    stored = state.get(STATE_KEY)
    if not isinstance(stored, Visibility):
        return frozenset()
    if not stored.applies_to(getattr(context, "turn_id", None)):
        return frozenset()
    return frozenset(stored.hidden)


async def decide(
    runtime: DecisionRuntime | None,
    *,
    tools: list[Any],
    task: str,
    context: Any,
) -> Visibility | None:
    """Work out what to hide this turn and stash it. ``None`` when nothing was decided.

    Never raises. Every uncertainty — no model, no task, a small catalogue, an
    unanswered question, an answer below the bar — leaves the tool visible.
    """
    if runtime is None or not runtime.enabled(USE_CASE) or not task:
        return None

    candidates = [
        tool for tool in tools if getattr(tool, "name", None) and tool.name not in ALWAYS_VISIBLE
    ]
    if len(tools) < MIN_TOOLS or not candidates:
        return None

    asked = candidates[:64]
    questions = {
        f"t{position}": noul(
            "An agent is about to work on a task. Could the tool "
            f"'{tool.name}' — {_describe(tool)} — plausibly be needed for it? "
            "Answer yes if there is any reasonable chance."
        )
        for position, tool in enumerate(asked)
    }
    decision = await runtime.ask(USE_CASE, {"task": task}, questions)

    threshold = runtime.threshold(USE_CASE)
    hidden: list[str] = []
    for position, tool in enumerate(asked):
        answer = decision.get(f"t{position}")
        if not runtime.may_act(USE_CASE, answer):
            continue
        # `is_false`, not `not is_true`: the model has to be confident the tool
        # is *not* needed. Measured on the same shape of question, it is sure
        # when it says hide (0.96-0.98) and unsure when it says keep
        # (0.57-0.72), so this is the direction it is actually good at.
        if answer.is_false(threshold):
            hidden.append(tool.name)

    if not hidden:
        return None

    visibility = Visibility(hidden=tuple(hidden), turn_id=getattr(context, "turn_id", None))
    state = getattr(context, "state", None)
    if isinstance(state, dict):
        state[STATE_KEY] = visibility
    get_logger().info(f"decisions[tools]: {len(hidden)}/{len(asked)} tools hidden for this turn")
    return visibility


def _describe(tool: Any) -> str:
    text = str(getattr(tool, "description", "") or "").strip().replace("\n", " ")
    return text[:200] or "no description"
