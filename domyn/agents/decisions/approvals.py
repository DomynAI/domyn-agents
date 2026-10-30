"""Letting a decision model have an opinion about a tool call.

The gate's order of business is fixed in ``ApprovalPolicy`` and that order *is*
the security model: hardline deny, the user's deny rules, a stored grant, then
whatever has an opinion, then ask the human. A stored grant returns immediately
and never reaches this code. So the only case a model ever speaks in is the one
whose answer would otherwise be **"ask the user"** — it can add friction, and it
can remove a question the user would have been asked, but it can never override
a hardline rule, a deny rule, or a permission the user already gave.

Two rules, and they point in opposite directions on purpose:

* **Allowing removes a human confirmation**, so it needs a real probability
  (``Provenance.LOGPROBS`` or better) over the use case's threshold. Measured:
  an LLM's self-reported confidence put its wrong answers at 0.78-0.85, above
  any threshold worth having, while the decision model's single mistake sat at
  0.53 against 0.83-0.98 for its correct ones.
* **Denying adds friction**, so a weaker number may do it. The cost of being
  wrong is an unnecessary refusal, not an unreviewed command.

One question per call, asked alone. Measured on this exact use case: putting
four supporting ``noul`` questions beside the verdict flipped ``rm -rf build/``
from ``ask`` (0.59) to ``allow`` (0.53), reproducibly, and moved every other
confidence toward the middle. The questions share one forward pass, so asking
"is this read-only?" next to the verdict is not free — it changes the verdict.
"""

from typing import Any, NamedTuple

from domyn.agents.approvals import ApprovalRequest, ApprovalVerdict
from domyn.agents.decisions.models import Answer, Decision, choice
from domyn.agents.decisions.runtime import DecisionMode, DecisionRuntime

USE_CASE = "approvals"
VERDICT = "verdict"

#: The only question asked. Named options rather than a score: the gate has
#: three outcomes and they are not points on a scale.
VERDICT_QUESTION = choice(
    "A coding agent wants to run this. What should the approval gate do?",
    {
        "allow": "Harmless and reversible: run it without interrupting the user.",
        "ask": "Potentially destructive, or it touches things outside the workspace: "
        "ask the user to confirm.",
        "deny": "Dangerous, or it exfiltrates secrets or credentials: refuse it.",
    },
)


class ApprovalConsultation(NamedTuple):
    """What came back, and what was done with it."""

    verdict: ApprovalVerdict | None
    """The verdict to hand the policy, or ``None`` to leave the decision alone.
    Always ``None`` in shadow mode, whatever the model said."""

    answer: Answer
    decision: Decision
    mode: DecisionMode
    acted: bool


def build_state(request: ApprovalRequest) -> dict[str, Any]:
    """What the model is shown about this call.

    The subject and the parts, because those are what the gate itself judges and
    what a grant would be keyed on — judging anything else would mean the model
    answers about a different string than the one being approved. The tool and
    agent names give it the difference between a shell line and a file write.
    """
    state: dict[str, Any] = {"tool": request.tool_name, "command": request.subject}
    if request.parts:
        state["parts"] = request.parts
    if request.agent_name:
        state["agent"] = request.agent_name
    return state


async def consult(
    runtime: DecisionRuntime | None, request: ApprovalRequest
) -> ApprovalConsultation:
    """Ask the model about this call, if there is one and it is switched on.

    Never raises, and never returns a verdict that the configuration did not
    authorise. A caller can hand the verdict straight to ``ApprovalPolicy``
    without re-checking anything.
    """
    nothing = Answer.unknown("not asked")
    if runtime is None or not runtime.enabled(USE_CASE):
        return ApprovalConsultation(None, nothing, Decision(), DecisionMode.OFF, False)

    mode = runtime.mode(USE_CASE)
    decision = await runtime.ask(USE_CASE, build_state(request), {VERDICT: VERDICT_QUESTION})
    answer = decision.get(VERDICT)

    verdict = _verdict_for(runtime, answer) if mode is DecisionMode.ACTIVE else None
    return ApprovalConsultation(
        verdict=verdict,
        answer=answer,
        decision=decision,
        mode=mode,
        acted=verdict is not None,
    )


def _verdict_for(runtime: DecisionRuntime, answer: Answer) -> ApprovalVerdict | None:
    """The verdict this answer earns, or ``None`` to stay out of it."""
    if not answer.known or not isinstance(answer.value, str):
        return None

    confidence = f"{answer.confidence:.2f}"
    rule = f"decision:{answer.value}:{confidence}"

    if answer.value == "deny":
        # Adding friction: a weaker number is allowed to do this, because being
        # wrong costs a refusal the user can override, not an unreviewed command.
        if runtime.may_act_cautiously(USE_CASE, answer):
            return ApprovalVerdict.deny(
                reason="A decision model judged this call dangerous.", rule=rule
            )
        return None

    if answer.value == "allow":
        # Removing a confirmation: the full bar, provenance included.
        if runtime.may_act(USE_CASE, answer):
            return ApprovalVerdict.allow(
                reason="A decision model judged this call safe.", rule=rule
            )
        return None

    # "ask" is what happens anyway. Returning an explicit ASK would be the same
    # outcome, but it would also be indistinguishable in the logs from the model
    # having nothing to say, and those are worth telling apart.
    return None


def event_metadata(use_case: str, consultation: ApprovalConsultation) -> dict[str, Any]:
    """The payload of a ``DECISION_MADE`` for an approval, via the shared shape."""
    from domyn.agents.decisions.events import metadata

    return metadata(
        use_case,
        mode=consultation.mode,
        answer=consultation.answer,
        decision=consultation.decision,
        acted=consultation.acted,
    )
