"""Saying what a decision model decided, in one shape for every use case.

``DECISION_MADE`` is the only output a use case running in ``shadow`` has — it
changes nothing about the turn by definition, so without the event it changes
nothing observable either, and the mode that exists to be measured cannot be.
That is why this is shared rather than written per use case: six slightly
different payloads would be six things to parse before anyone could compare
them.

**It never carries the state.** The command being judged, the task, the file
contents are customer data, and this event is emitted at info level — see the
logging rules in ``AGENTS.md`` (Logging Conventions). What it carries is the
verdict, the confidence, where that confidence came from, whether anything was
done about it, and the latency, which is the number the whole premise rests on
and which nothing else in the framework measures.
"""

from typing import Any

from domyn.agents.core import BaseEvent, ExecutionEventType
from domyn.agents.decisions.models import Answer, Decision
from domyn.agents.decisions.runtime import DecisionMode


def metadata(
    use_case: str,
    *,
    mode: DecisionMode,
    answer: Answer | None = None,
    decision: Decision | None = None,
    acted: bool = False,
    summary: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """The payload of one ``DECISION_MADE``.

    ``summary`` is for the use cases whose answer is not a single verdict — a
    set of hidden tools, a count of dropped sections — where the useful thing
    to show is what was done, not one probability.
    """
    payload: dict[str, Any] = {
        "use_case": use_case,
        "mode": mode.value,
        "acted": acted,
    }
    if summary is not None:
        payload["summary"] = summary
    if answer is not None:
        payload.update(
            {
                "known": answer.known,
                "value": answer.value,
                "confidence": round(answer.confidence, 4),
                "provenance": answer.provenance.value,
                "probabilities": {k: round(v, 4) for k, v in answer.probabilities.items()},
                "reason": answer.reason,
            }
        )
    if decision is not None:
        payload.update(
            {
                "latency_ms": decision.latency_ms,
                "model": decision.model,
                "provider_type": decision.provider_type,
            }
        )
    payload.update(extra)
    return payload


def event(author: str, payload: dict[str, Any]) -> BaseEvent:
    """Wrap a payload as the event itself."""
    return BaseEvent(event_type=ExecutionEventType.DECISION_MADE, author=author, metadata=payload)
