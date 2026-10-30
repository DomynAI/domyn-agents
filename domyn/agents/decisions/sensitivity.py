"""Which models a turn's data is allowed to reach.

The usual framing of this is routing — send the cheap turns to the cheap model.
For a platform that runs on customer data the valuable half is the other one:
*"secrets do not leave the cluster"* stops being discipline and becomes
configuration. A ``score`` rates what the turn is about, and a provider declares
the highest rating it may see.

.. code-block:: yaml

    llm_providers:
      - name: cloud_model
        type: anthropic
        model_name: claude-opus-5-5
        max_sensitivity: internal     # never sees confidential or regulated

    agents:
      - name: Assistant
        decisions:
          provider: local_clef
          sensitivity: {mode: active, min_confidence: 0.7}

**This use case breaks the rule the others follow, and it has to.** Everywhere
else a model that cannot answer degrades to today's behaviour, because today's
behaviour is safe. Here today's behaviour is "send it anyway", which is the
thing being prevented — a guardrail that fails open is not a guardrail. So when
the rating cannot be obtained, ``on_unknown`` decides, and it defaults to
``block``.

That default is deliberately the inconvenient one: a decision endpoint that is
down stops turns on an agent configured this way. Anyone who prefers the
opposite can say so, in configuration, where it is visible — and a reviewer can
see whose data they chose it for.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Any, NamedTuple

from domyn.agents.decisions.models import score
from domyn.agents.logger import get_logger

if TYPE_CHECKING:
    from domyn.agents.decisions.runtime import DecisionRuntime

USE_CASE = "sensitivity"
STATE_KEY = "sensitivity"


class Sensitivity(IntEnum):
    """What a turn is about, in the order that matters.

    An ``IntEnum`` because the only question ever asked of it is "is this more
    than that", and four names with a documented order beat a bare integer at
    every call site.
    """

    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    REGULATED = 3

    @classmethod
    def parse(cls, value: Any) -> Sensitivity | None:
        """Read a configured name or number. ``None`` when it is neither."""
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return None
        if isinstance(value, int) and 0 <= value <= 3:
            return cls(value)
        if isinstance(value, str):
            try:
                return cls[value.strip().upper()]
            except KeyError:
                return None
        return None


#: The wording the model is scored against. Written as what a reader would
#: recognise, not as a policy abstraction: "contains credentials" is answerable,
#: "is confidential" is not.
LEVELS: list[str] = [
    "Public: nothing here would matter if it were posted on the internet.",
    "Internal: ordinary company work — code, tickets, plans — not for outsiders.",
    "Confidential: customer data, personal data, financials, or unreleased work.",
    "Regulated: credentials, keys, secrets, health or payment data, or anything "
    "under a legal obligation such as GDPR or PCI.",
]

QUESTION = score(
    "An agent is about to work on the task below, which may name files it will "
    "read. How sensitive is the data this turn will handle?",
    LEVELS,
)


class Rating(NamedTuple):
    """What the turn was rated, and whether anyone actually rated it."""

    level: Sensitivity
    known: bool
    confidence: float = 0.0

    @property
    def name_(self) -> str:
        return self.level.name.lower()


async def rate(
    runtime: DecisionRuntime | None, *, task: str, state: dict[str, Any] | None = None
) -> Rating | None:
    """Rate this turn. ``None`` when the use case is not switched on at all.

    A ``Rating`` with ``known=False`` means it is on and could not tell, which
    is a different situation and a caller must handle it — see
    :func:`allows`.
    """
    if runtime is None or not runtime.enabled(USE_CASE):
        return None
    if not task:
        return Rating(level=Sensitivity.REGULATED, known=False)

    decision = await runtime.ask(USE_CASE, {"task": task}, {"level": QUESTION})
    answer = decision.get("level")
    if not runtime.may_act(USE_CASE, answer) or not isinstance(answer.value, float):
        return Rating(level=Sensitivity.REGULATED, known=False)

    # The score comes back fractional — a weighted mean over the levels — and
    # rounding *up* is the only safe direction: a turn sitting between internal
    # and confidential is treated as confidential.
    level = Sensitivity(min(int(answer.value + 0.999), Sensitivity.REGULATED))
    rating = Rating(level=level, known=True, confidence=answer.confidence)
    if isinstance(state, dict):
        state[STATE_KEY] = rating.name_
    get_logger().info(
        f"decisions[sensitivity]: this turn rated {rating.name_} ({answer.confidence:.2f})"
    )
    return rating


def provider_ceiling(provider: Any) -> Sensitivity | None:
    """The highest rating this provider may see, if it declares one."""
    return Sensitivity.parse(getattr(provider, "max_sensitivity", None))


def allows(
    provider: Any, rating: Rating | None, *, on_unknown: str = "block"
) -> tuple[bool, str | None]:
    """Whether this provider may serve this turn, and why not.

    Returns ``(True, None)`` for every case that is not a refusal, including
    every case where the question does not arise: no rating taken, or a
    provider that declares no ceiling. Declaring one is what opts a deployment
    into this.
    """
    ceiling = provider_ceiling(provider)
    if ceiling is None or rating is None:
        return True, None

    if not rating.known:
        if on_unknown == "allow":
            get_logger().warning(
                "decisions[sensitivity]: could not rate this turn; allowing it "
                "because on_unknown is 'allow'"
            )
            return True, None
        return False, (
            "This turn cannot run: the data-sensitivity check could not be "
            "completed, and this provider is configured to refuse rather than "
            "guess."
        )

    if rating.level <= ceiling:
        return True, None
    return False, (
        f"This turn handles {rating.name_} data and the configured model may "
        f"only see up to {ceiling.name.lower()}."
    )
