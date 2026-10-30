"""What an agent carries, and the rules about when it is allowed to matter.

The provider knows how to ask. The runtime knows *whether to ask at all*, and
whether an answer may change anything — which is the part with the safety
properties, so it lives in one place rather than at each call site.

Two rules are enforced here and nowhere else:

**A use case declares the quality of probability it needs.** Asking for
``active`` with a backend whose numbers are not good enough does not raise and
does not silently do it anyway: it logs once and stays in ``shadow``. The
minimum is in the code, not in the config, because it is a safety property and
not a preference — see :data:`USE_CASES`.

**Shadow is the default road to active.** Every use case starts ``off``; a
configured provider switches nothing on by itself. ``shadow`` asks and records
and changes nothing, which is what makes it safe to measure agreement on real
traffic before handing anything over.
"""

from enum import StrEnum
from typing import Any, NamedTuple

from domyn.agents.decisions.base import DecisionProvider
from domyn.agents.decisions.models import Answer, Decision, Provenance, Question
from domyn.agents.decisions.none import NoDecisions
from domyn.agents.logger import get_logger


def _rank(provenance: Provenance) -> int:
    """Order for comparing two minimums. Mirrors `Provenance.at_least`."""
    return sum(1 for other in Provenance if provenance.at_least(other))


class DecisionMode(StrEnum):
    """How much a use case lets the model matter."""

    OFF = "off"
    """Not asked. The default for everything."""

    SHADOW = "shadow"
    """Asked, recorded, ignored. Nothing about the turn changes."""

    ACTIVE = "active"
    """Asked, and acted on when the answer clears the bar."""


class UseCase(NamedTuple):
    """The fixed properties of one place a decision can be used."""

    name: str
    requires: Provenance
    """The weakest provenance that may act in this use case's *permissive*
    direction — the one that removes friction, and so needs a real number."""
    default_threshold: float
    why: str
    """Why the minimum is what it is. Read by whoever next wants to lower it."""
    cautious_requires: Provenance | None = None
    """The bar for the direction that only *adds* friction, when the use case
    has one. Approvals do: allowing skips a human confirmation, denying costs
    an unnecessary refusal the user can override, and those two should not need
    the same evidence. ``None`` means the use case has no cautious direction and
    ``requires`` governs everything."""

    @property
    def speaks_at(self) -> Provenance:
        """The weakest provenance at which this use case is worth running active
        at all — it can still do its cautious half."""
        if self.cautious_requires is None:
            return self.requires
        return min(self.requires, self.cautious_requires, key=_rank)


# The registry of use cases. Adding one here is a deliberate act: it is the
# list of places in the framework where a model is allowed to have an opinion.
USE_CASES: dict[str, UseCase] = {
    "approvals": UseCase(
        name="approvals",
        requires=Provenance.LOGPROBS,
        default_threshold=0.8,
        cautious_requires=Provenance.STATED,
        why=(
            "The gate's default is ASK, so allowing means *removing* a human "
            "confirmation. Measured: a stated confidence put its wrong answers "
            "at 0.78-0.85, above any threshold worth having. Denying only adds "
            "friction, so it runs on the weaker bar."
        ),
    ),
    "skills": UseCase(
        name="skills",
        requires=Provenance.STATED,
        default_threshold=0.5,
        why="Being wrong costs one expansion, and the index stays complete.",
    ),
    "instructions": UseCase(
        name="instructions",
        requires=Provenance.STATED,
        default_threshold=0.8,
        why=(
            "Not the same shape as skills, despite looking like it. Skills are "
            "*added*, so being wrong costs an unopened file; a conditional "
            "instruction section is *removed*, and that deletes a line the user "
            "wrote. The threshold governs the removal, so it is high and the "
            "decision reads `is_false`, never `not is_true` — between the two "
            "sits 'it does not know', and that case has to keep the text."
        ),
    ),
    "context_ladder": UseCase(
        name="context_ladder",
        requires=Provenance.LOGPROBS,
        default_threshold=0.9,
        why="Hiding is irreversible within the turn, so it needs a real number.",
    ),
    "sensitivity": UseCase(
        name="sensitivity",
        requires=Provenance.STATED,
        default_threshold=0.7,
        why=(
            "The odd one out, and knowingly. Every other use case acts only to "
            "*change* what would otherwise happen, so a weak number can do no "
            "harm by being ignored. This one is a data boundary: the thing it "
            "prevents is the default. It therefore runs on a stated confidence "
            "— a weak rating that triggers a refusal is a false alarm, which is "
            "the safe failure — and what happens when it cannot rate at all is "
            "`on_unknown`, which defaults to blocking."
        ),
    ),
    "tools": UseCase(
        name="tools",
        requires=Provenance.LOGPROBS,
        default_threshold=0.9,
        why="Same as the ladder: it may only ever hide, and only when sure.",
    ),
}


class UseCasePolicy(NamedTuple):
    """The configured settings for one use case."""

    mode: DecisionMode
    threshold: float
    on_unknown: str = "block"
    """What to do when the model cannot answer, for the one use case where that
    is a real choice. Everywhere else an unknown answer simply means "behave as
    today" and this is ignored; for ``sensitivity`` today's behaviour is the
    thing being prevented, so the operator has to pick. See
    ``decisions/sensitivity.py``."""


class DecisionRuntime:
    """A decision provider plus the policy for each place it may be used.

    Attached to an agent exactly as ``SkillsRuntime`` and ``InstructionRuntime``
    are: optional, ``None`` meaning "today's behaviour", shared between the
    agents that declare it. Stateless per turn — nothing derived is cached on
    it, because one agent serves concurrent conversations.
    """

    def __init__(
        self,
        provider: DecisionProvider | None = None,
        policies: dict[str, UseCasePolicy] | None = None,
    ) -> None:
        self.provider = provider or NoDecisions()
        self._policies = policies or {}
        self._warned: set[str] = set()

    # -- policy -------------------------------------------------------------------

    def mode(self, use_case: str) -> DecisionMode:
        """The mode this use case runs in, after the provenance rule is applied.

        An ``active`` use case whose provider cannot produce good enough
        probabilities is reported as ``shadow``: it still measures, it just does
        not act. Logged once per use case, not once per turn.
        """
        policy = self._policies.get(use_case)
        if policy is None or policy.mode is DecisionMode.OFF:
            return DecisionMode.OFF
        if not self.provider.acts:
            return DecisionMode.OFF

        spec = USE_CASES.get(use_case)
        if spec is None:
            # A name that is not in the registry is a typo in a config file, and
            # it must not inherit the threshold written next to it: that would
            # be an unreviewed use case running active with a number nobody
            # chose for it. The config loader rejects these outright; this is
            # the backstop for a runtime built in code.
            if use_case not in self._warned:
                self._warned.add(use_case)
                get_logger().warning(
                    f"decisions: '{use_case}' is not a known use case "
                    f"({', '.join(sorted(USE_CASES))}); it will never be asked"
                )
            return DecisionMode.OFF

        if policy.mode is DecisionMode.SHADOW:
            return DecisionMode.SHADOW

        if not self.provider.provenance.at_least(spec.speaks_at):
            if use_case not in self._warned:
                self._warned.add(use_case)
                get_logger().warning(
                    f"decisions: '{use_case}' is configured active but the "
                    f"{self.provider.type_name} backend only provides "
                    f"'{self.provider.provenance.value}' probabilities and this use case "
                    f"needs at least '{spec.speaks_at.value}'; staying in shadow. {spec.why}"
                )
            return DecisionMode.SHADOW
        return DecisionMode.ACTIVE

    def enabled(self, use_case: str) -> bool:
        """Whether it is worth building a state for this use case at all."""
        return self.mode(use_case) is not DecisionMode.OFF

    def on_unknown(self, use_case: str) -> str:
        """``block`` or ``allow`` for an unanswerable question. Blocks by default."""
        policy = self._policies.get(use_case)
        return policy.on_unknown if policy is not None else "block"

    def threshold(self, use_case: str) -> float:
        """The bar for this use case. Unreachable (1.0) for a name nobody declared."""
        spec = USE_CASES.get(use_case)
        if spec is None:
            return 1.0
        policy = self._policies.get(use_case)
        return policy.threshold if policy is not None else spec.default_threshold

    # -- asking -------------------------------------------------------------------

    async def ask(self, use_case: str, state: Any, questions: dict[str, Question]) -> Decision:
        """Ask on behalf of a use case. Returns a knows-nothing when it is off."""
        if not self.enabled(use_case):
            return Decision.nothing(f"'{use_case}' is off", list(questions))
        decision = await self.provider.decide(state, questions)
        get_logger().info(
            f"decisions[{use_case}]: mode={self.mode(use_case).value} "
            f"latency={decision.latency_ms}ms answered={sum(1 for a in decision.answers.values() if a.known)}"
        )
        return decision

    def may_act(self, use_case: str, answer: Answer) -> bool:
        """Whether this answer may act in the direction that *removes* friction.

        The three conditions are independent and all three are required: the
        use case is active, the number clears its threshold, and the number is
        the kind of number this use case accepts. This is the strict one — the
        one that skips a human confirmation or hides something from the model.
        """
        return self._may(use_case, answer, cautious=False)

    def may_act_cautiously(self, use_case: str, answer: Answer) -> bool:
        """Whether this answer may act in the direction that only *adds* friction.

        Same threshold, weaker provenance, where the use case says it has such a
        direction (``UseCase.cautious_requires``). Refusing a command that was
        fine costs an override; running one that was not costs rather more, and
        the two should not need the same evidence.
        """
        return self._may(use_case, answer, cautious=True)

    def _may(self, use_case: str, answer: Answer, *, cautious: bool) -> bool:
        if self.mode(use_case) is not DecisionMode.ACTIVE:
            return False
        spec = USE_CASES.get(use_case)
        if spec is None:
            return False
        minimum = spec.requires
        if cautious and spec.cautious_requires is not None:
            minimum = spec.cautious_requires
        return answer.acts_above(self.threshold(use_case), minimum=minimum)

    async def aclose(self) -> None:
        await self.provider.aclose()
