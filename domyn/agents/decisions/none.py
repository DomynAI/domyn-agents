"""The default: no decision model at all.

This is not a placeholder. It is the configuration almost every installation
will have, so it is the one whose behaviour has to be exactly right: every
caller must do precisely what it does today, with no extra call, no warning per
turn, and no code path that only runs when a model is present.

The rule it enforces, copied from the compaction strategies: **loud at load
time, silent at run time.** A configuration that names a provider which cannot
be built says so once, when it is built. After that, asking costs nothing and
says nothing — a per-turn warning about a feature nobody enabled is noise that
trains people to ignore warnings.
"""

from typing import ClassVar

from domyn.agents.decisions.base import DecisionProvider
from domyn.agents.decisions.models import Decision, Provenance, Question


class NoDecisions(DecisionProvider):
    """Answers nothing, cheaply, forever."""

    acts: ClassVar[bool] = False
    provenance: ClassVar[Provenance] = Provenance.NONE
    type_name: ClassVar[str] = "none"

    def __init__(self, reason: str = "no decision provider configured") -> None:
        super().__init__(model=None, timeout_s=0.0, max_failures=0)
        self.reason = reason

    async def _decide(self, state: str, questions: dict[str, Question]) -> Decision:
        # Unreachable: `decide()` short-circuits on `acts is False` before it
        # builds a state or looks at the clock.
        return Decision.nothing(self.reason, list(questions))
