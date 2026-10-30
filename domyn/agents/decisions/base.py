"""The provider interface, and the safety net every provider inherits.

Subclasses implement :meth:`DecisionProvider._decide`, which may raise, block or
misbehave. :meth:`DecisionProvider.decide` is what callers use, and it never
raises: a timeout, a bad response or an unreachable endpoint all arrive as a
:class:`~domyn.agents.decisions.models.Decision` that knows nothing. That split
is deliberate — the error handling is written once, here, instead of being
re-derived (and eventually forgotten) in each backend.
"""

import asyncio
import json
import math
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from domyn.agents.decisions.models import (
    MAX_QUESTIONS,
    Answer,
    Decision,
    Provenance,
    Question,
)
from domyn.agents.logger import get_logger

# Same ratio as HeuristicEstimator (context_management/budget.py), duplicated
# rather than imported to keep this package free of the planner machinery. It
# only has to avoid *under*-estimating; see _state_tokens for why that matters
# more here than it does there.
CHARS_PER_TOKEN = 3.0


def estimate_tokens(text: str) -> int:
    """A deliberately pessimistic token count for a state."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


class DecisionProvider(ABC):
    """One decision backend.

    ``acts`` is the honest answer to "will asking this thing ever change
    anything". ``NoDecisions`` says False, and callers can use it to skip
    building a state they will not send — the same trick ``CompactionStrategy``
    uses so a no-op strategy does not announce a threshold that will never fire.
    """

    acts: ClassVar[bool] = True
    provenance: ClassVar[Provenance] = Provenance.NONE
    type_name: ClassVar[str] = "base"

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout_s: float = 2.0,
        max_failures: int = 3,
        max_state_tokens: int | None = None,
    ) -> None:
        self.model = model
        self.timeout_s = timeout_s
        self.max_failures = max_failures
        self.max_state_tokens = max_state_tokens
        self._consecutive_failures = 0

    # -- what subclasses implement ------------------------------------------------

    @abstractmethod
    async def _decide(self, state: str, questions: dict[str, Question]) -> Decision:
        """Ask the backend. May raise; the caller handles it."""

    async def aclose(self) -> None:  # noqa: B027  (optional: most backends hold nothing)
        """Release whatever the backend holds. Safe to call twice."""

    # -- what callers use ---------------------------------------------------------

    @property
    def breaker_open(self) -> bool:
        """Whether this provider has given up after repeated failures."""
        return self.max_failures > 0 and self._consecutive_failures >= self.max_failures

    async def decide(self, state: Any, questions: dict[str, Question]) -> Decision:
        """Ask, and never raise.

        Every refusal is a ``Decision`` whose answers carry the reason, so a
        caller that branches on ``known`` behaves correctly without knowing
        which of the many ways to fail happened.
        """
        keys = list(questions)
        if not questions:
            return Decision()
        if not self.acts:
            return Decision.nothing("no decision provider configured", keys)
        if len(questions) > MAX_QUESTIONS:
            get_logger().warning(
                f"decisions: {len(questions)} questions asked, the protocol allows "
                f"{MAX_QUESTIONS}; not asking"
            )
            return Decision.nothing(f"more than {MAX_QUESTIONS} questions", keys)
        if self.breaker_open:
            return Decision.nothing("decision provider disabled after repeated failures", keys)

        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        too_long = self._state_too_long(text)
        if too_long is not None:
            return Decision.nothing(too_long, keys)

        started = time.monotonic()
        try:
            decision = await asyncio.wait_for(self._decide(text, questions), self.timeout_s)
        except TimeoutError:
            self._record_failure()
            get_logger().warning(f"decisions: {self.type_name} timed out after {self.timeout_s}s")
            return Decision.nothing("decision provider timed out", keys)
        except Exception as exc:
            self._record_failure()
            get_logger().warning(f"decisions: {self.type_name} failed: {type(exc).__name__}")
            get_logger().debug(f"decision provider detail: {exc}")
            return Decision.nothing("decision provider failed", keys)

        self._consecutive_failures = 0
        decision.latency_ms = decision.latency_ms or round((time.monotonic() - started) * 1000)
        decision.model = decision.model or self.model
        decision.provider_type = self.type_name
        # A backend that answered some questions and not others is normal; a
        # backend that answered none of them is a malformed response, not a
        # verdict, so the caller is told the difference.
        for key in keys:
            decision.answers.setdefault(key, Answer.unknown("not answered"))
        return decision

    # -- internals ----------------------------------------------------------------

    def _state_too_long(self, text: str) -> str | None:
        """The reason this state cannot be sent, if it cannot.

        **The client counts, because the server does not tell.** Measured on a
        served Clef: a state of 200 000 and one of 400 000 characters both came
        back ``200 OK`` with ``input_tokens: 73728``, the window size, and no
        warning of any kind. A silently truncated state produces a confident
        answer about a document the model only saw the start of, which is worse
        than no answer at all — so anything over the limit is refused here.
        """
        if self.max_state_tokens is None:
            return None
        tokens = estimate_tokens(text)
        if tokens <= self.max_state_tokens:
            return None
        get_logger().warning(
            f"decisions: state is about {tokens} tokens, over the "
            f"{self.max_state_tokens} this model accepts; not asking"
        )
        return "state longer than the model's context"

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self.breaker_open:
            get_logger().warning(
                f"decisions: {self.type_name} disabled after "
                f"{self._consecutive_failures} consecutive failures"
            )
