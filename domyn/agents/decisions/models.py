"""The vocabulary of a decision: what is asked, and what comes back.

A decision model is not an LLM. It takes a *state* and a set of *typed
questions*, and returns a probability distribution over the allowed answers for
each one, in a single pass. The contract is ``decide(state, questions) ->
answers``, which is why this package exists beside ``llm/`` rather than inside
it: no streaming, no tool calls, no token budget to manage on the way out.

Three question types, which is all the wire protocol has:

* ``noul`` — is this statement true? Answer in [0, 1].
* ``choice`` — one of 2..255 named options.
* ``score`` — one of 2..10 ordered levels.

The one thing in here that is not a transcription of the protocol is
:class:`Provenance`, and it is the most important type in the package. See its
docstring: every threshold in the framework is only as meaningful as the number
it is compared against.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# Protocol limits, measured against a served Clef on 2026-10-24 rather than read
# off a datasheet. Exceeding them is a 422 from the server, so the client checks
# first and says which limit was hit.
MAX_QUESTIONS = 64
MAX_CHOICE_OPTIONS = 255
MIN_CHOICE_OPTIONS = 2
MAX_SCORE_LEVELS = 10
MIN_SCORE_LEVELS = 2


class Provenance(StrEnum):
    """Where a probability came from, and therefore what it is worth.

    This is not bookkeeping. The whole design rests on thresholds — "act only
    above 0.8" — and a threshold is only a safety mechanism if the number it
    reads is a real probability. Measured on 13 approval cases: a decision
    model's wrong answer carried 0.53 while its right ones carried 0.83-0.98,
    so a 0.8 threshold rejected exactly the wrong one. The same eval on an LLM
    asked to state its own confidence put its wrong answers at 0.78-0.85 —
    *above* the same threshold. Same accuracy, opposite usefulness.

    So the provenance travels with every answer, and a use case declares the
    minimum it will act on.
    """

    NATIVE = "native"
    """A real distribution over the allowed answers, in one pass. Jev, Clef."""

    LOGPROBS = "logprobs"
    """A real distribution over the *answer token*, from an LLM's top_logprobs.
    Honest, but per-model: the spread is not comparable across models.

    Only claim this when the answer tokens can actually be isolated. On a
    reasoning model the logprobs array covers the thinking too, and reading the
    first token that looks like a label picks one out of a sentence like "I
    should respond with one word: allow". Measured: that misreading scored 7/13
    with confidence 1.00 everywhere, which looks excellent and is noise. A
    backend that cannot isolate the answer must declare ``STATED``."""

    STATED = "stated"
    """A number the model wrote because it looked like a confidence. Not a
    probability. Usable to *add* caution, never to remove it."""

    NONE = "none"
    """No probability at all: nothing was asked, or nothing answered."""

    def at_least(self, minimum: Provenance) -> bool:
        """Whether this provenance is good enough for a use case needing ``minimum``."""
        return _PROVENANCE_RANK[self] >= _PROVENANCE_RANK[minimum]


_PROVENANCE_RANK: dict[Provenance, int] = {
    Provenance.NONE: 0,
    Provenance.STATED: 1,
    Provenance.LOGPROBS: 2,
    Provenance.NATIVE: 3,
}


class Question(BaseModel):
    """One typed question. Build it with :func:`noul`, :func:`choice` or :func:`score`."""

    type: Literal["noul", "choice", "score"]
    instructions: str = Field(min_length=1)
    criteria: Any = None
    """``choice``: a mapping option -> description. ``score``: an ordered list of
    level descriptions. ``noul``: unused."""

    @model_validator(mode="after")
    def _check_criteria(self) -> Question:
        if self.type == "noul":
            if self.criteria is not None:
                raise ValueError("a noul question takes no criteria")
            return self
        if self.type == "choice":
            if not isinstance(self.criteria, dict) or not all(
                isinstance(k, str) for k in self.criteria
            ):
                raise ValueError("a choice question needs a mapping of option -> description")
            if not MIN_CHOICE_OPTIONS <= len(self.criteria) <= MAX_CHOICE_OPTIONS:
                raise ValueError(
                    f"a choice question needs {MIN_CHOICE_OPTIONS}-{MAX_CHOICE_OPTIONS} "
                    f"options, got {len(self.criteria)}"
                )
            return self
        if not isinstance(self.criteria, list) or not all(
            isinstance(level, str) for level in self.criteria
        ):
            raise ValueError("a score question needs a list of level descriptions")
        if not MIN_SCORE_LEVELS <= len(self.criteria) <= MAX_SCORE_LEVELS:
            raise ValueError(
                f"a score question needs {MIN_SCORE_LEVELS}-{MAX_SCORE_LEVELS} levels, "
                f"got {len(self.criteria)}"
            )
        return self

    def to_wire(self) -> dict[str, Any]:
        """The System One JSON for this question."""
        body: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            body["criteria"] = self.criteria
        return body

    def labels(self) -> list[str]:
        """The answers this question allows, for the backends that need them spelled out.

        Score levels are numbered from **0**, which is how the server returns
        them: ``legend`` and ``probabilities`` come back keyed ``"0"``, ``"1"``,
        ``"2"`` for three levels, and ``score`` is the fractional mean over
        those indices (verified on a served Clef, not assumed).
        """
        if self.type == "choice":
            return list(self.criteria)
        if self.type == "score":
            return [str(i) for i in range(len(self.criteria))]
        return ["yes", "no"]


def noul(instructions: str) -> Question:
    """A true/false question."""
    return Question(type="noul", instructions=instructions)


def choice(instructions: str, criteria: dict[str, str]) -> Question:
    """A pick-one question. ``criteria`` maps each option to what it means."""
    return Question(type="choice", instructions=instructions, criteria=criteria)


def score(instructions: str, criteria: list[str]) -> Question:
    """An ordered-levels question, lowest level first."""
    return Question(type="score", instructions=instructions, criteria=criteria)


class Answer(BaseModel):
    """What came back for one question.

    ``known`` is the field to branch on. Everything that can go wrong — no
    provider, a timeout, a malformed response, a tripped breaker — arrives here
    as ``known=False`` rather than as an exception, because every caller in this
    framework has a correct behaviour for "no opinion" and none of them has one
    for a raised error in the middle of a turn.
    """

    known: bool = False
    provenance: Provenance = Provenance.NONE
    value: float | str | None = None
    """``noul``: the probability. ``choice``: the chosen option. ``score``: the level."""
    confidence: float = 0.0
    probabilities: dict[str, float] = Field(default_factory=dict)
    reason: str | None = None
    """Why it is not known, when it is not. Never the model's prose."""

    @classmethod
    def unknown(cls, reason: str) -> Answer:
        return cls(known=False, reason=reason)

    def acts_above(self, threshold: float, *, minimum: Provenance) -> bool:
        """Whether a caller may act on this answer.

        Both halves are required and they are independent: a high number from a
        source that does not produce real probabilities is exactly the case this
        guards against.
        """
        return self.known and self.confidence >= threshold and self.provenance.at_least(minimum)

    def is_true(self, threshold: float, *, minimum: Provenance = Provenance.STATED) -> bool:
        """For a ``noul``: confidently true. Anything else, including "unknown", is False."""
        if not self.known or not self.provenance.at_least(minimum):
            return False
        return isinstance(self.value, float) and self.value >= threshold

    def is_false(self, threshold: float, *, minimum: Provenance = Provenance.STATED) -> bool:
        """For a ``noul``: confidently **false**, which is not the negation of
        :meth:`is_true`.

        Between the two sits "it does not know", and the difference matters
        wherever the two directions carry different costs. Dropping a line the
        user wrote on a 0.45 — a coin flip — is the mistake this exists to make
        impossible to write by accident.
        """
        if not self.known or not self.provenance.at_least(minimum):
            return False
        return isinstance(self.value, float) and (1.0 - self.value) >= threshold


class Decision(BaseModel):
    """The answers to one request, plus what it cost."""

    answers: dict[str, Answer] = Field(default_factory=dict)
    latency_ms: int = 0
    model: str | None = None
    provider_type: str | None = None

    def get(self, key: str) -> Answer:
        """The answer by name, or a known-nothing. Callers never have to check for a key."""
        return self.answers.get(key) or Answer.unknown("no answer for this question")

    @property
    def known(self) -> bool:
        """Whether anything at all came back."""
        return any(answer.known for answer in self.answers.values())

    @classmethod
    def nothing(cls, reason: str, keys: list[str] | None = None) -> Decision:
        """A decision that knows nothing, with the reason attached to every question."""
        return cls(answers={key: Answer.unknown(reason) for key in keys or []})
