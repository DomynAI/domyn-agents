"""An ordinary LLM standing in for a decision model.

Between "I have nothing" and "I have a Clef server" sits the case
almost every installation is in: *I already have an LLM*. Without this backend
every use case in the framework would be dead code for most people, there would
be no baseline to judge a decision model against, and the shadow mode that has
to run before anything is switched on could only run where Clef is.

**It is not a drop-in replacement, and the difference is not quality.** On the
13-command approval eval this backend matched Clef's accuracy — and that is
exactly why the numbers matter more than the score:

====================  ==========  ===========  ================================
backend               accuracy    median       confidence when **wrong**
====================  ==========  ===========  ================================
Clef                  12/13       147 ms       0.53  (below a 0.8 threshold)
LLM, ``stated``       12/13       354-477 ms   0.78-0.85  (**above** it)
LLM, ``logprobs``      9/13       419 ms       0.58-1.00
====================  ==========  ===========  ================================

A threshold filters Clef's mistake and lets the LLM's through, and the LLM was
not reproducible run to run at temperature 0. Hence :class:`Provenance`: this
provider declares what its numbers are worth, and a use case that removes
friction declines to act on ``stated``.
"""

import asyncio
import json
import math
import re
from typing import Any, ClassVar

from domyn.agents.decisions.base import DecisionProvider
from domyn.agents.decisions.models import Answer, Decision, Provenance, Question
from domyn.agents.llm import BaseLLMProvider, LLMMessage
from domyn.agents.logger import get_logger

_JSON_OBJECT = re.compile(r"\{[^{}]*\"answer\"[^{}]*\}", re.S)


class LLMDecisionProvider(DecisionProvider):
    """Asks an LLM one question at a time and reads a distribution out of it.

    One question per request, deliberately. The batching that makes a decision
    model cheap does not transfer: a decision model answers every question in
    one pass, an LLM writes them one after another and the last answer is not
    the quality of the first. Questions are asked concurrently instead, which
    recovers the wall-clock without sharing a generation.
    """

    type_name: ClassVar[str] = "llm"

    def __init__(
        self,
        *,
        llm: BaseLLMProvider | None = None,
        provider_name: str | None = None,
        mode: str = "logprobs",
        timeout_s: float = 10.0,
        max_failures: int = 3,
        max_state_tokens: int | None = None,
        max_concurrency: int = 4,
    ) -> None:
        super().__init__(
            model=None,
            timeout_s=timeout_s,
            max_failures=max_failures,
            max_state_tokens=max_state_tokens,
        )
        self._llm = llm
        self.provider_name = provider_name
        self.mode = mode if mode in ("logprobs", "stated") else "logprobs"
        self._semaphore = asyncio.Semaphore(max_concurrency)
        # Starts as asked for and is *lowered* by what actually happens: a
        # provider that cannot return usable logprobs must not keep claiming
        # them. Never raised, which is the direction that would do harm.
        self._provenance = Provenance.LOGPROBS if self.mode == "logprobs" else Provenance.STATED

    @property
    def provenance(self) -> Provenance:  # type: ignore[override]
        return self._provenance

    def bind(self, llm: BaseLLMProvider) -> None:
        """Lend this provider an agent's LLM, as ``LLMSelector.bind_llm`` does."""
        if self._llm is None:
            self._llm = llm

    def _resolve_llm(self) -> BaseLLMProvider | None:
        if self._llm is not None:
            return self._llm
        if self.provider_name is None:
            return None
        from domyn.agents.registry import get_llm_registry

        found = get_llm_registry().get_llm_provider(self.provider_name)
        if found is None:
            get_logger().warning(
                f"decisions: LLM provider '{self.provider_name}' is not registered"
            )
        return found

    async def _decide(self, state: str, questions: dict[str, Question]) -> Decision:
        llm = self._resolve_llm()
        if llm is None:
            return Decision.nothing("no LLM bound to the decision provider", list(questions))

        names = list(questions)
        results = await asyncio.gather(
            *(self._ask_one(llm, state, questions[name]) for name in names),
            return_exceptions=True,
        )
        answers: dict[str, Answer] = {}
        for name, result in zip(names, results, strict=True):
            if isinstance(result, BaseException):
                get_logger().warning(
                    f"decisions: question '{name}' failed: {type(result).__name__}"
                )
                get_logger().debug(f"decision question detail: {result}")
                answers[name] = Answer.unknown("question failed")
            else:
                answers[name] = result
        return Decision(answers=answers, model=getattr(llm, "model_name", None))

    async def _ask_one(self, llm: BaseLLMProvider, state: str, question: Question) -> Answer:
        async with self._semaphore:
            if self.mode == "logprobs":
                return await self._via_logprobs(llm, state, question)
            return await self._via_stated(llm, state, question)

    # -- the two ways to get a number ---------------------------------------------

    async def _via_logprobs(self, llm: BaseLLMProvider, state: str, question: Question) -> Answer:
        labels = question.labels()
        messages = [
            LLMMessage(role="system", content=_one_word_prompt(question, labels)),
            LLMMessage(role="user", content=state),
        ]
        try:
            response = await llm.generate(
                messages, temperature=0, max_tokens=600, logprobs=True, top_logprobs=20
            )
        except TypeError:
            # A provider that does not take logprobs at all (Anthropic): say so
            # once and answer the honest way instead of failing the decision.
            get_logger().warning(
                "decisions: this LLM provider does not support logprobs; "
                "falling back to a stated confidence"
            )
            self._downgrade()
            return await self._via_stated(llm, state, question)

        distribution = _distribution_from_logprobs(response.logprobs, response.content, labels)
        if distribution is None:
            # Asking again rather than reading the bare word off this reply.
            # That shortcut returned a *known* answer carrying confidence 0.0,
            # which is safe — nothing acts on a zero — and dishonest: it reports
            # a verdict while pretending to have measured something. One more
            # call buys a number the caller can actually reason about, and it
            # happens once, because the provider has just downgraded itself and
            # every later question goes straight down the stated path.
            self._downgrade()
            return await self._via_stated(llm, state, question)

        best = max(distribution, key=lambda key: distribution[key])
        return _build_answer(best, distribution[best], distribution, question, Provenance.LOGPROBS)

    async def _via_stated(self, llm: BaseLLMProvider, state: str, question: Question) -> Answer:
        labels = question.labels()
        messages = [
            LLMMessage(role="system", content=_json_prompt(question, labels)),
            LLMMessage(role="user", content=state),
        ]
        response = await llm.generate(messages, temperature=0, max_tokens=2000)
        match = _JSON_OBJECT.search(response.content or "")
        if match is None:
            return Answer.unknown("no JSON answer in the reply")
        try:
            parsed = json.loads(match.group(0))
        except ValueError:
            return Answer.unknown("malformed JSON answer")

        value = parsed.get("answer")
        if not isinstance(value, str) or value not in labels:
            return Answer.unknown("answer is not one of the allowed options")
        try:
            confidence = float(parsed.get("confidence"))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(max(confidence, 0.0), 1.0)
        return _build_answer(value, confidence, {value: confidence}, question, Provenance.STATED)

    def _downgrade(self) -> None:
        if self._provenance is not Provenance.STATED:
            self._provenance = Provenance.STATED


# -- prompts ----------------------------------------------------------------------


def _options_block(question: Question, labels: list[str]) -> str:
    if question.type == "noul":
        return "yes = the statement is true, no = it is false"
    if question.type == "choice":
        return "\n".join(f"{name} = {meaning}" for name, meaning in question.criteria.items())
    return "\n".join(f"{index} = {text}" for index, text in enumerate(question.criteria))


def _one_word_prompt(question: Question, labels: list[str]) -> str:
    return (
        f"{question.instructions}\n\n{_options_block(question, labels)}\n\n"
        f"Answer with exactly one word, one of: {', '.join(labels)}. Nothing else."
    )


def _json_prompt(question: Question, labels: list[str]) -> str:
    return (
        f"{question.instructions}\n\n{_options_block(question, labels)}\n\n"
        'Reply with JSON only: {"answer": "<one of '
        f'{", ".join(labels)}>", "confidence": <0..1>}}'
    )


# -- reading the numbers ----------------------------------------------------------


def _distribution_from_logprobs(
    logprobs: Any, content: str | None, labels: list[str]
) -> dict[str, float] | None:
    """A distribution over ``labels``, or ``None`` if it cannot be had honestly.

    **The trap this exists for.** On a reasoning model the logprobs array covers
    the thinking as well as the answer. Scanning forward for the first token
    that looks like a label finds it inside a sentence like *"I should respond
    with one word: allow"* — measured: 7/13 correct with confidence 1.00 on
    every single case, a result that looks excellent and is pure noise.

    So the scan runs **backwards**, from the end, where the answer is. And when
    the alternatives are not observable in ``top_logprobs``, this returns
    ``None`` rather than a fabricated 1.0: a distribution over one outcome is
    not a measurement, and the caller downgrades its provenance instead.

    **Measured against a real reasoning model, this tier mostly collapses**, and
    the reason is structural rather than a defect to fix: by the time the answer
    token is emitted the model has already reasoned its way to a conclusion, so
    the distribution over that token is near-degenerate — the uncertainty lived
    in the thinking and has been spent. Logprobs there measure the decoding, not
    the deliberation. On a DeepSeek-V4.1-Flash served behind a gateway one
    verdict came back at ``{allow: 1.0, deny: 0.0}`` and the rest had no second
    label at all. That is an argument for a real decision model, which answers
    in one pass and so has a distribution to report, and against treating
    ``logprobs`` as a free substitute for one.
    """
    entries = (logprobs or {}).get("content") if isinstance(logprobs, dict) else None
    if not isinstance(entries, list) or not entries:
        return None

    lowered = [label.lower() for label in labels]
    for entry in reversed(entries):
        if not isinstance(entry, dict):
            continue
        token = str(entry.get("token", "")).strip().lower()
        if not token or not any(label.startswith(token) for label in lowered):
            continue

        mass: dict[str, float] = {}
        for candidate in entry.get("top_logprobs") or []:
            text = str(candidate.get("token", "")).strip().lower()
            logprob = candidate.get("logprob")
            if not text or not isinstance(logprob, int | float):
                continue
            for label, original in zip(lowered, labels, strict=True):
                # A label may be split across tokens ("al" + "low"), so a token
                # is credited to every label it could begin.
                if label.startswith(text):
                    mass[original] = max(mass.get(original, 0.0), math.exp(float(logprob)))
        if len(mass) < 2:
            return None
        total = sum(mass.values()) or 1e-9
        return {label: value / total for label, value in mass.items()}
    return None


def _build_answer(
    label: str,
    confidence: float,
    probabilities: dict[str, float],
    question: Question,
    provenance: Provenance,
) -> Answer:
    """Turn a label into the shape the question's type promises.

    A caller asking a ``noul`` gets a probability in ``value`` whichever backend
    answered, so swapping the backend never changes how an answer is read.
    """
    if question.type == "noul":
        truth = probabilities.get("yes", confidence if label == "yes" else 1.0 - confidence)
        return Answer(
            known=True,
            provenance=provenance,
            value=truth,
            confidence=max(truth, 1.0 - truth),
            probabilities={"true": truth, "false": 1.0 - truth},
        )
    if question.type == "score":
        try:
            level = float(label)
        except ValueError:
            return Answer.unknown("score answer is not a level number")
        return Answer(
            known=True,
            provenance=provenance,
            value=level,
            confidence=confidence,
            probabilities=probabilities,
        )
    return Answer(
        known=True,
        provenance=provenance,
        value=label,
        confidence=confidence,
        probabilities=probabilities,
    )
