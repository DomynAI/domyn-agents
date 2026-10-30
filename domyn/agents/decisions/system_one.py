"""The System One API: Jev (TypeSafe) and Clef (Cloudflare).

Cloudflare made Clef speak Jev's API on purpose, so one client covers both
destinations and a third one served locally. Raw ``httpx``, no vendor SDK, like
every provider in ``llm/``.

Everything in here that looks like a quirk was measured against a served Clef on
2026-10-24, and each one is commented where it bites:

* ``model`` must be the name the **server** exposes (``Cloudflare/clef``, not
  ``clef``), or the request is refused outright. First thing anyone trips over.
* ``noul`` answers carry **no confidence field** — just the probability.
* ``score`` levels are numbered from zero and the score is fractional.
* A state past the context window is **truncated silently**, with a ``200 OK``.
  Handled in the base class, which refuses rather than sends.
"""

import time
from typing import Any, ClassVar

import httpx

from domyn.agents.decisions.base import DecisionProvider
from domyn.agents.decisions.models import Answer, Decision, Provenance, Question
from domyn.agents.logger import get_logger

DEFAULT_PATH = "/v1/systemone"


class SystemOneProvider(DecisionProvider):
    """A Jev-compatible decision endpoint."""

    provenance: ClassVar[Provenance] = Provenance.NATIVE
    type_name: ClassVar[str] = "system_one"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 2.0,
        max_failures: int = 3,
        max_state_tokens: int | None = None,
        path: str = DEFAULT_PATH,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            model=model,
            timeout_s=timeout_s,
            max_failures=max_failures,
            max_state_tokens=max_state_tokens,
        )
        self.url = base_url.rstrip("/") + path
        self.api_key = api_key
        self._client = client
        self._owns_client = client is None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            # Slightly over the decision timeout: `asyncio.wait_for` in the base
            # class is the real deadline, and a transport that gives up first
            # would report a transport error instead of a timeout.
            self._client = httpx.AsyncClient(timeout=self.timeout_s + 1.0)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None

    async def _decide(self, state: str, questions: dict[str, Question]) -> Decision:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = {
            "model": self.model,
            "state": state,
            "questions": {name: q.to_wire() for name, q in questions.items()},
        }
        get_logger().debug(f"decisions request: {body}")

        started = time.monotonic()
        response = await self._http().post(self.url, json=body, headers=headers)
        latency = round((time.monotonic() - started) * 1000)

        if response.status_code >= 400:
            # The server says which model it actually serves, and that message is
            # the whole answer to the commonest failure, so it is kept verbatim —
            # it names no customer data.
            raise RuntimeError(f"{response.status_code}: {response.text[:300]}")

        payload = response.json()
        get_logger().debug(f"decisions response: {payload}")
        answers = self._unwrap(payload)
        parsed = {
            name: _parse_answer(answers.get(name), questions[name])
            for name in questions
            if name in answers
        }
        decision = Decision(answers=parsed, latency_ms=latency, model=self.model)
        get_logger().info(
            f"decisions: {self.type_name} answered {len(parsed)}/{len(questions)} in {latency} ms"
        )
        return decision

    @staticmethod
    def _unwrap(payload: Any) -> dict[str, Any]:
        """The answers, wherever this destination puts them.

        A locally served Clef and the Jev API both return ``answers`` at the top
        level. Workers AI wraps every response in ``result``; accepting both here
        costs one line and saves a second subclass.
        """
        if not isinstance(payload, dict):
            raise ValueError(f"expected a JSON object, got {type(payload).__name__}")
        if "answers" not in payload and isinstance(payload.get("result"), dict):
            payload = payload["result"]
        answers = payload.get("answers")
        if not isinstance(answers, dict):
            raise ValueError("response carries no 'answers' object")
        return answers


class WorkersAIProvider(SystemOneProvider):
    """Clef on Cloudflare Workers AI.

    The same protocol behind a different URL shape
    (``/accounts/<id>/ai/run/@cf/cloudflare/clef``) and an envelope the parent
    already unwraps. It exists as its own type so the account id and the model
    slug can be configured as what they are, rather than pasted into a URL.
    """

    type_name: ClassVar[str] = "workers_ai"

    def __init__(
        self,
        *,
        account_id: str,
        model: str = "@cf/cloudflare/clef",
        api_key: str,
        base_url: str = "https://api.cloudflare.com/client/v4",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            path=f"/accounts/{account_id}/ai/run/{model}",
            **kwargs,
        )


def _parse_answer(raw: Any, question: Question) -> Answer:
    """One answer off the wire, or an unknown saying why it could not be read."""
    if not isinstance(raw, dict):
        return Answer.unknown("malformed answer")

    if question.type == "noul":
        value = raw.get("noul")
        if not isinstance(value, int | float):
            return Answer.unknown("malformed noul answer")
        probability = float(value)
        # There is no confidence field on a noul, so it is derived rather than
        # invented: 0.98 and 0.02 are both confident answers (true, false), 0.5
        # is the model saying it does not know. Collapsing that into a single
        # number is what lets one threshold work for all three question types.
        return Answer(
            known=True,
            provenance=Provenance.NATIVE,
            value=probability,
            confidence=max(probability, 1.0 - probability),
            probabilities={"true": probability, "false": 1.0 - probability},
        )

    if question.type == "choice":
        chosen = raw.get("choice")
        if not isinstance(chosen, str):
            return Answer.unknown("malformed choice answer")
        probabilities = _floats(raw.get("probabilities"))
        return Answer(
            known=True,
            provenance=Provenance.NATIVE,
            value=chosen,
            confidence=_confidence(raw, probabilities, chosen),
            probabilities=probabilities,
        )

    level = raw.get("score")
    if not isinstance(level, int | float):
        return Answer.unknown("malformed score answer")
    probabilities = _floats(raw.get("probabilities"))
    return Answer(
        known=True,
        provenance=Provenance.NATIVE,
        value=float(level),
        confidence=_confidence(raw, probabilities, None),
        probabilities=probabilities,
    )


def _floats(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): float(value) for key, value in raw.items() if isinstance(value, int | float)}


def _confidence(raw: dict[str, Any], probabilities: dict[str, float], chosen: str | None) -> float:
    """The server's confidence, or the best reconstruction of it.

    Falling back to the probability mass rather than to a constant: a missing
    field should not silently produce a confident-looking answer, and it should
    not produce a zero that a threshold reads as "unknown" either.
    """
    stated = raw.get("confidence")
    if isinstance(stated, int | float):
        return float(stated)
    if chosen is not None and chosen in probabilities:
        return probabilities[chosen]
    return max(probabilities.values(), default=0.0)
