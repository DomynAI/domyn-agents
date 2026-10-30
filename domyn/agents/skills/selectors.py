"""Selectors: which skills are active for this turn.

The selector is the axis that turns one skills implementation into many usage
styles. Swapping it changes behaviour from "everything always injected" to
"model picks from an index" to "a router model decides up front" to "only what
the user asked for", with no change to sources or delivery.

Selectors never *remove* a skill from the index because it looks irrelevant. They
decide what to expand. Keeping the index complete means the model can always ask
for something the selector did not anticipate — a selector that hides is worse
than no selector when it guesses wrong.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, PrivateAttr

from domyn.agents.logger import get_logger
from domyn.agents.skills.models import Selection, SkillIndex, SkillQuery, SkillRef

if TYPE_CHECKING:
    from domyn.agents.llm import BaseLLMProvider


class SkillSelector(BaseModel, ABC):
    """Base class for selection strategies, registered by ``type`` tag."""

    _registry: ClassVar[dict[str, type[SkillSelector]]] = {}

    type: str

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        # See SkillSource for why this reads model_fields rather than annotations.
        super().__pydantic_init_subclass__(**kwargs)
        field = cls.model_fields.get("type")
        tag = getattr(field, "default", None)
        if isinstance(tag, str) and tag:
            SkillSelector._registry[tag] = cls

    @abstractmethod
    async def select(self, index: SkillIndex, query: SkillQuery) -> Selection:
        """Decide what is visible and what gets expanded now."""

    def bind_llm(self, provider: BaseLLMProvider | None) -> None:
        """Offer the owning agent's LLM provider.

        Called when the runtime is attached to an agent. Ignored by default —
        only :class:`LLMSelector` has any use for it, and it is offered rather
        than required so that configuring a selector never depends on knowing
        whether it happens to need a model.
        """

    def bind_decisions(self, runtime: Any) -> None:
        """Offer the owning agent's decision runtime, if it has one.

        Offered on the same terms as ``bind_llm`` and for the same reason: a
        config that names a selector should not also have to know which of the
        agent's facilities that selector happens to use. Only
        :class:`DecisionSelector` keeps it.
        """


class IndexSelector(SkillSelector):
    """Default: expose the whole index, expand nothing up front.

    The model reads the index and opens what it needs via ``skill_view``. This is
    the behaviour Claude Code, Hermes and pi all converged on, and it keeps token
    cost proportional to what is actually used.
    """

    type: str = "index"

    async def select(self, index: SkillIndex, query: SkillQuery) -> Selection:
        return Selection(index=list(index.skills), expand=[], rationale="index only")


class AllSkillsSelector(SkillSelector):
    """Expand every available skill immediately.

    Kept for two reasons: it reproduces the "always inject" style some tools use,
    and it gives a measurable baseline to compare progressive disclosure against.
    Not the default — cost grows with the number of skills whether they are used
    or not.
    """

    type: str = "all"

    async def select(self, index: SkillIndex, query: SkillQuery) -> Selection:
        refs = [SkillRef(name=s.name) for s in index.skills if s.available]
        return Selection(index=list(index.skills), expand=refs, rationale="all skills")


class ExplicitSelector(SkillSelector):
    """Expand only what the caller asked for, e.g. via ``/skill <name>``.

    The most predictable option: nothing enters the context unless a human named
    it. Useful when token budget or auditability matters more than autonomy.
    """

    type: str = "explicit"

    async def select(self, index: SkillIndex, query: SkillQuery) -> Selection:
        requested = query.context.get("requested_skills")
        names: list[str] = []
        if isinstance(requested, list):
            names = [str(item) for item in requested]
        refs = [SkillRef(name=name) for name in names if index.by_name(name) is not None]
        return Selection(index=list(index.skills), expand=refs, rationale="explicit request")


_ROUTER_INSTRUCTION = """\
You route an agent's skills. Below is a catalogue of skills, each with a \
description, and the task the agent is about to work on. Decide which skills the \
agent needs to read in order to do this task well.

Choosing nothing is a valid and common answer: most tasks need no skill at all. \
Pick a skill only when its description tells you it is relevant to this specific \
task, not because it is broadly related to the domain. Pick at most {limit}.

When a skill lists sections and only one of them applies, name that section \
instead of the whole skill.

Reply with JSON and nothing else:
{{"skills": [{{"name": "<skill name>", "section": "<section or null>"}}]}}"""


class LLMSelector(SkillSelector):
    """Ask a model which skills the turn needs.

    The deterministic :class:`IndexSelector` leaves the choice to the agent
    mid-turn, which works well but pays for it: the agent sees the catalogue,
    decides, calls ``skill_view``, and only then starts the actual work. This
    selector moves that decision ahead of the turn and into a separate, cheap
    call, so the body of a relevant skill is already in context when the agent
    starts thinking.

    It is worth the extra call when the catalogue is large enough that matching a
    task to a description stops being obvious, and not worth it otherwise —
    hence opt-in, with ``index`` remaining the default.

    Routing failures never take a turn down. An unreachable model, a malformed
    reply or a hallucinated skill name degrades to index-only, which is exactly
    the default behaviour: the agent can still open anything through
    ``skill_view``. That is also why the full index is always returned — a router
    that hid what it did not pick would turn its own mistakes into dead ends.
    """

    type: str = "llm"

    provider: str | None = None
    """Name of an LLM provider in the registry. When unset, the owning agent's
    own provider is used — routing then needs no configuration at all, at the
    cost of paying the agent model's price for a classification task. Naming a
    small, fast model here is usually the better trade."""

    max_expansions: int = 3
    """Ceiling on what routing may pull in. Also stated in the prompt, so the
    model aims for it rather than being silently truncated afterwards."""

    timeout_s: float = 10.0
    """Routing runs before every turn, so a hung provider would hang the whole
    session. On timeout the turn proceeds index-only."""

    instruction: str | None = None
    """Overrides the built-in router prompt. ``{limit}`` is substituted."""

    _llm: BaseLLMProvider | None = PrivateAttr(default=None)

    def bind_llm(self, provider: BaseLLMProvider | None) -> None:
        self._llm = provider

    def _resolve_provider(self) -> BaseLLMProvider | None:
        """The named provider, or the agent's own as a fallback."""
        return resolve_provider(self.provider, self._llm, role="LLM router")

    async def select(self, index: SkillIndex, query: SkillQuery) -> Selection:
        full = list(index.skills)
        # Only available skills are offered: routing to one whose requirements
        # are unmet would just produce a load failure later.
        candidates = [skill for skill in full if skill.available]

        if not candidates or not query.task:
            return Selection(index=full, expand=[], rationale="index only (nothing to route)")

        provider = self._resolve_provider()
        if provider is None:
            get_logger().warning(
                "skills: LLM router has no provider (none named, none bound by an agent)"
            )
            return Selection(index=full, expand=[], rationale="index only (router unavailable)")

        try:
            content = await asyncio.wait_for(
                self._ask(provider, candidates, query.task), timeout=self.timeout_s
            )
        except TimeoutError:
            get_logger().warning(f"skills: LLM router timed out after {self.timeout_s}s")
            return Selection(index=full, expand=[], rationale="index only (router timed out)")
        except Exception as exc:
            get_logger().warning(f"skills: LLM router failed: {type(exc).__name__}")
            get_logger().debug(f"skills router detail: {exc}")
            return Selection(index=full, expand=[], rationale="index only (router failed)")

        refs = self._parse(content, candidates)
        rationale = f"llm router picked {len(refs)}/{len(candidates)}"
        return Selection(index=full, expand=refs, rationale=rationale)

    async def _ask(self, provider: BaseLLMProvider, candidates: list[Any], task: str) -> str:
        from domyn.agents.llm import LLMMessage

        template = self.instruction or _ROUTER_INSTRUCTION
        # Sections always: the router is told to name one when only one applies, and
        # this is a separate one-off call, not the per-request index the toggle trims.
        catalogue = "\n".join(skill.index_line(sections=True) for skill in candidates)
        response = await provider.generate(
            [
                LLMMessage(role="system", content=template.format(limit=self.max_expansions)),
                LLMMessage(role="user", content=f"Catalogue:\n{catalogue}\n\nTask:\n{task}"),
            ]
        )
        return response.content or ""

    def _parse(self, content: str, candidates: list[Any]) -> list[SkillRef]:
        """Turn the reply into refs, dropping anything that does not check out.

        Deliberately forgiving about shape and unforgiving about substance: models
        wrap JSON in prose or fences often enough that failing on it would make
        routing unreliable for no good reason, while a name that is not in the
        catalogue is a hallucination and must not reach a fetch.
        """
        payload = _extract_json(content)
        if payload is None:
            get_logger().warning("skills: LLM router reply was not JSON, ignoring it")
            get_logger().debug(f"skills router reply: {content!r}")
            return []

        entries = payload.get("skills") if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            get_logger().warning("skills: LLM router reply had no skill list, ignoring it")
            return []

        by_name = {skill.name: skill for skill in candidates}
        refs: list[SkillRef] = []
        seen: set[tuple[str, str | None]] = set()
        for entry in entries:
            name = entry.get("name") if isinstance(entry, dict) else entry
            section = entry.get("section") if isinstance(entry, dict) else None
            if not isinstance(name, str):
                continue
            skill = by_name.get(name)
            if skill is None:
                get_logger().warning(f"skills: LLM router picked unknown skill '{name}', ignoring")
                continue
            if section is not None and (
                not isinstance(section, str) or section not in skill.sections
            ):
                # The skill is real, the section is not — take the whole body
                # rather than dropping a correct routing decision.
                get_logger().debug(f"skills: unknown section '{section}' on '{name}', using body")
                section = None
            key = (name, section)
            if key in seen:
                continue
            seen.add(key)
            refs.append(SkillRef(name=name, section=section))
            if len(refs) == self.max_expansions:
                break
        return refs


def resolve_provider(
    name: str | None, fallback: BaseLLMProvider | None, *, role: str
) -> BaseLLMProvider | None:
    """The provider registered as ``name``, or ``fallback`` (the agent's own).

    Shared by everything in this package that asks a model on the side — routing
    and search — so a name means the same thing to both. An unknown name falls
    back with a warning rather than failing: the caller degrades either way.
    """
    if name is None:
        return fallback
    from domyn.agents.registry import get_llm_registry

    found = get_llm_registry().get_llm_provider(name)
    if found is None:
        get_logger().warning(
            f"skills: {role} provider '{name}' is not registered, "
            "falling back to the agent's provider"
        )
        return fallback
    return found


def _extract_json(content: str) -> Any:
    """Parse JSON from a reply that may be wrapped in fences or prose."""
    text = content.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass

    # Scan for the first balanced object or array. A plain "first { to last }"
    # slice breaks on trailing prose that contains braces of its own.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        depth = 0
        for position in range(start, len(text)):
            if text[position] == opener:
                depth += 1
            elif text[position] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : position + 1])
                    except ValueError:
                        break
    return None


class DecisionSelector(SkillSelector):
    """Ask a decision model, one yes/no per skill, in a single request.

    The same job as :class:`LLMSelector` and deliberately not the same code:
    that one asks a model to *write* a list of names, which means a prompt, a
    JSON parse, a hallucinated-name check and seconds of latency. This one asks
    a typed question per candidate and reads probabilities back — no parsing,
    no names to invent, and measured at 187-223 ms for a small catalogue.

    **This is the case where batching is right.** A decision with consequences
    has to be asked on its own, because questions in one request share a forward
    pass and contaminate each other (see ``decisions/approvals.py``). Here the
    questions are homogeneous and independent — "is this skill relevant?" asked
    of each skill — so there is nothing for them to contaminate, and the cost is
    almost entirely the state, which is paid once.

    Like every selector it only decides what to **expand**. The index stays
    complete, so a skill this model did not pick is still one ``skill_view``
    away, and a wrong answer costs one unopened file rather than a dead end.
    That is also why it needs nothing stronger than a stated confidence.
    """

    type: str = "decision"

    max_expansions: int = 3
    """Ceiling on what selection may pull in, highest confidence first."""

    min_confidence: float | None = None
    """Overrides the use case's default threshold."""

    max_questions: int = 64
    """The protocol's own limit. A catalogue larger than this is asked about in
    order and the remainder is left to the index, rather than refused: the
    index is complete either way, so the tail degrades to today's behaviour."""

    _decisions: Any = PrivateAttr(default=None)

    def bind_decisions(self, runtime: Any) -> None:
        self._decisions = runtime

    async def select(self, index: SkillIndex, query: SkillQuery) -> Selection:
        from domyn.agents.decisions import DecisionRuntime, noul

        full = list(index.skills)
        candidates = [skill for skill in full if skill.available]
        if not candidates or not query.task:
            return Selection(index=full, expand=[], rationale="index only (nothing to route)")

        runtime = self._decisions
        if not isinstance(runtime, DecisionRuntime) or not runtime.enabled("skills"):
            return Selection(index=full, expand=[], rationale="index only (no decision model)")

        asked = candidates[: self.max_questions]
        questions = {
            f"s{position}": noul(
                f"The agent is about to work on a task. Is the skill "
                f"'{skill.name}' — {skill.description} — needed to do it well? "
                "Most tasks need no skill at all."
            )
            for position, skill in enumerate(asked)
        }
        decision = await runtime.ask("skills", {"task": query.task}, questions)

        threshold = (
            self.min_confidence if self.min_confidence is not None else runtime.threshold("skills")
        )
        scored: list[tuple[float, Any]] = []
        for position, skill in enumerate(asked):
            answer = decision.get(f"s{position}")
            if not runtime.may_act("skills", answer):
                continue
            if answer.is_true(threshold):
                scored.append((float(answer.value or 0.0), skill))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        refs = [SkillRef(name=skill.name) for _, skill in scored[: self.max_expansions]]
        rationale = (
            f"decision model picked {len(refs)}/{len(asked)} in {decision.latency_ms} ms"
            if decision.known
            else "index only (decision model had no answer)"
        )
        return Selection(index=full, expand=refs, rationale=rationale)
