"""Searching the whole catalogue, for the ``skill_search`` tool.

The index is shortened to fit its budget (:meth:`SkillsRuntime.layout`), and past
a few hundred skills it is shortened to names, then to a count. Search is what
reaches the rest: the model asks for what it needs, and gets the few best
matches, never the catalogue.

Three ways to search, tried in order:

* the **source**'s own (:meth:`SkillSource.search`), for a catalogue that lives
  elsewhere and is better searched where it lives;
* a **model**, given the catalogue and the query. Skills may be written in any
  language, and a model matches by meaning: an Italian skill about "previsione
  delle vendite" is found by "sales forecast", which no word match does;
* **words**, when there is no model or it fails: folded to plain letters and
  compared by three-letter pieces, so "previsione vendite" still finds
  "previsão de vendas". Languages that share no words still miss each other,
  which is why the result says it came from here.

Like routing, a search never takes a turn down: whatever fails degrades to the
next way, and the last one needs nothing but the catalogue.
"""

from __future__ import annotations

import asyncio
import re
import unicodedata
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, PrivateAttr

from domyn.agents.logger import get_logger
from domyn.agents.skills.models import SkillContent, SkillDescriptor
from domyn.agents.skills.selectors import _extract_json, resolve_provider

if TYPE_CHECKING:
    from domyn.agents.llm import BaseLLMProvider


SearchMethod = Literal["source", "model", "words"]


class SearchResult(BaseModel):
    """The best matches for one query, best first."""

    hits: list[SkillDescriptor] = Field(default_factory=list)
    more: bool = False
    """More skills matched than were returned: the query can be narrowed."""
    method: SearchMethod = "words"


_SEARCH_INSTRUCTION = """\
You search a catalogue of skills for an agent. Below is the catalogue, one skill \
per line with what it is for, and a search query. List the skills that match the \
query, most relevant first.

Skills and queries may be written in different languages: match by meaning, not \
by shared words. List only skills that genuinely match; an empty list is a valid \
answer. List at most {limit}.

Reply with JSON and nothing else:
{{"skills": ["<skill name>", ...]}}"""


class SkillSearch(BaseModel):
    """How ``skill_search`` searches when the source does not do it itself.

    Declared on the runtime as ``search:``, so it is configured beside the
    selector it resembles: the same ``provider`` fallback, the same timeout.
    """

    provider: str | None = None
    """Name of an LLM provider in the registry. When unset, the owning agent's
    own provider is used. A small, fast model is the better trade."""

    timeout_s: float = 10.0
    """A search runs in the middle of a turn. On timeout it answers from words."""

    generation_params: dict[str, Any] = Field(default_factory=dict)
    """Passed to the provider's ``generate`` as keyword arguments. The place to turn
    off a reasoning model's thinking for this call, which providers spell
    differently (vLLM serving Qwen3: ``{chat_template_kwargs: {enable_thinking:
    false}}``): a model that thinks before picking a few names spends the timeout
    doing it."""

    max_candidates: int = 1_000
    """Past this many skills, the model is shown only the best word matches. The
    catalogue goes into a single request, and a few thousand descriptions would
    not fit one."""

    description_chars: int = 120
    """Descriptions are cut to this in the catalogue the model reads."""

    instruction: str | None = None
    """Overrides the built-in search prompt. ``{limit}`` is substituted."""

    _llm: BaseLLMProvider | None = PrivateAttr(default=None)

    def bind_llm(self, provider: BaseLLMProvider | None) -> None:
        self._llm = provider

    async def find(
        self,
        text: str,
        skills: list[SkillDescriptor],
        bodies: dict[str, SkillContent],
        limit: int,
    ) -> SearchResult:
        """The ``limit`` skills that best match ``text``, by model or else by words."""
        provider = resolve_provider(self.provider, self._llm, role="skill search")
        if provider is not None and skills:
            candidates = skills
            if len(candidates) > self.max_candidates:
                candidates = [
                    skill for skill, _ in rank_by_words(text, skills, bodies)[: self.max_candidates]
                ]
            try:
                names = await asyncio.wait_for(
                    self._ask(provider, text, candidates, limit + 1), timeout=self.timeout_s
                )
            except TimeoutError:
                get_logger().warning(f"skills: search timed out after {self.timeout_s}s")
            except Exception as exc:
                get_logger().warning(f"skills: search model failed: {type(exc).__name__}")
                get_logger().debug(f"skills search detail: {exc}")
            else:
                if names is not None:
                    by_name = {skill.name: skill for skill in candidates}
                    hits = [by_name[name] for name in names if name in by_name]
                    return SearchResult(hits=hits[:limit], more=len(hits) > limit, method="model")

        ranked = rank_by_words(text, skills, bodies)
        return SearchResult(
            hits=[skill for skill, _ in ranked[:limit]], more=len(ranked) > limit, method="words"
        )

    async def _ask(
        self,
        provider: BaseLLMProvider,
        text: str,
        candidates: list[SkillDescriptor],
        limit: int,
    ) -> list[str] | None:
        """Names the model picked, in its order; ``None`` when its reply is unusable.

        The catalogue goes before the query, so that consecutive searches share a
        prefix a provider can cache.
        """
        from domyn.agents.llm import LLMMessage

        template = self.instruction or _SEARCH_INSTRUCTION
        catalogue = "\n".join(
            skill.index_line(max_description_chars=self.description_chars) for skill in candidates
        )
        response = await provider.generate(
            [
                LLMMessage(role="system", content=template.format(limit=limit)),
                LLMMessage(role="user", content=f"Catalogue:\n{catalogue}\n\nQuery:\n{text}"),
            ],
            **self.generation_params,
        )
        payload = _extract_json(response.content or "")
        entries = payload.get("skills") if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            get_logger().warning("skills: search model reply had no skill list, ignoring it")
            get_logger().debug(f"skills search reply: {response.content!r}")
            return None
        names: list[str] = []
        for entry in entries:
            name = entry.get("name") if isinstance(entry, dict) else entry
            if isinstance(name, str) and name not in names:
                names.append(name)
        return names


# Words that match everything and say nothing, in the languages skills are most
# likely written in. Short words (under three letters) are dropped anyway.
_STOPWORDS = frozenset(
    " ".join(
        (
            "the and for with how use when what this that from into",  # English
            "una uno per con che del della delle dei come nel nella sono",  # Italian
            "para com uma dos das los las por que pelo pela",  # Portuguese, Spanish
            "une des pour avec dans les",  # French
            "und der die mit für von",  # German
        )
    ).split()
)

# How much a query word counts, by where in the skill it is found.
_WEIGHTS = {"name": 3.0, "description": 2.0, "sections": 1.5, "body": 0.5}


def rank_by_words(
    text: str, skills: list[SkillDescriptor], bodies: dict[str, SkillContent]
) -> list[tuple[SkillDescriptor, float]]:
    """Skills sharing words with ``text``, best first; ties keep catalogue order.

    Each query word scores the best place it is found — the name, the
    description, a section heading, the body — and a skill scores the average.
    Name, description and headings match loosely (:func:`_similar`), so a word
    in another form or a close language still counts; the body, which can be
    thousands of words, only matches exactly.

    Two cuts keep a loose match from becoming a list of everything. A skill must
    match at least half the query's words: one common word ("report") shared by
    half the catalogue is not a match for "travel expense report". And it must
    score at least half of the best: what trails far behind the top hit is
    noise. Measured on the builtin skills, without them a query that matched
    nothing still returned five skills.
    """
    query = _words(text)
    if not query:
        return []

    scored: list[tuple[SkillDescriptor, float]] = []
    for skill in skills:
        body = bodies.get(skill.name)
        fields = {
            "name": _words(skill.name),
            "description": _words(skill.description),
            "sections": _words(" ".join(skill.sections)),
        }
        body_words = set(_words(body.body)) if body is not None else set()
        total = 0.0
        matched = 0
        for word in query:
            best = 0.0
            for field, words in fields.items():
                if any(_similar(word, other) for other in words):
                    best = max(best, _WEIGHTS[field])
            if not best and word in body_words:
                best = _WEIGHTS["body"]
            total += best
            matched += best > 0
        if matched and 2 * matched >= len(query):
            scored.append((skill, total / len(query)))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    if not scored:
        return []
    top = scored[0][1]
    return [pair for pair in scored if pair[1] >= top / 2]


def _words(text: str) -> list[str]:
    """Lower-case words with accents removed, minus stopwords and short words."""
    folded = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return [
        word for word in re.findall(r"[^\W_]+", plain) if len(word) >= 3 and word not in _STOPWORDS
    ]


def _similar(a: str, b: str) -> bool:
    """The same word, or close enough: a shared start, or most three-letter pieces.

    The start catches a longer stem across close languages (``contract``/
    ``contratti`` share none of it past ``contr``), the pieces a changed ending or
    spelling (``previsione``/``previsao``, ``vendite``/``vendas``). Both still
    match the odd unrelated word (``quantum``/``quantity``), which is what the
    cuts in :func:`rank_by_words` are for. Five letters, not four: four matched
    ``contract`` to ``content``.
    """
    if a == b:
        return True
    if len(a) >= 6 and len(b) >= 6 and a[:5] == b[:5]:
        return True
    pieces_a, pieces_b = _pieces(a), _pieces(b)
    shared = len(pieces_a & pieces_b)
    return 2 * shared / (len(pieces_a) + len(pieces_b)) >= 0.45


@lru_cache(maxsize=4096)
def _pieces(word: str) -> frozenset[str]:
    padded = f" {word} "
    return frozenset(padded[i : i + 3] for i in range(len(padded) - 2))
