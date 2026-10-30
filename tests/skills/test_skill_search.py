"""Tests for ``skill_search``: finding a skill the index does not show in full.

Three things are pinned here. A search answers with a few lines whatever the
catalogue holds — it exists because the index cannot grow with it. A model's
pick is trusted only for names that exist. And every way the model can fail
ends in a word search, never in a broken turn.
"""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from pydantic import PrivateAttr

from domyn.agents.llm import BaseLLMProvider, LLMMessage, LLMResponse
from domyn.agents.skills import (
    InMemorySkillSource,
    LoadedSkills,
    SkillBudget,
    SkillContent,
    SkillDescriptor,
    SkillIndex,
    SkillQuery,
    SkillRef,
    SkillSource,
    SkillsRuntime,
)
from domyn.agents.skills.integration import MAX_SEARCH_RESULTS, build_skill_search_tool
from domyn.agents.skills.search import SkillSearch, rank_by_words


class FakeProvider(BaseLLMProvider):
    """Returns a canned reply and records what it was asked, and with what."""

    type: str = "fake"
    model_name: str = "fake-search"
    supported_structured_outputs: list[Any] | None = None

    reply: str = ""
    fail_with: str | None = None
    hang: bool = False

    _calls: list[tuple[list[LLMMessage], dict[str, Any]]] = PrivateAttr(default_factory=list)

    @property
    def calls(self) -> list[tuple[list[LLMMessage], dict[str, Any]]]:
        return self._calls

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        self._calls.append((messages, kwargs))
        if self.hang:
            await asyncio.sleep(10)
        if self.fail_with:
            raise RuntimeError(self.fail_with)
        return LLMResponse(content=self.reply, generated_tokens=1, execution_time=0.0, usage={})

    async def generate_streaming(
        self, messages: list[LLMMessage], **kwargs: Any
    ) -> AsyncGenerator[LLMResponse, None]:
        """Never used: a search is a single non-streaming call."""
        raise NotImplementedError
        yield  # pragma: no cover


def _skill(name: str, description: str, body: str = "body") -> str:
    return f'---\nname: {name}\ndescription: "{description}"\n---\n{body}\n'


CATALOGUE = {
    "forecast": _skill(
        "time-series-forecasting", "Forecast future values of a time series: sales, demand."
    ),
    "vendas": _skill("previsao-vendas", "Previsão de vendas mensais por região."),
    "slides": _skill("slides-style", "Build PowerPoint slides in the house style."),
    "email": _skill("communication", "Draft or rewrite an email or a Slack message."),
}


def _runtime(provider: FakeProvider | None = None, **search: Any) -> SkillsRuntime:
    runtime = SkillsRuntime(source=InMemorySkillSource(skills=CATALOGUE), search=search)
    runtime.bind_llm(provider)
    return runtime


async def _search(runtime: SkillsRuntime, query: str, **kwargs: Any) -> str:
    call = build_skill_search_tool(runtime)._actual_callable
    result: str = await call(query=query, **kwargs)
    return result


class TestTheModelSearches:
    @pytest.mark.asyncio
    async def test_its_picks_are_returned_in_its_order(self) -> None:
        provider = FakeProvider(reply='{"skills": ["previsao-vendas", "time-series-forecasting"]}')

        out = await _search(_runtime(provider), "sales forecast")

        lines = out.splitlines()
        assert lines[1].startswith("- previsao-vendas:")
        assert lines[2].startswith("- time-series-forecasting:")
        assert "shared words" not in out, "a model's answer is not a word match"

    @pytest.mark.asyncio
    async def test_it_reads_the_catalogue_before_the_query(self) -> None:
        """So consecutive searches share a prefix the provider can cache."""
        provider = FakeProvider(reply='{"skills": []}')

        await _search(_runtime(provider), "sales forecast")

        messages, _ = provider.calls[0]
        user = str(messages[-1].content)
        assert user.index("previsao-vendas") < user.index("sales forecast")

    @pytest.mark.asyncio
    async def test_it_is_told_languages_may_differ(self) -> None:
        provider = FakeProvider(reply='{"skills": []}')

        await _search(_runtime(provider), "x y z")

        messages, _ = provider.calls[0]
        assert "different languages" in str(messages[0].content)

    @pytest.mark.asyncio
    async def test_a_name_it_invents_is_dropped(self) -> None:
        provider = FakeProvider(reply='{"skills": ["made-up", "slides-style"]}')

        out = await _search(_runtime(provider), "presentation")

        assert "made-up" not in out
        assert "- slides-style:" in out

    @pytest.mark.asyncio
    async def test_an_empty_pick_is_an_answer(self) -> None:
        """Not a failure to fall back from: the model looked and found nothing."""
        provider = FakeProvider(reply='{"skills": []}')

        out = await _search(_runtime(provider), "slides")

        assert out.startswith('No skill matches "slides"')

    @pytest.mark.asyncio
    async def test_one_pick_past_the_limit_says_there_are_more(self) -> None:
        provider = FakeProvider(
            reply='{"skills": ["slides-style", "communication", "previsao-vendas"]}'
        )

        out = await _search(_runtime(provider), "anything", limit=2)

        assert "previsao-vendas" not in out
        assert "More skills match" in out

    @pytest.mark.asyncio
    async def test_generation_params_reach_the_provider(self) -> None:
        """The way to turn a reasoning model's thinking off for this call."""
        provider = FakeProvider(reply='{"skills": []}')
        params = {"chat_template_kwargs": {"enable_thinking": False}}

        await _search(_runtime(provider, generation_params=params), "slides")

        _, kwargs = provider.calls[0]
        assert kwargs == params

    @pytest.mark.asyncio
    async def test_a_huge_catalogue_is_narrowed_by_words_first(self) -> None:
        skills = {f"s{n}": _skill(f"filler-{n}", "Unrelated procedure") for n in range(20)}
        skills["target"] = _skill("payroll", "Monthly payroll run")
        provider = FakeProvider(reply='{"skills": ["payroll"]}')
        runtime = SkillsRuntime(
            source=InMemorySkillSource(skills=skills), search={"max_candidates": 3}
        )
        runtime.bind_llm(provider)

        out = await _search(runtime, "payroll")

        messages, _ = provider.calls[0]
        assert "payroll" in str(messages[-1].content)
        assert "filler-0" not in str(messages[-1].content)
        assert "- payroll:" in out


class TestItFallsBackToWords:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "provider",
        [
            FakeProvider(fail_with="boom"),
            FakeProvider(reply="I think slides-style fits."),
            FakeProvider(hang=True),
        ],
        ids=["raises", "not-json", "times-out"],
    )
    async def test_when_the_model_fails(self, provider: FakeProvider) -> None:
        out = await _search(_runtime(provider, timeout_s=0.05), "powerpoint slides")

        assert "- slides-style:" in out
        assert "Matched by shared words" in out

    @pytest.mark.asyncio
    async def test_with_no_model_at_all(self) -> None:
        out = await _search(_runtime(None), "powerpoint slides")

        assert "- slides-style:" in out
        assert "Matched by shared words" in out


class TestWordMatching:
    @staticmethod
    def _rank(query: str, *descriptions: str) -> list[str]:
        skills = [SkillDescriptor(name=f"s{n}", description=d) for n, d in enumerate(descriptions)]
        return [skill.name for skill, _ in rank_by_words(query, skills, {})]

    def test_accents_do_not_matter(self) -> None:
        assert self._rank("previsao", "Previsão de vendas") == ["s0"]

    def test_close_languages_meet(self) -> None:
        """Italian query, Portuguese skill: different endings, same words."""
        assert self._rank("previsione vendite", "Previsão de vendas mensais") == ["s0"]

    def test_unrelated_text_does_not_match(self) -> None:
        assert self._rank("payroll", "Build PowerPoint slides") == []

    def test_stopwords_and_short_words_alone_match_nothing(self) -> None:
        assert self._rank("the and of a", "The and of a everything") == []

    def test_one_shared_word_of_three_is_not_a_match(self) -> None:
        """ "report" is in half the catalogue; it does not make every one a travel expense report."""
        assert self._rank("travel expense report", "Write a monthly report") == []

    def test_half_the_words_is(self) -> None:
        assert self._rank("supplier contract", "Review a supplier agreement") == ["s0"]

    def test_four_shared_letters_are_not_a_word(self) -> None:
        """Measured: ``contract`` matched ``content``, ``quantum`` matched ``quantity``."""
        assert self._rank("contract", "Slide content and layout") == []

    def test_what_trails_far_behind_the_best_is_dropped(self) -> None:
        skills = [
            SkillDescriptor(name="payroll", description="Monthly payroll run"),
            SkillDescriptor(name="other", description="Unrelated"),
        ]
        bodies = {"other": SkillContent(name="other", body="mentions payroll once")}

        ranked = [skill.name for skill, _ in rank_by_words("payroll", skills, bodies)]

        assert ranked == ["payroll"]

    def test_the_name_outranks_the_description(self) -> None:
        skills = [
            SkillDescriptor(name="notes", description="Write release notes for a deploy"),
            SkillDescriptor(name="deploy", description="Ship to production"),
        ]

        ranked = [skill.name for skill, _ in rank_by_words("deploy", skills, {})]

        assert ranked == ["deploy", "notes"]

    def test_the_body_counts_when_nothing_else_does(self) -> None:
        skill = SkillDescriptor(name="ops", description="Operations")
        bodies = {"ops": SkillContent(name="ops", body="Rotate the kubernetes certificates.")}

        assert [s.name for s, _ in rank_by_words("kubernetes", [skill], bodies)] == ["ops"]


class TestTheToolStaysSmall:
    @pytest.mark.asyncio
    async def test_a_query_is_required(self) -> None:
        out = await _search(_runtime(None), "   ")

        assert out.startswith("A query is required")

    @pytest.mark.asyncio
    async def test_the_limit_is_capped(self) -> None:
        skills = {f"s{n}": _skill(f"report-{n}", "Monthly report") for n in range(30)}
        runtime = SkillsRuntime(source=InMemorySkillSource(skills=skills))

        out = await _search(runtime, "monthly report", limit=50)

        assert sum(line.startswith("- report-") for line in out.splitlines()) == MAX_SEARCH_RESULTS
        assert "More skills match" in out

    @pytest.mark.asyncio
    async def test_descriptions_are_shortened(self) -> None:
        long = "Monthly report " + "with many details " * 20
        runtime = SkillsRuntime(source=InMemorySkillSource(skills={"r": _skill("report", long)}))

        out = await _search(runtime, "monthly report")

        line = next(line for line in out.splitlines() if line.startswith("- report:"))
        assert len(line) <= len("- report: ") + 120


class TestItSearchesWhatTheIndexLeftOut:
    @pytest.mark.asyncio
    async def test_a_skill_not_listed_is_found(self) -> None:
        """The reason the tool exists: the index counted it, the search finds it."""
        skills = {f"s{n}": _skill(f"filler-{n}", "Unrelated procedure " * 5) for n in range(30)}
        skills["zz"] = _skill("zz-payroll", "Monthly payroll run")
        runtime = SkillsRuntime(
            source=InMemorySkillSource(skills=skills), budget=SkillBudget(max_index_chars=100)
        )
        loaded = await runtime.load(SkillQuery())
        assert "zz-payroll" in runtime.layout(loaded.index.skills).omitted

        call = build_skill_search_tool(runtime)._actual_callable
        out = await call(query="payroll", loaded_skills=loaded)

        assert "- zz-payroll:" in out

    @pytest.mark.asyncio
    async def test_the_index_points_at_the_tool(self) -> None:
        skills = {f"s{n}": _skill(f"filler-{n}", "Unrelated") for n in range(30)}
        runtime = SkillsRuntime(
            source=InMemorySkillSource(skills=skills), budget=SkillBudget(max_index_chars=100)
        )
        selection, contents = await runtime.resolve(SkillQuery())

        assert "find them with skill_search" in runtime.build_payload(selection, contents)


class TestASourceMaySearchItself:
    @pytest.mark.asyncio
    async def test_its_answer_is_used(self) -> None:
        class SearchingSource(SkillSource):
            type: str = "searching"

            async def index(self, query: SkillQuery) -> SkillIndex:
                return SkillIndex(skills=[])

            async def fetch(self, ref: SkillRef, query: SkillQuery) -> SkillContent:
                raise NotImplementedError

            async def search(
                self, text: str, limit: int, query: SkillQuery
            ) -> list[SkillDescriptor] | None:
                return [SkillDescriptor(name="remote-one", description="Found remotely")]

        provider = FakeProvider(reply='{"skills": []}')
        runtime = SkillsRuntime(source=SearchingSource())
        runtime.bind_llm(provider)

        result = await runtime.find("x", 5, LoadedSkills(), SkillQuery())

        assert [hit.name for hit in result.hits] == ["remote-one"]
        assert result.method == "source"
        assert provider.calls == [], "the source searched, the model need not"


def test_search_is_configurable_from_yaml() -> None:
    runtime = SkillsRuntime.model_validate(
        {
            "source": {"type": "memory", "skills": {}},
            "search": {"provider": "small", "timeout_s": 3, "max_candidates": 50},
        }
    )

    assert runtime.search == SkillSearch(provider="small", timeout_s=3, max_candidates=50)
