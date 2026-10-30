"""Tests for LLM-based skill routing.

Two things matter here and they pull in opposite directions: the router has to
cope with the sloppy output models actually produce, and it must never let a
hallucinated name reach a fetch. Most of these tests are about that line.

The other half is degradation. Routing runs before every turn, so any way it can
fail — no provider, timeout, prose instead of JSON — has to end in a normal
index-only turn rather than in a broken session.
"""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from pydantic import PrivateAttr

from domyn.agents.llm import BaseLLMProvider, LLMMessage, LLMResponse
from domyn.agents.skills import SkillDescriptor, SkillIndex, SkillQuery
from domyn.agents.skills.selectors import LLMSelector, _extract_json


class FakeProvider(BaseLLMProvider):
    """Returns a canned reply and records what it was asked."""

    type: str = "fake"
    model_name: str = "fake-router"
    supported_structured_outputs: list[Any] | None = None

    reply: str = ""
    fail_with: str | None = None
    hang: bool = False

    _calls: list[list[LLMMessage]] = PrivateAttr(default_factory=list)

    @property
    def calls(self) -> list[list[LLMMessage]]:
        return self._calls

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        self._calls.append(messages)
        if self.hang:
            await asyncio.sleep(10)
        if self.fail_with:
            raise RuntimeError(self.fail_with)
        return LLMResponse(content=self.reply, generated_tokens=1, execution_time=0.0, usage={})

    async def generate_streaming(
        self, messages: list[LLMMessage], **kwargs: Any
    ) -> AsyncGenerator[LLMResponse, None]:
        """Never used: routing is a single non-streaming call."""
        raise NotImplementedError
        yield  # pragma: no cover


def _index(*descriptors: SkillDescriptor) -> SkillIndex:
    return SkillIndex(skills=list(descriptors))


def _skill(name: str, **kwargs: Any) -> SkillDescriptor:
    kwargs.setdefault("description", f"how to {name}")
    return SkillDescriptor(name=name, **kwargs)


def _router(reply: str = "", **kwargs: Any) -> tuple[LLMSelector, FakeProvider]:
    provider = FakeProvider(reply=reply)
    selector = LLMSelector(**kwargs)
    selector.bind_llm(provider)
    return selector, provider


QUERY = SkillQuery(task="write a release note")


class TestRouting:
    @pytest.mark.asyncio
    async def test_picked_skills_are_expanded(self) -> None:
        selector, _ = _router('{"skills": [{"name": "changelog"}]}')
        selection = await selector.select(_index(_skill("changelog"), _skill("deploy")), QUERY)
        assert [ref.name for ref in selection.expand] == ["changelog"]

    @pytest.mark.asyncio
    async def test_picking_nothing_is_a_valid_answer(self) -> None:
        """The prompt says so explicitly, and it must not be read as a failure."""
        selector, _ = _router('{"skills": []}')
        selection = await selector.select(_index(_skill("changelog")), QUERY)
        assert selection.expand == []
        assert selection.rationale == "llm router picked 0/1"

    @pytest.mark.asyncio
    async def test_the_index_is_always_complete(self) -> None:
        """A router that hid what it did not pick would turn its own mistakes
        into dead ends — the agent could no longer open the skill itself."""
        selector, _ = _router('{"skills": [{"name": "changelog"}]}')
        index = _index(_skill("changelog"), _skill("deploy"), _skill("triage"))
        selection = await selector.select(index, QUERY)
        assert [d.name for d in selection.index] == ["changelog", "deploy", "triage"]

    @pytest.mark.asyncio
    async def test_a_section_can_be_routed_to(self) -> None:
        selector, _ = _router('{"skills": [{"name": "deploy", "section": "rollback"}]}')
        index = _index(_skill("deploy", sections=["rollback", "canary"]))
        selection = await selector.select(index, QUERY)
        assert selection.expand[0].section == "rollback"

    @pytest.mark.asyncio
    async def test_the_catalogue_and_task_reach_the_model(self) -> None:
        selector, provider = _router('{"skills": []}')
        await selector.select(_index(_skill("changelog")), QUERY)
        system, user = provider.calls[0]
        assert "at most 3" in str(system.content)
        assert "changelog" in str(user.content)
        assert "write a release note" in str(user.content)

    @pytest.mark.asyncio
    async def test_unavailable_skills_are_not_offered(self) -> None:
        """Routing to one whose requirements are unmet only produces a load
        failure later, so it is kept out of the catalogue the model sees — while
        staying in the index the agent sees."""
        selector, provider = _router('{"skills": []}')
        index = _index(
            _skill("changelog"),
            _skill("deploy", available=False, unavailable_reason="kubectl missing"),
        )
        selection = await selector.select(index, QUERY)
        assert "deploy" not in str(provider.calls[0][1].content)
        assert [d.name for d in selection.index] == ["changelog", "deploy"]


class TestGuardsAgainstBadOutput:
    @pytest.mark.asyncio
    async def test_hallucinated_names_are_dropped(self) -> None:
        selector, _ = _router('{"skills": [{"name": "nonexistent"}, {"name": "changelog"}]}')
        selection = await selector.select(_index(_skill("changelog")), QUERY)
        assert [ref.name for ref in selection.expand] == ["changelog"]

    @pytest.mark.asyncio
    async def test_an_unknown_section_falls_back_to_the_whole_body(self) -> None:
        """The skill is real and the routing decision was probably right; only
        the section is wrong, and dropping the whole pick would waste it."""
        selector, _ = _router('{"skills": [{"name": "deploy", "section": "nope"}]}')
        selection = await selector.select(_index(_skill("deploy", sections=["rollback"])), QUERY)
        assert selection.expand[0].section is None

    @pytest.mark.asyncio
    async def test_more_picks_than_allowed_are_truncated(self) -> None:
        selector, _ = _router(
            '{"skills": [{"name": "a"}, {"name": "b"}, {"name": "c"}]}', max_expansions=2
        )
        selection = await selector.select(_index(_skill("a"), _skill("b"), _skill("c")), QUERY)
        assert [ref.name for ref in selection.expand] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_duplicates_are_collapsed(self) -> None:
        selector, _ = _router('{"skills": [{"name": "a"}, {"name": "a"}]}')
        selection = await selector.select(_index(_skill("a")), QUERY)
        assert len(selection.expand) == 1

    @pytest.mark.parametrize(
        "reply",
        [
            '```json\n{"skills": [{"name": "a"}]}\n```',
            'Sure! Here you go:\n{"skills": [{"name": "a"}]}',
            '{"skills": [{"name": "a"}]}\nHope that helps {see above}.',
            '[{"name": "a"}]',
            '["a"]',
        ],
        ids=["fenced", "preamble", "trailing-braces", "bare-array", "bare-strings"],
    )
    @pytest.mark.asyncio
    async def test_json_is_recovered_from_realistic_replies(self, reply: str) -> None:
        """Models wrap JSON in fences and prose often enough that being strict
        here would make routing unreliable for no good reason."""
        selector, _ = _router(reply)
        selection = await selector.select(_index(_skill("a")), QUERY)
        assert [ref.name for ref in selection.expand] == ["a"]


class TestDegradation:
    """Routing runs before every turn, so no failure may cost more than the
    routing itself."""

    @pytest.mark.asyncio
    async def test_prose_instead_of_json_degrades_to_index_only(self) -> None:
        selector, _ = _router("I think you should read the changelog skill.")
        selection = await selector.select(_index(_skill("changelog")), QUERY)
        assert selection.expand == []
        assert [d.name for d in selection.index] == ["changelog"]

    @pytest.mark.asyncio
    async def test_a_provider_error_degrades_to_index_only(self) -> None:
        provider = FakeProvider(fail_with="connection reset")
        selector = LLMSelector()
        selector.bind_llm(provider)
        selection = await selector.select(_index(_skill("a")), QUERY)
        assert selection.expand == []
        assert selection.rationale == "index only (router failed)"

    @pytest.mark.asyncio
    async def test_a_hung_provider_does_not_hang_the_turn(self) -> None:
        provider = FakeProvider(hang=True)
        selector = LLMSelector(timeout_s=0.05)
        selector.bind_llm(provider)
        selection = await selector.select(_index(_skill("a")), QUERY)
        assert selection.rationale == "index only (router timed out)"

    @pytest.mark.asyncio
    async def test_no_provider_at_all_degrades_to_index_only(self) -> None:
        """An unbound selector — configured but never attached to an agent."""
        selection = await LLMSelector().select(_index(_skill("a")), QUERY)
        assert selection.rationale == "index only (router unavailable)"

    @pytest.mark.asyncio
    async def test_no_task_means_no_call(self) -> None:
        """Nothing to route on, so the model is not paid to say so."""
        selector, provider = _router('{"skills": [{"name": "a"}]}')
        selection = await selector.select(_index(_skill("a")), SkillQuery())
        assert provider.calls == []
        assert selection.rationale == "index only (nothing to route)"

    @pytest.mark.asyncio
    async def test_an_empty_catalogue_means_no_call(self) -> None:
        selector, provider = _router('{"skills": []}')
        await selector.select(_index(), QUERY)
        assert provider.calls == []


class TestProviderResolution:
    @pytest.mark.asyncio
    async def test_a_named_provider_wins_over_the_agents_own(self) -> None:
        from domyn.agents.registry import get_llm_registry

        named = FakeProvider(reply='{"skills": [{"name": "a"}]}')
        get_llm_registry().register_llm_provider("router-model", named)
        try:
            agents_own = FakeProvider(reply='{"skills": []}')
            selector = LLMSelector(provider="router-model")
            selector.bind_llm(agents_own)
            selection = await selector.select(_index(_skill("a")), QUERY)
            assert [ref.name for ref in selection.expand] == ["a"]
            assert agents_own.calls == []
        finally:
            get_llm_registry().clear()

    @pytest.mark.asyncio
    async def test_an_unregistered_name_falls_back_to_the_agents_own(self) -> None:
        """A typo in the config must not silently disable routing."""
        selector, provider = _router('{"skills": [{"name": "a"}]}', provider="typo")
        selection = await selector.select(_index(_skill("a")), QUERY)
        assert provider.calls, "the agent's own provider should have been used"
        assert [ref.name for ref in selection.expand] == ["a"]


class TestConfiguration:
    def test_it_is_resolvable_by_type_tag_from_yaml(self) -> None:
        from domyn.agents.skills import SkillsRuntime

        runtime = SkillsRuntime.model_validate(
            {
                "source": {"type": "memory", "skills": {}},
                "selector": {"type": "llm", "max_expansions": 5, "timeout_s": 2.0},
            }
        )
        assert isinstance(runtime.selector, LLMSelector)
        assert runtime.selector.max_expansions == 5

    def test_the_router_prompt_can_be_overridden(self) -> None:
        selector = LLMSelector(instruction="pick up to {limit}, reply in JSON")
        assert selector.instruction == "pick up to {limit}, reply in JSON"

    def test_index_stays_the_default_selector(self) -> None:
        """Routing costs a call per turn: it has to be asked for."""
        from domyn.agents.skills import SkillsRuntime

        runtime = SkillsRuntime.model_validate({"source": {"type": "memory", "skills": {}}})
        assert not isinstance(runtime.selector, LLMSelector)


class TestJsonExtraction:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ('{"a": 1}', {"a": 1}),
            ('noise {"a": 1} noise', {"a": 1}),
            ('{"a": {"b": 2}} trailing }', {"a": {"b": 2}}),
            ("[1, 2]", [1, 2]),
            ("no json here", None),
            ("{unterminated", None),
        ],
    )
    def test_balanced_scanning(self, text: str, expected: Any) -> None:
        """A naive first-brace-to-last-brace slice breaks on trailing prose that
        contains braces of its own — hence the depth counter."""
        assert _extract_json(text) == expected
