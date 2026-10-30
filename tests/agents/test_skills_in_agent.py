"""The agent selects from what the runner loaded, and hands the payload down.

This is the middle of the split: loading happened once, at the top of the turn,
in the runner; the planner only ever receives a rendered string. What the agent
owns is the part that depends on *which agent* is running and *what it was
asked* — the selector, the expansion, and the refusal when a required skill
cannot be produced.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from domyn.agents import BaseEvent, ExecutionEventType, Part, Runner
from domyn.agents.core import InteractionContext
from domyn.agents.skills import (
    AllSkillsSelector,
    LoadedSkills,
    SkillContent,
    SkillDescriptor,
    SkillIndex,
    SkillNotFoundError,
    SkillQuery,
    SkillRef,
    SkillSource,
    SkillsRuntime,
)


class PreloadingSource(SkillSource):
    """Local-shaped, and loudly refuses to be fetched from."""

    type: str = "agent_preloading"

    async def index(self, query: SkillQuery) -> SkillIndex:
        return (await self.load(query)).index

    async def fetch(self, ref: SkillRef, query: SkillQuery) -> SkillContent:
        raise AssertionError(f"fetch('{ref.name}') — the body was already loaded")

    async def load(self, query: SkillQuery) -> LoadedSkills:
        return LoadedSkills(
            index=SkillIndex(
                skills=[SkillDescriptor(name="sql", description="d", sections=["examples"])]
            ),
            bodies={"sql": SkillContent(name="sql", body="intro\n\n## examples\n\nSELECT 1")},
        )


class RequiredButMissingSource(SkillSource):
    """Advertises a ``required`` skill whose body cannot be read."""

    type: str = "agent_required_missing"

    async def index(self, query: SkillQuery) -> SkillIndex:
        return SkillIndex(
            skills=[
                SkillDescriptor(
                    name="refund-policy", description="mandatory", enforcement="required"
                )
            ]
        )

    async def fetch(self, ref: SkillRef, query: SkillQuery) -> SkillContent:
        raise OSError("disk gone")


class UnavailableRequiredSource(SkillSource):
    """Advertises a ``required`` skill whose *requirements* are unmet.

    The sibling of ``RequiredButMissingSource`` and not the same case: this one's
    body is perfectly readable, and the descriptor is refused on the index before
    anything is expanded.
    """

    type: str = "agent_unavailable_required"

    async def load(self, query: SkillQuery) -> LoadedSkills:
        return LoadedSkills(index=await self.index(query))

    async def index(self, query: SkillQuery) -> SkillIndex:
        return SkillIndex(
            skills=[
                SkillDescriptor(
                    name="refund-policy",
                    description="mandatory",
                    enforcement="required",
                    available=False,
                    unavailable_reason="binary 'nope-not-real'",
                )
            ]
        )

    async def fetch(self, ref: SkillRef, query: SkillQuery) -> SkillContent:
        raise AssertionError("expansion must never be reached")


class ExplodingSource(SkillSource):
    """A source that is simply down."""

    type: str = "agent_exploding"

    async def index(self, query: SkillQuery) -> SkillIndex:
        raise OSError("skills backend unreachable")

    async def fetch(self, ref: SkillRef, query: SkillQuery) -> SkillContent:
        raise SkillNotFoundError(ref.name)


def _recording_planner() -> tuple[Any, list[dict[str, Any]]]:
    """A planner that records the kwargs it was called with, then answers."""
    calls: list[dict[str, Any]] = []

    async def execute(agent, context, **kwargs):
        calls.append(kwargs)
        yield BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author=agent.name,
            content=[Part(text="done")],
        )

    planner = MagicMock()
    planner.execute = execute
    return planner, calls


def _agent(skills: Any) -> Any:
    from domyn.agents import Agent
    from domyn.agents.llm import CustomLLMProvider

    agent = Agent(
        name="A",
        description="d",
        instruction="i",
        llm_provider=CustomLLMProvider(model_name="m", url="http://x"),
        skills=skills,
    )
    return agent


def _context(text: str = "how do I write this query") -> InteractionContext:
    return InteractionContext(
        conversation_id="c",
        user_id="u",
        agent_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text=text)]
        ),
    )


async def _drive(agent, context, **kwargs) -> list[BaseEvent]:
    return [event async for event, _ in agent.execute(context, **kwargs)]


class TestThePayloadReachesThePlanner:
    @pytest.mark.asyncio
    async def test_a_preloaded_body_is_expanded_without_touching_the_source(self) -> None:
        """The whole point of preloading: expansion costs no second read."""
        agent = _agent(SkillsRuntime(source=PreloadingSource(), selector=AllSkillsSelector()))
        agent.planner, calls = _recording_planner()
        loaded = await agent.skills.load(SkillQuery())

        await _drive(agent, _context(), skills=loaded)

        assert len(calls) == 1
        payload = calls[0]["skills_payload"]
        assert "SELECT 1" in payload, "the preloaded body never reached the planner"

    @pytest.mark.asyncio
    async def test_a_section_request_is_served_from_the_preloaded_body(self) -> None:
        class SectionSelector(AllSkillsSelector):
            type: str = "agent_section"

            async def select(self, index, query):
                selection = await super().select(index, query)
                selection.expand = [SkillRef(name="sql", section="examples")]
                return selection

        agent = _agent(SkillsRuntime(source=PreloadingSource(), selector=SectionSelector()))
        agent.planner, calls = _recording_planner()
        loaded = await agent.skills.load(SkillQuery())

        await _drive(agent, _context(), skills=loaded)

        payload = calls[0]["skills_payload"]
        assert "section: examples" in payload
        assert "intro" not in payload, "the whole body leaked instead of the section"

    @pytest.mark.asyncio
    async def test_an_agent_with_no_skills_hands_the_planner_no_payload(self) -> None:
        """The common case, and the reason the kwarg is omitted rather than empty."""
        agent = _agent(None)
        agent.planner, calls = _recording_planner()

        await _drive(agent, _context())

        assert calls == [{}], f"a skill-less agent was handed {calls}"


class TestFailureModes:
    @pytest.mark.asyncio
    async def test_a_required_skill_that_cannot_expand_refuses_the_turn(self) -> None:
        """A guardrail must refuse, not degrade — but the session survives.

        The reason is the message, and it rides on ``error_message``: the hook
        this replaced reported a randomly chosen generic safety string, because
        `break_loop_response_event` drops the reason when `error_occurred` is set.
        That is a property of the hook path, not of the event type.
        """
        agent = _agent(
            SkillsRuntime(source=RequiredButMissingSource(), selector=AllSkillsSelector())
        )
        agent.planner, calls = _recording_planner()

        events = await _drive(agent, _context())

        assert calls == [], "the planner must never start"
        assert [e.event_type for e in events] == [ExecutionEventType.ERROR_OCCURRED]
        assert "refund-policy" in (events[0].error_message or ""), (
            f"the reason was swallowed: {events[0].error_message!r}"
        )

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.asyncio
    async def test_the_refusal_reaches_the_user_even_while_streaming(self, stream: bool) -> None:
        """Driven through the runner, because the drop this guards against is the runner's.

        A terminal RESPONSE could not carry this: the runner suppresses a
        RESPONSE that no RESPONSE_TOKEN sequence preceded whenever the agent
        streams, and a refusal raised before the planner starts never has tokens
        in front of it — so under ``stream=True`` the turn ended showing nothing
        at all. Driving ``agent.execute`` directly cannot see that, which is why
        this one pays for a runner.
        """
        agent = _agent(
            SkillsRuntime(source=RequiredButMissingSource(), selector=AllSkillsSelector())
        )
        agent.stream = stream

        events = [
            event
            async for event in Runner().run(
                application_name="app",
                user_id="u",
                root=agent,
                user_input=BaseEvent(
                    event_type=ExecutionEventType.USER_INPUT,
                    author="user",
                    content=[Part(text="hi")],
                ),
                conversation_id=f"conv-{stream}",
            )
        ]

        refusals = [e for e in events if e.event_type == ExecutionEventType.ERROR_OCCURRED]
        assert len(refusals) == 1, (
            f"the refusal never reached the consumer: {[e.event_type for e in events]}"
        )
        assert "refund-policy" in (refusals[0].error_message or "")

    @pytest.mark.asyncio
    async def test_an_index_level_refusal_also_reaches_the_user_outside_the_runner(self) -> None:
        """The route the runner's own block does not cover, so it is not dead code.

        Inside a turn the runner refuses an unavailable required skill before any
        agent starts, which makes this handler *look* unreachable. It is not: an
        agent driven directly — no runner, no ``skills=`` kwarg — loads for
        itself, and ``select`` raises on the index exactly as it does for the
        runner. The handler is what turns that into an event instead of a crash.
        """
        agent = _agent(
            SkillsRuntime(source=UnavailableRequiredSource(), selector=AllSkillsSelector())
        )
        agent.planner, calls = _recording_planner()

        events = await _drive(agent, _context())

        assert calls == [], "the planner must never start"
        assert [e.event_type for e in events] == [ExecutionEventType.ERROR_OCCURRED]
        assert events[0].error_code == "RequiredSkillUnavailable"
        message = events[0].error_message or ""
        for fragment in ("refund-policy", "nope-not-real"):
            assert fragment in message, f"{fragment!r} missing from {message!r}"

    @pytest.mark.asyncio
    async def test_an_advisory_failure_degrades_rather_than_refusing(self) -> None:
        agent = _agent(SkillsRuntime(source=ExplodingSource(), selector=AllSkillsSelector()))
        agent.planner, calls = _recording_planner()

        events = await _drive(agent, _context())

        assert calls == [{}], "the turn should have run, with no payload"
        assert any(e.event_type == ExecutionEventType.RESPONSE for e in events)
