"""What the runner tells a tool about the agent call it runs in, and when.

- ``agent_call_id`` (reserved, never in a schema): empty for the agent the
  conversation is addressed to, the call's interaction id in a sub-agent. The
  same through a pause for approval, different for every call, copies of one
  agent included. What a tool keeps per agent call (a browser's page) is keyed
  by it.
- The built-in tools are told when a call ends (``BuiltinTools.end_agent_call``),
  on every AGENT_END, and not while the call only waits for an answer.
- A tool that is not ``parallel_safe`` runs alone and in order among the calls
  of one response, as a call needing approval does, but without being asked.
"""

# No `from __future__ import annotations`: builds real agents and tools.
import asyncio
from typing import Any

import pytest

from domyn.agents import Agent
from domyn.agents.core import BaseEvent, ExecutionEventType
from domyn.agents.llm import LLMMessage, LLMResponse, LLMToolCall, ToolCallFunction
from domyn.agents.runner import Runner
from domyn.agents.tools import FunctionTool
from domyn.agents.tools.builtin import BuiltinTools
from tests.runner.test_parallel_handoffs import (
    PLANNERS,
    _approve,
    _ask,
    _planner,
    _run,
    _Script,
    _write_tool,
)


class _Ends(BuiltinTools):
    """A bundle that records the agent calls it is told have ended."""

    def __init__(self, ended: list[str]) -> None:
        super().__init__(tools=[])
        self.ended = ended

    async def end_agent_call(self, conversation_id: str, agent_call: str) -> None:
        self.ended.append(agent_call)


def _look(seen: list[tuple[str, str]]) -> FunctionTool:
    async def look(what: str, agent_call_id: str = "") -> str:
        """Look at something.

        Args:
            what: what to look at.
        """
        seen.append((what, agent_call_id))
        return f"saw {what}"

    return FunctionTool(func=look)


class _LooksOnce(_Script):
    """Looks, then answers once the look's result is the last thing it is shown:
    decided from the request, so every call to it looks once, copies sharing this
    provider included, whatever the order of the requests."""

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        self.shown.append(list(messages))
        if messages[-1].role == "tool":
            return LLMResponse(content="done", generated_tokens=1, execution_time=0.0, usage={})
        return LLMResponse(
            content="",
            generated_tokens=1,
            execution_time=0.0,
            usage={},
            tool_calls=[
                LLMToolCall(
                    id=f"{self.label}_look_{len(self.shown)}",
                    type="function",
                    function=ToolCallFunction(name="look", arguments='{"what": "x"}'),
                )
            ],
        )


def _sub(
    name: str, planner: str, seen: list[tuple[str, str]], replies: list[Any], **fields: Any
) -> Agent:
    """A sub-agent that answers with ``replies`` in turn; with none, it looks once."""
    provider = _Script(label=name, replies=replies) if replies else _LooksOnce(label=name)
    return Agent(
        name=name,
        description=f"looks at {name.lower()}",
        instruction="do the task",
        llm_provider=provider,
        planner=_planner(planner),
        tools=[_look(seen), *fields.pop("tools", [])],
        **fields,
    )


def _looks(what: str) -> list[tuple[str, str]]:
    return [("look", f'{{"what": "{what}"}}')]


def _starts(events: list[BaseEvent]) -> dict[str, list[str]]:
    """The interaction id of every sub-agent call, by agent."""
    found: dict[str, list[str]] = {}
    for e in events:
        if e.event_type == ExecutionEventType.AGENT_START and e.author != "MainAgent":
            found.setdefault(e.author, []).append(str(e.interaction_id))
    return found


class TestTheAgentCallId:
    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_the_agent_the_conversation_is_addressed_to_has_none(self, planner: str) -> None:
        seen: list[tuple[str, str]] = []
        root = Agent(
            name="MainAgent",
            description="the agent",
            instruction="look",
            llm_provider=_Script(label="main", replies=[_looks("root"), "Done."]),
            planner=_planner(planner),
            tools=[_look(seen)],
        )

        await _run(Runner(), root, f"c_root_{planner}")

        assert seen == [("root", "")]

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_sub_agents_at_the_same_time_each_have_their_own(self, planner: str) -> None:
        seen: list[tuple[str, str]] = []
        subs = [
            _sub(name, planner, seen, [_looks(name), f"{name} done"]) for name in ("Alpha", "Beta")
        ]
        root = Agent(
            name="MainAgent",
            description="the agent",
            instruction="delegate",
            llm_provider=_Script(
                label="main",
                replies=[
                    [("Alpha", '{"task": "a"}'), ("Beta", '{"task": "b"}')],
                    "Done.",
                ],
            ),
            planner=_planner(planner, parallel_handoffs=True),
            sub_agents=subs,
        )

        events = await _run(Runner(), root, f"c_two_{planner}")

        starts = _starts(events)
        assert dict(seen) == {"Alpha": starts["Alpha"][0], "Beta": starts["Beta"][0]}
        assert all(seen_id for _, seen_id in seen)
        assert len({seen_id for _, seen_id in seen}) == 2

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_copies_of_one_agent_and_a_lone_hand_off_each_have_their_own(
        self, planner: str
    ) -> None:
        """Two copies in one response, then the same agent alone in the next:
        three calls, three ids; a sub-agent called alone is a call too."""
        seen: list[tuple[str, str]] = []
        alpha = _sub("Alpha", planner, seen, [])
        main = _Script(
            label="main",
            replies=[
                [("Alpha", '{"task": "one"}'), ("Alpha", '{"task": "two"}')],
                [("Alpha", '{"task": "three"}')],
                "Done.",
            ],
        )
        root = Agent(
            name="MainAgent",
            description="the agent",
            instruction="delegate",
            llm_provider=main,
            planner=_planner(planner, parallel_handoffs=True),
            sub_agents=[alpha],
        )

        events = await _run(Runner(), root, f"c_copies_{planner}")

        ids = [seen_id for _, seen_id in seen]
        assert len(ids) == 3 and all(ids)
        assert len(set(ids)) == 3
        assert sorted(ids) == sorted(_starts(events)["Alpha"])

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_it_survives_a_pause_for_approval(self, planner: str) -> None:
        """A sub-agent waiting for an answer is the same call when it resumes, in
        another run: its browser is still its own."""
        seen: list[tuple[str, str]] = []
        ended: list[str] = []
        alpha = _sub(
            "Alpha",
            planner,
            seen,
            [
                _looks("before"),
                [("write", '{"path": "x"}')],
                _looks("after"),
                "done",
            ],
            tools=[_write_tool([])],
            builtin_tools=[_Ends(ended)],
        )
        root = Agent(
            name="MainAgent",
            description="the agent",
            instruction="delegate",
            llm_provider=_Script(label="main", replies=[[("Alpha", '{"task": "a"}')], "Done."]),
            planner=_planner(planner),
            sub_agents=[alpha],
        )
        runner = Runner()
        conversation = f"c_pause_{planner}"

        first = await _run(runner, root, conversation)
        assert ended == [], "a call waiting for an answer has not ended"
        await _run(runner, root, conversation, _approve())

        (call,) = _starts(first)["Alpha"]
        assert seen == [("before", call), ("after", call)]
        # Told once, when it ended (and of the caller's end, which keeps nothing).
        assert ended[0] == call and ended.count(call) == 1


class TestTheEndOfACall:
    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_every_sub_agent_call_that_ends_is_reported(self, planner: str) -> None:
        seen: list[tuple[str, str]] = []
        ended: list[str] = []
        subs = [
            _sub(name, planner, seen, [_looks(name), "done"], builtin_tools=[_Ends(ended)])
            for name in ("Alpha", "Beta")
        ]
        root = Agent(
            name="MainAgent",
            description="the agent",
            instruction="delegate",
            llm_provider=_Script(
                label="main",
                replies=[
                    [("Alpha", '{"task": "a"}'), ("Beta", '{"task": "b"}')],
                    "Done.",
                ],
            ),
            planner=_planner(planner, parallel_handoffs=True),
            sub_agents=subs,
        )

        events = await _run(Runner(), root, f"c_end_{planner}")

        calls = {i for ids in _starts(events).values() for i in ids}
        # Each bundle hears of every call that ends (both are told of both): a
        # bundle that kept nothing for a call ignores it.
        assert calls <= set(ended)

    @pytest.mark.asyncio
    async def test_a_bundle_that_fails_to_release_does_not_fail_the_turn(self) -> None:
        class _Failing(BuiltinTools):
            async def end_agent_call(self, conversation_id: str, agent_call: str) -> None:
                raise RuntimeError("the browser is gone")

        seen: list[tuple[str, str]] = []
        alpha = _sub("Alpha", "tool_use", seen, ["done"], builtin_tools=[_Failing(tools=[])])
        root = Agent(
            name="MainAgent",
            description="the agent",
            instruction="delegate",
            llm_provider=_Script(label="main", replies=[[("Alpha", '{"task": "a"}')], "Done."]),
            planner=_planner("tool_use"),
            sub_agents=[alpha],
        )

        events = await _run(Runner(), root, "c_failing_release")

        answers = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]
        assert answers and answers[-1].author == "MainAgent"


def _slow(log: list[str], name: str, parallel_safe: bool) -> FunctionTool:
    async def step(what: str) -> str:
        """Do a slow step.

        Args:
            what: the step.
        """
        log.append(f"start {what}")
        await asyncio.sleep(0.05)
        log.append(f"end {what}")
        return f"did {what}"

    step.__name__ = name
    return FunctionTool(func=step, parallel_safe=parallel_safe)


class TestToolsThatAreNotParallelSafe:
    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_their_calls_run_one_at_a_time_in_order(self, planner: str) -> None:
        log: list[str] = []
        calls = [
            ("read", '{"what": "r1"}'),
            ("browse", '{"what": "b1"}'),
            ("browse", '{"what": "b2"}'),
            ("read", '{"what": "r2"}'),
            ("read", '{"what": "r3"}'),
        ]
        root = Agent(
            name="MainAgent",
            description="the agent",
            instruction="work",
            llm_provider=_Script(label="main", replies=[calls, "Done."]),
            planner=_planner(planner),
            tools=[_slow(log, "read", True), _slow(log, "browse", False)],
        )

        await _run(Runner(), root, f"c_order_{planner}")

        assert log[:6] == [
            "start r1",
            "end r1",
            "start b1",
            "end b1",
            "start b2",
            "end b2",
        ]
        # What comes after them still runs together.
        assert log[6:8] == ["start r2", "start r3"]

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_they_are_not_asked_about(self, planner: str) -> None:
        log: list[str] = []
        root = Agent(
            name="MainAgent",
            description="the agent",
            instruction="work",
            llm_provider=_Script(label="main", replies=[[("browse", '{"what": "b"}')], "Done."]),
            planner=_planner(planner),
            tools=[_slow(log, "browse", False)],
        )

        events = await _run(Runner(), root, f"c_unasked_{planner}")

        assert log == ["start b", "end b"]
        assert not [e for e in events if e.event_type == ExecutionEventType.TOOL_FEEDBACK]
        assert not any(e.needs_feedback for e in events if hasattr(e, "needs_feedback"))

    def test_the_browser_tools_are_not(self, tmp_path: Any) -> None:
        from domyn.agents.tools.builtin import bundle

        built = bundle("browser", "files", root=tmp_path, backend="local")
        unsafe = sorted(t.name for t in built if not t.parallel_safe)
        assert unsafe == sorted(t.name for t in built if t.name.startswith("browser_"))
        assert len(unsafe) == 8

    def test_the_reserved_argument_is_in_no_schema(self, tmp_path: Any) -> None:
        from domyn.agents.tools.builtin import bundle

        for tool in bundle("browser", root=tmp_path, backend="local"):
            assert "agent_call_id" not in str(tool.declaration)
            assert "conversation_id" not in str(tool.declaration)


def test_ask_is_a_user_input() -> None:
    """`_ask` comes from the hand-off tests; this module relies on its shape."""
    assert _ask().event_type == ExecutionEventType.USER_INPUT
