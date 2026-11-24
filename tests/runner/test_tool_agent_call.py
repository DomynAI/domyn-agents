"""What the runner tells a tool about the agent call it runs in, and when.

- ``agent_call_id`` (reserved, never in a schema): empty for the agent the
  conversation is addressed to, the call's interaction id in a sub-agent. The
  same through a pause for approval, different for every call, copies of one
  agent included. What a tool keeps per agent call (a browser's page) is keyed
  by it.
- The built-in tools are told when a call ends (``BuiltinTools.end_agent_call``),
  on every AGENT_END, and not while the call only waits for an answer.
- Calls of one response to tools sharing a ``concurrency_key`` run one after the
  other, in the order made, without being asked about; the rest of the response
  runs alongside them, sub-agents included.
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
    APP,
    PLANNERS,
    USER,
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


def _delegating(planner: str, subs: list[Agent], replies: list[Any]) -> Agent:
    """The agent the conversation is addressed to: it calls ``subs`` all at once,
    then answers with ``replies``."""
    return Agent(
        name="MainAgent",
        description="the agent",
        instruction="delegate",
        llm_provider=_Script(
            label="main",
            replies=[[(sub.name, f'{{"task": "{sub.name}"}}') for sub in subs], *replies],
        ),
        planner=_planner(planner, parallel_handoffs=True),
        sub_agents=subs,
    )


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

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_an_agent_handed_off_to_inside_a_failed_branch_is_ended(
        self, planner: str
    ) -> None:
        # Alpha, in a branch, hands off to Inner, whose model fails. The branch ends
        # as failed; Inner, on that branch's stack, used to get no AGENT_END, and
        # its browser stayed open until the idle timeout.
        class _Down(_Script):
            async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
                raise RuntimeError("the model is down")

        seen: list[tuple[str, str]] = []
        ended: list[str] = []
        inner = Agent(
            name="Inner",
            description="does the inner part",
            instruction="do the task",
            llm_provider=_Down(label="Inner"),
            planner=_planner(planner),
            builtin_tools=[_Ends(ended)],
        )
        alpha = _sub(
            "Alpha",
            planner,
            seen,
            [[("Inner", '{"task": "i"}')], "Alpha done"],
            sub_agents=[inner],
            builtin_tools=[_Ends(ended)],
        )
        beta = _sub("Beta", planner, seen, ["Beta done"], builtin_tools=[_Ends(ended)])
        root = _delegating(planner, [alpha, beta], replies=["Done."])

        events = await _run(Runner(), root, f"c_nested_fail_{planner}")

        inner_call = _starts(events)["Inner"][0]
        assert inner_call in ended
        inner_end = [
            e
            for e in events
            if e.event_type == ExecutionEventType.AGENT_END and e.interaction_id == inner_call
        ]
        assert [e.error_code for e in inner_end] == ["SubAgentFailed"]
        # Top first: Inner's end before the end of the branch it was in.
        authors = [
            e.author
            for e in events
            if e.event_type == ExecutionEventType.AGENT_END and e.author in ("Inner", "Alpha")
        ]
        assert authors == ["Inner", "Alpha"]

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_an_agent_handed_off_to_inside_an_interrupted_branch_is_ended(
        self, planner: str
    ) -> None:
        seen: list[tuple[str, str]] = []
        ended: list[str] = []
        inner = Agent(
            name="Inner",
            description="does the inner part",
            instruction="do the task",
            llm_provider=_Script(label="Inner", replies=["Inner done"], delay=5.0),
            planner=_planner(planner),
            builtin_tools=[_Ends(ended)],
        )
        alpha = _sub(
            "Alpha",
            planner,
            seen,
            [[("Inner", '{"task": "i"}')], "Alpha done"],
            sub_agents=[inner],
            builtin_tools=[_Ends(ended)],
        )
        beta = Agent(
            name="Beta",
            description="looks at beta",
            instruction="do the task",
            llm_provider=_Script(label="Beta", replies=["Beta done"], delay=5.0),
            planner=_planner(planner),
            builtin_tools=[_Ends(ended)],
        )
        root = _delegating(planner, [alpha, beta], replies=["Done."])
        runner = Runner()
        conversation = f"c_nested_stop_{planner}"

        run = runner.run(
            application_name=APP,
            user_id=USER,
            conversation_id=conversation,
            root=root,
            user_input=_ask(),
        )
        inner_call = ""
        async for event in run:
            if event.event_type == ExecutionEventType.AGENT_START and event.author == "Inner":
                inner_call = str(event.interaction_id)
                break
        await run.aclose()

        assert inner_call
        assert inner_call in ended
        info = await runner.conversation_manager.get_conversation(APP, USER, conversation)
        assert info is not None
        interrupted = {
            e.author
            for e in info.events_history
            if e.event_type == ExecutionEventType.AGENT_END
            and e.error_code == "SubAgentInterrupted"
        }
        assert interrupted == {"Inner", "Alpha", "Beta"}

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


def _slow(log: list[str], name: str, key: str | None, delay: float = 0.05) -> FunctionTool:
    async def step(what: str) -> str:
        """Do a slow step.

        Args:
            what: the step.
        """
        log.append(f"start {what}")
        await asyncio.sleep(delay)
        log.append(f"end {what}")
        return f"did {what}"

    step.__name__ = name
    return FunctionTool(func=step, concurrency_key=key)


def _calls(*steps: tuple[str, str]) -> list[tuple[str, str]]:
    return [(name, f'{{"what": "{what}"}}') for name, what in steps]


def _working(planner: str, calls: list[tuple[str, str]], tools: list[FunctionTool]) -> Agent:
    return Agent(
        name="MainAgent",
        description="the agent",
        instruction="work",
        llm_provider=_Script(label="main", replies=[calls, "Done."]),
        planner=_planner(planner),
        tools=tools,
    )


def _before(log: list[str], first: str, second: str) -> bool:
    return log.index(first) < log.index(second)


class TestAConcurrencyKey:
    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_its_calls_run_in_order_and_the_rest_alongside(self, planner: str) -> None:
        log: list[str] = []
        calls = _calls(("read", "r1"), ("browse", "b1"), ("browse", "b2"), ("read", "r2"))
        root = _working(planner, calls, [_slow(log, "read", None), _slow(log, "browse", "page")])

        await _run(Runner(), root, f"c_key_{planner}")

        # b2 waits for b1, and only for b1: the reads start with b1.
        assert _before(log, "end b1", "start b2")
        assert log[:3] == ["start r1", "start b1", "start r2"]

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_two_keys_do_not_wait_for_each_other(self, planner: str) -> None:
        log: list[str] = []
        calls = _calls(("browse", "b1"), ("query", "q1"), ("browse", "b2"), ("query", "q2"))
        root = _working(planner, calls, [_slow(log, "browse", "page"), _slow(log, "query", "db")])

        await _run(Runner(), root, f"c_keys_{planner}")

        assert log[:2] == ["start b1", "start q1"]
        assert _before(log, "end b1", "start b2") and _before(log, "end q1", "start q2")

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_a_sub_agent_in_the_same_response_is_not_waited_for(self, planner: str) -> None:
        """The caller's page is not a sub-agent's: its browser call runs while the
        sub-agent works, as before sub-agents had browsers of their own."""
        log: list[str] = []
        alpha = Agent(
            name="Alpha",
            description="works slowly",
            instruction="do the task",
            llm_provider=_Script(label="Alpha", log=log, delay=0.3, replies=["alpha done"]),
            planner=_planner(planner),
        )
        root = Agent(
            name="MainAgent",
            description="the agent",
            instruction="delegate and browse",
            llm_provider=_Script(
                label="main",
                replies=[[("Alpha", '{"task": "a"}'), ("browse", '{"what": "snap"}')], "Done."],
            ),
            planner=_planner(planner, parallel_handoffs=True),
            sub_agents=[alpha],
            tools=[_slow(log, "browse", "page")],
        )

        await _run(Runner(), root, f"c_alongside_{planner}")

        assert _before(log, "end snap", "end Alpha"), log

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_a_call_waiting_its_turn_holds_no_place(self, planner: str) -> None:
        """Two places: b1 and r1 take them while b2 waits for b1. A call in line
        holding a place would leave r1 waiting for b1 instead."""
        log: list[str] = []
        calls = _calls(("browse", "b1"), ("browse", "b2"), ("read", "r1"))
        root = _working(planner, calls, [_slow(log, "read", None), _slow(log, "browse", "page")])

        await _run(Runner(max_parallel_tool_calls=2), root, f"c_place_{planner}")

        assert _before(log, "start r1", "end b1"), log
        assert _before(log, "end b1", "start b2"), log

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_its_calls_are_not_asked_about(self, planner: str) -> None:
        log: list[str] = []
        root = _working(planner, _calls(("browse", "b")), [_slow(log, "browse", "page")])

        events = await _run(Runner(), root, f"c_unasked_{planner}")

        assert log == ["start b", "end b"]
        assert not [e for e in events if e.event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK]

    def test_the_browser_tools_share_one(self, tmp_path: Any) -> None:
        from domyn.agents.tools.builtin import bundle

        built = bundle("browser", "files", root=tmp_path, backend="local")
        keys = {t.name: t.concurrency_key for t in built}
        assert {name for name, key in keys.items() if key == "browser"} == {
            name for name in keys if name.startswith("browser_")
        }
        assert len([k for k in keys.values() if k == "browser"]) == 8
        assert {k for name, k in keys.items() if not name.startswith("browser_")} == {None}

    def test_the_reserved_argument_is_in_no_schema(self, tmp_path: Any) -> None:
        from domyn.agents.tools.builtin import bundle

        for tool in bundle("browser", root=tmp_path, backend="local"):
            assert "agent_call_id" not in str(tool.declaration)
            assert "conversation_id" not in str(tool.declaration)
