"""``/compact``: compacting because someone asked, not because a limit was hit.

The request travels to the runner as an event rather than being carried out
inside the CLI, so the manual path exists for every front-end instead of only
this one. What lands is a **record**, not a rewritten history: the events are
untouched and the dropping is applied when the next request is rendered. That is
also why it survives — a compaction of a transient message list would be undone
by the very next re-render.
"""

# No `from __future__ import annotations`: builds real agents and tools.
import io
from typing import Any

import pytest
from rich.console import Console

from domyn.agents import Agent, Runner
from domyn.agents.core import BaseEvent, ExecutionEventType, Part, ToolAction
from domyn.agents.interfaces.cli import Theme, commands
from domyn.agents.interfaces.cli.repl import APPLICATION_NAME, USER_ID, Repl
from domyn.agents.llm import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    ToolCallFunction,
)
from domyn.agents.planner_strategy import ToolUsePlannerStrategy
from domyn.agents.planner_strategy.context_management.compaction import read_record


class _ToolCallingProvider(BaseLLMProvider):
    """Calls a tool `rounds` times, then answers.

    A real history is built rather than planted: the runner reads its own
    persisted conversation, so events injected into a state object it did not
    load are simply not there.
    """

    type: str = "tool-calling"
    model_name: str = "tool-calling"
    supported_structured_outputs: list[Any] | None = None
    rounds: int = 4
    calls: int = 0

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        self.calls += 1
        if self.calls <= self.rounds:
            return LLMResponse(
                content="",
                generated_tokens=1,
                execution_time=0.0,
                usage={},
                tool_calls=[
                    LLMToolCall(
                        id=f"call_{self.calls}",
                        type="function",
                        function=ToolCallFunction(
                            name="read_file", arguments='{"path": "big.txt"}'
                        ),
                    )
                ],
            )
        return LLMResponse(content="Done.", generated_tokens=1, execution_time=0.0, usage={})

    async def generate_streaming(self, messages: list[LLMMessage], **kwargs: Any) -> Any:
        raise NotImplementedError
        yield  # pragma: no cover


class _CountingProvider(BaseLLMProvider):
    """Answers, and counts how many times it was asked."""

    type: str = "counting"
    model_name: str = "counting"
    supported_structured_outputs: list[Any] | None = None
    calls: int = 0

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content="Done.", generated_tokens=1, execution_time=0.0, usage={})

    async def generate_streaming(self, messages: list[LLMMessage], **kwargs: Any) -> Any:
        raise NotImplementedError
        yield  # pragma: no cover


def _repl() -> tuple[Repl, io.StringIO, _CountingProvider]:
    provider = _CountingProvider()
    agent = Agent(
        name="Assistant",
        description="test agent",
        instruction="Be brief.",
        llm_provider=provider,
        planner=ToolUsePlannerStrategy(),
    )
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False)
    repl = Repl(
        agent,
        console,
        Theme.build(allow_animation=False),
        runner=Runner(),
        enable_enhanced_input=False,
    )
    return repl, buffer, provider


class TestTheCommandExists:
    def test_it_resolves(self) -> None:
        resolved = commands.resolve("/compact")
        assert resolved is not None and resolved[0].name == "/compact"

    def test_it_is_offered_by_the_autocompleter(self) -> None:
        """A slash command nobody can discover may as well not exist."""
        assert "/compact" in commands.completion_names()

    def test_it_appears_in_help(self) -> None:
        assert any(row[0] == "/compact" for row in commands.help_lines())


class TestAskingForIt:
    @pytest.mark.asyncio
    async def test_it_says_something_even_with_nothing_to_compact(self) -> None:
        """A command that answers nothing is indistinguishable from a broken one,
        and this is the case a user will hit first."""
        repl, buffer, _ = _repl()
        await repl._compact_now()
        assert "compact" in buffer.getvalue().lower()

    @pytest.mark.asyncio
    async def test_no_model_is_called(self) -> None:
        """`prune_only` is deterministic. If asking to compact costs a request,
        the cheapest strategy has stopped being cheap."""
        repl, _, provider = _repl()
        await repl._compact_now()
        assert provider.calls == 0

    @pytest.mark.asyncio
    async def test_it_does_not_count_as_a_turn(self) -> None:
        repl, _, _ = _repl()
        before = repl._state.turns
        await repl._compact_now()
        assert repl._state.turns == before

    @pytest.mark.asyncio
    async def test_the_request_never_enters_the_conversation(self) -> None:
        """Asking to compact is not something the agent should later read as
        having been said to it."""
        repl, _, provider = _repl()
        await repl._compact_now()
        await repl._run_turn("hello")

        assert provider.calls == 1
        # If the request had been stored as a user turn it would be rendered into
        # the next request, so the agent would see it.
        assert "compact the conversation" not in repl._console.file.getvalue()


class TestThroughTheRunner:
    @pytest.mark.asyncio
    async def test_a_history_with_large_tool_results_gets_recorded(self) -> None:
        """The event → runner → record path, over a history worth compacting."""
        from domyn.agents.tools import FunctionTool

        def read_file(path: str) -> str:
            """Read a file.

            Args:
                path: What to read.
            """
            return "exit code: 0\n" + "a line of the file\n" * 200

        provider = _ToolCallingProvider()
        agent = Agent(
            name="Assistant",
            description="test agent",
            instruction="Use your tools.",
            llm_provider=provider,
            planner=ToolUsePlannerStrategy(),
            tools=[FunctionTool(func=read_file)],
        )
        buffer = io.StringIO()
        console = Console(file=buffer, width=100, force_terminal=False)
        repl = Repl(
            agent,
            console,
            Theme.build(allow_animation=False),
            runner=Runner(),
            enable_enhanced_input=False,
        )

        await repl._run_turn("read the file four times")
        assert provider.calls > 1, "the fixture produced no tool calls, so no history to compact"

        collected: list[BaseEvent] = []
        async for event in repl._runner.run(
            application_name=APPLICATION_NAME,
            user_id=USER_ID,
            root=agent,
            conversation_id=repl._state.conversation_id,
            user_input=BaseEvent(
                event_type=ExecutionEventType.COMPACTION_REQUESTED,
                author="user",
                content=[Part(text="compact the conversation")],
            ),
        ):
            collected.append(event)

        kinds = [event.event_type for event in collected]
        assert ExecutionEventType.CONTEXT_COMPACTION_START in kinds, (
            "the start was not announced, so a summarizing strategy would look like a hang"
        )
        done = next(e for e in collected if e.event_type is ExecutionEventType.CONTEXT_COMPACTED)
        assert done.metadata["requested"] is True
        assert done.metadata["changed"] is True, str(done.content[0].text)
        assert done.metadata["events_dropped"] >= 1

    @pytest.mark.asyncio
    async def test_the_record_survives_into_the_next_request(self) -> None:
        """The property the whole design exists for. Everything else can be
        right and this still be wrong, and then the command is a placebo."""
        from domyn.agents.tools import FunctionTool

        def read_file(path: str) -> str:
            """Read a file.

            Args:
                path: What to read.
            """
            return "exit code: 0\n" + "a line of the file\n" * 200

        provider = _ToolCallingProvider()
        agent = Agent(
            name="Assistant",
            description="test agent",
            instruction="Use your tools.",
            llm_provider=provider,
            planner=ToolUsePlannerStrategy(),
            tools=[FunctionTool(func=read_file)],
        )
        buffer = io.StringIO()
        repl = Repl(
            agent,
            Console(file=buffer, width=100, force_terminal=False),
            Theme.build(allow_animation=False),
            runner=Runner(),
            enable_enhanced_input=False,
        )

        await repl._run_turn("read the file four times")
        await repl._compact_now()

        seen: list[list[LLMMessage]] = []
        original = agent.llm_provider.generate

        async def _capture(messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
            seen.append(list(messages))
            return await original(messages, **kwargs)

        object.__setattr__(agent.llm_provider, "generate", _capture)
        provider.rounds = 0  # answer straight away on the next turn
        await repl._run_turn("and now summarise")

        assert seen, "no request was captured"
        dropped = sum(
            1
            for message in seen[0]
            if message.role == "tool" and "dropped to save context" in str(message.content)
        )
        assert dropped >= 1, "the recorded compaction did not survive the re-render"

    @pytest.mark.asyncio
    async def test_it_says_what_it_saved_and_the_measurement_follows(self) -> None:
        """Reported as "/compact does nothing": it dropped the outputs, printed
        no number, and `/context` went on showing the context from before."""
        from domyn.agents.tools import FunctionTool

        def read_file(path: str) -> str:
            """Read a file.

            Args:
                path: What to read.
            """
            return "exit code: 0\n" + "a line of the file\n" * 200

        agent = Agent(
            name="Assistant",
            description="test agent",
            instruction="Use your tools.",
            llm_provider=_ToolCallingProvider(),
            planner=ToolUsePlannerStrategy(),
            tools=[FunctionTool(func=read_file)],
        )
        buffer = io.StringIO()
        repl = Repl(
            agent,
            Console(file=buffer, width=100, force_terminal=False),
            Theme.build(allow_animation=False),
            runner=Runner(),
            enable_enhanced_input=False,
        )

        await repl._run_turn("read the file four times")
        before = repl._state.last_usage
        await repl._compact_now()
        after = repl._state.last_usage

        assert "old tool outputs dropped (-" in buffer.getvalue(), "no saving was shown"
        assert before is not None and after is not None
        assert after.total_tokens < before.total_tokens, "/context still shows the old size"

    @pytest.mark.asyncio
    async def test_no_agent_is_started(self) -> None:
        """Nothing below the interception point applies: no agent runs."""
        repl, _, provider = _repl()
        collected = [
            event
            async for event in repl._runner.run(
                application_name=APPLICATION_NAME,
                user_id=USER_ID,
                root=repl._agent,
                conversation_id="conv-compact",
                user_input=BaseEvent(
                    event_type=ExecutionEventType.COMPACTION_REQUESTED,
                    author="user",
                    content=[Part(text="compact")],
                ),
            )
        ]
        assert provider.calls == 0
        assert ExecutionEventType.AGENT_START not in [e.event_type for e in collected]
        assert ExecutionEventType.RESPONSE not in [e.event_type for e in collected]

    @pytest.mark.asyncio
    async def test_a_planner_without_a_compactor_says_so(self) -> None:
        """Not every planner has one — the react branch does not. A silent no-op
        would be indistinguishable from a compaction that worked."""
        planner = ToolUsePlannerStrategy()
        # Stands in for the planners that have no compactor — the react branch
        # is one. Reached through the instance dict because the field is typed,
        # and building a react planner takes a full nested configuration that
        # would say nothing about what is under test here.
        planner.__dict__["compactor"] = None

        agent = Agent(
            name="Assistant",
            description="test agent",
            instruction="Be brief.",
            llm_provider=_CountingProvider(),
            planner=planner,
        )
        assert getattr(agent.planner, "compactor", None) is None, (
            "the fixture no longer proves anything"
        )

        collected = [
            event
            async for event in Runner().run(
                application_name=APPLICATION_NAME,
                user_id=USER_ID,
                root=agent,
                conversation_id="conv-no-compactor",
                user_input=BaseEvent(
                    event_type=ExecutionEventType.COMPACTION_REQUESTED,
                    author="user",
                    content=[Part(text="compact")],
                ),
            )
        ]
        done = next(e for e in collected if e.event_type is ExecutionEventType.CONTEXT_COMPACTED)
        assert done.metadata["changed"] is False
        assert "does not support compaction" in str(done.content[0].text)
