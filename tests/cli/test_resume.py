"""Getting back into a conversation that a previous session left behind.

The session used to keep its history in memory, so the id it printed on the way
out named nothing that still existed. These are the two halves of making it an
address: it is written somewhere, and there is a way back to it.
"""

import io
from datetime import datetime, timedelta
from typing import Any

import pytest
from rich.console import Console

from domyn.agents import Agent, BaseEvent, ExecutionEventType, Part
from domyn.agents.conversation_manager import ConversationInfo
from domyn.agents.interfaces.cli import Theme, conversations
from domyn.agents.interfaces.cli.repl import APPLICATION_NAME, USER_ID, Repl
from domyn.agents.llm import BaseLLMProvider, LLMMessage, LLMResponse


class _Scripted(BaseLLMProvider):
    type: str = "scripted"
    model_name: str = "scripted"
    supported_structured_outputs: list[Any] | None = None

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        return LLMResponse(content="an answer", generated_tokens=0, execution_time=0.0)

    async def generate_streaming(self, messages: list[LLMMessage], **kwargs: Any):
        yield LLMResponse(content="an answer", generated_tokens=0, execution_time=0.0)


class _Editor:
    def __init__(self, *lines: str) -> None:
        self._lines = list(lines)

    async def prompt_async(self, *args: Any, **kwargs: Any) -> str:
        if not self._lines:
            raise EOFError
        return self._lines.pop(0)


def _repl(*script: str) -> tuple[Repl, io.StringIO]:
    agent = Agent(
        name="Assistant",
        description="a test agent",
        instruction="Be brief.",
        llm_provider=_Scripted(),
    )
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False)
    repl = Repl(agent, console, Theme.build(allow_animation=False), enable_enhanced_input=False)
    repl._input._session = _Editor(*script)
    return repl, buffer


def _said(text: str) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text=text)]
    )


def _info(conversation_id: str, *texts: str, age: timedelta = timedelta()) -> ConversationInfo:
    return ConversationInfo(
        application_name=APPLICATION_NAME,
        user_id=USER_ID,
        conversation_id=conversation_id,
        events_history=[_said(t) for t in texts],
        updated_at=datetime.now() - age,
    )


class TestTheIdIsSaidOnTheWayOut:
    """It is the only way back in, and nothing else prints it."""

    @pytest.mark.asyncio
    async def test_a_session_that_did_something_names_itself(self) -> None:
        repl, buffer = _repl("hello")

        await repl.run()

        assert repl._state.conversation_id in buffer.getvalue()
        assert "--resume" in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_a_session_where_nothing_was_said_does_not(self) -> None:
        """An id for an empty conversation invites the user to resume nothing,
        which teaches them the feature does not work."""
        repl, buffer = _repl()

        await repl.run()

        assert repl._state.conversation_id not in buffer.getvalue()


class TestTheHistorySurvivesTheProcess:
    @pytest.mark.asyncio
    async def test_a_turn_is_still_there_in_a_new_session(self) -> None:
        """The whole feature in one test: what the first session stored is what
        the second one is handed."""
        first, _ = _repl("what is in this directory")
        await first.run()

        second, _ = _repl()
        found = await conversations.find(
            second._runner.conversation_manager,
            APPLICATION_NAME,
            USER_ID,
            first._state.conversation_id,
        )

        assert found is not None
        assert any(
            "what is in this directory" in (part.text or "")
            for event in found.events_history
            for part in event.content or []
        )

    @pytest.mark.asyncio
    async def test_a_conversation_nobody_started_is_not_invented(self) -> None:
        """Asking the store for an unknown id would create it, and a resume that
        silently starts an empty conversation is worse than a refusal."""
        repl, _ = _repl()

        found = await conversations.find(
            repl._runner.conversation_manager, APPLICATION_NAME, USER_ID, "conv_nothing"
        )

        assert found is None


class TestResumingByIdFromTheSession:
    @pytest.mark.asyncio
    async def test_the_session_moves_to_the_other_conversation(self) -> None:
        first, _ = _repl("the first thing")
        await first.run()
        second, buffer = _repl()

        await second._handle_slash(f"/resume {first._state.conversation_id}")

        assert second._state.conversation_id == first._state.conversation_id
        assert "the first thing" in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_an_unknown_id_changes_nothing(self) -> None:
        repl, buffer = _repl()
        before = repl._state.conversation_id

        await repl._handle_slash("/resume conv_nothing")

        assert repl._state.conversation_id == before
        assert "No conversation" in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_the_turn_count_comes_from_the_conversation(self) -> None:
        """It is what decides whether the id is printed on exit, and a resumed
        session that reported zero turns would leave without naming itself."""
        first, _ = _repl("one", "two")
        await first.run()
        second, _ = _repl()

        await second._handle_slash(f"/resume {first._state.conversation_id}")

        assert second._state.turns == 2


class TestTheScreenIsClearedOnlyForARealResume:
    """The screen holds the conversation in progress: wiping it is only worth it
    when another one is about to be drawn in its place."""

    @staticmethod
    def _counting_clears(repl: Repl, monkeypatch: pytest.MonkeyPatch) -> list[None]:
        cleared: list[None] = []
        monkeypatch.setattr(repl._console, "clear", lambda *a, **kw: cleared.append(None))
        return cleared

    @pytest.mark.asyncio
    async def test_a_conversation_found_is_drawn_on_a_clean_screen(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first, _ = _repl("the first thing")
        await first.run()
        second, _ = _repl()
        cleared = self._counting_clears(second, monkeypatch)

        await second._handle_slash(f"/resume {first._state.conversation_id}")

        assert len(cleared) == 1

    @pytest.mark.asyncio
    async def test_an_unknown_id_leaves_the_screen_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repl, _ = _repl()
        cleared = self._counting_clears(repl, monkeypatch)

        await repl._handle_slash("/resume conv_nothing")

        assert cleared == []

    @pytest.mark.asyncio
    async def test_a_cancelled_choice_leaves_the_screen_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repl, _ = _repl()
        cleared = self._counting_clears(repl, monkeypatch)
        monkeypatch.setattr(conversations, "pick", lambda *a, **kw: None)

        await repl._handle_slash("/resume")

        assert cleared == []


def _event(kind: ExecutionEventType, text: str = "", turn: str = "t1", **kw: Any) -> BaseEvent:
    return BaseEvent(
        event_type=kind,
        author="user" if kind is ExecutionEventType.USER_INPUT else "Assistant",
        content=[Part(text=text)] if text else [],
        turn_id=turn,
        **kw,
    )


def _replayed(*events: BaseEvent, verbose: bool = False) -> str:
    buffer = io.StringIO()
    info = ConversationInfo(
        application_name=APPLICATION_NAME,
        user_id=USER_ID,
        conversation_id="conv_x",
        events_history=list(events),
        updated_at=datetime.now(),
    )
    conversations.render_history(
        Console(file=buffer, width=100), Theme.build(allow_animation=False), info, verbose=verbose
    )
    return buffer.getvalue()


class TestTheConversationComesBackOnScreen:
    """A resume used to show the last three exchanges, one dimmed line each and
    cut at the edge, which read as a resume that had lost the conversation."""

    @staticmethod
    def _conversation(turns: int) -> list[BaseEvent]:
        events: list[BaseEvent] = []
        for n in range(turns):
            events.append(_event(ExecutionEventType.USER_INPUT, f"question {n:03d}", turn=f"t{n}"))
            events.append(_event(ExecutionEventType.RESPONSE, f"answer {n:03d}", turn=f"t{n}"))
        return events

    def test_a_short_conversation_is_shown_whole(self) -> None:
        shown = _replayed(*self._conversation(conversations.HISTORY_TURNS))

        for n in range(conversations.HISTORY_TURNS):
            assert f"question {n:03d}" in shown
            assert f"answer {n:03d}" in shown
        assert shown.index("question 000") < shown.index("answer 000") < shown.index("question 019")
        assert "not shown" not in shown

    def test_a_long_one_shows_its_last_turns_and_counts_the_rest(self) -> None:
        """Older turns would only scroll out of the terminal's scrollback."""
        total = conversations.HISTORY_TURNS + 25

        shown = _replayed(*self._conversation(total))

        assert "25 earlier turns not shown" in shown
        assert "question 024" not in shown
        assert "answer 024" not in shown
        for n in range(25, total):
            assert f"question {n:03d}" in shown
            assert f"answer {n:03d}" in shown
        assert shown.index("not shown") < shown.index("question 025")

    def test_a_steered_message_does_not_count_as_a_turn(self) -> None:
        events = self._conversation(conversations.HISTORY_TURNS)
        events.insert(1, _event(ExecutionEventType.USER_INPUT, "and also this", turn="t0"))

        shown = _replayed(*events)

        assert "not shown" not in shown
        assert "question 000" in shown
        assert "and also this" in shown

    def test_a_long_answer_is_shown_whole(self) -> None:
        answer = "\n".join(f"line {n} of a long answer" for n in range(40))

        shown = _replayed(
            _event(ExecutionEventType.USER_INPUT, "explain it"),
            _event(ExecutionEventType.RESPONSE, answer),
        )

        assert "line 0 of a long answer" in shown
        assert "line 39 of a long answer" in shown
        assert "…" not in shown

    def test_the_answer_is_drawn_as_an_answer(self) -> None:
        theme = Theme.build(allow_animation=False)

        shown = _replayed(
            _event(ExecutionEventType.USER_INPUT, "hi"),
            _event(ExecutionEventType.RESPONSE, "hello"),
        )

        assert f"you {theme.symbols.prompt} hi" in shown
        assert f"{theme.symbols.agent} hello" in shown

    def test_tool_calls_are_part_of_it(self) -> None:
        from domyn.agents.core import ToolAction

        call = ToolAction(name="list_dir", parameters={"path": "."}, observation="a.py b.py")

        shown = _replayed(
            _event(ExecutionEventType.USER_INPUT, "what is here"),
            _event(ExecutionEventType.TOOL_START, action=call),
            _event(ExecutionEventType.TOOL_END, action=call),
            _event(ExecutionEventType.RESPONSE, "two files"),
        )

        assert "list_dir" in shown
        assert "a.py b.py" in shown

    def test_an_approval_long_answered_is_not_asked_again(self) -> None:
        """The store keeps the request and the reply; the session drew them as its
        own prompt, and a panel waiting for an answer would be a lie now."""
        from domyn.agents.core import ToolAction

        call = ToolAction(name="write_file", parameters={"path": "x"})

        shown = _replayed(
            _event(ExecutionEventType.USER_INPUT, "write it"),
            _event(ExecutionEventType.TOOL_START, action=call),
            _event(ExecutionEventType.TOOL_NEEDS_FEEDBACK, "approve write_file?", action=call),
            _event(ExecutionEventType.TOOL_FEEDBACK, "approved", action=call),
            _event(ExecutionEventType.RESPONSE, "written"),
            verbose=True,
        )

        assert "tool_needs_feedback" not in shown
        assert "tool_feedback" not in shown
        assert "written" in shown

    def test_a_sub_agents_answer_is_not_shown_as_one(self) -> None:
        """It reached the screen as the return to its caller, not as an answer."""
        from domyn.agents.core import METADATA_CONVERTED_TO_AGENT_END

        shown = _replayed(
            _event(ExecutionEventType.USER_INPUT, "ask the helper"),
            _event(
                ExecutionEventType.RESPONSE,
                "the helper's own words",
                metadata={METADATA_CONVERTED_TO_AGENT_END: True},
            ),
            _event(ExecutionEventType.RESPONSE, "what the root said"),
        )

        assert "the helper's own words" not in shown
        assert "what the root said" in shown

    def test_a_steered_message_stays_inside_its_turn(self) -> None:
        theme = Theme.build(allow_animation=False)

        shown = _replayed(
            _event(ExecutionEventType.USER_INPUT, "start", turn="t1"),
            _event(ExecutionEventType.USER_INPUT, "and also this", turn="t1"),
            _event(ExecutionEventType.RESPONSE, "done both", turn="t1"),
        )

        assert f"you {theme.symbols.prompt} and also this" in shown
        assert shown.index("start") < shown.index("and also this") < shown.index("done both")

    @pytest.mark.asyncio
    async def test_a_streamed_answer_comes_back_once(self) -> None:
        """Streamed live, the final RESPONSE was not yielded but is what is stored."""
        first, _ = _repl("a question")
        first._agent.stream = True
        await first.run()
        second, buffer = _repl()

        await second._handle_slash(f"/resume {first._state.conversation_id}")

        assert buffer.getvalue().count("an answer") == 1
        assert "a question" in buffer.getvalue()


class TestChoosingOneInstead:
    def test_the_newest_is_offered_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        offered: list[list[str]] = []
        found = [
            _info("conv_old", "an old question", age=timedelta(days=2)),
            _info("conv_new", "a recent question", age=timedelta(minutes=5)),
        ]

        import domyn.agents.interfaces.cli.select as select_module

        monkeypatch.setattr(
            select_module,
            "select",
            lambda console, theme, label, options, **kw: offered.append(options) or options[0],
        )
        console = Console(file=io.StringIO(), width=100)
        chosen = conversations.pick(
            console,
            Theme.build(allow_animation=False),
            sorted(found, key=lambda i: i.updated_at, reverse=True),
        )

        assert chosen is not None and chosen.conversation_id == "conv_new"
        assert "a recent question" in offered[0][0]
        assert "5 min ago" in offered[0][0]

    def test_nothing_to_choose_from_says_so(self) -> None:
        buffer = io.StringIO()

        chosen = conversations.pick(
            Console(file=buffer, width=100), Theme.build(allow_animation=False), []
        )

        assert chosen is None
        assert "No past conversations" in buffer.getvalue()

    def test_two_conversations_that_look_identical_stay_distinct(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Opened a minute apart with the same first message, the rows are the
        same string — and the picker answers with the string."""
        import domyn.agents.interfaces.cli.select as select_module

        monkeypatch.setattr(
            select_module, "select", lambda console, theme, label, options, **kw: options[1]
        )
        found = [_info("conv_a", "same words"), _info("conv_b", "same words")]

        chosen = conversations.pick(
            Console(file=io.StringIO(), width=100), Theme.build(allow_animation=False), found
        )

        assert chosen is not None and chosen.conversation_id == "conv_b"


class TestWhatARowSays:
    def test_it_carries_the_first_thing_that_was_asked(self) -> None:
        row = conversations.label(_info("conv_x", "how do I build this", "and then?"))

        assert "how do I build this" in row
        assert "2 messages" in row

    def test_a_long_first_message_is_cut(self) -> None:
        row = conversations.label(_info("conv_x", "word " * 100))

        assert "…" in row
        assert len(row) < 140

    def test_a_conversation_with_nothing_in_it_still_has_a_row(self) -> None:
        """A crash between creating the conversation and the first message
        leaves one, and it must not take the listing down with it."""
        row = conversations.label(_info("conv_empty"))

        assert "0 messages" in row


class TestTheCommandLineForm:
    """``--resume`` takes its id as the next word rather than as an option value.

    Typer computes ``is_flag`` itself and mis-parses an option declared to take
    an optional one: declared that way, ``--resume --no-banner`` resolved the id
    to the string ``--no-banner`` and started a session against a conversation
    that could not exist.
    """

    def _invoke(self, monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> dict[str, Any]:
        from typer.testing import CliRunner

        from domyn.agents.cli.cli import app
        from domyn.agents.cli.commands.run import command as run_module

        seen: dict[str, Any] = {}

        async def fake_start_session(agent: Any, theme: Any, **kwargs: Any) -> int:
            seen.update(kwargs)
            return 0

        monkeypatch.setattr(run_module, "start_session", fake_start_session)

        async def fake_resolve(path: Any, theme: Any, **kwargs: Any) -> tuple[Any, dict]:
            return object(), {}

        monkeypatch.setattr(run_module, "resolve_agent_from_file", fake_resolve)
        CliRunner().invoke(app, argv)
        return seen

    def test_an_id_after_the_flag_is_the_conversation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = self._invoke(monkeypatch, ["run", "--no-banner", "--resume", "conv_abc123"])

        assert seen["resume"] == "conv_abc123"
        assert seen["resuming"] is True
        assert seen["initial_prompt"] is None, "the id must not also be sent as a message"

    def test_the_flag_alone_asks_for_a_choice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = self._invoke(monkeypatch, ["run", "--no-banner", "--resume"])

        assert seen["resuming"] is True
        assert seen["resume"] is None

    def test_another_option_after_it_is_not_eaten(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The regression that decided the shape of this option."""
        seen = self._invoke(monkeypatch, ["run", "--resume", "--no-banner"])

        assert seen["resume"] is None
        assert seen["show_status"] is False, "--no-banner was swallowed as the id"

    def test_without_the_flag_a_prompt_is_still_a_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = self._invoke(monkeypatch, ["run", "--no-banner", "what is this"])

        assert seen["initial_prompt"] == "what is this"
        assert seen["resuming"] is False


class TestStartingASessionOnIt:
    """``start_session`` is where the id becomes a conversation, and where the
    request has to fail if there is nothing to continue."""

    @pytest.mark.asyncio
    async def test_leaving_at_once_still_names_the_conversation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing new was said, but the conversation is not empty: the id is the
        way back to it. ``/resume`` counted its exchanges; ``--resume`` did not,
        so a session opened on it and closed straight away left without it."""
        import domyn.agents.cli.commands.run.command as run_module
        from domyn.agents.cli.commands.run.command import start_session

        first, _ = _repl("something earlier")
        await first.run()
        buffer = io.StringIO()
        monkeypatch.setattr(run_module, "console", Console(file=buffer, width=100))

        class _Quiet(Repl):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._input._session = _Editor()

        monkeypatch.setattr(run_module, "Repl", _Quiet)

        await start_session(
            _repl()[0]._agent,
            Theme.build(allow_animation=False),
            initial_prompt=None,
            verbose=False,
            show_status=False,
            resume=first._state.conversation_id,
            resuming=True,
        )

        assert f"--resume {first._state.conversation_id}" in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_resuming_from_the_command_line_draws_the_history(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import domyn.agents.cli.commands.run.command as run_module
        from domyn.agents.cli.commands.run.command import start_session

        first, _ = _repl("something earlier")
        await first.run()
        buffer = io.StringIO()
        monkeypatch.setattr(run_module, "console", Console(file=buffer, width=100))

        class _Quiet(Repl):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._input._session = _Editor()

        monkeypatch.setattr(run_module, "Repl", _Quiet)

        await start_session(
            _repl()[0]._agent,
            Theme.build(allow_animation=False),
            initial_prompt=None,
            verbose=False,
            show_status=False,
            resume=first._state.conversation_id,
            resuming=True,
        )

        shown = buffer.getvalue()
        assert "something earlier" in shown
        assert "an answer" in shown
        assert shown.index("an answer") < shown.index("resumed here")

    @pytest.mark.asyncio
    async def test_an_id_that_names_nothing_stops_the_launch(self) -> None:
        """Starting a fresh conversation instead would silently discard what was
        asked for, and look like a resume that lost everything."""
        from domyn.agents.cli.commands.run.command import start_session

        code = await start_session(
            _repl()[0]._agent,
            Theme.build(allow_animation=False),
            initial_prompt=None,
            verbose=False,
            show_status=False,
            resume="conv_nothing",
            resuming=True,
        )

        assert code == 1
