"""Typing while the agent is still working.

The REPL used to read a line, run a turn to completion, then read the next line,
strictly in that order. These drive the version that reads *alongside* the turn
and hands what it reads to ``Runner.steer``, which is the half of mid-turn
steering the CLI owns; the other half — the runner taking the message into the
next LLM call — is covered in ``tests/runner/test_steering.py``.

The editor is replaced at the ``InputSession`` level rather than at
prompt_toolkit's, so nothing here installs a stdout proxy or talks to a terminal.
What is deliberately *not* stubbed is the runner, the planner or the renderer: the
bugs this feature can cause all live in the seam between a coroutine reading the
terminal and a coroutine writing to it.
"""

# No `from __future__ import annotations`: builds real tools, and a stringified
# annotation is not a type the tool schema builder can read.
import asyncio
import io
from typing import Any

import pytest
from pydantic import Field
from rich.console import Console

from domyn.agents import Agent
from domyn.agents.core import ApprovalType, ExecutionEventType, Part
from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli.repl import Repl, _Queued, _SteeringReader
from domyn.agents.llm import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    ToolCallFunction,
)
from domyn.agents.planner_strategy import ToolUsePlannerStrategy
from domyn.agents.tools import FunctionTool

STEER = "actually, only the python files"


class _ToolThenAnswer(BaseLLMProvider):
    """Calls ``look`` once, then answers, recording what it was shown each time."""

    type: str = "tool-then-answer"
    model_name: str = "tool-then-answer"
    supported_structured_outputs: list[Any] | None = None
    rounds: int = 1
    calls: int = 0
    seen: list[list[str]] = Field(default_factory=list)
    # Run once, on the first call, while the turn is provably in flight. The only
    # way to observe the REPL mid-turn without a sleep.
    pause: Any = None

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        self.calls += 1
        if self.calls == 1 and self.pause is not None:
            await self.pause()
        self.seen.append([str(getattr(m, "content", "") or "") for m in messages])
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
                        function=ToolCallFunction(name="look", arguments='{"path": "a.txt"}'),
                    )
                ],
            )
        return LLMResponse(content="All done.", generated_tokens=1, execution_time=0.0, usage={})

    def saw(self, call: int, text: str) -> bool:
        return any(text in content for content in self.seen[call])

    async def generate_streaming(self, messages: list[LLMMessage], **kwargs: Any) -> Any:
        raise NotImplementedError
        yield  # pragma: no cover


class _ScriptedInput:
    """Stands in for ``InputSession``, handing out lines when a test releases them.

    Backed by a queue rather than a list so a read genuinely blocks: a stub that
    answers instantly would make every test pass whether or not the read is
    actually running concurrently with the turn.
    """

    def __init__(self, *lines: str) -> None:
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        for line in lines:
            self.queue.put_nowait(line)
        self.asked = 0
        self.defaults: list[str] = []
        self.concurrent: list[bool] = []
        self.pending: list[Any] = []
        self._draft = ""
        self.previews: list[Any] = []

    def rows(self) -> list[tuple[str, str]]:
        """The queue the editor would draw right now, from the newest read."""
        for callback in reversed(self.pending):
            if callback is not None:
                return list(callback())
        return []

    @property
    def enhanced(self) -> bool:
        return True

    @property
    def supports_concurrent_input(self) -> bool:
        return True

    def draft(self) -> str:
        return self._draft

    async def ask(
        self,
        label: str = "you",
        *,
        default: str = "",
        status: Any = None,
        concurrent: bool = False,
        pending: Any = None,
        preview: Any = None,
    ) -> str:
        self.asked += 1
        self.previews.append(preview)
        self.defaults.append(default)
        self.concurrent.append(concurrent)
        # The rows as they were *at the moment the read started*, which is what a
        # test can assert against. The real editor re-reads the callable on every
        # repaint; `rows()` below is how a test does the same.
        self.pending.append(pending)
        item = await self.queue.get()
        if isinstance(item, BaseException):
            raise item
        return str(item)


def _repl(
    input_lines: tuple[str, ...] = (),
    *,
    rounds: int = 1,
    steering: bool = True,
    gated: bool = False,
):
    """A Repl whose editor is scripted and whose reader is switched on explicitly.

    ``force_terminal=True`` because the reader is gated on the console being one:
    ``domyn run > log.txt`` has an editor but nowhere to repaint around.

    ``gated`` puts the tool behind ``USER_CONFIRM``, which is what every tool the
    CLI ships is — the ``files`` and ``shell`` bundles both — and what turns one
    turn into several ``run()`` calls.
    """
    provider = _ToolThenAnswer(rounds=rounds)

    async def look(path: str) -> str:
        """Look at a path.

        Args:
            path: what to look at.
        """
        return "looked"

    agent = Agent(
        name="Assistant",
        description="test agent",
        instruction="Be brief.",
        llm_provider=provider,
        planner=ToolUsePlannerStrategy(),
        tools=[
            FunctionTool(func=look, approval_type=ApprovalType.USER_CONFIRM)
            if gated
            else FunctionTool(func=look)
        ],
    )
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=True)
    repl = Repl(
        agent,
        console,
        Theme.build(allow_animation=False),
        enable_enhanced_input=False,
        enable_steering=steering,
    )
    scripted = _ScriptedInput(*input_lines)
    repl._input = scripted  # type: ignore[assignment]
    repl._reader = _SteeringReader(
        scripted,  # type: ignore[arg-type]
        enabled=steering,
        pending=repl._pending_rows,
        preview=repl._renderer.preview,
    )
    return repl, scripted, provider, buffer


class TestALineTypedWhileTheAgentWorks:
    @pytest.mark.asyncio
    async def test_it_steers_the_running_turn(self) -> None:
        """The line reaches the model on its next call, and does not open a turn
        of its own — the two halves of "steering, not queueing"."""
        repl, scripted, provider, _ = _repl()

        turn = asyncio.ensure_future(repl._run_turn("read every file"))
        await scripted.queue.put(STEER)
        await turn

        assert provider.calls >= 2, "the fixture produced no second call to steer into"
        assert not provider.saw(0, STEER)
        assert provider.saw(1, STEER)
        assert repl._state.turns == 1, "it must not have started a turn of its own"

    @pytest.mark.asyncio
    async def test_it_appears_in_the_transcript(self) -> None:
        """prompt_toolkit erases its line on submit, so without this the message
        the agent acted on is nowhere in the scrollback."""
        repl, scripted, _, buffer = _repl()

        turn = asyncio.ensure_future(repl._run_turn("read every file"))
        await scripted.queue.put(STEER)
        await turn

        assert STEER in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_it_appears_exactly_once(self) -> None:
        """Printed by the REPL when it is sent, and again by the renderer when the
        runner echoed it back as USER_INPUT — the same sentence twice, in two
        places, with the agent's work in between."""
        repl, scripted, _, buffer = _repl()

        turn = asyncio.ensure_future(repl._run_turn("read every file"))
        await scripted.queue.put(STEER)
        await turn

        assert buffer.getvalue().count(STEER) == 1

    @pytest.mark.asyncio
    async def test_the_read_is_told_it_is_concurrent(self) -> None:
        """Otherwise the editor does not install the stdout proxy, and the renderer
        writes over the prompt for the whole turn."""
        repl, scripted, _, _ = _repl()

        turn = asyncio.ensure_future(repl._run_turn("read every file"))
        await scripted.queue.put(STEER)
        await turn

        assert scripted.concurrent[0] is True

    @pytest.mark.asyncio
    async def test_a_quiet_turn_is_unchanged(self) -> None:
        repl, _, provider, buffer = _repl()

        await repl._run_turn("read every file")

        assert provider.calls == 2
        assert "All done." in buffer.getvalue()


class TestSlashCommandsTypedMidTurn:
    @pytest.mark.asyncio
    async def test_a_stateful_one_is_held_until_the_turn_ends(self) -> None:
        """``/clear`` mid-turn would swap the conversation id while the runner is
        still writing events under the old one. Held, not refused: the user typed
        it on purpose."""
        repl, scripted, _, buffer = _repl()
        before = repl._state.conversation_id

        turn = asyncio.ensure_future(repl._run_turn("read every file"))
        await scripted.queue.put("/clear")
        await turn

        assert repl._state.conversation_id == before
        assert "/clear" in repl._deferred
        # Not printed when it was typed — it was a queue row then. Printed as the
        # queue is drained, which is immediately above the run it causes.
        assert "/clear" in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_a_read_only_one_runs_immediately(self) -> None:
        repl, scripted, _, buffer = _repl()

        turn = asyncio.ensure_future(repl._run_turn("read every file"))
        await scripted.queue.put("/help")
        await turn

        assert "/help" not in repl._deferred
        assert "Show this help" in buffer.getvalue()

    def test_every_command_declares_whether_it_is_safe_mid_turn(self) -> None:
        """Pinned as a whole mapping rather than asserted case by case, so a
        command added later cannot inherit an answer silently in either
        direction."""
        from domyn.agents.interfaces.cli.commands import COMMANDS

        assert {c.name: c.mid_turn for c in COMMANDS} == {
            "/help": True,
            "/exit": False,
            "/clear": False,
            "/agent": True,
            "/skills": True,
            "/skill": True,
            # Rewrites config.yaml and rebuilds the agent, like `/model`.
            "/skills-autodiscover": False,
            "/auto": True,
            "/approvals": False,
            "/verbose": False,
            "/voice": False,
            "/model": False,
            "/config": True,
            # Writes a folder, and reloads a changed config first.
            "/export": False,
            "/context": True,
            "/compact": False,
            # `/ps` reads. It would *like* to write — noticing a command has
            # ended is worth recording — but mid-turn it skips that and stays a
            # pure reader, which is the line this mapping draws. Worth the
            # exception: "what is running" is a question asked while something
            # runs, and holding it until the turn ends answers it too late.
            "/ps": True,
            # `/stop` kills a process and records it. Held.
            "/stop": False,
            # `/resume` swaps the conversation id and reads the terminal to offer
            # a choice — both of the things mid-turn rules out.
            "/resume": False,
        }


class TestNothingTypedIsLost:
    @pytest.mark.asyncio
    async def test_a_line_submitted_as_the_turn_ends_is_carried_out(self) -> None:
        """``suspend()`` cancels and then awaits, so a read that completed in
        between is a submitted line rather than a cancellation. Without this the
        line disappears, which is the behaviour that stops people typing at all."""
        scripted = _ScriptedInput()
        reader = _SteeringReader(scripted, enabled=True)  # type: ignore[arg-type]
        reader.start()
        await scripted.queue.put("too late to steer")
        # Let the read complete, then suspend as the turn's teardown would.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await reader.suspend()

        assert reader.carried() == "too late to steer"
        assert reader.carried() is None, "consumed once, or it starts two turns"

    @pytest.mark.asyncio
    async def test_a_half_typed_draft_survives_a_suspend(self) -> None:
        """An approval prompt interrupts the read. Whatever was being typed has to
        come back, or every approval quietly eats a sentence."""
        scripted = _ScriptedInput()
        scripted._draft = "half a sen"
        reader = _SteeringReader(scripted, enabled=True)  # type: ignore[arg-type]
        reader.start()
        await asyncio.sleep(0)
        await reader.suspend()
        reader.start()
        await asyncio.sleep(0)

        assert scripted.defaults[-1] == "half a sen"
        await reader.suspend()

    @pytest.mark.asyncio
    async def test_a_carried_line_becomes_the_next_turn(self) -> None:
        """The other end of the carry: the turn's teardown hands it to the loop,
        which asks for nothing and runs it instead."""
        repl, _, _, _ = _repl()
        repl._reader._carried = "the next question"

        await repl._run_turn("read every file")

        assert "the next question" in repl._deferred


class TestOnlyOneReaderOfTheTerminal:
    @pytest.mark.asyncio
    async def test_the_reader_is_stopped_before_an_approval_prompt(self, monkeypatch) -> None:
        """The approval panel reads with a bare ``input()`` that does not know the
        editor exists; two live readers split the user's keystrokes between them at
        the terminal driver's discretion."""
        from domyn.agents.interfaces.cli import repl as repl_module

        repl, _, _, _ = _repl()
        seen: list[Any] = []

        def _fake_prompt(console, theme, event):
            seen.append(repl._reader.task)
            from domyn.agents import BaseEvent

            action: Any = event.action
            action.approve = False
            return BaseEvent(
                event_type=ExecutionEventType.TOOL_FEEDBACK, author="user", action=action
            )

        monkeypatch.setattr(repl_module, "prompt_for_approval", _fake_prompt)

        from domyn.agents import BaseEvent

        async def _one_approval(*args: Any, **kwargs: Any):
            yield BaseEvent(
                event_type=ExecutionEventType.TOOL_NEEDS_FEEDBACK,
                author="Assistant",
                action=__import__("domyn.agents.core", fromlist=["FeedbackAction"]).FeedbackAction(
                    name="look",
                    parameters=[],
                    approval_type=__import__(
                        "domyn.agents.core", fromlist=["ApprovalType"]
                    ).ApprovalType.USER_CONFIRM,
                ),
            )

        call = {"n": 0}
        real_run = repl._runner.run

        def _run(*args: Any, **kwargs: Any):
            call["n"] += 1
            if call["n"] == 1:
                return _one_approval()
            return real_run(*args, **kwargs)

        repl._runner.run = _run  # type: ignore[method-assign]

        await repl._run_turn("read every file")

        assert seen == [None], "a reader was still live while the approval prompt ran"

    @pytest.mark.asyncio
    async def test_it_is_reading_again_once_the_turn_continues(self) -> None:
        """Suspended for the prompt, resumed after — otherwise the rest of the turn
        cannot be steered at all."""
        repl, _, _, _ = _repl()
        repl._reader.start()
        await repl._reader.suspend()
        repl._reader.resume()

        assert repl._reader.task is not None
        await repl._reader.suspend()

    @pytest.mark.asyncio
    async def test_the_resumed_read_still_draws_the_queue(self) -> None:
        """The editor draws the queue from what it was handed when the read began,
        so a resume that forgets it leaves the rows gone for the rest of the turn —
        until the next submit happens to start a read that has them again. The
        reader holds the queue itself precisely so no caller can forget."""
        repl, scripted, _, _ = _repl()
        repl._queued.append(_Queued(tag="queued", text=STEER, event_id="pending"))

        # `sleep(0)` is one loop tick, not a timing guess: a task cancelled before
        # its first step never runs `ask()` at all, so without it there is nothing
        # to have forgotten the queue.
        repl._reader.start()
        await asyncio.sleep(0)
        await repl._reader.suspend()
        repl._reader.resume()
        await asyncio.sleep(0)
        await repl._reader.suspend()

        assert len(scripted.pending) == 2, "the read was not resumed"
        assert all(callback is not None for callback in scripted.pending)
        assert scripted.rows() == [("queued", STEER)]


class TestWithoutATerminalNothingChanges:
    def test_a_redirected_console_never_starts_a_reader(self) -> None:
        """The exact shape of every scripted test in this directory. Without the
        gate those stubs would answer instantly and steer their way through the
        scripts asserting how many turns ran."""
        provider = _ToolThenAnswer()
        agent = Agent(
            name="Assistant",
            description="t",
            instruction="i",
            llm_provider=provider,
            planner=ToolUsePlannerStrategy(),
        )
        console = Console(file=io.StringIO(), width=100, force_terminal=False)
        repl = Repl(agent, console, Theme.build(allow_animation=False))

        assert repl._reader.enabled is False

    def test_the_kill_switch_turns_it_off_on_a_real_terminal(self, monkeypatch) -> None:
        """Built through the real gate, not the test helper's override, because the
        gate is what is under test."""
        from domyn.agents.interfaces.cli.prompt import InputSession

        monkeypatch.setattr(InputSession, "supports_concurrent_input", property(lambda self: True))
        console = Console(file=io.StringIO(), width=100, force_terminal=True)
        agent = Agent(
            name="Assistant",
            description="t",
            instruction="i",
            llm_provider=_ToolThenAnswer(),
            planner=ToolUsePlannerStrategy(),
        )
        theme = Theme.build(allow_animation=False)

        monkeypatch.delenv("DOMYN_NO_STEERING", raising=False)
        assert Repl(agent, console, theme)._reader.enabled is True

        monkeypatch.setenv("DOMYN_NO_STEERING", "1")
        assert Repl(agent, console, theme)._reader.enabled is False

    @pytest.mark.asyncio
    async def test_a_disabled_reader_never_asks(self) -> None:
        repl, scripted, _, _ = _repl(steering=False)

        await repl._run_turn("read every file")

        assert scripted.asked == 0


class TestAMessageTheTurnNeverTookIn:
    """A steer is written to the store, so the REPL cannot assume it was read.

    The runner now absorbs at the end of a turn too, which leaves only the turns
    that end some other way — interrupted, or stopped to ask the user something.
    A message stranded there has been printed and then ignored, which is the one
    outcome worse than answering it late.
    """

    @pytest.mark.asyncio
    async def test_it_starts_the_next_turn(self) -> None:
        repl, _, _, buffer = _repl()
        repl._queued.append(_Queued(tag="queued", text=STEER, event_id="never_echoed"))

        await repl._run_turn("read every file")

        assert list(repl._deferred) == [STEER]
        assert STEER in buffer.getvalue(), "it left the queue without reaching the transcript"
        assert not repl._queued

    @pytest.mark.asyncio
    async def test_an_absorbed_one_does_not(self) -> None:
        """The receipt is the runner echoing it back, so a message that was taken
        in must not be run a second time as a turn of its own."""
        repl, scripted, provider, _ = _repl()

        turn = asyncio.ensure_future(repl._run_turn("read every file"))
        await scripted.queue.put(STEER)
        await turn

        assert provider.saw(1, STEER), "it was never absorbed, so this proves nothing"
        assert not repl._deferred
        assert not repl._queued

    @pytest.mark.asyncio
    async def test_the_bookkeeping_does_not_leak_between_turns(self) -> None:
        repl, _, _, _ = _repl()
        repl._queued.append(_Queued(tag="queued", text=STEER, event_id="never_echoed"))

        await repl._run_turn("read every file")
        repl._deferred.clear()
        await repl._run_turn("and again")

        assert not repl._deferred
        assert not repl._queued


class TestTheQueueIsVisibleWhileItWaits:
    """A message typed mid-turn is not history yet, so it does not go in the
    scrollback: it is drawn above the prompt until it takes effect, and printed
    at the point it does. Printing it on submit put a copy in the middle of an
    answer it had no part in, seconds before the agent read it."""

    @pytest.mark.asyncio
    async def test_a_steer_waiting_is_a_row_and_not_a_line(self) -> None:
        """Dispatched while the turn is genuinely in flight, because that is the
        only time there is a turn to hand the message to — ``steer()`` delivers to a
        running turn and writes nothing otherwise."""
        repl, _, provider, buffer = _repl()

        rows: list[list[tuple[str, str]]] = []
        printed: list[str] = []

        async def observe() -> None:
            """Runs while the first LLM call is suspended, i.e. mid-turn."""
            before = len(buffer.getvalue())
            await repl._dispatch_typed(STEER)
            rows.append(repl._pending_rows())
            printed.append(buffer.getvalue()[before:])

        provider.pause = observe
        await repl._run_turn("read every file")

        assert rows == [[("queued", STEER)]]
        assert STEER not in printed[0]

    @pytest.mark.asyncio
    async def test_a_held_command_is_a_row_too(self) -> None:
        """It waits on the turn rather than on the runner, but it is waiting either
        way, and the user has the same question about it."""
        repl, _, _, buffer = _repl()
        before = len(buffer.getvalue())

        await repl._dispatch_typed("/clear")

        assert repl._pending_rows() == [("held", "/clear")]
        assert list(repl._deferred) == ["/clear"]
        assert "/clear" not in buffer.getvalue()[before:]

    @pytest.mark.asyncio
    async def test_the_editor_is_given_the_queue_as_a_callable(self) -> None:
        """A snapshot could never lose a row: the prompt is re-rendered on a timer,
        and a message absorbed mid-read has to stop being drawn."""
        repl, scripted, _, _ = _repl()

        turn = asyncio.ensure_future(repl._run_turn("read every file"))
        await scripted.queue.put(STEER)
        await turn

        given = [callback for callback in scripted.pending if callback is not None]
        assert given, "the reader was started without the queue"
        assert all(callable(callback) for callback in given)
        # It was absorbed during the turn, so the callable now reports nothing.
        assert scripted.rows() == []

    @pytest.mark.asyncio
    async def test_every_read_draws_the_answers_unfinished_end(self) -> None:
        """The editor is the only thing that can show the part of a streaming
        answer not printed yet, so every read is handed the renderer's preview —
        the ones started again after an approval or a steer as well."""
        repl, scripted, _, _ = _repl()

        turn = asyncio.ensure_future(repl._run_turn("read every file"))
        await scripted.queue.put(STEER)
        await turn

        assert scripted.previews, "no read was started"
        assert all(preview == repl._renderer.preview for preview in scripted.previews)

    @pytest.mark.asyncio
    async def test_the_row_goes_when_the_runner_takes_it(self) -> None:
        repl, scripted, provider, buffer = _repl()

        turn = asyncio.ensure_future(repl._run_turn("read every file"))
        await scripted.queue.put(STEER)
        await turn

        assert provider.saw(1, STEER), "it was never absorbed, so this proves nothing"
        assert repl._pending_rows() == []
        assert buffer.getvalue().count(STEER) == 1

    @pytest.mark.asyncio
    async def test_one_that_could_not_be_sent_is_never_queued(self) -> None:
        """`steer()` returns None for a conversation that does not exist: nothing
        was written, so no echo is coming and waiting for one would strand it."""
        repl, _, _, buffer = _repl()

        await repl._dispatch_typed(STEER)

        assert repl._pending_rows() == []
        assert list(repl._deferred) == [STEER]
        assert STEER in buffer.getvalue()


async def _typed_mid_turn(repl: Repl, scripted: _ScriptedInput, text: str = STEER) -> None:
    """Submit a line and wait until the REPL has handed it to the turn.

    No sleep: the wait is on the queue row appearing, which is the REPL's own record
    that ``push_parts`` answered ``delivered``. Called from inside the first LLM
    call, so the message provably arrives before the tool call it is meant to change.
    """
    await scripted.queue.put(text)
    for _ in range(100):
        if repl._queued:
            return
        await asyncio.sleep(0)
    raise AssertionError("the reader never picked the line up")


class TestTheWholePathWithGatedTools:
    """End to end, on the configuration the CLI actually ships.

    Every built-in tool is gated, and ``/auto`` answers the gate rather than
    removing it, so one turn is several ``run()`` calls with a
    ``TOOL_NEEDS_FEEDBACK`` between them. The queue that holds a mid-turn message
    lives exactly one of those, which is why the gate has to be a boundary: this is
    the case that was broken — the message was accepted, dropped in teardown, and
    surfaced as the *next* turn once the agent had finished the work it was meant to
    change.
    """

    @pytest.mark.asyncio
    async def test_the_model_reads_it_on_the_call_after_the_approval(self) -> None:
        repl, scripted, provider, buffer = _repl(gated=True)
        repl._state.auto_approve = True
        provider.pause = lambda: _typed_mid_turn(repl, scripted)

        await repl._run_turn("read every file")

        assert "auto-approved" in buffer.getvalue(), "the gate was never reached"
        assert not provider.saw(0, STEER)
        assert provider.saw(1, STEER), "read only after the whole turn, as in the bug"

    @pytest.mark.asyncio
    async def test_it_does_not_come_back_as_a_turn_of_its_own(self) -> None:
        """What the bug looked like from the outside: the message ran as the next
        turn, after the work it was meant to redirect had been done."""
        repl, scripted, provider, _ = _repl(gated=True)
        repl._state.auto_approve = True
        provider.pause = lambda: _typed_mid_turn(repl, scripted)

        await repl._run_turn("read every file")

        assert repl._state.turns == 1
        assert not repl._deferred, "nothing was handed back to start a later turn"

    @pytest.mark.asyncio
    async def test_it_leaves_the_queue_and_enters_the_transcript_once(self) -> None:
        """The queue row is cleared by the runner echoing the event back, which is
        the only receipt there is — ``delivered`` says a turn took it, not that a
        turn read it."""
        repl, scripted, provider, buffer = _repl(gated=True)
        repl._state.auto_approve = True
        provider.pause = lambda: _typed_mid_turn(repl, scripted)

        await repl._run_turn("read every file")

        assert repl._queued == []
        assert buffer.getvalue().count(STEER) == 1


class _ChoosingInput(_ScriptedInput):
    """A scripted editor that can also carry a choice, as the real one does."""

    def __init__(self, *lines: str) -> None:
        super().__init__(*lines)
        self.choice: Any = None
        self.offered: list[list[str]] = []

    def offer_choice(self, question: str, options: list[str], *, start: int = 0) -> Any:
        self.offered.append(list(options))
        self.choice = asyncio.get_running_loop().create_future()
        return self.choice

    def withdraw_choice(self) -> None:
        if self.choice is not None and not self.choice.done():
            self.choice.set_result(None)
        self.choice = None


def _choosing_repl(rounds: int = 1):
    repl, _, provider, buffer = _repl(gated=True, rounds=rounds)
    scripted = _ChoosingInput()
    repl._input = scripted  # type: ignore[assignment]
    repl._reader = _SteeringReader(
        scripted,  # type: ignore[arg-type]
        enabled=True,
        pending=repl._pending_rows,
    )
    return repl, scripted, provider, buffer


async def _until(condition: Any, timeout: float = 5.0) -> None:
    """Wait for ``condition``, by the clock rather than by loop ticks.

    Reaching the gate goes through work done off the loop (the conversation store,
    the tool call), so a tick budget that is ample on one machine runs out before
    the approval is even asked on a slower CI runner.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("never happened")
        await asyncio.sleep(0.001)


async def _turn_with(repl: Repl, script: Any) -> None:
    """Run a turn while ``script`` plays the user, failing rather than hanging."""
    task = asyncio.ensure_future(script())
    await asyncio.wait_for(repl._run_turn("read every file"), timeout=15)
    await asyncio.wait_for(task, timeout=1)


class TestTheApprovalInsideTheEditor:
    """With the editor live, the approval is a choice drawn inside it, and what is
    typed meanwhile is handled as any line typed mid-turn."""

    @pytest.mark.asyncio
    async def test_the_choice_answers_the_gate(self) -> None:
        repl, scripted, provider, buffer = _choosing_repl()

        async def answer() -> None:
            await _until(lambda: scripted.choice is not None)
            scripted.choice.set_result(len(scripted.offered[0]) - 1)  # "no"

        await _turn_with(repl, answer)

        assert scripted.offered[0][0] == "yes, just this once"
        assert "denied" in buffer.getvalue()
        assert any("refused" in text.lower() for text in provider.seen[1])

    @pytest.mark.asyncio
    async def test_a_message_typed_during_it_reaches_the_resumed_turn(self) -> None:
        """No run() is in flight while the gate waits, so the message is held and
        pushed once the turn resumes — read by the very next model call, not
        turned into a turn of its own."""
        repl, scripted, provider, buffer = _choosing_repl()

        async def type_then_approve() -> None:
            await _until(lambda: scripted.choice is not None)
            await scripted.queue.put(STEER)
            await _until(lambda: bool(repl._queued))
            assert scripted.choice is not None, "a message must not answer the question"
            scripted.choice.set_result(0)

        await _turn_with(repl, type_then_approve)

        assert provider.saw(1, STEER)
        assert repl._state.turns == 1
        assert not repl._deferred
        assert repl._queued == []
        assert buffer.getvalue().count(STEER) == 1

    @pytest.mark.asyncio
    async def test_auto_typed_during_it_approves_this_call_and_the_next(self) -> None:
        repl, scripted, _, buffer = _choosing_repl(rounds=2)

        async def turn_auto_on() -> None:
            await _until(lambda: scripted.choice is not None)
            await scripted.queue.put("/auto")

        await _turn_with(repl, turn_auto_on)

        assert repl._state.auto_approve is True
        assert len(scripted.offered) == 1, "the second call must not ask"
        # One read before /auto, and one after it for the rest of the turn —
        # without the second the bar vanished as soon as auto-approve answered.
        assert scripted.asked == 2
        assert buffer.getvalue().count("auto-approved") == 2

    @pytest.mark.asyncio
    async def test_shift_tab_during_it_approves_this_call_and_the_next(self) -> None:
        """The key takes the question down rather than typing a line, so the
        choice settles with no pick. Read as an aborted prompt that is a denial
        — the opposite of what the user just asked for."""
        repl, scripted, _, buffer = _choosing_repl(rounds=2)

        async def press_shift_tab() -> None:
            await _until(lambda: scripted.choice is not None)
            repl._toggle_auto_from_key()  # what the editor's Shift+Tab binding calls

        await _turn_with(repl, press_shift_tab)

        assert repl._state.auto_approve is True
        assert len(scripted.offered) == 1, "the second call must not ask"
        assert buffer.getvalue().count("auto-approved") == 2
        assert "denied" not in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_ctrl_c_still_denies(self) -> None:
        """No pick and auto-approve off is still an aborted prompt."""
        repl, scripted, _, buffer = _choosing_repl()

        async def ctrl_c() -> None:
            await _until(lambda: scripted.choice is not None)
            scripted.choice.set_result(None)

        await _turn_with(repl, ctrl_c)

        assert repl._state.auto_approve is False
        assert "auto-approved" not in buffer.getvalue()


class TestShiftTabOutsideAnApproval:
    def test_it_flips_auto_approve_both_ways(self) -> None:
        repl, scripted, _, _ = _choosing_repl()
        repl._toggle_auto_from_key()
        assert repl._state.auto_approve is True
        repl._toggle_auto_from_key()
        assert repl._state.auto_approve is False
        assert scripted.choice is None, "with nothing open there is nothing to answer"

    def test_the_editor_reads_the_flag_the_session_owns(self) -> None:
        """`/auto` and the key change the same flag, and the toolbar follows it."""
        repl, _, _, _ = _repl()
        wired = Repl(
            repl._agent,
            Console(file=io.StringIO()),
            Theme.build(allow_animation=False),
            enable_enhanced_input=False,
        )
        assert wired._input._auto_on() is False
        wired._state.auto_approve = True  # as /auto leaves it
        assert wired._input._auto_on() is True
        wired._input.toggle_auto()  # as Shift+Tab does
        assert wired._state.auto_approve is False
