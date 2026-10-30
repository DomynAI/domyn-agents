"""Interactive conversation loop.

The important structural point: **a turn runs to completion, and only then is
the user asked for input.**

The previous implementation asked for the next message from *inside* the event
loop, in the branch handling a non-partial ``RESPONSE``. That coupled "the turn
ended" to "a response was produced", which is false whenever a turn ends any
other way — most importantly when it suspends on ``TOOL_NEEDS_FEEDBACK`` waiting
for a human decision. In that case the next user message was never read, and the
outer loop re-invoked the runner with the *previous* input, looping forever on
stale state. Separating "drive one turn" from "ask for the next message" removes
that class of bug entirely, and makes approval just another reason a turn
continues.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
import uuid
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import yaml
from rich.console import Console
from rich.text import Text

from domyn.agents.agents import Agent
from domyn.agents.audio import Transcriber, TranscriptionError, resolve_transcriber
from domyn.agents.core import ApprovalType, BaseEvent, ExecutionEventType, Part
from domyn.agents.decisions import attach_decisions, runtime_of
from domyn.agents.instructions import (
    InstructionRuntime,
)
from domyn.agents.interfaces.cli import commands as slash
from domyn.agents.interfaces.cli import context_view, conversations, grants, processes_view
from domyn.agents.interfaces.cli.approval import (
    begin_approval,
    prompt_for_approval,
    settle_approval,
)
from domyn.agents.interfaces.cli.calls import as_written
from domyn.agents.interfaces.cli.config import (
    LOCAL_SKILLS_DIRS,
    apply_context_window,
    approvals_paths,
    build_approval_policy,
    config_path,
    describe_configuration_v1,
    is_configuration_v1_usable,
    print_config_description,
    read_raw_config,
    skill_workspace_roots,
    unique_paths,
)
from domyn.agents.interfaces.cli.export import (
    ExportError,
    export_configuration,
    print_export_report,
)
from domyn.agents.interfaces.cli.inventory import Inventory, describe_change, take_inventory
from domyn.agents.interfaces.cli.prompt import InputSession
from domyn.agents.interfaces.cli.renderer import EventRenderer
from domyn.agents.interfaces.cli.setup import (
    run_setup,
    skill_autodiscovery,
    with_skill_autodiscovery,
)
from domyn.agents.interfaces.cli.skills_view import render_skills
from domyn.agents.interfaces.cli.spinner import Spinner
from domyn.agents.interfaces.cli.theme import Theme
from domyn.agents.interfaces.cli.voice import (
    PERMISSION_HINT,
    PROBE_TIMEOUT,
    Recorder,
    VoiceError,
    VoiceInput,
)
from domyn.agents.loader import load_application_from_file
from domyn.agents.logger import get_logger
from domyn.agents.planner_strategy.context_management.budget import (
    CONTEXT_USAGE_METADATA_KEY,
    ContextUsage,
)
from domyn.agents.runner import PushResult, Runner
from domyn.agents.skills import LocalDirSkillSource, SkillNotFoundError, SkillsRuntime
from domyn.agents.skills.integration import attach_skills

if TYPE_CHECKING:
    from domyn.agents.conversation_manager import ConversationInfo

_T = TypeVar("_T")

APPLICATION_NAME = "domyn-cli"
USER_ID = "local-user"


def _duration(seconds: float) -> str:
    """A turn's length, readable at a glance: ``4.2s``, ``2m 05s``."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes}m {rest:02d}s"


def _set_log_level(verbose: bool) -> None:
    """Raise or lower library logging to match ``/verbose``.

    The session starts quiet — inference URLs and httpx status lines belong to a
    server's output, not above the answer a user is waiting for. That leaves no
    way to get them back short of restarting, which is exactly when they are
    wanted: something just went wrong. ``/verbose`` is that switch.

    The root logger is what moves, because the noise comes from several loggers
    (``domyn.agents``, ``httpx``) and lowering only ours would leave half of it
    hidden.
    """
    import logging

    logging.getLogger().setLevel(logging.INFO if verbose else logging.WARNING)


@dataclass
class SessionState:
    """Mutable state of one interactive session."""

    conversation_id: str = field(default_factory=lambda: f"conv_{uuid.uuid4().hex}")
    verbose: bool = False
    turns: int = 0
    auto_approve: bool = False
    """Answer yes to every approval prompt.

    Deliberately *not* a standing grant: nothing is written to a store, nothing
    outlives the session, and there is nothing to revoke afterwards. It is the
    front-end saying yes on the user's behalf, which is why the hardline rules
    still apply — those are decided before the prompt is ever reached, so a
    catastrophic command is refused even here."""

    last_usage: ContextUsage | None = None
    """What the last request to the model cost.

    The planner measures it and ships it in ``LLM_START`` metadata; here it is
    only collected. The CLI cannot rebuild it on its own because it does not build
    the ``InteractionContext`` — and in any case the useful number is the one from
    the request that actually went out, not a hypothetical one."""

    def reset_conversation(self) -> None:
        self.conversation_id = f"conv_{uuid.uuid4().hex}"
        self.turns = 0
        # The measurement belongs to the conversation: keeping it past /clear
        # would say something false about the new one.
        self.last_usage = None

    def resume_conversation(self, info: ConversationInfo) -> None:
        """Continue ``info``, from ``--resume`` or ``/resume`` alike.

        Its exchanges count as turns: they decide whether the id is printed on
        the way out, and a conversation resumed and left at once is not empty.
        """
        self.conversation_id = info.conversation_id
        self.turns = conversations.exchanges(info)
        # It belongs to the conversation that was left, not to the one resumed;
        # the next request replaces it with a real measurement.
        self.last_usage = None


@dataclass
class _Queued:
    """Something typed mid-turn that has not taken effect yet.

    ``event_id`` is set for a steer — the id the runner will echo back when it
    absorbs the message, which is the only reliable receipt. A held slash command
    has none: it is waiting on the turn to end, not on the runner.
    """

    tag: str
    text: str
    event_id: str | None = None


class _InterruptedReadError(Exception):
    """The Ctrl-C that ended the background read, on its way to the REPL.

    That read runs in a task of its own, and asyncio re-raises a
    ``KeyboardInterrupt`` that ends a task out of the event loop itself, past every
    handler here: the first Ctrl-C mid-turn took the whole CLI down instead of
    stopping the turn. The read hands it over as this instead, and `take` turns it
    back into a ``KeyboardInterrupt`` raised where the REPL can catch it.
    """


class _SteeringReader:
    """Owns the one background read of the next typed line.

    One task, never two. Two coroutines awaiting the same stdin is not a race with
    a winner — each keystroke goes to whichever reader the terminal driver happens
    to wake, and nobody can be asked to reason about that. Every other place that
    reads the terminal (the approval prompt, the setup wizard, the end-of-turn
    prompt) therefore has to ``suspend()`` this one first, and be sure it stopped.

    ``suspend()`` is the interesting half. It cancels *and then awaits*, and treats
    a task that finished in between as a line the user submitted rather than as a
    cancellation: ``Task.cancel()`` on an already-finished task does nothing and
    returns False, and the following await hands back the result. Checking
    ``done()`` before cancelling instead would leave the same window open one
    statement earlier. This is what makes a line typed just as the turn ends start
    the next turn, rather than vanish.
    """

    def __init__(
        self,
        session: InputSession,
        *,
        enabled: bool,
        pending: Callable[[], list[tuple[str, str]]] | None = None,
        preview: Callable[[], str] | None = None,
    ) -> None:
        self._session = session
        self.enabled = enabled
        self.task: asyncio.Task[str] | None = None
        self._carried: str | None = None
        self._draft = ""
        self.ended: BaseException | None = None
        # Held here rather than passed to `start()`, because every read of the
        # session has to draw the same queue and there are four places that start
        # one. Passed per call, the one after an approval prompt was missed and the
        # rows stayed gone for the rest of the turn.
        self._pending = pending
        # Same reason: every read draws the answer's unfinished end.
        self._preview = preview

    def start(self, status: Callable[[], tuple[str, str]] | None = None) -> None:
        """Begin reading, unless one is already in flight or this is switched off."""
        if not self.enabled or self.task is not None or self.ended is not None:
            return
        draft, self._draft = self._draft, ""
        self.task = asyncio.ensure_future(self._read(draft, status))

    async def _read(self, draft: str, status: Callable[[], tuple[str, str]] | None) -> str:
        try:
            return await self._session.ask(
                default=draft,
                status=status,
                concurrent=True,
                pending=self._pending,
                preview=self._preview,
            )
        except KeyboardInterrupt as exc:
            raise _InterruptedReadError from exc

    async def take(self) -> str | None:
        """Consume a finished read. Records the exit keys instead of raising them."""
        task, self.task = self.task, None
        if task is None:
            return None
        try:
            return (await task).strip() or None
        except asyncio.CancelledError:
            return None
        except EOFError as exc:
            # Recorded rather than raised: this runs from inside the turn's event
            # loop, and the caller decides where each of the two exit keys lands.
            self.ended = exc
            return None
        except _InterruptedReadError:
            self.ended = KeyboardInterrupt()
            return None

    async def suspend(self) -> None:
        """Stop reading and leave the terminal to somebody else."""
        task = self.task
        if task is None:
            return
        # Read before the cancel: cancelling discards the buffer, and the buffer is
        # a sentence somebody was halfway through. `start()` seeds it back.
        self._draft = self._session.draft()
        task.cancel()
        line = await self.take()
        if line is not None:
            # Submitted between the cancel and the await. The user meant to send
            # it, so it is carried out rather than thrown away — and there is no
            # draft left to restore.
            self._carried = line
            self._draft = ""

    def resume(self, status: Callable[[], tuple[str, str]] | None = None) -> None:
        self.start(status)

    def carried(self) -> str | None:
        """A line taken during a suspend, consumed once."""
        line, self._carried = self._carried, None
        return line

    def take_draft(self) -> str:
        """Half-typed text left over from a cancelled read, consumed once."""
        draft, self._draft = self._draft, ""
        return draft


class Repl:
    """Drives an agent conversation on the terminal."""

    def __init__(
        self,
        agent: Agent,
        console: Console,
        theme: Theme,
        *,
        runner: Runner | None = None,
        config: dict[str, Any] | None = None,
        config_file: Path | None = None,
        verbose: bool = False,
        enable_enhanced_input: bool = True,
        enable_steering: bool = True,
        resumed: ConversationInfo | None = None,
        run_command: str = "domyn run",
        reload: Callable[[], Awaitable[tuple[Any, dict[str, Any]]]] | None = None,
    ) -> None:
        self._agent = agent
        self._route_browser_notices()
        # What the exit hint puts before `--resume <id>`: the agent file too, when
        # the session was started from one.
        self._run_command = run_command
        self._console = console
        self._theme = theme
        # A policy with the disk-backed stores, so "remember always" survives the
        # session. An injected runner keeps whatever policy it came with.
        # A persistent store, so a conversation can be resumed in another
        # session: an injected runner keeps whatever manager it came with.
        self._runner = runner or Runner(
            approval_policy=build_approval_policy(),
            conversation_manager=conversations.build_manager(),
        )
        self._config = config
        # Resolved once here so `/config` and `/model`'s default-config check
        # never have to care whether the caller passed a relative path.
        self._config_file = config_file.resolve() if config_file is not None else None
        self._state = SessionState(verbose=verbose)
        if resumed is not None:
            self._state.resume_conversation(resumed)
        self._spinner = Spinner(console, theme)
        # Logging writes to the stream on its own schedule, from anywhere in the
        # library, and never passes through the renderer that keeps the spinner
        # out of the way.
        self._spinner.attach_to_logging()
        self._renderer = EventRenderer(console, theme, verbose=verbose, spinner=self._spinner)
        self._input = InputSession(
            console,
            theme,
            enable_enhanced=enable_enhanced_input,
            auto_mode=lambda: self._state.auto_approve,
            on_toggle_auto=self._toggle_auto_from_key,
        )
        # The answer's unfinished end is drawn by the editor (`EventRenderer.preview`),
        # which repaints on a timer slower than tokens arrive.
        self._renderer.on_preview = self._input.refresh
        self._renderer.output = self._input.output
        # Typed mid-turn and not consumed there: a held slash command, a line
        # submitted as the turn ended, a steer the runner could not take.
        self._deferred: deque[str] = deque()
        # Typed mid-turn and not in effect *yet*: steers written to the store for
        # the running turn, and slash commands held until it ends. Drawn above the
        # prompt while they wait, and taken out as each one lands — a steer when
        # the runner echoes it back, a command when it runs. Whatever is still here
        # when the turn ends never took effect; see the end of `_run_turn`.
        self._queued: list[_Queued] = []
        self._closing = False
        self._inventory: Inventory | None = None
        # Rebuilds the agent from the config file, with the session's options; and
        # the file's content it was last built from (`_reload_if_changed`).
        self._reload = reload
        self._config_content = self._read_config()
        # True while a gated call waits on the user. There is no run() in flight
        # then, so nothing typed can be handed to one: messages wait here and are
        # pushed as soon as the turn resumes — see `_consume`.
        self._approving = False
        self._held_for_resume: list[str] = []
        self._reader = _SteeringReader(
            self._input,
            pending=self._pending_rows,
            preview=self._renderer.preview,
            enabled=(
                enable_steering
                and os.environ.get("DOMYN_NO_STEERING") is None
                and self._input.supports_concurrent_input
                # The console has to be a terminal too, not just stdin. `domyn run
                # > log.txt` from a tty keeps an editor but has nowhere to repaint
                # around, and it is also the shape of every scripted test: a stub
                # editor answering instantly into a StringIO console. Without this
                # those stubs would steer their way through the very scripts that
                # assert how many turns ran.
                and bool(getattr(console, "is_terminal", False))
            ),
        )
        self._turn_started = 0.0
        # Built on the first /voice, not before: it reads the configuration and
        # may load a speech model, which a session that never talks should not pay.
        self._voice: VoiceInput | None = None
        # `domyn run -v` should start with the logs already on, the same ones
        # `/verbose` toggles later.
        _set_log_level(verbose)

    # --- public API --------------------------------------------------------

    async def run(self, initial_prompt: str | None = None) -> int:
        """Run the session until the user exits. Returns a process exit code."""
        try:
            return await self._loop(initial_prompt)
        finally:
            # In a finally, not after the loop: the session can also end by
            # exception or by Ctrl-C, and the thing this releases is a sandbox
            # container. A container leaked on every crash is a container that
            # is still running tomorrow.
            await self.close()

    async def close(self) -> None:
        """Release what the tools hold. Safe to call more than once."""
        await self._say_what_survives_us()
        self._say_where_to_find_us()
        await self._agent.close()

    def _say_where_to_find_us(self) -> None:
        """Print the id on the way out, because it is the only way back in.

        Once, inside the command that reopens it: a line of its own would only
        repeat it.

        Only once something has been said: an id for a session where nobody
        typed anything names an empty conversation, and inviting someone to
        resume one teaches them the feature does nothing.
        """
        if not self._state.turns:
            return
        self._console.print(
            f"  resume it with  {self._run_command} --resume {self._state.conversation_id}",
            style=self._theme.style("meta"),
            highlight=False,
        )

    async def _say_what_survives_us(self) -> None:
        """Name the commands that will outlive this session.

        They are not stopped on the way out, and that is the decision rather than
        an oversight: "start it, close the terminal, come back tomorrow" is a
        reason this exists. But a process nobody was told about is one the user
        finds out about from the fan. Said, not asked — a prompt on exit is a
        prompt nobody reads.
        """
        try:
            running = processes_view.summary((await self._conversation_state()).state)
        except Exception as exc:
            get_logger().debug(f"could not report surviving processes: {exc}")
            return
        if not running:
            return
        self._console.print(
            f"  {running} and will keep going after this session. "
            "Resume this conversation to follow them.",
            style=self._theme.style("warning"),
            highlight=False,
        )

    async def _loop(self, initial_prompt: str | None = None) -> int:
        pending_user_text = initial_prompt
        # What the startup line under the banner described: the baseline every
        # turn's end is compared with (`_say_what_changed`).
        self._inventory = await take_inventory(self._agent)

        while True:
            # Drained before asking, so nobody is prompted for something they have
            # already typed.
            if pending_user_text is None and self._deferred:
                pending_user_text = self._deferred.popleft()

            if pending_user_text is None:
                if self._closing:
                    # Ctrl-D arrived mid-turn. The turn was allowed to finish:
                    # ending a session by discarding work the agent had already
                    # done is a worse reading of one keystroke than waiting.
                    self._console.print()
                    self._say("Session closed.")
                    return 0
                try:
                    # The keyword is passed only when there is a draft to restore,
                    # so the ordinary between-turns read keeps the exact call shape
                    # it has always had — embedders and tests that supply their own
                    # one-argument `ask` are not broken by a feature they never
                    # switched on.
                    draft = self._reader.take_draft()
                    pending_user_text = await (
                        self._input.ask(default=draft) if draft else self._input.ask()
                    )
                except (EOFError, KeyboardInterrupt):
                    self._console.print()
                    self._say("Session closed.")
                    return 0

            text = pending_user_text
            pending_user_text = None

            if not text:
                continue

            if slash.is_slash(text):
                if await self._handle_slash(text) is False:
                    return 0
                continue

            try:
                await self._run_turn(text)
            except KeyboardInterrupt:
                # Abort the turn, keep the session alive: the user interrupted
                # the agent, not the conversation.
                self._renderer.finish()
                self._console.print()
                self._say("Turn interrupted.")

    # --- turn execution ----------------------------------------------------

    async def _run_turn(self, user_text: str) -> None:
        """Drive one turn to completion, resuming across approval pauses."""
        pending_event: BaseEvent | None = BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            author="user",
            content=[Part(text=user_text)],
        )
        self._state.turns += 1
        self._renderer.begin_turn()
        self._turn_started = time.monotonic()
        if self._reader.enabled:
            # Stopped, not paused, and the spinner does not run at all: the
            # renderer resumes the spinner after every event it draws, so a pause
            # would let the next event restart an in-place redraw that then fights
            # the editor for the bottom line. What the spinner was saying moves
            # into the editor, on the row above the prompt, repainted by whoever
            # owns that line.
            #
            # The renderer has to know too: with an editor on screen, anything
            # written without a trailing newline is erased by its next repaint,
            # so the answer can no longer be streamed a token at a time.
            self._renderer.set_live_prompt(True)
            self._reader.start(self._elapsed)
        else:
            await self._spinner.start()
        completed = False
        try:
            await self._reload_if_changed()
            await self._drive(pending_event)
            completed = True
        finally:
            # The reader first: it is the thing holding the terminal, and an
            # exception on its way out of _drive (Ctrl-C) passes through here too
            # and must not reach the handler in _loop with a live editor on screen.
            await self._reader.suspend()
            if completed:
                # Once, at the end: the spinner's clock only ever times one model
                # call, so this is the only place the whole turn is measured.
                self._console.print(
                    f"  Completed in {_duration(time.monotonic() - self._turn_started)}",
                    style=self._theme.style("meta"),
                    highlight=False,
                )
                await self._say_what_changed()
            # Nothing is drawing the queue any more, so whatever is left in it has
            # to land in the transcript now or be lost from view entirely.
            #
            # Every one of them is printed, and this is the right place for it:
            # whatever is still queued runs next, so the anchor lands immediately
            # above the work it causes.
            #
            # A held command is already in `_deferred`. A steer still queued was
            # handed to the turn and never taken in — the turn ended some other way,
            # interrupted or stopped to ask the user something, which is rare now
            # that the runner absorbs at the end of a turn too. It starts the next
            # turn instead, as text, and there is no copy of it anywhere else: a
            # steer the turn never took in was never written, so re-sending it
            # cannot make the model read the sentence twice.
            for missed in self._queued:
                self._renderer.echo_user(missed.text)
                if missed.event_id is not None:
                    self._deferred.append(missed.text)
            self._queued.clear()
            # Typed during an approval and never pushed: the turn ended first.
            self._deferred.extend(self._held_for_resume)
            self._held_for_resume.clear()
            carried = self._reader.carried()
            if carried is not None:
                # Submitted as the turn was ending. Too late to steer this turn, so
                # it starts the next one — which is what the user would expect from
                # having pressed Enter at all. Printed here rather than when that
                # turn begins, because here is where it lands in the transcript:
                # immediately above the work it causes, exactly as a message typed
                # between turns would.
                self._renderer.echo_user(carried)
                self._deferred.append(carried)
            if isinstance(self._reader.ended, EOFError):
                self._closing = True
            # In a finally so an interrupted turn does not leave the line half
            # drawn and the cursor hidden.
            await self._spinner.stop()
            self._renderer.set_live_prompt(False)
            # After the spinner and after the answer: it is the last line left
            # in view while the next message is being typed, which is when it is
            # wanted. It stays quiet on its own when there is nothing to say.
            await self._render_context_indicator()

    def _pending_rows(self) -> list[tuple[str, str]]:
        """The queue, as the editor wants to draw it.

        Handed to the reader as a callable rather than a snapshot: the prompt is
        re-rendered on a timer, so a row added or taken away while a read is in
        flight shows up without the read being restarted.
        """
        return [(item.tag, item.text) for item in self._queued]

    def _dequeue(self, event_id: str | None) -> _Queued | None:
        """Take a steer out of the queue by the id the runner echoed back."""
        if event_id is None:
            return None
        for index, item in enumerate(self._queued):
            if item.event_id == event_id:
                return self._queued.pop(index)
        return None

    def _elapsed(self) -> tuple[str, str]:
        """What the spinner would be saying, for the editor to draw instead.

        Nothing while an approval is open: the human is the slow part then, and
        a timer ticking next to a question reads as pressure to answer.
        """
        if self._approving:
            return "", ""
        return self._spinner.status()

    async def _drive(self, pending_event: BaseEvent | None) -> None:
        while pending_event is not None:
            if pending_event.event_type is ExecutionEventType.TOOL_FEEDBACK:
                # The approved call runs now, with no TOOL_START of its own to
                # restart the clock: that one was drawn before the question.
                self._spinner.executing()
            needs_feedback = await self._consume(
                self._runner.run(
                    application_name=APPLICATION_NAME,
                    user_id=USER_ID,
                    user_input=pending_event,
                    root=self._agent,
                    conversation_id=self._state.conversation_id,
                )
            )

            self._renderer.finish()

            if needs_feedback is None:
                pending_event = None
                continue

            if self._state.auto_approve:
                self._spinner.pause()
                pending_event = self._approve_automatically(needs_feedback)
                continue

            if self._approves_in_editor():
                pending_event = await self._approve_in_editor(needs_feedback)
                continue

            # Two readers of stdin must not both be live: prompt_for_approval runs
            # its own editor, which does not know the session's exists, and the
            # terminal would split the user's keystrokes between them. The *await*
            # is the point — a cancel that has been requested is not a reader that
            # has stopped.
            await self._reader.suspend()
            try:
                try:
                    # The human is the slow part now, not the agent; a timer ticking
                    # next to a question reads as pressure to answer.
                    self._spinner.pause()
                    pending_event = prompt_for_approval(self._console, self._theme, needs_feedback)
                except (EOFError, KeyboardInterrupt):
                    # Treat an aborted prompt as a denial rather than leaving the
                    # turn suspended forever.
                    self._console.print()
                    self._say("Approval aborted — treated as denied.")
                    pending_event = self._deny(needs_feedback)
            finally:
                # Whatever the human answered, the turn continues, so reading
                # resumes. In a finally because the answer can arrive as an
                # exception and the reader has to come back either way.
                self._reader.resume(self._elapsed)
                await self._dispatch_typed(self._reader.carried())

    def _approves_in_editor(self) -> bool:
        """Whether the approval can be asked inside the editor running alongside.

        Only when that editor is live: with no reader (a pipe, a redirected
        console, steering off) there is nothing to put it in, and the blocking
        prompt is the only reader of the terminal.
        """
        return (
            self._reader.enabled
            and self._reader.ended is None
            and callable(getattr(self._input, "offer_choice", None))
        )

    async def _approve_in_editor(self, needs_feedback: BaseEvent) -> BaseEvent:
        """Ask for approval inside the live editor, which stays usable meanwhile.

        A line sent while the question is open is handled as any line typed
        mid-turn — a read-only command runs, others are held, a message is queued
        — except that a message cannot reach the runner yet, so it waits for the
        turn to resume. ``/auto`` answers the open question too: it says "stop
        asking", and leaving the one on screen would be the opposite.

        The reader stands down only for what reads whole lines on its own: the new
        values of a ``PARAMS_UPDATE`` call and an edited pattern.
        """
        action: Any = needs_feedback.action
        self._approving = True
        try:
            if getattr(action, "approval_type", None) == ApprovalType.PARAMS_UPDATE:
                menu = await self._with_reader_stood_down(
                    lambda: begin_approval(self._console, self._theme, needs_feedback)
                )
            else:
                menu = begin_approval(self._console, self._theme, needs_feedback)

            self._reader.start(self._elapsed)
            answer = self._input.offer_choice(menu.question, menu.options)
            try:
                while not answer.done():
                    reader = self._reader.task
                    waiting: set[asyncio.Future[Any]] = {answer}
                    if reader is not None:
                        waiting.add(reader)
                    await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
                    if reader is None or not reader.done() or answer.done():
                        continue
                    await self._dispatch_typed(await self._reader.take())
                    # Restarted before anything below can return: the turn goes on
                    # either way, and without a read the bar is gone for the rest
                    # of it. A no-op once the reader has ended.
                    if not self._closing:
                        self._reader.start(self._elapsed)
                    if self._state.auto_approve:
                        return self._approve_automatically(needs_feedback)
                    if self._reader.ended is not None:
                        # Ctrl-D on an empty line. Denied, as an aborted prompt
                        # always has been; the session closes once the turn ends.
                        self._console.print()
                        self._say("Approval aborted — treated as denied.")
                        return self._deny(needs_feedback)
            finally:
                self._input.withdraw_choice()

            picked = answer.result()
            if picked is None and self._state.auto_approve:
                # Shift+Tab turned auto-approve on while this was open, and took
                # the question down with it. Approved once, as auto-approve always
                # does; read as an aborted prompt it would be a denial.
                return self._approve_automatically(needs_feedback)
            if menu.edits_pattern(picked):
                return await self._with_reader_stood_down(
                    lambda: settle_approval(
                        self._console, self._theme, needs_feedback, menu, picked
                    )
                )
            return settle_approval(self._console, self._theme, needs_feedback, menu, picked)
        finally:
            self._approving = False

    async def _with_reader_stood_down(self, read: Callable[[], _T]) -> _T:
        """Run something that reads whole lines itself, with the editor gone."""
        await self._reader.suspend()
        try:
            return read()
        finally:
            self._reader.resume(self._elapsed)
            await self._dispatch_typed(self._reader.carried())

    async def _consume(self, stream: AsyncGenerator[BaseEvent, None]) -> BaseEvent | None:
        """Render one runner stream, reading the terminal alongside it.

        Advanced one event at a time with an explicit ``__anext__`` task raced
        against the pending read, rather than with ``async for``, and for the same
        reason ``Runner._merge_with_injections`` does it: ``async for`` parks this
        coroutine inside the runner for the whole of a slow tool call, so a line
        typed during a sixty-second command could not be written until the call was
        already over. That is exactly the wrong moment — the runner takes steering
        in at ``TOOL_END``, so a write landing there is a coin flip against the
        absorber, and losing it costs a full extra tool cycle before the model sees
        the message. Racing writes it comfortably mid-call.

        Only one ``__anext__`` is outstanding at a time, and the next is created
        only after the current event has been rendered, so the runner never runs
        ahead of what has actually been shown.
        """
        needs_feedback: BaseEvent | None = None
        next_event: asyncio.Task[BaseEvent] | None = asyncio.ensure_future(stream.__anext__())
        try:
            while next_event is not None:
                waiting: set[asyncio.Future[Any]] = {next_event}
                reader = self._reader.task
                if reader is not None:
                    waiting.add(reader)

                done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)

                # The reader branch first, deliberately. When both finish in the
                # same wakeup — the line submitted as the final event arrives —
                # handling the event first would end the stream and tear the reader
                # down in the cleanup, turning a submitted line into a lost one.
                if reader is not None and reader in done:
                    await self._dispatch_typed(await self._reader.take())
                    if self._reader.ended is None and not self._closing:
                        self._reader.start(self._elapsed)

                if next_event in done:
                    try:
                        event = next_event.result()
                    except StopAsyncIteration:
                        next_event = None
                    else:
                        if self._held_for_resume:
                            # The resumed run is in flight from its first event on,
                            # so a message typed during the approval can reach it.
                            await self._send_held()
                        self._record_usage(event)
                        if event.event_type is ExecutionEventType.USER_INPUT:
                            # The runner echoes back every message it absorbs, so
                            # this is the receipt: whatever is never echoed is a
                            # message the turn did not take in. It leaves the queue
                            # and enters the transcript here, which is the point it
                            # actually took effect. `echo_user` also records it, so
                            # the renderer drops the event below rather than
                            # drawing the same sentence twice.
                            taken = self._dequeue(event.event_id)
                            if taken is not None:
                                self._renderer.echo_user(taken.text)
                        self._renderer.render(event)
                        if event.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK:
                            # The runner has suspended. Remember it and prompt once
                            # the event stream is drained, so the panel is not
                            # interleaved with trailing events.
                            needs_feedback = event
                        next_event = asyncio.ensure_future(stream.__anext__())

                if isinstance(self._reader.ended, KeyboardInterrupt):
                    # While a prompt_toolkit application runs it installs its own
                    # SIGINT handling, so a mid-turn Ctrl-C surfaces inside the
                    # reader task rather than interrupting this one. Re-raised here
                    # so it reaches the "Turn interrupted." handler in _loop, which
                    # is the escape hatch users are told to reach for.
                    ended, self._reader.ended = self._reader.ended, None
                    raise ended
        finally:
            if next_event is not None:
                next_event.cancel()
            with contextlib.suppress(Exception):
                await stream.aclose()
        return needs_feedback

    async def _send_held(self) -> None:
        held, self._held_for_resume = self._held_for_resume, []
        for line in held:
            # Its placeholder row goes; the steer sent below puts its own back.
            for index, item in enumerate(self._queued):
                if item.event_id is None and item.tag == "queued" and item.text == line:
                    del self._queued[index]
                    break
            await self._dispatch_typed(line)

    async def _dispatch_typed(self, line: str | None) -> None:
        """Decide what a line that arrived mid-turn is.

        Three kinds, and the distinction is about what can safely happen while a
        ``run()`` is in flight on this conversation. A read-only slash command runs
        now; any other slash command is *held* rather than refused, because
        refusing would throw away something typed on purpose; anything else is a
        message for the agent.

        A command that runs now is printed first: the editor erased itself on
        submit, and its output would otherwise arrive under nothing. Anything that
        has to wait is *queued* instead — drawn above the prompt until it takes
        effect, and printed into the transcript at that point. Printing it here
        would put it in the scrollback above work it had no part in, seconds before
        the agent read it.
        """
        if not line:
            return

        if slash.is_slash(line):
            resolved = slash.resolve(line)
            if resolved is not None and resolved[0].mid_turn:
                self._renderer.echo_user(line)
                if await self._handle_slash(line, mid_turn=True) is False:  # pragma: no cover
                    self._closing = True
                return
            self._deferred.append(line)
            self._queued.append(_Queued(tag="held", text=line))
            return

        if self._approving:
            # No run() to hand it to until the approval is answered.
            self._held_for_resume.append(line)
            self._queued.append(_Queued(tag="queued", text=line))
            return

        # Imported here rather than at module scope, like `Runner` above: pulling the
        # runner in at import time is the cycle this module keeps avoiding.
        from domyn.agents.runner import PushTiming

        sent: PushResult | None = None
        try:
            # A user message, so it waits for the turn's next boundary rather than
            # being spliced in wherever the agent happens to be: that is where the
            # planner rebuilds its messages, and so the only timing at which the
            # model reliably reads it.
            sent = await self._runner.push_parts(
                APPLICATION_NAME,
                self._state.conversation_id,
                [Part(text=line)],
                event_type=ExecutionEventType.USER_INPUT,
                timing=PushTiming.SCHEDULED,
            )
        except Exception as exc:
            get_logger().warning(f"could not steer the running turn: {type(exc).__name__}")
            get_logger().debug(f"steer detail: {exc}")

        if sent is None or not sent.delivered or sent.event.event_id is None:
            # Either it raised, or the turn ended between the line being typed and
            # this call — the message is handed to a running turn and nothing else,
            # so there is no echo coming back and nothing to wait in the queue for.
            # Nothing is holding the message now, so it is printed and handed to the
            # next turn: dropping it would leave the user watching an agent that
            # visibly ignored them.
            self._renderer.echo_user(line)
            self._say("couldn't send that mid-turn — it will start the next turn instead")
            self._deferred.append(line)
            return

        self._queued.append(_Queued(tag="queued", text=line, event_id=sent.event.event_id))

    def _toggle_auto_from_key(self) -> None:
        """Shift+Tab: flip auto-approve, from the editor, mid-turn or not.

        Nothing is printed: the line above the prompt is the answer, and
        between turns nothing else may write to the terminal while the editor is
        drawing. What auto-approve lets through is still printed, call by call.

        Turned on with an approval open, it answers that one too — it is the
        call the user is looking at, and leaving it open would make the key look
        like it did nothing.
        """
        self._state.auto_approve = not self._state.auto_approve
        if self._state.auto_approve and self._approving:
            self._input.withdraw_choice()

    def _approve_automatically(self, needs_feedback: BaseEvent) -> BaseEvent:
        """Say yes without asking, and show what was let through.

        Still printed, and in the warning colour: the point of auto-approve is
        to stop being interrupted, not to stop being told. A command that ran
        unseen is one nobody can notice was wrong.
        """
        action: Any = needs_feedback.action
        subject = str(needs_feedback.metadata.get("approval_subject") or action.name)
        self._console.print(
            f"  {self._theme.symbols.approved} auto-approved: {as_written(subject)}",
            style=self._theme.style("auto_mode"),
            highlight=False,
            markup=False,
            emoji=False,
        )
        action.approve = True
        # No scope: auto-approve does not silently accumulate standing
        # permissions that outlive the session it was turned on in.
        action.remember_scope = None
        return BaseEvent(event_type=ExecutionEventType.TOOL_FEEDBACK, author="user", action=action)

    @staticmethod
    def _deny(needs_feedback: BaseEvent) -> BaseEvent:
        # Typed as Any: the core declares `action` through an unresolved union
        # (`ActionsTypes`, itself `# type: ignore`d), so mypy mis-narrows it and
        # calls the defensive branch unreachable. The check is real at runtime.
        action: Any = needs_feedback.action
        if action is not None:
            action.approve = False
        return BaseEvent(
            event_type=ExecutionEventType.TOOL_FEEDBACK,
            author="user",
            action=action,
        )

    # --- slash commands ----------------------------------------------------

    async def _handle_slash(self, text: str, *, mid_turn: bool = False) -> bool:
        """Handle a slash command. Returns ``False`` when the session must end."""
        resolved = slash.resolve(text)
        if resolved is None:
            self._console.print(
                f"Unknown command: {text.split()[0]}  (try /help)",
                style=self._theme.style("warning"),
                highlight=False,
            )
            return True

        command, argument = resolved
        name = command.name

        if name == "/exit":
            self._say("Session closed.")
            return False
        if name == "/help":
            self._render_help()
        elif name == "/clear":
            self._state.reset_conversation()
            # The screen holds the conversation just left: kept, it would read as
            # part of the new one. rich's clear() leaves the scrollback, so ESC[3J
            # erases that too, after it: some terminals move the cleared screen
            # into the scrollback.
            self._console.clear()
            if self._console.is_terminal and not self._console.is_dumb_terminal:
                self._console.file.write("\x1b[3J")
            self._say("Started a new conversation.")
        elif name == "/agent":
            await self._reload_if_changed()
            self._render_agent()
        elif name == "/auto":
            self._state.auto_approve = not self._state.auto_approve
            if self._state.auto_approve:
                stop = "/auto or shift+tab" if self._input.enhanced else "/auto again"
                self._console.print(
                    "  Auto-approve on: every tool call runs without asking, "
                    "including writes and shell commands. Only the hardline "
                    f"rules still apply. {stop} to stop.",
                    style=self._theme.style("auto_mode"),
                    highlight=False,
                )
            else:
                self._say("Auto-approve off.")
        elif name == "/approvals":
            await self._handle_approvals(argument)
        elif name == "/voice":
            await self.toggle_voice()
        elif name == "/verbose":
            self._state.verbose = not self._state.verbose
            self._renderer = EventRenderer(
                self._console, self._theme, verbose=self._state.verbose, spinner=self._spinner
            )
            _set_log_level(self._state.verbose)
            self._say(f"Verbose {'on' if self._state.verbose else 'off'}.")
        elif name in ("/skills", "/skill"):
            await self._render_skills(argument if name == "/skill" else None)
        elif name == "/config":
            await self._reload_if_changed()
            await self._show_config_file()
        elif name == "/compact":
            await self._compact_now()
        elif name == "/context":
            context_view.render(self._console, self._theme, self._state.last_usage)
        elif name == "/model":
            await self._change_model()
        elif name == "/skills-autodiscover":
            await self._set_autodiscovery(argument)
        elif name == "/export":
            await self._export(argument)
        elif name == "/ps":
            await self._show_processes(mid_turn=mid_turn)
        elif name == "/stop":
            await self._stop_process(argument)
        elif name == "/resume":
            await self._resume(argument)
        return True

    async def _resume(self, argument: str) -> None:
        """Continue a past conversation: by id, or by picking one.

        Switching is all it takes — the runner loads an existing conversation,
        its state and its agent stack from the id alone. What is done here is
        refusing an id that names nothing, which would otherwise be created on
        first use and look like a resume that lost the history.
        """
        manager = self._runner.conversation_manager
        if argument:
            info = await conversations.find(manager, APPLICATION_NAME, USER_ID, argument.strip())
            if info is None:
                self._console.print(
                    f"  No conversation {argument.strip()}. /resume on its own lists them.",
                    style=self._theme.style("warning"),
                    highlight=False,
                )
                return
        else:
            found = await conversations.recent(manager, APPLICATION_NAME, USER_ID)
            found = [c for c in found if c.conversation_id != self._state.conversation_id]
            info = conversations.pick(self._console, self._theme, found)
            if info is None:
                return

        self._state.resume_conversation(info)
        # The screen holds the conversation that was left, and the one resumed
        # is about to be drawn in full: below it, the two would read as one.
        self._console.clear()
        conversations.render_history(self._console, self._theme, info, verbose=self._state.verbose)

    def _backend(self) -> Any:
        """The shell backend the agent's ``run_command`` uses, if it has one.

        Read from ``Agent.builtin_tools``, where the loader keeps the bundles.
        This used to read a ``_builtin_tools`` attribute nothing set any more, so
        ``/ps`` never asked the machine and ``/stop`` could stop nothing.
        """
        for built in getattr(self._agent, "builtin_tools", None) or []:
            backend = getattr(built, "backend", None)
            if backend is not None:
                return backend
        return None

    async def _show_processes(self, *, mid_turn: bool = False) -> None:
        """What is still running, asked of the machine rather than of the session.

        The listing enumerates the process directory, so it also finds anything
        started before a crash — those are the ones nothing else can reach.
        Looking is also what makes "it has finished" knowable, so what it learns
        is written back; without that the prompt goes on claiming a command is
        running hours after it stopped.

        Except mid-turn, where it stays a pure reader. Writing conversation state
        while the runner is using it is the line ``mid_turn`` draws, and this
        command is worth having *now* — "what is running" is a question asked
        while something runs. What it learns is simply not recorded; the next
        call outside a turn records it.
        """
        holder = await self._conversation_state()
        updated = await processes_view.render(
            self._console,
            self._theme,
            holder.state,
            self._backend(),
            self._state.conversation_id,
        )
        if updated is not None and not mid_turn:
            await self._persist_conversation_state(holder)

    async def _stop_process(self, handle: str) -> None:
        holder = await self._conversation_state()
        updated = await processes_view.stop(
            self._console, self._theme, holder.state, self._backend(), (handle or "").strip()
        )
        if updated is None:
            return
        # Written back, or the next turn rebuilds a registry that still believes
        # the process is live and counts it against the cap.
        await self._persist_conversation_state(holder)

    async def _compact_now(self) -> None:
        """Ask for a compaction instead of waiting for the threshold.

        Goes through the runner as an event rather than being done here, so the
        manual path exists for every front-end and not only for this one. The
        effect is recorded and applies to the next request: the history itself is
        never rewritten.
        """
        self._renderer.begin_turn()
        await self._spinner.start()
        try:
            await self._drive(
                BaseEvent(
                    event_type=ExecutionEventType.COMPACTION_REQUESTED,
                    author="user",
                    content=[Part(text="compact the conversation")],
                )
            )
        finally:
            await self._spinner.stop()
        # Read as a turn's would: the measurement now has the saving taken off.
        await self._render_context_indicator()

    async def _render_context_indicator(self) -> None:
        """The lines nobody asked for, when there is a reason to show them.

        Background commands first: one still running is a fact about what
        happens next, context pressure a fact about the turn just past. Both stay
        quiet when there is nothing to say — a status line that is always there
        stops being read.
        """
        running = processes_view.summary((await self._conversation_state()).state)
        if running:
            self._console.print(
                f"  {running} — /ps to look, /stop to end one",
                style=self._theme.style("accent"),
                highlight=False,
            )
        shown = context_view.indicator(self._state.last_usage)
        if shown is None:
            return
        text, style_name = shown
        self._console.print(f"  {text}", style=self._theme.style(style_name), highlight=False)

    def _record_usage(self, event: BaseEvent) -> None:
        """Collect the context measurement if the event carries one.

        A manual compaction carries no measurement of its own, only what it
        saved: taken off the last one, so the indicator and ``/context`` do not
        go on showing the context as it was before the compaction.

        Silent about everything: a malformed piece of telemetry must not bring
        down the turn that carried it.
        """
        metadata = event.metadata or {}
        usage = self._state.last_usage
        if event.event_type is ExecutionEventType.CONTEXT_COMPACTED:
            saved = metadata.get("tokens_saved")
            if usage is not None and isinstance(saved, int) and saved > 0:
                self._state.last_usage = usage.model_copy(
                    update={"message_tokens": max(0, usage.message_tokens - saved)}
                )
            return
        if event.event_type is not ExecutionEventType.LLM_START:
            return
        raw = metadata.get(CONTEXT_USAGE_METADATA_KEY)
        if not isinstance(raw, dict):
            return
        try:
            self._state.last_usage = ContextUsage.model_validate(raw)
        except Exception as exc:
            get_logger().debug(f"context usage metadata ignored: {type(exc).__name__}: {exc}")

    # --- standing approvals ------------------------------------------------

    class _StateHolder:
        """The one attribute ``ConversationApprovalStore`` needs.

        It reads and writes ``context.state``; outside a turn there is no
        interaction context, so the state is fetched from the conversation
        manager and handed over in something that looks like one. Passing
        ``None`` instead would make conversation-scoped grants silently invisible
        to ``/approvals`` — the worst outcome for a command whose whole job is to
        show what is there.
        """

        def __init__(self, state: dict[str, Any]) -> None:
            self.state = state

    async def _conversation_state(self) -> _StateHolder:
        try:
            info = await self._runner.conversation_manager.get_conversation(
                APPLICATION_NAME, USER_ID, self._state.conversation_id
            )
        except Exception as exc:
            get_logger().warning(f"could not read conversation state: {type(exc).__name__}")
            get_logger().debug(f"conversation state detail: {exc}")
            info = None
        return self._StateHolder(dict(getattr(info, "state", None) or {}))

    async def _persist_conversation_state(self, holder: _StateHolder) -> None:
        """Write back a state the conversation store mutated in place."""
        try:
            await self._runner.conversation_manager.update_conversation_state(
                APPLICATION_NAME, USER_ID, self._state.conversation_id, holder.state
            )
        except Exception as exc:
            # Reported, never swallowed: the user was told a permission is gone.
            get_logger().warning(f"could not persist revocation: {type(exc).__name__}")
            get_logger().debug(f"revocation persist detail: {exc}")
            self._console.print(
                "  Removed for now, but the conversation state could not be saved — "
                "it may come back. /verbose for the reason.",
                style=self._theme.style("warning"),
                highlight=False,
            )

    async def _handle_approvals(self, argument: str) -> None:
        holder = await self._conversation_state()
        policy = self._runner.approval_policy
        entries = await grants.collect(policy, context=holder)

        verb, _, rest = argument.strip().partition(" ")
        verb = verb.lower()

        if not verb:
            grants.render(self._console, self._theme, entries, paths=approvals_paths())
            return

        if verb not in ("revoke", "clear"):
            self._say(f"Unknown: /approvals {verb}. Try /approvals, revoke <n>, or clear.")
            return

        if not entries:
            self._say("Nothing to remove — there are no standing approvals.")
            return

        if verb == "clear":
            targets = entries
        else:
            wanted, problem = grants.parse_selector(rest, len(entries))
            if problem:
                self._say(problem)
                return
            targets = [e for e in entries if e.index in wanted]

        removed: list[str] = []
        for entry in targets:
            try:
                if await entry.store.revoke(entry.grant, holder):
                    removed.append(f"{entry.label}  ({entry.scope})")
            except Exception as exc:
                get_logger().warning(f"could not revoke a grant: {type(exc).__name__}")
                get_logger().debug(f"revoke detail: {exc}")

        if any(e.scope == "conversation" for e in targets):
            await self._persist_conversation_state(holder)

        if not removed:
            self._say("Nothing was removed.")
            return
        # Echoed back rather than counted: indices shift, and "revoked 2" gives
        # the user no way to notice they removed the wrong line.
        for label in removed:
            self._console.print(
                f"  removed  {label}", style=self._theme.style("meta"), highlight=False
            )

    async def _show_config_file(self) -> None:
        """Show the config file backing this session as a redacted summary —
        its path, then a comprehensive but secret-free rundown of its
        contents. The path alone would leave no way to tell, from the output,
        whether it's the saved default config or a file run explicitly, which
        is the whole reason this command exists.

        This is a summary, not the raw file: `config.yaml` is a full
        `ConfigurationV1` file and nothing stops a hand-edited one from
        carrying a literal secret (an API key, an MCP `client_secret`, a
        bearer token in `headers`) instead of an `${ENV_VAR}` reference.
        Dumping it verbatim would echo that straight to the terminal, so this
        goes through `describe_configuration_v1` instead, the same
        allowlisted-fields renderer `domyn setup` uses.
        """
        if self._config_file is None:
            self._say("No configuration file — this agent was loaded from a Python module.")
            return
        raw = read_raw_config(self._config_file)
        if raw is None:
            self._say(f"Could not read or parse {self._config_file}.")
            return
        description = describe_configuration_v1(raw, path=self._config_file)
        print_config_description(self._console, self._theme, description)

    async def _change_model(self) -> None:
        """Reconfigure the model mid-session and reload the agent from disk.

        A full reload through the same `load_application_from_file` path used
        at startup — not a one-off patch of `self._agent.llm_provider` — so the
        live agent and the persisted config can never drift apart. Only the
        saved default config can be re-pointed this way: a session started
        from an explicitly-named `.yaml` file or a `.py` module is a no-op
        here, checked against the resolved path rather than `self._config`
        being set — an explicit file gets a `self._config` too now, but it
        isn't the default config `run_setup`/the reload below are hardcoded to.
        """
        if self._config_file is None or self._config_file != config_path().resolve():
            self._say("`/model` only applies to the default configuration.")
            return
        # From the file as it is now: a hand edit made since the last message would
        # otherwise be overwritten by a rewrite of the configuration loaded before it.
        await self._reload_if_changed()

        declared_instructions = getattr(self._agent, "standing_instructions", None)
        instructions_runtime = (
            declared_instructions if isinstance(declared_instructions, InstructionRuntime) else None
        )
        decisions_runtime = runtime_of(self._agent)
        # Before reloading the agent capture skill runtime: the agent file `domyn run` was
        # given may list skill paths that config.yaml does not.
        declared_skills = getattr(self._agent, "skills", None)
        skills_runtime = declared_skills if isinstance(declared_skills, SkillsRuntime) else None
        skill_source = getattr(skills_runtime, "source", None) if skills_runtime else None
        if isinstance(skill_source, LocalDirSkillSource):
            runtime_skill_dirs = skill_source.paths
        else:
            runtime_skill_dirs = []

        updated = run_setup(self._console, self._theme, self._config, provider_only=True)
        if updated is None:
            return
        self._config = updated
        if not is_configuration_v1_usable(updated):
            return

        # Read skill dirs from config, skills_runtime contains main agent only path
        config_skill_dirs = skill_workspace_roots(self._config)
        # Merge skill dirs from config with those from the skills runtime without duplicates.
        workspace_roots = unique_paths([*config_skill_dirs, *runtime_skill_dirs])
        try:
            reloaded = await load_application_from_file(
                str(config_path()), extra_workspace_roots=[str(d) for d in workspace_roots]
            )
        except Exception as exc:
            self._console.print(
                f"Could not apply the new provider: {type(exc).__name__}: {exc}",
                style=self._theme.style("error"),
                highlight=False,
            )
            return

        if skills_runtime is not None:
            reloaded = attach_skills(reloaded, skills_runtime)
        # The reloaded agent comes off disk without the session's standing
        # instructions (`~/.domyn` lives only on the runtime), so they are
        # carried across like the skills runtime above.
        # Without this the session keeps announcing them while the model stops
        # receiving them.
        if hasattr(reloaded, "standing_instructions"):
            reloaded.standing_instructions = instructions_runtime
        await apply_context_window(getattr(reloaded, "llm_provider", None))
        # Same reason again for the decision runtime: it holds a live client and
        # a circuit breaker, and the reloaded agent would come back with a fresh
        # one — or, when it was attached in code rather than declared in the
        # file, with none at all, silently switching the feature off.
        if decisions_runtime is not None:
            attach_decisions(reloaded, decisions_runtime)
        self._agent = reloaded
        self._route_browser_notices()
        # Written by this command: not a change for `_reload_if_changed` to redo.
        self._config_content = self._read_config()

        provider = updated["llm_providers"][0]
        self._say(f"Now using {provider['type']} · {provider['model_name']}")

    def _render_help(self) -> None:
        rows = slash.help_lines()
        width = max(len(invocation) for invocation, _ in rows)
        self._console.print()
        for invocation, summary in rows:
            self._console.print(
                f"  {invocation.ljust(width)}   {summary}",
                style=self._theme.style("meta"),
                highlight=False,
                # `/approvals [revoke <n> | clear]` is a rich markup tag to a
                # console that has markup on, and rich deletes what it cannot
                # resolve. Said here as well as on the session's console, because
                # the console is handed in and an embedder's may differ.
                markup=False,
            )
        self._console.print(
            "\n  Enter sends · Alt+Enter or Ctrl+J for a newline · Shift+Tab auto-approve"
            "\n  Ctrl-C clears the line, or stops a running turn · twice in a row quits\n",
            style=self._theme.style("meta"),
            highlight=False,
        )

    def _render_agent(self) -> None:
        agent = self._agent
        sym = self._theme.symbols
        self._console.print()
        self._console.print(
            f"  {agent.name}", style=self._theme.style("accent_bold"), highlight=False
        )
        if agent.description:
            self._console.print(
                f"  {agent.description}", style=self._theme.style("meta"), highlight=False
            )
        tools = [t.name for t in getattr(agent, "tools", []) or []]
        subs = [a.name for a in getattr(agent, "sub_agents", []) or []]
        self._console.print(
            f"  {sym.bullet} tools: {', '.join(tools) if tools else '(none)'}",
            style=self._theme.style("meta"),
            highlight=False,
        )
        self._console.print(
            f"  {sym.bullet} sub-agents: {', '.join(subs) if subs else '(none)'}",
            style=self._theme.style("meta"),
            highlight=False,
        )
        # The id is what the working directory is named after, so showing it is
        # what makes that directory findable afterwards.
        self._console.print(
            f"  {sym.bullet} conversation: {self._state.conversation_id}",
            style=self._theme.style("meta"),
            highlight=False,
        )
        home = self._conversation_home()
        if home is not None:
            self._console.print(
                f"  {sym.bullet} working dir: {home}",
                style=self._theme.style("meta"),
                highlight=False,
            )
        self._console.print()

    def _conversation_home(self) -> Any | None:
        """Where this conversation keeps its environment and output, if any."""
        for built in self._agent.builtin_tools:
            sessions = getattr(built.backend, "sessions", None)
            if sessions is not None:
                return sessions.paths(self._state.conversation_id).home
        return None

    def _read_config(self) -> bytes | None:
        """The config file's content, to tell whether it changed; ``None`` if unreadable."""
        if self._config_file is None:
            return None
        try:
            return self._config_file.read_bytes()
        except OSError:
            return None

    async def _export(self, argument: str) -> None:
        """``domyn export <dir>`` for the session's configuration file: the same
        function and the same report, read from the file as it is now. A relative
        folder is relative to where the session started, as in a shell."""
        if not argument.strip():
            self._say(
                "Usage: /export <folder>. Saves this configuration, with its skills and "
                "instruction files, into a new folder you can share or run as a project."
            )
            return
        if self._config_file is None:
            self._say(
                "Nothing to export: this agent was loaded from a Python file, not from a "
                "configuration file."
            )
            return
        await self._reload_if_changed()
        raw = read_raw_config(self._config_file)
        if raw is None:
            self._say(f"Could not read or parse {self._config_file}.")
            return
        try:
            report = export_configuration(raw, Path(argument.strip()))
        except (ExportError, OSError) as exc:
            self._console.print(
                f"Export failed, nothing was written: {exc}",
                style=self._theme.style("error"),
                highlight=False,
            )
            return
        print_export_report(self._console, self._theme, self._config_file, report)

    async def _set_autodiscovery(self, argument: str) -> None:
        """Show, or turn on or off, autodiscovery of ``./.domyn/skills`` and
        ``./.agents/skills``: the switch ``domyn
        setup`` asks about, written to the same file through the same function
        (`with_skill_autodiscovery`), then picked up by the reload every change to the
        file gets. Only the default configuration, as for ``/model``: a file run by
        name is the user's own, and rewriting it would drop its comments."""
        if self._config_file is None or self._config_file != config_path().resolve():
            self._say("`/skills-autodiscover` only applies to the default configuration.")
            return
        await self._reload_if_changed()
        raw = read_raw_config(self._config_file)
        agents = (raw or {}).get("agents") or []
        if raw is None or not agents or not isinstance(agents[0], dict):
            self._say(f"Could not read the agent from {self._config_file}.")
            return
        skills = agents[0].get("skills")
        on = skill_autodiscovery(skills)
        if on is None:
            self._say(
                "This agent's skills do not come from folders, so there is nothing to discover."
            )
            return

        folders = " and ".join(f"./{d.as_posix()}" for d in LOCAL_SKILLS_DIRS)
        where = f"{folders} of the folder domyn runs in"
        choice = argument.strip().lower()
        if not choice:
            self._say(
                f"  Skill autodiscovery is {'on' if on else 'off'}: skills in {where} "
                f"are {'' if on else 'not '}loaded. `/skills-autodiscover on|off` to change it."
            )
            return
        if choice not in ("on", "off"):
            # `[on|off]` is a rich markup tag to a console with markup on.
            self._console.print(
                "Usage: /skills-autodiscover [on|off]",
                style=self._theme.style("meta"),
                highlight=False,
                markup=False,
            )
            return
        enabled = choice == "on"
        if enabled == on:
            self._say(f"  Skill autodiscovery is already {choice}.")
            return

        updated = with_skill_autodiscovery(skills, enabled)
        if updated is None:
            agents[0].pop("skills", None)
        else:
            agents[0]["skills"] = updated
        self._config_file.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        if not await self._reload_if_changed(announce=False):
            return
        self._say(
            f"  Skill autodiscovery {choice}: skills in {where} are "
            f"{'' if enabled else 'no longer '}loaded, from your next message. "
            f"Saved in {self._config_file}."
        )
        await self._say_what_changed()

    async def _reload_if_changed(self, *, announce: bool = True) -> bool:
        """Rebuild the agent from ``config.yaml`` if the file changed since it was built.

        Checked before every turn, after every turn and before the commands that show
        the configuration, so an edit — by the user, or by the agent during a turn —
        is in use from the next message: model, tools, skills, instructions, all of
        it. Rebuilt the way startup builds it, with the session's command-line options
        (``reload``, from `start_session`). The conversation is the runner's, not the
        agent's, so it carries over, as it does for ``/model``.

        A file that no longer loads keeps the agent the session has, and says why once:
        the content is remembered either way, so the same broken file is not retried
        every turn.

        Returns whether the agent was rebuilt; ``announce=False`` leaves saying so to a
        caller that changed the file itself.
        """
        if self._reload is None:
            return False
        content = self._read_config()
        if content is None or content == self._config_content:
            return False
        self._config_content = content
        try:
            agent, raw = await self._reload()
        except Exception as exc:
            self._console.print(
                f"  {self._config_file} changed but cannot be loaded "
                f"({type(exc).__name__}: {exc}); keeping the previous configuration.",
                style=self._theme.style("warning"),
                highlight=False,
            )
            return False
        self._agent, self._config = agent, raw
        if announce:
            self._say(f"  Reloaded {self._config_file}: it changed since the last message.")
        return True

    async def _say_what_changed(self) -> None:
        """Say which skills or instruction files changed since they were last named.

        The line under the banner is scrollback and cannot be redrawn, while both are
        read from disk every turn: a skill added to ./.domyn/skills, or one the agent wrote
        itself, is in play from the next turn on. Silent when nothing changed.
        """
        # The agent may have changed config.yaml during the turn.
        await self._reload_if_changed()
        current = await take_inventory(self._agent)
        change = describe_change(self._inventory, current) if self._inventory else None
        self._inventory = current
        if change:
            self._say(change)

    async def _render_skills(self, name: str | None) -> None:
        await self._reload_if_changed()
        skills_runtime = getattr(self._agent, "skills", None)
        skills_runtime = skills_runtime if isinstance(skills_runtime, SkillsRuntime) else None
        try:
            await render_skills(self._console, self._theme, skills_runtime, name)
        except SkillNotFoundError as exc:
            self._console.print(Text(str(exc), style=self._theme.style("warning")), highlight=False)

    async def toggle_voice(self) -> bool:
        """Turn push-to-talk on or off; True when it is on afterwards.

        The first time, the speech model is loaded (downloaded first, once, if it
        is not there yet), so the first message spoken is not the one that waits
        for it. Off drops a recording in progress.
        """
        voice = self._voice
        if voice is not None and voice.enabled:
            voice.cancel()
            voice.enabled = False
            self._say("Voice off.")
            return False
        if not self._input.enhanced:
            self._say("Voice needs an interactive terminal.")
            return False
        try:
            Recorder.check()
            if voice is None:
                voice = VoiceInput(self._voice_transcriber())
            # Here rather than on the first press: the first opening is when
            # macOS asks for permission, and it blocks until answered.
            await asyncio.wait_for(asyncio.to_thread(Recorder.probe), PROBE_TIMEOUT)
        except TimeoutError:
            self._console.print(
                f"  {PERMISSION_HINT}",
                style=self._theme.style("warning"),
                highlight=False,
                markup=False,
            )
            return False
        except (VoiceError, ValueError) as exc:
            self._console.print(
                f"  {exc}", style=self._theme.style("warning"), highlight=False, markup=False
            )
            return False
        if not voice.transcriber.is_ready():
            self._console.print(
                f"  {voice.transcriber.describe()}: the speech model is downloaded "
                "the first time, this once.",
                style=self._theme.style("meta"),
                highlight=False,
            )
        try:
            await voice.transcriber.prepare(on_download=self._say)
        except TranscriptionError as exc:
            self._console.print(
                f"  {exc}", style=self._theme.style("warning"), highlight=False, markup=False
            )
            return False
        self._voice = voice
        self._input.voice = voice
        voice.enabled = True
        self._say(
            "Voice on: hold space on an empty line to talk, let go to send. "
            "Space during a turn stops it. /voice to turn off."
        )
        return True

    def _voice_transcriber(self) -> Transcriber:
        """`voice.transcriber` from the configuration, or the local one.

        Read from the file this session was started with: the setting lives
        beside the agent, in `~/.domyn/config.yaml`, and is not part of it.
        """
        settings = (self._config or {}).get("voice") or {}
        configured = settings.get("transcriber") if isinstance(settings, dict) else None
        return resolve_transcriber(configured or {"type": "local"})

    def _route_browser_notices(self) -> None:
        """Have the browser tools say, on screen, what they download on first use."""
        for built in getattr(self._agent, "builtin_tools", None) or []:
            browser = getattr(built, "browser", None)
            if browser is not None:
                browser.on_notice = self._say

    def _say(self, message: str) -> None:
        self._console.print(message, style=self._theme.style("meta"), highlight=False)
