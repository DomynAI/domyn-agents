"""Commands that outlive the call that started them.

Real processes throughout. The claims here are about signals, process groups and
file descriptors, and a mock of any of those would only restate the assumption
being checked — three of these tests exist precisely because the obvious
assumption was wrong when measured.
"""

# No `from __future__ import annotations`: this reaches the tool modules, which
# need real annotations at construction time.
import asyncio
import contextlib
import os
import re
import shlex
import shutil
import signal
import time
from pathlib import Path

import pytest

from domyn.agents.tools.builtin import LocalBackend, Workspace, bundle, processes, shell


@pytest.fixture
def backend(tmp_path: Path) -> LocalBackend:
    return LocalBackend(Workspace.at(tmp_path), isolate_environment=False)


async def _launch(backend: LocalBackend, command: str, *, yield_s: float = 0.15, **kw):  # type: ignore[no-untyped-def]
    """Start something and read the first report.

    The default yield is short on purpose. This goes through ``build_launch``
    rather than the tool, so it is not subject to the one-second floor that
    stops ``run_command`` handing back a handle for ``ls`` — and a second per
    launch, forty times over, was most of what this file cost.
    """
    record = processes.new_record(command)
    out = await backend.run(processes.build_launch(record, yield_s=yield_s, **kw), timeout=120)
    return record, processes.parse_report(out.stdout, record, from_cursor=False)


async def _call(tool, **kwargs):  # type: ignore[no-untyped-def]
    """Run a tool whether or not it streams.

    ``check_command`` is a streamable tool — it reports what a command is
    printing while it waits — so awaiting it yields an async generator rather
    than a result. Everything before the last item is progress.
    """
    import inspect

    from domyn.agents.tools import ToolResult

    if not inspect.isasyncgenfunction(tool.execute):
        return await tool.execute(**kwargs)
    last = None
    async for item in tool.execute(**kwargs):
        if isinstance(item, ToolResult):
            last = item
    assert last is not None, "a streaming tool must end with a result"
    return last


async def _finished(backend: LocalBackend, record: processes.ProcessRecord) -> bool:
    return not (await _probe(backend, record)).running


async def _compacted(backend: LocalBackend, record: processes.ProcessRecord) -> bool:
    """Whether the log has been trimmed at least once — the compactor records
    how many bytes it dropped, so a non-zero count is the event itself."""
    seen = await backend.run(f"cat {record.dir}/out.log.cut 2>/dev/null || echo 0", timeout=30)
    return int(seen.stdout.strip() or 0) > 0


async def _wrapped_up(backend: LocalBackend, record: processes.ProcessRecord) -> bool:
    """Whether the wrapper itself has exited, not just the command.

    ``_finished`` turns true on the exit file, and the wrapper writes it *before*
    its last ``compact_once``: a test that measures the log on ``_finished`` can
    land in the middle of that pass."""
    seen = await backend.run(
        f'kill -0 "$(cat {record.dir}/run.pid)" 2>/dev/null && echo alive || echo gone',
        timeout=30,
    )
    return seen.stdout.strip() == "gone"


async def _until(condition, timeout: float = 20.0, step: float = 0.2):  # type: ignore[no-untyped-def]
    """Wait for something to become true, rather than for a fixed time.

    Faster on an idle machine and — the part that matters — still correct on a
    busy one. A fixed sleep has to be long enough for the worst case, which
    makes every run pay for it, and is still too short on the day it is not:
    that is exactly how one of these failed the first time it ran alongside
    thirteen others.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        result = await condition()
        if result:
            return result
        await asyncio.sleep(step)
    return await condition()


async def _probe(backend: LocalBackend, record: processes.ProcessRecord) -> processes.ProcessState:
    out = await backend.run(processes.build_probe(record), timeout=60)
    return processes.parse_report(out.stdout, record, from_cursor=True)


class TestACommandThatFinishesInTime:
    """Almost every command. It must be indistinguishable from before."""

    @pytest.mark.asyncio
    async def test_it_reports_as_finished(self, backend: LocalBackend) -> None:
        _, state = await _launch(backend, "echo hello; exit 3", yield_s=10)
        assert not state.running
        assert state.exit_code == 3
        assert state.out.strip() == "hello"

    @pytest.mark.asyncio
    async def test_the_streams_stay_apart(self, backend: LocalBackend) -> None:
        _, state = await _launch(backend, "echo out; echo err >&2", yield_s=10)
        assert "out" in state.out
        assert "err" in state.err


class TestACommandThatDoesNot:
    @pytest.mark.asyncio
    async def test_the_call_yields_and_the_process_lives(self, backend: LocalBackend) -> None:
        loop = asyncio.get_running_loop()
        started = loop.time()
        record, state = await _launch(
            backend, "for i in $(seq 1 60); do echo line $i; sleep 0.3; done", yield_s=1
        )
        assert state.running
        assert loop.time() - started < 20
        assert state.out_bytes > 0, "it should already have printed something"

        # The point of the whole design: it is still going a moment later.
        await asyncio.sleep(1)
        later = await _probe(backend, record)
        assert later.running
        assert later.out_bytes > state.out_bytes

    @pytest.mark.asyncio
    async def test_a_forking_command_does_not_hold_the_call(self, backend: LocalBackend) -> None:
        """`communicate()` returns when the pipes close, not when the child
        exits, so a background child that inherited them blocks the caller for
        its whole life. Redirecting to files is what breaks that grip — this is
        the measurement the design rests on."""
        loop = asyncio.get_running_loop()
        started = loop.time()
        _, state = await _launch(backend, "sleep 40 & wait", yield_s=0.15)
        assert state.running
        assert loop.time() - started < 20


class TestFollowingIt:
    @pytest.mark.asyncio
    async def test_a_probe_only_brings_back_what_is_new(self, backend: LocalBackend) -> None:
        record, first = await _launch(
            backend, "for i in $(seq 1 60); do echo line $i; sleep 0.2; done", yield_s=1
        )
        record.cursor = first.out_bytes
        # Blank lines are dropped from both sides. An empty chunk splits into
        # `{""}`, which is a *truthy* set: the loop below would take it for
        # something new, and the empty string matches the empty string on the
        # other side. That is the whole failure, and it only shows up under load.
        already_seen = {line for line in first.out.splitlines() if line.strip()}

        # Polled until something new arrives rather than sleeping a fixed second
        # and assuming it did. The invariant under test is that the cursor does
        # not re-send; *when* the next line appears is incidental, and on a busy
        # machine — a parallel test run, CI — a second buys nothing at all. That
        # is how this failed the first time it ran alongside thirteen others.
        fresh: set[str] = set()
        for _ in range(40):
            await asyncio.sleep(0.5)
            state = await _probe(backend, record)
            record.cursor = state.out_bytes
            fresh = {line for line in state.out.splitlines() if line.strip()}
            if fresh:
                break
        assert fresh, "the command printed nothing new at all"
        # Compared as whole lines: "line 1" is a substring of "line 10", and an
        # `in` check here passes for the wrong reason.
        assert not (fresh & already_seen), "the cursor re-sent lines already shown"

    @pytest.mark.asyncio
    async def test_silence_is_visible_as_silence(self, backend: LocalBackend) -> None:
        """ "It has printed nothing for ten minutes" is the judgement the model
        has to be able to make, and it cannot be made from the text alone."""
        record, _ = await _launch(backend, "echo start; sleep 30", yield_s=0.15)

        async def gone_quiet() -> bool:
            return (await _probe(backend, record)).idle_s >= 1

        assert await _until(gone_quiet), "time since the last byte must be reported"
        state = await _probe(backend, record)
        assert state.running
        assert state.elapsed_s >= state.idle_s


class TestStoppingIt:
    @pytest.mark.asyncio
    async def test_it_takes_the_grandchildren_too(self, backend: LocalBackend) -> None:
        """A benchmark forks workers. Killing the wrapper alone leaves them
        burning CPU with nobody watching."""
        record, _ = await _launch(
            backend, "( while true; do echo tick; sleep 0.2; done ) & wait", yield_s=1
        )
        await backend.run(processes.build_kill(record, grace=0), timeout=60)
        await asyncio.sleep(0.5)
        state = await _probe(backend, record)
        assert not state.running

        before = state.out_bytes
        await asyncio.sleep(1)
        assert (await _probe(backend, record)).out_bytes == before, "a worker survived"

    @pytest.mark.asyncio
    async def test_a_stopped_command_never_reports_success(self, backend: LocalBackend) -> None:
        """The most dangerous defect found while designing this.

        A wrapper asked for its own exit status while being signalled reports
        whatever it happened to be doing — `exited 0` was measured, twice, with
        and without a TERM trap. A model reading that would tell the user their
        benchmark completed successfully. So the fact that it was stopped is
        recorded by whoever stopped it, and never inferred."""
        record, _ = await _launch(
            backend, "( while true; do echo tick; sleep 0.2; done ) & wait", yield_s=1
        )
        await backend.run(processes.build_kill(record, grace=0), timeout=60)
        await asyncio.sleep(0.5)
        state = await _probe(backend, record)
        assert state.killed is True
        assert "stopped" in processes.describe(state, following=True)

    @pytest.mark.asyncio
    async def test_stopping_something_already_finished_is_not_an_error(
        self, backend: LocalBackend
    ) -> None:
        record, _ = await _launch(backend, "echo done", yield_s=10)
        result = await backend.run(processes.build_kill(record, grace=0), timeout=60)
        assert "not running" in result.stdout


class TestTheWatchdog:
    @pytest.mark.asyncio
    async def test_it_ends_the_process_without_anyone_watching(self, backend: LocalBackend) -> None:
        """The backstop for the guarantee that detaching gives up.

        It has to live inside the process group, because the only thing
        guaranteed to be alive when it fires is the process itself — the CLI may
        have been closed hours earlier. Two failures are baked into its shape:
        the watchdog must ignore its own TERM or it dies first and never
        escalates, and the escalation to KILL is not politeness — a bash loop
        that receives TERM while waiting on a child simply carries on."""
        record, state = await _launch(
            backend,
            "while true; do echo tick; sleep 0.2; done",
            yield_s=0.15,
            max_lifetime=1,
            grace=0,
        )
        assert state.running
        assert await _until(lambda: _finished(backend, record)), "the watchdog never fired"


class TestTheRegistry:
    def test_a_record_survives_a_round_trip_through_json(self) -> None:
        """It is stored in `InteractionContext.state`, which the runner
        serialises with `json.dumps`, whatever store is behind it."""
        import json

        state: dict = {}
        processes.remember(state, processes.new_record("echo hi"))
        assert processes.registry(json.loads(json.dumps(state)))

    def test_an_unreadable_record_is_dropped_not_raised(self) -> None:
        """Following `CompactionRecord`: losing one costs the ability to follow
        that process, whose files are still on disk. Raising would make a
        resumed conversation unusable."""
        state = {
            "background_processes": {"good": processes.new_record("x").model_dump(), "bad": 42}
        }
        assert list(processes.registry(state)) == ["good"]

    def test_no_output_text_is_ever_stored(self) -> None:
        """`state` is rewritten after every event, so anything kept here is
        re-serialised continuously. The log is a file; this only says where."""
        record = processes.new_record("echo hi")
        assert "out" not in record.model_dump()
        assert set(record.model_dump()) <= {
            "handle",
            "command",
            "conversation",
            "finished",
            "started_at",
            "cursor",
            "err_cursor",
        }


class TestKeepingItReadable:
    def test_repeated_lines_are_folded(self) -> None:
        folded = processes.collapse_repeats("\n".join(["retrying"] * 4000 + ["done"]))
        assert "x 4000" in folded
        assert len(folded) < 100

    def test_an_increment_is_clipped_from_the_end(self) -> None:
        """Where a run is *now* is at the end; the start of an increment is of
        no interest."""
        text, dropped = processes.clip("A" * 100 + "B" * 100, limit=100, tail_only=True)
        assert text == "B" * 100
        assert dropped == 100

    def test_a_finished_command_keeps_both_ends(self) -> None:
        text, _ = processes.clip("START" + "x" * 5000 + "END", limit=100, tail_only=False)
        assert text.startswith("START")
        assert text.endswith("END")

    @pytest.mark.asyncio
    async def test_one_observation_stays_small(self, backend: LocalBackend) -> None:
        """Following a long run must not be the thing that fills the context
        window and triggers a compaction."""
        record, _ = await _launch(
            backend, "for i in $(seq 1 5000); do echo line $i of noise; done; sleep 30", yield_s=2
        )
        text = processes.describe(await _probe(backend, record), following=True)
        assert len(text) < processes.POLL_CHARS * 2


class TestTheToolsTogether:
    @pytest.mark.asyncio
    async def test_start_follow_and_stop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The grace between TERM and KILL is five seconds of waiting that this
        # test does not exercise; `TestStoppingIt` does.
        monkeypatch.setattr(processes, "DEFAULT_GRACE_S", 0)
        tools = {t.name: t for t in bundle("shell", root=tmp_path)}
        state: dict = {}

        started = await tools["run_command"].execute(
            command="for i in $(seq 1 60); do echo line $i; sleep 0.3; done",
            timeout=1,
            state=state,
        )
        assert "still running" in str(started.data)
        state = started.state or {}
        handle = next(iter(processes.registry(state)))

        await asyncio.sleep(0.4)
        checked = await _call(tools["check_command"], handle=handle, state=state)
        assert "running" in str(checked.data)

        stopped = await tools["stop_command"].execute(handle=handle, state=checked.state or {})
        assert "Stopped" in str(stopped.data)

    @pytest.mark.asyncio
    async def test_an_unknown_handle_says_what_is_there(self, tmp_path: Path) -> None:
        tools = {t.name: t for t in bundle("shell", root=tmp_path)}
        out = await _call(tools["check_command"], handle="nope", state={})
        assert "nope" in str(out.data)

    @pytest.mark.asyncio
    async def test_too_many_at_once_is_refused_with_a_way_out(self, tmp_path: Path) -> None:
        """The cap exists so a conversation cannot leak processes. Refusing
        without naming them would leave the model unable to act on it."""
        tools = {t.name: t for t in bundle("shell", root=tmp_path)}
        # Started together: they are independent, and eight yields in a row is
        # eight seconds spent proving nothing.
        results = await asyncio.gather(
            *(
                tools["run_command"].execute(command="sleep 30", timeout=1, state={})
                for _ in range(processes.MAX_PROCESSES)
            )
        )
        state: dict = {}
        for result in results:
            for record in processes.registry(result.state).values():
                processes.remember(state, record)

        refused = await tools["run_command"].execute(command="sleep 30", timeout=1, state=state)
        assert "which is the limit" in str(refused.data)
        assert "stop_command" in str(refused.data), "it has to say how to get out of it"
        assert "sleep 30" in str(refused.data), "and which ones, by what they run"

    @pytest.mark.asyncio
    async def test_a_conversation_that_never_looked_is_not_wedged(self, tmp_path: Path) -> None:
        """Nothing marks a command finished on its own, so the live count is only
        as fresh as the last time anybody checked. A conversation that started
        eight and never looked believed all eight were running and refused to
        start another — for good, since the thing that would have unstuck it was
        the very call being refused.

        The fix is not checking more often. It is checking *before refusing*: one
        sweep, on the path that was about to fail."""
        tools = {t.name: t for t in bundle("shell", root=tmp_path)}

        # Started together rather than one after another: each launch waits out
        # its own yield, and eight of those in a row is eight seconds spent
        # proving nothing. They are independent, so they can overlap.
        results = await asyncio.gather(
            *(
                # Longer than the yield, or it finishes inside the window and
                # leaves no record at all — which is the state this test needs.
                tools["run_command"].execute(
                    command="sleep 2", timeout=1, state={}, conversation_id="c1"
                )
                for _ in range(processes.MAX_PROCESSES)
            )
        )
        state: dict = {}
        for result in results:
            for record in processes.registry(result.state).values():
                processes.remember(state, record)
        assert len([r for r in processes.registry(state).values() if not r.finished]) == (
            processes.MAX_PROCESSES
        )

        # Wait for them to end, with nothing noticing that they have.
        watcher = LocalBackend(Workspace.at(tmp_path), isolate_environment=False)
        last = list(processes.registry(state).values())[-1]
        assert await _until(lambda: _finished(watcher, last))

        allowed = await tools["run_command"].execute(
            command="echo through", timeout=10, state=state, conversation_id="c1"
        )
        assert "through" in str(allowed.data)

    def test_finished_records_do_not_pile_up_forever(self) -> None:
        """`state` is re-serialised after every event of every turn, so an
        unbounded list of records is a cost paid continuously for something
        nobody will read again."""
        state: dict = {}
        for i in range(processes.MAX_RECORDS * 2):
            record = processes.new_record(f"cmd {i}", "c1")
            record.finished = True
            processes.remember(state, record)
        processes.prune(state)
        assert len(processes.registry(state)) == processes.MAX_RECORDS

    def test_a_running_command_is_never_dropped(self) -> None:
        """Losing a live record orphans a running process — the failure this
        whole module exists to avoid."""
        state: dict = {}
        alive = processes.new_record("the important one", "c1")
        processes.remember(state, alive)
        for i in range(processes.MAX_RECORDS * 2):
            done = processes.new_record(f"cmd {i}", "c1")
            done.finished = True
            processes.remember(state, done)
        processes.prune(state)
        assert alive.handle in processes.registry(state)

    @pytest.mark.asyncio
    async def test_only_starting_needs_permission(self, tmp_path: Path) -> None:
        """Watching reads a file and stopping can only reach what this
        conversation started. Asking permission for either would be asking
        permission to notice."""
        tools = {t.name: t for t in bundle("shell", root=tmp_path)}
        assert tools["run_command"].approval_type is not None
        assert tools["check_command"].approval_type is None
        assert tools["stop_command"].approval_type is None


def _whole_lines(out: str) -> set[str]:
    """The lines a report returned in full. A report is cut by bytes, so either
    end of it can be a piece of a line (`7` of `line 84757`), and two pieces can
    match without anything having been read twice."""
    return {line for line in out.split("\n")[:-1] if re.fullmatch(r"line \d+", line)}


class TestTheCursorUnderLoad:
    @pytest.mark.asyncio
    async def test_a_busy_process_is_never_re_read(self, backend: LocalBackend) -> None:
        """The race that only showed up in a full test run.

        The size is measured before the content is read, and a running command
        keeps writing in between — so more came back than had been counted, the
        cursor advanced to the smaller number, and the next poll repeated lines
        it had already shown. Every read is now bounded by the measured size.

        Written with no sleep between polls and a command that writes as fast as
        it can, because that is the only way the window is wide enough to catch.
        """
        record, first = await _launch(
            backend, "for i in $(seq 1 200000); do echo line $i; done; sleep 5", yield_s=1
        )
        record.cursor = first.out_bytes
        seen = _whole_lines(first.out)

        for _ in range(4):
            state = await _probe(backend, record)
            fresh = _whole_lines(state.out)
            assert not (fresh & seen), "a poll repeated lines it had already shown"
            seen |= fresh
            record.cursor = state.out_bytes


class TestWaitingForIt:
    """One call that waits, instead of a model polling in a loop.

    A poll loop costs a turn per look and re-reads the conversation each time;
    two minutes of watching would be dozens of turns. Waiting happens in the
    shell, on the machine the process is on, which is also why it works
    unchanged against a pod behind a one-shot endpoint.
    """

    @pytest.mark.asyncio
    async def test_it_returns_the_moment_the_command_ends(self, backend: LocalBackend) -> None:
        record, _ = await _launch(backend, "sleep 2", yield_s=0.5)
        loop = asyncio.get_running_loop()
        started = loop.time()
        out = await backend.run(processes.build_probe(record, wait_s=60), timeout=120)
        elapsed = loop.time() - started
        assert not processes.parse_report(out.stdout, record, from_cursor=True).running
        assert elapsed < 20, f"waited {elapsed:.0f}s for something that took 2"

    @pytest.mark.asyncio
    async def test_it_gives_up_at_the_deadline(self, backend: LocalBackend) -> None:
        record, _ = await _launch(backend, "sleep 60", yield_s=0.5)
        state = processes.parse_report(
            (await backend.run(processes.build_probe(record, wait_s=1), timeout=60)).stdout,
            record,
            from_cursor=True,
        )
        assert state.running, "it should still be going when the wait runs out"

    def test_a_single_wait_is_bounded(self) -> None:
        """Not by what would be useful but by what the transport tolerates: in
        a remote backend the call is an HTTP request, and one held open for
        several minutes is one something will drop."""
        allowed, _ = processes.wait_allowance({}, 10_000)
        assert allowed == processes.MAX_WAIT_S

    def test_the_conversation_has_a_total_budget(self) -> None:
        """The per-call bound alone does not stop a model asking for another
        sixty seconds forty times over."""
        state: dict = {}
        spent = 0
        while True:
            allowed, _ = processes.wait_allowance(state, processes.MAX_WAIT_S)
            if allowed == 0:
                break
            processes.spend_wait(state, allowed)
            spent += allowed
        assert spent == processes.WAIT_BUDGET_S

    def test_the_budget_survives_the_turn_that_spent_it(self) -> None:
        import json

        state: dict = {}
        processes.spend_wait(state, 30)
        assert processes.wait_allowance(json.loads(json.dumps(state)), 60)[1] == (
            processes.WAIT_BUDGET_S - 30
        )


class TestGettingAHandleBack:
    """A handle reaches the model in a tool reply, and tool replies are pruned
    when the conversation is compacted. Without a way back in, a running process
    becomes unreachable while its record sits in conversation state, intact and
    unread — the model can neither follow it nor stop it."""

    @pytest.mark.asyncio
    async def test_the_handles_are_recoverable_from_state_alone(self, tmp_path: Path) -> None:
        tools = {t.name: t for t in bundle("shell", root=tmp_path)}
        started = await tools["run_command"].execute(
            command="sleep 30", timeout=1, state={}, conversation_id="c1"
        )
        handle = next(iter(processes.registry(started.state)))

        # The reply that carried the handle is gone; only the state remains.
        listed = await tools["list_commands"].execute(state=started.state, conversation_id="c1")

        assert handle in str(listed.data)
        assert "sleep 30" in str(listed.data), "the command is how a person recognises it"

    @pytest.mark.asyncio
    async def test_it_says_so_when_there_is_nothing(self, tmp_path: Path) -> None:
        tools = {t.name: t for t in bundle("shell", root=tmp_path)}
        out = await tools["list_commands"].execute(state={}, conversation_id="c1")
        assert "Nothing running" in str(out.data)


class TestAskingForTheBackground:
    """Asking to "run it in the background" used to have no answer but `&`, which detaches
    the command from everything that could follow or stop it."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("flag", [True, "true"])
    async def test_it_hands_back_a_handle_at_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: object
    ) -> None:
        monkeypatch.setattr(processes, "DEFAULT_GRACE_S", 0)
        tools = {t.name: t for t in bundle("shell", root=tmp_path)}

        began = time.monotonic()
        started = await tools["run_command"].execute(command="sleep 30", background=flag, state={})
        elapsed = time.monotonic() - began

        assert elapsed < 10, "background must not wait the default timeout"
        assert "still running" in str(started.data)
        handle = next(iter(processes.registry(started.state)))
        await tools["stop_command"].execute(handle=handle, state=started.state)

    def test_the_model_is_offered_it(self, tmp_path: Path) -> None:
        """Every other test here passes `timeout=1`, which a model asked to start
        a server does not: it sends no timeout, and the default kept the session
        waiting two minutes before the handle came back. What a model can choose
        is what its schema and description offer, so that is what is pinned."""
        tool = {t.name: t for t in bundle("shell", root=tmp_path)}["run_command"]
        schema = tool.invocation_json_schema

        assert "background" in schema["properties"]["arguments"]["properties"]
        assert "background=true" in schema["description"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("command", ["true &", "true & true", "true &\ntrue"])
    async def test_a_detaching_line_is_called_out(self, tmp_path: Path, command: str) -> None:
        """Not only a trailing `&`: `server & sleep 1; curl …` is the shape a
        model actually writes, and it leaves the server just as untracked."""
        tools = {t.name: t for t in bundle("shell", root=tmp_path)}
        out = await tools["run_command"].execute(command=command, state={})

        assert "not tracked" in str(out.data)
        assert "background=true" in str(out.data)
        assert not processes.registry(out.state)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("command", ["true && true", "echo disown", "true & wait"])
    async def test_a_line_that_detaches_nothing_is_not(self, tmp_path: Path, command: str) -> None:
        """The note tells the model to stop a process; said of a line that left
        none behind, it sends it looking for something that does not exist."""
        tools = {t.name: t for t in bundle("shell", root=tmp_path)}
        out = await tools["run_command"].execute(command=command, state={})

        assert "not tracked" not in str(out.data)


class TestWhatCountsAsDetaching:
    """`_detaches` on its own, against the shapes the tokenizer has to tell apart."""

    @pytest.mark.parametrize(
        "command",
        [
            "server &",
            "server &  \n",
            "server & sleep 1; curl localhost:8000",
            "server &\ncurl localhost:8000",
            "a & b",
            "nohup server > log 2>&1 &",
            "server 2>&1 & echo started",
            "server & disown",
            "disown -a",
            "X=1 disown",
            "a & wait; b &",  # the wait joins `a`, not what comes after it
            "a & wait &",  # a backgrounded wait joins nothing
            "setsid -f server",
            "setsid --fork server",
            "setsid -cf server",
            "(sleep 1; true) &",  # a subshell: the line's end still says it
        ],
    )
    def test_it_detaches(self, command: str) -> None:
        assert shell._detaches(command)

    @pytest.mark.parametrize(
        "command",
        [
            "true",
            "a && b",
            "a || b",
            "a | b",
            "a |& b",
            "make 2>&1",
            "make &> log",
            "make &>/dev/null",
            "echo hi >&2",
            "read x <&3",
            "echo disown",
            "grep -rn disown src/",
            'git commit -m "do not disown it"',
            "echo 'a & b'",
            'echo "trailing &"',
            "ls # then run it with &",
            "a & b & wait",
            "a &\nb &\nwait",
            "setsid server",
            "setsid -f -w server",
            "setsid --fork --wait server",
            "setsid -- -f",
            "cat <<'EOF'\nserver &\nEOF",
        ],
    )
    def test_it_does_not(self, command: str) -> None:
        assert not shell._detaches(command)

    def test_a_subshell_hides_what_it_runs(self) -> None:
        """The tokenizer gives up on `(…)`, so only the line's end is read: an `&`
        inside one goes unnoticed. Pinned so a change here is a decision."""
        assert not shell._detaches("(sleep 1 &)")


class TestOneConversationCannotSeeAnother:
    @pytest.mark.asyncio
    async def test_the_files_are_grouped_by_conversation(self, backend: LocalBackend) -> None:
        """Logs live in the workspace, which the file tools can read. Without a
        folder per conversation, every conversation's output is readable from
        every other one — and nothing a conversation left behind can be found,
        or removed, as a unit."""
        mine = processes.new_record("echo mine", "conversation-a")
        theirs = processes.new_record("echo theirs", "conversation-b")
        assert mine.dir != theirs.dir
        assert "conversation-a" in mine.dir
        assert not mine.dir.startswith(theirs.dir)

    def test_an_id_that_would_escape_the_folder_is_flattened(self) -> None:
        """Conversation ids arrive from callers — a gateway, a resumed session —
        and one containing `../` would otherwise place logs anywhere on disk."""
        record = processes.new_record("echo hi", "../../etc")
        assert ".." not in record.dir


class TestNoCallBlocksForever:
    """Two holes that only showed once `wait` existed: the model can choose how
    long a call blocks in *two* places, and only one of them was bounded."""

    def test_the_initial_yield_is_capped_too(self) -> None:
        """Without this, `run_command(..., timeout=7200)` blocks for two hours
        and walks straight past the waiting budget, which only ever saw
        `check_command`. The pod's own ceiling would catch it there; nothing
        would catch it locally."""
        assert processes.block_seconds(7200, 120) == processes.MAX_BLOCK_S

    def test_an_unset_timeout_still_gets_the_default(self) -> None:
        assert processes.block_seconds(0, 120) == 120

    def test_it_never_collapses_to_nothing(self) -> None:
        """A yield of zero would hand back a handle for `ls` instead of its
        output — a far worse failure than the one being prevented."""
        assert processes.block_seconds(-5, 0) >= 1

    def test_the_budget_charges_what_was_spent_not_what_was_asked(self) -> None:
        """A wait ends the moment the command does: waiting 300s for something
        that takes two costs two. Charging the allowance would let a handful of
        quick checks exhaust a budget they never used."""
        state: dict = {}
        allowed, _ = processes.wait_allowance(state, processes.MAX_WAIT_S)
        assert allowed == processes.MAX_WAIT_S
        processes.spend_wait(state, 2)  # it came back almost at once
        _, remaining = processes.wait_allowance(state, 10)
        assert remaining == processes.WAIT_BUDGET_S - 2

    @pytest.mark.asyncio
    async def test_a_quick_command_barely_touches_the_budget(self, tmp_path: Path) -> None:
        tools = {t.name: t for t in bundle("shell", root=tmp_path)}
        started = await tools["run_command"].execute(
            command="sleep 2", timeout=1, state={}, conversation_id="c1"
        )
        state = started.state or {}
        handle = next(iter(processes.registry(state)))

        checked = await _call(
            tools["check_command"],
            handle=handle,
            wait=processes.MAX_WAIT_S,
            state=state,
            conversation_id="c1",
        )

        _, remaining = processes.wait_allowance(checked.state, 1)
        assert remaining > processes.WAIT_BUDGET_S - 30, "it was charged for waiting it never did"


class TestHowAProcessIsNamed:
    """A handle is an internal identifier — a pid by another name.

    Leading with it taught the model to lead with it too, and people were shown
    tables of hex strings to describe "the benchmark". What identifies a command
    to a person is what it runs.
    """

    def test_the_command_comes_first(self) -> None:
        state = processes.ProcessState(handle="deadbeef", running=True, elapsed_s=90)
        text = processes.describe(state, following=False, command="python bench.py --full")
        assert text.startswith("`$ python bench.py --full`")

    def test_the_handle_is_still_there_and_labelled(self) -> None:
        """It is not noise to be removed: the model needs it to call these tools,
        and the person needs it to type /stop."""
        state = processes.ProcessState(handle="deadbeef", running=True)
        text = processes.describe(state, following=False, command="python bench.py")
        assert "deadbeef" in text
        assert "/stop" in text

    def test_without_a_command_it_falls_back_to_the_handle(self) -> None:
        state = processes.ProcessState(handle="deadbeef", running=True)
        assert processes.describe(state, following=False).startswith("`deadbeef`")

    def test_a_very_long_command_line_is_trimmed(self) -> None:
        state = processes.ProcessState(handle="deadbeef", running=True)
        text = processes.describe(state, following=False, command="x" * 500)
        assert len(text.splitlines()[0]) < 200


class TestWhatIsShownWhileWaiting:
    """A blocking call cannot say anything while it blocks, so a wait is chunked
    into probes and each one reports. Without that the person watching gets a
    spinner and nothing else for as long as the wait lasts.

    Only a wait the model asked for is chunked: it costs a round trip per chunk,
    and the first yield of `run_command` — which nearly every command finishes
    inside — stays a single call."""

    @pytest.mark.asyncio
    async def test_progress_reaches_the_client_before_the_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from domyn.agents.tools.base import ToolResult, UpdateToolResult
        from domyn.agents.tools.builtin import shell

        # A shorter reporting step. The behaviour is the same at any cadence,
        # and at the real one a wait short enough to be quick would contain a
        # single chunk — which reports nothing, because there is no "meanwhile".
        monkeypatch.setattr(shell, "PROGRESS_STEP_S", 1)

        tools = {t.name: t for t in bundle("shell", root=tmp_path)}
        started = await tools["run_command"].execute(
            command="for i in $(seq 1 60); do echo tick $i; sleep 0.5; done",
            timeout=1,
            state={},
            conversation_id="c1",
        )
        state = started.state or {}
        handle = next(iter(processes.registry(state)))

        updates, results = [], []
        async for item in tools["check_command"].execute(
            handle=handle,
            wait=2,
            state=state,
            conversation_id="c1",
        ):
            if isinstance(item, UpdateToolResult):
                updates.append(item)
            elif isinstance(item, ToolResult):
                results.append(item)

        assert updates, "nothing was reported while waiting"
        assert "tick" in updates[0].data["progress"]
        assert len(results) == 1, "exactly one final result, at the end"

    @pytest.mark.asyncio
    async def test_a_command_that_ends_at_once_reports_no_progress(self, tmp_path: Path) -> None:
        """Progress is for waiting. Something already finished has nothing to
        report along the way."""
        from domyn.agents.tools.base import UpdateToolResult

        tools = {t.name: t for t in bundle("shell", root=tmp_path)}
        started = await tools["run_command"].execute(
            command="sleep 2", timeout=1, state={}, conversation_id="c1"
        )
        state = started.state or {}
        handle = next(iter(processes.registry(state)))

        updates = [
            item
            async for item in tools["check_command"].execute(
                handle=handle, wait=30, state=state, conversation_id="c1"
            )
            if isinstance(item, UpdateToolResult)
        ]
        assert not updates

    def test_the_progress_line_is_the_last_thing_printed(self) -> None:
        """One line, not a screenful, and never a blank one — a run that just
        printed a separator has not stopped talking."""
        assert processes.tail_line("epoch 1\nepoch 2\n\n  \n") == "epoch 2"
        assert processes.tail_line("") == ""


class TestTheYieldIsHonoured:
    @pytest.mark.asyncio
    async def test_a_short_yield_is_not_rounded_up(self, backend: LocalBackend) -> None:
        """The wait is fine-grained at first and coarse afterwards, and the fine
        phase used to run to its own length regardless of what was asked — so a
        yield of one second waited two. Found by a `sleep 2` asked to yield after
        1s: it ran to completion inside the wait and never produced a handle."""
        loop = asyncio.get_running_loop()
        started = loop.time()
        _, state = await _launch(backend, "sleep 30", yield_s=0.5)
        assert state.running
        assert loop.time() - started < 1.5, "it waited longer than it was asked to"


class TestWhenTheWaitingIsGone:
    """What a refusal has to look like.

    Found in manual use: the model kept asking to wait and kept being answered
    instantly, and from outside there was no reason for it. The budget had run
    out and the explanation was the last line of a long reply, under the
    command's output, where it reads as a footnote."""

    @pytest.mark.asyncio
    async def test_the_refusal_comes_before_everything_else(self, tmp_path: Path) -> None:
        from domyn.agents.tools import ToolResult

        tools = {t.name: t for t in bundle("shell", root=tmp_path)}
        started = await tools["run_command"].execute(
            command="sleep 60", timeout=1, state={}, conversation_id="c1"
        )
        state = started.state or {}
        handle = next(iter(processes.registry(state)))
        state[processes.WAIT_BUDGET_KEY] = processes.WAIT_BUDGET_S

        text = ""
        async for item in tools["check_command"].execute(
            handle=handle, wait=45, state=state, conversation_id="c1"
        ):
            if isinstance(item, ToolResult):
                text = str(item.data)

        first = text.splitlines()[0]
        assert "no waiting was done" in first
        assert "Asking again will not wait" in text, "it has to stop the model retrying"

    def test_there_is_enough_of_it_for_a_real_run(self) -> None:
        """Ten minutes was the first guess and too tight: one five-minute run,
        watched twice, used it up."""
        assert processes.WAIT_BUDGET_S >= 2 * 15 * 60 / 2


class TestTheFilesystemIsTheTruth:
    """`state` is a cache of what the machine already knows.

    Each process has a directory holding its pid, its command and its log, so a
    listing answers "what did this conversation leave running" without being
    told. That matters because a process outlives the session that started it:
    anything that loses the state — a crash before the turn was persisted, a
    store trimmed, a front-end keeping its own — used to orphan it permanently.
    """

    @pytest.mark.asyncio
    async def test_processes_are_found_with_no_state_at_all(self, backend: LocalBackend) -> None:
        for command in ("sleep 30", "echo done"):
            record = processes.new_record(command, "conv-x")
            await backend.run(processes.build_launch(record, yield_s=0.15), timeout=60)

        found = processes.parse_discover(
            (await backend.run(processes.build_discover("conv-x"), timeout=60)).stdout
        )

        assert len(found) == 2
        assert {v["command"] for v in found.values()} == {"sleep 30", "echo done"}
        assert sum(1 for v in found.values() if v["running"]) == 1

    @pytest.mark.asyncio
    async def test_it_does_not_see_another_conversation(self, backend: LocalBackend) -> None:
        await backend.run(
            processes.build_launch(processes.new_record("sleep 30", "mine"), yield_s=0.15),
            timeout=60,
        )
        found = processes.parse_discover(
            (await backend.run(processes.build_discover("theirs"), timeout=60)).stdout
        )
        assert found == {}

    @pytest.mark.asyncio
    async def test_the_model_can_recover_a_process_after_losing_its_state(
        self, tmp_path: Path
    ) -> None:
        tools = {t.name: t for t in bundle("shell", root=tmp_path)}
        await tools["run_command"].execute(
            command="sleep 30", timeout=1, state={}, conversation_id="c1"
        )

        # The state is gone entirely — not merely missing this handle.
        listed = await tools["list_commands"].execute(state={}, conversation_id="c1")

        assert "sleep 30" in str(listed.data)


class TestTheLogCannotGrowForever:
    """A run nobody is watching must not be able to fill a disk.

    The cap is enforced from inside the process group, like the lifetime limit
    and for the same reason: when it matters, nothing outside is alive. The head
    is copied aside once — it cannot change, so there is nothing to race — and
    only the tail is rewritten.
    """

    @pytest.fixture
    def small_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Real sizes would need minutes and megabytes to reach. The behaviour
        is the same at any scale; the numbers are not the point."""
        monkeypatch.setattr(processes, "LOG_COMPACT_AT", 200_000)
        monkeypatch.setattr(processes, "LOG_HEAD_BYTES", 20_000)
        monkeypatch.setattr(processes, "LOG_TAIL_BYTES", 50_000)
        monkeypatch.setattr(processes, "LOG_CHECK_EVERY_S", 1)

    @staticmethod
    def _noisy(linger: bool = True) -> str:
        """3.4 MB as fast as the shell can write it.

        ``linger`` keeps the process alive afterwards, for the tests that need
        something still running. The ones that only care about what was kept let
        it end instead — waiting out a sleep they do not need was most of this
        class's running time.
        """
        writing = "i=0; while [ $i -lt 300000 ]; do echo riga-$i; i=$((i+1)); done"
        return f"{writing}; sleep 20" if linger else writing

    @pytest.mark.asyncio
    async def test_it_is_brought_back_under_the_cap(
        self, backend: LocalBackend, tmp_path: Path, small_cap: None
    ) -> None:
        """ "Brought back under", not "never above": the compactor runs on a
        timer, and a command writing as fast as it can overshoots between two
        checks. What is guaranteed is that it does not keep growing — this one
        writes 3.4 MB into a 200 KB cap.

        Measured once the wrapper has exited, which is the only moment the
        question has a stable answer. Chasing the instant *between* two
        compactions of a command writing at full speed is chasing a window of
        milliseconds: it passed on an idle machine and failed under coverage,
        which is a fact about the test and not about the cap.

        The cap is on the live log. The head is set aside once and kept on top
        of it, so the files together may reach the cap plus the head: a command
        that ends just under the trigger is left alone, rightly, and counting
        the head against the cap failed on a loaded runner with ``out.log`` at
        198 KB.
        """
        record = processes.new_record(self._noisy(linger=False), "c1")
        await backend.run(processes.build_launch(record, yield_s=0.15), timeout=60)
        directory = tmp_path / record.dir

        async def settled() -> bool:
            return await _compacted(backend, record) and await _wrapped_up(backend, record)

        assert await _until(settled, timeout=40), "it never settled"

        log = (directory / "out.log").stat().st_size
        assert log <= processes.LOG_COMPACT_AT, f"{log} bytes in the live log"
        kept = sum(f.stat().st_size for f in directory.glob("out.log*"))
        # 64: what `out.log.cut` holds, a byte count and a newline.
        limit = processes.LOG_COMPACT_AT + processes.LOG_HEAD_BYTES + 64
        assert kept <= limit, f"{kept} bytes kept"

    @pytest.mark.asyncio
    async def test_both_ends_survive(
        self, backend: LocalBackend, tmp_path: Path, small_cap: None
    ) -> None:
        """The head says what the run was doing, the tail how it went. Keeping
        only one of them is the same as keeping neither."""
        record = processes.new_record(self._noisy(linger=False), "c1")
        await backend.run(processes.build_launch(record, yield_s=0.15), timeout=60)

        async def finished_and_compacted() -> bool:
            # The last line is only the last once the command has stopped
            # writing, and the log only stays put once the wrapper's last
            # compaction is over: it rewrites the file, which is empty for a
            # moment in between.
            return await _compacted(backend, record) and await _wrapped_up(backend, record)

        assert await _until(finished_and_compacted, timeout=40), "it never settled"

        raw = (await backend.run(processes._report(record, from_cursor=False), timeout=60)).stdout
        state = processes.parse_report(raw, record, from_cursor=False)
        lines = [line for line in state.out.splitlines() if line.strip()]

        assert lines[0].startswith("riga-0")
        assert lines[-1].startswith("riga-299999")
        assert any("dropped" in line for line in lines), "the gap has to be named"
        # Two compactions at once counted the same bytes twice.
        written = sum(len(f"riga-{i}\n") for i in range(300_000))
        assert 0 < state.omitted_bytes < written

    @pytest.mark.asyncio
    async def test_a_cursor_past_the_end_is_reset_not_left_blind(
        self, backend: LocalBackend, small_cap: None
    ) -> None:
        """A cursor is an offset into the log, and a compaction makes the log
        shorter. Left alone, the caller points past the end and reads nothing at
        all — silently, for the rest of the run."""
        record = processes.new_record(self._noisy(), "c1")
        await backend.run(processes.build_launch(record, yield_s=0.15), timeout=60)
        assert await _until(lambda: _compacted(backend, record)), "it never compacted"

        record.cursor = 10_000_000
        state = await _probe(backend, record)
        assert state.out.strip(), "it went blind after the log was compacted"

    @pytest.mark.asyncio
    async def test_the_helpers_do_not_outlive_the_command(
        self, backend: LocalBackend, small_cap: None
    ) -> None:
        """The watchdog sleeps for a whole day and the compactor loops forever.
        Left running, every command ever started leaves a pair behind."""
        record = processes.new_record("echo quick", "c1")
        await backend.run(processes.build_launch(record, yield_s=10), timeout=60)
        # One step of the helpers' own cadence, plus slack: they notice the
        # command has finished and leave on their own.
        await asyncio.sleep(processes.LOG_CHECK_EVERY_S + 2)

        pgid = int(
            (await backend.run(f"cat {record.dir}/run.pgid", timeout=30)).stdout.strip() or 0
        )
        survivors = (await backend.run(f"pgrep -g {pgid} | wc -l", timeout=30)).stdout.strip()
        assert survivors == "0", f"{survivors} helpers left behind"


class TestTheHelpersLeaveWhenTheirDirectoryCannotBeWritten:
    """The watchdog and the compactor learnt that the command was over from one
    file, ``run.exit``, in a directory that can be removed under them: pytest
    removes a test's three sessions later — any session of any project on the
    machine — and ``HOME`` is meant to be removed as a unit. Once it was gone,
    the file could never appear. The watchdog waited out its day, and the
    compactor's lock, a ``mkdir`` in that directory, could never be taken: it
    retried ten times a second, for the whole of that day.

    A directory still in place can refuse that ``mkdir`` as well, on a full disk
    or a read-only file system. On a full disk it was worse: the exit file is
    created empty, so the helpers left, and the wrapper retried on its own with
    nothing left to stop it, for as long as the disk stayed full.
    """

    @pytest.fixture
    def workspace(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Its own directory, so it can be removed without taking the test's.

        The helpers' cadence is shortened, but kept longer than it takes to
        remove the directory: they have to be asleep when it goes, as they were.
        """
        monkeypatch.setattr(processes, "LOG_CHECK_EVERY_S", 3)
        root = tmp_path / "workspace"
        root.mkdir()
        return root

    @staticmethod
    async def _group(directory: Path) -> int:
        """The group, once the wrapper has written it down."""
        path = directory / "run.pgid"

        async def written() -> bool:
            return path.exists() and path.read_text().strip() != ""

        assert await _until(written), "the wrapper never recorded its group"
        return int(path.read_text())

    @staticmethod
    async def _emptied(pgid: int) -> bool:
        async def gone() -> bool:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return True
            return False

        return await _until(gone, timeout=processes.LOG_CHECK_EVERY_S + 5)

    @pytest.mark.asyncio
    async def test_removed_once_the_command_has_finished(self, workspace: Path) -> None:
        """What was found: the command over, the wrapper gone, the helpers
        asleep between two checks when the directory went."""
        backend = LocalBackend(Workspace.at(workspace), isolate_environment=False)
        record = processes.new_record("echo quick", "c1")
        await backend.run(processes.build_launch(record, yield_s=10), timeout=60)
        pgid = await self._group(workspace / record.dir)
        try:
            assert await _until(lambda: _wrapped_up(backend, record))
            shutil.rmtree(workspace)
            assert await self._emptied(pgid), "helpers left behind"
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)

    @pytest.mark.asyncio
    async def test_removed_while_the_command_is_still_running(self, workspace: Path) -> None:
        """Then the exit code cannot be written either, and the wrapper's own
        last compaction waited for that same lock: the wrapper spun too, and
        kept the helpers believing the command was still going."""
        backend = LocalBackend(Workspace.at(workspace), isolate_environment=False)
        record, state = await _launch(backend, "sleep 2")
        assert state.running
        pgid = await self._group(workspace / record.dir)
        try:
            shutil.rmtree(workspace)
            assert await self._emptied(pgid), "the wrapper or its helpers left behind"
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)

    @pytest.mark.asyncio
    async def test_still_there_but_unwritable(self, workspace: Path) -> None:
        """The lock's ``mkdir`` fails with the directory in place: taking away its
        write permission stands in for a full disk or a read-only file system.
        The wrapper's last compaction then waited for good, and with no exit code
        written the helpers took the command for still running. Only a lock
        another pass holds is worth waiting for."""
        if os.geteuid() == 0:
            pytest.skip("root creates files in a directory without write permission")
        backend = LocalBackend(Workspace.at(workspace), isolate_environment=False)
        record, state = await _launch(backend, "sleep 2")
        assert state.running
        directory = workspace / record.dir
        pgid = await self._group(directory)
        try:
            directory.chmod(0o555)
            assert await self._emptied(pgid), "the wrapper or its helpers left behind"
        finally:
            directory.chmod(0o755)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)


class TestAShortNoisyCommandIsCompactedToo:
    @pytest.mark.asyncio
    async def test_it_is_trimmed_even_though_it_outran_the_timer(
        self, backend: LocalBackend, tmp_path: Path
    ) -> None:
        """The hole the timer alone leaves.

        The compactor checks on an interval and leaves when the command does, so
        a command that writes megabytes and finishes well inside one interval
        was never compacted at all — and at the real cadence that is most noisy
        commands. Found by a test, not by reasoning: the log was 1.7 MB inside a
        200 KB cap.

        No monkeypatched interval here on purpose: the point is precisely that
        the timer never gets a turn.
        """
        import pytest as _pytest

        with _pytest.MonkeyPatch.context() as patch:
            patch.setattr(processes, "LOG_COMPACT_AT", 200_000)
            patch.setattr(processes, "LOG_HEAD_BYTES", 20_000)
            patch.setattr(processes, "LOG_TAIL_BYTES", 50_000)
            record = processes.new_record(
                "i=0; while [ $i -lt 300000 ]; do echo riga-$i; i=$((i+1)); done", "c1"
            )
            await backend.run(processes.build_launch(record, yield_s=0.15), timeout=60)

        # Not `_finished`: it turns true before the wrapper's last compaction.
        assert await _until(lambda: _wrapped_up(backend, record), timeout=40)
        kept = sum(f.stat().st_size for f in (tmp_path / record.dir).glob("out.log*"))
        assert kept < 200_000, f"{kept} bytes kept by a command the timer never saw"


class TestTheLastCompactionIsNotGivenUp:
    """The wrapper's last ``compact_once`` is the only pass after the command
    ends: given up, it leaves what the command wrote at its end above the cap. It
    waits for a lock another pass holds, and takes one released between its
    failed ``mkdir`` and its look. The timer keeps its real interval here, so that
    it never takes the lock first.
    """

    NOISY = "yes riga | head -c 1000000"

    @pytest.fixture(autouse=True)
    def small_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(processes, "LOG_COMPACT_AT", 200_000)
        monkeypatch.setattr(processes, "LOG_HEAD_BYTES", 20_000)
        monkeypatch.setattr(processes, "LOG_TAIL_BYTES", 50_000)

    @pytest.mark.asyncio
    async def test_a_lock_released_between_its_two_looks_is_taken(
        self, backend: LocalBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``mkdir`` that fails once, with no lock there when it is looked for,
        is what another pass releasing the lock at that moment looks like. The
        pass was given up, and the whole log stayed."""
        real = shutil.which("mkdir")
        assert real is not None
        failed = tmp_path / "failed-once"
        shim = tmp_path / "bin"
        shim.mkdir()
        (shim / "mkdir").write_text(
            "#!/bin/sh\n"
            'for a in "$@"; do last=$a; done\n'
            'case "$last" in */compact.lock) '
            f"[ -e {shlex.quote(str(failed))} ] || {{ : > {shlex.quote(str(failed))}; exit 1; }} ;; "
            "esac\n"
            f'exec {shlex.quote(real)} "$@"\n'
        )
        (shim / "mkdir").chmod(0o755)
        monkeypatch.setenv("PATH", f"{shim}{os.pathsep}{os.environ['PATH']}")

        record = processes.new_record(self.NOISY, "c1")
        await backend.run(processes.build_launch(record, yield_s=10), timeout=60)
        assert await _until(lambda: _wrapped_up(backend, record))
        assert failed.exists(), "the lock's mkdir never went through the shim"
        assert await _compacted(backend, record), "the wrapper gave its last pass up"

    @pytest.mark.asyncio
    async def test_a_lock_another_pass_holds_is_waited_for(
        self, backend: LocalBackend, tmp_path: Path
    ) -> None:
        """What the lock is for: two passes at once write each other's half-read
        tail back. A pass that gave up whenever ``mkdir`` failed would skip the
        last one each time the timer's pass ran into the command's end. Held for a
        second, not for good: what is checked is the wait, then the pass once the
        lock comes free."""
        record = processes.new_record(self.NOISY, "c1")
        lock = tmp_path / record.dir / "compact.lock"
        lock.mkdir(parents=True)
        await backend.run(processes.build_launch(record, yield_s=10), timeout=60)
        assert await _until(lambda: _finished(backend, record))

        await asyncio.sleep(1)
        assert not await _wrapped_up(backend, record), "it gave up a lock another pass holds"
        assert not await _compacted(backend, record)

        lock.rmdir()
        assert await _until(lambda: _wrapped_up(backend, record))
        assert await _compacted(backend, record), "it never took the lock once it came free"


class TestAFinishedCommandIsFinished:
    @pytest.mark.asyncio
    async def test_the_wrapper_tidying_up_does_not_count(self, backend: LocalBackend) -> None:
        """The wrapper outlives the command by a moment: it writes the exit code
        and then compacts the log one last time. Asking only whether its pid is
        alive reported a finished command as running for that window — which
        showed up as a wait that "did not return when the command ended", one
        run in three.

        Which half is authoritative is not symmetric: an exit code means it is
        over, whatever else is still tidying up.
        """
        record, _ = await _launch(backend, "echo done", yield_s=0.1)

        # The instant after the command ends, without waiting for the wrapper.
        for _ in range(60):
            state = await _probe(backend, record)
            if state.exit_code is not None:
                assert not state.running, "an exited command reported as running"
                return
            await asyncio.sleep(0.05)
        raise AssertionError("it never recorded an exit code")


class TestEveryReadIsBoundedByTheCountItReports:
    """The invariant behind the cursor, checked where it lives.

    A caller advances its cursor to the byte count it was handed, so a report
    that returns *more* bytes than it counted guarantees a repeat: the extra
    lines were shown and are still ahead of the cursor. That happened — the tail
    was bounded and the head was not, so on a small file being written fast the
    head picked up lines that arrived after the count.

    Asserted against the generated shell rather than by running something,
    because the window is a few milliseconds wide: it never opened on a
    developer machine and opened in CI, where coverage slows the reader down.
    A behavioural test for it would pass for the wrong reason almost always.
    """

    def _report_for(self, from_cursor: bool) -> str:
        return processes._report(processes.new_record("x", "c1"), from_cursor=from_cursor)

    def test_the_live_log_is_never_read_unbounded(self) -> None:
        for from_cursor in (False, True):
            shell = self._report_for(from_cursor)
            # The bound appears as `$OUTB` or, for an incremental read, inside
            # `$((OUTB - CURSOR))` — what matters is that the measured size is
            # what limits it.
            for stream, size in (("out.log", "OUTB"), ("err.log", "ERRB")):
                # Every read of the live file passes through the measured size.
                # `.head` and `.cut` are different files and never change once
                # written, so they need no bound.
                reads = [
                    part
                    for part in shell.split(";")
                    # Reads of the *content*. `stat` and the `wc` that defines
                    # the bound touch the same file and are not reads of it.
                    if f"/{stream}" in part
                    and f"{stream}." not in part
                    and ("head -c" in part or "tail -c" in part)
                ]
                assert reads, f"expected the report to read {stream}"
                for part in reads:
                    assert size in part, f"unbounded read of {stream}: {part.strip()}"

    def test_the_size_is_measured_before_anything_is_read(self) -> None:
        """Measuring after would report a number the content had already
        outgrown, which is the same failure from the other end."""
        shell = self._report_for(from_cursor=False)
        assert shell.index("OUTB=") < shell.index("OUT_HEAD")

    @pytest.mark.asyncio
    async def test_and_the_reported_text_fits_the_count(self, backend: LocalBackend) -> None:
        """The same claim, end to end. Cheap, and it would catch a way of
        breaking this that the shell-shape check above would not."""
        _, state = await _launch(
            backend, "i=0; while [ $i -lt 4000 ]; do echo line $i; i=$((i+1)); done", yield_s=0.1
        )
        shown = len(state.out.encode("utf-8", "replace"))
        assert shown <= state.out_bytes, (
            f"reported {shown} bytes having counted {state.out_bytes} — "
            "a cursor set to the count would re-send the difference"
        )
