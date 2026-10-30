"""What the session shows about commands still running.

The view exists because a process nobody can see is a process the user finds out
about from the fan. So the things pinned here are: that it stays quiet when there
is nothing to say, that it never asks the model anything, and that it survives
whatever the conversation store hands back.
"""

import io

import pytest
from rich.console import Console

from domyn.agents.interfaces.cli import Theme, processes_view
from domyn.agents.tools.builtin import CommandResult, processes


@pytest.fixture
def console() -> Console:
    return Console(file=io.StringIO(), width=100, no_color=True)


@pytest.fixture
def theme() -> Theme:
    return Theme.build(allow_animation=False)


class _FakeBackend:
    """Answers a listing with a fixed verdict, so the view can be tested without
    starting anything.

    It is told which handles exist, because the real listing enumerates a
    directory and the command carries no handles at all — which is the point of
    it: the view asks the machine what is there rather than being told.
    """

    def __init__(
        self,
        *,
        running: bool = True,
        killed: bool = False,
        handles: dict[str, str] | None = None,
    ) -> None:
        self.running = running
        self.killed = killed
        self.handles = handles or {}
        self.commands: list[str] = []

    async def run(self, command: str, **_: object) -> CommandResult:
        self.commands.append(command)
        lines = []
        for handle, cmd in self.handles.items():
            lines.append(
                f"P {handle} {1 if self.running else 0} - {1 if self.killed else 0} 880 1000 970 42"
            )
            lines.append(f"C {handle} {cmd}")
        return CommandResult(exit_code=0, stdout="\n".join(lines), stderr="")


def _state_with(command: str = "pytest -q") -> dict:
    state: dict = {}
    processes.remember(state, processes.new_record(command))
    return state


def _backend_for(state: dict, **kwargs: object) -> _FakeBackend:
    """A backend that reports exactly the processes this state knows about."""
    return _FakeBackend(
        handles={h: r.command for h, r in processes.registry(state).items()},
        **kwargs,  # type: ignore[arg-type]
    )


class TestTheIndicatorStaysQuiet:
    def test_nothing_running_says_nothing(self) -> None:
        assert processes_view.summary({}) == ""

    def test_a_finished_command_is_not_counted(self) -> None:
        state = _state_with()
        for raw in state["background_processes"].values():
            raw["finished"] = True
        assert processes_view.summary(state) == ""

    def test_it_counts_what_is_live(self) -> None:
        assert processes_view.summary(_state_with()) == "1 command running"

    def test_it_never_probes(self) -> None:
        """Drawn next to the prompt, so it must not run a shell command. Being
        one turn stale is the right trade for something whose job is to remind."""
        state = _state_with()
        # No backend is passed at all: if it needed one, this could not work.
        assert processes_view.summary(state)

    def test_a_shape_it_does_not_recognise_is_not_a_crash(self) -> None:
        """The store is written by another process and read by front-ends that
        keep their own state; an unexpected shape means nothing to show."""
        assert processes_view.summary({"background_processes": "nonsense"}) == ""
        assert processes_view.summary(None) == ""


class TestListing:
    @pytest.mark.asyncio
    async def test_it_says_so_when_there_is_nothing(self, console: Console, theme: Theme) -> None:
        await processes_view.render(console, theme, {}, _FakeBackend())
        assert "No commands running" in console.file.getvalue()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_it_shows_the_command_and_where_its_log_is(
        self, console: Console, theme: Theme
    ) -> None:
        """Identifying a process by its handle alone asks the user to remember a
        hex string; the command line is what they actually recognise. The log
        path is there so it can be followed from another terminal."""
        state = _state_with("pytest -q")
        await processes_view.render(console, theme, state, _backend_for(state))
        shown = console.file.getvalue()  # type: ignore[attr-defined]
        assert "pytest -q" in shown
        assert "out.log" in shown
        assert "running" in shown

    @pytest.mark.asyncio
    async def test_a_backend_that_cannot_answer_is_reported_not_raised(
        self, console: Console, theme: Theme
    ) -> None:
        class _Broken:
            async def run(self, command: str, **_: object) -> CommandResult:
                return CommandResult(
                    exit_code=None, stdout="", stderr="", backend_error="pod is gone"
                )

        await processes_view.render(console, theme, _state_with(), _Broken())
        assert "could not be checked" in console.file.getvalue()  # type: ignore[attr-defined]


class TestStopping:
    @pytest.mark.asyncio
    async def test_it_asks_which_one_when_not_told(self, console: Console, theme: Theme) -> None:
        assert await processes_view.stop(console, theme, _state_with(), _FakeBackend(), "") is None
        assert "/ps" in console.file.getvalue()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_an_unknown_handle_is_refused(self, console: Console, theme: Theme) -> None:
        assert (
            await processes_view.stop(console, theme, _state_with(), _FakeBackend(), "nope") is None
        )

    @pytest.mark.asyncio
    async def test_a_stopped_command_stops_counting_against_the_cap(
        self, console: Console, theme: Theme
    ) -> None:
        """The returned state is what the REPL persists. Without it the next turn
        rebuilds a registry that still believes the process is live."""
        state = _state_with()
        handle = next(iter(processes.registry(state)))
        backend = _backend_for(state)

        updated = await processes_view.stop(console, theme, state, backend, handle)

        assert updated is not None
        assert processes_view.summary(updated) == ""
        assert any("kill -TERM" in c for c in backend.commands)


class TestLookingIsWhatRecordsIt:
    @pytest.mark.asyncio
    async def test_a_command_that_ended_stops_being_counted(
        self, console: Console, theme: Theme
    ) -> None:
        """Nothing watches a process between turns, so the indicator would go on
        claiming one is running hours after it stopped. Looking is what makes
        that knowable, so looking is what writes it down."""
        state = _state_with()
        assert processes_view.summary(state) == "1 command running"

        updated = await processes_view.render(
            console, theme, state, _backend_for(state, running=False)
        )

        assert updated is not None, "the REPL has to be told to persist this"
        assert processes_view.summary(updated) == ""

    @pytest.mark.asyncio
    async def test_nothing_to_persist_when_nothing_changed(
        self, console: Console, theme: Theme
    ) -> None:
        state = _state_with()
        assert await processes_view.render(console, theme, state, _backend_for(state)) is None

    @pytest.mark.asyncio
    async def test_many_processes_cost_one_call(self, console: Console, theme: Theme) -> None:
        """Against a pod each probe is an HTTP round trip. A listing that cost
        one per process is a listing nobody can afford to draw."""
        state: dict = {}
        for i in range(5):
            processes.remember(state, processes.new_record(f"cmd {i}"))
        backend = _backend_for(state)

        await processes_view.render(console, theme, state, backend)

        assert len(backend.commands) == 1


class TestItAsksTheMachineNotTheSession:
    @pytest.mark.asyncio
    async def test_a_process_with_no_record_is_still_listed(
        self, console: Console, theme: Theme
    ) -> None:
        """A process outlives the session that started it. Anything that loses
        the conversation's state — a crash before the turn was persisted, a store
        trimmed — used to orphan it permanently: still running, still writing,
        unreachable. The listing enumerates the directory, so it is one call
        away."""
        backend = _FakeBackend(handles={"beefcafe": "python train.py"})

        await processes_view.render(console, theme, {}, backend, "c1")

        shown = console.file.getvalue()  # type: ignore[attr-defined]
        assert "beefcafe" in shown
        assert "python train.py" in shown
        assert "before it" in shown, "it has to say why the session does not know it"


class TestItListsOnlyWhatIsStillGoing:
    @pytest.mark.asyncio
    async def test_a_finished_command_is_not_listed(self, console: Console, theme: Theme) -> None:
        """A list that keeps yesterday's commands is one nobody reads to the
        end, and "what is running" is the question being asked."""
        state = _state_with("pytest -q")

        await processes_view.render(console, theme, state, _backend_for(state, running=False))

        shown = console.file.getvalue()  # type: ignore[attr-defined]
        assert "pytest -q" not in shown
        assert "No commands running" in shown

    @pytest.mark.asyncio
    async def test_it_is_still_noticed_even_though_it_is_not_shown(
        self, console: Console, theme: Theme
    ) -> None:
        """Not listing it is a display decision; forgetting it would leave the
        prompt claiming it runs and leave it counting against the cap."""
        state = _state_with()

        updated = await processes_view.render(
            console, theme, state, _backend_for(state, running=False)
        )

        assert updated is not None
        assert processes_view.summary(updated) == ""

    @pytest.mark.asyncio
    async def test_one_that_cannot_be_checked_is_still_reported(
        self, console: Console, theme: Theme
    ) -> None:
        """Silence about a process nobody could reach is the one case where
        saying nothing is worse than a short list."""

        class _Broken:
            async def run(self, command: str, **_: object) -> CommandResult:
                return CommandResult(exit_code=None, stdout="", stderr="", backend_error="gone")

        await processes_view.render(console, theme, _state_with(), _Broken())
        assert "could not be checked" in console.file.getvalue()  # type: ignore[attr-defined]
