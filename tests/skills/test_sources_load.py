"""One loader per source: ``load()`` is where the files are read, once.

``parse_skill_file`` has always returned both the descriptor and the body, but
``index()`` discarded the body and ``fetch()`` then re-walked every directory and
re-parsed every file to get it back. Expanding N skills cost N+1 full walks of
work that had already been done. These are the guards on that not coming back.
"""

import asyncio
import threading
import time
from pathlib import Path

import pytest

from domyn.agents.skills import (
    AllSkillsSelector,
    InMemorySkillSource,
    LocalDirSkillSource,
    SkillContent,
    SkillIndex,
    SkillNotFoundError,
    SkillQuery,
    SkillRef,
    SkillSource,
    SkillsRuntime,
)
from domyn.agents.skills import sources as sources_module

SKILL = """---
name: {name}
description: skill number {n}
---

Body of {name}.

## details

The detail of {name}.
"""


@pytest.fixture
def five_skills(tmp_path: Path) -> Path:
    for n in range(5):
        directory = tmp_path / f"skill-{n}"
        directory.mkdir()
        (directory / "SKILL.md").write_text(SKILL.format(name=f"skill-{n}", n=n), encoding="utf-8")
    return tmp_path


class TestOneWalkNotN:
    @pytest.mark.asyncio
    async def test_each_file_is_parsed_once_for_a_full_expansion(
        self, five_skills: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Five skills, all expanded: five parses.

        Before the split this was 30 — one walk to index plus one walk per skill
        fetched, each walk re-parsing all five files.
        """
        calls: list[Path] = []
        original = sources_module.parse_skill_file

        def counting(path: Path, **kwargs: object):
            calls.append(path)
            return original(path, **kwargs)

        monkeypatch.setattr(sources_module, "parse_skill_file", counting)

        runtime = SkillsRuntime(
            source=LocalDirSkillSource(paths=[five_skills]), selector=AllSkillsSelector()
        )
        selection, contents = await runtime.resolve(SkillQuery())

        assert len(selection.index) == 5
        assert len(contents) == 5, "every skill should have been expanded"
        assert len(calls) == 5, f"parsed {len(calls)}x for 5 skills — the loader ran more than once"


class TestBodiesComeBackWithTheIndex:
    @pytest.mark.asyncio
    async def test_local_bodies_are_keyed_by_the_parsed_name(self, tmp_path: Path) -> None:
        """Not the filename: frontmatter ``name:`` wins, and the index reports it.

        Keying the bodies by the file's name instead would file them under
        something ``SkillRef.name`` never asks for.
        """
        (tmp_path / "on-disk.md").write_text(
            "---\nname: declared-name\ndescription: d\n---\n\nthe body", encoding="utf-8"
        )
        loaded = await LocalDirSkillSource(paths=[tmp_path]).load(SkillQuery())

        assert [s.name for s in loaded.index.skills] == ["declared-name"]
        assert set(loaded.bodies) == {"declared-name"}
        assert loaded.body("declared-name").body == "the body"

    @pytest.mark.asyncio
    async def test_in_memory_bodies_are_keyed_by_the_parsed_name(self) -> None:
        source = InMemorySkillSource(
            skills={"mapping-key": "---\nname: declared-name\ndescription: d\n---\n\nthe body"}
        )
        loaded = await source.load(SkillQuery())

        assert set(loaded.bodies) == {"declared-name"}
        fetched = await source.fetch(SkillRef(name="declared-name"), SkillQuery())
        assert fetched.body == "the body"

    @pytest.mark.asyncio
    async def test_the_first_directory_wins_for_bodies_too(self, tmp_path: Path) -> None:
        """Precedence is expressed by ordering, and must not differ per half.

        An index from the first directory paired with a body from the second is a
        skill whose description and content disagree.
        """
        first, second = tmp_path / "first", tmp_path / "second"
        for directory, body in ((first, "winner"), (second, "loser")):
            directory.mkdir()
            (directory / "dup.md").write_text(
                f"---\nname: dup\ndescription: d\n---\n\n{body}", encoding="utf-8"
            )

        loaded = await LocalDirSkillSource(paths=[first, second]).load(SkillQuery())

        assert len(loaded.index.skills) == 1
        assert loaded.body("dup").body == "winner"


class TestMalformedFiles:
    @pytest.mark.asyncio
    async def test_a_broken_neighbour_no_longer_takes_the_others_down(self, tmp_path: Path) -> None:
        """``fetch`` used to raise SkillParseError from an unrelated file.

        ``index`` skipped malformed files, ``fetch`` did not, and the guard in
        ``SkillsRuntime`` catches only (SkillNotFoundError, OSError) — so one bad
        neighbour wiped the whole payload rather than itself.
        """
        (tmp_path / "broken.md").write_text("---\n: : not yaml : :\n---\n\nx", encoding="utf-8")
        (tmp_path / "good.md").write_text(
            "---\nname: good\ndescription: d\n---\n\nfine", encoding="utf-8"
        )
        source = LocalDirSkillSource(paths=[tmp_path])

        assert [s.name for s in (await source.index(SkillQuery())).skills] == ["good"]
        assert (await source.fetch(SkillRef(name="good"), SkillQuery())).body == "fine"

        with pytest.raises(SkillNotFoundError):
            await source.fetch(SkillRef(name="broken"), SkillQuery())


class TestTheBaseClassDefault:
    @pytest.mark.asyncio
    async def test_a_source_that_only_implements_index_and_fetch_still_works(self) -> None:
        """``load`` is concrete, so every source written before it keeps working.

        Making it abstract would break every third-party source at construction.
        """

        class OldStyleSource(SkillSource):
            type: str = "old_style"

            async def index(self, query: SkillQuery) -> SkillIndex:
                return SkillIndex()

            async def fetch(self, ref: SkillRef, query: SkillQuery) -> SkillContent:
                raise SkillNotFoundError(ref.name)

        loaded = await OldStyleSource().load(SkillQuery())

        assert loaded.index.skills == []
        assert loaded.bodies == {}, "index-only sources must not invent bodies"


class _RecordingLogger:
    """Captures what ``load`` decided to say, without a real logging backend.

    ``get_logger()`` returns a wrapper around whatever logger the host injected,
    so there is nothing for ``caplog`` to attach to.
    """

    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.debugs: list[str] = []

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        self.warnings.append(message)

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        self.debugs.append(message)

    def info(self, message: str, *args: object, **kwargs: object) -> None:
        pass

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        pass


def _record_logs(monkeypatch: pytest.MonkeyPatch) -> _RecordingLogger:
    recorder = _RecordingLogger()
    monkeypatch.setattr(sources_module, "get_logger", lambda: recorder)
    return recorder


def _parse_taking(monkeypatch: pytest.MonkeyPatch, delay_for) -> None:
    """Make each file take a chosen time to parse, inverting completion order.

    Without this the parses finish in roughly submission order by luck, and a
    loader rebuilt around ``as_completed`` would pass the ordering tests anyway.
    """
    original = sources_module.parse_skill_file

    def slow(path: Path, **kwargs: object):
        time.sleep(delay_for(path))
        return original(path, **kwargs)

    monkeypatch.setattr(sources_module, "parse_skill_file", slow)


def _parse_raising(monkeypatch: pytest.MonkeyPatch, failures: dict[str, BaseException]) -> None:
    """Raise a chosen exception from the thread, keyed by file name."""
    original = sources_module.parse_skill_file

    def failing(path: Path, **kwargs: object):
        if path.name in failures:
            raise failures[path.name]
        return original(path, **kwargs)

    monkeypatch.setattr(sources_module, "parse_skill_file", failing)


def _write_flat(directory: Path, *names: str) -> None:
    directory.mkdir(exist_ok=True)
    for name in names:
        (directory / f"{name}.md").write_text(
            f"---\nname: {name}\ndescription: d\n---\n\nbody of {name}", encoding="utf-8"
        )


class TestOrderIsSubmissionOrderNotCompletionOrder:
    """The walk and the parses run in threads; the order must not come from them.

    ``self.paths`` order then the sort inside ``_iter_skill_files`` is the whole
    precedence rule — first occurrence of a name wins — and it is also what keeps
    the prompt prefix stable across runs. Nothing else in the suite pins it.
    """

    @pytest.mark.asyncio
    async def test_a_directory_keeps_its_sorted_order_when_the_parses_finish_backwards(
        self, five_skills: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # skill-0 is slowest, skill-4 fastest: completion order is the reverse of
        # the order the result has to come back in.
        _parse_taking(monkeypatch, lambda path: 0.02 * (5 - int(path.parent.name[-1])))

        loaded = await LocalDirSkillSource(paths=[five_skills]).load(SkillQuery())

        assert [s.name for s in loaded.index.skills] == [f"skill-{n}" for n in range(5)]

    @pytest.mark.asyncio
    async def test_directories_keep_their_configured_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first, second = tmp_path / "first", tmp_path / "second"
        _write_flat(first, "alpha", "bravo")
        _write_flat(second, "charlie", "delta")
        # The first directory's files are the slow ones.
        _parse_taking(monkeypatch, lambda path: 0.06 if path.parent.name == "first" else 0.0)

        loaded = await LocalDirSkillSource(paths=[first, second]).load(SkillQuery())

        assert [s.name for s in loaded.index.skills] == ["alpha", "bravo", "charlie", "delta"]

    @pytest.mark.asyncio
    async def test_the_first_occurrence_still_wins_when_it_parses_last(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The existing first-directory test cannot tell first-wins from last-wins.

        With one file per directory either rule produces the same answer as long as
        they are merged in order; making the winner finish last is what separates
        them.
        """
        first, second = tmp_path / "first", tmp_path / "second"
        for directory, body in ((first, "winner"), (second, "loser")):
            directory.mkdir()
            (directory / "dup.md").write_text(
                f"---\nname: dup\ndescription: d\n---\n\n{body}", encoding="utf-8"
            )
        _parse_taking(monkeypatch, lambda path: 0.06 if path.parent.name == "first" else 0.0)

        loaded = await LocalDirSkillSource(paths=[first, second]).load(SkillQuery())

        assert len(loaded.index.skills) == 1
        assert loaded.body("dup").body == "winner"


class TestExceptionsCrossingTheThreadBoundary:
    """``return_exceptions=True`` hands back every failure as a value.

    Only the two a malformed file raises may be swallowed. Anything else has to
    keep leaving ``load()`` the way it did when the parse was inline — the failure
    mode being a bug demoted to a warning about a bad skill file.
    """

    @pytest.mark.asyncio
    async def test_an_unexpected_error_still_leaves_load(
        self, five_skills: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _record_logs(monkeypatch)
        _parse_raising(monkeypatch, {"SKILL.md": RuntimeError("boom")})

        with pytest.raises(RuntimeError, match="boom"):
            await LocalDirSkillSource(paths=[five_skills]).load(SkillQuery())

        assert recorder.warnings == [], "a bug was reported as an unreadable skill file"

    @pytest.mark.asyncio
    async def test_a_cancelled_parse_is_re_raised_not_swallowed(
        self, five_skills: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Swallowing it would leave the turn running inside a cancelled task."""
        recorder = _record_logs(monkeypatch)
        _parse_raising(monkeypatch, {"SKILL.md": asyncio.CancelledError()})

        with pytest.raises(asyncio.CancelledError):
            await LocalDirSkillSource(paths=[five_skills]).load(SkillQuery())

        assert recorder.warnings == []

    @pytest.mark.asyncio
    async def test_an_unreadable_file_and_a_broken_one_are_both_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both halves of the caught pair, together, with the survivors in order.

        ``TestMalformedFiles`` covers only the ``ValueError`` side; the ``OSError``
        one reaches the same branch and used to be an ``except`` clause on an
        inline call.
        """
        _write_flat(tmp_path, "aaa", "bbb", "ccc", "locked")
        (tmp_path / "broken.md").write_text("---\n: : not yaml : :\n---\n\nx", encoding="utf-8")
        recorder = _record_logs(monkeypatch)
        _parse_raising(monkeypatch, {"locked.md": PermissionError("no read")})

        loaded = await LocalDirSkillSource(paths=[tmp_path]).load(SkillQuery())

        assert [s.name for s in loaded.index.skills] == ["aaa", "bbb", "ccc"]
        assert len(recorder.warnings) == 2
        assert any("broken.md" in w and "SkillParseError" in w for w in recorder.warnings)
        assert any("locked.md" in w and "PermissionError" in w for w in recorder.warnings)

    @pytest.mark.asyncio
    async def test_a_directory_that_cannot_be_listed_still_raises(
        self, five_skills: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The per-file tolerance is scoped to files, deliberately.

        A configured directory that cannot be listed is a deployment fault, and
        swallowing it returns a silently short index instead. It already degrades
        one level up, where the runner logs which agent lost its skills.
        """

        def boom(directory: Path) -> list[Path]:
            raise PermissionError(f"cannot list {directory}")

        monkeypatch.setattr(sources_module, "_iter_skill_files", boom)

        with pytest.raises(PermissionError):
            await LocalDirSkillSource(paths=[five_skills]).load(SkillQuery())


class TestTheLoopIsNotBlocked:
    """The point of the exercise: this runs once per turn for every agent at once."""

    @pytest.mark.asyncio
    async def test_the_files_are_parsed_concurrently(
        self, five_skills: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A barrier the five parses can only clear together: serialised, the
        first one waits for four that will never arrive."""
        together = threading.Barrier(5)
        original = sources_module.parse_skill_file

        def meeting(path: Path, **kwargs: object):
            together.wait(timeout=5)
            return original(path, **kwargs)

        monkeypatch.setattr(sources_module, "parse_skill_file", meeting)

        loaded = await LocalDirSkillSource(paths=[five_skills]).load(SkillQuery())

        assert len(loaded.index.skills) == 5

    @pytest.mark.asyncio
    async def test_other_coroutines_run_while_the_files_are_parsed(
        self, five_skills: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every parse waits for a coroutine that can only run on a free loop, so
        a blocking loader cannot complete the load at all.

        Counting the ticks of a timer instead measured the machine: under a full
        parallel suite a 5ms sleep is not 5ms, and the load finished before the
        ticker had been scheduled often enough.
        """
        entered, released = threading.Event(), threading.Event()
        original = sources_module.parse_skill_file

        def blocking(path: Path, **kwargs: object):
            entered.set()
            assert released.wait(2), "the parse ran with the loop blocked behind it"
            return original(path, **kwargs)

        monkeypatch.setattr(sources_module, "parse_skill_file", blocking)

        async def release() -> None:
            # Not released until a parse is under way, or the loop could have
            # been free only before the work started.
            while not entered.is_set():
                await asyncio.sleep(0)
            released.set()

        releaser = asyncio.create_task(release())
        loaded = await LocalDirSkillSource(paths=[five_skills]).load(SkillQuery())
        await releaser

        assert len(loaded.index.skills) == 5


class TestDegenerateInputs:
    @pytest.mark.asyncio
    async def test_no_configured_paths_returns_an_empty_payload(self) -> None:
        """Guards the zero-awaitable ``gather``, which is a branch not taken."""
        loaded = await LocalDirSkillSource(paths=[]).load(SkillQuery())

        assert loaded.index.skills == []
        assert loaded.bodies == {}

    @pytest.mark.asyncio
    async def test_the_worker_passes_the_path_positionally(
        self, five_skills: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The counting wrapper above binds ``path`` either way; this does not.

        A refactor to ``parse_skill_file(path=path, ...)`` would keep every other
        test green and break any wrapper written against the current call shape.
        """
        seen: list[tuple[tuple, tuple]] = []
        original = sources_module.parse_skill_file

        def spy(*args: object, **kwargs: object):
            seen.append((args, tuple(sorted(kwargs))))
            return original(*args, **kwargs)

        monkeypatch.setattr(sources_module, "parse_skill_file", spy)

        await LocalDirSkillSource(paths=[five_skills]).load(SkillQuery())

        assert len(seen) == 5
        assert all(len(args) == 1 and isinstance(args[0], Path) for args, _ in seen)
        assert all(keywords == ("source",) for _, keywords in seen)


class TestHomeRelativePaths:
    """A ``~`` path is walked as the home directory, but shown as written."""

    @pytest.fixture
    def home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        home = tmp_path / "home"
        home.mkdir()
        _write_flat(home / "skills", "alpha")
        monkeypatch.setenv("HOME", str(home))
        elsewhere = tmp_path / "cwd"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        return home

    @pytest.mark.asyncio
    async def test_the_walk_expands_the_home_directory(self, home: Path) -> None:
        """Walked literally, ``~/skills`` is a folder named ``~`` under the cwd:
        no skills and no error."""
        index = await LocalDirSkillSource(paths=["~/skills"]).index(SkillQuery())

        assert [skill.name for skill in index.skills] == ["alpha"]
        assert index.skills[0].path is not None
        assert index.skills[0].path.is_relative_to(home)

    def test_the_path_is_kept_as_written(self, home: Path) -> None:
        source = LocalDirSkillSource(paths=["~/skills"])

        assert source.paths == [Path("~/skills")]
        assert "~/skills" in source.describe()


class TestAPathThatIsItselfASkill:
    """`domyn setup` lists each installed skill's own folder in the config, two
    levels down from where a directory scan would find it."""

    @pytest.mark.asyncio
    async def test_a_skill_folder_loads_just_that_skill(self, five_skills: Path) -> None:
        source = LocalDirSkillSource(paths=[five_skills / "skill-2"])

        index = await source.index(SkillQuery(task=""))

        assert [d.name for d in index.skills] == ["skill-2"]

    @pytest.mark.asyncio
    async def test_a_skill_folder_and_a_folder_of_skills_mix(
        self, five_skills: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        other = tmp_path_factory.mktemp("other")
        (other / "extra").mkdir()
        (other / "extra" / "SKILL.md").write_text(SKILL.format(name="extra", n=9), encoding="utf-8")
        source = LocalDirSkillSource(paths=[five_skills / "skill-4", other])

        index = await source.index(SkillQuery(task=""))

        assert [d.name for d in index.skills] == ["skill-4", "extra"]

    @pytest.mark.asyncio
    async def test_a_skill_folder_serves_its_body(self, five_skills: Path) -> None:
        source = LocalDirSkillSource(paths=[five_skills / "skill-1"])

        content = await source.fetch(SkillRef(name="skill-1"), SkillQuery(task=""))

        assert "Body of skill-1." in content.body
