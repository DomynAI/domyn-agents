"""Instruction sources and the runtime's budget.

Three properties carry most of the weight, and none is obvious from the code:
the **order** is by kind and not by the order paths were given, an agent has
**one soul** (the first in path order), and nothing is ever dropped **silently**.
No link is followed. The rest is failure modes — a missing directory, an
unreadable file, an empty one — every one of which must leave a turn able to run.
"""

from pathlib import Path

import pytest

from domyn.agents.instructions import (
    AgentInstructions,
    InMemoryInstructionsSource,
    Instruction,
    InstructionKind,
    InstructionRuntime,
    LocalDirInstructionsSource,
)


@pytest.fixture
def home(tmp_path: Path) -> Path:
    """A directory of instructions."""
    directory = tmp_path / "domyn"
    directory.mkdir()
    return directory


def _write(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


async def _load(*paths: Path, max_bytes: int | None = None) -> AgentInstructions:
    runtime = InstructionRuntime(
        source=LocalDirInstructionsSource(paths=list(paths)),
        **({"max_bytes": max_bytes} if max_bytes is not None else {}),
    )
    return await runtime.load()


class TestWhatGetsFound:
    @pytest.mark.asyncio
    async def test_both_files_in_one_directory(self, home: Path) -> None:
        _write(home, "SOUL.md", "identity")
        _write(home, "AGENTS.md", "project")

        loaded = await _load(home)

        assert loaded.names == ("SOUL.md", "AGENTS.md")
        assert [one.text for one in loaded.soul] == ["identity"]
        assert [one.text for one in loaded.agent] == ["project"]

    @pytest.mark.asyncio
    async def test_soul_comes_first_whatever_order_the_paths_were_given(
        self, tmp_path: Path
    ) -> None:
        """Identity before task, always: the model weighs what is nearest the task
        most, so the project instructions have to land last."""
        first, second = tmp_path / "a", tmp_path / "b"
        first.mkdir()
        second.mkdir()
        _write(first, "AGENTS.md", "project")
        _write(second, "SOUL.md", "identity")

        loaded = await _load(first, second)

        assert loaded.names == ("SOUL.md", "AGENTS.md")
        assert loaded.text.index("identity") < loaded.text.index("project")

    @pytest.mark.asyncio
    async def test_every_agents_file_is_kept_in_path_order(self, tmp_path: Path) -> None:
        """Several paths contribute one file each; within a kind, the order the
        paths were given is kept."""
        first, second = tmp_path / "a", tmp_path / "b"
        first.mkdir()
        second.mkdir()
        _write(first, "AGENTS.md", "general")
        _write(second, "AGENTS.md", "specific")

        loaded = await _load(first, second)

        assert [one.text for one in loaded.agent] == ["general", "specific"]

    @pytest.mark.asyncio
    async def test_the_first_soul_wins(self, tmp_path: Path) -> None:
        """An agent has one identity: a later soul is named, not read."""
        first, second = tmp_path / "a", tmp_path / "b"
        first.mkdir()
        second.mkdir()
        _write(first, "SOUL.md", "first soul")
        later = _write(second, "SOUL.md", "second soul")

        loaded = await _load(first, second)

        assert [one.text for one in loaded.soul] == ["first soul"]
        assert "second soul" not in loaded.text
        assert loaded.superseded == (str(later.resolve()),)
        assert loaded.truncated is False

    @pytest.mark.asyncio
    async def test_an_empty_soul_gives_way_to_the_next(self, tmp_path: Path) -> None:
        first, second = tmp_path / "a", tmp_path / "b"
        first.mkdir()
        second.mkdir()
        _write(first, "SOUL.md", "  ")
        _write(second, "SOUL.md", "second soul")

        loaded = await _load(first, second)

        assert [one.text for one in loaded.soul] == ["second soul"]
        assert loaded.superseded == ()

    @pytest.mark.asyncio
    async def test_a_soul_not_used_survives_the_budget(self, tmp_path: Path) -> None:
        first, second = tmp_path / "a", tmp_path / "b"
        first.mkdir()
        second.mkdir()
        _write(first, "SOUL.md", "x" * 40)
        later = _write(second, "SOUL.md", "second soul")

        loaded = await _load(first, second, max_bytes=10)

        assert loaded.superseded == (str(later.resolve()),)

    @pytest.mark.asyncio
    async def test_a_directory_and_a_single_file_together(self, home: Path, tmp_path: Path) -> None:
        _write(home, "AGENTS.md", "project")
        brief = _write(tmp_path, "brief.md", "one-off brief")

        loaded = await _load(home, brief)

        assert loaded.names == ("AGENTS.md", "brief.md")

    @pytest.mark.asyncio
    async def test_the_same_file_reached_twice_is_read_once(self, home: Path) -> None:
        _write(home, "SOUL.md", "identity")

        loaded = await _load(home, home / "SOUL.md")

        assert loaded.names == ("SOUL.md",)

    @pytest.mark.asyncio
    async def test_a_single_file_is_the_task_brief(self, tmp_path: Path) -> None:
        """A declared ``./brief.md`` points at one file with any name."""
        _write(tmp_path, "brief.md", "one-off brief")

        loaded = await _load(tmp_path / "brief.md")

        assert loaded.names == ("brief.md",)
        assert [one.text for one in loaded.task] == ["one-off brief"]
        assert loaded.agent == ()

    @pytest.mark.asyncio
    async def test_a_single_agents_file_is_the_agent_brief(self, tmp_path: Path) -> None:
        """The kind follows the name, not how the file was reached."""
        agents = _write(tmp_path, "AGENTS.md", "project")

        loaded = await _load(agents)

        assert [one.text for one in loaded.agent] == ["project"]
        assert loaded.task == ()

    @pytest.mark.asyncio
    async def test_a_task_comes_last_whatever_order_the_paths_were_given(
        self, home: Path, tmp_path: Path
    ) -> None:
        """Soul, agent, task: the one-off brief is nearest the task, so it is
        rendered last even when it was typed first."""
        _write(home, "SOUL.md", "identity")
        _write(home, "AGENTS.md", "project")
        brief = _write(tmp_path, "brief.md", "one-off brief")

        loaded = await _load(brief, home)

        assert loaded.names == ("SOUL.md", "AGENTS.md", "brief.md")

    @pytest.mark.asyncio
    async def test_a_single_soul_file_is_the_soul(self, tmp_path: Path) -> None:
        _write(tmp_path, "SOUL.md", "identity")

        loaded = await _load(tmp_path / "SOUL.md")

        assert len(loaded.soul) == 1 and loaded.agent == ()

    @pytest.mark.asyncio
    async def test_agent_md_is_accepted_as_a_spelling_of_agents_md(self, home: Path) -> None:
        _write(home, "AGENT.md", "project")

        assert (await _load(home)).names == ("AGENT.md",)

    @pytest.mark.asyncio
    async def test_only_one_agents_spelling_is_read(self, home: Path) -> None:
        """Two spellings are two attempts at one file, not two files."""
        _write(home, "AGENTS.md", "canonical")
        _write(home, "AGENT.md", "older draft")

        loaded = await _load(home)

        assert loaded.names == ("AGENTS.md",)
        assert "older draft" not in loaded.text

    @pytest.mark.asyncio
    async def test_nothing_is_read_from_the_current_directory_unless_listed(
        self, home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cwd = tmp_path / "somewhere"
        cwd.mkdir()
        _write(cwd, "SOUL.md", "not mine")
        monkeypatch.chdir(cwd)

        assert not await _load(home)

    @pytest.mark.asyncio
    async def test_a_relative_path_reads_the_current_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """How ``./.domyn`` autodiscovery works: resolved on every load."""
        (tmp_path / ".domyn").mkdir()
        _write(tmp_path / ".domyn", "AGENTS.md", "project")
        monkeypatch.chdir(tmp_path)

        assert (await _load(Path(".domyn"))).names == ("AGENTS.md",)

    @pytest.mark.asyncio
    async def test_an_edit_is_seen_by_the_next_load(self, home: Path) -> None:
        """The point of loading per turn: nothing is cached between loads."""
        soul = _write(home, "SOUL.md", "before")
        runtime = InstructionRuntime(source=LocalDirInstructionsSource(paths=[home]))
        assert "before" in (await runtime.load()).text

        soul.write_text("after", encoding="utf-8")

        assert "after" in (await runtime.load()).text


def _put(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


class TestTheAutodiscoveryWalk:
    """``.domyn`` and ``.agents``, as written, are walked on every load; every other
    directory contributes only the files directly in it."""

    @pytest.fixture
    def project(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.chdir(tmp_path)
        return tmp_path

    @pytest.mark.asyncio
    async def test_subfolders_are_read(self, project: Path) -> None:
        _put(project / ".domyn" / "instructions" / "SOUL.md", "identity")
        _put(project / ".domyn" / "instructions" / "additional" / "AGENTS.md", "project")

        loaded = await _load(Path(".domyn"))

        assert [one.text for one in loaded.soul] == ["identity"]
        assert [one.text for one in loaded.agent] == ["project"]

    @pytest.mark.asyncio
    async def test_however_the_entry_is_spelled(self, project: Path) -> None:
        _put(project / ".domyn" / "sub" / "AGENTS.md", "project")

        assert (await _load(Path("./.domyn/"))).names == ("AGENTS.md",)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "where",
        [
            ".domyn/conversations/c1",
            ".domyn/browser",
            ".domyn/processes/p1",
            ".domyn/skills/s",
            ".agents/skills/s",
        ],
    )
    async def test_domyns_own_folders_are_not(self, project: Path, where: str) -> None:
        """An ``AGENTS.md`` the agent wrote in its scratch, or a skill's own, must never
        reach the system prompt."""
        _put(project / where / "AGENTS.md", "scratch")

        assert not await _load(Path(".domyn"), Path(".agents"))

    @pytest.mark.asyncio
    async def test_agents_files_go_shallow_to_deep_then_domyn_to_agents(
        self, project: Path
    ) -> None:
        _put(project / ".agents" / "AGENTS.md", "agents top")
        _put(project / ".domyn" / "a" / "b" / "AGENTS.md", "domyn deep")
        _put(project / ".domyn" / "AGENTS.md", "domyn top")

        loaded = await _load(Path(".domyn"), Path(".agents"))

        assert [one.text for one in loaded.agent] == ["domyn top", "domyn deep", "agents top"]

    @pytest.mark.asyncio
    async def test_a_deeper_soul_is_superseded(self, project: Path) -> None:
        _put(project / ".domyn" / "SOUL.md", "top soul")
        deeper = _put(project / ".domyn" / "sub" / "SOUL.md", "deeper soul")
        other = _put(project / ".agents" / "SOUL.md", "agents soul")

        loaded = await _load(Path(".domyn"), Path(".agents"))

        assert [one.text for one in loaded.soul] == ["top soul"]
        assert loaded.superseded == (str(deeper.resolve()), str(other.resolve()))

    @pytest.mark.asyncio
    async def test_a_declared_soul_beats_the_walk(self, project: Path, tmp_path: Path) -> None:
        user = _put(tmp_path / "user" / "SOUL.md", "user soul")
        _put(project / ".domyn" / "sub" / "SOUL.md", "project soul")

        loaded = await _load(user, Path(".domyn"))

        assert [one.text for one in loaded.soul] == ["user soul"]

    @pytest.mark.asyncio
    async def test_any_other_directory_stays_flat(self, project: Path) -> None:
        _put(project / "notes" / "sub" / "AGENTS.md", "relative")
        _put(project / "abs" / "sub" / "AGENTS.md", "absolute")

        assert not await _load(Path("notes"), project / "abs")

    @pytest.mark.asyncio
    async def test_a_linked_subfolder_is_not_entered(self, project: Path, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        _put(outside / "AGENTS.md", "secret")
        (project / ".domyn").mkdir()
        (project / ".domyn" / "linked").symlink_to(outside)

        assert not await _load(Path(".domyn"))


class TestNoLinkIsFollowed:
    """A relative path is chosen by whatever project the CLI starts in, and a
    followed link would send any file on the machine to the model."""

    @pytest.mark.asyncio
    async def test_a_linked_file_in_a_directory(self, home: Path, tmp_path: Path) -> None:
        secret = _write(tmp_path, "secret.txt", "secret")
        (home / "AGENTS.md").symlink_to(secret)
        _write(home, "SOUL.md", "identity")

        loaded = await _load(home)

        assert loaded.names == ("SOUL.md",)
        assert "secret" not in loaded.text

    @pytest.mark.asyncio
    async def test_a_linked_file_does_not_hide_the_other_spelling(
        self, home: Path, tmp_path: Path
    ) -> None:
        (home / "AGENTS.md").symlink_to(_write(tmp_path, "secret.txt", "secret"))
        _write(home, "AGENT.md", "project")

        assert (await _load(home)).names == ("AGENT.md",)

    @pytest.mark.asyncio
    async def test_a_linked_listed_file(self, tmp_path: Path) -> None:
        link = tmp_path / "brief.md"
        link.symlink_to(_write(tmp_path, "secret.txt", "secret"))

        assert not await _load(link)

    @pytest.mark.asyncio
    async def test_a_linked_folder_on_the_way_to_a_relative_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``.agents -> ~`` would make ``.agents`` read the home directory."""
        outside = tmp_path / "outside"
        outside.mkdir()
        _write(outside, "AGENTS.md", "secret")
        project = tmp_path / "project"
        project.mkdir()
        (project / ".agents").symlink_to(outside)
        monkeypatch.chdir(project)

        assert not await _load(Path(".agents"))


class TestNothingTakesTheTurnDown:
    @pytest.mark.asyncio
    async def test_a_missing_directory(self, tmp_path: Path) -> None:
        assert not await _load(tmp_path / "nowhere")

    @pytest.mark.asyncio
    async def test_a_directory_with_neither_file(self, home: Path) -> None:
        _write(home, "config.yaml", "provider: {}")

        assert not await _load(home)

    @pytest.mark.asyncio
    async def test_an_empty_file_is_not_announced_to_the_model(self, home: Path) -> None:
        """Rendering its header would tell the model there are instructions and
        then show it none."""
        _write(home, "SOUL.md", "   \n\n")
        _write(home, "AGENTS.md", "project")

        assert (await _load(home)).names == ("AGENTS.md",)

    @pytest.mark.asyncio
    async def test_an_unreadable_file_is_skipped(self, home: Path) -> None:
        soul = _write(home, "SOUL.md", "identity")
        _write(home, "AGENTS.md", "project")
        soul.chmod(0o000)
        try:
            loaded = await _load(home)
        finally:
            soul.chmod(0o644)

        assert loaded.names == ("AGENTS.md",)

    @pytest.mark.asyncio
    async def test_no_paths_at_all(self) -> None:
        loaded = await _load()

        assert not loaded
        assert loaded.text == ""


class TestTheBudget:
    @pytest.mark.asyncio
    async def test_an_instruction_that_fits_is_untouched(self, home: Path) -> None:
        _write(home, "SOUL.md", "identity")

        loaded = await _load(home, max_bytes=1_000)

        assert loaded.truncated is False
        assert loaded.soul[0].text == "identity"

    @pytest.mark.asyncio
    async def test_the_one_that_crosses_the_line_is_cut_not_dropped(self, home: Path) -> None:
        _write(home, "SOUL.md", "x" * 40)
        _write(home, "AGENTS.md", "y" * 40)

        loaded = await _load(home, max_bytes=50)

        assert loaded.names == ("SOUL.md", "AGENTS.md")
        assert loaded.soul[0].truncated is False
        assert loaded.agent[0].truncated is True
        assert len(loaded.agent[0].text) == 10

    @pytest.mark.asyncio
    async def test_what_did_not_fit_at_all_is_recorded(self, home: Path) -> None:
        """A standing instruction that vanished without a trace is a rule the
        agent stops following that nobody can see was removed."""
        _write(home, "SOUL.md", "x" * 40)
        agents = _write(home, "AGENTS.md", "y" * 40)

        loaded = await _load(home, max_bytes=40)

        assert loaded.names == ("SOUL.md",)
        assert loaded.dropped == (str(agents.resolve()),)
        assert loaded.truncated is True

    @pytest.mark.asyncio
    async def test_truncation_never_splits_a_character(self, home: Path) -> None:
        _write(home, "SOUL.md", "è" * 10)

        loaded = await _load(home, max_bytes=5)

        assert "�" not in loaded.soul[0].text

    @pytest.mark.asyncio
    async def test_it_applies_to_every_source(self) -> None:
        """The budget belongs to the runtime, not to the file source."""
        runtime = InstructionRuntime(source=InMemoryInstructionsSource(soul="x" * 40), max_bytes=10)

        loaded = await runtime.load()

        assert loaded.soul[0].truncated


class TestWhatTheModelSees:
    @pytest.mark.asyncio
    async def test_each_block_carries_its_absolute_path(self, home: Path) -> None:
        """The file tools cannot reach ``~/.domyn``, so the path in the text is
        the only route from "why do you keep doing that" back to the file."""
        soul = _write(home, "SOUL.md", "identity")

        assert str(soul.resolve()) in (await _load(home)).text

    @pytest.mark.asyncio
    async def test_truncation_is_visible_in_the_block(self, home: Path) -> None:
        _write(home, "SOUL.md", "x" * 40)

        assert "truncated" in (await _load(home, max_bytes=10)).text

    @pytest.mark.asyncio
    async def test_an_in_memory_instruction_is_rendered_by_name(self) -> None:
        source = InMemoryInstructionsSource(soul="Answer in Italian.", agent="  ")

        loaded = await source.load()

        assert loaded.names == ("soul",)
        assert loaded.text == "[soul]\nAnswer in Italian."


class TestTheModels:
    def test_from_parts_sorts_by_kind_and_keeps_order_within_one(self) -> None:
        def one(text: str, kind: InstructionKind) -> Instruction:
            return Instruction(text=text, instruction_kind=kind)

        loaded = AgentInstructions.from_parts(
            [
                one("t", InstructionKind.TASK),
                one("a1", InstructionKind.AGENT),
                one("s", InstructionKind.SOUL),
                one("a2", InstructionKind.AGENT),
            ]
        )

        assert [part.text for part in loaded.parts] == ["s", "a1", "a2", "t"]

    @pytest.mark.asyncio
    async def test_an_in_memory_source_has_all_three_slots(self) -> None:
        loaded = await InMemoryInstructionsSource(soul="s", agent="a", task="t").load()

        assert loaded.names == ("soul", "agent", "task")

    def test_a_slot_rejects_the_wrong_kind(self) -> None:
        wrong = Instruction(text="t", instruction_kind=InstructionKind.SOUL)

        with pytest.raises(ValueError):
            AgentInstructions(agent=(wrong,))

    def test_a_source_is_resolved_from_its_type_tag(self, home: Path) -> None:
        """So ``standing_instructions:`` can be written in YAML."""
        runtime = InstructionRuntime.model_validate(
            {"source": {"type": "local_dir", "paths": [str(home)]}}
        )

        assert isinstance(runtime.source, LocalDirInstructionsSource)

    @pytest.mark.asyncio
    async def test_each_agent_declares_its_own_in_yaml(self, tmp_path: Path) -> None:
        """The shape skills use: a ``standing_instructions:`` block per agent,
        beside ``skills:``, surviving the loader's whitelist of keys."""
        from domyn.agents import load_application_from_file

        expert_dir = tmp_path / "expert"
        expert_dir.mkdir()
        (expert_dir / "AGENTS.md").write_text("Use the house SQL style.", encoding="utf-8")
        config = tmp_path / "app.yaml"
        config.write_text(
            f"""
version: "1.0"
entrypoint: Orchestrator
llm_providers:
  - name: local
    type: custom
    model_name: m
    url: http://127.0.0.1:9/v1/chat/completions
planner_strategies:
  - name: tools
    type: tool_use
agents:
  - name: SqlExpert
    description: sql specialist
    instruction: follow the conventions
    llm_provider: local
    planner_strategy: tools
    standing_instructions:
      source:
        type: local_dir
        paths: ["{expert_dir}"]
      max_bytes: 1000
  - name: Orchestrator
    description: coordinator
    instruction: delegate
    llm_provider: local
    planner_strategy: tools
    sub_agents: [SqlExpert]
""",
            encoding="utf-8",
        )

        root = await load_application_from_file(str(config))

        assert root.standing_instructions is None, "the orchestrator declared none"
        expert = root.sub_agents[0]
        assert isinstance(expert.standing_instructions, InstructionRuntime)
        assert expert.standing_instructions.max_bytes == 1000
        loaded = await expert.standing_instructions.load()
        assert loaded.text and "house SQL style" in loaded.text

    def test_an_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown instructions source type"):
            InstructionRuntime.model_validate({"source": {"type": "nope"}})

    @pytest.mark.parametrize("value", ["local_dir", None, ["local_dir"], 42])
    def test_a_value_that_is_not_a_mapping_raises(self, value: object) -> None:
        """Stored as is, it would load, fail on every turn and be degraded by the
        runner to no instructions with a warning — silently, in effect."""
        with pytest.raises(ValueError, match="valid dictionary or instance of"):
            InstructionRuntime.model_validate({"source": value})
