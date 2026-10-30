"""`find_instruction_files`: the walk ``domyn setup`` runs over the folder it is given, and
the loader over ``./.domyn`` and ``./.agents`` on every load."""

from pathlib import Path

from domyn.agents.instructions.discovery import find_instruction_files


def _write(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    return path


class TestTheSoul:
    def test_the_shallowest_one_wins(self, tmp_path: Path) -> None:
        _write(tmp_path / "a" / "SOUL.md")
        top = _write(tmp_path / "SOUL.md")

        assert find_instruction_files(tmp_path).soul == top

    def test_breadth_first_not_depth_first(self, tmp_path: Path) -> None:
        """``a/b/SOUL.md`` sorts before ``z/SOUL.md`` by name, but is deeper."""
        _write(tmp_path / "a" / "b" / "SOUL.md")
        near = _write(tmp_path / "z" / "SOUL.md")

        assert find_instruction_files(tmp_path).soul == near

    def test_none_when_there_is_none(self, tmp_path: Path) -> None:
        _write(tmp_path / "AGENTS.md")

        assert find_instruction_files(tmp_path).soul is None

    def test_the_others_are_superseded_shallowest_first(self, tmp_path: Path) -> None:
        top = _write(tmp_path / "SOUL.md")
        deep = _write(tmp_path / "a" / "b" / "SOUL.md")
        mid = _write(tmp_path / "z" / "SOUL.md")

        found = find_instruction_files(tmp_path)

        assert found.soul == top
        assert found.superseded == [mid, deep]
        assert deep not in found.paths


class TestTheAgentsFiles:
    def test_every_one_is_found_shallowest_first(self, tmp_path: Path) -> None:
        deep = _write(tmp_path / "a" / "b" / "AGENTS.md")
        top = _write(tmp_path / "AGENTS.md")
        mid = _write(tmp_path / "z" / "AGENTS.md")

        assert find_instruction_files(tmp_path).agents == [top, mid, deep]

    def test_agent_md_only_where_there_is_no_agents_md(self, tmp_path: Path) -> None:
        both = _write(tmp_path / "AGENTS.md")
        _write(tmp_path / "AGENT.md")
        alone = _write(tmp_path / "sub" / "AGENT.md")

        assert find_instruction_files(tmp_path).agents == [both, alone]

    def test_the_soul_comes_first_in_paths(self, tmp_path: Path) -> None:
        agents = _write(tmp_path / "AGENTS.md")
        soul = _write(tmp_path / "sub" / "SOUL.md")

        assert find_instruction_files(tmp_path).paths == [soul, agents]


class TestWhatIsSkipped:
    def test_hidden_and_dependency_folders(self, tmp_path: Path) -> None:
        _write(tmp_path / ".git" / "AGENTS.md")
        _write(tmp_path / "node_modules" / "pkg" / "AGENTS.md")
        _write(tmp_path / ".hidden" / "SOUL.md")

        assert find_instruction_files(tmp_path).paths == []

    def test_links_are_not_followed(self, tmp_path: Path) -> None:
        elsewhere = tmp_path / "elsewhere"
        _write(elsewhere / "AGENTS.md")
        _write(elsewhere / "SOUL.md")
        root = tmp_path / "root"
        root.mkdir()
        (root / "linked").symlink_to(elsewhere)
        (root / "SOUL.md").symlink_to(elsewhere / "SOUL.md")

        assert find_instruction_files(root).paths == []

    def test_a_linked_root_gives_nothing(self, tmp_path: Path) -> None:
        real = tmp_path / "real"
        _write(real / "SOUL.md")
        (tmp_path / "link").symlink_to(real)

        assert find_instruction_files(tmp_path / "link").paths == []

    def test_a_missing_root_gives_nothing(self, tmp_path: Path) -> None:
        assert find_instruction_files(tmp_path / "missing").paths == []


class TestSkip:
    def test_it_drops_the_named_folders_below_the_root(self, tmp_path: Path) -> None:
        _write(tmp_path / "conversations" / "c1" / "AGENTS.md")
        _write(tmp_path / "skills" / "s" / "SOUL.md")
        kept = _write(tmp_path / "instructions" / "AGENTS.md")

        found = find_instruction_files(tmp_path, skip=frozenset({"conversations", "skills"}))

        assert found.paths == [kept]

    def test_a_folder_of_that_name_deeper_down_is_walked(self, tmp_path: Path) -> None:
        """The state folders sit directly under ``.domyn``; a user's ``docs/skills`` is theirs."""
        deeper = _write(tmp_path / "docs" / "skills" / "AGENTS.md")

        assert find_instruction_files(tmp_path, skip=frozenset({"skills"})).agents == [deeper]

    def test_nothing_is_skipped_by_default(self, tmp_path: Path) -> None:
        agents = _write(tmp_path / "skills" / "AGENTS.md")

        assert find_instruction_files(tmp_path).agents == [agents]
