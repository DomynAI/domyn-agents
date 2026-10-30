"""The one walk over a folder of skills, and what the loader makes of it.

`domyn setup` and `LocalDirSkillSource` share it, so what setup reports and what a
session loads cannot disagree about a layout.
"""

from pathlib import Path

import pytest

from domyn.agents.skills import LocalDirSkillSource, SkillQuery
from domyn.agents.skills.layout import listed_path_is_linked, walk_skill_tree


def _skill(path: Path, name: str) -> None:
    if path.suffix == ".md":
        path.parent.mkdir(parents=True, exist_ok=True)
        file = path
    else:
        path.mkdir(parents=True, exist_ok=True)
        file = path / "SKILL.md"
    file.write_text(f"---\nname: {name}\ndescription: d\n---\nbody\n")


async def _loaded(*paths: Path) -> list[str]:
    index = await LocalDirSkillSource(paths=list(paths)).index(SkillQuery(task=""))
    return [d.name for d in index.skills]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "skills"
    _skill(root / "writing" / "essays" / "draft", "draft")
    _skill(root / "writing" / "outline", "outline")
    _skill(root / "notes.md", "notes")
    _skill(root / "top", "top")
    return root


class TestTheWalk:
    def test_group_folders_are_entered_at_any_depth(self, root: Path) -> None:
        found = {f.path.relative_to(root).as_posix(): f.group for f in walk_skill_tree(root)}

        assert found == {
            "notes.md": (),
            "top": (),
            "writing/essays/draft": ("writing", "essays"),
            "writing/outline": ("writing",),
        }

    def test_a_symlinked_root_gives_nothing(self, root: Path, tmp_path: Path) -> None:
        (tmp_path / "link").symlink_to(root)

        assert walk_skill_tree(tmp_path / "link") == []


class TestTheLoader:
    @pytest.mark.asyncio
    async def test_it_loads_nested_skills(self, root: Path) -> None:
        assert sorted(await _loaded(root)) == ["draft", "notes", "outline", "top"]

    @pytest.mark.asyncio
    async def test_hidden_and_dependency_folders_are_not_loaded(self, root: Path) -> None:
        _skill(root / ".hidden" / "a", "hidden")
        _skill(root / "node_modules" / "b", "dependency")

        assert "hidden" not in await _loaded(root)
        assert "dependency" not in await _loaded(root)

    @pytest.mark.asyncio
    async def test_a_loose_file_below_the_root_is_not_a_skill(self, root: Path) -> None:
        _skill(root / "writing" / "readme.md", "readme")

        assert "readme" not in await _loaded(root)

    @pytest.mark.asyncio
    async def test_a_listed_skill_folder_or_file_loads_just_that_one(self, root: Path) -> None:
        assert await _loaded(root / "writing" / "outline") == ["outline"]
        assert await _loaded(root / "notes.md") == ["notes"]


class TestSymlinksAreNotFollowed:
    @pytest.mark.asyncio
    async def test_a_listed_folder_that_is_a_link(self, root: Path, tmp_path: Path) -> None:
        (tmp_path / "link").symlink_to(root)

        assert await _loaded(tmp_path / "link") == []

    @pytest.mark.asyncio
    async def test_a_listed_skill_that_is_a_link(self, root: Path, tmp_path: Path) -> None:
        (tmp_path / "one").symlink_to(root / "top")
        (tmp_path / "one.md").symlink_to(root / "notes.md")

        assert await _loaded(tmp_path / "one", tmp_path / "one.md") == []

    @pytest.mark.asyncio
    async def test_a_link_inside_a_listed_folder(self, root: Path, tmp_path: Path) -> None:
        _skill(tmp_path / "outside" / "secret", "secret")
        (root / "secret").symlink_to(tmp_path / "outside" / "secret")
        (root / "group").symlink_to(tmp_path / "outside")

        assert "secret" not in await _loaded(root)

    @pytest.mark.asyncio
    async def test_a_symlinked_parent_of_a_relative_path_is_not_followed(
        self, root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The relative part comes from the project domyn starts in: `.agents -> ~` there
        must not make `.agents/skills` read `~/skills`."""
        (tmp_path / "repo").mkdir()
        (tmp_path / "repo" / ".agents").symlink_to(root.parent)
        monkeypatch.chdir(tmp_path / "repo")

        assert await _loaded(Path(".agents/skills")) == []

    @pytest.mark.asyncio
    async def test_a_symlinked_parent_is_fine(self, root: Path, tmp_path: Path) -> None:
        (tmp_path / "parent").symlink_to(root.parent)

        assert sorted(await _loaded(tmp_path / "parent" / "skills")) == [
            "draft",
            "notes",
            "outline",
            "top",
        ]


class TestAListedPathIsLinked:
    @pytest.fixture(autouse=True)
    def repo(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        (tmp_path / "real" / "skills").mkdir(parents=True)
        (tmp_path / "repo").mkdir()
        monkeypatch.chdir(tmp_path / "repo")
        return tmp_path / "repo"

    def test_a_plain_relative_path_is_not(self, repo: Path) -> None:
        (repo / ".domyn" / "skills").mkdir(parents=True)

        assert listed_path_is_linked(Path(".domyn/skills")) is False

    def test_a_missing_path_is_not(self) -> None:
        assert listed_path_is_linked(Path(".agents/skills")) is False

    def test_the_last_component_linked_is(self, repo: Path, tmp_path: Path) -> None:
        (repo / ".domyn").mkdir()
        (repo / ".domyn" / "skills").symlink_to(tmp_path / "real" / "skills")

        assert listed_path_is_linked(Path(".domyn/skills")) is True

    def test_a_linked_folder_on_a_relative_path_is(self, repo: Path, tmp_path: Path) -> None:
        (repo / ".agents").symlink_to(tmp_path / "real")

        assert listed_path_is_linked(Path(".agents/skills")) is True
        assert listed_path_is_linked(Path("./.agents/skills")) is True

    def test_a_linked_folder_above_an_absolute_path_is_not(
        self, repo: Path, tmp_path: Path
    ) -> None:
        """`/tmp` is a link on macOS: the folders above an absolute path are the user's."""
        (repo / ".agents").symlink_to(tmp_path / "real")

        assert listed_path_is_linked(repo / ".agents" / "skills") is False
