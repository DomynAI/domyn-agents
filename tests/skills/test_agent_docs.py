"""The domyn-docs skill: the documentation pages, where the CLI agent can read them,
behind an index (`skills/builtin/cli/domyn-docs/`).

In the repository `domyn-docs/pages` is a symlink to `docs/docs`; the sdist and the
wheel hold the pages themselves (`pyproject.toml`), and `domyn setup` copies them into
the installed skill as files.
The index in `SKILL.md` is written by hand, and checked here against the pages'
front matter and the order of `docs/sidebars.js`, so a page added, removed or
retitled fails until its line is right.
"""

import re
from pathlib import Path

import pytest
import yaml

import domyn.agents
from domyn.agents.skills import LocalDirSkillSource, SkillQuery
from domyn.agents.skills.catalog import install_skills
from domyn.agents.tools.builtin import Workspace

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs" / "docs"
SIDEBARS = ROOT / "docs" / "sidebars.js"
SKILL = Path(domyn.agents.__file__).parent / "skills" / "builtin" / "cli" / "domyn-docs"

_SIDEBAR_ENTRY = re.compile(r"""^\s*(?:section\('(?P<section>[^']+)'\)|'(?P<page>[^']+)'),?\s*$""")
_INDEX_LINE = re.compile(r"^(?:## .+|- `pages/.+)$")


def _sidebar_order() -> list[tuple[str, str]]:
    """``(section, doc id)`` for every page in the sidebars, in order."""
    entries: list[tuple[str, str]] = []
    section = ""
    for line in SIDEBARS.read_text(encoding="utf-8").splitlines():
        match = _SIDEBAR_ENTRY.match(line)
        if match is None:
            continue
        if match["section"] is not None:
            section = match["section"]
        else:
            entries.append((section, match["page"]))
    return entries


def _front_matter(doc_id: str) -> dict[str, str]:
    text = (SOURCE / f"{doc_id}.md").read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{doc_id}.md has no front matter"
    return yaml.safe_load(text[4 : text.index("\n---", 4)]) or {}


def _expected_index() -> list[str]:
    lines: list[str] = []
    current = None
    for section, doc_id in _sidebar_order():
        if section != current:
            current = section
            lines.append(f"## {section}")
        data = _front_matter(doc_id)
        description = " ".join(str(data["description"]).split())
        lines.append(f"- `pages/{doc_id}.md`: **{data['title']}**. {description}")
    return lines


class TestTheIndexFollowsThePages:
    def test_every_page_is_in_the_sidebar(self) -> None:
        on_disk = {
            p.relative_to(SOURCE).with_suffix("").as_posix()
            for p in SOURCE.rglob("*.md")
            if not p.name.startswith("_")
        }
        assert {doc_id for _, doc_id in _sidebar_order()} == on_disk

    def test_the_index_matches_the_pages(self) -> None:
        text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
        actual = [line for line in text.splitlines() if _INDEX_LINE.match(line)]
        expected = _expected_index()
        assert actual == expected, (
            "The index in domyn-docs/SKILL.md does not match docs/docs and docs/sidebars.js. "
            "Its sections and lines should read:\n\n" + "\n".join(expected)
        )


class TestTheModelCanFollowIt:
    def test_the_pages_are_the_documentation(self) -> None:
        assert (SKILL / "pages").resolve() == SOURCE.resolve()

    def test_every_path_in_the_index_exists(self) -> None:
        index = (SKILL / "SKILL.md").read_text(encoding="utf-8")
        paths = re.findall(r"`(pages/[^`]+\.md)`", index)
        assert paths
        assert [p for p in paths if not (SKILL / p).is_file()] == []

    def test_links_between_pages_point_to_pages(self) -> None:
        broken = []
        for page in SOURCE.rglob("*.md"):
            text = page.read_text(encoding="utf-8")
            for target in re.findall(r"\]\(([^)#:]+\.md)(?:#[^)]*)?\)", text):
                if not (page.parent / target).resolve().is_file():
                    broken.append(f"{page.relative_to(SOURCE)} -> {target}")
        assert broken == []

    def test_the_file_tools_can_read_the_installed_pages(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        (installed,) = install_skills(["domyn-docs"], tmp_path / "skills")
        workspace = Workspace.at(project, also=[installed])

        assert workspace.resolve(installed / "pages" / "guides" / "tools.md", must_exist=True)

    @pytest.mark.asyncio
    async def test_it_is_indexed_as_a_skill(self) -> None:
        index = await LocalDirSkillSource(paths=[SKILL.parent]).index(SkillQuery(task=""))
        names = {skill.name for skill in index.skills}
        # The pages beside it are read by path, never indexed as skills of their own.
        assert names == {"domyn-docs", "domyn-cli"}
