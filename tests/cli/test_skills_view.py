"""Tests for ``/skills`` and ``/skill <name>`` (also ``domyn skills`` / ``domyn skill show``).

The list has one promise worth protecting above the layout: every skill shows
where its file is. It is the first question when a skill misbehaves, and with
several skill directories the name alone does not answer it.
"""

import io
from pathlib import Path

import pytest
from rich.console import Console

from domyn.agents.interfaces.cli.skills_view import render_skills
from domyn.agents.interfaces.cli.theme import Capabilities, Styles, Symbols, Theme
from domyn.agents.skills import (
    IndexSelector,
    LocalDirSkillSource,
    SkillBudget,
    SkillNotFoundError,
    SkillsRuntime,
)

LONG = (
    "How to write a git commit message for this project. Use when asked to commit, "
    "to write or rewrite a commit message, or to review one before pushing."
)

BODY = """Intro line.

## Format

```
<type>: <subject>
```

## Checklist

- [ ] Subject is imperative
"""


def _skill(
    directory: Path,
    name: str,
    *,
    flat: bool = False,
    description: str = "Does a thing.",
    extra: str = "",
    body: str = BODY,
) -> Path:
    path = directory / f"{name}.md" if flat else directory / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: {description}\n{extra}---\n{body}")
    return path


def _theme(*, unicode: bool = True) -> Theme:
    return Theme(
        capabilities=Capabilities(color=False, unicode=unicode, animation=False),
        symbols=Symbols(unicode=unicode),
        styles=Styles,
    )


def _runtime(*dirs: Path, budget: SkillBudget | None = None) -> SkillsRuntime:
    return SkillsRuntime(
        source=LocalDirSkillSource(paths=list(dirs)),
        selector=IndexSelector(),
        budget=budget or SkillBudget(),
    )


async def _render(
    runtime: SkillsRuntime | None, name: str | None = None, *, width: int = 80, **theme_kw: bool
) -> str:
    buffer = io.StringIO()
    console = Console(file=buffer, width=width, force_terminal=False, color_system=None)
    await render_skills(console, _theme(**theme_kw), runtime, name)
    return "\n".join(line.rstrip() for line in buffer.getvalue().splitlines())


@pytest.fixture
def layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A project skills dir under the working directory, and one under home."""
    home = tmp_path / "home"
    work = tmp_path / "work"
    project, personal = work / "skills", home / ".domyn" / "skills"
    project.mkdir(parents=True)
    personal.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)
    return project, personal


class TestTheListIsGroupedByDirectory:
    @pytest.mark.asyncio
    async def test_each_directory_heads_its_own_skills(self, layout: tuple[Path, Path]) -> None:
        project, personal = layout
        _skill(project, "commit-message")
        _skill(personal, "journal", flat=True)

        out = await _render(_runtime(project, personal), width=400)

        assert out.index(str(project)) < out.index("commit-message") < out.index(str(personal))
        assert out.index(str(personal)) < out.index("journal")

    @pytest.mark.asyncio
    async def test_directories_are_shown_as_absolute_paths(self, layout: tuple[Path, Path]) -> None:
        project, personal = layout
        _skill(project, "commit-message")
        _skill(personal, "journal", flat=True)

        out = await _render(_runtime(project, personal), width=400)

        assert f"  {project}\n" in out
        assert f"  {personal}\n" in out
        assert "./skills" not in out
        assert "~/" not in out

    @pytest.mark.asyncio
    async def test_every_skill_still_shows_its_file(self, layout: tuple[Path, Path]) -> None:
        project, personal = layout
        _skill(project, "commit-message")
        _skill(personal, "journal", flat=True)

        out = await _render(_runtime(project, personal))

        # And the two layouts can be told apart, which the name alone cannot do.
        assert "commit-message/SKILL.md" in out
        assert "journal.md" in out

    @pytest.mark.asyncio
    async def test_a_loose_skill_listed_by_its_own_file_shows_the_file_name(
        self, layout: tuple[Path, Path]
    ) -> None:
        """`domyn setup` lists a loose `notes.md` singly, so the "directory" is the
        file itself; its label used to be "." and told nobody anything."""
        project, _ = layout
        loose = _skill(project, "notes", flat=True)

        out = await _render(_runtime(loose), width=400)

        lines = out.splitlines()
        row = next(line for line in lines if line.strip().startswith("notes "))
        assert "notes.md" in lines[lines.index(row) + 1]
        assert not any(line.strip() == "." for line in lines)

    @pytest.mark.asyncio
    async def test_a_directory_with_nothing_in_it_is_reported(
        self, layout: tuple[Path, Path]
    ) -> None:
        project, personal = layout
        _skill(project, "commit-message")

        out = await _render(_runtime(project, personal), width=400)

        assert f"also searched: {personal} (none found)" in out


class TestTheListLayout:
    @pytest.mark.asyncio
    async def test_a_long_description_wraps_inside_its_column(
        self, layout: tuple[Path, Path]
    ) -> None:
        project, _ = layout
        _skill(project, "commit-message", description=LONG)

        out = await _render(_runtime(project), width=60)

        lines = out.splitlines()
        first = next(i for i, line in enumerate(lines) if "commit-message " in line)
        column = lines[first].index("How to")
        continuation = lines[first + 1]
        assert continuation[:column].strip() == ""
        assert continuation[column:].strip()

    @pytest.mark.asyncio
    async def test_sections_are_listed(self, layout: tuple[Path, Path]) -> None:
        project, _ = layout
        _skill(project, "commit-message")

        out = await _render(_runtime(project))

        assert "Format · Checklist" in out

    @pytest.mark.asyncio
    async def test_required_and_unavailable_are_both_visible(
        self, layout: tuple[Path, Path]
    ) -> None:
        project, _ = layout
        _skill(
            project,
            "deploy",
            extra="enforcement: required\nmetadata:\n  requires:\n    bins: [surely-not-on-path-xyz]\n",
        )

        out = await _render(_runtime(project))

        assert "required" in out
        assert "✗" in out
        assert "unavailable:" in out
        assert "surely-not-on-path-xyz" in out
        assert "3 available" not in out and "0 available · 1 unavailable" in out

    @pytest.mark.asyncio
    async def test_a_missing_external_file_is_shown_as_the_reason(
        self, layout: tuple[Path, Path]
    ) -> None:
        project, _ = layout
        _skill(project, "report", extra="external_files:\n  - ../shared/helper.py\n")

        out = await _render(_runtime(project))

        assert "unavailable:" in out
        assert "file '../shared/helper.py' (does not exist)" in out

    @pytest.mark.asyncio
    async def test_external_files_are_counted_in_the_requires_row(
        self, layout: tuple[Path, Path]
    ) -> None:
        project, _ = layout
        _skill(project, "report", extra="external_files:\n  - a.py\n  - b.py\n")
        (project / "report" / "a.py").write_text("x")
        (project / "report" / "b.py").write_text("x")

        out = await _render(_runtime(project), "report")

        assert "files 2" in out
        assert "a.py" not in out
        assert "unavailable" not in out

    @pytest.mark.asyncio
    async def test_a_description_is_printed_as_written_not_as_markup(
        self, layout: tuple[Path, Path]
    ) -> None:
        project, _ = layout
        _skill(project, "odd", description="Keeps [bold]this[/bold] literal.")

        out = await _render(_runtime(project))

        assert "[bold]this[/bold]" in out

    @pytest.mark.asyncio
    async def test_the_ascii_theme_uses_no_unicode_glyphs(self, layout: tuple[Path, Path]) -> None:
        project, _ = layout
        _skill(project, "commit-message")
        _skill(
            project, "deploy", extra="metadata:\n  requires:\n    bins: [surely-not-on-path-xyz]\n"
        )

        out = await _render(_runtime(project), unicode=False)

        assert not {"·", "✗", "•", "─"} & set(out)
        assert "[no]" in out

    @pytest.mark.asyncio
    async def test_an_index_over_budget_is_announced(self, layout: tuple[Path, Path]) -> None:
        """Every skill is still listed here; what changes is how the model sees them."""
        project, _ = layout
        for n in range(6):
            _skill(project, f"skill-{n}", description=LONG)

        # Six whole lines are ~950 chars, six shortened ones under 800.
        out = await _render(_runtime(project, budget=SkillBudget(max_index_chars=900)))

        assert all(f"skill-{n}" in out for n in range(6))
        assert "the model sees shortened descriptions" in out

    @pytest.mark.asyncio
    async def test_names_only_and_unlisted_are_announced(self, layout: tuple[Path, Path]) -> None:
        project, _ = layout
        for n in range(6):
            _skill(project, f"skill-{n}", description=LONG)

        out = await _render(_runtime(project, budget=SkillBudget(max_index_chars=60)), width=200)

        assert "the model sees skills by name only, and not at all: skill-" in out

    @pytest.mark.asyncio
    async def test_an_index_within_budget_says_nothing(self, layout: tuple[Path, Path]) -> None:
        project, _ = layout
        _skill(project, "commit-message", description=LONG)

        out = await _render(_runtime(project))

        assert "index budget" not in out


class TestTheEmptyCases:
    @pytest.mark.asyncio
    async def test_no_runtime_says_how_to_configure_one(self) -> None:
        out = await _render(None)

        assert "No skills configured" in out

    @pytest.mark.asyncio
    async def test_an_empty_directory_says_so(self, layout: tuple[Path, Path]) -> None:
        project, _ = layout

        out = await _render(_runtime(project))

        assert "No skills found in" in out


class TestTheDetailView:
    @pytest.mark.asyncio
    async def test_the_header_says_where_and_what(self, layout: tuple[Path, Path]) -> None:
        project, _ = layout
        _skill(
            project,
            "commit-message",
            extra='version: "1.0"\nenforcement: required\nmetadata:\n  requires:\n    env: [HOME]\n',
        )

        out = await _render(_runtime(project), "commit-message", width=400)

        header = out.splitlines()[1]
        assert "commit-message" in header
        assert "v1.0" in header
        assert "required" in header
        assert str(project / "commit-message" / "SKILL.md") in out
        assert "Format · Checklist" in out
        assert "env HOME" in out

    @pytest.mark.asyncio
    async def test_the_body_is_rendered_not_dumped(self, layout: tuple[Path, Path]) -> None:
        project, _ = layout
        _skill(project, "commit-message")

        out = await _render(_runtime(project), "commit-message")

        assert "Format" in out
        assert "## " not in out
        assert "```" not in out
        assert "<type>: <subject>" in out
        assert "☐ Subject is imperative" in out

    @pytest.mark.asyncio
    async def test_a_section_is_shown_on_its_own_and_named(self, layout: tuple[Path, Path]) -> None:
        project, _ = layout
        _skill(project, "commit-message")

        out = await _render(_runtime(project), "commit-message#Checklist")

        assert "commit-message #Checklist" in out
        assert "Subject is imperative" in out
        assert "Intro line." not in out

    @pytest.mark.asyncio
    async def test_an_unknown_section_lists_the_real_ones(self, layout: tuple[Path, Path]) -> None:
        project, _ = layout
        _skill(project, "commit-message")

        with pytest.raises(SkillNotFoundError) as raised:
            await _render(_runtime(project), "commit-message#Nope")

        assert (
            str(raised.value) == "No section 'Nope' in commit-message. Sections: Format · Checklist"
        )

    @pytest.mark.asyncio
    async def test_a_misspelled_name_suggests_the_close_one(
        self, layout: tuple[Path, Path]
    ) -> None:
        project, _ = layout
        _skill(project, "commit-message")

        with pytest.raises(SkillNotFoundError) as raised:
            await _render(_runtime(project), "comit-message")

        assert str(raised.value) == "Skill not found: comit-message. Did you mean commit-message?"

    @pytest.mark.asyncio
    async def test_a_name_with_no_close_match_is_reported(self, layout: tuple[Path, Path]) -> None:
        """Without a hint at `/skills`: the message is printed by `domyn skill show` too."""
        project, _ = layout
        _skill(project, "commit-message")

        with pytest.raises(SkillNotFoundError) as raised:
            await _render(_runtime(project), "zzz")

        assert str(raised.value) == "Skill not found: zzz."

    @pytest.mark.asyncio
    async def test_an_unreadable_file_is_reported(
        self, layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project, _ = layout
        _skill(project, "commit-message")

        async def unreadable(*args: object, **kwargs: object) -> None:
            raise PermissionError

        # On the class: the runtime is a pydantic model, which refuses a new attribute.
        monkeypatch.setattr(SkillsRuntime, "fetch", unreadable)

        with pytest.raises(SkillNotFoundError) as raised:
            await _render(_runtime(project), "commit-message")

        assert str(raised.value) == "Cannot read skill 'commit-message': PermissionError"

    @pytest.mark.asyncio
    async def test_the_name_is_matched_regardless_of_case(self, layout: tuple[Path, Path]) -> None:
        project, _ = layout
        _skill(project, "Tone")

        out = await _render(_runtime(project), "tone#Format")

        assert "Tone #Format" in out
