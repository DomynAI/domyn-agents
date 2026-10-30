"""Tests for the skills subsystem."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import pytest

from domyn.agents.skills import (
    AllSkillsSelector,
    ExplicitSelector,
    IndexSelector,
    InMemorySkillSource,
    LoadedSkills,
    LocalDirSkillSource,
    SkillBudget,
    SkillContent,
    SkillDescriptor,
    SkillIndex,
    SkillNotFoundError,
    SkillParseError,
    SkillQuery,
    SkillRef,
    SkillRequiredError,
    SkillSource,
    SkillsRuntime,
    clip_text,
    extract_section,
    parse_skill_text,
    section_names,
)
from domyn.agents.skills.integration import (
    _SEARCH_DESCRIPTION,
    _VIEW_DESCRIPTION,
    attach_skills,
    build_skill_view_tool,
)
from domyn.agents.skills.sources import narrow_to_section

SQL_SKILL = """---
name: sql-style
description: Internal SQL conventions
version: "3"
---
Always use CTEs.

## examples
SELECT 1;

## anti-patterns
Avoid SELECT *.
"""

NEEDS_BINARY = """---
name: needs-binary
description: Requires a missing binary
metadata:
  requires:
    bins: [definitely-not-a-real-binary-xyz]
---
body
"""

REQUIRED_SKILL = """---
name: compliance
description: Mandatory rules
enforcement: required
metadata:
  requires:
    bins: [definitely-not-a-real-binary-xyz]
---
never disclose internal data
"""

# Named to sort *last*, because that is the condition the index budget used to lose
# a guardrail on: the walk drops whatever it reaches after the cap, and order is
# alphabetical within a directory (and mapping order for InMemorySkillSource).
REQUIRED_LAST_UNAVAILABLE = """---
name: zz-compliance
description: Mandatory rules
enforcement: required
metadata:
  requires:
    bins: [definitely-not-a-real-binary-xyz]
---
never disclose internal data
"""

REQUIRED_LAST_AVAILABLE = """---
name: zz-compliance
description: Mandatory rules
enforcement: required
---
never disclose internal data
"""


def _advisory(name: str) -> str:
    return f"---\nname: {name}\ndescription: Advisory {name}\n---\nbody of {name}\n"


def _skill_with_files(root: Path, name: str, entries: list[str], front: str = "") -> Path:
    """``<root>/<name>/SKILL.md`` listing ``entries`` under ``external_files:``."""
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    listed = "".join(f"  - {entry}\n" for entry in entries)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: d\n{front}external_files:\n{listed}---\nbody\n",
        encoding="utf-8",
    )
    return folder


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    """A directory exercising both layouts: <skill>/SKILL.md and flat <skill>.md."""
    nested = tmp_path / "sql-style"
    nested.mkdir()
    (nested / "SKILL.md").write_text(SQL_SKILL, encoding="utf-8")
    (tmp_path / "needs-binary.md").write_text(NEEDS_BINARY, encoding="utf-8")
    return tmp_path


class TestParser:
    def test_parses_frontmatter_and_sections(self) -> None:
        descriptor, content = parse_skill_text(SQL_SKILL, fallback_name="x")
        assert descriptor.name == "sql-style"
        assert descriptor.description == "Internal SQL conventions"
        assert descriptor.version == "3"
        assert descriptor.sections == ["examples", "anti-patterns"]
        assert content.body.startswith("Always use CTEs.")

    def test_missing_description_is_derived_not_fatal(self) -> None:
        """A skill without a description still shows up, so it can be fixed."""
        descriptor, _ = parse_skill_text("# Title\n\nfirst useful line", fallback_name="fb")
        assert descriptor.name == "fb"
        assert descriptor.description

    def test_no_frontmatter_is_accepted(self) -> None:
        descriptor, content = parse_skill_text("just prose", fallback_name="plain")
        assert descriptor.name == "plain"
        assert content.body == "just prose"

    def test_broken_yaml_raises_readable_error(self) -> None:
        with pytest.raises(SkillParseError, match="invalid YAML frontmatter"):
            parse_skill_text("---\n: : :\n---\nbody", fallback_name="bad")

    def test_section_extraction(self) -> None:
        _, content = parse_skill_text(SQL_SKILL, fallback_name="x")
        assert extract_section(content.body, "examples") == "SELECT 1;"
        assert extract_section(content.body, "EXAMPLES") == "SELECT 1;"
        assert extract_section(content.body, "nope") is None

    def test_section_names_deduplicates(self) -> None:
        assert section_names("## a\nx\n## b\ny\n## a\nz") == ["a", "b"]


class TestLocalDirSource:
    @pytest.mark.asyncio
    async def test_indexes_both_layouts(self, skills_dir: Path) -> None:
        source = LocalDirSkillSource(paths=[skills_dir])
        index = await source.index(SkillQuery())
        assert {s.name for s in index.skills} == {"sql-style", "needs-binary"}

    @pytest.mark.asyncio
    async def test_unmet_requirement_marks_unavailable_not_hidden(self, skills_dir: Path) -> None:
        """An unavailable skill must stay visible with a reason: hiding it silently
        makes the situation undebuggable."""
        index = await LocalDirSkillSource(paths=[skills_dir]).index(SkillQuery())
        needs = index.by_name("needs-binary")
        assert needs is not None
        assert needs.available is False
        assert "definitely-not-a-real-binary-xyz" in (needs.unavailable_reason or "")

    @pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested"])
    @pytest.mark.asyncio
    async def test_a_missing_external_file_marks_unavailable_not_hidden(
        self, tmp_path: Path, nested: bool
    ) -> None:
        if nested:
            folder = tmp_path / "report"
            folder.mkdir()
            (folder / "SKILL.md").write_text(
                "---\nname: report\ndescription: d\n"
                "metadata:\n  requires:\n    external_files: [gone.py]\n---\nbody\n",
                encoding="utf-8",
            )
        else:
            _skill_with_files(tmp_path, "report", ["gone.py"])

        index = await LocalDirSkillSource(paths=[tmp_path]).index(SkillQuery())

        report = index.by_name("report")
        assert report is not None
        assert report.available is False
        assert report.unavailable_reason == "file 'gone.py' (does not exist)"

    @pytest.mark.asyncio
    async def test_a_skill_with_every_external_file_there_is_available(
        self, tmp_path: Path
    ) -> None:
        folder = _skill_with_files(tmp_path, "report", ["scripts/", "../shared.py"])
        (folder / "scripts").mkdir()
        (tmp_path / "shared.py").write_text("x", encoding="utf-8")

        index = await LocalDirSkillSource(paths=[tmp_path]).index(SkillQuery())

        report = index.by_name("report")
        assert report is not None and report.available is True

    @pytest.mark.asyncio
    async def test_an_absolute_external_file_is_unusable(self, tmp_path: Path) -> None:
        _skill_with_files(tmp_path, "report", [str(tmp_path / "abs.py")])
        (tmp_path / "abs.py").write_text("x", encoding="utf-8")

        index = await LocalDirSkillSource(paths=[tmp_path]).index(SkillQuery())

        report = index.by_name("report")
        assert report is not None and report.available is False
        assert "(absolute path)" in (report.unavailable_reason or "")

    @pytest.mark.asyncio
    async def test_a_missing_file_and_a_missing_binary_are_both_said(self, tmp_path: Path) -> None:
        _skill_with_files(
            tmp_path,
            "report",
            ["gone.py"],
            front="metadata:\n  requires:\n    bins: [definitely-not-a-real-binary-xyz]\n",
        )

        index = await LocalDirSkillSource(paths=[tmp_path]).index(SkillQuery())

        report = index.by_name("report")
        assert report is not None and report.unavailable_reason is not None
        reason = report.unavailable_reason
        assert "binary 'definitely-not-a-real-binary-xyz'" in reason
        assert "file 'gone.py' (does not exist)" in reason

    @pytest.mark.asyncio
    async def test_external_files_are_checked_again_on_every_load(self, tmp_path: Path) -> None:
        """A file restored mid-session counts from the next turn, with no restart."""
        folder = _skill_with_files(tmp_path, "report", ["helper.py"])
        source = LocalDirSkillSource(paths=[tmp_path])
        before = (await source.index(SkillQuery())).by_name("report")

        (folder / "helper.py").write_text("x", encoding="utf-8")
        after = (await source.index(SkillQuery())).by_name("report")

        assert before is not None and before.available is False
        assert after is not None and after.available is True

    @pytest.mark.asyncio
    async def test_fetch_full_and_section(self, skills_dir: Path) -> None:
        source = LocalDirSkillSource(paths=[skills_dir])
        full = await source.fetch(SkillRef(name="sql-style"), SkillQuery())
        assert full.partial is False
        assert "Always use CTEs." in full.body

        part = await source.fetch(SkillRef(name="sql-style", section="examples"), SkillQuery())
        assert part.partial is True
        assert part.body == "SELECT 1;"

    @pytest.mark.asyncio
    async def test_fetch_unknown_raises(self, skills_dir: Path) -> None:
        with pytest.raises(SkillNotFoundError):
            await LocalDirSkillSource(paths=[skills_dir]).fetch(SkillRef(name="nope"), SkillQuery())

    @pytest.mark.asyncio
    async def test_missing_directory_is_not_an_error(self, tmp_path: Path) -> None:
        index = await LocalDirSkillSource(paths=[tmp_path / "absent"]).index(SkillQuery())
        assert index.skills == []

    @pytest.mark.asyncio
    async def test_malformed_skill_does_not_hide_the_others(self, skills_dir: Path) -> None:
        (skills_dir / "broken.md").write_text("---\n: : :\n---\nx", encoding="utf-8")
        index = await LocalDirSkillSource(paths=[skills_dir]).index(SkillQuery())
        assert "sql-style" in {s.name for s in index.skills}

    @pytest.mark.asyncio
    async def test_earlier_path_wins_on_name_collision(self, tmp_path: Path) -> None:
        first, second = tmp_path / "a", tmp_path / "b"
        for directory, body in ((first, "from-first"), (second, "from-second")):
            directory.mkdir()
            (directory / "dup.md").write_text(
                f"---\nname: dup\ndescription: d\n---\n{body}\n", encoding="utf-8"
            )
        source = LocalDirSkillSource(paths=[first, second])
        content = await source.fetch(SkillRef(name="dup"), SkillQuery())
        assert content.body == "from-first"


class TestSelectors:
    @pytest.mark.asyncio
    async def test_index_selector_expands_nothing(self, skills_dir: Path) -> None:
        """Progressive disclosure: the catalogue is visible, bodies are not loaded."""
        runtime = SkillsRuntime(source=LocalDirSkillSource(paths=[skills_dir]))
        selection, contents = await runtime.resolve(SkillQuery())
        assert len(selection.index) == 2
        assert contents == []

    @pytest.mark.asyncio
    async def test_all_selector_skips_unavailable(self, skills_dir: Path) -> None:
        runtime = SkillsRuntime(
            source=LocalDirSkillSource(paths=[skills_dir]), selector=AllSkillsSelector()
        )
        _, contents = await runtime.resolve(SkillQuery())
        assert [c.name for c in contents] == ["sql-style"]

    @pytest.mark.asyncio
    async def test_explicit_selector_expands_only_requested(self, skills_dir: Path) -> None:
        runtime = SkillsRuntime(
            source=LocalDirSkillSource(paths=[skills_dir]), selector=ExplicitSelector()
        )
        query = SkillQuery(context={"requested_skills": ["sql-style", "unknown"]})
        _, contents = await runtime.resolve(query)
        assert [c.name for c in contents] == ["sql-style"]

    def test_selectors_self_register_by_type(self) -> None:
        from domyn.agents.skills import SkillSelector

        assert {"index", "all", "explicit"} <= set(SkillSelector._registry)
        assert IndexSelector().type == "index"


class TestBudget:
    @pytest.mark.asyncio
    async def test_body_budget_skips_oversized_bodies(self, skills_dir: Path) -> None:
        runtime = SkillsRuntime(
            source=LocalDirSkillSource(paths=[skills_dir]),
            selector=AllSkillsSelector(),
            budget=SkillBudget(max_body_chars=1),
        )
        _, contents = await runtime.resolve(SkillQuery())
        assert contents == []

    @pytest.mark.asyncio
    async def test_expansion_cap(self, tmp_path: Path) -> None:
        for i in range(8):
            (tmp_path / f"s{i}.md").write_text(
                f"---\nname: s{i}\ndescription: d{i}\n---\nbody{i}\n", encoding="utf-8"
            )
        runtime = SkillsRuntime(
            source=LocalDirSkillSource(paths=[tmp_path]),
            selector=AllSkillsSelector(),
            budget=SkillBudget(max_expansions_per_turn=3),
        )
        _, contents = await runtime.resolve(SkillQuery())
        assert len(contents) == 3


def _many(count: int, description: str) -> InMemorySkillSource:
    return InMemorySkillSource(
        skills={
            f"s{n:02d}": f"---\nname: s{n:02d}\ndescription: {description}\n---\nbody\n"
            for n in range(count)
        }
    )


LONG_DESCRIPTION = " ".join(["Use when the task involves something rather specific"] * 6)


class TestTheIndexIsShortenedNotCut:
    """Over budget, every advisory skill is shown in a shorter form; none disappears
    until even their names do not fit, and then the index says how many are missing."""

    @staticmethod
    async def _payload(runtime: SkillsRuntime) -> str:
        selection, contents = await runtime.resolve(SkillQuery())
        return runtime.build_payload(selection, contents)

    @pytest.mark.asyncio
    async def test_an_index_that_fits_is_shown_whole(self) -> None:
        runtime = SkillsRuntime(source=_many(3, LONG_DESCRIPTION))

        layout = runtime.layout((await runtime.index(SkillQuery())).skills)

        assert layout.tier == "full"
        assert not layout.shortened
        assert LONG_DESCRIPTION in await self._payload(runtime)

    @pytest.mark.asyncio
    async def test_descriptions_are_shortened_first(self) -> None:
        runtime = SkillsRuntime(
            source=_many(10, LONG_DESCRIPTION), budget=SkillBudget(max_index_chars=1_500)
        )

        payload = await self._payload(runtime)

        assert runtime.layout((await runtime.index(SkillQuery())).skills).tier == "short"
        assert LONG_DESCRIPTION not in payload
        lines = [line for line in payload.splitlines() if line.startswith("- s")]
        assert len(lines) == 10, "a skill was dropped instead of shortened"
        assert all(line.endswith("…") for line in lines)
        assert all(len(line) <= len("- s00: ") + 120 for line in lines)

    @pytest.mark.asyncio
    async def test_then_only_names_are_shown(self) -> None:
        runtime = SkillsRuntime(
            source=_many(10, LONG_DESCRIPTION), budget=SkillBudget(max_index_chars=200)
        )

        payload = await self._payload(runtime)

        assert "Also available, by name only: s00, s01, s02" in payload
        assert "Use when" not in payload
        assert "more skills" not in payload, "every name fitted, nothing is missing"

    @pytest.mark.asyncio
    async def test_what_does_not_fit_even_by_name_is_counted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        runtime = SkillsRuntime(
            source=_many(10, LONG_DESCRIPTION), budget=SkillBudget(max_index_chars=60)
        )

        with caplog.at_level(logging.WARNING, logger="domyn.agents"):
            payload = await self._payload(runtime)

        layout = runtime.layout((await runtime.index(SkillQuery())).skills)
        assert layout.omitted, "the cap was not tight enough to test"
        assert f"+{len(layout.omitted)} more skills" in payload
        assert any("not listed" in record.getMessage() for record in caplog.records)

    @pytest.mark.asyncio
    async def test_one_tier_for_every_skill_whatever_the_order(self) -> None:
        """Which skill keeps its description must not come down to its filename."""
        short = "---\nname: {name}\ndescription: Short\n---\nb\n"
        long = f"---\nname: {{name}}\ndescription: {LONG_DESCRIPTION}\n---\nb\n"
        texts = {"a-long": long, "b-short": short, "c-long": long}
        budget = SkillBudget(max_index_chars=400)

        tiers = set()
        for order in (list(texts), list(reversed(texts))):
            source = InMemorySkillSource(
                skills={name: texts[name].format(name=name) for name in order}
            )
            runtime = SkillsRuntime(source=source, budget=budget)
            tiers.add(runtime.layout((await runtime.index(SkillQuery())).skills).tier)

        assert tiers == {"short"}

    def test_an_unavailable_skill_says_so_by_name_too(self) -> None:
        runtime = SkillsRuntime(
            source=InMemorySkillSource(skills={}), budget=SkillBudget(max_index_chars=80)
        )
        skills = [
            SkillDescriptor(name="ok", description=LONG_DESCRIPTION),
            SkillDescriptor(name="gone", description=LONG_DESCRIPTION, available=False),
        ]

        rendered = runtime.render_index(runtime.layout(skills))

        assert "ok, gone (unavailable)" in rendered

    def test_the_description_length_is_configurable(self) -> None:
        runtime = SkillsRuntime.model_validate(
            {
                "source": {"type": "memory", "skills": {}},
                "budget": {"max_index_chars": 100, "max_description_chars": 30},
            }
        )
        skills = [SkillDescriptor(name="x", description=LONG_DESCRIPTION)]

        assert runtime.render_index(runtime.layout(skills)).splitlines()[1] == (
            "- x: Use when the task involves…"
        )


class TestClipText:
    def test_short_text_is_untouched(self) -> None:
        assert clip_text("short", 10) == "short"

    def test_cut_at_a_word_and_marked(self) -> None:
        assert clip_text("one two three four", 12) == "one two…"

    def test_one_long_word_is_cut_inside_it(self) -> None:
        assert clip_text("abcdefghijklmnop", 6) == "abcde…"

    def test_never_longer_than_the_limit(self) -> None:
        for limit in range(30):
            assert len(clip_text(LONG_DESCRIPTION, limit)) <= limit


class TestIndexSections:
    """Whether the index lists each skill's `##` sections is a toggle, off by default."""

    @staticmethod
    async def _payload(runtime: SkillsRuntime) -> str:
        selection, contents = await runtime.resolve(SkillQuery())
        return runtime.build_payload(selection, contents)

    @pytest.mark.asyncio
    async def test_are_left_out_by_default(self, skills_dir: Path) -> None:
        runtime = SkillsRuntime(source=LocalDirSkillSource(paths=[skills_dir]))
        payload = await self._payload(runtime)
        assert "sql-style: Internal SQL conventions" in payload
        assert "[sections:" not in payload

    @pytest.mark.asyncio
    async def test_are_listed_when_turned_on(self, skills_dir: Path) -> None:
        runtime = SkillsRuntime(source=LocalDirSkillSource(paths=[skills_dir]), index_sections=True)
        assert "[sections: examples, anti-patterns]" in await self._payload(runtime)

    def test_an_unavailable_skill_says_why_either_way(self) -> None:
        descriptor = SkillDescriptor(
            name="n", description="d", available=False, unavailable_reason="missing bin"
        )
        for sections in (False, True):
            assert "[unavailable: missing bin]" in descriptor.index_line(sections=sections)

    @pytest.mark.asyncio
    async def test_the_budget_charges_what_is_shown(self, skills_dir: Path) -> None:
        """A budget measured on lines with sections, while the model is shown lines
        without, would shorten an index that fits."""
        index = await SkillsRuntime(source=LocalDirSkillSource(paths=[skills_dir])).index(
            SkillQuery()
        )
        short = sum(len(d.index_line()) for d in index.skills)
        long = sum(len(d.index_line(sections=True)) for d in index.skills)
        assert short < long
        budget = SkillBudget(max_index_chars=short)

        hidden = SkillsRuntime(source=LocalDirSkillSource(paths=[skills_dir]), budget=budget)
        assert hidden.layout(index.skills).tier == "full"

        shown = SkillsRuntime(
            source=LocalDirSkillSource(paths=[skills_dir]), budget=budget, index_sections=True
        )
        assert shown.layout(index.skills).tier != "full"

    def test_can_be_set_from_config(self, skills_dir: Path) -> None:
        """An agent's `skills:` mapping is validated into the runtime as-is."""
        runtime = SkillsRuntime.model_validate(
            {"source": {"type": "local_dir", "paths": [str(skills_dir)]}, "index_sections": True}
        )
        assert runtime.index_sections is True


class TestTheIndexBudgetNeverShortensAGuardrail:
    """``enforcement: required`` outranks ``max_index_chars``.

    The budget used to walk the index in order and drop everything past the cap,
    and the three ``raise_if_required_unavailable`` call sites all read the result.
    A required skill that sorted late was therefore dropped with only a warning:
    an unavailable one stopped refusing the turn, and an available one never
    reached the model. The budget now shortens rather than drops and is applied
    only to the rendered index, but a required skill is still never the one it
    shortens: a guardrail the model reads by name only is one it does not follow.
    """

    @staticmethod
    def _source(*, required: str, advisory: tuple[str, ...]) -> InMemorySkillSource:
        """Advisory skills first, the required one last — mapping order is index order."""
        skills = {name: _advisory(name) for name in advisory}
        skills["zz-compliance"] = required
        return InMemorySkillSource(skills=skills)

    @staticmethod
    async def _line_len(source: InMemorySkillSource, name: str) -> int:
        """The rendered length of one index line, read rather than guessed."""
        index = await SkillsRuntime(source=source).index(SkillQuery())
        descriptor = index.by_name(name)
        assert descriptor is not None
        return len(descriptor.index_line())

    @staticmethod
    async def _layout(runtime: SkillsRuntime) -> Any:
        return runtime.layout((await runtime.index(SkillQuery())).skills)

    @pytest.mark.asyncio
    async def test_a_required_skill_is_shown_whole_at_every_tier(self) -> None:
        source = self._source(
            required=REQUIRED_LAST_AVAILABLE, advisory=("a-one", "b-two", "c-three")
        )
        runtime = SkillsRuntime(source=source, budget=SkillBudget(max_index_chars=60))

        layout = await self._layout(runtime)
        rendered = runtime.render_index(layout)

        assert layout.tier == "names", "the cap was not tight enough to test"
        assert "- zz-compliance: Mandatory rules" in rendered

    @pytest.mark.asyncio
    async def test_the_refusal_fires_even_when_the_budget_is_tight(self) -> None:
        """The headline regression: an unavailable guardrail must still stop the turn."""
        source = self._source(
            required=REQUIRED_LAST_UNAVAILABLE, advisory=("a-one", "b-two", "c-three")
        )
        runtime = SkillsRuntime(source=source, budget=SkillBudget(max_index_chars=60))
        query = SkillQuery()

        loaded = await runtime.load(query)

        with pytest.raises(SkillRequiredError, match="zz-compliance"):
            await runtime.select(loaded, query)

    @pytest.mark.asyncio
    async def test_an_available_required_skill_still_reaches_the_payload(self) -> None:
        source = self._source(
            required=REQUIRED_LAST_AVAILABLE, advisory=("a-one", "b-two", "c-three")
        )
        runtime = SkillsRuntime(
            source=source,
            selector=AllSkillsSelector(),
            budget=SkillBudget(max_index_chars=50),
        )
        query = SkillQuery()

        selection, contents = await runtime.select(await runtime.load(query), query)

        assert "zz-compliance" in [descriptor.name for descriptor in selection.index]
        assert "zz-compliance" in runtime.build_payload(selection, contents)

    @pytest.mark.asyncio
    async def test_required_skills_still_spend_the_budget_they_use(self) -> None:
        """Counted, not exempted — otherwise the cap grows with its own input."""
        source = self._source(
            required=REQUIRED_LAST_AVAILABLE, advisory=("a-one", "b-two", "c-three")
        )
        runtime = SkillsRuntime(
            source=source,
            budget=SkillBudget(max_index_chars=await self._line_len(source, "zz-compliance")),
        )

        layout = await self._layout(runtime)

        assert [descriptor.name for descriptor in layout.listed] == ["zz-compliance"]
        assert sorted(layout.omitted) == ["a-one", "b-two", "c-three"]

    @pytest.mark.asyncio
    async def test_required_skills_alone_may_exceed_the_budget_and_are_kept(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Over budget is a warning, never a drop — and never a raise from here."""
        runtime = SkillsRuntime(
            source=InMemorySkillSource(
                skills={
                    "zz-compliance": REQUIRED_LAST_AVAILABLE,
                    "compliance": REQUIRED_SKILL,
                }
            ),
            budget=SkillBudget(max_index_chars=10),
        )

        with caplog.at_level(logging.WARNING, logger="domyn.agents"):
            layout = await self._layout(runtime)

        assert sorted(descriptor.name for descriptor in layout.listed) == [
            "compliance",
            "zz-compliance",
        ]
        # Nothing was hidden: the index simply costs more than the cap.
        assert layout.omitted == []
        assert not layout.shortened
        assert any(
            "required skills alone need" in record.getMessage() for record in caplog.records
        ), "an index over its own budget has to say so"

    @pytest.mark.asyncio
    async def test_the_index_is_not_reordered(self) -> None:
        """Guards against re-solving this by sorting guardrails to the front."""
        source = self._source(required=REQUIRED_LAST_AVAILABLE, advisory=("a-first", "m-second"))
        # Room for the required line and both names (49 chars with the line's lead-in),
        # not for their two whole lines (56).
        fits = await self._line_len(source, "zz-compliance") + 52
        runtime = SkillsRuntime(source=source, budget=SkillBudget(max_index_chars=fits))

        layout = await self._layout(runtime)

        assert layout.tier == "names", "the cap did not force the names-only tier"

        assert [descriptor.name for descriptor in layout.listed] == [
            "a-first",
            "m-second",
            "zz-compliance",
        ]


class TestInMemorySource:
    @pytest.mark.asyncio
    async def test_works_without_filesystem(self) -> None:
        """Backs the standalone-SDK promise: skills with no files and no server."""
        runtime = SkillsRuntime(source=InMemorySkillSource(skills={"sql": SQL_SKILL}))
        index = await runtime.index(SkillQuery())
        assert [s.name for s in index.skills] == ["sql-style"]
        content = await runtime.fetch(SkillRef(name="sql-style"), SkillQuery())
        assert "CTE" in content.body

    @pytest.mark.asyncio
    async def test_external_files_do_not_make_a_skill_without_a_path_unavailable(self) -> None:
        """Nothing to resolve them against: there is no SKILL.md on disk."""
        text = "---\nname: s\ndescription: d\nexternal_files:\n  - gone.py\n---\nbody\n"
        index = await InMemorySkillSource(skills={"s": text}).index(SkillQuery())

        assert [(s.name, s.available) for s in index.skills] == [("s", True)]


class TestAttachToAgent:
    """``attach_skills`` sets the runtime and ``skill_view`` together, and nothing else."""

    class _FakeAgent:
        name = "A"

        def __init__(self) -> None:
            self.tools: list[object] = []

    class _FakeAgentWithSkills(_FakeAgent):
        def __init__(self, skills: SkillsRuntime | None = None, provider: Any = None) -> None:
            super().__init__()
            self.skills = skills
            self.llm_provider = provider

    def test_attaches_tools_that_do_not_carry_the_index(self) -> None:
        """The planner injects the catalogue every turn; the tools must not repeat it."""
        agent = self._FakeAgentWithSkills()
        runtime = SkillsRuntime(source=InMemorySkillSource(skills={"sql": SQL_SKILL}))
        attach_skills(agent, runtime)

        assert [t.name for t in agent.tools] == ["skill_view", "skill_search"]  # type: ignore[attr-defined]
        assert agent.tools[0].declaration.description == _VIEW_DESCRIPTION  # type: ignore[attr-defined]
        assert agent.tools[1].declaration.description == _SEARCH_DESCRIPTION  # type: ignore[attr-defined]

    def test_is_idempotent(self) -> None:
        agent = self._FakeAgentWithSkills()
        runtime = SkillsRuntime(source=InMemorySkillSource(skills={"sql": SQL_SKILL}))
        attach_skills(agent, runtime)
        attach_skills(agent, runtime)
        assert [t.name for t in agent.tools] == ["skill_view", "skill_search"]

    def test_sets_the_runtime_on_the_agent(self) -> None:
        agent = self._FakeAgentWithSkills()
        runtime = SkillsRuntime(source=InMemorySkillSource(skills={"sql": SQL_SKILL}))
        attach_skills(agent, runtime)
        assert agent.skills is runtime

    @pytest.mark.asyncio
    async def test_replacing_the_runtime_rebinds_the_tool(self) -> None:
        """The split this prevents: shown skills by one runtime, fetching from another."""
        agent = self._FakeAgentWithSkills()
        attach_skills(agent, SkillsRuntime(source=InMemorySkillSource(skills={"sql": SQL_SKILL})))

        other = SQL_SKILL.replace("sql-style", "other-style")
        replacement = SkillsRuntime(source=InMemorySkillSource(skills={"other": other}))
        attach_skills(agent, replacement)

        assert agent.skills is replacement
        assert [t.name for t in agent.tools] == ["skill_view", "skill_search"]
        call = agent.tools[0]._actual_callable  # type: ignore[attr-defined]
        assert "Always use CTEs." in await call(name="other-style")
        assert await call(name="sql-style") == "Skill not found: sql-style"

    def test_an_agent_without_a_skills_field_is_left_untouched(self) -> None:
        """The catalogue reaches the model only through ``agent.skills``: without
        the field, ``skill_view`` would be a tool for skills no model is shown."""
        agent = self._FakeAgent()
        runtime = SkillsRuntime(source=InMemorySkillSource(skills={"sql": SQL_SKILL}))
        attach_skills(agent, runtime)

        assert agent.tools == []
        assert not hasattr(agent, "skills")

    def test_the_selector_is_offered_the_agent_provider(self) -> None:
        provider = object()
        seen: list[Any] = []
        runtime = SkillsRuntime(source=InMemorySkillSource(skills={"sql": SQL_SKILL}))
        object.__setattr__(runtime.selector, "bind_llm", seen.append)

        attach_skills(self._FakeAgentWithSkills(provider=provider), runtime)
        assert seen == [provider]

    def test_an_unavailable_required_skill_does_not_raise_here(self) -> None:
        """The guardrail is the runner's, once per turn; wiring checks nothing."""
        agent = self._FakeAgentWithSkills()
        runtime = SkillsRuntime(source=InMemorySkillSource(skills={"c": REQUIRED_SKILL}))
        attach_skills(agent, runtime)
        assert agent.skills is runtime


class TestRequiredEnforcementOnTheTurnPath:
    """``enforcement: required`` also bites on the path a *turn* takes.

    ``attach_skills`` does not check it — it only wires the runtime and the tool.
    Inside a turn the runner calls
    ``SkillsRuntime.load`` and the agent calls ``SkillsRuntime.select``; the
    check that covers that path is ``raise_if_required_unavailable`` at the top
    of ``select``, on the **index**. It has to be on the index and before
    selection: the ``fetch``-failure handler further down sees an unavailable
    skill never, because a source that preloads bodies does not call ``fetch``
    (both shipped sources preload) and every selector drops unavailable skills
    before expansion.
    """

    @pytest.mark.parametrize("selector", [AllSkillsSelector(), IndexSelector()])
    @pytest.mark.asyncio
    async def test_select_refuses_an_unavailable_required_skill(self, selector: Any) -> None:
        """Both selectors, because the check belongs *before* selection, not inside it.

        ``IndexSelector`` expands nothing at all and ``AllSkillsSelector`` filters
        on ``available``; a check that lives in either one is a check the other
        does not get.
        """
        runtime = SkillsRuntime(
            source=InMemorySkillSource(skills={"c": REQUIRED_SKILL}), selector=selector
        )
        query = SkillQuery(task="anything")
        loaded = await runtime.load(query)

        assert [(s.name, s.enforcement, s.available) for s in loaded.index.skills] == [
            ("compliance", "required", False)
        ], "the fixture stopped describing an unavailable required skill"

        with pytest.raises(SkillRequiredError, match="compliance"):
            await runtime.select(loaded, query)

    @pytest.mark.asyncio
    async def test_the_refusal_says_which_agent_which_skill_and_why(self) -> None:
        """The message contract, pinned so its three parts cannot drift apart.

        It is the only thing an operator gets: the CLI prints it verbatim, and
        the runner wraps it into the error event that ends the turn. Naming the
        skill without the reason sends them to ``/skills`` to find out why.
        """
        runtime = SkillsRuntime(source=InMemorySkillSource(skills={"c": REQUIRED_SKILL}))
        query = SkillQuery(agent_name="BillingAgent", task="anything")
        loaded = await runtime.load(query)

        with pytest.raises(SkillRequiredError) as caught:
            await runtime.select(loaded, query)

        message = str(caught.value)
        for fragment in ("BillingAgent", "compliance", "definitely-not-a-real-binary-xyz"):
            assert fragment in message, f"{fragment!r} missing from {message!r}"

    @pytest.mark.asyncio
    async def test_an_unavailable_advisory_skill_does_not_block(self) -> None:
        """The other half of the rule, and the invariant it must not break.

        An unmet requirement on an ``advisory`` skill leaves it *visible* and
        unusable, not hidden — the model is told the skill exists and why it
        cannot be used, which is a different thing from it never having existed.
        """
        runtime = SkillsRuntime(source=InMemorySkillSource(skills={"n": NEEDS_BINARY}))
        query = SkillQuery(agent_name="BillingAgent", task="anything")
        loaded = await runtime.load(query)

        selection, contents = await runtime.select(loaded, query)

        assert [s.name for s in selection.index] == ["needs-binary"]
        assert contents == [], "an unavailable skill must not be expanded"

    @pytest.mark.asyncio
    async def test_a_required_skill_with_a_missing_file_refuses_the_turn(
        self, tmp_path: Path
    ) -> None:
        _skill_with_files(tmp_path, "compliance", ["rules.py"], front="enforcement: required\n")
        runtime = SkillsRuntime(source=LocalDirSkillSource(paths=[tmp_path]))
        query = SkillQuery(task="anything")
        loaded = await runtime.load(query)

        with pytest.raises(SkillRequiredError, match=r"file 'rules\.py' \(does not exist\)"):
            await runtime.select(loaded, query)

    @pytest.mark.asyncio
    async def test_an_advisory_skill_with_a_missing_file_does_not_block(
        self, tmp_path: Path
    ) -> None:
        _skill_with_files(tmp_path, "report", ["helper.py"])
        runtime = SkillsRuntime(source=LocalDirSkillSource(paths=[tmp_path]))
        query = SkillQuery(task="anything")
        loaded = await runtime.load(query)

        selection, contents = await runtime.select(loaded, query)

        assert [s.name for s in selection.index] == ["report"]
        assert contents == [], "an unavailable skill must not be expanded"

    @pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-000 file regardless")
    @pytest.mark.asyncio
    async def test_an_unreadable_required_skill_cannot_be_known_to_be_required(
        self, tmp_path: Path
    ) -> None:
        """The limit of the check above, pinned so it is not mistaken for the same bug.

        ``enforcement`` is declared *inside* the file. A file that cannot be read
        cannot be known to have declared ``required``, so no check at any layer
        can catch this one — it is indistinguishable from a skill that was never
        written. ``load`` drops it with a warning and the turn runs without it.
        Asserted rather than xfailed because there is no fix to wait for; what a
        reader needs is to know the silence here is a different silence.
        """
        directory = tmp_path / "compliance"
        directory.mkdir()
        skill_file = directory / "SKILL.md"
        skill_file.write_text(REQUIRED_SKILL, encoding="utf-8")
        skill_file.chmod(0o000)

        runtime = SkillsRuntime(source=LocalDirSkillSource(paths=[str(tmp_path)]))
        loaded = await runtime.load(SkillQuery(task="anything"))

        assert loaded.index.skills == [], "an unreadable file somehow yielded a descriptor"


class TestSkillViewTool:
    def test_declares_section_as_optional(self) -> None:
        runtime = SkillsRuntime(source=InMemorySkillSource(skills={"sql": SQL_SKILL}))
        tool = build_skill_view_tool(runtime)
        by_name = {p.name: p for p in tool.declaration.parameters}
        assert by_name["name"].is_required is True
        assert by_name["section"].is_required is False

    @pytest.mark.asyncio
    async def test_returns_body_and_section(self) -> None:
        runtime = SkillsRuntime(source=InMemorySkillSource(skills={"sql": SQL_SKILL}))
        call = build_skill_view_tool(runtime)._actual_callable
        assert "Always use CTEs." in await call(name="sql-style")
        assert (await call(name="sql-style", section="examples")).endswith("SELECT 1;")

    @pytest.mark.asyncio
    async def test_missing_skill_is_reported_to_the_model_not_raised(self) -> None:
        """The model can recover from a bad name; a crash would end the turn."""
        runtime = SkillsRuntime(source=InMemorySkillSource(skills={"sql": SQL_SKILL}))
        call = build_skill_view_tool(runtime)._actual_callable
        assert await call(name="nope") == "Skill not found: nope"
        assert await call(name="sql-style", section="nope") == "Skill not found: sql-style#nope"


class TestSkillViewSpendsTheTurnLoad:
    """The preloaded-body branch of ``skill_view``.

    The runner loads every agent's skills once per turn; before this branch the
    tool ignored that and went back to the source on every call, which for a local
    directory meant re-walking and re-parsing every skill file to keep one body.

    The condition that matters is *which* thing is checked. Branching on "did
    anyone preload" rather than "is this body preloaded" looks equivalent and is
    not: a source whose bodies cost a round-trip returns an index with no bodies at
    all, so the first reading answers "yes" and then finds nothing.
    """

    BODY = "Always use CTEs.\n\n## examples\nSELECT 1;\n"

    class _CountingSource(SkillSource):
        """Records every read, so "the preload was used" is asserted, not hoped."""

        type: str = "counting_view"
        loads: int = 0
        fetches: int = 0
        with_bodies: bool = True

        async def load(self, query: SkillQuery) -> LoadedSkills:
            self.loads += 1
            index = SkillIndex(skills=[SkillDescriptor(name="sql-style", description="how we sql")])
            if not self.with_bodies:
                # The ``SkillSource.load`` default: a source whose bodies cost a
                # round-trip each returns none of them up front.
                return LoadedSkills(index=index)
            return LoadedSkills(
                index=index,
                bodies={
                    "sql-style": SkillContent(
                        name="sql-style", body=TestSkillViewSpendsTheTurnLoad.BODY
                    )
                },
            )

        async def index(self, query: SkillQuery) -> SkillIndex:
            return (await self.load(query)).index

        async def fetch(self, ref: SkillRef, query: SkillQuery) -> SkillContent:
            self.fetches += 1
            if ref.name != "sql-style":
                raise SkillNotFoundError(ref.name)
            return narrow_to_section(
                SkillContent(name=ref.name, body=TestSkillViewSpendsTheTurnLoad.BODY), ref
            )

    async def _preloaded(self, source: SkillSource) -> LoadedSkills:
        """What the runner puts in ``state.skills`` at the top of the turn."""
        loaded = await source.load(SkillQuery())
        source.loads = 0  # type: ignore[attr-defined]
        return loaded

    @pytest.mark.asyncio
    async def test_a_preloaded_body_never_reaches_the_source(self) -> None:
        """The point of the whole branch: no second read for a body already held."""
        source = self._CountingSource()
        loaded = await self._preloaded(source)
        call = build_skill_view_tool(SkillsRuntime(source=source))._actual_callable

        result = await call(name="sql-style", loaded_skills=loaded)

        assert "Always use CTEs." in result
        assert (source.loads, source.fetches) == (0, 0)

    @pytest.mark.asyncio
    async def test_a_section_is_sliced_from_the_preloaded_body(self) -> None:
        """L2 costs no read either: the slice is the one the source would have made."""
        source = self._CountingSource()
        loaded = await self._preloaded(source)
        call = build_skill_view_tool(SkillsRuntime(source=source))._actual_callable

        result = await call(name="sql-style", section="examples", loaded_skills=loaded)

        assert result.endswith("SELECT 1;")
        assert "(section: examples)" in result
        assert (source.loads, source.fetches) == (0, 0)

    @pytest.mark.asyncio
    async def test_a_missing_section_of_a_preloaded_body_is_reported_not_raised(self) -> None:
        """``narrow_to_section`` raises; the model must get the same string either way.

        The fallback path has always answered a bad section with this sentence.
        A preloaded path that raised instead would end the turn on a typo.
        """
        source = self._CountingSource()
        loaded = await self._preloaded(source)
        call = build_skill_view_tool(SkillsRuntime(source=source))._actual_callable

        assert await call(name="sql-style", section="nope", loaded_skills=loaded) == (
            "Skill not found: sql-style#nope"
        )

    @pytest.mark.asyncio
    async def test_an_index_only_source_still_fetches(self) -> None:
        """The regression guard for the condition itself.

        A source that serves bodies from somewhere the index does not reach
        returns an index with no bodies — the ``SkillSource.load`` default.
        ``LoadedSkills`` is then present but empty, so branching on the container
        rather than on the body made every such skill report itself missing.
        """
        source = self._CountingSource(with_bodies=False)
        loaded = await self._preloaded(source)
        assert loaded is not None and loaded.bodies == {}, "the shape this guards against"

        call = build_skill_view_tool(SkillsRuntime(source=source))._actual_callable
        result = await call(name="sql-style", loaded_skills=loaded)

        assert "Always use CTEs." in result
        assert source.fetches == 1

    @pytest.mark.asyncio
    async def test_a_name_absent_from_the_preload_falls_back_to_the_source(self) -> None:
        """A skill the turn's load did not see is still worth one look."""
        source = self._CountingSource()
        loaded = await self._preloaded(source)
        call = build_skill_view_tool(SkillsRuntime(source=source))._actual_callable

        assert await call(name="nope", loaded_skills=loaded) == "Skill not found: nope"
        assert source.fetches == 1, "the miss went to the source rather than being assumed"

    @pytest.mark.asyncio
    async def test_the_fallback_fetch_runs_on_the_runners_loop(self) -> None:
        """A source may hold loop-bound resources (a shared ``httpx.AsyncClient``, a
        cached task). A fetch moved to a fresh loop fails on them with "bound to a
        different event loop", so it has to run on the loop that called the tool."""
        caller = asyncio.get_running_loop()
        seen: list[asyncio.AbstractEventLoop] = []

        class _LoopRecordingSource(self._CountingSource):  # type: ignore[name-defined]
            async def fetch(self, ref: SkillRef, query: SkillQuery) -> SkillContent:
                seen.append(asyncio.get_running_loop())
                return await super().fetch(ref, query)

        source = _LoopRecordingSource(with_bodies=False)
        tool = build_skill_view_tool(SkillsRuntime(source=source))

        # Through `execute`, as the runner calls it.
        result = await tool.execute(name="sql-style", loaded_skills=LoadedSkills())

        assert result.success, result.error
        assert seen == [caller]

    @pytest.mark.asyncio
    async def test_without_a_preload_the_source_is_still_the_answer(self) -> None:
        """A direct ``agent.execute`` outside a runner preloads nothing."""
        source = self._CountingSource()
        call = build_skill_view_tool(SkillsRuntime(source=source))._actual_callable

        assert "Always use CTEs." in await call(name="sql-style", loaded_skills=None)
        assert source.fetches == 1


class TestAgentDeclaredSkills:
    """S2: skills as a declared property of the agent, not CLI wiring."""

    @staticmethod
    def _provider() -> Any:
        from domyn.agents.llm import CustomLLMProvider

        return CustomLLMProvider(model_name="m", url="http://x")

    def _agent(self, skills: Any) -> Any:
        from domyn.agents import Agent

        return Agent(
            name="A",
            description="d",
            instruction="i",
            llm_provider=self._provider(),
            skills=skills,
        )

    def test_declaring_skills_attaches_only_the_tools(self, skills_dir: Path) -> None:
        """Declaring is enough: no initialize call the caller could forget.

        The tools need no I/O so they are built at construction. Reading the index
        does, and depends on who is asking, so the planner resolves it per turn
        and keeps it in a local — no hook, and nothing on the agent.
        """
        agent = self._agent({"source": {"type": "local_dir", "paths": [str(skills_dir)]}})
        assert [t.name for t in agent.tools] == ["skill_view", "skill_search"]
        assert agent.hooks == []

    def test_agent_without_skills_is_untouched(self) -> None:
        from domyn.agents import Agent

        agent = Agent(name="B", description="d", instruction="i", llm_provider=self._provider())
        assert agent.tools == []
        assert agent.hooks == []

    def test_source_and_selector_resolve_by_type(self, skills_dir: Path) -> None:
        agent = self._agent(
            {
                "source": {"type": "local_dir", "paths": [str(skills_dir)]},
                "selector": {"type": "all"},
                "budget": {"max_index_chars": 999},
            }
        )
        assert agent.skills is not None
        assert type(agent.skills.source).__name__ == "LocalDirSkillSource"
        assert type(agent.skills.selector).__name__ == "AllSkillsSelector"
        assert agent.skills.budget.max_index_chars == 999

    def test_unknown_type_raises_instead_of_defaulting(self) -> None:
        """A typo in `type:` must not silently produce different behaviour."""
        with pytest.raises(ValueError, match="unknown skill source type"):
            self._agent({"source": {"type": "nope"}})

    @pytest.mark.parametrize("field", ["source", "selector"])
    @pytest.mark.parametrize("value", ["local_dir", None, ["local_dir"], 42])
    def test_a_value_that_is_not_a_mapping_raises(self, field: str, value: Any) -> None:
        """Not stored as is, to fail on first use: ``source: local_dir`` (no
        nested ``type:``) is a mistake to report at load time."""
        config: dict[str, Any] = {"source": {"type": "memory"}, field: value}

        with pytest.raises(ValueError, match="valid dictionary or instance of"):
            self._agent(config)

    def test_attach_is_idempotent(self, skills_dir: Path) -> None:
        agent = self._agent({"source": {"type": "local_dir", "paths": [str(skills_dir)]}})
        agent._attach_skills()
        assert [t.name for t in agent.tools].count("skill_view") == 1

    @staticmethod
    def _turn_context() -> Any:
        from domyn.agents import BaseEvent, ExecutionEventType, Part
        from domyn.agents.core import InteractionContext

        return InteractionContext(
            conversation_id="c",
            user_id="u",
            agent_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="q")],
            ),
        )

    @pytest.mark.asyncio
    async def test_resolve_returns_the_payload_and_writes_nothing(self, skills_dir: Path) -> None:
        """The payload is a return value now, published nowhere.

        Not on the Agent (one instance serves concurrent conversations) and not
        on context.state (the conversation's, reseeded into sub-agents').
        """
        from domyn.agents.skills.turn import build_skills_payload

        agent = self._agent({"source": {"type": "local_dir", "paths": [str(skills_dir)]}})
        context = self._turn_context()

        # No `loaded`: nobody preloaded, so the runtime loads for itself. That is
        # the path a direct agent.execute() outside a runner takes.
        payload = await build_skills_payload(agent, context)

        assert "sql-style" in payload
        assert context.state == {}
        assert not hasattr(agent, "skills_payload")

    @pytest.mark.asyncio
    async def test_agent_without_skills_resolves_to_empty(self) -> None:
        from domyn.agents import Agent
        from domyn.agents.skills.turn import build_skills_payload

        agent = Agent(name="B", description="d", instruction="i", llm_provider=self._provider())
        assert await build_skills_payload(agent, self._turn_context()) == ""

    @pytest.mark.asyncio
    async def test_a_mock_agent_resolves_to_empty(self) -> None:
        """`getattr(agent, "skills", None) is None` would NOT be enough.

        BaseAgent declares no `skills`, and the planner suites build agents as
        bare MagicMocks, which answer every getattr with a truthy child mock.
        The guard is an isinstance check; this pins it.
        """
        from unittest.mock import MagicMock

        from domyn.agents.skills.turn import build_skills_payload

        assert await build_skills_payload(MagicMock(), self._turn_context()) == ""


class TestYamlConfiguration:
    """Skills must survive the loader, which drops unlisted keys silently."""

    @pytest.mark.asyncio
    async def test_per_agent_skills_load_from_yaml(self, tmp_path: Path, skills_dir: Path) -> None:
        """Each agent carries its own skills; the orchestrator needs none.

        This is the shape a multi-agent deployment uses: specialists with their own
        knowledge, coordinated by an agent that has none of its own.
        """
        from domyn.agents import load_application_from_file

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
    skills:
      source:
        type: local_dir
        paths: ["{skills_dir}"]
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
        assert root.name == "Orchestrator"
        assert root.skills is None, "the orchestrator declared none"

        expert = root.sub_agents[0]
        assert expert.skills is not None, (
            "skills were dropped by the loader — check the from_config kwargs whitelist"
        )
        assert [t.name for t in expert.tools] == ["skill_view", "skill_search"]
