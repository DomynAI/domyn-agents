"""Terminal views for skills: the ``/skills`` and ``/skill <name>`` commands.

The list is grouped by the directory a skill was read from, because "where is
this coming from" is the first question when a skill misbehaves, and with
several skill directories the name alone does not answer it. Every entry still
shows its file — relative to the absolute directory heading its group, since
the layout (``<name>/SKILL.md`` or ``<name>.md``) is not derivable from the name.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from rich.markdown import Heading, Markdown
from rich.padding import Padding
from rich.table import Table
from rich.text import Text
from rich.theme import Theme as RichTheme

from domyn.agents.skills import SkillNotFoundError, SkillQuery, SkillRef

if TYPE_CHECKING:
    from rich.console import Console, JustifyMethod
    from rich.markdown import MarkdownElement

    from domyn.agents.interfaces.cli.theme import Theme
    from domyn.agents.skills import (
        IndexLayout,
        SkillDescriptor,
        SkillIndex,
        SkillRequirements,
        SkillsRuntime,
    )

_NOT_CONFIGURED = (
    "No skills configured. Run `domyn setup` to install the built-in ones or load the skills "
    "in ./.domyn/skills/ and ./.agents/skills/."
)

# Past this a name column stops being a column and starts eating the description.
# Longer names fold onto a second line rather than pushing everything right.
_MAX_NAME_WIDTH = 28

# Prose wider than this is hard to read, however wide the terminal is.
_MAX_BODY_WIDTH = 100

_INDENT = 2


def _display_path(path: Path) -> str:
    """The absolute, resolved path: unambiguous, and pasteable into any shell or editor."""
    return str(path.expanduser().resolve())


class _Group:
    def __init__(self, label: str, root: Path | None) -> None:
        self.label = label
        self.root = root
        self.skills: list[SkillDescriptor] = []

    def file_label(self, skill: SkillDescriptor) -> str | None:
        if skill.path is None:
            return None
        if self.root is None:
            return _display_path(skill.path)
        relative = skill.path.expanduser().resolve().relative_to(self.root)
        # A group whose root is the skill's own file (a loose skill listed singly):
        # the relative path is "." and says nothing; the file's name does.
        return str(relative) if str(relative) != "." else skill.path.name


def _group(skills: list[SkillDescriptor], roots: list[Path], fallback: str) -> list[_Group]:
    """One group per configured directory, in precedence order, plus one for the rest.

    A skill goes to the deepest root that contains it, so nested directories
    still get the shortest file label. Skills with no file, or one outside every
    root, share a trailing group labelled by the source itself.
    """
    groups = [_Group(_display_path(root), root.expanduser().resolve()) for root in roots]
    other = _Group(fallback, None)
    for skill in skills:
        owner: _Group | None = None
        depth = -1
        if skill.path is not None:
            resolved = skill.path.expanduser().resolve()
            for group in groups:
                root = group.root
                if root is not None and resolved.is_relative_to(root) and len(root.parts) > depth:
                    owner, depth = group, len(root.parts)
        (owner or other).skills.append(skill)
    return [*groups, other]


async def render_skills(
    console: Console,
    theme: Theme,
    runtime: SkillsRuntime | None,
    name: str | None = None,
) -> None:
    """List skills, or show one in full when ``name`` is given.

    Async because it is called from inside the REPL's running event loop:
    ``asyncio.run`` would raise there.

    Raises `SkillNotFoundError` when ``name`` is given and cannot be shown (see
    `_show_one`); listing never raises.
    """
    if runtime is None:
        console.print(_NOT_CONFIGURED, style=theme.style("meta"), highlight=False)
        return

    if name:
        await _show_one(console, theme, runtime, name)
        return

    index = await runtime.index(SkillQuery())
    if not index.skills:
        console.print(
            f"No skills found in {runtime.source.describe()}.",
            style=theme.style("meta"),
            highlight=False,
        )
        return

    sym = theme.symbols
    roots = list(getattr(runtime.source, "paths", None) or [])
    groups = _group(index.skills, roots, runtime.source.describe())

    unavailable = sum(1 for s in index.skills if not s.available)
    header = Text("Skills", style=theme.style("accent_bold"))
    counts = f"{len(index.skills) - unavailable} available"
    if unavailable:
        counts += f" {sym.separator} {unavailable} unavailable"
    header.append(f"  {counts}", style=theme.style("meta"))
    console.print()
    _line(console, header)

    # One width for every group, so the descriptions line up down the whole list
    # and not only within one directory. Capped by the terminal too: on a narrow
    # one a long name should fold, not leave the description a sliver.
    name_width = min(
        _MAX_NAME_WIDTH,
        max(12, console.width // 4),
        max(max(len(s.name) for s in index.skills), len("required")),
    )
    marker_width = len(sym.denied) if unavailable else 0

    for group in groups:
        if not group.skills:
            continue
        console.print()
        _line(console, Text(group.label, style=theme.style("accent")))
        table = Table.grid(padding=(0, 2))
        if marker_width:
            table.add_column(width=marker_width)
        table.add_column(width=name_width, overflow="fold")
        table.add_column(ratio=1, overflow="fold")
        for skill in group.skills:
            cells = _row(theme, skill, group.file_label(skill))
            table.add_row(*(cells if marker_width else cells[1:]))
        # Unavailable skills put their marker in the gutter, so names stay aligned
        # with or without one.
        indent = _INDENT if marker_width else _INDENT * 2
        console.print(Padding(table, (0, 0, 0, indent)))

    empty = [g.label for g in groups if g.root is not None and not g.skills]
    if empty:
        console.print()
        _line(
            console,
            Text(f"also searched: {', '.join(empty)} (none found)", style=theme.style("meta")),
        )

    layout = runtime.layout(index.skills)
    if layout.shortened:
        # Never let a budget silently change what the model sees: an author whose
        # skill is not being picked needs to know it was shown by name only.
        console.print()
        _line(console, Text(_layout_note(sym.denied, layout), style=theme.style("warning")))

    console.print()
    _line(
        console,
        Text(
            f"/skill <name> to read one {sym.separator} /skill <name>#<section> for one section",
            style=theme.style("meta"),
        ),
    )
    console.print()


def _layout_note(marker: str, layout: IndexLayout) -> str:
    """What the index budget does to the model's view of these skills, in one line."""
    if layout.tier == "short":
        note = "over the index budget: the model sees shortened descriptions"
    else:
        note = "over the index budget: the model sees skills by name only"
    if layout.omitted:
        note += f", and not at all: {', '.join(layout.omitted)}"
    return f"{marker} {note}"


def _line(console: Console, text: Text) -> None:
    """One indented block of text; wrapped lines keep the indent."""
    console.print(Padding(text, (0, 0, 0, _INDENT)), highlight=False)


def _row(theme: Theme, skill: SkillDescriptor, file_label: str | None) -> tuple[Text, Text, Text]:
    """``(marker, name, details)`` for one list entry.

    Built from plain text, never markup: descriptions are written by whoever
    wrote the skill, and a ``[bold]`` in one must print as itself.
    """
    sym = theme.symbols
    warning = theme.style("warning")
    meta = theme.style("meta")

    marker = Text(sym.denied, style=warning) if not skill.available else Text()

    name = Text(skill.name, style=theme.style("emphasis") if skill.available else warning)
    if skill.enforcement == "required":
        name.append("\nrequired", style=warning)

    details = Text(skill.description)
    if not skill.available:
        reason = skill.unavailable_reason or "requirements not met"
        details.append(f"\nunavailable: {reason}", style=warning)
    extras: list[str] = []
    if skill.sections:
        extras.append(f" {sym.separator} ".join(skill.sections))
    if file_label:
        extras.append(file_label)
    if extras:
        # A plain gap, not another separator: the sections are already joined by
        # one, and a second would make the file read as one more section.
        details.append("\n" + "   ".join(extras), style=meta)
    return marker, name, details


def find_skill(index: SkillIndex, name: str) -> SkillDescriptor | None:
    """The skill a name typed by hand means: the exact name, else a different case.

    Typed by hand, so the case is a guess: `/skill elia` for a skill named `Elia` is
    not worth a round trip. Only when the match is unambiguous.
    """
    descriptor = index.by_name(name)
    if descriptor is None:
        folded = [s for s in index.skills if s.name.casefold() == name.casefold()]
        if len(folded) == 1:
            descriptor = folded[0]
    return descriptor


async def _show_one(console: Console, theme: Theme, runtime: SkillsRuntime, name: str) -> None:
    """Print one skill, or one of its ``##`` sections (``<name>#<section>``), in full.

    Raises `SkillNotFoundError` when the skill or the section does not exist, or the
    file cannot be read. Its message is written for the user, in words that fit both
    `/skill` and `domyn skill show`; the caller prints it.
    """
    target, _, section = name.partition("#")
    ref = SkillRef(name=target.strip(), section=section.strip() or None)
    index = await runtime.index(SkillQuery())
    # The descriptor carries what the body does not — sections, requirements.
    descriptor = find_skill(index, ref.name)
    if descriptor is not None:
        ref = ref.model_copy(update={"name": descriptor.name})
    sym = theme.symbols

    try:
        content = await runtime.fetch(ref, SkillQuery())
    except SkillNotFoundError:
        if descriptor is not None and ref.section is not None:
            available = f" {sym.separator} ".join(descriptor.sections) or "none"
            message = f"No section '{ref.section}' in {ref.name}. Sections: {available}"
        else:
            known = [s.name for s in index.skills]
            close = difflib.get_close_matches(ref.name, known, n=3)
            message = f"Skill not found: {name}."
            if close:
                message += f" Did you mean {', '.join(close)}?"
        raise SkillNotFoundError(message) from None
    except OSError as exc:
        raise SkillNotFoundError(f"Cannot read skill '{name}': {type(exc).__name__}") from exc

    width = min(console.width, _MAX_BODY_WIDTH)
    meta = theme.style("meta")
    warning = theme.style("warning")

    header = Text(" " * _INDENT)
    header.append(content.name, style=theme.style("accent_bold"))
    if content.section:
        header.append(f" #{content.section}", style=theme.style("accent"))
    version = (descriptor.version if descriptor else None) or content.version
    tags: list[tuple[str, str]] = []
    if version:
        tags.append((version if version.startswith("v") else f"v{version}", meta))
    if descriptor is not None and descriptor.enforcement == "required":
        tags.append(("required", warning))
    for i, (tag, style) in enumerate(tags):
        header.append("  " if i == 0 else f" {sym.separator} ", style=meta)
        header.append(tag, style=style)

    console.print()
    console.print(header, highlight=False)

    if descriptor is not None:
        facts = Table.grid(padding=(0, 2))
        facts.add_column(style=meta)
        facts.add_column(overflow="fold")
        if descriptor.path is not None:
            facts.add_row("file", Text(_display_path(descriptor.path)))
        if descriptor.sections:
            listed = Text()
            for i, title in enumerate(descriptor.sections):
                if i:
                    listed.append(f" {sym.separator} ", style=meta)
                current = content.section is not None and title == content.section
                listed.append(title, style=theme.style("accent") if current else "")
            facts.add_row("sections", listed)
        requires = _requirements(descriptor.requires)
        if requires:
            facts.add_row("requires", Text(f" {sym.separator} ".join(requires)))
        if not descriptor.available:
            reason = descriptor.unavailable_reason or "requirements not met"
            facts.add_row("status", Text(f"unavailable: {reason}", style=warning))
        if facts.row_count:
            # Full terminal width, not the prose cap: a path folded mid-way is one
            # that can no longer be copied.
            console.print(Padding(facts, (0, 0, 0, _INDENT)))

    console.print(
        Text(" " * _INDENT + sym.rule * (width - _INDENT * 2), style=theme.style("rule")),
        highlight=False,
    )
    console.print()
    with console.use_theme(_markdown_theme(theme)):
        console.print(
            Padding(
                _SkillMarkdown(
                    _checkboxes(content.body.strip(), unicode=theme.capabilities.unicode),
                    code_theme="ansi_dark",
                    hyperlinks=True,
                ),
                (0, 0, 0, _INDENT),
            ),
            width=width,
        )
    console.print()


_TASK_ITEM = re.compile(r"^(\s*[-*+]\s+)\[([ xX])\]\s", re.MULTILINE)


def _checkboxes(body: str, *, unicode: bool) -> str:
    """Task-list items as boxes: rich's markdown has no task lists, and prints ``• [ ]``."""
    if not unicode:
        return body

    def box(match: re.Match[str]) -> str:
        return f"{match.group(1)}{'☐' if match.group(2) == ' ' else '☑'} "

    return _TASK_ITEM.sub(box, body)


def _requirements(requires: SkillRequirements) -> list[str]:
    fields = (("bins", requires.bins), ("env", requires.env), ("os", requires.os))
    parts = [f"{label} {', '.join(values)}" for label, values in fields if values]
    if requires.tools:
        parts.append(f"tools {', '.join(requires.tools)}")
    if requires.external_files:
        parts.append(f"files {len(requires.external_files)}")
    return parts


class _LeftHeading(Heading):
    """Every heading level left-aligned: rich centres ``#``, which reads as a banner."""

    LEVEL_ALIGN: ClassVar[dict[str, JustifyMethod]] = {}


class _SkillMarkdown(Markdown):
    elements: ClassVar[dict[str, type[MarkdownElement]]] = {
        **Markdown.elements,
        "heading_open": _LeftHeading,
    }


def _markdown_theme(theme: Theme) -> RichTheme:
    """Rich's markdown defaults, redone in this CLI's palette.

    The defaults put inline code in ``cyan on black`` and headings in magenta,
    which is a black box on a light terminal and a colour this CLI uses for
    nothing else. Headings take the brand accent, code keeps no background.

    Weight and underline are not colour, so they stay when colour is off: a
    heading that loses its colour must still read as a heading.
    """

    def s(name: str, fallback: str = "none") -> str:
        return theme.style(name) or fallback

    return RichTheme(
        {
            "markdown.h1": f"{s('accent_bold', 'bold')} underline",
            "markdown.h2": s("accent_bold", "bold"),
            "markdown.h3": "bold",
            "markdown.h4": "italic",
            "markdown.code": "bold",
            "markdown.code_block": "none",
            "markdown.block_quote": s("thought", "italic"),
            "markdown.list": "none",
            "markdown.item.number": "none",
            "markdown.link": s("accent"),
            "markdown.link_url": s("meta"),
            "markdown.table.border": s("rule"),
            "markdown.table.header": "bold",
        }
    )
