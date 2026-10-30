"""Parsing of ``SKILL.md`` files.

Format is agentskills.io-compatible: a YAML frontmatter block delimited by ``---``
followed by a Markdown body. Staying on that format means skills written here are
readable by Claude Code, Cursor and Hermes, and vice versa — the portability is
the main reason not to invent a private schema.

Sections are derived from Markdown ``##`` headings, which gives L2 addressing for
free without asking authors to declare anything extra.
"""

import re
from pathlib import Path
from typing import Any

import yaml

from domyn.agents.logger import get_logger
from domyn.agents.skills.models import (
    Enforcement,
    SkillContent,
    SkillDescriptor,
    SkillRequirements,
)

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)
_HEADING = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


class SkillParseError(ValueError):
    """Raised when a skill file cannot be interpreted."""


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split raw file text into ``(metadata, body)``.

    A file without frontmatter is valid: it yields empty metadata, and the caller
    decides whether the missing ``description`` is fatal. That keeps hand-written
    notes usable while still steering authors towards the portable format.
    """
    match = _FRONTMATTER.match(text)
    if match is None:
        return {}, text.strip()

    raw_meta, body = match.group(1), match.group(2)
    try:
        loaded = yaml.safe_load(raw_meta)
    except yaml.YAMLError as exc:
        raise SkillParseError(f"invalid YAML frontmatter: {type(exc).__name__}") from exc

    if loaded is None:
        return {}, body.strip()
    if not isinstance(loaded, dict):
        raise SkillParseError("frontmatter must be a YAML mapping")
    return loaded, body.strip()


def section_names(body: str) -> list[str]:
    """Markdown ``##`` headings, in document order, deduplicated."""
    seen: dict[str, None] = {}
    for heading in _HEADING.findall(body):
        seen.setdefault(heading.strip(), None)
    return list(seen)


def extract_section(body: str, section: str) -> str | None:
    """Return the text under a ``##`` heading, excluding the heading itself."""
    wanted = section.strip().lower()
    matches = list(_HEADING.finditer(body))
    for position, match in enumerate(matches):
        if match.group(1).strip().lower() != wanted:
            continue
        start = match.end()
        end = matches[position + 1].start() if position + 1 < len(matches) else len(body)
        return body[start:end].strip()
    return None


def _requirements(meta: dict[str, Any], name: str) -> SkillRequirements:
    """Read requirements from the agentskills.io metadata namespace.

    Accepts both a nested ``metadata.requires`` mapping and flat top-level keys,
    because real-world skill files in the wild use both.
    """
    nested = meta.get("metadata")
    block: dict[str, Any] = {}
    if isinstance(nested, dict):
        requires = nested.get("requires")
        if isinstance(requires, dict):
            block = dict(requires)
        if "os" in nested and "os" not in block:
            block["os"] = nested["os"]

    for key in ("bins", "env", "os", "tools", "external_files"):
        if key in meta and key not in block:
            block[key] = meta[key]

    def as_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value]
        return []

    return SkillRequirements(
        bins=as_list(block.get("bins")),
        env=as_list(block.get("env")),
        os=as_list(block.get("os")),
        tools=as_list(block.get("tools")),
        external_files=_paths(block.get("external_files"), name),
    )


def _paths(value: Any, name: str) -> list[str]:
    """The ``external_files`` requirement: file paths relative to the skill file, as written.

    Stricter than ``as_list``: a single string is one entry, and anything else, or an
    item that is not a non-empty string, is left out with a debug line. An empty entry
    kept as ``""`` would resolve to the skill's own folder, which always exists, and
    pass the check without a word. Like the other keys, a malformed value must not
    stop a skill from loading.
    """
    if value is None:
        return []
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list):
        get_logger().debug(f"skill '{name}': 'external_files' is not a list of paths; ignored")
        return []
    kept: list[str] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            kept.append(item.strip())
        else:
            get_logger().debug(f"skill '{name}': an 'external_files' entry is not a path; ignored")
    return kept


def parse_skill_text(
    text: str,
    *,
    fallback_name: str,
    source: str = "unknown",
    path: Path | None = None,
) -> tuple[SkillDescriptor, SkillContent]:
    """Parse skill text into its L0 descriptor and L1 content."""
    meta, body = split_frontmatter(text)

    name = str(meta.get("name") or fallback_name).strip()
    description = str(meta.get("description") or "").strip()
    if not description:
        # Fall back to the first non-heading line so the skill still appears in
        # the index instead of vanishing. Logged, because an author who omits a
        # description gets worse selection quality and should know.
        for line in body.splitlines():
            candidate = line.strip().lstrip("#").strip()
            if candidate:
                description = candidate[:200]
                break
        get_logger().debug(
            f"skill '{name}' has no description in frontmatter; derived one from the body"
        )

    enforcement_raw = str(meta.get("enforcement") or "advisory").strip().lower()
    enforcement: Enforcement = "required" if enforcement_raw == "required" else "advisory"

    version = meta.get("version")

    descriptor = SkillDescriptor(
        name=name,
        description=description or f"Skill {name}",
        version=str(version) if version is not None else None,
        source=source,
        sections=section_names(body),
        requires=_requirements(meta, name),
        enforcement=enforcement,
        path=path,
    )
    content = SkillContent(name=name, body=body, version=descriptor.version, path=path)
    return descriptor, content


def parse_skill_file(path: Path, *, source: str = "local") -> tuple[SkillDescriptor, SkillContent]:
    """Parse a skill from disk.

    The fallback name is the containing directory for ``<skill>/SKILL.md`` layouts
    and the file stem for flat ``<skill>.md`` ones, so both conventions work.
    """
    text = path.read_text(encoding="utf-8")
    fallback = path.parent.name if path.name.lower() == "skill.md" else path.stem
    return parse_skill_text(text, fallback_name=fallback, source=source, path=path)
