"""Finding the skills a project keeps in its own folder, as ``domyn setup`` shows them.

The walk itself is `skills.layout.walk_skill_tree`, the one loading uses too: when
the config lists ``.domyn/skills`` and ``.agents/skills``, ``domyn run`` reads them
every turn from the folder it starts in. What this module adds is what a picker or a report needs on top:
the parsed name and description, a label no other row has, the group as one string,
and the ``external_files`` that are missing. Nothing is copied, and no file is read
except to parse a skill.
"""

from dataclasses import dataclass, field
from pathlib import Path

from domyn.agents.skills.external_files import missing_external_files
from domyn.agents.skills.layout import walk_skill_tree
from domyn.agents.skills.parser import parse_skill_file


@dataclass(frozen=True)
class LocalSkill:
    name: str
    """The skill's own name (its frontmatter ``name``, else its folder or file name)."""
    label: str
    """What the picker calls it: ``name``, made unique when two skills or a group share it."""
    description: str
    path: Path
    """The skill's folder, or its ``.md`` file when it is a loose one, as ``root / ...``.

    Relative when ``root`` was, which is what ends up in the config."""
    group: str | None = None
    """The folders between the root and the skill, ``/``-joined; ``None`` when it sits in the root."""
    missing: tuple[str, ...] = ()
    """Entries of its ``external_files:`` list that point at nothing, each with why.

    Only reported: a local skill stays where it is, so nothing is copied, and a skill
    half way through being written is still offered."""


@dataclass(frozen=True)
class LocalDiscovery:
    skills: list[LocalSkill] = field(default_factory=list)
    skipped: list[tuple[Path, str]] = field(default_factory=list)
    """Skill files that could not be read, and why: left out, never hiding the rest."""

    def groups(self) -> dict[str, list[str]]:
        """Group name to the labels of its skills, in discovery order."""
        found: dict[str, list[str]] = {}
        for skill in self.skills:
            if skill.group is not None:
                found.setdefault(skill.group, []).append(skill.label)
        return found


def discover_local_skills(root: Path) -> LocalDiscovery:
    """The skills under ``root``, grouped by the folders that hold them.

    Hidden folders, ``__pycache__`` and the usual dependency folders are skipped, and a
    symlinked folder is never entered or counted, so a loop, or a link out of the
    project, cannot be walked. A skill that cannot be parsed goes to ``skipped``.
    A missing or empty ``root`` gives nothing.
    """
    skills: list[LocalSkill] = []
    skipped: list[tuple[Path, str]] = []
    for found in walk_skill_tree(root):
        try:
            descriptor, _ = parse_skill_file(found.skill_file, source="local")
        except (OSError, ValueError) as exc:
            skipped.append((found.path, type(exc).__name__))
            continue
        skills.append(
            LocalSkill(
                name=descriptor.name,
                label=descriptor.name,
                description=descriptor.description,
                path=found.path,
                group="/".join(found.group) or None,
                missing=tuple(
                    missing_external_files(found.skill_file, descriptor.requires.external_files)
                ),
            )
        )

    return LocalDiscovery(skills=_with_unique_labels(skills), skipped=skipped)


def _with_unique_labels(skills: list[LocalSkill]) -> list[LocalSkill]:
    """Give every skill a label no other skill or group has.

    The picker keys its rows by label, so two skills called ``summarise`` in different
    folders, or a skill named like a group, would be one row. The later ones get their
    path appended; the first keeps the plain name.
    """
    taken = {skill.group for skill in skills if skill.group is not None}
    labelled: list[LocalSkill] = []
    for skill in skills:
        label = skill.name
        if label in taken:
            label = f"{skill.name} ({skill.path.as_posix()})"
        taken.add(label)
        labelled.append(
            LocalSkill(
                name=skill.name,
                label=label,
                description=skill.description,
                path=skill.path,
                group=skill.group,
                missing=skill.missing,
            )
        )
    return labelled
