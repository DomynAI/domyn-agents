"""Where the skills inside a folder are: one walk, for ``domyn setup`` and for loading.

Shared so the two cannot disagree about a layout. ``./.domyn/skills`` and
``./.agents/skills`` are read at run time from whatever folder ``domyn run`` starts in,
and setup offers what it finds in a folder of the user's own; a group folder that one
of them entered and the other did not would be a skill the user was shown and never
got, or the other way round.

The layout: a directory holding a ``SKILL.md`` is one skill and is not entered; a
loose ``<name>.md`` directly in the root is a skill too; any other directory is a
**group** of what it holds. Hidden folders, ``__pycache__`` and the usual dependency
folders are skipped, and a symlink is never followed, the root included
(`listed_path_is_linked`): a folder of skills is read every turn from a project the
user may just have cloned, and a link in it (``.agents -> ~``) must not decide what is
read, or what the file tools can reach.
"""

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path

# Names a skill file may have inside its own folder, in priority order.
SKILL_FILENAMES = ("SKILL.md", "skill.md")

# Directories never worth entering, and often enormous. Matches the search tools'
# list (`tools/builtin/workspace.py`) without importing it: that package's init pulls
# in most of the framework, and a skills module has no business depending on it.
SKIPPED_DIRS = frozenset(
    {
        "__pycache__",
        "node_modules",
        "venv",
        "dist",
        "build",
        "target",
    }
)


@dataclass(frozen=True)
class FoundSkill:
    path: Path
    """The skill's folder, or its ``.md`` file when it is a loose one, as ``root / ...``."""
    skill_file: Path
    """The file to parse: ``path / "SKILL.md"``, or ``path`` itself for a loose one."""
    group: tuple[str, ...] = ()
    """The folders between the root and the skill; empty when it sits in the root."""


def listed_path_is_linked(path: Path) -> bool:
    """Whether a listed skills path goes through a symbolic link that is not followed.

    The path itself, always. For a relative path, also every folder on the way to it
    from the working directory: that part comes from the project domyn was started in,
    which may be a clone, and ``.agents -> ~`` would make ``.agents/skills`` read
    ``~/skills``. The folders above an absolute path are left alone, so a symlinked
    parent the user did not choose (``/tmp`` on macOS) is fine.
    """
    if path.is_symlink():
        return True
    if path.is_absolute():
        return False
    prefix = Path()
    for part in path.parts[:-1]:
        prefix /= part
        if prefix.is_symlink():
            return True
    return False


def own_skill_file(directory: Path) -> Path | None:
    """The ``SKILL.md`` that makes ``directory`` a skill of its own, if it has one."""
    return next((directory / n for n in SKILL_FILENAMES if (directory / n).is_file()), None)


def walk_skill_tree(root: Path) -> list[FoundSkill]:
    """The skills under ``root``, depth first, sorted by name within each folder.

    A missing root, a file, or a linked one (`listed_path_is_linked`) gives nothing.
    A folder that cannot be listed ends the walk with what was found so far.
    """
    if listed_path_is_linked(root) or not root.is_dir():
        return []

    found: list[FoundSkill] = []

    def walk(directory: Path, parts: tuple[str, ...]) -> None:
        own = own_skill_file(directory)
        if own is not None:
            found.append(FoundSkill(path=directory, skill_file=own, group=parts[:-1]))
            return
        with os.scandir(directory) as entries:
            children = sorted(entries, key=lambda e: e.name)
        for entry in children:
            if entry.is_symlink() or entry.name.startswith(".") or entry.name in SKIPPED_DIRS:
                continue
            if entry.is_dir(follow_symlinks=False):
                walk(directory / entry.name, (*parts, entry.name))
            elif (
                not parts
                and entry.is_file(follow_symlinks=False)
                and entry.name.lower().endswith(".md")
                and entry.name.lower() not in (n.lower() for n in SKILL_FILENAMES)
            ):
                path = directory / entry.name
                found.append(FoundSkill(path=path, skill_file=path))

    with contextlib.suppress(OSError):
        walk(root, ())
    return found
