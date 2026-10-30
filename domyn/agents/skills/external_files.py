"""The files a skill says it needs, and whether they are there.

A skill lists them in its ``external_files:`` frontmatter key as paths **relative to its
own ``SKILL.md``**. External to that file, not to the skill's folder: ``scripts/`` is an
entry like ``../shared.py``. What each entry means is decided here and nowhere else, so the
installer (which copies them for a built-in skill), the setup picker and the startup
check (which only look) cannot read the same entry two ways.

An entry is a file, or a folder meaning everything under it; ``.`` is the skill's own
folder. It may go up with ``..``: the installed layout mirrors the package's, so a path
that is right from the skill file in one is right in the other.
"""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

# Never carried along when a folder is expanded: running a skill's scripts makes them.
_SKIPPED = ("__pycache__",)


@dataclass(frozen=True)
class ExternalFile:
    entry: str
    """The entry as written in the skill file."""
    path: Path | None
    """Where it resolves to, or ``None`` when the entry is unusable."""
    problem: str | None = None
    """Why it is unusable, or ``None``: ``"absolute path"``, ``"does not exist"``."""


def resolve_external_files(skill_file: Path, entries: list[str]) -> list[ExternalFile]:
    """Resolve each entry against the folder holding ``skill_file``.

    Nothing is copied or created. An absolute path is refused rather than followed:
    the key describes files that travel with the skill, and a path fixed to one
    machine cannot.
    """
    base = skill_file.expanduser().resolve().parent
    resolved: list[ExternalFile] = []
    for entry in entries:
        normal = entry.replace("\\", "/")
        if PurePosixPath(normal).is_absolute() or PureWindowsPath(entry).is_absolute():
            resolved.append(ExternalFile(entry, None, "absolute path"))
            continue
        target = (base / normal).resolve()
        resolved.append(ExternalFile(entry, target, None if target.exists() else "does not exist"))
    return resolved


def expand_files(path: Path) -> list[Path]:
    """The files a resolved entry stands for: itself, or everything under a folder.

    Bytecode caches are left out, and so are symlinks inside a folder: a link could
    point anywhere, and a skill's files are the ones it holds.
    """
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    return sorted(
        child
        for child in path.rglob("*")
        if child.is_file()
        and not child.is_symlink()
        # Judged on the part below the entry only: the folders above it are the user's.
        and not any(
            part in _SKIPPED or part.endswith(".pyc") for part in child.relative_to(path).parts
        )
    )


def missing_external_files(skill_file: Path, entries: list[str]) -> list[str]:
    """What is wrong with a skill's list, as short phrases naming the entry.

    Empty when every entry resolves. This is what is shown for a skill that stays
    where it is: it is not copied, so the only question is whether its files exist.
    """
    return [
        f"{item.entry} ({item.problem})"
        for item in resolve_external_files(skill_file, entries)
        if item.problem
    ]
