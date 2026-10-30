"""The skills that ship with the package, and how they are installed.

``domyn setup`` lets the user choose among them and writes the choice to a skills
directory (``~/.domyn/builtin-skills`` for the CLI), which the config then names skill by
skill. Nothing here reads the config or knows where that directory is: the caller
passes it.

What gets copied for a skill is what its ``SKILL.md`` says: the file itself, and every
path in its ``external_files:`` frontmatter key (relative to the skill file; see
``skills/external_files.py``). Nothing else in the skill's folder is copied, so a script a
skill needs is listed in the skill, next to the text that tells the model to run it,
and not in a table in this module.

The installed tree **mirrors the package's own layout** (``builtin/<category>/<skill>``,
with shared files at the category level and ``resources/`` at the top). Several
skills ship scripts that find shared modules by their own place in that tree
(``Path(__file__).parents[2]`` and ``[3]``) and whose SKILL.md says where those
modules live, so a copy that keeps the relative layout needs no edit to any of it, and
an ``external_files`` path that is right from the skill file in the package is right from the
installed one.

An install **replaces** the directory instead of adding to it: a skill the user
dropped disappears, and one that changed in the package arrives in its new version.
That is only safe because the directory is owned by setup, so :func:`install_skills`
refuses a symlink (it may point at something that is not ours to replace) and
:func:`unrecognized_entries` tells the caller what a replace would delete that did
not come from here.
"""

import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from domyn.agents.skills.external_files import expand_files, resolve_external_files
from domyn.agents.skills.parser import parse_skill_file

BUILTIN_SKILLS_ROOT = Path(__file__).resolve().parent / "builtin"
"""Skills shipped with the package, grouped one directory per category."""


class SkillInstallError(OSError):
    """The skills directory cannot be replaced, or a skill's file list cannot be followed."""


@dataclass(frozen=True)
class BuiltinSkill:
    name: str
    """The skill's folder name, which is what a config's path ends in."""
    category: str
    description: str
    external_files: tuple[str, ...]
    """The skill's ``external_files:`` entries as written: paths relative to its ``SKILL.md``."""

    @property
    def relative_path(self) -> Path:
        return Path(self.category) / self.name

    @property
    def source(self) -> Path:
        return BUILTIN_SKILLS_ROOT / self.relative_path

    @property
    def skill_file(self) -> Path:
        return self.source / "SKILL.md"


def list_builtin_skills() -> list[BuiltinSkill]:
    """Every skill under ``builtin/<category>/<skill>/SKILL.md``, in a stable order."""
    skills: list[BuiltinSkill] = []
    if not BUILTIN_SKILLS_ROOT.is_dir():
        return skills
    for category in sorted(p for p in BUILTIN_SKILLS_ROOT.iterdir() if p.is_dir()):
        if category.name == "__pycache__":
            continue
        for folder in sorted(p for p in category.iterdir() if p.is_dir()):
            skill_file = folder / "SKILL.md"
            if not skill_file.is_file():
                continue
            descriptor, _ = parse_skill_file(skill_file, source="builtin")
            skills.append(
                BuiltinSkill(
                    name=folder.name,
                    category=category.name,
                    description=descriptor.description,
                    external_files=tuple(descriptor.requires.external_files),
                )
            )
    return skills


def skill_groups() -> dict[str, list[str]]:
    """The builtin skill names by category, in catalog order.

    The categories are the folders under ``builtin/``, so a group is whatever the
    package ships and there is no second list to keep in step with it.
    """
    groups: dict[str, list[str]] = {}
    for skill in list_builtin_skills():
        groups.setdefault(skill.category, []).append(skill.name)
    return groups


def install_plan(skill: BuiltinSkill) -> list[tuple[Path, Path]]:
    """What installing a skill copies: ``(source file, destination under the skills root)``.

    The ``SKILL.md`` itself, then every file its ``external_files`` entries stand for (a
    folder entry is everything under it). Destinations are the paths relative to
    ``BUILTIN_SKILLS_ROOT``, which is how the installed tree mirrors the package.

    Raises :class:`SkillInstallError` for an entry that is absolute, does not exist, or
    leads outside the built-in skills: a typo, or a path that would copy a file from
    somewhere else, is an error, not a quiet omission.

    An entry is judged by where it is written to be, not where a link leads: the tree
    is the package's own, and in a source checkout ``domyn-docs/pages`` is a link to
    ``docs/docs``. Copying through it gives the installed skill its pages as files. A
    link *inside* a listed folder is still left out (`expand_files`).
    """
    root = BUILTIN_SKILLS_ROOT
    sources: dict[Path, None] = {skill.skill_file: None}
    for item in resolve_external_files(skill.skill_file, list(skill.external_files)):
        if item.problem or item.path is None:
            raise SkillInstallError(
                f"skill {skill.name!r}: external_files entry {item.entry!r} is unusable ({item.problem})"
            )
        listed = Path(os.path.normpath(skill.source / item.entry.replace("\\", "/")))
        if listed != root and root not in listed.parents:
            raise SkillInstallError(
                f"skill {skill.name!r}: external_files entry {item.entry!r} leads outside the "
                "built-in skills"
            )
        sources.update(dict.fromkeys(expand_files(listed)))
    return [(source, source.relative_to(root)) for source in sources]


def install_skills(names: Iterable[str], dest: Path) -> list[Path]:
    """Replace ``dest`` with the named builtin skills and the files they list.

    Returns the absolute path of each installed skill's folder, in catalog order.
    ``dest`` ends up holding exactly these skills, whatever it held before.

    Every skill's file list is worked out first, so a broken list fails before anything
    is touched. The tree is then built next to ``dest`` and swapped in, so a failure
    while copying leaves the old directory as it was rather than empty. Unknown names
    raise ``ValueError``; a symlinked ``dest`` raises :class:`SkillInstallError`.
    """
    wanted = set(names)
    catalog = {skill.name: skill for skill in list_builtin_skills()}
    unknown = sorted(wanted - catalog.keys())
    if unknown:
        raise ValueError(f"unknown builtin skill: {', '.join(unknown)}")
    chosen = [skill for skill in catalog.values() if skill.name in wanted]
    # A file two skills list (a shared module) is one destination, copied once.
    plan = dict.fromkeys(pair for skill in chosen for pair in install_plan(skill))

    dest = dest.expanduser().absolute()
    if dest.is_symlink():
        raise SkillInstallError(
            f"{dest} is a symbolic link; it is not replaced, remove the link to let setup manage it"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.parent / f".{dest.name}.new"
    retired = dest.parent / f".{dest.name}.old"
    # Left behind only by a run that was killed half way.
    for leftover in (staging, retired):
        shutil.rmtree(leftover, ignore_errors=True)

    try:
        staging.mkdir()
        for source, relative in plan:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if dest.exists():
        dest.rename(retired)
    try:
        staging.rename(dest)
    except OSError:
        if retired.exists():
            retired.rename(dest)
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(retired, ignore_errors=True)

    return [dest / skill.relative_path for skill in chosen]


def add_skills(names: Iterable[str], dest: Path) -> list[Path]:
    """Add the named builtin skills to ``dest``, leaving what it holds as it is.

    Returns the absolute path of each added skill's folder, in catalog order. The
    counterpart of :func:`install_skills` for a directory whose contents are not this
    package's to replace, such as the built-ins a project brought from an export:
    a file already there, a shared module or a skill's own, is kept rather than
    overwritten with the package's version.

    Every skill's file list is worked out first, so a broken list fails before anything
    is touched, and a failure while copying removes the files this call wrote. Unknown
    names raise ``ValueError``; a symlinked ``dest`` raises :class:`SkillInstallError`.
    """
    wanted = set(names)
    catalog = {skill.name: skill for skill in list_builtin_skills()}
    unknown = sorted(wanted - catalog.keys())
    if unknown:
        raise ValueError(f"unknown builtin skill: {', '.join(unknown)}")
    chosen = [skill for skill in catalog.values() if skill.name in wanted]
    plan = dict.fromkeys(pair for skill in chosen for pair in install_plan(skill))

    dest = dest.expanduser().absolute()
    if dest.is_symlink():
        raise SkillInstallError(
            f"{dest} is a symbolic link; skills are not added to it, remove the link to let "
            "setup manage it"
        )

    written: list[Path] = []
    try:
        for source, relative in plan:
            target = dest / relative
            if target.exists() or target.is_symlink():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            written.append(target)
    except BaseException:
        for target in written:
            target.unlink(missing_ok=True)
        raise

    return [dest / skill.relative_path for skill in chosen]


def unrecognized_entries(dest: Path) -> list[Path]:
    """What a replace of ``dest`` would delete that did not come from this catalog.

    Paths relative to ``dest``, outermost first: an unknown folder is reported once,
    not file by file. A file is recognised by *where it is*, never by its content:
    an install from an older version differs from the package without anyone having
    edited it, so comparing bytes would raise the alarm after every upgrade.
    What is known is what `install_plan` would write, so the two cannot disagree.
    ``__pycache__`` is ignored, since running a skill's scripts creates it.
    """
    dest = dest.expanduser()
    if not dest.is_dir() or dest.is_symlink():
        return []

    known: set[Path] = set()
    for skill in list_builtin_skills():
        try:
            known.update(relative for _, relative in install_plan(skill))
        except SkillInstallError:
            # A built-in whose own list is broken cannot be installed at all, and the
            # install says so; what it would have written is not a thing to know about.
            continue
    # The folders above a known entry (a category, `data/`) are known too.
    known.update(parent for path in list(known) for parent in path.parents)
    known.discard(Path("."))

    unrecognized: list[Path] = []

    def walk(directory: Path) -> None:
        for child in sorted(directory.iterdir()):
            if child.name == "__pycache__" or child.suffix == ".pyc":
                continue
            relative = child.relative_to(dest)
            if relative not in known:
                unrecognized.append(relative)
            elif child.is_dir() and not child.is_symlink():
                walk(child)

    walk(dest)
    return unrecognized
