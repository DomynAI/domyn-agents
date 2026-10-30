"""``domyn export``: a copy of the CLI configuration someone else can run.

The result is a folder laid out like ``~/.domyn`` — ``config.yaml``, ``builtin-skills/``,
``skills/``, ``instructions/`` — that holds everything the configuration reads from disk, with every path to a copied file rewritten as ``${DOMYN_PROJECT_DIR}/...``,
and a ``.info`` recording the version of domyn that wrote it (which an
import compares with its own), when, and on what.
That is the layout of a **project**: used with ``--project <its path>``, or moved to
``~/.domyn/projects/<name>`` and used with ``--project <name>``, as if
``domyn setup --project`` had written it — the CLI sets ``DOMYN_PROJECT_DIR`` to the
selected folder.

What a skills path stands for is decided the way the loader decides it
(`skills.layout.walk_skill_tree`, symlinks never followed): a group folder is walked
at any depth, a skill folder or a loose ``.md`` is that skill.

**What is copied is opt-in, never a folder's contents.** A skill travels as its
``SKILL.md`` (or its loose ``.md``) plus exactly what its ``external_files`` declares
— the list `domyn setup` installs the built-in skills from, so their copies are
complete by construction. Anything else in a skill's folder stays behind and is named
in the report, so a script a hand-written skill forgot to declare is visible rather
than silently missing; the fix is to declare it. A folder entry (``scripts/``, ``.``)
is the author's opt-in to what it holds, hidden files under it excepted, and no
entry ever brings a file with a hidden part in its path (``../../.ssh/id_rsa``): a
skill is often someone else's, and the export goes to someone else again. The
instruction files are ``SOUL.md`` and the ``AGENTS.md`` a folder contributes, nothing
beside them. Instructions are the files the config declares, and only those: nothing
is read without being declared, in a project as anywhere else.

Those declared files can reach above their own folder (``../shared.py``,
``plotly_style.py``). So a listed path is copied together with its **anchor** — the
nearest folder holding it and everything its skills need — and the layout below the
anchor is kept, which is what keeps those relative references true in the copy.

``~/.domyn/builtin-skills`` (what ``domyn setup`` installs) is ``builtin-skills/``,
so the export, used as a project, has its built-ins where setup installs them. Every
other anchor is ``skills/<folder name>/``, with a suffix when the name is taken. The
config's paths are rewritten one by one, so their order, which settles a name clash,
is kept.

The ``.domyn/skills`` and ``.agents/skills`` autodiscovery entries are
**snapshotted** and **merged**: the skills in those folders of the folder the export
runs in are copied straight into ``skills/``, categories kept
(``.domyn/skills/<category>/<name>`` → ``skills/<category>/<name>``), and listed one
by one where the first entry was, so the export carries what a run here would load. A
name both folders hold is settled before copying, as the loader settles it: the
earlier folder's skill (``.domyn``'s, in the order setup writes) is copied, the other
is named in the report. A place already taken in ``skills/`` gets a suffix, unless
what holds it is the same skill, reached through another listed path (named on its
own, or inside a listed folder), or an identical copy of it from elsewhere (every
file the same, byte for byte, and nothing more): that one is copied once, and listed
by that copy. A folder missing there is left out quietly.

The file a ``tools`` or ``hooks`` entry runs is copied into ``code/`` (the file
alone), and the report warns about what it imports that the machine running the
export may lack (`import_warnings`: a module beside it, a package domyn does not
install). After all that, the exported config names nothing outside the exported
folder: a path left that is not ``${DOMYN_PROJECT_DIR}/…`` — a workspace, a browser
executable, an OAuth token store, a file that could not be copied — refuses the
export (`paths_outside_project`), and so does a skill's ``external_files`` entry that
leads out of the folder from where the skill was copied to, because a file was not
copied (`external_files_outside`). `domyn setup --import` checks both before
copying anything, since an export can be edited after it is made.

Secrets are not exported. A literal API key, header or client secret becomes a
``${VAR}`` reference and the report names the variables to set. ``credentials``,
approvals and conversation history are never copied: nothing names them, and a file
called like one is refused even when an ``external_files`` entry does (`is_private`),
since a project keeps all three in its own folder, next to the files that are exported.
"""

import ast
import contextlib
import copy
import filecmp
import importlib.metadata
import importlib.util
import os
import platform
import posixpath
import re
import shutil
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from rich.console import Console

import domyn.agents
from domyn.agents.instructions import LocalDirInstructionsSource
from domyn.agents.instructions.discovery import (
    AUTODISCOVERY_DIRS,
    STATE_DIRS,
    find_instruction_files,
)
from domyn.agents.interfaces.cli import config as cli_config
from domyn.agents.interfaces.cli.config import (
    APPROVALS_FILENAME,
    BUILTIN_SKILLS_DIRNAME,
    CONFIG_FILENAME,
    CREDENTIALS_FILENAME,
    PROJECT_DIR_REF,
    PROJECT_DIR_VAR,
    PROVIDER_HINTS,
    ProjectError,
    is_skill_autodiscovery,
    project_dir,
    projects_dir,
    skills_install_dir,
)
from domyn.agents.interfaces.cli.conversations import STORE_FILENAME
from domyn.agents.interfaces.cli.theme import Theme
from domyn.agents.loader import interpolate_env_vars
from domyn.agents.skills import LocalDirSkillSource, parse_skill_file
from domyn.agents.skills.external_files import expand_files, resolve_external_files
from domyn.agents.skills.layout import own_skill_file, walk_skill_tree

SKILLS_DIR = "skills"
INSTRUCTIONS_DIR = "instructions"

CODE_DIR = "code"
"""Where the Python files of ``tools`` and ``hooks`` entries are copied."""

INFO_FILENAME = ".info"
"""What an export says about itself, beside ``config.yaml`` (`export_info`): which
version of domyn made it, when, and on what. Not a configuration: nothing
loads it, and an import reads it and does not copy it."""

EXPORT_FORMAT_VERSION = 1
"""The version of what an export is: its folder layout (``builtin-skills/``,
``skills/``, ``instructions/``, ``code/``), how its config names those files, and the
keys of `INFO_FILENAME`. Raised when any of them changes in a way an import has to
know about."""

INFO_VERSION_KEY = "domyn_agents_version"


def installed_version() -> str:
    """The version of domyn running, ``unknown`` when it is not installed."""
    return domyn.agents.__version__


def export_info() -> dict[str, Any]:
    """What `INFO_FILENAME` records, as ``key: value`` lines.

    What may explain why a project behaves differently elsewhere, and nothing that
    names the machine or its user: no path, host or user name, as the export is
    meant to be handed on. The platform SDK's version only when it is installed,
    since the ``domyn`` provider needs it.
    """
    info: dict[str, Any] = {
        "export_format_version": EXPORT_FORMAT_VERSION,
        INFO_VERSION_KEY: installed_version(),
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}".strip(),
    }
    with contextlib.suppress(importlib.metadata.PackageNotFoundError):
        info["domyn_platform_sdk_version"] = importlib.metadata.version("domyn-platform-sdk")
    return info


LOCAL_INSTRUCTION_DIRS: tuple[Path, ...] = AUTODISCOVERY_DIRS
"""The instruction counterpart of `LOCAL_SKILLS_DIRS`: ``./.domyn`` and ``./.agents``
of the folder ``domyn run`` starts in, listed relative under an agent's
``standing_instructions`` and walked by the loader on every load. Export **snapshots**
them: it walks them as the loader does (`find_instruction_files`, domyn's own state
folders skipped) and lists every file it uses one by one, as a copied folder would be
read flat. One with no file is left out quietly, as most folders have none."""

_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
# Header names whose value is a credential. Anything else (``Accept``, a tenant id)
# is configuration and travels as written.
_SECRET_HEADER = re.compile(r"auth|key|token|secret|cookie|password", re.IGNORECASE)


class ExportError(Exception):
    """The export cannot be made; nothing was written."""


@dataclass
class ExportReport:
    dest: Path
    skill_paths: list[str] = field(default_factory=list)
    """The skills paths of the exported config, as written in it."""
    instruction_files: list[str] = field(default_factory=list)
    """The instruction files copied, relative to the export."""
    files_copied: int = 0
    dropped: list[str] = field(default_factory=list)
    """Paths the config lists that are not in the export, each with the reason."""
    warnings: list[str] = field(default_factory=list)
    env_vars: list[str] = field(default_factory=list)
    """Every ``${VAR}`` the exported config needs set to load."""


@dataclass
class _InstructionSource:
    """A folder or file an agent's ``standing_instructions`` reads."""

    path: Path
    """Resolved: a folder, or one file."""
    files: list[Path]
    """What it contributes: its ``SOUL.md`` and ``AGENTS.md``, or the file itself."""


@dataclass
class _MergedSkill:
    """One skill of the merged autodiscovery folders."""

    root: Path
    """Its folder, or its file for a loose ``.md``."""
    folder: Path
    """The autodiscovery folder it comes from."""
    name: str
    files: list[Path]
    """Its ``SKILL.md`` and what its ``external_files`` declare."""
    place: PurePosixPath = field(default_factory=PurePosixPath)
    """Where it is copied, relative to ``skills/``: categories kept."""


@dataclass
class _MergedSkills:
    """The autodiscovery folders of one agent, merged into ``skills/``.

    A skill whose name an earlier folder already has is not in it: the loader would
    read the earlier one, so that is the one exported.
    """

    folders: tuple[Path, ...]
    skills: list[_MergedSkill]


@dataclass
class _SkillsEntry:
    """One listed skills path that exists, and what it needs copied."""

    agent: int
    listed: Path
    """Resolved: a folder of skills, a skill folder, or a loose ``.md``."""
    files: set[Path]
    anchor: Path = field(default=Path())


class _Exporter:
    def __init__(self, raw: dict[str, Any], cwd: Path, install_dir: Path) -> None:
        self.config = copy.deepcopy(raw)
        self.cwd = cwd
        self.install_dir = install_dir.expanduser().resolve()
        self.copies: dict[Path, Path] = {}
        """Destination, relative to the export → source file."""
        self.skill_files: set[Path] = set()
        """Every skill file read, resolved: whose ``external_files`` the copy must keep
        inside the export (`_refuse_paths_outside`)."""
        self.report = ExportReport(dest=Path())

    # -- paths ------------------------------------------------------------

    def _listed_path(self, written: Any, what: str) -> Path | None:
        """A path as the config writes it, made absolute the way the loader reads it."""
        try:
            path = Path(interpolate_env_vars(str(written), "the configuration")).expanduser()
        except ValueError:
            self.report.dropped.append(
                f"{written} ({what}: uses an environment variable that is not set)"
            )
            return None
        return path if path.is_absolute() else self.cwd / path

    def _copy(self, source: Path, dest: Path) -> None:
        if is_private(source):
            # Only an `external_files` entry walking up into a domyn home can name one:
            # refused all the same, as the export is for someone else.
            note = f"{source} (API keys, approvals and history are never exported)"
            if note not in self.report.dropped:
                self.report.dropped.append(note)
            return
        previous = self.copies.setdefault(dest, source)
        if previous != source:
            # Anchors are disjoint and instruction folders uniquely named, so two
            # sources for one destination is a bug here, not a property of the config.
            raise ExportError(f"two files would be exported as {dest}: {previous} and {source}")

    def _unique(self, parent: str, name: str, taken: set[str]) -> str:
        candidate, n = f"{parent}/{name or 'skills'}", 2
        while candidate in taken:
            candidate, n = f"{parent}/{name or 'skills'}-{n}", n + 1
        taken.add(candidate)
        return candidate

    # -- skills -----------------------------------------------------------

    def _autodiscovery_folder(self, written: Any) -> Path | None:
        """The folder an autodiscovery entry reads here, or ``None`` when there is
        nothing to export: missing (the normal state of most folders) or a link (named)."""
        listed = self._listed_path(written, "skills")
        if listed is None:
            return None
        if listed.is_symlink():
            self.report.dropped.append(f"{written} (a symbolic link, which export does not follow)")
            return None
        if not listed.is_dir():
            return None
        return listed.resolve()

    def _merged_skills(self, folders: tuple[Path, ...]) -> _MergedSkills | None:
        """The skills of the autodiscovery folders, in their order.

        A name already taken by an earlier folder is left out, as the loader would
        never read it, and so is a skill whose place (``category/name``) an earlier
        one already has.
        """
        names: dict[str, Path] = {}
        places: dict[Path, Path] = {}
        skills: list[_MergedSkill] = []
        for folder in folders:
            for skill_root, skill_file in _listed_skills(folder):
                name = _skill_name(skill_file, skill_root)
                place = skill_root.relative_to(folder)
                earlier = names.get(name) or places.get(place)
                if earlier is not None:
                    self.report.dropped.append(
                        f"{skill_root} (skill {name!r}: {earlier} has the same name and "
                        "takes precedence, so only that one is exported)"
                    )
                    continue
                names[name] = places[place] = skill_root
                declared = {skill_file, *self._external_files(skill_file)}
                if skill_root.is_dir():
                    self._name_undeclared(skill_root, declared)
                skills.append(_MergedSkill(skill_root, folder, name, sorted(declared)))
        return _MergedSkills(folders, skills) if skills else None

    def _occupied(self, dest: PurePosixPath) -> bool:
        """Whether something is already exported at ``dest`` or under it."""
        target = Path(dest)
        return any(k == target or target in k.parents for k in self.copies)

    def _copy_merged(self, group: _MergedSkills, walked: set[str], bases: dict[Path, str]) -> None:
        """Copy the merged skills into ``skills/``, each at ``category/name`` as in its
        folder, and set the place each one is listed by.

        A skill another listed path also reaches (named on its own, or inside a listed
        folder) is the same files: it keeps the place that path's copy gives it
        (`_place_in_anchor`) and is copied once. So does one whose place already holds
        an identical copy from elsewhere (`_identical_copy_at`): the same files, byte
        for byte, and nothing else in its folder.

        Anything else can already be there: a different skill another entry copied to
        the same place, or the copy of another listed folder of skills (``walked``),
        whose entry would then load the skill a second time. The first moves the skill to ``name-2`` in its
        category, the second its category to ``category-2``; either is named. The skill
        is still listed ahead of them, so it wins its name as it did.

        A file the skill shares from beside or above it (a category's ``shared.py``,
        ``../../shared.py``) keeps its position relative to the skill, anywhere in the
        export (`_shared_place_problem`). One whose place another file already has
        keeps the first. A refusal is named, and the skill is unavailable in the export,
        as the loader reports.
        """
        for skill in group.skills:
            if (reused := self._place_in_anchor(skill, bases)) is not None:
                skill.place = reused
                continue
            original = PurePosixPath(skill.root.relative_to(skill.folder).as_posix())
            if self._identical_copy_at(skill, original):
                skill.place = original
                continue
            place = original
            top = place.parts[0]
            n = 2
            while f"{SKILLS_DIR}/{top}" in walked and len(place.parts) > 1:
                top = f"{place.parts[0]}-{n}"
                n += 1
            place = PurePosixPath(top, *place.parts[1:])
            n = 2
            while self._occupied(PurePosixPath(SKILLS_DIR) / place) or (
                len(place.parts) == 1 and f"{SKILLS_DIR}/{place}" in walked
            ):
                stem, suffix = (
                    (original.stem, original.suffix)
                    if skill.root.is_file()
                    else (
                        original.name,
                        "",
                    )
                )
                place = place.with_name(f"{stem}-{n}{suffix}")
                n += 1
            if place != original:
                self.report.warnings.append(
                    f"Skill {skill.name!r} was copied to {SKILLS_DIR}/{place}, because "
                    f"{SKILLS_DIR}/{original} was already taken"
                )
            skill.place = place
            anchor = skill.root if skill.root.is_dir() else skill.root.parent
            dest_anchor = PurePosixPath(SKILLS_DIR) / (
                place if skill.root.is_dir() else place.parent
            )
            for source_file in skill.files:
                relative = Path(os.path.relpath(source_file, anchor)).as_posix()
                dest = posixpath.normpath(f"{dest_anchor}/{relative}")
                if (problem := _shared_place_problem(dest)) is not None:
                    self.report.warnings.append(
                        f"Skill {skill.name!r} will not work in the export: its file "
                        f"{source_file} would have to be copied to {dest}, {problem}"
                    )
                    continue
                earlier = self.copies.get(Path(dest))
                if earlier is not None and earlier != source_file:
                    self.report.warnings.append(
                        f"Skill {skill.name!r}: its file {source_file} would be copied to "
                        f"{dest}, where {earlier} already is; the skill uses that one instead"
                    )
                    continue
                self._copy(source_file, Path(dest))

    def _identical_copy_at(self, skill: _MergedSkill, place: PurePosixPath) -> bool:
        """Whether ``skills/<place>`` already holds this skill, copied from another source.

        Pairwise identical: every file of the skill, shared ones above its folder
        included, has an exported file at the position it would be copied to, with the
        same bytes; and the skill's own place holds no file the skill does not have, so
        a copy with an extra script, or one missing a file, is a different skill.
        """
        is_dir = skill.root.is_dir()
        anchor = skill.root if is_dir else skill.root.parent
        dest_anchor = PurePosixPath(SKILLS_DIR) / (place if is_dir else place.parent)
        wanted = {
            Path(
                posixpath.normpath(f"{dest_anchor}/{Path(os.path.relpath(f, anchor)).as_posix()}")
            ): f
            for f in skill.files
        }
        own = Path(SKILLS_DIR) / place
        there = {dest for dest in self.copies if dest == own or own in dest.parents}
        if not there or there != {dest for dest in wanted if dest == own or own in dest.parents}:
            return False
        return all(
            dest in self.copies and filecmp.cmp(self.copies[dest], source, shallow=False)
            for dest, source in wanted.items()
        )

    def _place_in_anchor(self, skill: _MergedSkill, bases: dict[Path, str]) -> PurePosixPath | None:
        """The place, relative to ``skills/``, of a merged skill whose files all sit in
        an anchor another entry copies, after copying any of them that entry did not
        need; ``None`` for a skill in no such anchor.

        Copied at the same position below the anchor's folder, so a file both reach is
        one destination with one source, and the skill's shared files keep their place.
        """
        for anchor, base in bases.items():
            if not all(f.is_relative_to(anchor) for f in [skill.root, *skill.files]):
                continue
            for source_file in skill.files:
                self._copy(source_file, Path(base) / source_file.relative_to(anchor))
            inside = PurePosixPath(base, skill.root.relative_to(anchor).as_posix())
            return inside.relative_to(SKILLS_DIR)
        return None

    def _skills_entry(self, agent: int, written: Any) -> _SkillsEntry | None:
        """A listed path other than an autodiscovery entry (`_merged_skills`)."""
        listed = self._listed_path(written, "skills")
        if listed is None:
            return None
        if listed.is_symlink():
            self.report.dropped.append(f"{written} (a symbolic link, which export does not follow)")
            return None
        if not listed.exists():
            self.report.dropped.append(f"{written} (does not exist)")
            return None
        listed = listed.resolve()
        files: set[Path] = set()
        for skill_root, skill_file in _listed_skills(listed):
            declared = {skill_file, *self._external_files(skill_file)}
            files.update(declared)
            if skill_root.is_dir():
                self._name_undeclared(skill_root, declared)
        if not files:
            self.report.dropped.append(f"{written} (no skill found in it)")
            return None
        return _SkillsEntry(agent, listed, files)

    def _name_undeclared(self, skill_root: Path, declared: set[Path]) -> None:
        """Warn about what a skill's folder holds beyond its declared files."""
        left = [
            path.relative_to(skill_root).as_posix()
            for path in _own_files(skill_root)
            if path not in declared
        ]
        if left:
            shown = ", ".join(left[:5]) + (f" and {len(left) - 5} more" if len(left) > 5 else "")
            self.report.warnings.append(
                f"{skill_root}: left out, because its SKILL.md does not list them under "
                f"external_files: {shown}"
            )

    def _external_files(self, skill_file: Path) -> list[Path]:
        self.skill_files.add(skill_file.resolve())
        try:
            descriptor, _ = parse_skill_file(skill_file)
        except (OSError, ValueError) as exc:
            # The loader skips it too; copied as it is, so it fails the same way there.
            self.report.warnings.append(
                f"{skill_file} could not be read as a skill ({type(exc).__name__}); "
                "copied unchanged"
            )
            return []
        found: list[Path] = []
        for item in resolve_external_files(skill_file, descriptor.requires.external_files):
            if item.problem or item.path is None:
                self.report.warnings.append(
                    f"Skill {descriptor.name!r} lists {item.entry!r} under external_files, "
                    f"but cannot use it ({item.problem}): the skill does not work here, and "
                    "will not in the export either"
                )
                continue
            # A folder entry opts in to what it holds, but not to hidden files: a
            # `.env` or a `.git` under `scripts/` is not what the author declared.
            files = _own_files(item.path) if item.path.is_dir() else expand_files(item.path)
            for file in files:
                if _hidden_below(file, skill_file.parent):
                    # Named outright, it is still refused: the skill may be a stranger's,
                    # and `../../.ssh/id_rsa` would travel in the export without a word.
                    note = f"{file} (hidden: listed by skill {descriptor.name!r}, never exported)"
                    if note not in self.report.dropped:
                        self.report.dropped.append(note)
                    continue
                found.append(file)
        return found

    def _export_skills(self) -> None:
        local_dir_type = LocalDirSkillSource.model_fields["type"].default
        entries: list[_SkillsEntry] = []
        sources: list[tuple[int, list[_SkillsEntry | _MergedSkills]]] = []
        merged: dict[tuple[Path, ...], _MergedSkills | None] = {}
        for index, agent in enumerate(self.config.get("agents") or []):
            skills = agent.get("skills") if isinstance(agent, dict) else None
            source = skills.get("source") if isinstance(skills, dict) else None
            if not isinstance(source, dict):
                continue
            if source.get("type") != local_dir_type:
                self.report.warnings.append(
                    f"Agent {agent.get('name')!r} loads its skills from a "
                    f"{source.get('type')!r} source, not from files: nothing to copy, the "
                    "configuration keeps it as it is"
                )
                continue
            # In the config's order, which settles a name clash. The autodiscovery
            # entries become one merged folder, listed where the first of them was.
            written_paths = list(source.get("paths") or [])
            folders = tuple(
                folder
                for written in written_paths
                if is_skill_autodiscovery(str(written))
                and (folder := self._autodiscovery_folder(written)) is not None
            )
            listed: list[_SkillsEntry | _MergedSkills] = []
            placed = False
            for written in written_paths:
                if is_skill_autodiscovery(str(written)):
                    if not placed and folders:
                        if folders not in merged:
                            merged[folders] = self._merged_skills(folders)
                        if (group := merged[folders]) is not None:
                            listed.append(group)
                        placed = True
                    continue
                if (entry := self._skills_entry(index, written)) is not None:
                    entries.append(entry)
                    listed.append(entry)
            sources.append((index, listed))

        bases = self._place_anchors(entries)
        for entry in entries:
            base = bases[entry.anchor]
            for source_file in entry.files:
                self._copy(source_file, Path(base) / source_file.relative_to(entry.anchor))
        # The folders another entry's copy is listed by, and so walked when loading.
        walked = set(bases.values())
        for group in merged.values():
            if group is not None:
                self._copy_merged(group, walked, bases)

        for index, listed in sources:
            paths: list[str] = []
            for item in listed:
                if isinstance(item, _MergedSkills):
                    # One entry per skill, in order: `skills/` itself also holds the
                    # other listed folders, which would load a second time.
                    for skill in item.skills:
                        path = _in_project(f"{SKILLS_DIR}/{skill.place}")
                        if path not in paths:
                            paths.append(path)
                    continue
                else:
                    relative = item.listed.relative_to(item.anchor).as_posix()
                    base = bases[item.anchor]
                    path = _in_project(base if relative == "." else f"{base}/{relative}")
                if path not in paths:
                    paths.append(path)
            self.config["agents"][index]["skills"]["source"]["paths"] = paths
            self.report.skill_paths.extend(p for p in paths if p not in self.report.skill_paths)

    def _place_anchors(self, entries: list[_SkillsEntry]) -> dict[Path, str]:
        """Give every entry its anchor, and every anchor its folder in the export.

        An anchor inside another one is merged into it, so a file is copied once
        whichever entries reach it.
        """
        for entry in entries:
            anchor = Path(
                os.path.commonpath(
                    [entry.listed if entry.listed.is_dir() else entry.listed.parent]
                    + [f.parent for f in entry.files]
                )
            )
            if anchor == self.install_dir or self.install_dir in anchor.parents:
                anchor = self.install_dir
            entry.anchor = anchor

        outer: list[Path] = []
        # Outermost first, then in the config's order: which of two same-named folders
        # gets the suffix must not change from one export to the next.
        in_order = list(dict.fromkeys(e.anchor for e in entries))
        for anchor in sorted(in_order, key=lambda p: len(p.parts)):
            if not any(o == anchor or o in anchor.parents for o in outer):
                outer.append(anchor)
        for entry in entries:
            entry.anchor = next(o for o in outer if o == entry.anchor or o in entry.anchor.parents)

        bases: dict[Path, str] = {}
        taken: set[str] = set()
        for anchor in outer:
            if anchor == self.install_dir:
                # Setup's directory, where a project's setup installs too.
                bases[anchor] = BUILTIN_SKILLS_DIRNAME
            else:
                bases[anchor] = self._unique(SKILLS_DIR, _folder_name(anchor), taken)
        return bases

    # -- standing instructions --------------------------------------------

    def _instruction_files(self, folder: Path) -> list[Path]:
        """What a folder contributes, as `LocalDirInstructionsSource` reads it: the
        soul, and the first of the accepted ``AGENTS.md`` spellings found."""
        soul = folder / LocalDirInstructionsSource.soul_filename
        agents = next(
            (
                folder / name
                for name in LocalDirInstructionsSource.agents_filenames
                if (folder / name).is_file()
            ),
            None,
        )
        return [f for f in (soul, agents) if f is not None and f.is_file()]

    def _export_instructions(self) -> None:
        # Every source, in the config's order: a listed folder or file, and the files
        # `.domyn` / `.agents` of the folder exported from hold (`LOCAL_INSTRUCTION_DIRS`).
        home = cli_config.GLOBAL_DIR.expanduser().resolve()
        sources: dict[Path, _InstructionSource] = {}
        rewritten: list[tuple[dict[str, Any], list[Path | str]]] = []
        for agent in self.config.get("agents") or []:
            runtime = agent.get("standing_instructions") if isinstance(agent, dict) else None
            source = runtime.get("source") if isinstance(runtime, dict) else None
            if not (isinstance(source, dict) and source.get("type") == "local_dir"):
                continue
            items: list[Path | str] = []
            for written in source.get("paths") or []:
                local = Path(str(written)) in LOCAL_INSTRUCTION_DIRS
                listed = self._listed_path(written, "instructions")
                if listed is None:
                    continue
                if local:
                    # Most folders have neither: normal, left out quietly. Exported from
                    # `$HOME`, `./.domyn` is `~/.domyn`, which is not an instructions
                    # folder (`resolve_instruction_dirs`).
                    if listed.is_symlink() or not listed.is_dir() or listed.resolve() == home:
                        continue
                    found = find_instruction_files(listed.resolve(), skip=STATE_DIRS)
                    for file in found.paths:
                        file = file.resolve()
                        sources.setdefault(file, _InstructionSource(file, [file]))
                        items.append(file)
                    continue
                if not listed.exists():
                    self.report.dropped.append(f"{written} (instructions: does not exist)")
                    continue
                listed = listed.resolve()
                if listed not in sources:
                    files = [listed] if listed.is_file() else self._instruction_files(listed)
                    if not files:
                        self.report.dropped.append(
                            f"{written} (instructions: no SOUL.md or AGENTS.md in it)"
                        )
                        continue
                    sources[listed] = _InstructionSource(listed, files)
                items.append(listed)
            rewritten.append((source, items))

        places = _place_instruction_sources(list(sources.values()))
        for item in sources.values():
            for file in item.files:
                copied = f"{places[item.path]}/{file.name}"
                self._copy(file.resolve(), Path(copied))
                self.report.instruction_files.append(copied)

        for source, items in rewritten:
            paths: list[str] = []
            for entry in items:
                if isinstance(entry, str):
                    path = entry
                elif entry.is_file():
                    path = _in_project(f"{places[entry]}/{entry.name}")
                else:
                    path = _in_project(places[entry])
                if path not in paths:
                    paths.append(path)
            source["paths"] = paths

    # -- secrets and what is not copied -----------------------------------

    def _scrub_secrets(self) -> None:
        taken: set[str] = set()

        def reference(value: Any, name: str) -> Any:
            if not isinstance(value, str) or not value or _ENV_REFERENCE.search(value):
                return value
            env = re.sub(r"[^A-Z0-9_]", "_", name.upper())
            n = 2
            candidate = env
            while candidate in taken:
                candidate, n = f"{env}_{n}", n + 1
            taken.add(candidate)
            return f"${{{candidate}}}"

        for provider in self.config.get("llm_providers") or []:
            if isinstance(provider, dict) and "api_key" in provider:
                hint = PROVIDER_HINTS.get(str(provider.get("type")), {}).get("env", "LLM_API_KEY")
                provider["api_key"] = reference(provider["api_key"], hint)

        for section in ("mcp_servers", "tools"):
            for entry in self.config.get(section) or []:
                headers = entry.get("headers") if isinstance(entry, dict) else None
                if isinstance(headers, dict):
                    for header, value in headers.items():
                        if _SECRET_HEADER.search(str(header)):
                            headers[header] = reference(value, f"{entry.get('name')}_{header}")

        for oauth in self.config.get("oauth_providers") or []:
            if isinstance(oauth, dict) and "client_secret" in oauth:
                oauth["client_secret"] = reference(
                    oauth["client_secret"], f"{oauth.get('name')}_client_secret"
                )

    def _export_code(self) -> None:
        """Copy the file each ``tools``/``hooks`` entry runs into ``code/``, and point the
        entry at the copy. The file alone, as declared: a module it imports from beside
        it is not copied. One that cannot be copied keeps its path, which
        `_refuse_paths_outside` then refuses."""
        places: dict[Path, str] = {}
        taken: set[str] = set()
        for section in ("tools", "hooks"):
            for entry in self.config.get(section) or []:
                if not isinstance(entry, dict) or not entry.get("path"):
                    continue
                written = entry["path"]
                if isinstance(written, str) and _inside_project(written):
                    continue  # a project's own file, already named in its terms
                path = self._listed_path(written, f"{section[:-1]} code")
                if path is None or path.is_symlink() or not path.is_file():
                    continue
                path = path.resolve()
                if path not in places:
                    name, n = path.name, 2
                    while name in taken:
                        name, n = f"{path.stem}-{n}{path.suffix}", n + 1
                    taken.add(name)
                    places[path] = f"{CODE_DIR}/{name}"
                    self._copy(path, Path(places[path]))
                    self.report.warnings.extend(import_warnings(path, places[path]))
                entry["path"] = _in_project(places[path])

    def _refuse_paths_outside(self) -> None:
        """Refuse the export when the config it would write still names a file or
        folder outside the exported folder.

        An export goes to someone else, and is run there as a project: a path that is
        not ``${DOMYN_PROJECT_DIR}/…`` names a place on the machine that runs it, which
        then reads it (an instruction file goes to the model every turn), runs it (a
        tool, a hook, a browser) or works in it (a workspace, a token store). What
        export copied in is rewritten into the project, so only what it left as written
        is refused, every one in one message.
        """
        outside = paths_outside_project(self.config) + external_files_outside(
            (PurePosixPath(dest.as_posix()), source)
            for dest, source in self.copies.items()
            if source.resolve() in self.skill_files
        )
        if outside:
            raise ExportError(
                "the configuration names paths outside the exported folder, which would "
                "be read on the machine that imports it:\n"
                + "".join(f"  - {line}\n" for line in outside)
                + f"Remove them, or put the files in the project and name them as "
                f"{PROJECT_DIR_REF}/…"
            )

    # -- the whole thing --------------------------------------------------

    def plan(self) -> None:
        self._export_skills()
        self._export_instructions()
        self._export_code()
        self._scrub_secrets()
        self._refuse_paths_outside()
        text = yaml.safe_dump(self.config, sort_keys=False)
        # DOMYN_PROJECT_DIR is set by the CLI for whichever folder is the project.
        self.report.env_vars = [
            name for name in dict.fromkeys(_ENV_REFERENCE.findall(text)) if name != PROJECT_DIR_VAR
        ]
        self.report.files_copied = len(self.copies)


def import_warnings(source: Path, copied: str) -> list[str]:
    """What the copy of a tool's or hook's file at ``copied`` imports that may not be
    there where the export runs, one warning per kind, for the person sending it.

    Read, never run: the file is parsed and each top-level module looked up as Python
    would, without importing it. The standard library and domyn itself with its own
    dependencies are there wherever domyn runs; anything else is named — a module
    beside the file (not copied), a relative import (the same), a package installed
    here (to install there) and a module found nowhere (it fails here already).
    """
    try:
        tree = ast.parse(source.read_bytes(), filename=str(source))
    except (OSError, SyntaxError, ValueError) as exc:
        return [
            f"{copied}: could not be read as Python ({type(exc).__name__}), so its imports were not checked"
        ]
    names: set[str] = set()
    relative: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative.add("." * node.level + (node.module or ""))
            elif node.module:
                names.add(node.module.split(".")[0])

    available = _available_everywhere()
    distributions = importlib.metadata.packages_distributions()
    beside: list[str] = []
    installed: dict[str, list[str]] = {}
    missing: list[str] = []
    for name in sorted(names):
        if name in sys.stdlib_module_names or name in available:
            continue
        if (source.parent / f"{name}.py").is_file() or (source.parent / name).is_dir():
            beside.append(name)
            continue
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(name)
            continue
        packages = distributions.get(name) or [name]
        if not {_normal(p) for p in packages} & available:
            installed.setdefault(", ".join(sorted(set(packages))), []).append(name)

    warnings: list[str] = []
    if beside or relative:
        shown = ", ".join([*beside, *sorted(relative)])
        warnings.append(
            f"{copied} imports modules from beside {source} that export does not copy "
            f"({shown}): it will not load where they are not importable"
        )
    if installed:
        shown = ", ".join(f"{package} ({', '.join(mods)})" for package, mods in installed.items())
        warnings.append(
            f"{copied} needs packages domyn does not install: {shown}. The machine running "
            "the export must have them installed, or it will not load"
        )
    if missing:
        warnings.append(
            f"{copied} imports {', '.join(missing)}, which cannot be found here either: "
            "it does not load on this machine, and will not in the export"
        )
    return warnings


def _available_everywhere() -> set[str]:
    """What any machine running domyn can import besides the standard library: domyn
    itself and the packages it requires (names normalized, `_normal`), extras aside."""
    names = {"domyn"}
    try:
        requirements = importlib.metadata.requires("domyn") or []
    except importlib.metadata.PackageNotFoundError:
        return names
    for requirement in requirements:
        if "extra ==" in requirement:
            continue
        if match := re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement):
            names.add(_normal(match.group()))
    return names


def _normal(name: str) -> str:
    """A distribution's name as PEP 503 compares them."""
    return re.sub(r"[-_.]+", "-", name).lower()


def paths_outside_project(config: dict[str, Any]) -> list[str]:
    """Every path ``config`` names outside its project, as ``<where>: <path>``.

    Empty for what export writes, and checked again by an import
    (`project_import.import_project`): an export is a folder anyone can edit before
    sending it, so the check that protects the machine importing it is the one that
    runs there."""
    return [
        f"{where}: {value}" for where, value in config_paths(config) if not _inside_project(value)
    ]


def external_files_outside(skill_files: Iterable[tuple[PurePosixPath, Path]]) -> list[str]:
    """Every ``external_files`` entry that leads out of the project, as ``<where>:
    <entry>``, for skill files given as ``(place in the project, file to read)``.

    An entry is relative to its ``SKILL.md``, so where the copy sits decides where it
    leads: a file export did not copy (a hidden one, a shared one it could not place)
    leaves the entry pointing above the project, at whatever the machine running it
    has there, and a later export of that project would copy it in. An absolute entry
    is not counted: the loader never resolves one."""
    found: list[str] = []
    for place, source in skill_files:
        try:
            descriptor, _ = parse_skill_file(source)
        except (OSError, ValueError):
            continue
        for entry in descriptor.requires.external_files:
            if posixpath.isabs(entry):
                continue
            normal = posixpath.normpath(posixpath.join(str(place.parent), entry))
            if normal == ".." or normal.startswith("../"):
                found.append(f"skill {descriptor.name!r} ({place}) external_files: {entry}")
    return found


def _copy_into(staging: Path, relative: Path, source: Path, chosen: set[Path]) -> None:
    """Copy ``source`` to ``staging / relative``, checking both ends first.

    What reaches here was planned by `_Exporter`, which already refuses what may not
    travel; this is the last check before a byte is read, and holds whatever a later
    change to the planning does: the target must resolve inside ``staging`` (no ``..``,
    no absolute ``relative``), and the source must be one the plan chose, a regular file
    and no link, and none of the files a domyn home keeps for its owner."""
    root = staging.resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or target == root:
        raise ExportError(f"{relative} would be written outside the exported folder")
    if source not in chosen:
        raise ExportError(f"{source} was not chosen for the export")
    if source.is_symlink() or not source.is_file() or is_private(source):
        raise ExportError(f"{source} is not a file export may copy")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target, follow_symlinks=False)


def config_paths(config: dict[str, Any]) -> list[tuple[str, Any]]:
    """Every value in ``config`` that names a file or folder, with where it is.

    The places the loader reads a path from: the ``local_dir`` skills and instructions
    of every agent, the code of ``tools`` and ``hooks``, a tool bundle's workspace and
    browser, an OAuth provider's token store. A browser ``binary`` without a ``/`` is
    a program looked up on ``PATH``, not a path.
    """
    found: list[tuple[str, Any]] = []
    for agent in config.get("agents") or []:
        if not isinstance(agent, dict):
            continue
        for key in ("skills", "standing_instructions"):
            spec = agent.get(key)
            source = spec.get("source") if isinstance(spec, dict) else None
            if isinstance(source, dict) and source.get("type", "local_dir") == "local_dir":
                for path in source.get("paths") or []:
                    found.append((f"agent {agent.get('name')!r} {key}", path))
    for section in ("tools", "hooks"):
        for entry in config.get(section) or []:
            if isinstance(entry, dict) and entry.get("path"):
                found.append((f"{section[:-1]} {entry.get('name')!r} code", entry["path"]))
    for bundle in config.get("tool_bundles") or []:
        if not isinstance(bundle, dict):
            continue
        name = bundle.get("name")
        if bundle.get("workspace"):
            found.append((f"tool bundle {name!r} workspace", bundle["workspace"]))
        browser = bundle.get("browser")
        if isinstance(browser, dict):
            if browser.get("executable_path"):
                found.append(
                    (f"tool bundle {name!r} browser.executable_path", browser["executable_path"])
                )
            binary = browser.get("binary")
            if isinstance(binary, str) and ("/" in binary or binary.startswith("~")):
                found.append((f"tool bundle {name!r} browser.binary", binary))
    for oauth in config.get("oauth_providers") or []:
        if isinstance(oauth, dict) and oauth.get("storage_dir"):
            found.append(
                (f"OAuth provider {oauth.get('name')!r} storage_dir", oauth["storage_dir"])
            )
    return found


def _inside_project(value: Any) -> bool:
    """Whether a path as the config writes it stays inside the project:
    ``${DOMYN_PROJECT_DIR}`` or below it, with no ``..`` climbing out and no other
    variable, which could be set to anything where it is run."""
    if not isinstance(value, str):
        return False
    if value == PROJECT_DIR_REF:
        return True
    if not value.startswith(f"{PROJECT_DIR_REF}/"):
        return False
    rest = value.removeprefix(f"{PROJECT_DIR_REF}/")
    if "$" in rest:
        return False
    normal = posixpath.normpath(rest)
    return normal != ".." and not normal.startswith("../")


def _in_project(relative: str) -> str:
    """A path inside the export, as its config names it."""
    return f"{PROJECT_DIR_REF}/{relative}"


def is_private(path: Path) -> bool:
    """Whether ``path`` is one of the files a domyn home keeps for its owner alone:
    the credentials, the approvals, the conversation history (with SQLite's side
    files). Matched by name wherever it is, so no layout of skills can carry one out.
    """
    name = path.name
    return name in (CREDENTIALS_FILENAME, APPROVALS_FILENAME) or name.startswith(STORE_FILENAME)


def _listed_skills(listed: Path) -> list[tuple[Path, Path]]:
    """``(skill folder or loose file, its skill file)`` for what a listed path loads.

    The loader's reading of a path (`skills.sources._iter_skill_files`), keeping the
    skill's own folder, which is what gets copied.
    """
    if listed.is_file():
        return [(listed, listed)] if listed.suffix.lower() == ".md" else []
    own = own_skill_file(listed)
    if own is not None:
        return [(listed, own)]
    return [(found.path, found.skill_file) for found in walk_skill_tree(listed)]


def _folder_name(anchor: Path) -> str:
    """What an anchor is called inside ``skills/``.

    A folder called ``skills`` is named after the folder holding it, or the export
    would read ``skills/skills``. A leading dot is dropped, so a copy never lands in a
    folder the search tools skip.
    """
    name = anchor.parent.name if anchor.name == SKILLS_DIR else anchor.name
    return name.lstrip(".") or SKILLS_DIR


def _place_instruction_sources(sources: list[_InstructionSource]) -> dict[Path, str]:
    """Each source's folder in the export, with the least nesting that keeps them all.

    No two files with the same name (``AGENTS.md`` and ``AGENT.md`` counting as one, as
    a folder contributes only the first): all of them in ``instructions/``. Otherwise
    each source gets a folder of its own, named by the fewest trailing parts of its
    path (a file's: its folder's) that tell them apart — ``dir/a`` and ``dir/b`` are
    ``instructions/a`` and ``instructions/b``, ``x/a`` and ``y/a`` are
    ``instructions/x/a`` and ``instructions/y/a`` — without leading dots, so nothing
    lands in a folder the search tools skip. Nothing is lost either way: every
    ``SOUL.md`` and ``AGENTS.md`` is read, not just the first.
    """
    agents_names = {n.upper() for n in LocalDirInstructionsSource.agents_filenames}

    def slot(file: Path) -> str:
        return "AGENTS.md" if file.name.upper() in agents_names else file.name

    slots = [slot(f) for source in sources for f in source.files]
    if len(slots) == len(set(slots)):
        return {source.path: INSTRUCTIONS_DIR for source in sources}

    def folder_of(source: _InstructionSource) -> Path:
        return source.path if source.path.is_dir() else source.path.parent

    folders = list(dict.fromkeys(folder_of(s) for s in sources))
    names: dict[Path, str] = {}
    for k in range(1, max(len(f.parts) for f in folders) + 1):
        candidate = {
            f: "/".join(part.lstrip(".") or "_" for part in f.parts[1:][-k:]) for f in folders
        }
        if len(set(candidate.values())) == len(folders):
            names = candidate
            break
    return {s.path: f"{INSTRUCTIONS_DIR}/{names[folder_of(s)]}" for s in sources}


def _shared_place_problem(dest: str) -> str | None:
    """Why a file a skill shares cannot be exported at ``dest`` (relative to the
    export), or ``None``.

    Anywhere inside the export will do, except what the project itself reads there:
    its ``config.yaml``, the ``instructions/`` folder, whose files are those of the
    config, and ``builtin-skills/``, which the project's setup rewrites from scratch.
    """
    if dest == ".." or dest.startswith("../") or posixpath.isabs(dest):
        return "outside the exported folder"
    top = PurePosixPath(dest).parts[0]
    if dest == CONFIG_FILENAME:
        return "where the project's own configuration goes"
    if dest == INFO_FILENAME:
        return "where the export records the version that made it"
    if top == INSTRUCTIONS_DIR:
        return f"inside {INSTRUCTIONS_DIR}/, which holds the config's instruction files"
    if top == BUILTIN_SKILLS_DIRNAME:
        return f"inside {BUILTIN_SKILLS_DIRNAME}/, which setup rewrites from scratch"
    return None


def _skill_name(skill_file: Path, skill_root: Path) -> str:
    """The name the loader gives a skill: its frontmatter's, else its folder's or file's."""
    try:
        descriptor, _ = parse_skill_file(skill_file)
        return descriptor.name
    except (OSError, ValueError):
        return skill_root.stem


def _hidden_below(path: Path, folder: Path) -> bool:
    """Whether ``path``, written relative to ``folder``, has a hidden folder or file in
    it (``.ssh``, ``.env``); the ``..`` that climb out of ``folder`` are not."""
    parts = Path(os.path.relpath(path, folder)).parts
    return any(part.startswith(".") and part not in (".", "..") for part in parts)


def _own_files(skill_folder: Path) -> list[Path]:
    """What a folder holds, without hidden files and folders: ``.DS_Store``, a
    ``.git``, a ``.env`` with a key in it."""
    return [
        path
        for path in expand_files(skill_folder)
        if not any(part.startswith(".") for part in path.relative_to(skill_folder).parts)
    ]


def export_configuration(
    raw: dict[str, Any],
    dest: Path,
    *,
    cwd: Path | None = None,
    install_dir: Path | None = None,
) -> ExportReport:
    """Write the configuration ``raw`` and every file it reads into a new folder ``dest``.

    ``cwd`` is what a relative path in ``raw`` is read against (the working directory,
    as the loader reads it), and ``install_dir`` is where ``domyn setup`` installs skills.

    ``dest`` must not exist, or be an empty folder. It is built beside itself and
    moved into place, so a failure leaves nothing behind. ``raw`` is not modified.
    """
    dest = dest.expanduser().absolute()
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ExportError(
            f"{dest} already exists and is not an empty folder. Choose a new folder name."
        )

    exporter = _Exporter(
        raw,
        cwd=(cwd or Path.cwd()).absolute(),
        install_dir=install_dir or skills_install_dir(),
    )
    exporter.plan()

    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dest.name}.", dir=dest.parent))
    try:
        chosen = set(exporter.copies.values())
        for relative, source in exporter.copies.items():
            _copy_into(staging, relative, source, chosen)
        (staging / CONFIG_FILENAME).write_text(
            yaml.safe_dump(exporter.config, sort_keys=False), encoding="utf-8"
        )
        (staging / INFO_FILENAME).write_text(
            yaml.safe_dump(export_info(), sort_keys=False), encoding="utf-8"
        )
        if dest.exists():
            dest.rmdir()
        staging.rename(dest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    exporter.report.dest = dest
    return exporter.report


def print_export_report(console: Console, theme: Theme, source: Path, report: ExportReport) -> None:
    """What an export copied, what it left behind, and how to use the result: printed
    the same by ``domyn export`` and ``/export``."""
    meta, warning = theme.style("meta"), theme.style("warning")
    # Soft-wrapped: a path cut across lines by the console cannot be copied whole.
    console.print(
        f"Exported {source} to {report.dest}",
        style=theme.style("accent_bold"),
        highlight=False,
        soft_wrap=True,
    )
    count = report.files_copied
    console.print(f"  Copied {count} file{'' if count == 1 else 's'}", style=meta)
    _print_list(console, "Skills:", [_in_export(path) for path in report.skill_paths], meta)
    _print_list(console, "Instructions:", report.instruction_files, meta)
    _print_list(console, "Not exported:", report.dropped, warning)
    _print_list(console, "Check before using it:", report.warnings, warning)
    if report.env_vars:
        console.print(
            "  Before running it, set "
            + ", ".join(report.env_vars)
            + ": in your shell, in a .env file, or in the project's credentials file",
            style=warning,
            highlight=False,
        )

    # Named after the folder when that is a valid project name, as it usually is.
    try:
        project_dir(report.dest.name)
        name = report.dest.name
    except ProjectError:
        name = "<name>"
    console.print()
    console.print("  The folder is a project. To use it:", style=meta, highlight=False)
    for how, command in (
        ("where it is", f"domyn run --project {report.dest}"),
        (
            f"with your other projects, once moved to {_with_tilde(projects_dir() / name)}",
            f"domyn run --project {name}",
        ),
        ("as a new project, reviewing each setting", f"domyn setup --import {report.dest}"),
    ):
        console.print(f"    {how}:", style=meta, highlight=False)
        console.print(f"      {command}", style=meta, highlight=False, markup=False, soft_wrap=True)


def _print_list(console: Console, title: str, items: list[str], style: str) -> None:
    if not items:
        return
    console.print(f"  {title}", style=style, highlight=False)
    for item in items:
        console.print(f"    {item}", style=style, highlight=False, markup=False)


def _in_export(path: str) -> str:
    """A path of the exported config as the reader sees it: relative to the export."""
    return path.removeprefix(f"{PROJECT_DIR_REF}/")


def _with_tilde(path: Path) -> str:
    try:
        return f"~/{path.relative_to(Path.home()).as_posix()}"
    except ValueError:
        return str(path)
