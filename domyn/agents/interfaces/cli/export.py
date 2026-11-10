"""``domyn export``: a copy of the CLI configuration someone else can run.

The result is a folder laid out like ``~/.domyn`` — ``config.yaml``, ``builtin-skills/``,
``resources/``, ``code/`` — that holds everything the configuration reads from disk,
with every path to a copied file rewritten as ``${DOMYN_PROJECT_DIR}/...``,
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
other anchor is ``resources/<folder name>/``, with a suffix when the name is taken:
listed skill folders, and the roots of a resource read through a subpath, alike.
Everything a resource reads that is not a built-in is under ``resources/``. The
config's paths are rewritten one by one, so their order, which settles a name clash,
is kept.

The ``.domyn/skills`` and ``.agents/skills`` autodiscovery entries are
**snapshotted** and **merged**: the skills in those folders of the folder the export
runs in are copied straight into ``resources/skills/``, categories kept
(``.domyn/skills/<category>/<name>`` → ``resources/skills/<category>/<name>``), and listed one
by one where the first entry was, so the export carries what a run here would load. A
name both folders hold is settled before copying, as the loader settles it: the
earlier folder's skill (``.domyn``'s, in the order setup writes) is copied, the other
is named in the report. A place already taken in ``resources/skills/`` gets a suffix, unless
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
approvals, the state file and conversation history are never copied: nothing names them, and a file
called like one is refused even when an ``external_files`` entry does (`is_private`),
since a project keeps all three in its own folder, next to the files that are exported.
Nor is the session's working state (`StateFolders`): what is below a home's or
project's ``conversations/``, ``processes/`` or ``browser/`` is refused the same way.
"""

from __future__ import annotations

import ast
import contextlib
import copy
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
    find_instruction_files,
)
from domyn.agents.interfaces.cli import config as cli_config
from domyn.agents.interfaces.cli.config import (
    CONFIG_FILENAME,
    PROJECT_DIR_VAR,
    PROVIDER_HINTS,
    PrivateFiles,
    ProjectError,
    is_skill_autodiscovery,
    project_dir,
    projects_dir,
)
from domyn.agents.interfaces.cli.theme import Theme
from domyn.agents.loader import interpolate_env_vars
from domyn.agents.skills import parse_skill_file
from domyn.agents.skills.external_files import expand_files, resolve_external_files
from domyn.agents.skills.layout import own_skill_file, walk_skill_tree
from domyn.agents.tools.builtin.session import StateFolders

AUTODISCOVERY_EXPORT = "autodiscovery_export"
"""What the autodiscovery resource found when the export was made, copied: read beside
it, which goes on reading the folder the project runs in."""

RESOURCES_DIR = "resources"
"""Where every folder the configuration's skills and instructions are read from is
copied, one folder each (`_Exporter._place_roots`)."""

CODE_DIR = "code"
"""Where the Python files of ``tools`` and ``hooks`` entries are copied."""

INFO_FILENAME = ".info"
"""What an export says about itself, beside ``config.yaml`` (`export_info`): which
version of domyn made it, when, and on what. Not a configuration: nothing
loads it, and an import reads it and does not copy it."""

EXPORT_FORMAT_VERSION = 2
"""The version of what an export is: its folder layout (``builtin-skills/``,
``resources/`` with ``resources/skills/`` and ``resources/instructions/``, ``code/``;
version 1 had ``skills/`` and ``instructions/`` at the top), how its config names those files, and the
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
class _Root:
    """One folder (or file) a skills or instructions source reads, and what it reads
    there: one entry of a ``local_dir`` resource's or source's ``paths``."""

    holder: dict[str, Any]
    """The mapping whose ``paths`` lists it: a ``local_dir`` resource or source."""
    index: int
    """Its place in that list."""
    path: Path
    """Resolved."""
    files: set[Path]
    """What its sources read under it, every one inside `folder`."""
    instructions: list[Path] = field(default_factory=list)
    """Those of `files` that are instruction files, for the report."""
    anchor: Path = field(default=Path())
    """The outermost root folder holding it (`_place_roots`)."""

    @property
    def folder(self) -> Path:
        """The folder it is: itself, or the folder of a root that is one file."""
        return self.path.parent if self.path.is_file() else self.path


class _Exporter:
    def __init__(self, raw: dict[str, Any], cwd: Path) -> None:
        self.config = copy.deepcopy(raw)
        self.cwd = cwd
        self.copies: dict[Path, Path] = {}
        """Destination, relative to the export → source file."""
        self.skill_files: set[Path] = set()
        """Every skill file read, resolved: whose ``external_files`` the copy must keep
        inside the export (`_refuse_paths_outside`)."""
        self.outside: list[str] = []
        """Files a source needs from outside the folder it reads them through."""
        self.report = ExportReport(dest=Path())
        self.places = PrivatePlaces.now()

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
        if is_private(source, self.places):
            # Only an `external_files` entry walking up into a domyn home can name one
            # (a file, or what is in its `conversations/`, `processes/` or `browser/`):
            # refused all the same, as the export is for someone else.
            note = f"{source} (API keys, approvals, history and session state are never exported)"
            if note not in self.report.dropped:
                self.report.dropped.append(note)
            return
        previous = self.copies.setdefault(dest, source)
        if previous != source:
            # Root folders are merged when nested and uniquely named otherwise, so two
            # sources for one destination is a bug here, not a property of the config.
            raise ExportError(f"two files would be exported as {dest}: {previous} and {source}")

    def _unique(self, parent: str, name: str, taken: set[str]) -> str:
        candidate, n = f"{parent}/{name}", 2
        while candidate in taken:
            candidate, n = f"{parent}/{name}-{n}", n + 1
        taken.add(candidate)
        return candidate

    # -- what the sources read ---------------------------------------------

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

    def _skill_files(self, under: Path, folder: Path) -> set[Path]:
        """The skill files under ``under`` and what their ``external_files`` declare.

        A declared file outside ``folder``, the root it is read through, is recorded in
        `outside`: the export would have no place for it that keeps the skill working.
        """
        if under.is_symlink() or not under.exists():
            return set()
        files: set[Path] = set()
        for skill_root, skill_file in _listed_skills(under.resolve()):
            declared = {skill_file, *self._external_files(skill_file)}
            for file in sorted(declared):
                if not file.resolve().is_relative_to(folder):
                    self.outside.append(
                        f"{file} (needed by the skill at {skill_root}, outside {folder}, "
                        "the folder it is read through)"
                    )
                    declared.discard(file)
            files.update(declared)
            if skill_root.is_dir():
                self._name_undeclared(skill_root, declared)
        return files

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

    def _read_instruction_files(self, under: Path, walk: bool) -> list[Path]:
        """The ``SOUL.md`` and ``AGENTS.md`` a root contributes, as the loader reads it:
        walked when its entry says ``walk``, and nothing when that is ``~/.domyn``."""
        if under.is_symlink():
            return []
        if walk:
            home = cli_config.GLOBAL_DIR.expanduser().resolve()
            if not under.is_dir() or under.resolve() == home:
                return []
            found = find_instruction_files(under.resolve())
            return [file.resolve() for file in found.paths]
        if under.is_file():
            return [under.resolve()]
        if under.is_dir():
            return [file.resolve() for file in self._instruction_files(under)]
        return []

    def _reads(self) -> dict[int, tuple[dict[str, Any], list[tuple[str, str, bool]]]]:
        """Every list of paths a skills or instructions source reads, by identity, with
        what is read under each path: ``(key, subpath, walk)``.

        The list of each ``local_dir`` resource a ``type: resource`` source names. A
        resource two sources read is one entry, read for both. Any other source has no
        files to copy, and is said so; a ``local_dir`` source, which no configuration
        loads, refuses the export.
        """
        declared = cli_config.declared_resources(self.config)
        reads: dict[int, tuple[dict[str, Any], list[tuple[str, str, bool]]]] = {}

        def read(holder: dict[str, Any], use: tuple[str, str, bool]) -> None:
            if isinstance(holder.get("paths"), str):
                holder["paths"] = [holder["paths"]]
            uses = reads.setdefault(id(holder), (holder, []))[1]
            if use not in uses:
                uses.append(use)

        for agent in self.config.get("agents") or []:
            if not isinstance(agent, dict):
                continue
            for key in ("skills", "standing_instructions"):
                runtime = agent.get(key)
                source = runtime.get("source") if isinstance(runtime, dict) else None
                if not isinstance(source, dict):
                    continue
                kind = source.get("type", "local_dir")
                if kind == "local_dir":
                    raise ExportError(
                        f"Agent {agent.get('name')!r} reads its {key} from a 'local_dir' "
                        "source, which a configuration no longer accepts: declare the "
                        "folders as a resource and name it, or run `domyn setup` again"
                    )
                if kind == "resource":
                    for resource_id, subpath, walk in cli_config.source_refs(source, key):
                        resource = declared.get(resource_id)
                        if isinstance(resource, dict) and resource.get("type") == "local_dir":
                            read(resource, (key, subpath, walk))
                        else:
                            self.report.warnings.append(
                                f"Agent {agent.get('name')!r} reads its {key} from resource "
                                f"{resource_id!r}, which is not a local folder: nothing to copy"
                            )
                else:
                    self.report.warnings.append(
                        f"Agent {agent.get('name')!r} loads its {key} from a {kind!r} source, "
                        "not from files: nothing to copy, the configuration keeps it as it is"
                    )
        return reads

    def _roots(
        self, reads: dict[int, tuple[dict[str, Any], list[tuple[str, str, bool]]]]
    ) -> list[_Root]:
        """Every path each list of `_reads` holds, with the files read under it.

        A path that is a link, or that names nothing here, is left out, the latter
        quietly (``.domyn/skills`` is missing from most folders); `_rewrite` drops it
        from its list. Every file kept is inside the path's folder (`_skill_files`)."""
        roots: list[_Root] = []
        for holder, uses in reads.values():
            for index, written in enumerate(list(holder.get("paths") or [])):
                listed = self._listed_path(written, "a resource path")
                if listed is None:
                    continue
                if listed.is_symlink():
                    self.report.dropped.append(
                        f"{written} (a symbolic link, which export does not follow)"
                    )
                    continue
                if not listed.exists():
                    if not (
                        is_skill_autodiscovery(str(written))
                        or Path(str(written)) in LOCAL_INSTRUCTION_DIRS
                    ):
                        self.report.dropped.append(f"{written} (does not exist)")
                    continue
                path = listed.resolve()
                folder = path.parent if path.is_file() else path
                files: set[Path] = set()
                instructions: list[Path] = []
                for key, subpath, walk in uses:
                    under = path if subpath == "." else path / subpath
                    if key == "skills":
                        files.update(self._skill_files(under, folder))
                    else:
                        found = self._read_instruction_files(under, walk)
                        files.update(found)
                        instructions.extend(f for f in found if f not in instructions)
                if not files:
                    self.report.dropped.append(f"{written} (nothing is read from it)")
                    continue
                roots.append(_Root(holder, index, path, files, instructions))
        return roots

    # -- placing and copying ------------------------------------------------

    def _place_roots(self, roots: list[_Root]) -> dict[Path, str]:
        """Each root's anchor — the outermost root folder holding it — and each anchor's
        folder in the export, ``resources/<its name>``.

        A root inside another root's folder (``.domyn`` beside ``.``) is copied as part of
        it, so its files keep their place relative to the other's. Names follow the
        folder (a folder called ``skills`` takes its parent's name, a leading dot is
        dropped), with ``-2``… when taken.
        """
        outer: list[Path] = []
        # Outermost first, then in the config's order: which of two same-named folders
        # gets the suffix must not change from one export to the next.
        in_order = list(dict.fromkeys(root.folder for root in roots))
        for folder in sorted(in_order, key=lambda p: len(p.parts)):
            if not any(o == folder or o in folder.parents for o in outer):
                outer.append(folder)
        for root in roots:
            root.anchor = next(o for o in outer if o == root.folder or o in root.folder.parents)
        taken: set[str] = set()
        return {
            anchor: self._unique(RESOURCES_DIR, _folder_name(anchor), taken) for anchor in outer
        }

    def _export_resources(self) -> None:
        """Copy what every skills and instructions source reads, by position, and point
        each path at its copy (see the module docstring)."""
        reads = self._reads()
        roots = self._roots(reads)
        if self.outside:
            raise ExportError(
                "skills need files outside the folder they are read through, which the "
                "export could not carry without breaking them:\n"
                + "".join(f"  - {line}\n" for line in self.outside)
                + "Move the files inside that folder, or list the folder holding both."
            )
        bases = self._place_roots(roots)
        for root in roots:
            for file in sorted(root.files):
                dest = Path(bases[root.anchor]) / file.relative_to(root.anchor)
                self._copy(file, dest)
                if (
                    file in root.instructions
                    and dest.as_posix() not in self.report.instruction_files
                ):
                    self.report.instruction_files.append(dest.as_posix())
        self._rewrite(reads, roots, bases)

    def _rewrite(
        self,
        reads: dict[int, tuple[dict[str, Any], list[tuple[str, str, bool]]]],
        roots: list[_Root],
        bases: dict[Path, str],
    ) -> None:
        """Every list of paths, rewritten in place: each path that was copied, as its copy,
        in its order; one that was not, left out.

        Except the autodiscovery resource (`_snapshot_autodiscovery`): it keeps its
        paths, and its copies become a resource beside it."""
        rewritten: dict[int, tuple[dict[str, Any], list[str]]] = {}
        for holder, _ in reads.values():
            rewritten.setdefault(id(holder), (holder, []))
        for root in sorted(roots, key=lambda r: r.index):
            relative = root.path.relative_to(root.anchor).as_posix()
            base = bases[root.anchor]
            path = _in_project(base if relative == "." else f"{base}/{relative}")
            paths = rewritten[id(root.holder)][1]
            if path not in paths:
                paths.append(path)
        snapshot: list[str] = []
        for holder, paths in rewritten.values():
            if _is_autodiscovery(self.config, holder):
                snapshot = paths
            else:
                holder["paths"] = paths
        # Last: a snapshot an earlier export left is rewritten above, then added to.
        self._snapshot_autodiscovery(snapshot)
        cli_config.drop_empty_resources(self.config)
        for agent in self.config.get("agents") or []:
            if isinstance(agent, dict):
                for written in (
                    cli_config.source_paths(self.config, agent.get("skills"), "skills") or []
                ):
                    if written not in self.report.skill_paths:
                        self.report.skill_paths.append(written)

    def _snapshot_autodiscovery(self, copies: list[str]) -> None:
        """Keep ``autodiscovery`` reading the folder the project runs in, and add what
        it found here, copied, as `AUTODISCOVERY_EXPORT`, read beside it.

        Skills name the snapshot just after the live folders, so the folder a session
        starts in wins a name; instructions just before them, so its brief is still the
        last one read. An export of a project that already has a snapshot adds to it,
        and the sources already name it.
        """
        if not copies:
            return
        declared = cli_config.declared_resources(self.config)
        existing = declared.get(AUTODISCOVERY_EXPORT)
        if existing is not None:
            paths = existing.get("paths") or []
            paths = [paths] if isinstance(paths, str) else list(paths)
            existing["paths"] = [*paths, *(c for c in copies if c not in paths)]
            return
        self.config.setdefault("resources", []).append(
            {"id": AUTODISCOVERY_EXPORT, "type": "local_dir", "paths": copies}
        )
        for agent in self.config.get("agents") or []:
            if not isinstance(agent, dict):
                continue
            for key in ("skills", "standing_instructions"):
                runtime = agent.get(key)
                source = runtime.get("source") if isinstance(runtime, dict) else None
                if not isinstance(source, dict) or source.get("type") != "resource":
                    continue
                entries = source.get("resource")
                entries = [entries] if isinstance(entries, str | dict) else list(entries or [])
                for at, entry in enumerate(entries):
                    entry_id = entry if isinstance(entry, str) else (entry or {}).get("id")
                    if entry_id != cli_config.CLI_AUTODISCOVERY_RESOURCE:
                        continue
                    copy_entry = (
                        AUTODISCOVERY_EXPORT
                        if isinstance(entry, str)
                        else {**entry, "id": AUTODISCOVERY_EXPORT}
                    )
                    entries.insert(at + 1 if key == "skills" else at, copy_entry)
                    break
                source["resource"] = entries

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
                f"${{{PROJECT_DIR_VAR}}}/…"
            )

    # -- the whole thing --------------------------------------------------

    def plan(self) -> None:
        self._export_resources()
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


def _copy_into(
    staging: Path, relative: Path, source: Path, chosen: set[Path], places: PrivatePlaces
) -> None:
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
    if source.is_symlink() or not source.is_file() or is_private(source, places):
        raise ExportError(f"{source} is not a file export may copy")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target, follow_symlinks=False)


def config_paths(config: dict[str, Any]) -> list[tuple[str, Any]]:
    """Every value in ``config`` that names a file or folder, with where it is.

    The places the loader reads a path from: every ``local_dir`` resource, the
    ``local_dir`` skills and instructions of every agent, the code of ``tools`` and
    ``hooks``, a tool bundle's browser, an OAuth provider's token store. A browser ``binary`` without a ``/`` is
    a program looked up on ``PATH``, not a path.
    """
    found: list[tuple[str, Any]] = []
    for resource in config.get("resources") or []:
        if isinstance(resource, dict) and resource.get("type") == "local_dir":
            paths = resource.get("paths") or []
            for path in [paths] if isinstance(paths, str) else paths:
                # The folder domyn runs in, `.` (a workspace naming none always meant it)
                # and a project's `.domyn` and `.agents` (autodiscovery): not places the
                # configuration carries with it.
                if _working_folder(path):
                    continue
                found.append((f"resource {resource.get('id')!r}", path))
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


def _working_folder(path: Any) -> bool:
    """Whether a path names the folder domyn runs in, or the project folders read in it
    (``.domyn``, ``.agents``): relative to wherever a session starts, so nothing an
    export carries, nor a place outside a project."""
    written = str(path).rstrip("/")
    return written in (".", "") or Path(written) in LOCAL_INSTRUCTION_DIRS


def _is_autodiscovery(config: dict[str, Any], holder: dict[str, Any]) -> bool:
    """Whether ``holder`` is the config's autodiscovery resource, read in the folder a
    session starts in."""
    return holder is cli_config.declared_resources(config).get(
        cli_config.CLI_AUTODISCOVERY_RESOURCE
    )


def _inside_project(value: Any) -> bool:
    """Whether a path as the config writes it stays inside the project:
    ``${DOMYN_PROJECT_DIR}`` or below it, with no ``..`` climbing out and no other
    variable, which could be set to anything where it is run."""
    if not isinstance(value, str):
        return False
    if value == f"${{{PROJECT_DIR_VAR}}}":
        return True
    if not value.startswith(f"${{{PROJECT_DIR_VAR}}}/"):
        return False
    rest = value.removeprefix(f"${{{PROJECT_DIR_VAR}}}/")
    if "$" in rest:
        return False
    normal = posixpath.normpath(rest)
    return normal != ".." and not normal.startswith("../")


def _in_project(relative: str) -> str:
    """A path inside the export, as its config names it."""
    return f"${{{PROJECT_DIR_VAR}}}/{relative}"


@dataclass(frozen=True)
class PrivatePlaces:
    """Where a domyn home's session state can be: the folders `is_private` looks
    for a `StateFolders` folder in. Resolved once per export or import
    (`PrivatePlaces.now`), since `is_private` is asked about every file they copy."""

    homes: frozenset[Path]
    """The active home (`home_dir`) and ``~/.domyn``, resolved."""
    projects: Path | None
    """``~/.domyn/projects``, resolved: each project in it is a home too."""

    @classmethod
    def now(cls) -> PrivatePlaces:
        try:
            homes = frozenset(
                place.resolve() for place in (cli_config.home_dir(), cli_config.GLOBAL_DIR)
            )
            return cls(homes=homes, projects=cli_config.projects_dir().resolve())
        except (OSError, RuntimeError):
            return cls(homes=frozenset(), projects=None)


_STATE_FOLDER_NAMES = frozenset(StateFolders)


def is_private(path: Path, places: PrivatePlaces) -> bool:
    """Whether ``path`` is one of the files a domyn home keeps for its owner alone:
    the credentials, the approvals, the state file (what the CLI remembers
    between sessions), the conversation history (with SQLite's side
    files), matched by name wherever it is, so no layout of skills can carry one out;
    and anything in the session's working state (`StateFolders`: the
    conversations' environments and output, the commands' logs, the browser's files)
    of one of ``places``. Those folders are matched by place, not by name: a skill
    may well have a ``browser/`` folder of its own.
    """
    name = path.name
    if name in (
        PrivateFiles.CREDENTIALS,
        PrivateFiles.APPROVALS,
        PrivateFiles.STATE,
    ) or name.startswith(PrivateFiles.CONVERSATIONS):
        return True
    try:
        resolved = path.absolute().resolve()
    except (OSError, RuntimeError):
        return False
    return any(
        folder.name in _STATE_FOLDER_NAMES
        and (folder.parent in places.homes or folder.parent.parent == places.projects)
        for folder in (resolved, *resolved.parents)
    )


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
    """What a root folder is called inside ``resources/``.

    A folder called ``skills`` is named after the folder holding it (``.domyn/skills``
    is ``domyn``): a name saying where the skills came from tells two of them apart. A
    leading dot is dropped, so a copy never lands in a folder the search tools skip.
    """
    name = anchor.parent.name if anchor.name == "skills" else anchor.name
    return name.lstrip(".") or "skills"


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
) -> ExportReport:
    """Write the configuration ``raw`` and every file it reads into a new folder ``dest``.

    ``cwd`` is what a relative path in ``raw`` is read against (the working directory,
    as the loader reads it).

    ``dest`` must not exist, or be an empty folder. It is built beside itself and
    moved into place, so a failure leaves nothing behind. ``raw`` is not modified.
    """
    dest = dest.expanduser().absolute()
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ExportError(
            f"{dest} already exists and is not an empty folder. Choose a new folder name."
        )

    exporter = _Exporter(raw, cwd=(cwd or Path.cwd()).absolute())
    exporter.plan()

    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dest.name}.", dir=dest.parent))
    try:
        chosen = set(exporter.copies.values())
        for relative, source in exporter.copies.items():
            _copy_into(staging, relative, source, chosen, exporter.places)
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
    return path.removeprefix(f"${{{PROJECT_DIR_VAR}}}/")


def _with_tilde(path: Path) -> str:
    try:
        return f"~/{path.relative_to(Path.home()).as_posix()}"
    except ValueError:
        return str(path)
