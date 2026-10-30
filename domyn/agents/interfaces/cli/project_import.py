"""``domyn setup [--project <p>] --import <dir>``: a new project started from an export,
``~/.domyn/projects/<the name of dir>`` unless ``--project`` names one.

The export (``domyn export``) is a starting point, not a copy to keep in step with:
its files are copied into the new project, and its configuration becomes the default
of every question the wizard asks, which then writes the project's own
``config.yaml`` as any setup does. The wizard offers the project's own skills in
place of the built-in catalog (`setup._choose_project_skills`), and asks whether to
keep its instructions (`setup._choose_imported_instructions`).

**The skills keep their place**, every one as exported, whether it began as a
built-in or not: the export's ``builtin-skills/`` becomes the project's, where its
setup installs, and ``skills/`` stays ``skills/``, so the config's
``${DOMYN_PROJECT_DIR}/…`` paths and a skill's ``../`` references stay true. A later
setup of the project offers the built-ins in its catalog, and reinstalls the ones
ticked from the package.

**The version is compared, never enforced.** The export's ``.info`` records the
version of domyn that made it, among the rest (`ProjectImport.info`); another
version, or none recorded, is a warning before the wizard starts
(`ProjectImport.version_warning`), and the import goes on. The file describes the
export, so it is not copied into the project.

**Nothing outside the project is named.** An export can be edited after it is made, so
a config naming any path that is not ``${DOMYN_PROJECT_DIR}/…`` (`paths_outside_project`:
skills, instructions, tool and hook code, a workspace, a browser, a token store), or a
skill whose ``external_files`` lead out of it (`external_files_outside`), is
refused before anything is copied: it would have this machine read, run or write
what the export's author chose.

**Code is not imported.** The export's ``tools`` and ``hooks`` entries that run a file
of their own are not kept by the wizard, which writes the project's ``config.yaml``
from its own questions, so their ``code/`` is not copied either, and the import names
them (`ProjectImport.code_left_out`). A project that needs them gets them by a hand
edit; a project imported is used by name, and trusted as it is (`select_project`).

Everything else at the export's top — ``instructions/``, and any file a skill shares
from there — is copied as it is, so the config's paths into it stay true. Hidden entries,
links and the files a domyn home keeps for its owner alone (`is_private`) are not.
"""

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from domyn.agents.interfaces.cli.config import (
    CONFIG_FILENAME,
    read_raw_config,
)
from domyn.agents.interfaces.cli.export import (
    CODE_DIR,
    INFO_FILENAME,
    INFO_VERSION_KEY,
    external_files_outside,
    installed_version,
    is_private,
    paths_outside_project,
)


class ProjectImportError(Exception):
    """The export cannot be imported; nothing was written."""


@dataclass
class ProjectImport:
    source: Path
    """The export's folder, resolved."""
    config: dict[str, Any]
    """Its configuration, as exported: its paths name the project's files with
    ``${DOMYN_PROJECT_DIR}``, which the copy keeps true."""
    copied: list[str] = field(default_factory=list)
    """The top-level entries copied, by name, the same in the project as in the export."""
    code_left_out: list[str] = field(default_factory=list)
    """The export's ``tools`` and ``hooks`` entries that run code of their own, as
    ``tool <name>`` / ``hook <name>``: not imported (`_copy_plan`)."""
    info: dict[str, str] = field(default_factory=dict)
    """What the export's ``.info`` records (`export.export_info`), every value
    as text; empty when it has none (an export older than the file, or edited)."""

    @property
    def exported_with(self) -> str | None:
        """The version of domyn that made the export, ``None`` when not recorded."""
        return self.info.get(INFO_VERSION_KEY) or None

    def version_warning(self) -> str | None:
        """What to tell the user when the export was made by another version of
        domyn than the one importing it, or does not say which; else ``None``."""
        running = installed_version()
        if self.exported_with is None:
            return (
                f"{self.source} does not say which version of domyn exported it "
                f"(no {INFO_VERSION_KEY} in {INFO_FILENAME}); this is {running}. Its "
                "configuration and built-in skills may not match what this version expects."
            )
        if self.exported_with != running:
            when = f" on {self.info['exported_at']}" if self.info.get("exported_at") else ""
            return (
                f"{self.source} was exported with domyn {self.exported_with}{when}; "
                f"this is {running}. Its configuration and built-in skills are as that "
                "version wrote them, and may not match what this one expects."
            )
        return None


def import_project(source: Path, project: Path) -> ProjectImport:
    """Copy the export at ``source`` into the new project folder ``project``, and
    return its configuration as the wizard's starting point.

    ``project`` must not exist, or be an empty folder: an import starts a project, it
    does not merge into one. It is built beside itself and moved into place, so a
    failure leaves nothing behind.
    """
    source = source.expanduser().resolve()
    project = project.expanduser().absolute()
    raw = _read_export(source)
    if outside := paths_outside_project(raw) + external_files_outside(skill_files_in(source)):
        # An export can be edited after it is made: a path out of the project would
        # have this machine read, run or write what the export's author chose.
        raise ProjectImportError(
            f"{source} names paths outside itself, which would be read on this machine:\n"
            + "".join(f"  - {line}\n" for line in outside)
            + "Nothing was imported. An export made by `domyn export` names none."
        )
    if project.resolve() == source:
        raise ProjectImportError(
            f"{source} is already a project, the one being imported. To use it where it "
            f"is, run `domyn run --project {source}`; to change its settings, run "
            f"`domyn setup --project {source}`."
        )
    if project.exists() and (not project.is_dir() or any(project.iterdir())):
        raise ProjectImportError(
            f"{project} already exists and is not an empty folder. Choose another name "
            "with --project <name>, or remove that folder first."
        )

    plan = _copy_plan(source)

    project.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{project.name}.", dir=project.parent))
    try:
        for origin in plan:
            target = staging / origin.name
            if origin.is_dir():
                shutil.copytree(origin, target, ignore=_ignore)
            else:
                shutil.copy2(origin, target)
        if project.exists():
            project.rmdir()
        staging.rename(project)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return ProjectImport(
        source=source,
        config=raw,
        info=_read_info(source),
        copied=[origin.name for origin in plan],
        code_left_out=[
            f"{section[:-1]} {entry.get('name')}"
            for section in ("tools", "hooks")
            for entry in raw.get(section) or []
            if isinstance(entry, dict) and entry.get("path")
        ],
    )


def _read_export(source: Path) -> dict[str, Any]:
    if not source.is_dir():
        raise ProjectImportError(
            f"{source} is not a folder. Give the folder that `domyn export` created."
        )
    raw = read_raw_config(source / CONFIG_FILENAME)
    if raw is None:
        raise ProjectImportError(
            f"{source} has no readable {CONFIG_FILENAME}. Is it a folder created by `domyn export`?"
        )
    return raw


def _read_info(source: Path) -> dict[str, str]:
    """The export's ``.info``, every plain value as text; empty when the file is
    missing or unreadable. Anything but a mapping of plain values is ignored, as the
    file can be edited like the rest of the export."""
    try:
        raw = yaml.safe_load((source / INFO_FILENAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in raw.items()
        if isinstance(value, (str, int, float)) and not isinstance(value, bool) and str(value)
    }


def _copy_plan(source: Path) -> list[Path]:
    """The export's top-level entries to copy, each to the same name in the project."""
    plan: list[Path] = []
    for child in sorted(source.iterdir()):
        # `code/` is what the export's tools and hooks run, and setup writes neither
        # back: copied, it would be code nothing names, waiting for a hand edit to run.
        # The manifest describes the export, not the project made from it.
        if child.name in (CONFIG_FILENAME, CODE_DIR, INFO_FILENAME) or _left_behind(child):
            continue
        plan.append(child)
    return plan


def skill_files_in(source: Path) -> list[tuple[PurePosixPath, Path]]:
    """Every ``.md`` in the folder ``source`` but hidden ones, links and what is below
    them, with its place in it, for `external_files_outside`. In an export, a skill's
    place is its place in the project imported from it. A file that is not a skill
    declares no ``external_files``."""
    return [
        (PurePosixPath(path.relative_to(source).as_posix()), path)
        for path in sorted(source.rglob("*.md"))
        if path.is_file()
        and not any(
            _left_behind(source.joinpath(*path.relative_to(source).parts[: i + 1]))
            for i in range(len(path.relative_to(source).parts))
        )
    ]


def _left_behind(path: Path) -> bool:
    return path.name.startswith(".") or path.is_symlink() or is_private(path)


def _ignore(directory: str, names: list[str]) -> set[str]:
    """What `shutil.copytree` leaves out below a copied folder: what `_left_behind`
    does at the top, and bytecode."""
    return {
        name
        for name in names
        if name == "__pycache__" or name.endswith(".pyc") or _left_behind(Path(directory) / name)
    }
