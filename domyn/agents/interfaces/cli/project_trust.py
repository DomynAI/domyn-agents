"""What a project folder would have this machine read or run, checked every time it is
used.

A project is a folder, and a folder can come from anyone: an export someone sent, run
in place with ``--project <its path>``, or moved under ``~/.domyn/projects``. Its
``config.yaml`` decides what the session reads (skills, instruction files — sent to the
model every turn) and runs (the code of ``tools`` and ``hooks``, executed when the agent
is built, before any approval prompt). Nothing about a folder is remembered: a record of
which folders were accepted would be one more thing to keep honest, and a folder cannot
vouch for itself. So every command selecting a project **by path** looks again
(`select_project`); one selected by name, under ``~/.domyn/projects``, is trusted, those
being the folders `domyn setup` made or imported:

- **Files outside the folder** (`project_problems`): every path the config names
  (`config_paths`), where it really leads — a link, ``..`` or an absolute path into the
  project are judged by where they land — and every skill's ``external_files`` entry
  leading out of it (`external_files_outside`).
- **Code** (`code_files`): the file of every ``tools``/``hooks`` entry, shown with what
  it imports before ``domyn run`` runs it.

Both are shown and asked about on a terminal, every time, and refused without one. A
project ``domyn setup`` made, used by its path, can name files outside itself (a skills
folder or an instruction file elsewhere is listed where it is), so it is asked about
too: from a path alone, a project you made and one you were sent look the same.
"""

from pathlib import Path
from typing import Any

from domyn.agents.interfaces.cli.export import config_paths, external_files_outside
from domyn.agents.interfaces.cli.project_import import skill_files_in
from domyn.agents.loader import interpolate_env_vars


def _resolved(written: Any) -> Path | None:
    """A path as the loader reads it, links followed; ``None`` when it uses a variable
    that is not set (the loader refuses that config before reading anything)."""
    try:
        text = interpolate_env_vars(str(written), "the configuration")
    except ValueError:
        return None
    return Path(text).expanduser().absolute().resolve()


def project_problems(folder: Path, raw: dict[str, Any]) -> list[str]:
    """Every file or folder outside ``folder`` that ``raw`` would have read, run or
    worked in, each as one line; empty when it names none."""
    root = folder.resolve()
    problems: list[str] = []
    for where, written in config_paths(raw):
        path = _resolved(written)
        if path is None or not path.is_relative_to(root):
            problems.append(f"{where}: {written}")
    return problems + external_files_outside(skill_files_in(folder))


def code_files(raw: dict[str, Any]) -> list[Path]:
    """The file every ``tools``/``hooks`` entry of ``raw`` runs, as the loader finds it
    (``${VAR}`` filled in, relative to the working directory), each once."""
    found: list[Path] = []
    for section in ("tools", "hooks"):
        for entry in raw.get(section) or []:
            if not isinstance(entry, dict) or not entry.get("path"):
                continue
            path = _resolved(entry["path"]) or Path(str(entry["path"]))
            if path not in found:
                found.append(path)
    return found
