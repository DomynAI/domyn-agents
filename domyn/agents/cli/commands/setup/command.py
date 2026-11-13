"""``domyn setup`` — configure the provider, model and default agent."""

import shutil
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console

from domyn.agents.cli.project import ProjectOption, ask_yes, select_project
from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli.config import (
    ProjectError,
    is_project_path,
    project_dir,
    read_raw_config,
)
from domyn.agents.interfaces.cli.project_import import (
    ProjectImport,
    ProjectImportError,
    import_project,
)
from domyn.agents.interfaces.cli.project_trust import code_files
from domyn.agents.interfaces.cli.prompt import interactive_terminal
from domyn.agents.interfaces.cli.setup import render_config, run_setup

console = Console()

# Annotated, so a plain call of the command gets a real ``None`` (see `ProjectOption`).
ImportOption = Annotated[
    Path | None,
    typer.Option(
        "--import",
        help=(
            "Create a new project from a folder made by `domyn export`, named after "
            "that folder unless --project gives a name. Setup then shows each exported "
            "setting and asks whether to keep it"
        ),
    ),
]


def setup_command(
    show: bool = typer.Option(False, "--show", help="Print the active configuration and exit"),
    project: ProjectOption = None,
    import_from: ImportOption = None,
) -> None:
    """Configure domyn: the model provider, API key, model, tools, skills and instructions.

    Run it again at any time to change the configuration. With --project <name>, the
    configuration is saved in its own folder, ~/.domyn/projects/<name>, created if
    needed. With --import <folder>, a new project is created from a folder made by
    `domyn export`.
    """
    theme = Theme.build(allow_animation=False)
    if import_from is not None:
        if show:
            _fail(
                "--import cannot be combined with --show: --import creates a new project, "
                "--show only prints an existing configuration.",
                theme,
            )
        _import_and_set_up(import_from, project, theme)
        return

    # A project shown has to exist; one being set up is created by saving it.
    select_project(project, console, theme, must_exist=show)

    existing = read_raw_config()

    if show:
        render_config(console, theme, existing)
        return

    if run_setup(console, theme, existing) is None:
        raise typer.Exit(1)


def _fail(message: str, theme: Theme) -> NoReturn:
    console.print(message, style=theme.style("error"), highlight=False, markup=False)
    raise typer.Exit(1)


def _accept_code(imported: ProjectImport, folder: Path, theme: Theme) -> bool:
    """List the files the imported project would run, warn that they run and must be
    checked, and ask: no unless the user says yes, and no without a terminal to ask on.

    Asked here, once, because it is never asked again: a project imported is used by
    name, and trusted as it is (`select_project`). The files listed are the project's
    copies, so what is checked is what runs. Their content is never shown: the user
    reads the files themselves."""
    files = "".join(f"    {path}\n" for path in code_files(imported.config))
    console.print(
        f"  The export runs Python code of its own ({', '.join(imported.code)}), from "
        f"these files, now in the new project:\n{files}"
        "  Imported, they run with everything you can do on this machine, every time the "
        "project's agent starts, without being asked about again. Check them carefully "
        "before you accept.",
        style=theme.style("warning"),
        highlight=False,
        markup=False,
    )
    if not interactive_terminal():
        return False
    return ask_yes(console, theme, "Import this code?")


def _import_and_set_up(source: Path, project: str | None, theme: Theme) -> None:
    """Copy the export into ``project`` (a name or a path, as ``--project`` takes it),
    by default the project under ``~/.domyn/projects`` named after the export's folder,
    then ask every question with its settings as the defaults. An aborted wizard
    removes the project again: it did not exist, or was empty, before the import."""
    if project is None:
        project = source.expanduser().resolve().name
        try:
            project_dir(project)
        except ProjectError as exc:
            _fail(
                f"Cannot import {source}: a new project is named after its folder. {exc} "
                "Rename the folder, or choose a name with --project <name>.",
                theme,
            )
    folder = select_project(project, console, theme, must_exist=False)
    try:
        imported = import_project(source, folder)
    except (ProjectImportError, OSError) as exc:
        _fail(f"Cannot import {source}: {exc}", theme)
    if warning := imported.version_warning():
        console.print(
            f"  {warning}\n",
            style=theme.style("caution"),
            highlight=False,
            markup=False,
        )
    if imported.code and not _accept_code(imported, folder, theme):
        shutil.rmtree(folder, ignore_errors=True)
        _fail(
            f"Import aborted: the code of {source} was not accepted, and nothing was "
            "imported. Check the files listed above, then import again from a terminal "
            "and accept them.",
            theme,
        )

    if run_setup(console, theme, imported.config, imported=imported) is None:
        shutil.rmtree(folder, ignore_errors=True)
        raise typer.Exit(1)

    console.print(
        f"  Created project {folder.name} in {folder}.",
        style=theme.style("success"),
        highlight=False,
        markup=False,
    )
    # A relative path names the project only from this folder: the hint spells it out.
    option = str(folder) if is_project_path(project) else project
    console.print(
        f"  Start it with:            domyn run --project {option}\n"
        f"  Change its settings with: domyn setup --project {option}\n",
        style=theme.style("meta"),
        highlight=False,
        markup=False,
        soft_wrap=True,
    )
