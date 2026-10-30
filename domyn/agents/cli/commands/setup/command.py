"""``domyn setup`` — configure the provider, model and default agent."""

import shutil
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console

from domyn.agents.cli.project import ProjectOption, select_project
from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli.config import (
    ProjectError,
    is_project_path,
    project_dir,
    read_raw_config,
)
from domyn.agents.interfaces.cli.project_import import ProjectImportError, import_project
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
    if imported.code_left_out:
        console.print(
            f"  Not imported: {', '.join(imported.code_left_out)}, which run code of their "
            "own. Setup does not keep tools or hooks; add them to the project's "
            "config.yaml by hand if you need them.\n",
            style=theme.style("warning"),
            highlight=False,
            markup=False,
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
