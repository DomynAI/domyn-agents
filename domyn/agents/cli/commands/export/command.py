"""``domyn export`` — the CLI configuration as a folder someone else can run."""

from pathlib import Path

import typer
from rich.console import Console

from domyn.agents.cli.project import ProjectOption, select_project
from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli.config import config_path, read_raw_config
from domyn.agents.interfaces.cli.export import (
    ExportError,
    export_configuration,
    print_export_report,
)

console = Console()


def export_command(
    name: Path = typer.Argument(
        ..., help="The folder to create. It must not exist yet, or be empty"
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="The configuration file to export. Default: the one `domyn run` uses",
    ),
    project: ProjectOption = None,
) -> None:
    """Save the configuration, with the skills and instruction files it uses, into a new folder.

    The folder is a project that works on any machine: run it where it is with
    `domyn run --project <folder>`, move it to ~/.domyn/projects/<name> and run
    `domyn run --project <name>`, or create a new project from it, reviewing each
    setting, with `domyn setup --import <folder>`. API keys, approvals and conversation
    history are never included.
    """
    theme = Theme.build(allow_animation=False)
    select_project(project, console, theme)
    source = config or config_path()
    raw = read_raw_config(source)
    if raw is None:
        console.print(
            f"No configuration found at {source}. Run `domyn setup` to create one first.",
            style=theme.style("error"),
        )
        raise typer.Exit(1)

    try:
        report = export_configuration(raw, name)
    except (ExportError, OSError) as exc:
        console.print(
            f"Export failed, nothing was written: {exc}",
            style=theme.style("error"),
            highlight=False,
        )
        raise typer.Exit(1) from exc

    print_export_report(console, theme, source, report)
