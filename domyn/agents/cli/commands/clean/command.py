"""``domyn clean`` — delete old conversations and the files created for them."""

import typer
from rich.console import Console

from domyn.agents.cli.project import ProjectOption, select_project
from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli.clean import apply_clean, human, plan_clean
from domyn.agents.interfaces.cli.config import home_dir
from domyn.agents.interfaces.cli.prompt import interactive_terminal

console = Console()


def clean_command(
    older_than: int | None = typer.Option(
        None,
        "--older-than",
        min=0,
        help="Remove conversations not touched for this many days",
    ),
    all_: bool = typer.Option(False, "--all", help="Remove every conversation"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Say what would go, remove nothing"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask first"),
    project: ProjectOption = None,
) -> None:
    """Delete past conversations and the files the agent created for them.

    Works on the history of the selected project, else the global one in ~/.domyn
    (`--project <name>` for a project's). Use `--older-than <days>` to keep recent ones,
    or `--all` to start empty. A removed conversation can no longer be resumed.
    """
    theme = Theme.build(allow_animation=False)
    select_project(project, console, theme)
    if (older_than is None) == (not all_):
        console.print(
            "Say what to remove: --older-than <days>, or --all. "
            "Add --project <name> to clean a project's history instead of the global one.",
            style=theme.style("error"),
            highlight=False,
        )
        raise typer.Exit(1)

    home = home_dir()
    plan = plan_clean(home, older_than_days=None if all_ else older_than)
    if plan.nothing:
        console.print(f"Nothing to remove in {home}.", highlight=False)
        return
    console.print(
        f"In {home}: {len(plan.ids)} conversation(s) to remove"
        f" ({plan.kept} kept), {len(plan.folders)} environment folder(s)"
        f" ({human(plan.folder_bytes)}); history file is {human(plan.store_bytes)}.",
        highlight=False,
    )
    if dry_run:
        return
    if not yes:
        if not interactive_terminal():
            console.print("Not asking without a terminal: pass --yes.", style=theme.style("error"))
            raise typer.Exit(1)
        if not typer.confirm("Remove them?", default=False):
            console.print("Nothing removed.")
            return
    freed = apply_clean(plan)
    console.print(f"Removed. {human(freed)} given back.", highlight=False)
