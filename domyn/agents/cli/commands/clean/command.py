"""``domyn clean`` — delete old conversations and memories, and the files created for them."""

import typer
from rich.console import Console

from domyn.agents.cli.project import ProjectOption, select_project
from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli.clean import (
    apply_clean,
    apply_memory_clean,
    human,
    plan_clean,
    plan_memory_clean,
)
from domyn.agents.interfaces.cli.config import home_dir
from domyn.agents.interfaces.cli.prompt import interactive_terminal

console = Console()


def clean_command(
    older_than: int | None = typer.Option(
        None,
        "--older-than",
        min=0,
        help="Remove what was not touched (conversations) or written (memories) for this many days",
    ),
    all_: bool = typer.Option(False, "--all", help="Remove everything of the chosen kind"),
    conversations: bool = typer.Option(
        False,
        "--conversations",
        help="Clean conversations and their files (the default when --memory is not given)",
    ),
    memory: bool = typer.Option(False, "--memory", help="Clean memories (memory.md entries)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Say what would go, remove nothing"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask first"),
    project: ProjectOption = None,
) -> None:
    """Delete past conversations and/or memories.

    Works on the history of the selected project, else the global one in ~/.domyn
    (`--project <name>` for a project's). Say how much with `--older-than <days>` (keep
    recent ones) or `--all` (start empty), and of what with `--conversations` (the default:
    each one with the files the agent created for it) and/or `--memory` (entries of that
    history's memory.md, by the date they were written). A removed conversation can no
    longer be resumed.
    """
    theme = Theme.build(allow_animation=False)
    select_project(project, console, theme)
    if (older_than is None) == (not all_):
        console.print(
            "Say how much to remove: --older-than <days>, or --all. Choose what with "
            "--conversations (default) and/or --memory. "
            "Add --project <name> to clean a project's history instead of the global one.",
            style=theme.style("error"),
            highlight=False,
        )
        raise typer.Exit(1)

    days = None if all_ else older_than
    do_conv = conversations or not memory
    home = home_dir()
    plan = plan_clean(home, older_than_days=days) if do_conv else None
    memory_plan = plan_memory_clean(home, older_than_days=days) if memory else None
    if (plan is None or plan.nothing) and (memory_plan is None or memory_plan.nothing):
        console.print(f"Nothing to remove in {home}.", highlight=False)
        return
    if plan is not None and not plan.nothing:
        console.print(
            f"In {home}: {len(plan.ids)} conversation(s) to remove"
            f" ({plan.kept} kept), {len(plan.folders)} environment folder(s)"
            f" ({human(plan.folder_bytes)}); history file is {human(plan.store_bytes)}.",
            highlight=False,
        )
    if memory_plan is not None and not memory_plan.nothing:
        console.print(
            f"In {memory_plan.path}: {memory_plan.count} memory entr"
            f"{'y' if memory_plan.count == 1 else 'ies'} to remove ({memory_plan.kept} kept).",
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
    freed = apply_clean(plan) if plan is not None else 0
    removed = apply_memory_clean(memory_plan) if memory_plan is not None else 0
    memory_note = f" {removed} memor{'y' if removed == 1 else 'ies'} removed." if memory else ""
    console.print(f"Removed. {human(freed)} given back.{memory_note}", highlight=False)
