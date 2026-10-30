"""``--project``: the option every command reading the saved configuration takes."""

from pathlib import Path
from typing import Annotated, overload

import typer
from rich.console import Console

from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli.config import (
    ProjectError,
    is_project_path,
    load_credentials_into_environment,
    read_raw_config,
    use_project,
)
from domyn.agents.interfaces.cli.export import import_warnings
from domyn.agents.interfaces.cli.project_trust import code_files, project_problems
from domyn.agents.interfaces.cli.prompt import interactive_terminal
from domyn.agents.interfaces.cli.select import select

# Annotated, so the parameter's default is a real ``None``: the commands are also
# called as plain functions, where a ``typer.Option`` default would arrive as itself.
ProjectOption = Annotated[
    str | None,
    typer.Option(
        "--project",
        help=(
            "Use a project: a separate configuration with its own model, API keys, "
            "skills, instructions, approvals and conversation history, instead of the "
            "default one in ~/.domyn. A name means ~/.domyn/projects/<name>; a path "
            "(containing '/', or starting with '.' or '~') means that folder"
        ),
    ),
]


@overload
def select_project(
    name: str,
    console: Console,
    theme: Theme,
    *,
    must_exist: bool = True,
    runs_code: bool = False,
) -> Path: ...


@overload
def select_project(
    name: None,
    console: Console,
    theme: Theme,
    *,
    must_exist: bool = True,
    runs_code: bool = False,
) -> None: ...


@overload
def select_project(
    name: str | None,
    console: Console,
    theme: Theme,
    *,
    must_exist: bool = True,
    runs_code: bool = False,
) -> Path | None: ...


def select_project(
    name: str | None,
    console: Console,
    theme: Theme,
    *,
    must_exist: bool = True,
    runs_code: bool = False,
) -> Path | None:
    """Select the project for the rest of the process and return its folder (``None``
    for no project), or exit saying why it cannot be.

    ``must_exist=False`` is for `domyn setup`, the one command that creates a project.
    A project given by **path** is checked every time: one naming files outside its
    folder is shown and asked about (`project_trust`), and with ``runs_code``
    (`domyn run`) so is the code of its tools and hooks, before anything runs it;
    without a terminal to ask on, it is refused. A project given by **name** is not:
    the folders under ``~/.domyn/projects`` are the ones `domyn setup` made or
    imported, and are trusted.
    """
    try:
        folder = use_project(name, must_exist=must_exist)
    except ProjectError as exc:
        console.print(str(exc), style=theme.style("error"), highlight=False, markup=False)
        raise typer.Exit(1) from exc
    if folder is not None and name is not None and is_project_path(name):
        _review(folder, console, theme, runs_code=runs_code)
    if name is not None:
        # Startup put ~/.domyn/credentials in the environment, before the option was
        # parsed: the project's own keys go over it, before any config is read.
        load_credentials_into_environment()
    return folder


def _review(folder: Path, console: Console, theme: Theme, *, runs_code: bool) -> None:
    """Show what the project would read outside itself, and the code it would run, and
    ask; refused without a terminal to ask on. Every time: nothing is remembered."""
    raw = read_raw_config()
    if raw is None:
        return  # nothing to use yet: setup is creating it, or the loader will say why
    if problems := project_problems(folder, raw):
        _say(
            console,
            theme,
            "warning",
            f"The project {folder} names files outside its folder, which this machine "
            "would read or use:\n" + "".join(f"  - {line}\n" for line in problems),
        )
        _confirm(
            console,
            theme,
            "Use this project anyway?",
            f"Not used: check the paths above, then use the project from a terminal to "
            f"confirm them, or change its config.yaml so it names only files inside "
            f"{folder}.",
        )
    if runs_code and (code := code_files(raw)):
        _say(
            console,
            theme,
            "warning",
            f"The project {folder} runs code of its own when the agent starts, before any "
            "approval is asked: Python with everything you can do on this machine.",
        )
        for path in code:
            lines = (
                [f"    {w}" for w in import_warnings(path, path.name)]
                or ["    imports nothing beyond Python and domyn"]
                if path.is_file()
                else ["    does not exist"]
            )
            _say(console, theme, "meta", f"  {path}\n" + "\n".join(lines))
        _confirm(
            console,
            theme,
            "Run this code?",
            "Not run: read the files above, then run the project from a terminal to confirm them.",
        )


def _confirm(console: Console, theme: Theme, question: str, refusal: str) -> None:
    """Exit unless the user answers yes; without a terminal, exit saying ``refusal``."""
    if not interactive_terminal():
        _say(console, theme, "error", refusal)
        raise typer.Exit(1)
    if not _yes(console, theme, question):
        raise typer.Exit(1)


def _yes(console: Console, theme: Theme, question: str) -> bool:
    answer = select(
        console,
        theme,
        question,
        ["no", "yes"],
        current="no",
        aliases={"y": "yes", "n": "no"},
        max_tries=3,
    )
    return answer == "yes"


def _say(console: Console, theme: Theme, style: str, text: str) -> None:
    # Paths are the user's, and may hold `[`.
    console.print(text, style=theme.style(style), highlight=False, markup=False)
