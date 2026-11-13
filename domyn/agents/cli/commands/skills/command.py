"""``domyn skills`` and ``domyn skill`` — inspect and unlist configured skills from the shell."""

import asyncio
import difflib
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.text import Text

from domyn.agents.cli.commands.run.command import build_skills_runtime
from domyn.agents.cli.project import ProjectOption, select_project
from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli.config import (
    collect_skill_workspace_roots,
    config_path,
    drop_empty_resources,
    is_skill_autodiscovery,
    read_raw_config,
    skills_install_dir,
    source_path_lists,
    source_paths,
)
from domyn.agents.interfaces.cli.skills_view import find_skill, render_skills
from domyn.agents.loader import interpolate_env_vars
from domyn.agents.skills import SkillIndex, SkillNotFoundError, SkillQuery
from domyn.agents.skills.layout import listed_path_is_linked

console = Console()


def _configured_skill_dirs() -> list[Path]:
    """The skills the saved configuration lists: what `domyn setup` installed, and
    ``.domyn/skills`` and ``.agents/skills`` when it turned on loading the project's.

    A command that does not build the agent from the configuration has to read them
    from it, or it would show only the folder it was given.
    """
    raw = read_raw_config()
    return collect_skill_workspace_roots(raw) if raw else []


def skills_command(project: ProjectOption = None) -> None:
    """List the skills that would be loaded, and why any are unavailable."""
    theme = Theme.build(allow_animation=False)
    select_project(project, console, theme)
    runtime = build_skills_runtime(_configured_skill_dirs(), expand_all=False)
    asyncio.run(render_skills(console, theme, runtime, None))


def skill_command(
    action: str = typer.Argument(..., help="show | remove"),
    name: str = typer.Argument(
        ..., help="Skill name; for show, <name>#<section> prints one ## section"
    ),
    project: ProjectOption = None,
) -> None:
    """Show a configured skill, or remove it from the config."""
    theme = Theme.build(allow_animation=False)
    select_project(project, console, theme)

    if action not in ("show", "remove"):
        console.print(
            f"Unknown action '{action}'. Use show or remove.",
            style=theme.style("error"),
            highlight=False,
        )
        raise typer.Exit(1)

    path = config_path()
    raw = read_raw_config(path)

    if action == "show":
        _show(theme, name, path, raw)
    else:
        _remove(theme, name, path, raw)


def _show(theme: Theme, name: str, path: Path, raw: dict[str, Any] | None) -> None:
    runtime = build_skills_runtime(collect_skill_workspace_roots(raw or {}), expand_all=False)
    if runtime is None:
        console.print(
            f"No skills configured in {path}.",
            style=theme.style("warning"),
            highlight=False,
            soft_wrap=True,
        )
        raise typer.Exit(1)
    try:
        asyncio.run(render_skills(console, theme, runtime, name))
    except SkillNotFoundError as exc:
        console.print(Text(str(exc), style=theme.style("warning")), highlight=False, soft_wrap=True)
        raise typer.Exit(1) from None


def _remove(theme: Theme, name: str, path: Path, raw: dict[str, Any] | None) -> None:
    """Unlist a skill from the config. Its files are never deleted.

    Every message is soft-wrapped: it names paths, and one cut across lines by the
    console cannot be copied whole."""
    index = _index(raw)
    descriptor = find_skill(index, name) if index is not None else None
    if raw is None or index is None:
        console.print(
            f"No skills configured in {path}.",
            style=theme.style("warning"),
            highlight=False,
            soft_wrap=True,
        )
        raise typer.Exit(1)
    if descriptor is None or descriptor.path is None:
        known = [s.name for s in index.skills]
        close = difflib.get_close_matches(name, known, n=3)
        message = f"Skill '{name}' is not among the skills {path} loads."
        message += f" Did you mean {', '.join(close)}?" if close else " `domyn skills` lists them."
        console.print(message, style=theme.style("warning"), highlight=False, soft_wrap=True)
        raise typer.Exit(1)

    skill_file = descriptor.path.expanduser().resolve()
    # A listed path names the skill when it is its loose `.md` or its folder.
    own = {skill_file, skill_file.parent} if skill_file.name == "SKILL.md" else {skill_file}
    explicit: list[str] = []
    covering: list[str] = []

    def resolved(entry: Any) -> Path | None:
        """A path as the loader reads it: `${VAR}` filled in, relative to the current
        folder, and no symlink followed (the loader loads nothing through one)."""
        try:
            listed = Path(interpolate_env_vars(str(entry), "the configuration")).expanduser()
        except ValueError:
            return None
        return None if listed_path_is_linked(listed) else listed.resolve()

    for agent_dict in raw.get("agents") or []:
        skills = agent_dict.get("skills") if isinstance(agent_dict, dict) else None
        # Unlisted from the resources the agent's skills source reads each path of
        # whole (what `domyn setup` writes for the user's own).
        for paths in (holder["paths"] for holder in source_path_lists(raw, skills, "skills")):
            # Over a copy, since a matching entry is removed from `paths` as it goes.
            for entry in list(paths):
                if resolved(entry) in own:
                    explicit.append(entry)
                    paths.remove(entry)
        # Every folder the source reads, subpaths included (`.domyn/skills` of the
        # autodiscovery resource): one holding the skill is named, never edited.
        for entry in source_paths(raw, skills, "skills") or []:
            where = resolved(entry)
            if where is not None and where not in own and where in skill_file.parents:
                covering.append(entry)

    if explicit:
        drop_empty_resources(raw)
        path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        listed_entries = ", ".join(dict.fromkeys(explicit))
        console.print(
            f"Removed '{descriptor.name}' ({listed_entries}) from {path}",
            style=theme.style("accent"),
            highlight=False,
            soft_wrap=True,
        )
        install = skills_install_dir().expanduser().resolve()
        if install in skill_file.parents:
            console.print(
                f"  Its files stay in {install} until the next `domyn setup`.",
                style=theme.style("meta"),
                highlight=False,
                soft_wrap=True,
            )

    if covering:
        # Listed only through a folder: unlisting that would unload its other skills too.
        autodiscovered = next((e for e in covering if is_skill_autodiscovery(e)), None)
        install = skills_install_dir().expanduser().resolve()
        if install in skill_file.parents:
            # The built-ins are read through the folder setup installs them into.
            message = (
                f"'{descriptor.name}' is a built-in skill, loaded from {install} with every "
                "other built-in setup installed. To stop loading it, run `domyn setup` and "
                "untick it."
            )
        elif autodiscovered is not None:
            message = (
                f"'{descriptor.name}' is loaded by ./{Path(str(autodiscovered)).as_posix()} "
                f"autodiscovery, not listed on its own in {path}. To stop loading it, turn "
                f"autodiscovery off with `domyn setup`, or delete or move {skill_file}."
            )
        else:
            message = (
                f"'{descriptor.name}' is loaded through the folder "
                f"{', '.join(dict.fromkeys(covering))} listed in {path}, not on its own. "
                f"Removing that entry would unload every skill in it: edit {path} by hand, "
                f"or delete or move {skill_file}."
            )
        console.print(message, style=theme.style("warning"), highlight=False, soft_wrap=True)
        if not explicit:
            raise typer.Exit(1)
        return

    # Another skill of the same name, hidden by this one until now, loads in its place.
    after = _index(raw)
    shadowed = after.by_name(descriptor.name) if after is not None else None
    if shadowed is not None:
        console.print(
            f"  A skill named '{shadowed.name}' still loads, from {shadowed.path}.",
            style=theme.style("warning"),
            highlight=False,
            soft_wrap=True,
        )


def _index(raw: dict[str, Any] | None) -> SkillIndex | None:
    """Every skill the config loads, or ``None`` when it lists none."""
    runtime = build_skills_runtime(collect_skill_workspace_roots(raw or {}), expand_all=False)
    return asyncio.run(runtime.index(SkillQuery())) if runtime is not None else None
