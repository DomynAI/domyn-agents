import asyncio
import functools
import importlib.util
import inspect
import shlex
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.text import Text

from domyn.agents import __version__, instructions
from domyn.agents.agents import Agent
from domyn.agents.cli.cli_dependencies import install_cli_requirements_here
from domyn.agents.cli.project import ProjectOption, select_project
from domyn.agents.interfaces.cli import Theme, conversations, render_banner
from domyn.agents.interfaces.cli.config import (
    active_project,
    apply_context_window,
    config_path,
    current_settings_from,
    is_configuration_v1_usable,
    is_skill_autodiscovery,
    project_location,
    read_raw_config,
    resolve_instruction_dirs,
    skill_workspace_roots,
    unique_paths,
)
from domyn.agents.interfaces.cli.prompt import interactive_terminal
from domyn.agents.interfaces.cli.repl import Repl
from domyn.agents.interfaces.cli.setup import run_setup
from domyn.agents.loader import load_application_from_file
from domyn.agents.logger import get_logger
from domyn.agents.skills import (
    AllSkillsSelector,
    IndexSelector,
    LocalDirSkillSource,
    SkillIndex,
    SkillQuery,
    SkillSelector,
    SkillsRuntime,
)
from domyn.agents.skills.external_files import missing_external_files
from domyn.agents.skills.integration import attach_skills
from domyn.agents.skills.layout import listed_path_is_linked

# markup=False for the whole session: nothing here writes rich's square-bracket
# tags, and everything here prints text from somewhere else — tool arguments, a
# model's reasoning, a path, a command's own usage line. With markup on, rich
# reads `[revoke <n> | clear]` as a tag and silently deletes it, which is how
# /help came to hide the usage of the one command that has one.
console = Console(markup=False)


def load_module_from_path(file_path: str) -> Any:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix not in (".py",):
        raise ValueError(f"Expected a .py file, got: {file_path}")

    spec = importlib.util.spec_from_file_location("user_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["user_module"] = module
    spec.loader.exec_module(module)
    return module


async def load_agent_from_yaml(file_path: str) -> Agent:
    """Load root agent from a YAML config file."""
    from domyn.agents.loader import load_application_from_file

    return await load_application_from_file(file_path)


def collect_all_agents(module: Any) -> dict[str, Agent]:
    """Collect all Agent instances from the module and its imports.

    Returns a dictionary mapping agent variable names to Agent instances.
    """
    agents: dict[str, Agent] = {}
    visited_modules: set[Any] = set()

    def _collect_from_module(mod: Any, mod_name: str = "") -> None:
        if mod in visited_modules or mod is None:
            return
        visited_modules.add(mod)

        for attr_name in dir(mod):
            if attr_name.startswith("_"):
                continue
            try:
                attr = getattr(mod, attr_name)
                if isinstance(attr, Agent):
                    full_name = f"{mod_name}.{attr_name}" if mod_name else attr_name
                    agents[full_name.lower()] = attr
                    get_logger().debug(f"discovered agent: {full_name}")
            except (AttributeError, ImportError):
                continue

        for attr_name in dir(mod):
            if attr_name.startswith("_"):
                continue
            try:
                attr = getattr(mod, attr_name)
                is_imported_module = inspect.ismodule(attr) and hasattr(attr, "__file__")
                if (
                    is_imported_module
                    and attr.__file__
                    and not attr.__file__.startswith("/usr/")
                    and not attr.__file__.startswith("/opt/")
                    and "site-packages" not in attr.__file__
                ):
                    _collect_from_module(attr, attr_name)
            except (AttributeError, ImportError):
                continue

    _collect_from_module(module)
    return agents


def find_root_agents(agents: dict[str, Agent]) -> dict[str, Agent]:
    """Find agents that are not sub_agents of any other agent."""
    child_agents = {id(sub_agent) for a in agents.values() for sub_agent in a.sub_agents}
    return {name: a for name, a in agents.items() if id(a) not in child_agents}


def find_main_agent(all_agents: dict[str, Agent], agent_name: str | None = None) -> Agent:
    """Find the main agent using dependency tree analysis.

    Priority:
    1. Exact match by --agent-name
    2. Root agent (not a sub_agent of any other)
    3. Name-based fallback ("main", "service", "customer")
    4. First agent found
    """
    if not all_agents:
        raise ValueError("No Agent instances found in the module or its imports")

    get_logger().debug(f"discovered {len(all_agents)} agent(s): {list(all_agents.keys())}")

    if agent_name:
        agent_name = agent_name.lower()
        if agent := all_agents.get(agent_name):
            return agent
        matches = [a for a in all_agents if a.endswith(f".{agent_name}") or a == agent_name]
        if matches:
            return all_agents[matches[0]]
        raise ValueError(f"Agent '{agent_name}' not found. Available: {list(all_agents.keys())}")

    root_agents = find_root_agents(all_agents)

    if not root_agents:
        typer.echo("No root agents found. Falling back to name-based detection.", err=True)
        for name, agent in all_agents.items():
            if any(k in name.lower() for k in ("main", "service", "customer")):
                return agent
        return next(iter(all_agents.values()))

    if len(root_agents) > 1:
        typer.echo(
            f"Multiple root agents found: {list(root_agents.keys())}. Using: {next(iter(root_agents))}",
            err=True,
        )

    return next(iter(root_agents.values()))


def apply_session_instructions(agent: Any) -> None:
    """Give the root agent this session's standing instructions — the same rule
    the skills follow, so only the root is touched and every sub-agent keeps
    exactly what its own definition declares.

    A ``local_dir`` source the agent declares keeps its paths, through
    `resolve_instruction_dirs`; ``./.domyn`` and ``./.agents`` are among them when
    setup turned autodiscovery on, and read from the folder the session starts in.
    An agent that declares none gets none. Any other kind of source is its
    author's and is kept as it is.
    """
    declared = getattr(agent, "standing_instructions", None)
    if isinstance(declared, instructions.InstructionRuntime) and isinstance(
        declared.source, instructions.LocalDirInstructionsSource
    ):
        paths = resolve_instruction_dirs(declared.source.paths)
        # A copy, not an edit: the declared runtime may be shared with
        # another agent built from the same config.
        agent.standing_instructions = declared.model_copy(
            update={"source": instructions.LocalDirInstructionsSource(paths=paths)}
        )


def build_skills_runtime(
    directories: list[Path] | None,
    *,
    expand_all: bool,
    index_sections: bool = False,
) -> SkillsRuntime | None:
    """Build a skills runtime, or ``None`` when there is nothing to load.

    Returning ``None`` rather than an empty runtime keeps the promise that an
    agent without skills behaves exactly as before.
    """
    directories = unique_paths(directories or [])

    if not directories:
        return None

    selector: SkillSelector = AllSkillsSelector() if expand_all else IndexSelector()
    return SkillsRuntime(
        source=LocalDirSkillSource(paths=directories),
        selector=selector,
        index_sections=index_sections,
    )


async def start_session(
    agent: Agent,
    theme: Theme,
    *,
    initial_prompt: str | None,
    verbose: bool,
    show_status: bool,
    config: dict[str, Any] | None = None,
    config_file: Path | None = None,
    resume: str | None = None,
    resuming: bool = False,
    run_command: str = "domyn run",
    reload: Callable[[], Awaitable[tuple[Agent, dict[str, Any]]]] | None = None,
    voice: bool = False,
) -> int:
    """Print the status line, then hand over to the REPL.

    Skills and standing instructions are already on the agent (attached by the
    caller); they are read here only to say which are in play. The banner is drawn
    by the caller too, before anything else, so it appears even when the setup
    wizard has to run first — branding then belongs to the app, not to the happy
    path only.
    """
    # Read once, for the status line and for the warning below: each listed skill file
    # is parsed here once and not once per use.
    index = await _startup_index(agent)
    if show_status:
        details = [agent.name]
        skills = _skills_status(agent, index)
        if skills:
            details.append(skills)
        loaded_instructions = await instructions.load_declared(agent)
        if loaded_instructions:
            # Named, not counted. These files change how the agent behaves for the
            # whole session and may come from the folder it started in, so the one
            # moment to say which ones are in play is before the first message.
            details.append(", ".join(loaded_instructions.names))
        # Shift+Tab is said here because nothing else shows it while auto-approve
        # is off. Only where the editor runs: a pipe has no such key.
        keys = "  ·  shift+tab auto-approve" if interactive_terminal() else ""
        console.print(
            f"  {'  ·  '.join(details)}  ·  /help for commands{keys}  ·  Ctrl-C twice to quit\n",
            style=theme.style("meta"),
            highlight=False,
        )
        if _lacks_chosen_skills(config, config_file):
            console.print(
                "  No skills chosen: run `domyn setup` to pick the built-in ones.\n",
                style=theme.style("meta"),
                highlight=False,
            )

    # The CLI's own configuration: declared, else asked of the provider, else the
    # default. An explicit file is left to the runner, which asks per turn and
    # assumes nothing — as for any agent built outside the CLI.
    if config_file is not None and config_file.resolve() == config_path().resolve():
        await apply_context_window(getattr(agent, "llm_provider", None))

    # Outside the status block: a listed skill that is not there is a warning, not
    # decoration, so `--no-banner` and a one-shot prompt still say it.
    missing = _missing_skills_warning(agent, index)
    if missing:
        console.print(missing, style=theme.style("warning"), highlight=False, markup=False)

    resumed = await _resolve_resume(theme, resume or "") if resuming else None
    if resuming and resumed is None:
        # Asked to continue something and there was nothing to continue: starting
        # a fresh conversation instead would silently discard the request.
        return 1

    repl = Repl(
        agent,
        console,
        theme,
        config=config,
        config_file=config_file,
        verbose=verbose,
        resumed=resumed,
        run_command=run_command,
        reload=reload,
    )
    if resumed is not None:
        conversations.render_history(console, theme, resumed, verbose=verbose)
    if voice:
        await repl.toggle_voice()
    return await repl.run(initial_prompt)


def _lacks_chosen_skills(config: dict[str, Any] | None, config_file: Path | None) -> bool:
    """Whether the saved configuration was never given a choice of skills.

    Only for the saved configuration: an agent file, or one named on the command
    line, declares its own skills (or none) and is not setup's to ask about. A
    configuration written before setup offered the choice has no ``skills`` entry,
    and with the package fallback gone it loads none of the built-in ones.
    """
    if not config or config_file is None or config_file != config_path():
        return False
    return not current_settings_from(config)[1].skills


_MISSING_SHOWN = 5


def _missing_skills_warning(agent: Agent, index: SkillIndex | None = None) -> str | None:
    """The warning for what the root agent's skills list names but is not there, or ``None``.

    Three things, all a check of what the config names:

    * a listed **path** that goes through a symbolic link (`listed_path_is_linked`: the
      path itself, or for a relative one a folder on the way to it). The loader follows
      none, and would skip it without a word;
    * a listed **path** that does not exist. The loader skips it without a word, which
      would leave a stale relative entry — the folder was started from somewhere else, or
      the skill was deleted — looking like a skill that was never there. Not the
      ``.domyn/skills`` / ``.agents/skills`` autodiscovery entries, which are read from
      wherever domyn starts;
    * a listed **skill whose ``external_files``** are not there (``index`` is what the
      runtime loaded; without it this part is skipped).
    """
    runtime = getattr(agent, "skills", None)
    if not isinstance(runtime, SkillsRuntime) or not isinstance(
        runtime.source, LocalDirSkillSource
    ):
        return None

    lines: list[str] = []
    # Links first, so a broken one is named once, as a link: the loader follows none.
    linked = [path for path in runtime.source.paths if listed_path_is_linked(path.expanduser())]
    if linked:
        lines.append(
            "  Skills not loaded, the path is a symbolic link (links are not followed): "
            f"{_first_names(linked)}"
        )
    # The autodiscovery folders are read from wherever domyn starts, so their absence is normal.
    missing = [
        path
        for path in runtime.source.paths
        if path not in linked
        and not is_skill_autodiscovery(path)
        and not path.expanduser().exists()
    ]
    if missing:
        lines.append(f"  Skill not found: {_first_names(missing)}")
        relative = any(not path.expanduser().is_absolute() for path in missing)
        if relative:
            lines.append(
                f"  Relative paths are read from the folder domyn is started in ({Path.cwd()})."
            )
        lines.append(f"  Run `domyn setup`{' there' if relative else ''} to update the list.")

    lines.extend(_missing_external_files_lines(index))
    return "\n".join([*lines, ""]) if lines else None


def _first_names(paths: list[Path]) -> str:
    names = [str(path) for path in paths[:_MISSING_SHOWN]]
    if len(paths) > _MISSING_SHOWN:
        names.append(f"and {len(paths) - _MISSING_SHOWN} more")
    return ", ".join(names)


def _missing_external_files_lines(index: SkillIndex | None) -> list[str]:
    """One line per listed skill whose ``external_files:`` are not all there."""
    if index is None:
        return []
    broken = [
        (skill.name, problems)
        for skill in index.skills
        if skill.path is not None and skill.requires.external_files
        if (problems := missing_external_files(skill.path, skill.requires.external_files))
    ]
    lines = [
        f"  Skill '{name}' needs files that are missing: {', '.join(problems)}"
        for name, problems in broken[:_MISSING_SHOWN]
    ]
    if len(broken) > _MISSING_SHOWN:
        lines.append(f"  and {len(broken) - _MISSING_SHOWN} more skills with missing files")
    return lines


def _skills_status(agent: Agent, index: SkillIndex | None) -> str | None:
    """What the status line says about skills; ``None`` when there are none or they cannot be read.

    The count, and whether the index is over its budget: the model then sees the
    skills in a shorter form, and the start of a session is the one moment that is
    said without being asked (``/skills`` says which). From the index `_startup_index`
    read once for the status line and the warnings.
    """
    runtime = getattr(agent, "skills", None)
    if index is None or not index.skills or not isinstance(runtime, SkillsRuntime):
        return None
    count = len(index.skills)
    status = f"{count} skill{'s' if count != 1 else ''}"
    layout = runtime.layout(index.skills)
    if layout.omitted:
        status += f" ({len(layout.omitted)} not listed, see /skills)"
    elif layout.tier == "names":
        status += " (by name only, see /skills)"
    elif layout.tier == "short":
        status += " (descriptions shortened, see /skills)"
    return status


async def _startup_index(agent: Agent) -> SkillIndex | None:
    """The root agent's skill index, for the startup status line and warnings.

    ``None`` when the agent has no skills or they cannot be read. Display and checks
    only: the runner loads the catalogue again at the top of every turn, and that is
    where an unreadable source or a required skill is dealt with — so a failure here
    must not stop the session from starting.
    """
    runtime = getattr(agent, "skills", None)
    if not isinstance(runtime, SkillsRuntime):
        return None
    try:
        return await runtime.index(SkillQuery(agent_name=agent.name))
    except Exception as exc:
        get_logger().warning(f"skills: cannot read the skills at startup: {type(exc).__name__}")
        get_logger().debug(f"skills: startup index detail: {exc}")
        return None


async def _resolve_resume(theme: Theme, resume: str) -> Any:
    """Turn ``--resume [id]`` into a conversation, or ``None`` with a reason said.

    Done before the REPL is built rather than inside it, so a bad id fails at
    the point the user typed it instead of one prompt later.
    """
    from domyn.agents.interfaces.cli.repl import APPLICATION_NAME, USER_ID

    manager = conversations.build_manager()
    if resume:
        found = await conversations.find(manager, APPLICATION_NAME, USER_ID, resume)
        if found is None:
            console.print(
                f"  No conversation {resume}.  Run --resume with no id to choose one.",
                style=theme.style("warning"),
                highlight=False,
            )
        return found
    return conversations.pick(
        console, theme, await conversations.recent(manager, APPLICATION_NAME, USER_ID)
    )


def _run_command_for(file: Path | None, agent_name: str | None) -> str:
    """The command that opens this same agent again, for the resume hint.

    A conversation id alone is not enough for a session started from a file:
    `domyn run --resume <id>` would reopen it with the configured agent instead.
    """
    if file is None:
        return "domyn run"
    command = f"domyn run {shlex.quote(str(file))}"
    if agent_name:
        command += f" --agent-name {shlex.quote(agent_name)}"
    return command


def _looks_like_agent_file(value: Path) -> bool:
    """Whether a positional argument is meant as an agent file rather than a prompt."""
    if value.exists():
        return True
    return value.suffix.lower() in (".py", ".yaml", ".yml")


async def resolve_agent_from_file(
    path: Path,
    theme: Theme,
    *,
    allow_setup: bool,
) -> tuple[Agent, dict[str, Any]] | None:
    """Load an agent from a ``ConfigurationV1`` YAML file, applying the skill
    directories its agents declare (``skills.source``) the same way regardless of
    whether the file is the saved default config or one named on the command line.

    ``allow_setup`` gates the interactive wizard — exclusive to the saved
    default config. A file the user named explicitly is a mistake to report
    if it's missing or incomplete, not something to interactively build.

    Returns ``None`` when the file can't be used, so the caller can exit
    quietly instead of raising at them.
    """

    raw = read_raw_config(path)

    if allow_setup and (raw is None or not is_configuration_v1_usable(raw)):
        if raw is None:
            console.print(
                f"  No agent file given and no configuration yet ({path}).",
                style=theme.style("meta"),
                highlight=False,
            )
        else:
            console.print(
                "  Configuration is incomplete.", style=theme.style("warning"), highlight=False
            )
        raw = run_setup(console, theme, raw)
        if raw is None or not is_configuration_v1_usable(raw):
            return None

    if raw is None:
        console.print(f"  Could not read {path}.", style=theme.style("error"), highlight=False)
        return None

    # The skill directories become reachable by the agent's built-in file
    # tools, so it can edit a skill it has been told about. Writing still
    # needs approval. Passed in rather than declared in the YAML — see
    # `ConfigurationV1.from_dict`'s `extra_workspace_roots`.
    config_skill_dirs = skill_workspace_roots(raw)
    skill_dirs = unique_paths(config_skill_dirs)
    try:
        agent = await load_application_from_file(
            str(path), extra_workspace_roots=[str(d) for d in skill_dirs]
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(
            f"  Could not load {path}: {type(exc).__name__}: {exc}",
            style=theme.style("error"),
            highlight=False,
        )
        return None

    return agent, raw


@dataclass(frozen=True)
class SessionOptions:
    """What the command line adds to the configuration's agent. Kept, so the agent can
    be rebuilt the same way when ``config.yaml`` changes mid-session."""

    expand_skills: bool = False
    skill_sections: bool = False


def configure_session_agent(
    agent: Any, options: SessionOptions, *, warn: Callable[[str], None] | None = None
) -> Any:
    """Apply the session's options to the root agent: its skill paths through
    `unique_paths`, the skill selector flags, and the standing instructions.
    The one place this happens, at startup and on every reload."""
    # The main agent's skill paths go through `unique_paths` (duplicates) and
    # the runtime takes --expand-skills / --skill-sections, so a new one is attached
    # to the agent.
    declared_skills = getattr(agent, "skills", None)
    existing_skills = declared_skills if isinstance(declared_skills, SkillsRuntime) else None
    updated_skill_runtime = existing_skills
    if existing_skills:
        if isinstance(existing_skills.source, LocalDirSkillSource):
            skill_runtime_params: dict[str, LocalDirSkillSource | SkillSelector | bool] = {
                "source": LocalDirSkillSource(paths=unique_paths(existing_skills.source.paths))
            }
            if options.expand_skills:
                skill_runtime_params["selector"] = AllSkillsSelector()
            if options.skill_sections:
                skill_runtime_params["index_sections"] = True
            updated_skill_runtime = existing_skills.model_copy(update=skill_runtime_params)
        elif (options.expand_skills or options.skill_sections) and warn is not None:
            # An agent can have only one type of skill source.
            warn(
                "The main agent already has a skill source that is not a local directory. "
                "--expand-skills and --skill-sections will be ignored."
            )

    if updated_skill_runtime:
        agent = attach_skills(agent, updated_skill_runtime)

    # Attached here rather than inside `start_session`: configuring the agent is part
    # of building it, and the session should be handed one that is already complete.
    # The files themselves are read by the runner per turn.
    apply_session_instructions(agent)
    return agent


async def load_session_agent(path: Path, options: SessionOptions) -> tuple[Agent, dict[str, Any]]:
    """Build the session's agent from the config file ``path`` as startup does, minus
    the setup wizard. For the REPL's reload when the file changes: raises when it
    cannot be read or loaded, and the session keeps the agent it has."""
    # Parsed here rather than through `read_raw_config`, which logs a warning of its
    # own: the session reports the failure, once, with where the YAML broke.
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1}" if mark is not None else ""
        raise ValueError(f"not valid YAML{where}") from exc
    if not isinstance(raw, dict):
        raise ValueError("not a configuration: the top level is not a mapping")
    if not is_configuration_v1_usable(raw):
        raise ValueError("incomplete: its agent has no usable model")
    skill_dirs = unique_paths(skill_workspace_roots(raw))
    agent = await load_application_from_file(
        str(path), extra_workspace_roots=[str(d) for d in skill_dirs]
    )
    return configure_session_agent(agent, options), raw


def run_command(
    file: Path | None = typer.Argument(
        None, help="Agent .py or .yaml file. Omit to use the saved configuration."
    ),
    prompt: str | None = typer.Argument(
        None, help="Optional first message. Omit it to open an interactive session."
    ),
    agent_name: str | None = typer.Option(
        None,
        "--agent-name",
        "-a",
        help="Variable name of the agent to run (auto-detected if omitted)",
    ),
    expand_skills: bool = typer.Option(
        False,
        "--expand-skills",
        help="Load every skill body up front instead of on demand (costs more context)",
    ),
    skill_sections: bool = typer.Option(
        False,
        "--skill-sections",
        help="List each skill's sections in the index shown to the model (costs more context)",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show internal events (LLM, hooks, agent lifecycle)"
    ),
    no_banner: bool = typer.Option(False, "--no-banner", help="Skip the startup banner"),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Continue a past conversation: --resume <id>, or --resume alone to choose one",
    ),
    project: ProjectOption = None,
    voice: bool = typer.Option(
        False, "--voice", help="Talk instead of typing: hold space to speak (needs the audio extra)"
    ),
) -> None:
    """Run an agent interactively, or send it a single prompt.

    With no FILE, the agent comes from the saved configuration — and the setup
    wizard runs first if there is none, so a fresh install is usable immediately.
    """
    theme = Theme.build(allow_animation=not no_banner)
    config: dict[str, Any] | None = None
    config_file: Path | None = None
    # Before anything reads a path: the config, the history `--resume` opens and the
    # default instructions all come from the project's folder.
    select_project(project, console, theme, runs_code=True)

    # Drawn first, before the agent is even resolved: on a fresh install the next
    # thing on screen is the setup wizard, and starting that with no branding at
    # all makes the tool feel like it crashed into a form.
    if not no_banner:
        # The project is the one thing on screen saying which configuration, keys and
        # history this session uses, so it sits under the logo with the version.
        selected = active_project()
        subtitle = Text(f"v{__version__}", style=theme.style("meta"))
        if selected is not None:
            # The name in the brand colour: easy to miss in a dim line, and the one
            # word that says which keys and history this session uses.
            subtitle.append("  ·  Project ", style=theme.style("meta"))
            subtitle.append(selected.name, style=theme.style("accent_bold"))
            subtitle.append(f" ({project_location(selected)})", style=theme.style("meta"))
        render_banner(console, theme, subtitle=subtitle, animate=True)

    # Every read on a terminal goes through the line editor, the setup wizard's
    # first field included, so it has to be there before anything is asked. A pipe
    # is read with input() and needs nothing.
    if interactive_terminal():
        install_cli_requirements_here()

    # `domyn run "some question"` is what a user naturally types once FILE is
    # optional, and typer would bind that single argument to FILE. Treat a first
    # argument that is not an agent file as the prompt instead of failing with a
    # confusing "File not found".
    if file is not None and prompt is None and not _looks_like_agent_file(file):
        prompt, file = str(file), None

    # `--resume` takes its id as the next word rather than as an option value.
    # Typer builds `is_flag` itself and mis-parses an option declared to take an
    # optional one: `--resume --no-banner` resolved the id to "--no-banner".
    # Reading the positional is also what the command already does to tell an
    # agent file from a prompt.
    resume_id: str | None = None
    if resume and prompt is not None:
        resume_id, prompt = prompt, None

    try:
        if file is None:
            resolved = asyncio.run(
                resolve_agent_from_file(
                    config_path(),
                    theme,
                    allow_setup=True,
                )
            )
            if resolved is None:
                raise typer.Exit(1)
            agent, config = resolved
            config_file = config_path()
        else:
            path = Path(file)
            if path.suffix in (".yaml", ".yml"):
                resolved = asyncio.run(
                    resolve_agent_from_file(
                        path,
                        theme,
                        allow_setup=False,
                    )
                )
                if resolved is None:
                    raise typer.Exit(1)
                agent, config = resolved
                config_file = path
            else:
                module = load_module_from_path(str(path))
                all_agents = collect_all_agents(module)
                agent = find_main_agent(all_agents, agent_name)

        options = SessionOptions(expand_skills=expand_skills, skill_sections=skill_sections)
        agent = configure_session_agent(
            agent,
            options,
            warn=lambda message: console.print(
                f"  Warning: {message}", style=theme.style("warning"), highlight=False
            ),
        )
        exit_code = asyncio.run(
            start_session(
                agent,
                theme,
                initial_prompt=prompt,
                verbose=verbose,
                show_status=not no_banner,
                config=config,
                config_file=config_file,
                resume=resume_id if resume else None,
                resuming=resume,
                run_command=_run_command_for(file, agent_name),
                # Rebuilt from the file whenever it changes, with the same options.
                reload=(
                    functools.partial(load_session_agent, config_file, options)
                    if config_file is not None
                    else None
                ),
                voice=voice,
            )
        )
        raise typer.Exit(exit_code)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error running agent: {type(e).__name__}: {e}", err=True)
        sys.exit(1)
