import typer

from domyn.agents.cli.commands import (
    connect_command,
    init_command,
    run_command,
    serve_command,
)
from domyn.agents.cli.commands.export import export_command
from domyn.agents.cli.commands.setup import setup_command
from domyn.agents.cli.commands.skills import skill_command, skills_command
from domyn.agents.cli.utils import get_version, package_name, setup_logging

app = typer.Typer(
    name="cli",
    help="Main entrypoint for the CLI",
    no_args_is_help=True,
    add_completion=False,
    # `-h` alongside `--help`, on the root app and every subcommand.
    context_settings={"help_option_names": ["-h", "--help"]},
)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(get_version())
        raise typer.Exit()


# Commands the user watches happen, as opposed to services they run.
#
# The distinction decides the default log level, and the two answers are
# genuinely opposite. A server has no other output, so INFO *is* its interface:
# every request logged is a request observable. An interactive session already
# shows the user what is going on, and the same lines become a wall of noise
# between their question and the answer — inference URLs, httpx status lines,
# runner bookkeeping, all of it above the reply they are waiting for.
#
# So these default to WARNING. ``--log-level`` still overrides, and inside a
# session ``/verbose`` turns them back on without leaving it.
_QUIET_COMMANDS = frozenset({"run", "setup", "init", "skills", "skill", "export"})


@app.callback()
def cli(
    ctx: typer.Context,
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        help="Set the logging level (default: WARNING for interactive commands, INFO for servers)",
        case_sensitive=False,
    ),
    version: bool | None = typer.Option(
        None, "--version", callback=version_callback, help="Show version and exit"
    ),
) -> None:
    """Main entrypoint for the CLI"""

    quiet = ctx.invoked_subcommand in _QUIET_COMMANDS
    resolved = log_level or ("WARNING" if quiet else "INFO")
    setup_logging(resolved)

    # The startup line is bookkeeping for a service and clutter above a prompt.
    if not quiet:
        typer.echo(f"Starting {package_name} CLI version {get_version()} with log level {resolved}")


app.command("init", help="Initialize a new Domyn Agents project")(init_command)
app.command(
    "setup",
    help="Configure the model, API key, tools, skills and instructions",
    epilog=(
        "Run it again at any time to change the configuration.\n\n"
        "Projects keep separate configurations side by side: `domyn setup --project work` "
        "creates or changes the project 'work', and `domyn run --project work` starts it.\n\n"
        "`domyn setup --import <folder>` creates a new project from a folder made by "
        "`domyn export`."
    ),
)(setup_command)
app.command("skills", help="List the skills the configuration loads")(skills_command)
app.command("skill", help="Show or remove a configured skill (show|remove <name>)")(skill_command)
app.command(
    "export",
    help="Save the configuration, its skills and instructions into a new folder to share",
    epilog=(
        "The folder is a project that works on any machine. Use it where it is with "
        "`domyn run --project <folder>`, or move it to ~/.domyn/projects/<name> and use "
        "`domyn run --project <name>`, or create a new project from it, reviewing each "
        "setting, with `domyn setup --import <folder>`.\n\n"
        "API keys, approvals and conversation history are never included."
    ),
)(export_command)
app.command("run", help="Chat with an agent — interactively, or with a single prompt")(run_command)
app.command(
    "serve",
    hidden=True,
    help="Serve an agent from a Python file using the specified protocol",
)(serve_command)
app.command("connect", help="Connect a runnable to the Domyn platform as a relay worker")(
    connect_command
)
app.command("expose", hidden=True, help="Alias for `domyn connect`")(connect_command)
