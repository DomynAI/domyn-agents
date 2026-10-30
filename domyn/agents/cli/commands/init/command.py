import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import tomlkit
import typer

from domyn.agents.cli.cli_dependencies import install_cli_requirements_here

DOMYN_AGENTS_REQUIREMENT = "domyn[cli]"


def run_command(cmd: list[str], cwd: Path | None = None) -> bool:
    """Run a shell command and return success status."""
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        typer.echo(f"Command succeeded: {' '.join(cmd)}")
        typer.echo(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        typer.echo(f"Command failed: {' '.join(cmd)}", err=True)
        typer.echo(f"Error: {e.stderr}", err=True)
        return False
    except FileNotFoundError:
        typer.echo(f"Command not found: {cmd[0]}", err=True)
        return False


def check_uv_installed() -> bool:
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def create_agent_file(project_dir: Path, project_name: str) -> None:
    """Create the main agent.py with a starter agent and two example tools."""
    agent_content = f'''"""Main agent configuration for {project_name}."""

from domyn.agents.agents import Agent
from domyn.agents.llm import OpenAIProvider
from domyn.agents.tools import FunctionTool

from domyn.agents.logger import get_logger

logger = get_logger()


async def greet_user(name: str) -> str:
    """Greet the user by name."""
    return f"Hello {{name}}! Nice to meet you."


async def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Main agent configuration
# Replace the OpenAIProvider with your preferred LLM provider
agent = Agent(
    name="{project_name.replace("-", "_").replace(" ", "_")}Agent",
    description="A helpful agent for {project_name}",
    instruction="You are a helpful assistant. Use the available tools when needed to help users.",
    llm_provider=OpenAIProvider(
        model_name="gpt-5",  # Replace with your actual model
        generation_params={{"temperature": 0.7}}
    ),
    tools=[
        FunctionTool(greet_user),
        FunctionTool(get_current_time),
    ],
)

# You can add more agents here and set up sub_agents relationships
# Example:
# sub_agent = Agent(...)
# agent.sub_agents = [sub_agent]
'''

    agent_file = project_dir / "agent.py"
    agent_file.write_text(agent_content)
    typer.echo(f"Created agent configuration: {agent_file}")


def create_dockerfile(project_dir: Path) -> None:
    """Create a Dockerfile that packages the project and serves it with domyn."""
    dockerfile_content = """\
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock* ./

# Install dependencies (no dev extras, sync from lock file)
RUN uv sync --no-dev --frozen

# Copy project source
COPY . .

EXPOSE 8000

CMD ["uv", "run", "domyn", "serve", "agent.py", "--host", "0.0.0.0", "--port", "8000"]
"""

    dockerfile = project_dir / "Dockerfile"
    dockerfile.write_text(dockerfile_content)
    typer.echo(f"Created Dockerfile: {dockerfile}")


def create_readme(project_dir: Path, project_name: str) -> None:
    readme_content = f"""# {project_name}

A Domyn Agents project created with the Domyn CLI.

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Configure your LLM provider in `agent.py`

3. Run your agent:
   ```bash
   # Using domyn CLI
   domyn run agent.py "Hello, world!"


   # Or using uv
   uv run domyn run agent.py "Hello, world!"
   ```

## Project Structure

- `agent.py` - Main agent configuration
- `pyproject.toml` - Project dependencies and metadata
- `README.md` - This file

## Customization

Edit `agent.py` to:
- Add new tools/functions
- Configure different LLM providers
- Set up agent hierarchies with sub_agents
- Customize the agent's behavior and instructions

## Available Scripts

- `uv run domyn run agent.py "your prompt"` - Run with custom prompt

## Documentation

For more information about Domyn Agents, visit the documentation at
https://domynai.github.io/domyn-agents/.
"""

    readme_file = project_dir / "README.md"
    readme_file.write_text(readme_content)
    typer.echo(f"Created README: {readme_file}")


def find_domyn_agents_checkout(start: Path) -> Path | None:
    """The domyn-agents source tree ``start`` is inside, or ``None``.

    A project created from a checkout (``uv run domyn init`` in this repo) should
    use that checkout rather than a published ``domyn``. Recognised by its ``pyproject.toml`` naming the project, not by a
    directory name, since a clone can be called anything.
    """
    for directory in (start, *start.parents):
        candidate = directory / "pyproject.toml"
        if not candidate.is_file():
            continue
        try:
            name = tomlkit.parse(candidate.read_text()).get("project", {}).get("name")
        except Exception:
            continue
        if name == "domyn":
            return directory
    return None


def modify_pyproject_toml(
    project_dir: Path,
    local_checkout: Path | None = None,
) -> None:
    pyproject_file = project_dir / "pyproject.toml"

    if not pyproject_file.exists():
        typer.echo("pyproject.toml not found. uv init may have failed.", err=True)
        return

    pyproject: tomlkit.TOMLDocument | None = None
    with open(pyproject_file) as f:
        pyproject = tomlkit.load(f)

    # The line editor (completion, history, multiline) for ``uv run domyn run``.
    pyproject["project"]["dependencies"].add_line(DOMYN_AGENTS_REQUIREMENT)  # type: ignore
    pyproject["project"]["dependencies"].add_line(indent="")  # type: ignore
    pyproject.add(tomlkit.nl())

    if local_checkout is not None:
        # Relative, so a project created inside the checkout keeps working in
        # every clone of it; editable, so changes to the framework show up
        # without a reinstall.
        path = Path(os.path.relpath(local_checkout, project_dir.resolve())).as_posix()
        typer.echo(f"Configuring pyproject.toml to use domyn-agents from {local_checkout}")
        tool = tomlkit.table()
        pyproject["tool"] = tool

        uv = tomlkit.table()
        tool["uv"] = uv

        sources = tomlkit.table()
        uv["sources"] = sources
        sources.append("domyn", tomlkit.inline_table().add("path", path).add("editable", True))

    with open(pyproject_file, "w") as f:
        tomlkit.dump(pyproject, f)

    typer.echo(f"Updated pyproject.toml with {DOMYN_AGENTS_REQUIREMENT} dependency")


def init_command(
    project_name: str | None = typer.Argument(None, help="Name of the project to create"),
    directory: str | None = typer.Option(
        None,
        "--directory",
        "-d",
        help="Directory to create the project in (defaults to project name)",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing directory"),
) -> None:
    """Initialize a new Domyn Agents project.


    Creates a new Python project using uv with Domyn Agents
    and includes a basic agent configuration.
    """

    if not check_uv_installed():
        typer.echo("Error: uv is not installed. Please install uv first:", err=True)
        typer.echo("  Visit: https://docs.astral.sh/uv/getting-started/installation/", err=True)
        sys.exit(1)

    # Determine project name and directory
    if not project_name:
        project_name = cast(str, typer.prompt("Project name"))

    project_dir = Path(directory).resolve() if directory else Path.cwd() / project_name

    # Check if directory exists
    if project_dir.exists() and any(project_dir.iterdir()):  # Directory exists and is not empty
        if force:
            typer.echo(f"Warning: Overwriting existing directory '{project_dir}'", err=True)
            shutil.rmtree(project_dir)
        else:
            typer.echo(
                f"Error: Directory '{project_dir}' already exists and is not empty.", err=True
            )
            typer.echo("Use --force to overwrite or choose a different name.", err=True)
            sys.exit(1)

    # Create directory if it doesn't exist
    project_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Initializing Domyn Agents project: {project_name}")
    typer.echo(f"Project directory: {project_dir}")

    try:
        # Initialize uv project
        typer.echo("Creating Python project with uv...")
        if not run_command(
            [
                "uv",
                "init",
                "--no-workspace",
                "--name",
                project_name,
                "--description",
                f"A Domyn Agents project for {project_name}",
            ],
            cwd=project_dir,
        ):
            typer.echo("Failed to initialize uv project", err=True)
            sys.exit(1)

        # Remove the default main.py if it exists
        package_name = project_name.replace("-", "_")
        main_py = project_dir / package_name / "main.py"
        if main_py.exists():
            main_py.unlink()

        # Also remove the package directory if it's empty
        package_dir = project_dir / package_name
        if package_dir.exists():
            with contextlib.suppress(OSError):
                package_dir.rmdir()  # Only removes if empty

        typer.echo(f"Adding {DOMYN_AGENTS_REQUIREMENT} dependency...")
        modify_pyproject_toml(
            project_dir,
            local_checkout=find_domyn_agents_checkout(Path.cwd().resolve()),
        )

        typer.echo("Creating agent configuration...")
        create_agent_file(project_dir, project_name)

        typer.echo("Creating README...")
        create_readme(project_dir, project_name)

        typer.echo("Creating Dockerfile...")
        create_dockerfile(project_dir)

        typer.echo("Installing dependencies...")
        if not run_command(["uv", "sync"], cwd=project_dir):
            typer.echo("Warning: Failed to install dependencies. Run 'uv sync' manually.", err=True)
        install_cli_requirements_here()

        # Adding .gitignore if not present
        gitignore_file = project_dir / ".gitignore"
        if not gitignore_file.exists():
            gitignore_file.write_text("")

        # Success message
        typer.echo(f"\n✅ Successfully created Domyn Agents project: {project_name}")
        typer.echo(f"📁 Project directory: {project_dir}")
        typer.echo("\n🚀 Next steps:")
        typer.echo(f"   cd {project_dir.name}")
        typer.echo("   # Configure your LLM provider in agent.py")
        typer.echo('   uv run domyn run agent.py "Hello, world!"')
        typer.echo("   # Serve via HTTP:")
        typer.echo("   uv run domyn serve agent.py")
        typer.echo("   # Or build and run with Docker:")
        typer.echo("   docker build -t my-agent . && docker run -p 8000:8000 my-agent")

    except Exception as e:
        typer.echo(f"Failed to create project: {e}", err=True)
        typer.echo(f"Error: Failed to create project: {e}", err=True)
        sys.exit(1)
