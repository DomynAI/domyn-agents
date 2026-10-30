"""The ``cli`` extra (the line editor), installed into the environment ``domyn`` runs from.

Shared by ``domyn init`` and ``domyn run``: the interactive session reads every
line through ``prompt_toolkit``, and an environment synced without extras (``uv
run`` from a checkout, a plain ``pip install domyn``) does not have it.
"""

import importlib
import subprocess
import sys
from importlib.util import find_spec

import typer


def cli_requirements() -> list[str]:
    """The ``cli`` extra's requirements, read from the installed package's metadata.

    Not from ``pyproject.toml``: there is none beside an installed wheel. Not
    ``domyn[cli]`` either: that would have uv resolve domyn-agents again,
    and replace the editable checkout ``domyn`` is running from.
    """
    from importlib.metadata import requires

    return [
        requirement.split(";")[0].strip()
        for requirement in requires("domyn") or ()
        if requirement.replace('"', "'").endswith("extra == 'cli'")
    ]


def _install(command: list[str]) -> bool:
    """Run one installer; False when it is missing or fails."""
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError as e:
        typer.echo(f"Command failed: {' '.join(command)}", err=True)
        typer.echo(f"Error: {e.stderr}", err=True)
        return False


def install_cli_requirements_here() -> bool:
    """Install the ``cli`` extra into the running environment if it is missing.

    Returns whether ``prompt_toolkit`` can be imported afterwards. ``uv pip`` first,
    ``pip`` when there is no ``uv`` (an environment ``uv`` made has no ``pip``, and
    one ``pip`` made may have no ``uv``). ``uv run`` syncs inexactly, so a package
    installed here stays installed.
    """
    if find_spec("prompt_toolkit") is not None:
        return True
    if sys.prefix == sys.base_prefix:
        # Not a virtual environment: never install into a system interpreter.
        typer.echo(
            f"prompt-toolkit is not installed in {sys.prefix}; "
            'install "domyn[cli]" there for autocomplete and history.',
            err=True,
        )
        return False
    requirements = cli_requirements()
    typer.echo("Installing the CLI dependencies into the current environment...")
    installed = bool(requirements) and (
        _install(["uv", "pip", "install", "--python", sys.executable, *requirements])
        or _install([sys.executable, "-m", "pip", "install", *requirements])
    )
    if not installed:
        typer.echo(
            f'Warning: could not install "domyn[cli]" into {sys.prefix}.',
            err=True,
        )
        return False
    # The import system caches directory listings: without this, the package just
    # written into site-packages stays invisible to this process.
    importlib.invalidate_caches()
    return find_spec("prompt_toolkit") is not None
