"""Resolving a backend by the name a configuration gives it.

The outermost module of the package: it knows every backend, and none of them knows
it, which is what lets ``base`` be imported by all the rest without a cycle.
"""

import os
from collections.abc import Iterable

from domyn.agents.tools.builtin.backends.auto import AutoBackend
from domyn.agents.tools.builtin.backends.base import BackendUnavailableError, CommandBackend
from domyn.agents.tools.builtin.backends.docker import DockerBackend, DockerBackendOptions
from domyn.agents.tools.builtin.backends.local import this_machine
from domyn.agents.tools.builtin.workspace import Workspace

BACKEND_NAMES: tuple[str, ...] = ("auto", "local", "docker")


def build_backend(
    name: str,
    workspace: Workspace,
    *,
    isolate_environment: bool = True,
    allowed_env: Iterable[str] = (),
    state_dir: str | os.PathLike[str] | None = None,
    docker: DockerBackendOptions | None = None,
) -> CommandBackend:
    """Resolve a backend by name.

    An unknown name raises, unlike every other unknown config value here, which
    degrades with a warning: handing back a local backend to someone who asked
    for a container gives the illusion of a boundary, which is worse than none.

    ``local`` is this machine's shell: ``/bin/sh`` on macOS and Linux, Git Bash
    on Windows, where ``cmd.exe`` is never used. ``git_bash`` names the second
    explicitly, and exists on Windows only: elsewhere there is no Git Bash, and
    quietly running ``/bin/sh`` instead would be running something else than
    the configuration says.
    """
    if name == "git_bash":
        if os.name != "nt":
            raise BackendUnavailableError(
                "the 'git_bash' command backend runs on Windows only; use 'local' here. "
                "Refusing to fall back to another shell."
            )
        name = "local"
    if name in ("auto", "docker") and os.environ.get("DOMYN_BACKEND") == "docker":
        # A `domyn` the agent runs inside our container (it sets this variable): there is
        # no Docker there to look for, and the container already is the boundary.
        name = "local"
    if name == "local":
        return this_machine(
            workspace,
            isolate_environment=isolate_environment,
            allowed_env=allowed_env,
            state_dir=state_dir,
        )
    if name == "auto":
        return AutoBackend(
            workspace,
            isolate_environment=isolate_environment,
            allowed_env=allowed_env,
            docker=docker,
            state_dir=state_dir,
        )
    if name == "docker":
        return DockerBackend(
            workspace, options=docker, allowed_env=allowed_env, state_dir=state_dir
        )
    raise BackendUnavailableError(
        f"unknown command backend '{name}' (available: {', '.join(BACKEND_NAMES)}, git_bash on Windows). "
        "Refusing to fall back to running on this machine."
    )
