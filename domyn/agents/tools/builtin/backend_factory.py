"""Resolving a backend by the name a configuration gives it."""

import os
from collections.abc import Iterable

from domyn.agents.tools.builtin.auto_backend import AutoBackend
from domyn.agents.tools.builtin.backends import (
    BackendUnavailableError,
    CommandBackend,
    LocalBackend,
)
from domyn.agents.tools.builtin.docker_sandbox import DockerBackend, DockerBackendOptions
from domyn.agents.tools.builtin.workspace import Workspace

BACKEND_NAMES = ("auto", "local", "docker")


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
    to be sandboxed gives the illusion of a boundary, which is worse than none.
    """
    if name == "local":
        return LocalBackend(
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
        f"unknown command backend '{name}' (available: {', '.join(BACKEND_NAMES)}). "
        "Refusing to fall back to running on this machine."
    )
