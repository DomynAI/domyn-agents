"""Where a command actually runs. See ``base`` for the interface.

``base`` is imported by everything else here; ``backends`` is the outermost module
and imports them all. The names below are what code outside the package uses.
"""

from domyn.agents.tools.builtin.backends.auto import AutoBackend
from domyn.agents.tools.builtin.backends.backends import BACKEND_NAMES, build_backend
from domyn.agents.tools.builtin.backends.base import (
    BackendUnavailableError,
    CommandBackend,
    CommandResult,
    ToolHost,
)
from domyn.agents.tools.builtin.backends.docker import DockerBackend, DockerBackendOptions
from domyn.agents.tools.builtin.backends.local import LocalBackend

__all__ = [
    "BACKEND_NAMES",
    "AutoBackend",
    "BackendUnavailableError",
    "CommandBackend",
    "CommandResult",
    "DockerBackend",
    "DockerBackendOptions",
    "LocalBackend",
    "ToolHost",
    "build_backend",
]
