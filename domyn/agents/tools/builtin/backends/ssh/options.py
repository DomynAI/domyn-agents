"""The ``ssh:`` block of a tool bundle: which server, and where on it."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class SshBackendOptions(BaseModel):
    """Where the SSH backend runs the tools.

    ``host`` is passed to the system ``ssh`` as it is, so a ``Host`` alias of
    ``~/.ssh/config`` works, with everything that file says about it (user, port, key,
    jump host). The other connection fields are for a server the file does not name.
    """

    model_config = ConfigDict(extra="forbid")

    host: str = ""
    """The server: a ``Host`` of ``~/.ssh/config``, ``name`` or ``user@name``."""

    workspace: str = ""
    """The folder on the server the tools work in, as an absolute path: commands start
    there and the file tools may not leave it."""

    user: str | None = None
    port: int | None = None
    identity_file: str | None = None
    """A private key on this machine, when neither the agent nor ``~/.ssh/config``
    gives one."""

    state_dir: str | None = None
    """Where the tools keep their state on the server: the logs of background commands
    and the copy of the skills. Absolute, or relative to ``workspace``. By default
    ``~/.cache/domyn/<id>`` on the server, one per server and workspace, outside the
    workspace so the file tools and ``git`` do not see it."""

    browser: Literal["local", "remote"] = "local"
    """Where the ``browser`` bundle runs: on this machine (the default), or on the
    server, which then needs agent-browser and Chrome."""

    config_file: str | None = None
    """An ``ssh`` configuration file to read instead of ``~/.ssh/config`` (``ssh -F``)."""

    connect_timeout_s: int = 15
    binary: str = "ssh"
    """The ``ssh`` command: a name looked up on PATH, or a path."""

    @field_validator("workspace", "state_dir")
    @classmethod
    def _no_tilde(cls, value: str | None) -> str | None:
        # Every path built on these goes into a shell line quoted, where `~` is not
        # expanded: it would name a folder called `~`.
        if value is not None and value.startswith("~"):
            raise ValueError(f"{value!r}: write the absolute path on the server, without '~'")
        return value

    @field_validator("workspace")
    @classmethod
    def _absolute(cls, value: str) -> str:
        if value and not value.startswith("/"):
            raise ValueError(f"{value!r}: the workspace must be an absolute path on the server")
        return value
