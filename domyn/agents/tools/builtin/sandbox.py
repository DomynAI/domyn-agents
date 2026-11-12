"""Backends that run commands in an environment of their own.

A container (``docker_sandbox/``), or whatever ``auto`` settles on, is somewhere
that has to be checked before it is used, prepared the first time, made ready for
each conversation and torn down with it. The local machine has none of that, so none
of it is on :class:`~domyn.agents.tools.builtin.backends.CommandBackend`: a front-end
that wants to do these things asks for a ``SandboxBackend``.
"""

from abc import abstractmethod
from collections.abc import Callable

from domyn.agents.tools.builtin.backends import CommandBackend


class SandboxBackend(CommandBackend):
    destroys_on_close: bool = False
    """Whether what a command started is stopped when its conversation ends or the
    session closes. A container answers ``True``: unlike on this machine, a command
    left running in it does not outlive the session."""

    on_notice: Callable[[str], None] | None = None
    """Where the backend says what it is doing that takes a while (pulling an image,
    starting a container). The front-end sets it; ``None`` is silent."""

    @abstractmethod
    async def check(self) -> None:
        """Raise :class:`BackendUnavailableError` if this backend cannot be used at all
        (Docker not installed, daemon not running). Fast, and downloads nothing: a
        front-end calls it at start-up to refuse to run, rather than discover it at the
        first command."""

    @abstractmethod
    async def prepare(self) -> None:
        """Do the slow, first-time part of ``start`` now, in the foreground, so the
        front-end can say what is happening."""

    @abstractmethod
    async def warm_up(self, conversation_id: str) -> None:
        """Have what ``conversation_id`` will need ready before its first command.
        Quiet, and must not raise: the first command meets the same failure."""

    @abstractmethod
    async def end_conversation(self, conversation_id: str) -> None:
        """The conversation is over: release what it holds (a container)."""

    async def explain_failure(self, since: float) -> str:
        """Something the backend knows about why a command that started at ``since``
        (``time.time()``) failed, which the command's own output cannot say: a
        connection the sandbox refused. Called by the shell tool for a failed command
        and appended to what the model reads; ``""`` when there is nothing to add."""
        return ""
