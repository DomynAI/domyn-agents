"""Where a command actually runs.

One backend ships here (the local machine); the interface is meant for others,
e.g. a remote executor, implemented outside the package. The three a user can pick
are ``local`` (``local.py``), ``docker`` (``docker/``) and ``auto`` (``auto.py``: Docker,
or this machine when Docker cannot be used). Docker has what a front-end needs to
check, prepare and tear down an environment of its own, and a local machine has none
of that, so none of it is here.

Each member is there for what a remote backend needs and a local one does not:

* **``start``/``close``.** A container has to be created and destroyed; without
  a lifecycle it would create itself lazily inside ``run`` and never be torn
  down.
* **One backend per session, not per command.** It has to survive between calls,
  or ``pip install x`` then ``python -c "import x"`` fails — which is most of
  what a shell is for.
* **A structured result, not a rendered string.** Rendering is the tool's job,
  and a remote backend's own failures ("the container is gone") must stay
  distinguishable from the command's exit code.
* **The command is a string, not an argv.** Both ends run it through a shell;
  see ``shell.py``.
* **The workspace is bound at construction.** A container mounts it once.

``is_isolated`` exists purely for later: an approval policy would key on it to
relax the gate, the way Hermes drops its dangerous-command checks behind a
container. Nothing reads it while every backend answers ``False``.

One seam did not hold: ``run`` cannot carry a backend's own routing information,
because ``FunctionTool.execute`` drops arguments a callable does not declare. An
external backend that needs routing data must currently widen ``run`` and ship its
own tool; widening it here is what would let the shipped tool serve both.

**Anything added here has a consumer outside this repository**: a defaulted
parameter is safe, a required one is a breaking change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from domyn.agents.tools.builtin.session import HOME_DIRNAME, StateFolders


class BackendUnavailableError(RuntimeError):
    """The requested backend cannot be used.

    Raised, never swallowed into a fallback: running on the host while the user
    believes they are isolated turns a safety feature into a hazard.

    ``problem`` says what is wrong and ``fix`` what to do about it, when the backend
    knows: a front-end can lay them out, while ``str(error)`` has both in one line
    for everyone else (a tool error the model reads).
    """

    def __init__(self, problem: str, *, fix: str | None = None) -> None:
        super().__init__(f"{problem} {fix}" if fix else problem)
        self.problem = problem
        self.fix = fix


@dataclass(frozen=True)
class ToolHost:
    """Where and how a tool body runs when it is not in this process."""

    command: str
    """Run through ``CommandBackend.run`` with the request on stdin: it prints the answer."""

    roots: tuple[Path, ...] = ()
    """What the body may reach besides the workspace and the skills. The boundary a
    workspace draws protects what is the user's on this machine; elsewhere the boundary
    is what the backend mounts, and these are what it adds."""


@dataclass(frozen=True)
class CommandResult:
    """What a command did, before it is turned into text for the model."""

    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    backend_error: str | None = None
    """The backend itself failed, as opposed to the command failing. Kept
    separate so "the container is gone" is never reported as "your command
    exited non-zero"."""


class CommandBackend(ABC):
    """Somewhere commands can be run."""

    name: str = "backend"

    is_isolated: bool = False
    """Whether a destructive command is contained by something other than
    trust. See the module docstring: no policy consults this yet."""

    @property
    def state_dir(self) -> str:
        """Where the tools keep their state on the machine commands run on: the
        logs of background commands (``<state_dir>/processes``) and the browser's
        files (``<state_dir>/browser``).

        Relative to the directory commands run in, ``.domyn``, unless a backend
        knows better: a remote one that answers nothing else keeps its state
        beside the workspace it mounts, as before. Put into shell lines quoted."""
        return HOME_DIRNAME

    @property
    def processes_dir(self) -> str:
        """The background commands' folders: ``<state_dir>/processes``."""
        return f"{self.state_dir}/{StateFolders.PROCESSES}"

    @abstractmethod
    async def run(
        self,
        command: str,
        *,
        timeout: float,
        conversation_id: str = "",
        stdin: str | None = None,
    ) -> CommandResult:
        """Run one command. Must not raise for an ordinary command failure.

        ``conversation_id`` selects the working environment. A container backend
        will use it to pick a container; the local one uses it to pick a
        virtualenv, so a package installed in one conversation does not appear
        in another — and, more importantly, not in the runtime running the CLI.

        ``stdin`` is the whole input the command will ever get, supplied up
        front. ``None`` means none — which is not the same as leaving it unset;
        see ``LocalBackend.run``.
        """

    @property
    def tool_host(self) -> "ToolHost | None":
        """How a tool's body runs where this backend runs commands, or ``None`` for in
        this process (see ``files``).

        The tool stays on this side (its schema, its approval) and only the body moves,
        through ``run``: the command named here is run with one JSON request on stdin.
        ``None``, the default, is right for a backend that runs commands on this machine,
        and for any written outside the package: it needs nothing more than ``run``."""
        return None

    # Not abstract, and empty on purpose: a backend with nothing to set up —
    # the local one — should not have to write two empty methods to say so.
    # Forcing them would make "nothing to do" and "forgot to implement" look
    # identical in every subclass.
    async def start(self) -> None:  # noqa: B027
        """Prepare the backend. Idempotent, and called before the first command."""

    async def close(self) -> None:  # noqa: B027
        """Release whatever ``start`` acquired. Must tolerate never having started."""
