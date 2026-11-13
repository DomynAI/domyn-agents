"""What a container can see of the host: the workspace, and very little else.

A command and a tool may write what is mounted, as they may on this machine: the
workspace, the skill folders and the session's state. What a write must not do is
run code on the host or change its policy, so two things differ from this machine:

* git hooks and config are mounted over, read-only: a planted hook runs at the next
  ``git commit``, a ``core.hooksPath`` or ``core.fsmonitor`` entry the same. A mask
  exists only for what exists when the container is created.
* the session's ``config.yaml`` is there to read and not to write: it names the code
  the host runs (tools, hooks, MCP servers) and which backend to use. Credentials and
  approvals are not mounted at all.
"""

from collections.abc import Iterator
from pathlib import Path

from domyn.agents.logger import get_logger
from domyn.agents.tools.builtin.backends.base import BackendUnavailableError
from domyn.agents.tools.builtin.backends.docker.state import StateFolder
from domyn.agents.tools.builtin.workspace import Workspace


def refuse_unsafe_workspace(root: Path) -> None:
    """Mounting the home folder, or ``/``, would hand the container everything in it."""
    home = Path.home().resolve()
    if root == Path(root.anchor) or root == home or root in home.parents:
        raise BackendUnavailableError(
            f"The workspace is {root}: mounting it would hand the container everything in it.",
            fix="Run domyn from a project folder, or set `backend: local`.",
        )


def skill_roots(workspace: Workspace) -> list[Path]:
    """The skill folders that exist, which builtin skills run their scripts from."""
    return [p for p in workspace.extra_roots if p.is_dir() and "," not in str(p)]


def build_mounts(workspace: Workspace, state: StateFolder, conversation: str) -> list[str]:
    """The ``--mount`` arguments for a conversation's container.

    Outermost first: a mount nested under another is applied after it, which is how
    the git masks sit under the workspace.
    """
    root = workspace.root
    state.prepare(conversation)
    mounts = _Mounts()
    mounts.add(root, readonly=False)
    mounts.add(state.processes, readonly=False)
    mounts.add(state.browser, readonly=False)
    mounts.add(state.conversation_home(conversation), readonly=False)
    if state.config is not None and _is_plain_file(state.config):
        mounts.add(state.config, readonly=True)
    for masked in _masked_in(root):
        mounts.add(masked, readonly=True)
    for skills in skill_roots(workspace):
        mounts.add(skills, readonly=False)
    return mounts.arguments


class _Mounts:
    def __init__(self) -> None:
        self.arguments: list[str] = []

    def add(self, path: Path, *, readonly: bool) -> None:
        if "," in str(path):
            get_logger().warning("docker: a path with a comma cannot be mounted; skipped")
            return
        spec = f"type=bind,source={path},target={path}" + (",readonly" if readonly else "")
        self.arguments += ["--mount", spec]


def _masked_in(root: Path) -> Iterator[Path]:
    git = root / ".git"
    if _is_plain_dir(git):
        hooks, config = git / "hooks", git / "config"
        if _is_plain_dir(hooks):
            yield hooks
        if _is_plain_file(config):
            yield config


def _is_plain_dir(path: Path) -> bool:
    """A real folder: a bind mount follows a link, wherever it leads."""
    return path.is_dir() and not path.is_symlink()


def _is_plain_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()
