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

The state folder's own place in the container is an empty, writable folder (see
``StateFolder.scratch_home``) with those few mounts over it, so ``domyn run`` there can
keep a history of its own without seeing the real one.
"""

from collections.abc import Iterator
from pathlib import Path

from domyn.agents.logger import get_logger
from domyn.agents.tools.builtin.backends.base import BackendUnavailableError
from domyn.agents.tools.builtin.backends.docker.containers import container_home
from domyn.agents.tools.builtin.backends.docker.state import StateFolder
from domyn.agents.tools.builtin.session import HOME_DIRNAME
from domyn.agents.tools.builtin.workspace import Workspace


def refuse_unsafe_workspace(root: Path) -> None:
    """Mounting ``/``, or a folder above the home folder, would hand the container
    everything in it. The home folder itself is allowed, with ``~/.domyn`` (credentials,
    approvals, history) hidden: see ``build_mounts``."""
    home = Path.home().resolve()
    if root == Path(root.anchor) or root in home.parents:
        raise BackendUnavailableError(
            f"The workspace is {root}: mounting it would hand the container everything in it.",
            fix="Run domyn from a project folder, or set `backend: local`.",
        )


def _hidden_global_home(root: Path, state: StateFolder) -> Path | None:
    """``~/.domyn`` when the workspace contains it and the session does not run on it.

    It holds the credentials, the approvals and the whole history, and a workspace that
    is the home folder (or holds ``.domyn``) would mount it with the rest. The session's
    own folder, when it is that one, is already an empty scratch folder over it.
    """
    hidden = Path.home().absolute() / HOME_DIRNAME
    inside = root == hidden or root in hidden.parents
    if not inside or hidden == state.project_dir or "," in str(hidden):
        return None
    return hidden


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
    home = Path(container_home())
    scratch = state.scratch_home(conversation) if state.project_dir else None
    # The folder the container's own `~/.domyn` is: the same scratch folder as the state
    # folder's place, so a `domyn run` without `--project` keeps the history of one
    # conversation and sees none of the real one. Not when the state folder is that folder
    # already, or below it (a project under `~/.domyn`): the scratch would hold itself.
    alias = home / HOME_DIRNAME
    if (
        scratch is None
        or state.project_dir == alias
        or alias in (state.project_dir or alias).parents
    ):
        alias_scratch = None
    else:
        alias_scratch = scratch
    hidden = _hidden_global_home(root, state)
    if root != home:
        # The home folder is empty and writable, with what is mounted below it: Docker
        # makes the parents of a mount as root, so without it nothing could be created
        # there (`~/.cache`, a `.local` of a tool). A workspace that *is* the home is
        # writable already, and a tmpfs would hide it.
        mounts.arguments += ["--tmpfs", f"{home}:rw,mode=1777,exec"]
    mounts.add(root, readonly=False)
    if hidden is not None and alias_scratch is None:
        # An empty folder where the global `~/.domyn` is: nothing of it is reachable.
        # A project of the session, which lives below it, is mounted over this one.
        mounts.arguments += ["--tmpfs", f"{hidden}:rw,size=1m,mode=0755"]
    if scratch is not None and state.project_dir:
        mounts.add(scratch, readonly=False, target=state.project_dir)
    if alias_scratch is not None:
        mounts.add(alias_scratch, readonly=False, target=alias)
    mounts.add(state.processes, readonly=False)
    mounts.add(state.browser, readonly=False)
    mounts.add(state.conversation_home(conversation), readonly=False)
    if state.config is not None and _is_plain_file(state.config):
        mounts.add(state.config, readonly=True)
        if alias_scratch is not None:
            mounts.add(state.config, readonly=True, target=alias / "config.yaml")
    for masked in _masked_in(root):
        mounts.add(masked, readonly=True)
    for skills in skill_roots(workspace):
        mounts.add(skills, readonly=False)
    return mounts.arguments


class _Mounts:
    def __init__(self) -> None:
        self.arguments: list[str] = []

    def add(self, path: Path, *, readonly: bool, target: Path | None = None) -> None:
        where = target or path
        if "," in str(path) or "," in str(where):
            get_logger().warning("docker: a path with a comma cannot be mounted; skipped")
            return
        spec = f"type=bind,source={path},target={where}" + (",readonly" if readonly else "")
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
