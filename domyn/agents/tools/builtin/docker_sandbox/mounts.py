"""What a container can see of the host: the workspace, and very little else.

The workspace is the one thing a command may write, so every way a write there
could reach the host or the next prompt is mounted over, read-only: git hooks and
config (a planted hook runs on the host at the next ``git commit``), and the
folders and files whose text is read into the next turn's system prompt. A mask
exists only for what exists when the container is created.
"""

from collections.abc import Iterator
from pathlib import Path

from domyn.agents.logger import get_logger
from domyn.agents.tools.builtin.backends import BackendUnavailableError
from domyn.agents.tools.builtin.docker_sandbox.state import StateFolder
from domyn.agents.tools.builtin.session import HOME_DIRNAME
from domyn.agents.tools.builtin.workspace import Workspace

_INSTRUCTION_FILES = ("AGENTS.md", "AGENT.md", "SOUL.md")


def refuse_unsafe_workspace(root: Path) -> None:
    """Mounting the home folder, or ``/``, would hand the container everything in it."""
    home = Path.home().resolve()
    if root == Path(root.anchor) or root == home or root in home.parents:
        raise BackendUnavailableError(
            f"The workspace is {root}: mounting it would hand the container everything in it.",
            fix="Run domyn from a project folder, or set `sandbox: local`.",
        )


def skill_roots(workspace: Workspace) -> list[Path]:
    """The skill folders that exist, which builtin skills run their scripts from."""
    return [p for p in workspace.extra_roots if p.is_dir() and "," not in str(p)]


def build_mounts(workspace: Workspace, state: StateFolder, conversation: str) -> list[str]:
    """The ``--mount`` arguments for a conversation's container.

    Outermost first: a mount nested under another is applied after it, which is how
    the places written in again (the state folder's two) sit under a masked folder.
    """
    root = workspace.root
    state.prepare(conversation)
    mounts = _Mounts()
    mounts.add(root, readonly=False)
    project_folder = root / HOME_DIRNAME
    if _is_plain_dir(project_folder):  # config, skills and instructions of the project
        mounts.add(project_folder, readonly=True)
    mounts.add(state.processes, readonly=False)
    mounts.add(state.conversation_home(conversation), readonly=False)
    for masked in _masked_in(root):
        mounts.add(masked, readonly=True)
    for skills in skill_roots(workspace):
        mounts.add(skills, readonly=True)
    return mounts.arguments


class _Mounts:
    def __init__(self) -> None:
        self.arguments: list[str] = []

    def add(self, path: Path, *, readonly: bool) -> None:
        if "," in str(path):
            get_logger().warning("sandbox: a path with a comma cannot be mounted; skipped")
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
    if _is_plain_dir(root / ".agents"):
        yield root / ".agents"
    for name in _INSTRUCTION_FILES:
        if _is_plain_file(root / name):
            yield root / name


def _is_plain_dir(path: Path) -> bool:
    """A real folder: a bind mount follows a link, wherever it leads."""
    return path.is_dir() and not path.is_symlink()


def _is_plain_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()
