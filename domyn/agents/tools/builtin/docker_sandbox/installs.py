"""What the agent installed in a conversation, kept so a new container has it back.

A container lives as long as its conversation, but it is made again when the
conversation resumes or the configuration changes. After a command that installed
Python or global npm packages, the packages are written under the conversation's
folder on the host (``requirements.txt`` and ``npm-global.txt``), and installed
again when a container is made for it. A local ``npm install`` needs no record: its
``node_modules`` is in the workspace. Not kept: ``npx`` caches, and system packages
(there is no root).
"""

import json
import os
import re
import shlex
from collections.abc import Callable
from pathlib import Path

from domyn.agents.logger import get_logger
from domyn.agents.tools.builtin.docker_sandbox.cli import DockerCli
from domyn.agents.tools.builtin.docker_sandbox.containers import DockerContainer
from domyn.agents.tools.builtin.docker_sandbox.state import StateFolder

REQUIREMENTS_FILE = "requirements.txt"
NPM_GLOBALS_FILE = "npm-global.txt"

# A quote may come before the program: `run_command` sends the backend a launcher
# script with the model's command quoted inside it, not the command itself.
_BEFORE_PROGRAM = r"(?:^|[\s;&|('\"])"

PIP_COMMAND = re.compile(_BEFORE_PROGRAM + r"(?:pip3?|pipx|uv|python3?\s+-m\s+pip)(?:\s|$)")
"""A command that may have changed what is installed. Checked on the text only to
decide whether to look; what is recorded is what ``pip freeze`` then says."""

NPM_GLOBAL_COMMAND = re.compile(
    _BEFORE_PROGRAM + r"npm\s[^;&|\n]*?\s(?:-g|--global|--location[= ]global)(?:\s|$)"
)
"""An ``npm`` command that may have changed the global packages."""

_RESTORE_SECONDS = 900


class InstallLog:
    def __init__(
        self,
        cli: DockerCli,
        state: StateFolder,
        workdir: Path,
        notify: Callable[[str], None],
        *,
        can_download: bool,
    ) -> None:
        self._cli = cli
        self._state = state
        self._workdir = workdir
        self._notify = notify
        self._can_download = can_download

    def requirements_path(self, conversation: str) -> Path:
        return self._state.conversation_home(conversation) / REQUIREMENTS_FILE

    def npm_globals_path(self, conversation: str) -> Path:
        return self._state.conversation_home(conversation) / NPM_GLOBALS_FILE

    async def record(self, container: DockerContainer, conversation: str, command: str) -> None:
        """Note what a command that succeeded installed, if it may have installed anything."""
        if PIP_COMMAND.search(command):
            await self._record_python(container, conversation)
        if NPM_GLOBAL_COMMAND.search(command):
            await self._record_npm(container, conversation)

    async def replay(self, container: DockerContainer, conversation: str) -> None:
        """Install again what this conversation installed in an earlier container."""
        python = _read_lines(self.requirements_path(conversation))
        npm = _read_lines(self.npm_globals_path(conversation))
        if not python and not npm:
            return
        count = len(python) + len(npm)
        if not self._can_download:
            self._notify(
                f"{count} packages were installed in this conversation, but the "
                "sandbox has no network to restore them."
            )
            return
        self._notify(f"Restoring {count} package(s) installed earlier in this conversation…")
        if python:
            requirements = shlex.quote(str(self.requirements_path(conversation)))
            await self._restore(
                container, f"pip install --user --quiet -r {requirements}", "Python"
            )
        if npm:
            packages = " ".join(shlex.quote(line) for line in npm)
            await self._restore(
                container, f"npm install -g --no-audit --no-fund --loglevel=error {packages}", "npm"
            )

    async def _restore(self, container: DockerContainer, command: str, kind: str) -> None:
        result = await self._cli(
            "exec", "-w", str(self._workdir), container.name, "bash", "-c", command,
            timeout=_RESTORE_SECONDS,
        )  # fmt: skip
        if not result.ok:
            self._notify(f"Could not restore every {kind} package: {result.error(160)}")

    async def _record_python(self, container: DockerContainer, conversation: str) -> None:
        frozen = await self._cli(
            "exec", container.name, "bash", "-c",
            'export PATH="$HOME/.local/bin:$PATH"; pip freeze --user 2>/dev/null',
            timeout=60,
        )  # fmt: skip
        if frozen.ok:
            _write(self.requirements_path(conversation), frozen.stdout, "Python packages")

    async def _record_npm(self, container: DockerContainer, conversation: str) -> None:
        listed = await self._cli(
            "exec", container.name, "bash", "-c",
            'export PATH="$HOME/.npm-global/bin:$PATH"; npm ls -g --depth=0 --json 2>/dev/null',
            timeout=60,
        )  # fmt: skip
        packages = _npm_packages(listed.stdout)
        if packages is not None:
            _write(self.npm_globals_path(conversation), packages, "npm packages")


def _npm_packages(npm_ls_json: str) -> str | None:
    """``name@version`` lines for the top-level global packages, or ``None`` for output
    that is not what ``npm ls --json`` prints."""
    try:
        found = json.loads(npm_ls_json).get("dependencies") or {}
        lines = sorted(
            f"{name}@{info['version']}"
            for name, info in found.items()
            if isinstance(info, dict) and info.get("version")
        )
    except (ValueError, AttributeError):
        return None
    return "".join(f"{line}\n" for line in lines)


def _read_lines(path: Path) -> list[str]:
    try:
        if path.is_symlink() or not path.is_file():
            return []
        return [
            line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    except OSError:
        return []


def _write(path: Path, text: str, what: str) -> None:
    """Replace the record atomically, unless it is already this or is a link (a
    write through one would land wherever it leads)."""
    try:
        if path.is_symlink() or (path.exists() and path.read_text(encoding="utf-8") == text):
            return
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    except OSError as exc:
        get_logger().warning(f"sandbox: could not record installed {what}: {type(exc).__name__}")
        get_logger().debug(f"sandbox install record detail: {exc}")
