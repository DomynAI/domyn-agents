"""The skill folders of this machine, copied to the server for the commands that use them.

Skills are found and read on this machine, as with every backend: the index, the
bodies, ``read_file`` on a skill's files. What a skill *runs* runs on the server, where a
path of this machine (``/Users/me/.domyn/builtin-skills/…``) names nothing. So each skill
folder is copied there, one way, before the first command and again whenever a file in
it changed, and a command's text has each folder's path replaced with its copy's: the
model goes on writing the paths it was shown, and ``cd <skill> && python scripts/x.py``
works as written. Docker does the same by mounting the folders at their own paths, which
a server cannot do.

Nothing is installed: the copy is a ``tar`` stream unpacked by the server's ``tar``,
under the backend's state folder. A script that imports a package the server lacks fails
there, and the agent installs it the way it installs anything, through ``run_command``.
"""

import asyncio
import hashlib
import io
import os
import shlex
import tarfile
from collections.abc import Iterable
from pathlib import Path

from domyn.agents.logger import get_logger
from domyn.agents.tools.builtin.backends.ssh.connection import SshConnection

MAX_COPY_BYTES = 64 * 1024 * 1024
"""Largest skill folder copied. Skills are text and small scripts; a folder this large
is something else listed as a skill, and is left on this machine with a warning."""

_SKIPPED = frozenset({"__pycache__", ".git", "node_modules", ".venv"})

Fingerprint = tuple[tuple[str, int, int], ...]


class SkillCopies:
    """Which skill folders are copied where, and whether a copy is still current."""

    def __init__(self, roots: Iterable[Path]) -> None:
        # Longest first, so a folder inside another is replaced before the one around it.
        self._roots = sorted(dict.fromkeys(roots), key=lambda root: len(str(root)), reverse=True)
        self._copied: dict[Path, Fingerprint] = {}

    @property
    def roots(self) -> list[Path]:
        return list(self._roots)

    def destination(self, root: Path, base: str) -> str:
        """Where ``root`` is copied, under ``base``: a name of safe characters only, so
        it can replace the path wherever the command had it, quoted or not."""
        return f"{base}/{hashlib.sha256(str(root).encode()).hexdigest()[:16]}"

    def rewrite(self, command: str, base: str) -> str:
        """``command`` with every skill folder's path replaced by its copy's."""
        home = Path.home()
        for root in self._roots:
            copy = self.destination(root, base)
            command = command.replace(str(root), copy)
            if home in root.parents:
                command = command.replace(f"~/{root.relative_to(home).as_posix()}", copy)
        return command

    async def sync(self, connection: SshConnection, base: str) -> None:
        """Copy every folder that changed since its last copy. A failure is logged and
        leaves the old copy: a command that does not use the skill should still run."""
        for root in self._roots:
            try:
                fingerprint = await asyncio.to_thread(_fingerprint, root)
                if self._copied.get(root) == fingerprint:
                    continue
                if sum(size for _, size, _ in fingerprint) > MAX_COPY_BYTES:
                    get_logger().warning(f"ssh: {root} is too large to copy to the server")
                    self._copied[root] = fingerprint
                    continue
                archive = await asyncio.to_thread(_archive, root)
                target = shlex.quote(self.destination(root, base))
                out = await connection.run(
                    f"sh -c {shlex.quote(_UNPACK.format(target=target))}",
                    stdin=archive,
                    timeout=300,
                )
                if out is None or out.code != 0:
                    reason = "timed out" if out is None else out.stderr.decode(errors="replace")
                    get_logger().warning("ssh: could not copy a skill folder to the server")
                    get_logger().debug(f"ssh: copying {root} failed: {reason[-300:]}")
                    continue
                self._copied[root] = fingerprint
            except OSError as exc:
                get_logger().warning(f"ssh: could not read a skill folder ({type(exc).__name__})")
                get_logger().debug(f"ssh: reading {root} failed: {exc}")


# Unpacked beside the copy and swapped in, so a command never sees half of one.
_UNPACK = (
    'd={target}; rm -rf "$d.new" && mkdir -p "$d.new" && tar -xf - -C "$d.new" '
    '&& rm -rf "$d" && mv "$d.new" "$d"'
)


def _walk(root: Path) -> list[Path]:
    """The files of a skill folder that are copied: no link (skills never follow one),
    nothing in a cache or a repository's own folder."""
    found = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(
            d for d in dirs if d not in _SKIPPED and not (Path(folder) / d).is_symlink()
        )
        for name in sorted(files):
            path = Path(folder) / name
            if not path.is_symlink() and path.is_file():
                found.append(path)
    return found


def _fingerprint(root: Path) -> Fingerprint:
    result = []
    for path in _walk(root):
        info = path.stat()
        result.append((path.relative_to(root).as_posix(), info.st_size, info.st_mtime_ns))
    return tuple(result)


def _archive(root: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path in _walk(root):
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    return buffer.getvalue()
