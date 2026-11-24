"""The files of the server, for the file tools, through a few standard commands.

Nothing is installed there: every operation is a short shell line of GNU coreutils,
findutils and grep, run over the shared connection. What the tools do with the files
(windows of lines, exact edits, documents converted, images decoded, the workspace's
boundary) happens here, as on this machine; only the bytes travel.

Each tool call gets its own ``RemoteDisk``, which remembers what it learned about a
path for the length of the call: resolving a path also says what it is, and a listing
says what every entry is, so a call costs a few round trips, not one per file.

Where it differs from this machine, on purpose:

* A glob does not walk into a linked folder (``find`` without ``-L``); a link it finds
  is resolved and checked against the workspace like any other match.
* ``grep`` runs on the server, with PCRE (``grep -P``), and every line it returns is
  matched again with Python's ``re``, so nothing is reported that the pattern does not
  match here. A pattern PCRE cannot read, or a ``grep`` without ``-P``, sends the
  files' contents instead, searched here.
"""

import io
import os
import re
import shlex
import subprocess
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import IO

from domyn.agents.tools.builtin.backends.base import BackendUnavailableError
from domyn.agents.tools.builtin.backends.ssh.connection import SSH_ERROR, Output, SshConnection
from domyn.agents.tools.builtin.disk import Disk, DiskError, Hit
from domyn.agents.tools.builtin.workspace import IGNORED_DIRS, Workspace, WorkspaceError

CALL_SECONDS = 120.0
"""How long one operation on the server may take."""

_MAGIC = frozenset("*?[")

# Folders a listing does not enter: the search tools skip them anyway
# (`is_ignored`), and a `node_modules` would be most of what crossed the network.
_PRUNE = (
    r"\( -type d \( "
    + " -o ".join(f"-name {shlex.quote(name)}" for name in sorted(IGNORED_DIRS))
    + r" \) -prune \)"
)


@dataclass(frozen=True)
class _Info:
    kind: str
    """``find``'s letter for what the path leads to: ``f`` a file, ``d`` a folder."""
    size: int | None = None
    links: int | None = None


def _q(path: object) -> str:
    return shlex.quote(os.fspath(path))  # type: ignore[call-overload]


def _sh(script: str) -> str:
    """``script`` run by ``sh``, whatever the login shell that reads the line is."""
    return f"sh -c {shlex.quote(script)}"


def _records(data: bytes) -> list[str]:
    return [os.fsdecode(one) for one in data.split(b"\0")]


class RemoteDisk(Disk):
    """The server's files, one operation per round trip."""

    def __init__(self, connection: SshConnection, *, home: str, timeout: float = CALL_SECONDS):
        self._connection = connection
        self._home = home
        self._timeout = timeout
        self._info: dict[str, _Info | None] = {}
        self._real: dict[str, str] = {}

    # --- talking to the server ---------------------------------------------------------

    def _call(self, script: str, stdin: bytes | None = None) -> Output:
        try:
            out = self._connection.run_sync(_sh(script), stdin=stdin, timeout=self._timeout)
        except subprocess.TimeoutExpired:
            raise DiskError(f"the server did not answer within {self._timeout:.0f}s") from None
        except (BackendUnavailableError, OSError) as exc:
            raise DiskError(f"the server could not be reached: {exc}") from None
        if out.code == SSH_ERROR:
            raise DiskError(f"the server could not be reached: {_tail(out.stderr)}")
        return out

    def _checked(self, script: str, stdin: bytes | None = None) -> Output:
        """``_call``, with a failing command raised as the ``OSError`` pathlib would."""
        out = self._call(script, stdin)
        if out.code != 0:
            raise OSError(_tail(out.stderr) or f"exit code {out.code}")
        return out

    def _remember(self, record: str, path: str) -> None:
        """Keep what ``find -printf '%y %s %n'`` said of a path, tab-separated."""
        kind, size, links = record.split("\t", 2)
        self._info[path] = _Info(kind, int(size), int(links))

    def _stat(self, path: Path) -> _Info | None:
        key = str(path)
        if key not in self._info:
            out = self._call(
                f"find -L {_q(path)} -maxdepth 0 -printf '%y\\t%s\\t%n\\0' 2>/dev/null; true"
            )
            record = _records(out.stdout)[0]
            if record and not record.startswith(("l", "N", "L")):
                self._remember(record, key)
            else:
                # Missing, or a link that leads nowhere: what `exists` says of it here.
                self._info[key] = None
        return self._info[key]

    def _forget(self, path: Path) -> None:
        self._info.pop(str(path), None)

    # --- Disk -------------------------------------------------------------------------

    def resolve(
        self, workspace: Workspace, candidate: str | os.PathLike[str], *, must_exist: bool = False
    ) -> Path:
        raw = os.fspath(candidate)
        if raw == "~" or raw.startswith("~/"):
            raw = self._home + raw[1:]
        target = raw if raw.startswith("/") else f"{workspace.root}/{raw}"
        if target in self._real:
            resolved = self._real[target]
        else:
            out = self._call(
                f"p=$(realpath -m -- {_q(target)}) || exit 1; printf '%s\\0' \"$p\"; "
                "find \"$p\" -maxdepth 0 -printf '%y\\t%s\\t%n\\0' 2>/dev/null; true"
            )
            if out.code != 0:
                raise WorkspaceError(f"'{candidate}' cannot be used: {_tail(out.stderr)}")
            real, record, *_ = [*_records(out.stdout), ""]
            resolved = real
            self._real[target] = real
            if record:
                self._remember(record, real)
            else:
                self._info[real] = None
        path = Path(resolved)
        if not workspace.contains(path):
            allowed = ", ".join(str(root) for root in workspace.roots)
            raise WorkspaceError(
                f"'{candidate}' is outside the workspace ({allowed}). "
                "Only paths inside it can be accessed."
            )
        if must_exist and not self.exists(path):
            raise WorkspaceError(f"'{candidate}' does not exist.")
        return path

    def real(self, path: Path) -> Path:
        key = str(path)
        if key not in self._real:
            out = self._checked(f"realpath -e -z -- {_q(path)}")
            self._real[key] = _records(out.stdout)[0]
        return Path(self._real[key])

    def exists(self, path: Path) -> bool:
        return self._stat(path) is not None

    def is_dir(self, path: Path) -> bool:
        info = self._stat(path)
        return info is not None and info.kind == "d"

    def is_file(self, path: Path) -> bool:
        info = self._stat(path)
        return info is not None and info.kind == "f"

    def size(self, path: Path) -> int:
        info = self._stat(path)
        if info is None or info.size is None:
            self._forget(path)
            info = self._stat(path)
        if info is None or info.size is None:
            raise FileNotFoundError(str(path))
        return info.size

    def links(self, path: Path) -> int:
        info = self._stat(path)
        if info is None or info.links is None:
            self._forget(path)
            info = self._stat(path)
        if info is None or info.links is None:
            raise FileNotFoundError(str(path))
        return info.links

    @contextmanager
    def lines(self, path: Path) -> Iterator[Iterable[str]]:
        with self._connection.stream(_sh(f"cat -- {_q(path)}")) as process:
            assert process.stdout is not None
            yield io.TextIOWrapper(process.stdout, encoding="utf-8", newline=None)
            # Read to the end: the command's own failure. Stopped early: killed on
            # the way out, and nothing to report.
            if process.poll() not in (None, 0):
                stderr = process.stderr.read() if process.stderr is not None else b""
                if process.returncode == SSH_ERROR:
                    raise DiskError(f"the server could not be reached: {_tail(stderr)}")
                raise OSError(_tail(stderr) or f"could not read {path}")

    def read_bytes(self, path: Path) -> bytes:
        return self._checked(f"cat -- {_q(path)}").stdout

    def read_text(self, path: Path) -> str:
        text = self.read_bytes(path).decode("utf-8")
        # As a file opened in text mode reads it here.
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def write_text(self, path: Path, text: str) -> None:
        self._forget(path)
        self._checked(
            f"mkdir -p -- {_q(path.parent)} && cat > {_q(path)}", stdin=text.encode("utf-8")
        )

    def append_text(self, path: Path, text: str) -> None:
        self._forget(path)
        self._checked(
            f"mkdir -p -- {_q(path.parent)} && cat >> {_q(path)}", stdin=text.encode("utf-8")
        )

    def iterdir(self, path: Path) -> list[Path]:
        out = self._checked(
            f"find {_q(path)} -mindepth 1 -maxdepth 1 -printf '%Y\\t%s\\t%n\\t%p\\0'"
        )
        entries = []
        for record in _records(out.stdout):
            if record:
                kind, _, _, name = record.split("\t", 3)
                # A link's own size and count say nothing of what it leads to: only
                # its kind is kept, the rest asked when needed.
                self._info[name] = _Info(kind)
                entries.append(Path(name))
        return entries

    def glob(self, base: Path, pattern: str) -> list[Path]:
        parts = PurePosixPath(pattern).parts
        if ".." in parts or not parts or parts[0] == "/":
            return []
        literal = 0
        while literal < len(parts) and not _MAGIC & set(parts[literal]) and parts[literal] != "**":
            literal += 1
        start = base.joinpath(*parts[:literal])
        rest = parts[literal:]
        if not rest:
            return [start] if self.exists(start) else []
        depth = "" if "**" in rest else f"-maxdepth {len(rest)}"
        walk = f"find {_q(start)} -mindepth 1 {depth} {_PRUNE} -o"
        out = self._call(
            f"realpath -z -m -- {_q(start)}; "
            f"{walk} -printf '%y\\t%Y\\t%s\\t%n\\t%p\\0' 2>/dev/null; printf '\\0'; "
            f"{walk} \\( -type l -printf '%p\\0' -exec realpath -z -m -- {{}} \\; \\) 2>/dev/null; "
            "true"
        )
        records = _records(out.stdout)
        real_start, records = records[0], records[1:]
        split = records.index("") if "" in records else len(records)
        entries, links = records[:split], records[split + 1 :]
        for name, real in zip(links[0::2], links[1::2], strict=False):
            if name:
                self._real[name] = real

        dirs_only = rest[-1] == "**"
        prefix = f"{start}/"
        found = []
        for record in entries:
            own, kind, size, count, name = record.split("\t", 4)
            if own == "l":
                self._info[name] = _Info(kind) if kind not in ("N", "L") else None
            else:
                self._info[name] = _Info(kind, int(size), int(count))
                self._real.setdefault(name, real_start + name[len(str(start)) :])
            names = tuple(name.removeprefix(prefix).split("/"))
            if (not dirs_only or kind == "d") and _match(names, rest):
                found.append(Path(name))
        return found

    def search(
        self, files: Sequence[Path], regex: re.Pattern[str], limit: int
    ) -> tuple[list[Hit], int]:
        if not files:
            return [], 0
        listing = b"".join(os.fsencode(os.fspath(one)) + b"\0" for one in files)
        pattern = shlex.quote(regex.pattern)
        script = (
            'export LC_ALL=C.UTF-8; t=$(mktemp) || exit 1; cat > "$t"; '
            'exec 3<"$t" 4<"$t"; rm -f "$t"; '
            f"printf '' | grep -qP -e {pattern} 2>/dev/null; "
            "if [ $? -le 1 ]; then printf 'P\\n'; "
            "xargs -0 grep -I -l -Z -e '' -- <&3 | tr -dc '\\000' | wc -c; "
            f"xargs -0 grep -I -n -H -Z -P -e {pattern} -- <&4; "
            "else printf 'F\\n'; "
            'xargs -0 sh -c \'for f; do s=$(wc -c < "$f" 2>/dev/null) || continue; '
            'printf "%s\\000%s\\000" "$f" "$s"; cat -- "$f"; done\' sh <&3; fi; true'
        )
        with self._connection.stream(_sh(script), stdin=listing) as process:
            assert process.stdout is not None
            mode = process.stdout.readline().strip()
            if mode == b"P":
                return _grepped(process.stdout, regex, limit)
            if mode == b"F":
                return _searched_here(process.stdout, regex, limit)
            stderr = process.stderr.read() if process.stderr is not None else b""
        raise DiskError(f"the search could not run on the server: {_tail(stderr)}")


def _grepped(stream: IO[bytes], regex: re.Pattern[str], limit: int) -> tuple[list[Hit], int]:
    """Read ``grep -I -n -H -Z``'s lines, each matched again here."""
    searched = int(stream.readline().strip() or 0)
    hits: list[Hit] = []
    for raw in stream:
        name, _, rest = raw.rstrip(b"\n").partition(b"\0")
        number, _, line = rest.partition(b":")
        text = line.decode("utf-8", "replace").rstrip("\r")
        if regex.search(text):
            hits.append((Path(os.fsdecode(name)), int(number), text))
            if len(hits) >= limit:
                break
    return hits, searched


def _searched_here(stream: IO[bytes], regex: re.Pattern[str], limit: int) -> tuple[list[Hit], int]:
    """Read the files as name, NUL, size, NUL, bytes, one after the other, and search
    them here."""
    hits: list[Hit] = []
    searched = 0
    while True:
        name = _until_nul(stream)
        if name is None:
            break
        size = int(_until_nul(stream) or 0)
        data = stream.read(size)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        searched += 1
        for number, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                hits.append((Path(os.fsdecode(name)), number, line))
                if len(hits) >= limit:
                    return hits, searched
    return hits, searched


def _until_nul(stream: IO[bytes]) -> bytes | None:
    data = bytearray()
    while True:
        byte = stream.read(1)
        if not byte:
            return bytes(data) if data else None
        if byte == b"\0":
            return bytes(data)
        data += byte


def _match(names: tuple[str, ...], pattern: tuple[str, ...]) -> bool:
    """Whether a path, as its names below the folder searched, matches a glob's parts,
    ``**`` standing for any number of folders."""
    if not pattern:
        return not names
    if pattern[0] == "**":
        return any(_match(names[i:], pattern[1:]) for i in range(len(names) + 1))
    return bool(names) and fnmatchcase(names[0], pattern[0]) and _match(names[1:], pattern[1:])


def _tail(data: bytes) -> str:
    return data.decode("utf-8", "replace").strip()[-300:]
