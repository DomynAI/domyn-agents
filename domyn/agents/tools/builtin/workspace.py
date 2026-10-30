"""The boundary every built-in tool works inside.

One directory is the workspace, and no built-in reads or writes outside it —
except the roots explicitly added to it (see ``Workspace.extra_roots``). That
is the only structural protection these tools have — the approval gate protects
against the *unexpected*, this protects against the expected-but-wrong: a model
that reasonably decides ``~/.ssh/id_rsa`` is relevant to the task.

Containment is done by resolving both paths and comparing, never by string
prefix. ``/home/me/work`` is a prefix of ``/home/me/work-secrets`` and shares no
directory with it; ``../../etc/passwd`` has the right prefix until it is
resolved. Resolution also follows symlinks, which is the case a prefix check
cannot see at all: a link inside the workspace pointing anywhere on the disk.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


class WorkspaceError(Exception):
    """A path escapes the workspace, or cannot be used.

    Raised rather than returned so a tool cannot forget to check. The message is
    read by the model, so it says what is allowed instead of only what failed —
    an error the model can act on saves a turn.
    """


@dataclass(frozen=True)
class Workspace:
    """A resolved root, and the rules for staying inside it."""

    root: Path
    extra_roots: tuple[Path, ...] = ()
    """Directories reachable besides the root.

    They exist for files that belong to the *agent* rather than to the task: its
    skills, above all, which live outside any project workspace — in a directory
    the config names, or in `~/.domyn/builtin-skills` with the files they share. An agent
    that can read its own skill but not edit it is told where the file is and then
    refused, which is worse than not being told at all.

    This does not weaken the approval gate — writing and editing still require
    confirmation, keyed on the path — so widening reach widens what can be
    *asked for*, not what happens unasked. It does widen the structural
    boundary, so nothing belongs here that was not deliberately put here.
    """

    @classmethod
    def at(
        cls,
        path: str | os.PathLike[str] | None = None,
        *,
        also: Iterable[str | os.PathLike[str]] = (),
    ) -> Workspace:
        root = Path(path or Path.cwd()).expanduser().resolve()
        candidates = dict.fromkeys(
            resolved
            for resolved in (Path(one).expanduser().resolve() for one in also)
            if resolved != root and root not in resolved.parents
        )
        # Dropped when already covered, by the root or by another extra root: a
        # redundant root would make the error message list the same place twice
        # (an install directory and each skill inside it) and `display` pick
        # arbitrarily.
        extra = tuple(
            one
            for one in candidates
            if not any(other in one.parents for other in candidates if other != one)
        )
        return cls(root=root, extra_roots=extra)

    @property
    def roots(self) -> tuple[Path, ...]:
        """Every allowed root, the workspace first."""
        return (self.root, *self.extra_roots)

    def anchor(self, resolved: Path) -> Path | None:
        """The root a resolved path sits in or is, or ``None`` when it is in none."""
        return next(
            (root for root in self.roots if resolved == root or root in resolved.parents), None
        )

    def contains(self, resolved: Path) -> bool:
        """Whether an already **resolved** path is inside the workspace or an extra root.

        For paths that came from somewhere other than the model's own argument — the
        matches of a glob, which can name a symlink pointing anywhere. Resolve first:
        a prefix check on an unresolved path is the check the module note rules out.
        """
        return self.anchor(resolved) is not None

    def resolve(self, candidate: str | os.PathLike[str], *, must_exist: bool = False) -> Path:
        """Turn a model-supplied path into a real one inside the workspace.

        Relative paths are relative to the root, not to the process working
        directory: those two drift apart the moment anything calls ``chdir``, and
        a tool that silently resolves against the wrong one is a tool that reads
        the wrong file.
        """
        raw = Path(candidate).expanduser()
        target = raw if raw.is_absolute() else self.root / raw

        # strict=False so a path that does not exist yet still gets its symlinks
        # and `..` resolved — writing a new file has to be checked too.
        resolved = target.resolve()

        if not self.contains(resolved):
            allowed = ", ".join(str(root) for root in self.roots)
            raise WorkspaceError(
                f"'{candidate}' is outside the workspace ({allowed}). "
                "Only paths inside it can be accessed."
            )
        if must_exist and not resolved.exists():
            raise WorkspaceError(f"'{candidate}' does not exist.")
        return resolved

    def display(self, path: Path) -> str:
        """Path as the user should see it: relative to the root when possible.

        Absolute paths in a transcript are noise, and in an approval prompt they
        are worse than noise — a long path pushes the interesting part of the
        line off the screen.
        """
        for root in self.roots:
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            # An extra root is named, because "SKILL.md" alone does not say which
            # of several places it is in — and in an approval prompt that is the
            # whole question.
            return str(relative) if root == self.root else str(path)
        return str(path)


# Directories that are never worth walking and frequently enormous. Skipped by
# the search tools: without this, a single `grep` in a repo with a virtualenv
# spends its whole output budget on third-party code.
IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".next",
        "target",
    }
)


def is_ignored(path: Path, root: Path) -> bool:
    """Whether a path sits inside a directory not worth searching."""
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    return any(part in IGNORED_DIRS for part in parts)
