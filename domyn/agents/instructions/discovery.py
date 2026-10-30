"""Finding ``SOUL.md`` / ``AGENTS.md`` under a folder.

Two callers. ``domyn setup`` walks once, when the user names a folder, and saves
each file it finds by path, so what a session reads is exactly what setup showed: a
file added there later is read only once setup runs again. The loader walks only the
two autodiscovery entries, ``./.domyn`` and ``./.agents`` (`AUTODISCOVERY_DIRS`), and
does it on every load, so a file added anywhere below them is read on the next
message; every other declared directory contributes the files directly in it
(`LocalDirInstructionsSource`).
"""

import contextlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from domyn.agents.skills.layout import SKIPPED_DIRS, listed_path_is_linked

SOUL_FILENAME = "SOUL.md"

AGENTS_FILENAMES: tuple[str, ...] = ("AGENTS.md", "AGENT.md")
"""Accepted spellings, in precedence order. Only the first one found in a directory
is read: ``AGENTS.md`` is the de-facto standard, ``AGENT.md`` the spelling Amp falls
back to, and reading both would concatenate two attempts at the same file."""

AUTODISCOVERY_DIRS: tuple[Path, ...] = (Path(".domyn"), Path(".agents"))
"""The project's own instruction folders, relative to where the CLI is started.

Listed as written in an agent's paths, they are walked on every load (`STATE_DIRS`
skipped) rather than read flat. When ``domyn setup`` is told to, it writes them in this
order **after** the user's own paths — the reverse of the skills folders, because the
order means the opposite: the first soul wins, so the user's beats a project's (and
``.domyn``'s beats ``.agents``'s), and every ``AGENTS.md`` is included with the last one
weighed most, so the project's brief lands after the user's.
"""

STATE_DIRS = frozenset({"conversations", "browser", "processes", "skills"})
"""Folders domyn itself writes directly under ``.domyn`` (and ``.agents/skills``),
skipped by the autodiscovery walk.

Their content is the agent's own scratch files or skill folders, and an ``AGENTS.md``
in there must never reach the system prompt. `tests/instructions/test_state_dirs.py`
fails when a built-in starts writing to a folder that is not listed here.
"""


def is_regular_file(path: Path) -> bool:
    """A regular file that is not a link: a linked ``SOUL.md`` is not followed."""
    return not path.is_symlink() and path.is_file()


@dataclass(frozen=True)
class FoundInstructions:
    soul: Path | None = None
    """The shallowest ``SOUL.md``: an agent has one identity, so the walk keeps the first."""
    agents: list[Path] = field(default_factory=list)
    """Every ``AGENTS.md`` (or ``AGENT.md`` where there is none), shallowest first."""
    superseded: list[Path] = field(default_factory=list)
    """Every ``SOUL.md`` after the first, shallowest first, so the loader can say
    which ones it did not use."""

    @property
    def paths(self) -> list[Path]:
        """Everything used, the soul first."""
        return [*([self.soul] if self.soul else []), *self.agents]


def find_instruction_files(root: Path, *, skip: frozenset[str] = frozenset()) -> FoundInstructions:
    """The instruction files under ``root``, breadth first, sorted by name within a level.

    Breadth first, so "the first ``SOUL.md``" is the one nearest the root, and the
    ``AGENTS.md`` files come out from the most general to the most specific — the
    order the loader keeps within a kind, where the later one is what a model follows.
    In each folder only the first of the accepted spellings is taken, as the loader
    does. Hidden and dependency folders are skipped and no link is followed, as in
    the skills walk. ``skip`` names folders skipped among the root's direct children
    only, where domyn keeps its state; a folder of that name deeper down is walked.
    A missing root, a file, or a linked one gives nothing; a folder that cannot be
    listed ends the walk with what was found so far.
    """
    if listed_path_is_linked(root) or not root.is_dir():
        return FoundInstructions()

    souls: list[Path] = []
    agents: list[Path] = []
    level = [root]
    with contextlib.suppress(OSError):
        while level:
            below: list[Path] = []
            for directory in level:
                if is_regular_file(directory / SOUL_FILENAME):
                    souls.append(directory / SOUL_FILENAME)
                agents_file = next(
                    (
                        directory / name
                        for name in AGENTS_FILENAMES
                        if is_regular_file(directory / name)
                    ),
                    None,
                )
                if agents_file is not None:
                    agents.append(agents_file)
                with os.scandir(directory) as entries:
                    children = sorted(entries, key=lambda e: e.name)
                for entry in children:
                    if entry.name.startswith(".") or entry.name in SKIPPED_DIRS:
                        continue
                    if directory == root and entry.name in skip:
                        continue
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    below.append(directory / entry.name)
            level = below
    return FoundInstructions(soul=next(iter(souls), None), agents=agents, superseded=souls[1:])
