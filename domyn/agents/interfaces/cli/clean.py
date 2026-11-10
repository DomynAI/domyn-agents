"""``domyn clean`` — take old conversations out of the history, and what they left on disk.

A conversation is a row in ``conversations.db`` (its whole event history, tool outputs
included) and, once the agent ran a command, a folder ``conversations/<id>/`` beside it
holding the virtualenv the agent installed into. Neither is ever removed by itself, so a
heavy user's ``~/.domyn`` only grows; this is the way to give the space back.

Done on the file with ``sqlite3`` rather than through the conversation manager, which
is per application and user: this clears what is in the *store*. SQLite does not shrink
its file when rows go, so the file is ``VACUUM``-ed afterwards — without it the command
would report a saving the disk never saw.
"""

import shutil
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from domyn.agents.tools.builtin.session import safe_id

CONVERSATIONS_DIRNAME = "conversations"


@dataclass
class CleanPlan:
    """What a clean would remove, worked out without removing anything."""

    store: Path
    ids: list[str] = field(default_factory=list)
    kept: int = 0
    folders: list[Path] = field(default_factory=list)
    store_bytes: int = 0
    folder_bytes: int = 0

    @property
    def nothing(self) -> bool:
        return not self.ids and not self.folders


def folder_size(path: Path) -> int:
    """Bytes under ``path``, links counted as themselves and never followed."""
    total = 0
    stack = [path]
    while stack:
        here = stack.pop()
        try:
            for entry in here.iterdir():
                if entry.is_symlink():
                    continue
                if entry.is_dir():
                    stack.append(entry)
                else:
                    total += entry.stat().st_size
        except OSError:
            continue
    return total


def _cutoff(older_than_days: int | None) -> str | None:
    if older_than_days is None:
        return None
    # The format the store writes (`conversation_manager.sqlite._stamp`): text, so
    # it orders only when compared in the same shape.
    return (datetime.now() - timedelta(days=older_than_days)).isoformat(sep=" ")


def plan_clean(home: Path, *, older_than_days: int | None) -> CleanPlan:
    """The conversations to remove: those not touched for ``older_than_days``, or all
    of them with ``None``. Reads the store; a store that is not there is an empty plan."""
    store = home / "conversations.db"
    plan = CleanPlan(store=store)
    kept: set[str] = set()
    cutoff = _cutoff(older_than_days)
    if store.is_file():
        plan.store_bytes = store.stat().st_size
        with sqlite3.connect(store) as db:
            try:
                rows = db.execute("SELECT conversation_id, updated_at FROM conversation").fetchall()
            except sqlite3.OperationalError:  # no table yet: never used
                rows = []
        for conversation_id, updated_at in rows:
            if cutoff is None or str(updated_at) < cutoff:
                plan.ids.append(conversation_id)
            else:
                kept.add(safe_id(conversation_id))
        plan.kept = len(kept)

    root = home / CONVERSATIONS_DIRNAME
    if root.is_dir() and not root.is_symlink():
        removed = {safe_id(i) for i in plan.ids}
        cutoff_time = (
            None if older_than_days is None else datetime.now() - timedelta(days=older_than_days)
        )
        for folder in sorted(root.iterdir()):
            if folder.is_symlink() or not folder.is_dir() or folder.name in kept:
                continue
            # The folder of a conversation being removed, or an orphan: one whose
            # conversation is not in the store (deleted before, another store) and
            # that has not been touched for as long.
            orphan_and_old = folder.name not in removed and (
                cutoff_time is None or datetime.fromtimestamp(folder.stat().st_mtime) < cutoff_time
            )
            if folder.name in removed or orphan_and_old:
                plan.folders.append(folder)
                plan.folder_bytes += folder_size(folder)
    return plan


def apply_clean(plan: CleanPlan) -> int:
    """Remove what ``plan`` lists; return the bytes the store file shrank by plus the
    folders' size. A folder that cannot be removed is skipped and left where it is."""
    freed = 0
    if plan.ids and plan.store.is_file():
        with sqlite3.connect(plan.store) as db:
            db.executemany(
                "DELETE FROM conversation WHERE conversation_id = ?", [(i,) for i in plan.ids]
            )
        # VACUUM cannot run inside a transaction, so on a connection of its own.
        vacuum = sqlite3.connect(plan.store, isolation_level=None)
        try:
            vacuum.execute("VACUUM")
        finally:
            vacuum.close()
        freed += max(0, plan.store_bytes - plan.store.stat().st_size)
    for folder in plan.folders:
        size = folder_size(folder)
        shutil.rmtree(folder, ignore_errors=True)
        if not folder.exists():
            freed += size
    return freed


def human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"
