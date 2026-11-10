"""``domyn clean``: old conversations leave the store and the disk."""

import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

from typer.testing import CliRunner

from domyn.agents.cli.cli import app
from domyn.agents.interfaces.cli import clean


def _store(home: Path, rows: dict[str, int]) -> None:
    """``rows``: conversation id -> days since it was last touched."""
    with sqlite3.connect(home / "conversations.db") as db:
        db.execute(
            "CREATE TABLE conversation (application_name TEXT, user_id TEXT, "
            "conversation_id TEXT, events_history TEXT, updated_at DATETIME)"
        )
        for cid, days in rows.items():
            stamp = (datetime.now() - timedelta(days=days)).isoformat(sep=" ")
            db.execute(
                "INSERT INTO conversation VALUES ('a', 'u', ?, ?, ?)", (cid, "x" * 50_000, stamp)
            )
        db.commit()


def _venv(home: Path, cid: str, age_days: int = 0) -> Path:
    folder = home / "conversations" / cid
    (folder / "venv").mkdir(parents=True)
    (folder / "venv" / "big").write_bytes(b"0" * 10_000)
    stamp = time.time() - age_days * 86400
    os.utime(folder, (stamp, stamp))
    return folder


def test_only_old_conversations_are_planned(tmp_path: Path) -> None:
    _store(tmp_path, {"old": 90, "new": 1})
    old, new = _venv(tmp_path, "old"), _venv(tmp_path, "new")

    plan = clean.plan_clean(tmp_path, older_than_days=30)

    assert plan.ids == ["old"] and plan.kept == 1
    assert plan.folders == [old] and new not in plan.folders


def test_an_old_orphan_folder_goes_and_a_recent_one_stays(tmp_path: Path) -> None:
    _store(tmp_path, {"new": 1})
    gone, fresh = _venv(tmp_path, "gone", age_days=90), _venv(tmp_path, "fresh")

    plan = clean.plan_clean(tmp_path, older_than_days=30)

    assert plan.folders == [gone] and fresh not in plan.folders


def test_applying_it_removes_rows_folders_and_shrinks_the_file(tmp_path: Path) -> None:
    _store(tmp_path, {"old": 90, "new": 1})
    old = _venv(tmp_path, "old")
    before = (tmp_path / "conversations.db").stat().st_size

    freed = clean.apply_clean(clean.plan_clean(tmp_path, older_than_days=30))

    assert not old.exists()
    with sqlite3.connect(tmp_path / "conversations.db") as db:
        assert [r[0] for r in db.execute("SELECT conversation_id FROM conversation")] == ["new"]
    assert (tmp_path / "conversations.db").stat().st_size < before
    assert freed > 10_000


def test_a_missing_store_is_an_empty_plan(tmp_path: Path) -> None:
    assert clean.plan_clean(tmp_path, older_than_days=0).nothing


def test_the_command_needs_to_be_told_what_to_remove(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("domyn.agents.cli.commands.clean.command.home_dir", lambda: tmp_path)
    result = CliRunner().invoke(app, ["clean"])
    assert result.exit_code == 1


def test_the_command_removes_with_yes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("domyn.agents.cli.commands.clean.command.home_dir", lambda: tmp_path)
    _store(tmp_path, {"old": 90})

    dry = CliRunner().invoke(app, ["clean", "--all", "--dry-run"])
    assert dry.exit_code == 0 and clean.plan_clean(tmp_path, older_than_days=None).ids == ["old"]

    done = CliRunner().invoke(app, ["clean", "--all", "--yes"])
    assert done.exit_code == 0 and clean.plan_clean(tmp_path, older_than_days=None).nothing
