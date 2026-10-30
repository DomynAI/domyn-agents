"""The suite's own cleanup: background commands a test leaves running are stopped."""

import subprocess
import time
from pathlib import Path

from tests.conftest import _PROCESS_FILES, _stop_background_commands


def _start(workspace: Path, *, recorded_pgid: int | None = None) -> subprocess.Popen[bytes]:
    """A long command in its own group, with the files the built-in tools write."""
    process = subprocess.Popen(["sleep", "60"], start_new_session=True)
    run = workspace / ".domyn" / "processes" / "default" / "abc123"
    run.mkdir(parents=True)
    (run / "run.pid").write_text(f"{process.pid}\n")
    (run / "run.pgid").write_text(f"{recorded_pgid or process.pid}\n")
    return process


def _gone(process: subprocess.Popen[bytes]) -> bool:
    for _ in range(50):
        if process.poll() is not None:
            return True
        time.sleep(0.02)
    return False


def test_a_running_command_is_stopped(tmp_path: Path) -> None:
    process = _start(tmp_path / "test_x")
    assert _stop_background_commands(tmp_path, f"*/{_PROCESS_FILES}") == 1
    assert _gone(process)


def test_a_group_the_pid_is_no_longer_in_is_left_alone(tmp_path: Path) -> None:
    """A recorded group id the process is not in may have been reused by
    something else: nothing is signalled."""
    process = _start(tmp_path / "test_x", recorded_pgid=999_999)
    try:
        assert _stop_background_commands(tmp_path, f"*/{_PROCESS_FILES}") == 0
        assert process.poll() is None
    finally:
        process.kill()
        process.wait()


def test_a_command_that_already_ended_is_skipped(tmp_path: Path) -> None:
    process = _start(tmp_path / "test_x")
    process.kill()
    process.wait()
    assert _stop_background_commands(tmp_path, f"*/{_PROCESS_FILES}") == 0


def test_the_scan_reaches_the_workspace_root_and_one_level_down(tmp_path: Path) -> None:
    """The pattern the fixture uses: a command run in ``tmp_path`` itself, and one
    run in a directory inside it, are both found.

    The fixture scans the test's own ``tmp_path`` rather than the session's base
    directory, which holds every test's directory and grew all run long: two
    globs over it after each test made the suite quadratic. What it must still
    reach is what a test can start a command in.
    """
    at_root = _start(tmp_path)
    deeper = _start(tmp_path / "sub" / "deep")
    try:
        assert _stop_background_commands(tmp_path, f"**/{_PROCESS_FILES}") == 2
        assert _gone(at_root)
        assert _gone(deeper)
    finally:
        for process in (at_root, deeper):
            process.kill()
            process.wait()
