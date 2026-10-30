"""Fixtures for the whole suite."""

import contextlib
import os
import signal
from collections.abc import Iterator
from pathlib import Path

import pytest

from domyn.agents.llm import context_window

# A stand-in for the platform, so tests need no ``domyn-platform-sdk``: the ``platform`` fixture.
pytest_plugins = ["tests.fake_platform"]

# Where a background command started by the built-in tools keeps its pid and
# group (`tools/builtin/processes.py`), relative to the workspace it ran in.
_PROCESS_FILES = ".domyn/processes/*/*/run.pgid"


def _stop_background_commands(root: Path, pattern: str) -> int:
    """Kill the process group of every background command still running under
    ``root``; return how many were stopped.

    A background command outlives the call that started it by design, for up to
    a day (``DEFAULT_MAX_LIFETIME_S``), so every test that starts one and does not
    stop it leaves it running: one run of ``tests/tools`` left about two hundred,
    which then slowed the timing tests of the next run. The pid is checked alive
    and still in the recorded group first, as ``build_kill`` does, so a reused
    group id is never signalled.
    """
    stopped = 0
    for pgid_file in root.glob(pattern):
        with contextlib.suppress(OSError, ValueError):
            pgid = int(pgid_file.read_text().strip())
            pid = int((pgid_file.parent / "run.pid").read_text().strip())
            if pgid <= 1 or os.getpgid(pid) != pgid:
                continue
            os.killpg(pgid, signal.SIGKILL)
            stopped += 1
    return stopped


@pytest.fixture(autouse=True)
def _forget_context_windows() -> None:
    """Start every test with no context window remembered.

    `llm/context_window.py` remembers windows in two module-level dicts for as long as
    the process lives, and pytest runs all of a worker's tests in one process. Without
    this, a window one test taught (a rejection stating 32768 for an endpoint and model)
    would still be there in the next test that builds a provider for them, and that test
    would pass or fail depending on which of the two ran first.
    """
    context_window._answers.clear()
    context_window._stated.clear()


@pytest.fixture(autouse=True)
def _no_background_commands_left(tmp_path: Path) -> Iterator[None]:
    """After each test, stop the background commands it started.

    Tests run their commands in a workspace under their own ``tmp_path``, at
    most two levels down (``tmp_path`` itself, or a directory in it).

    Scanned from ``tmp_path``, not from the session's base directory as it was
    before: that one holds every test's directory, it grows all run long, and
    two globs over it after each test made the suite quadratic — 1250
    directories by the end of ``tests/cli`` alone, 138s for the suite against
    63s without the scan. A ``tmp_path`` is a handful of entries and belongs to
    the test that is ending. ``pytest_sessionfinish`` still sweeps the whole
    base directory once, for whatever is left deeper.
    """
    yield
    _stop_background_commands(tmp_path, f"**/{_PROCESS_FILES}")


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Whatever a test left deeper, or a test interrupted before its teardown:
    the whole temporary directory, once, as the session ends (Ctrl-C included)."""
    factory = getattr(session.config, "_tmp_path_factory", None)
    if factory is None:
        return
    with contextlib.suppress(OSError, ValueError):
        _stop_background_commands(factory.getbasetemp(), f"**/{_PROCESS_FILES}")


@pytest.fixture(autouse=True)
def _no_speech_model_downloads(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the speech model (670 MB) off the network and out of ``~/.domyn``.

    A ``LocalTranscriber`` built with its defaults downloads into the real
    ``~/.domyn/models`` the first time it is used, and a test reaching one by
    accident (a ``/voice`` in a REPL test) did exactly that once. Its models
    directory points at a temporary one, and Hugging Face is told it is offline,
    so such a test fails fast instead of downloading.
    """
    from domyn.agents.audio import local

    models: Path = tmp_path_factory.mktemp("models")
    monkeypatch.setattr(local, "_default_models_dir", lambda: models)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
