"""The daily check for a new version shown by `domyn run`, and `/update-check`."""

import json
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import respx

import domyn.agents
from domyn.agents.interfaces.cli import config as cli_config
from domyn.agents.interfaces.cli import update_check

# Taken before the autouse fixture replaces them for every test.
_REAL_LATEST_VERSION = update_check.latest_version
_REAL_UPDATE_CHECKS_PATH = update_check._update_checks_path


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Installed 1.0.0, PyPI answering 2.0.0; the list counts the requests."""
    made: list[int] = []

    def latest() -> str:
        made.append(1)
        return "2.0.0"

    monkeypatch.setattr(domyn.agents, "__version__", "1.0.0")
    monkeypatch.setattr(update_check, "latest_version", latest)
    monkeypatch.setattr(update_check, "_update_checks_path", _REAL_UPDATE_CHECKS_PATH)
    monkeypatch.setattr(sys, "prefix", "/nowhere/python")
    # The request runs where the thread would, but at once, so a test can read its answer.
    monkeypatch.setattr(update_check, "threading", SimpleNamespace(Thread=_RunNow))
    return made


class _RunNow:
    def __init__(self, target: Callable[[], None], daemon: bool) -> None:
        self._target = target

    def start(self) -> None:
        self._target()


def _update_checks() -> dict:
    return json.loads(update_check._update_checks_path().read_text())


class TestTheNotice:
    def test_names_a_newer_version_from_the_next_run(self, calls: list[int]) -> None:
        assert update_check.update_notice() is None
        notice = update_check.update_notice()
        assert notice is not None
        assert "2.0.0" in notice
        assert 'pip install -U "domyn[cli]"' in notice

    def test_is_kept_in_the_global_domyn_folder(self, calls: list[int]) -> None:
        update_check.update_notice()
        assert (cli_config.GLOBAL_DIR / "update-check.json").is_file()

    @pytest.mark.parametrize("installed", ["2.0.0", "2.1.0", "2.0.1a3"])
    def test_says_nothing_when_up_to_date(
        self, calls: list[int], monkeypatch: pytest.MonkeyPatch, installed: str
    ) -> None:
        monkeypatch.setattr(domyn.agents, "__version__", installed)
        assert update_check.update_notice() is None

    @pytest.mark.parametrize("installed", ["1.0.1.dev3+gabc", "unknown"])
    def test_a_development_build_is_never_checked(
        self, calls: list[int], monkeypatch: pytest.MonkeyPatch, installed: str
    ) -> None:
        monkeypatch.setattr(domyn.agents, "__version__", installed)
        assert update_check.update_notice() is None
        assert calls == []


class TestOnceADay:
    def test_a_fresh_answer_is_reused(self, calls: list[int]) -> None:
        update_check.update_notice()
        assert update_check.update_notice() is not None
        assert calls == [1]

    def test_a_day_old_answer_is_asked_again(self, calls: list[int]) -> None:
        update_check.update_notice()
        update_checks = _update_checks()
        update_checks["checked_at"] = time.time() - update_check.CHECK_INTERVAL_SECONDS - 1
        update_check._write_update_checks(update_checks)
        update_check.update_notice()
        assert calls == [1, 1]

    def test_a_failed_check_waits_a_day_too(
        self, calls: list[int], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(update_check, "latest_version", lambda: calls.append(1))
        assert update_check.update_notice() is None
        assert update_check.update_notice() is None
        assert calls == [1]

    def test_startup_does_not_wait_for_the_answer(
        self, calls: list[int], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(update_check, "threading", threading)
        answer = threading.Event()

        def slow() -> str:
            answer.wait(5)
            return "2.0.0"

        monkeypatch.setattr(update_check, "latest_version", slow)
        started = time.monotonic()
        assert update_check.update_notice() is None
        assert time.monotonic() - started < 1

        answer.set()
        deadline = time.monotonic() + 5
        while not update_check._update_checks_path().is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert update_check.update_notice() is not None


class TestTurningItOff:
    def test_off_asks_nothing_and_says_nothing(self, calls: list[int]) -> None:
        update_check.set_enabled(False)
        assert not update_check.is_enabled()
        assert update_check.update_notice() is None
        assert calls == []

    def test_on_again(self, calls: list[int]) -> None:
        update_check.set_enabled(False)
        update_check.set_enabled(True)
        update_check.update_notice()
        assert update_check.update_notice() is not None

    @pytest.mark.asyncio
    async def test_the_slash_command(self, calls: list[int]) -> None:
        from tests.cli.test_repl_session import _repl

        repl, _ = _repl("/update-check off")
        await repl.run()
        assert not update_check.is_enabled()

        repl, _ = _repl("/update-check on")
        await repl.run()
        assert update_check.is_enabled()


class TestTheCommand:
    def test_standalone(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / ".domyn-runtime").write_text("domyn 1.0.0, installed by install.sh")
        monkeypatch.setattr(sys, "prefix", str(tmp_path / "python"))
        monkeypatch.setattr(sys, "platform", "darwin")
        assert update_check.update_command().endswith("install.sh | sh")
        monkeypatch.setattr(sys, "platform", "win32")
        assert update_check.update_command().endswith("install.ps1 | iex")

    def test_uv_tool(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "uv-receipt.toml").write_text("")
        monkeypatch.setattr(sys, "prefix", str(tmp_path))
        assert update_check.update_command() == "uv tool upgrade domyn"


class TestAskingPyPI:
    @respx.mock
    def test_reads_the_latest_release(self) -> None:
        respx.get(update_check.PYPI_URL).mock(
            return_value=httpx.Response(200, json={"info": {"version": "3.1.0"}})
        )
        assert _REAL_LATEST_VERSION() == "3.1.0"

    @respx.mock
    def test_a_failure_is_no_answer(self) -> None:
        respx.get(update_check.PYPI_URL).mock(side_effect=httpx.ConnectError("offline"))
        assert _REAL_LATEST_VERSION() is None
