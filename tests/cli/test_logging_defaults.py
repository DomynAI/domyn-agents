"""Tests for how loud the CLI is.

A server and an interactive session want opposite defaults. For a server, INFO
*is* the interface: every logged request is an observable request. In a session
the same lines land between the user's question and the answer — inference URLs,
httpx status lines, runner bookkeeping — and push the reply they are waiting for
off the top of the screen.

So the split is deliberate, and these tests pin both halves of it plus the two
escape hatches: ``--log-level`` from outside, ``/verbose`` from within.
"""

import io
import logging

import pytest
from rich.console import Console
from typer.testing import CliRunner

from domyn.agents.cli.cli import app
from domyn.agents.interfaces.cli.repl import _set_log_level

runner = CliRunner()


@pytest.fixture(autouse=True)
def restore_root_level() -> object:
    """Log level is global state; leaking it would make other tests flaky."""
    root = logging.getLogger()
    previous = root.level
    yield
    root.setLevel(previous)


class TestDefaultsPerCommand:
    @pytest.mark.parametrize("command", ["run", "setup", "init", "skills", "skill", "export"])
    def test_interactive_commands_are_quiet(self, command: str) -> None:
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0
        assert logging.getLogger().level == logging.WARNING

    @pytest.mark.parametrize("command", ["serve", "connect", "expose"])
    def test_server_commands_stay_at_info(self, command: str) -> None:
        """A server has no other output: silencing it would blind the operator."""
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0
        assert logging.getLogger().level == logging.INFO


class TestTheStartupBanner:
    def test_it_is_suppressed_for_interactive_commands(self) -> None:
        """Bookkeeping for a service, clutter above a prompt."""
        result = runner.invoke(app, ["skills", "--help"])
        assert "Starting domyn CLI" not in result.output

    def test_it_is_printed_for_servers(self) -> None:
        result = runner.invoke(app, ["serve", "--help"])
        assert "Starting domyn CLI" in result.output


class TestExplicitOverride:
    def test_log_level_wins_over_the_quiet_default(self) -> None:
        runner.invoke(app, ["--log-level", "DEBUG", "skills", "--help"])
        assert logging.getLogger().level == logging.DEBUG

    def test_log_level_wins_over_the_server_default(self) -> None:
        runner.invoke(app, ["--log-level", "ERROR", "serve", "--help"])
        assert logging.getLogger().level == logging.ERROR

    def test_it_is_case_insensitive(self) -> None:
        runner.invoke(app, ["--log-level", "debug", "skills", "--help"])
        assert logging.getLogger().level == logging.DEBUG


class TestVerboseToggle:
    def test_it_turns_logs_on_and_off(self) -> None:
        _set_log_level(True)
        assert logging.getLogger().level == logging.INFO
        _set_log_level(False)
        assert logging.getLogger().level == logging.WARNING

    def test_it_moves_the_root_logger_not_just_ours(self) -> None:
        """Half the noise is httpx's. Lowering only ``domyn.agents`` would hide
        our lines and leave the HTTP ones, which is the worst of both."""
        _set_log_level(False)
        assert not logging.getLogger("httpx").isEnabledFor(logging.INFO)
        _set_log_level(True)
        assert logging.getLogger("httpx").isEnabledFor(logging.INFO)

    @pytest.mark.asyncio
    async def test_the_slash_command_flips_it(self) -> None:
        from tests.cli.test_repl_session import _repl

        _set_log_level(False)
        repl, _ = _repl("/verbose")
        await repl.run()
        assert repl._state.verbose
        assert logging.getLogger().level == logging.INFO

    def test_starting_verbose_turns_them_on_immediately(self) -> None:
        """``domyn run -v`` should not need a `/verbose` afterwards."""
        from tests.cli.test_repl_session import _agent

        _set_log_level(False)
        from domyn.agents.interfaces.cli import Theme
        from domyn.agents.interfaces.cli.repl import Repl

        Repl(
            _agent(),
            Console(file=io.StringIO()),
            Theme.build(allow_animation=False),
            verbose=True,
            enable_enhanced_input=False,
        )
        assert logging.getLogger().level == logging.INFO
