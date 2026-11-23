"""Tests for domyn/agents/cli/main.py"""

import os
from pathlib import Path

from domyn.agents.cli.main import _load_credentials, _load_dotenv


def test_loads_a_dotenv_file_from_the_working_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MY_TEST_VAR", raising=False)
    (tmp_path / ".env").write_text("MY_TEST_VAR=from_dotenv\n", encoding="utf-8")

    _load_dotenv()

    assert os.environ["MY_TEST_VAR"] == "from_dotenv"


def test_an_already_exported_var_is_not_overridden_by_dotenv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MY_TEST_VAR", "from_shell")
    (tmp_path / ".env").write_text("MY_TEST_VAR=from_dotenv\n", encoding="utf-8")

    _load_dotenv()

    assert os.environ["MY_TEST_VAR"] == "from_shell"


def test_is_a_silent_noop_when_no_dotenv_file_is_found(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    _load_dotenv()  # should not raise


def test_finds_a_dotenv_file_in_a_parent_directory(tmp_path: Path, monkeypatch) -> None:
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    monkeypatch.chdir(subdir)
    monkeypatch.delenv("MY_TEST_VAR", raising=False)
    (tmp_path / ".env").write_text("MY_TEST_VAR=from_parent_dotenv\n", encoding="utf-8")

    _load_dotenv()

    assert os.environ["MY_TEST_VAR"] == "from_parent_dotenv"


def test_credentials_file_entries_become_env_vars(tmp_path: Path, monkeypatch) -> None:
    from domyn.agents.interfaces.cli import config as cli_config

    monkeypatch.delenv("MY_CRED_KEY", raising=False)
    creds = tmp_path / "credentials"
    creds.write_text("MY_CRED_KEY=from_credentials_file\n", encoding="utf-8")
    monkeypatch.setattr(cli_config, "credentials_path", lambda: creds)

    _load_credentials()

    assert os.environ["MY_CRED_KEY"] == "from_credentials_file"


def test_a_real_env_var_is_not_overridden_by_the_credentials_file(
    tmp_path: Path, monkeypatch
) -> None:
    from domyn.agents.interfaces.cli import config as cli_config

    monkeypatch.setenv("MY_CRED_KEY", "from_shell")
    creds = tmp_path / "credentials"
    creds.write_text("MY_CRED_KEY=from_credentials_file\n", encoding="utf-8")
    monkeypatch.setattr(cli_config, "credentials_path", lambda: creds)

    _load_credentials()

    assert os.environ["MY_CRED_KEY"] == "from_shell"


def test_is_a_silent_noop_when_no_credentials_file_exists(tmp_path: Path, monkeypatch) -> None:
    from domyn.agents.interfaces.cli import config as cli_config

    monkeypatch.setattr(cli_config, "credentials_path", lambda: tmp_path / "credentials")

    _load_credentials()  # should not raise


def test_importing_the_entry_point_loads_no_command() -> None:
    """The commands make their rich Consoles when imported, and on Windows rich keeps
    the sixteen colours of the console API if the console did not yet take escape
    sequences: `cli()` turns those on first, so importing this module must not."""
    import subprocess
    import sys

    code = (
        "import sys, domyn.agents.cli.main; "
        "print([m for m in sys.modules if m.startswith('domyn.agents.cli.commands')])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "[]"


def test_cli_turns_escape_sequences_on_before_the_commands(monkeypatch) -> None:
    """In this order: the commands' rich Consoles are made when they are imported."""
    import domyn.agents.cli.cli as commands
    from domyn.agents.cli import main
    from domyn.agents.interfaces.cli import theme

    calls: list[str] = []
    monkeypatch.setattr(theme, "enable_escape_sequences", lambda: calls.append("escapes"))
    monkeypatch.setattr(main, "_load_dotenv", lambda: calls.append("dotenv"))
    monkeypatch.setattr(main, "_load_credentials", lambda: calls.append("credentials"))
    monkeypatch.setattr(commands, "app", lambda: calls.append("app"))

    main.cli()

    assert calls == ["escapes", "dotenv", "credentials", "app"]
