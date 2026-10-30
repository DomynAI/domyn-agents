"""Tests for domyn/agents/cli/main.py"""

import os
from pathlib import Path

from domyn.agents.cli.main import _load_credentials, _load_dotenv


def test_loads_a_dotenv_file_from_the_working_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MY_TEST_VAR", raising=False)
    (tmp_path / ".env").write_text("MY_TEST_VAR=from_dotenv\n")

    _load_dotenv()

    assert os.environ["MY_TEST_VAR"] == "from_dotenv"


def test_an_already_exported_var_is_not_overridden_by_dotenv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MY_TEST_VAR", "from_shell")
    (tmp_path / ".env").write_text("MY_TEST_VAR=from_dotenv\n")

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
    (tmp_path / ".env").write_text("MY_TEST_VAR=from_parent_dotenv\n")

    _load_dotenv()

    assert os.environ["MY_TEST_VAR"] == "from_parent_dotenv"


def test_credentials_file_entries_become_env_vars(tmp_path: Path, monkeypatch) -> None:
    from domyn.agents.interfaces.cli import config as cli_config

    monkeypatch.delenv("MY_CRED_VAR", raising=False)
    creds = tmp_path / "credentials"
    creds.write_text("MY_CRED_VAR=from_credentials_file\n")
    monkeypatch.setattr(cli_config, "credentials_path", lambda: creds)

    _load_credentials()

    assert os.environ["MY_CRED_VAR"] == "from_credentials_file"


def test_a_real_env_var_is_not_overridden_by_the_credentials_file(
    tmp_path: Path, monkeypatch
) -> None:
    from domyn.agents.interfaces.cli import config as cli_config

    monkeypatch.setenv("MY_CRED_VAR", "from_shell")
    creds = tmp_path / "credentials"
    creds.write_text("MY_CRED_VAR=from_credentials_file\n")
    monkeypatch.setattr(cli_config, "credentials_path", lambda: creds)

    _load_credentials()

    assert os.environ["MY_CRED_VAR"] == "from_shell"


def test_is_a_silent_noop_when_no_credentials_file_exists(tmp_path: Path, monkeypatch) -> None:
    from domyn.agents.interfaces.cli import config as cli_config

    monkeypatch.setattr(cli_config, "credentials_path", lambda: tmp_path / "credentials")

    _load_credentials()  # should not raise
