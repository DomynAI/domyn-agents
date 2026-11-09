"""What the CLI remembers between sessions (``state.yaml``).

The top-level conftest points `CliState.path` at a temporary file, so none of this
touches the developer's real ``~/.domyn``; the location tests undo that on purpose.
"""

from pathlib import Path

import pytest
import yaml

from domyn.agents.interfaces.cli import config as cli_config
from domyn.agents.interfaces.cli import state as cli_state
from domyn.agents.interfaces.cli.export import is_private
from domyn.agents.interfaces.cli.state import CliState
from tests.cli.test_repl_session import _repl


class TestTheFile:
    def test_defaults_when_there_is_no_file(self) -> None:
        assert CliState.load() == CliState()
        assert CliState.load().auto_approve is False

    def test_the_whole_state_round_trips(self) -> None:
        CliState(auto_approve=True, verbose=True).save()
        assert CliState.load() == CliState(auto_approve=True, verbose=True)
        CliState().save()
        assert CliState.load() == CliState()

    def test_a_key_this_version_does_not_know_survives_a_write(self) -> None:
        CliState.path().write_text("from_the_future: 7\n", encoding="utf-8")
        CliState(auto_approve=True).save()
        raw = yaml.safe_load(CliState.path().read_text(encoding="utf-8"))
        assert raw == {"from_the_future": 7, "verbose": False, "auto_approve": True}
        assert CliState.load() == CliState(auto_approve=True)

    def test_an_unreadable_or_invalid_file_is_the_defaults(self) -> None:
        path = CliState.path()
        path.write_text("auto_approve: [not, a, bool\n", encoding="utf-8")
        assert CliState.load() == CliState()
        path.write_text("auto_approve: banana\n", encoding="utf-8")
        assert CliState.load() == CliState()
        path.write_text("- a list\n", encoding="utf-8")
        assert CliState.load() == CliState()

    def test_owner_only(self) -> None:
        CliState(auto_approve=True).save()
        assert CliState.path().stat().st_mode & 0o777 == 0o600


class TestInTheSession:
    def test_a_toggle_is_remembered_whole(self) -> None:
        repl, _ = _repl()
        repl._prefs.auto_approve = True
        repl._prefs.save()
        assert CliState.load() == CliState(auto_approve=True)

    def test_a_new_session_starts_as_the_last_one_left_it(self) -> None:
        CliState(auto_approve=True, verbose=True).save()
        repl, _ = _repl()
        assert repl._prefs == CliState(auto_approve=True, verbose=True)

    def test_dash_v_sets_verbose_and_it_is_remembered_like_any_toggle(self) -> None:
        CliState(auto_approve=True).save()
        assert _repl(verbose=True)[0]._prefs.verbose is True
        assert CliState.load() == CliState(auto_approve=True, verbose=True)

    def test_no_flag_leaves_what_was_remembered(self) -> None:
        CliState(verbose=True).save()
        assert _repl()[0]._prefs.verbose is True

    def test_the_conversation_is_not_part_of_it(self) -> None:
        repl, _ = _repl()
        repl._state.turns = 3
        repl._prefs.save()
        assert "turns" not in yaml.safe_load(CliState.path().read_text(encoding="utf-8"))


class TestWhereItLives:
    def test_globally_when_no_project_is_selected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.undo()  # the real function, not the conftest's stand-in
        monkeypatch.setattr(cli_config, "_active_project", None)
        assert CliState.path() == cli_config.GLOBAL_DIR / "state.yaml"

    def test_in_the_project_folder_when_one_is_selected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.undo()
        monkeypatch.setattr(cli_config, "_active_project", tmp_path)
        assert CliState.path() == tmp_path / "state.yaml"

    def test_projects_do_not_share_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.undo()
        monkeypatch.setattr(cli_config, "_active_project", tmp_path / "a")
        CliState(auto_approve=True).save()
        monkeypatch.setattr(cli_config, "_active_project", tmp_path / "b")
        assert CliState.load().auto_approve is False


class TestExport:
    def test_the_state_is_never_exported(self, tmp_path: Path) -> None:
        assert is_private(tmp_path / "state.yaml")
