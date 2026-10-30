"""Ctrl-B in `domyn setup` goes back to the previous question asked (`run_setup`).

Driven through the numbered pickers with scripted answers, where a ``GoBack`` in the
script stands for Ctrl-B on that question: the key itself is tested in ``test_select.py``
and ``test_prompt.py``. The draft is restored as it was before the step gone back to,
and its questions open on the answers given there last time (`_Recall`).
"""

import io
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from domyn.agents.interfaces.cli import Theme, setup
from domyn.agents.interfaces.cli import config as cfg
from domyn.agents.interfaces.cli import select as select_mod
from domyn.agents.interfaces.cli.prompt import GoBack

# Captured before the CLI conftest replaces it for every wizard test.
_REAL_CUSTOM_SKILLS = setup._choose_custom_skills


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "GLOBAL_DIR", tmp_path / "home" / ".domyn")
    monkeypatch.setattr(cfg, "PROJECT_DIR", tmp_path / "nowhere" / ".domyn")
    monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)
    monkeypatch.setattr(setup, "_choose_builtin_skills", lambda console, theme, current: current)
    for name in ("OPENAI_API_KEY", "VLLM_API_KEY", "LLM_API_KEY"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


@pytest.fixture
def theme() -> Theme:
    return Theme.build(allow_animation=False)


@pytest.fixture
def console() -> MagicMock:
    return MagicMock()


def _run(console: MagicMock, theme: Theme, answers: list[Any], tmp_path: Path) -> Any:
    """A first setup; for openai the answers are provider, model, url, api key, max
    tokens and context window."""
    with patch("builtins.input", side_effect=answers):
        return setup.run_setup(console, theme, path=tmp_path / "config.yaml")


def _provider(result: Any) -> dict[str, Any]:
    provider: dict[str, Any] = result["llm_providers"][0]
    return provider


class TestGoingBack:
    def test_a_wrong_model_name_is_fixed_from_the_next_question(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        answers = ["openai", "wrong", GoBack(), "right", "", "", "", ""]

        result = _run(console, theme, answers, tmp_path)

        assert _provider(result)["model_name"] == "right"

    def test_the_question_opens_on_the_answer_given_there(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        # Enter after going back keeps what was typed, not the config's old value.
        answers = ["openai", "typed", GoBack(), "", "", "", "", ""]

        result = _run(console, theme, answers, tmp_path)

        assert _provider(result)["model_name"] == "typed"

    def test_back_on_the_first_question_asks_it_again(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        answers = [GoBack(), "openai", "m", "", "", "", ""]

        result = _run(console, theme, answers, tmp_path)

        assert _provider(result)["type"] == "openai"
        assert _provider(result)["model_name"] == "m"

    def test_going_back_says_so(self, console: MagicMock, theme: Theme, tmp_path: Path) -> None:
        _run(console, theme, ["openai", GoBack(), "openai", "m", "", "", "", ""], tmp_path)

        printed = [str(c.args[0]) for c in console.print.call_args_list if c.args]
        assert "  ← back" in printed

    def test_back_crosses_into_the_previous_section(
        self, console: MagicMock, theme: Theme, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Back on the instructions path, the first question after the model's, goes
        # back to the context window.
        instructions = MagicMock(side_effect=[GoBack(), None])
        monkeypatch.setattr(setup, "_choose_instructions", instructions)
        answers = ["openai", "m", "", "", "", "", "2000"]

        result = _run(console, theme, answers, tmp_path)

        assert _provider(result)["context_window"] == 2000
        assert instructions.call_count == 2

    def test_a_step_that_asked_nothing_is_passed_over(
        self, console: MagicMock, theme: Theme, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The key is found, so its step asks nothing: back on the max tokens goes
        # to the URL, rather than to the key step and straight forward again.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        answers = ["openai", "m", "http://u", GoBack(), "http://v", "", ""]

        result = _run(console, theme, answers, tmp_path)

        assert _provider(result)["url"] == "http://v"

    def test_back_to_a_key_pasted_in_this_run_offers_it_again(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        # Back on the max tokens lands on the key just pasted, offered masked, rather
        # than on a step that asks nothing and comes straight forward again.
        answers = ["openai", "m", "", "sk-first", GoBack(), "sk-second", "", ""]

        _run(console, theme, answers, tmp_path)

        assert cfg.read_credentials() == {"OPENAI_API_KEY": "sk-second"}

    def test_ctrl_c_still_aborts(self, console: MagicMock, theme: Theme, tmp_path: Path) -> None:
        answers = ["openai", "m", GoBack(), KeyboardInterrupt()]

        assert _run(console, theme, answers, tmp_path) is None
        assert not (tmp_path / "config.yaml").exists()


class TestAnotherProvider:
    def test_the_same_provider_keeps_what_was_typed(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        # Back from the URL to the model, then to the provider, Enter on both.
        answers = ["openai", "m1", GoBack(), GoBack(), "", "", "", "", "", ""]

        result = _run(console, theme, answers, tmp_path)

        assert _provider(result)["model_name"] == "m1"

    def test_another_provider_forgets_what_was_typed_for_the_first(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        answers = ["vllm", "llama", GoBack(), GoBack(), "openai", "", "", "", "", ""]

        result = _run(console, theme, answers, tmp_path)

        assert _provider(result)["type"] == "openai"
        assert _provider(result)["model_name"] == ""


class TestAStepAskedAgain:
    def test_a_folder_of_skills_is_not_listed_twice(
        self, console: MagicMock, theme: Theme, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The custom skills step adds to the list: asked again on the draft it left,
        # its folder would be listed twice. The draft from before it is restored.
        mine = tmp_path / "mine"
        (mine / "review").mkdir(parents=True)
        (mine / "review" / "SKILL.md").write_text(
            "---\nname: review\ndescription: Review code.\n---\n\nbody\n", encoding="utf-8"
        )
        monkeypatch.setattr(setup, "_choose_custom_skills", _REAL_CUSTOM_SKILLS)
        # Back on the autodiscovery question after it, once.
        local = MagicMock(side_effect=[GoBack(), None])
        monkeypatch.setattr(
            setup, "_choose_local_skills", lambda c, t, current, **k: local() or current
        )
        # Enter on the folder question asked again: the folder typed the first time.
        answers = ["openai", "m", "", "", "", "", str(mine), ""]

        result = _run(console, theme, answers, tmp_path)

        paths = result["agents"][0]["skills"]["source"]["paths"]
        assert paths == [str(mine / "review")]


class TestTheScreen:
    def test_counts_the_rows_written(self) -> None:
        screen = setup._Screen(io.StringIO())

        screen.write("one\ntwo\n")
        screen.write("three")

        assert screen.rows == 2

    def test_erases_up_to_a_mark(self) -> None:
        file = io.StringIO()
        screen = setup._Screen(file)
        screen.write("kept\n")
        mark = screen.rows
        screen.write("gone\ngone too\n")

        screen.erase_to(mark)

        assert file.getvalue().endswith("\x1b[2A\r\x1b[J")
        assert screen.rows == mark

    def test_nothing_printed_since_the_mark_writes_nothing(self) -> None:
        file = io.StringIO()
        screen = setup._Screen(file)
        screen.write("kept\n")

        screen.erase_to(screen.rows)

        assert file.getvalue() == "kept\n"


class TestOnATerminal:
    def test_going_back_erases_up_to_the_question_gone_back_to(
        self, theme: Theme, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The model step prints one row (the model names) before its question; back on
        # the URL erases it, rather than saying "back" below.
        monkeypatch.setattr(setup, "interactive_terminal", lambda: True)
        out = io.StringIO()
        console = Console(file=out, force_terminal=True)
        answers = ["openai", "m", GoBack(), "m2", "", "", "", ""]

        with patch("builtins.input", side_effect=answers):
            result = setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        assert _provider(result)["model_name"] == "m2"
        assert "\x1b[1A\r\x1b[J" in out.getvalue()
        assert "\u2190 back" not in out.getvalue()
        # The caller's console is left as it was.
        assert console.file is out
