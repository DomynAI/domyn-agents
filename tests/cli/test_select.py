"""Tests for the arrow-key option selector."""

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli import select as select_mod
from domyn.agents.interfaces.cli.prompt import GoBack, back_key
from domyn.agents.interfaces.cli.select import select

OPTIONS = ["openai", "anthropic", "custom"]


@pytest.fixture
def theme() -> Theme:
    return Theme.build(allow_animation=False)


@pytest.fixture(autouse=True)
def force_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests run without a TTY, which is exactly the fallback path."""
    monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)


class TestNumberedFallback:
    """The wizard must stay drivable from a pipe: scripted installs depend on it."""

    def test_number_picks_the_option(self, theme: Theme) -> None:
        with patch("builtins.input", return_value="2"):
            assert select(MagicMock(), theme, "provider", OPTIONS) == "anthropic"

    def test_name_picks_the_option(self, theme: Theme) -> None:
        with patch("builtins.input", return_value="custom"):
            assert select(MagicMock(), theme, "provider", OPTIONS) == "custom"

    def test_empty_keeps_the_current_value(self, theme: Theme) -> None:
        with patch("builtins.input", return_value=""):
            assert select(MagicMock(), theme, "provider", OPTIONS, current="custom") == "custom"

    def test_empty_without_current_takes_the_first(self, theme: Theme) -> None:
        with patch("builtins.input", return_value=""):
            assert select(MagicMock(), theme, "provider", OPTIONS) == "openai"

    def test_invalid_input_reprompts(self, theme: Theme) -> None:
        with patch("builtins.input", side_effect=["nope", "99", "1"]):
            assert select(MagicMock(), theme, "provider", OPTIONS) == "openai"

    def test_abort_returns_none(self, theme: Theme) -> None:
        with patch("builtins.input", side_effect=EOFError):
            assert select(MagicMock(), theme, "provider", OPTIONS) is None

    def test_descriptions_are_shown(self, theme: Theme) -> None:
        console = MagicMock()
        with patch("builtins.input", return_value="1"):
            select(
                console,
                theme,
                "provider",
                OPTIONS,
                descriptions={"openai": "OpenAI API"},
            )
        printed = " ".join(str(c.args[0]) for c in console.print.call_args_list if c.args)
        assert "OpenAI API" in printed

    def test_empty_option_list_returns_none(self, theme: Theme) -> None:
        assert select(MagicMock(), theme, "provider", []) is None


class TestInteractiveGate:
    """Arrow mode must never engage where it cannot work."""

    def test_requires_a_tty(self, theme: Theme, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.undo()  # restore the real _interactive
        fake_stdin: Any = type("S", (), {"isatty": lambda self: False})()
        monkeypatch.setattr("sys.stdin", fake_stdin)
        assert select_mod._interactive(theme) is False


class TestHelpAlias:
    """`-h` is what most people type; it must work everywhere `--help` does."""

    @pytest.mark.parametrize("flag", ["-h", "--help"])
    @pytest.mark.parametrize("command", [[], ["run"], ["setup"], ["skills"]])
    def test_help_flags(self, flag: str, command: list[str]) -> None:
        from typer.testing import CliRunner

        from domyn.agents.cli.cli import app

        result = CliRunner().invoke(app, [*command, flag])
        assert result.exit_code == 0
        assert "Usage:" in result.output


class TestInsideARunningLoop:
    """``/model`` opens the picker from inside the REPL's event loop.

    Same trap as the input editor: prompt_toolkit's synchronous ``run()`` calls
    ``asyncio.run()``, which raises when a loop is already running. The picker
    sidesteps it with ``in_thread=True`` — prompt_toolkit's own worker thread and
    loop — and that only stays true if something checks it asks for one.
    """

    @pytest.mark.asyncio
    async def test_the_arrow_picker_asks_for_its_own_thread(
        self, theme: Theme, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import prompt_toolkit.application

        started: list[bool] = []

        class StubApplication:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def run(self, *, in_thread: bool = False) -> None:
                started.append(in_thread)

            def exit(self) -> None:  # pragma: no cover - not reached by the stub
                pass

        monkeypatch.setattr(prompt_toolkit.application, "Application", StubApplication)
        monkeypatch.setattr(select_mod, "_interactive", lambda theme: True)

        chosen = select(MagicMock(), theme, "provider", OPTIONS, current="anthropic")

        assert started == [True], "the picker must run on its own thread, off the outer loop"
        assert chosen == "anthropic", "it should return the highlighted option"


def test_the_selected_row_is_a_reversed_bar() -> None:
    """A cyan bar with dark text, not cyan text: set by our own style, not inherited
    from prompt_toolkit's built-in rule for a class that happens to be `selected`."""
    from prompt_toolkit.styles import Attrs, Style

    from domyn.agents.interfaces.cli.select import CHOICE_STYLES

    attrs: Attrs = Style.from_dict(CHOICE_STYLES).get_attrs_for_style_str("class:choice-selected")
    assert attrs.reverse and attrs.bold


class TestDigitsInTheArrowPicker:
    """A digit moves the highlight and Enter confirms, as in the session's editor.

    Driven through a real ``Application`` on a pipe: what is under test is the key
    bindings, which a stub has none of.
    """

    def _pick(self, theme: Theme, keys: str, options: list[str]) -> str | None:
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            pipe.send_text(keys)
            return select_mod._select_with_arrows(theme, "pick", options, 0, {})

    def test_a_digit_then_enter_picks_that_row(self, theme: Theme) -> None:
        assert self._pick(theme, "3\r", OPTIONS) == "custom"

    def test_two_digits_reach_a_row_past_nine(self, theme: Theme) -> None:
        options = [f"model-{n}" for n in range(1, 13)]
        assert self._pick(theme, "12\r", options) == "model-12"

    def test_a_digit_naming_no_row_leaves_the_highlight(self, theme: Theme) -> None:
        assert self._pick(theme, "\x1b[B9\r", OPTIONS) == "anthropic"


class TestGoingBackInTheArrowPickers:
    """Inside ``back_key`` (the setup wizard) Ctrl-B goes back; Esc cancels everywhere."""

    def _run(self, keys: str, picker: Any) -> Any:
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            pipe.send_text(keys)
            return picker()

    def _select(self, theme: Theme) -> Any:
        return lambda: select_mod._select_with_arrows(theme, "pick", OPTIONS, 0, {})

    def _multiselect(self, theme: Theme) -> Any:
        ticks = select_mod._TickList(OPTIONS, None, [])
        return lambda: select_mod._multiselect_with_arrows(theme, "tick", ticks, {})

    def test_ctrl_b_goes_back_from_a_pick_inside_back_key(self, theme: Theme) -> None:
        with back_key(), pytest.raises(GoBack):
            self._run("\x02", self._select(theme))

    def test_ctrl_b_does_nothing_to_a_pick_elsewhere(self, theme: Theme) -> None:
        assert self._run("\x02\r", self._select(theme)) == "openai"

    def test_esc_still_cancels_a_pick_inside_back_key(self, theme: Theme) -> None:
        with back_key():
            assert self._run("\x1b", self._select(theme)) is None

    def test_ctrl_c_still_cancels_a_pick_inside_back_key(self, theme: Theme) -> None:
        with back_key():
            assert self._run("\x03", self._select(theme)) is None

    def test_ctrl_b_goes_back_from_a_tick_list_inside_back_key(self, theme: Theme) -> None:
        with back_key(), pytest.raises(GoBack):
            self._run("\x02", self._multiselect(theme))

    def test_esc_still_cancels_a_tick_list_inside_back_key(self, theme: Theme) -> None:
        with back_key():
            assert self._run("\x1b", self._multiselect(theme)) is None

    def test_a_pick_is_printed_back_inside_back_key(
        self, theme: Theme, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The picker erases itself when it ends, so what stays is printed through the console.
        monkeypatch.setattr(select_mod, "_interactive", lambda theme: True)
        console = MagicMock()

        with back_key():
            chosen = self._run("\x1b[B\r", lambda: select(console, theme, "Provider", OPTIONS))

        assert chosen == "anthropic"
        assert str(console.print.call_args.args[0]) == "  Provider: anthropic"

    def test_a_pick_is_not_printed_back_elsewhere(
        self, theme: Theme, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(select_mod, "_interactive", lambda theme: True)
        console = MagicMock()

        self._run("\r", lambda: select(console, theme, "Provider", OPTIONS))

        console.print.assert_not_called()

    def test_the_hint_names_ctrl_b_only_where_it_goes_back(self, theme: Theme) -> None:
        ticks = select_mod._TickList(OPTIONS, None, [])

        def hint(back: bool) -> str:
            return select_mod.multi_choice_rows(theme, "tick", ticks, 0, back=back)[-1][1]

        assert "ctrl-b back" in hint(True)
        assert "ctrl-b back" not in hint(False)


def test_rows_are_numbered(theme: Theme) -> None:
    rows = select_mod.choice_rows(theme, "pick", OPTIONS, 0, ["enter select"])
    texts = [text.rstrip() for _, text in rows]
    assert any(text.endswith("1. openai") for text in texts)
    assert any(text.endswith("3. custom") for text in texts)
