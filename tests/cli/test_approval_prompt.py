"""The approval prompt is an arrow-key choice on a terminal, not a typed Y/n."""

import io
from typing import Any
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from domyn.agents.core import (
    ApprovalType,
    BaseEvent,
    ExecutionEventType,
    FeedbackAction,
    FeedbackParameters,
)
from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli import select as select_mod
from domyn.agents.interfaces.cli.approval import begin_approval, prompt_for_approval


def _event() -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.TOOL_NEEDS_FEEDBACK,
        author="Assistant",
        metadata={
            "approval_subject": "$ python a.py",
            "approval_scopes": ["conversation", "user"],
            "approval_pattern": "$ python *",
        },
        action=FeedbackAction(
            name="run_command",
            parameters=[],
            approval_type=ApprovalType.USER_CONFIRM,
        ),
    )


@pytest.fixture
def picked(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Stands in for the arrow picker; each test sets what it returns."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(select_mod, "_interactive", lambda theme: True)
    monkeypatch.setattr("builtins.input", lambda: pytest.fail("typed input on a terminal"))
    return calls


def _arrows(monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, Any]], answer: Any) -> None:
    def fake(theme: Any, label: str, options: list[str], start: int, descriptions: Any) -> Any:
        calls.append({"label": label, "options": options, "start": start})
        return answer(options) if callable(answer) else answer

    monkeypatch.setattr(select_mod, "_select_with_arrows", fake)


def _ask() -> Any:
    return prompt_for_approval(MagicMock(), Theme.build(allow_animation=False), _event()).action


class TestTheArrowPicker:
    def test_offers_every_scope_and_no_in_one_list(
        self, picked: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _arrows(monkeypatch, picked, lambda options: options[0])
        action = _ask()

        options = picked[0]["options"]
        assert options[0] == "yes, just this once"
        assert any("for the rest of this conversation" in o for o in options)
        assert any("anything matching  $ python *" in o for o in options)
        assert options[-1] == "no"
        assert action.approve is True
        assert (action.remember_scope, action.remember_pattern) == (None, None)

    def test_picking_a_scope_remembers_it(
        self, picked: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _arrows(monkeypatch, picked, lambda options: options[2])
        action = _ask()
        assert action.approve is True
        assert action.remember_scope == "user"

    def test_picking_no_denies(
        self, picked: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _arrows(monkeypatch, picked, "no")
        action = _ask()
        assert action.approve is False
        assert action.remember_scope is None

    def test_cancelling_the_picker_denies(
        self, picked: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _arrows(monkeypatch, picked, None)
        assert _ask().approve is False


class TestThePipeFallback:
    def test_y_and_n_still_answer_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)
        monkeypatch.setattr("builtins.input", lambda: "n")
        assert _ask().approve is False
        monkeypatch.setattr("builtins.input", lambda: "y")
        assert _ask().approve is True

    def test_a_stuck_stdin_gives_up_denying(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)
        monkeypatch.setattr("builtins.input", lambda: "maybe")
        assert _ask().approve is False


class TestTheCallReadsAsWritten:
    """The question is about a call, so it shows that call: once, as it will run.
    Not a ``repr`` with every backslash doubled, not something that drives the
    terminal instead of being printed, not the same text three times."""

    @staticmethod
    def _shown(tool: str, subject: str, *, width: int = 500, **params: object) -> str:
        buffer = io.StringIO()
        event = BaseEvent(
            event_type=ExecutionEventType.TOOL_NEEDS_FEEDBACK,
            author="Assistant",
            metadata={"approval_subject": subject, "approval_scopes": ["conversation"]},
            action=FeedbackAction(
                name=tool,
                approval_type=ApprovalType.USER_CONFIRM,
                parameters=[
                    FeedbackParameters(name=name, value=value, type=type(value).__name__)
                    for name, value in params.items()
                ],
            ),
        )
        # rich's default console, markup on: an embedder may hand in one like it.
        begin_approval(Console(file=buffer, width=width), Theme.build(allow_animation=False), event)
        return "\n".join(line.rstrip() for line in buffer.getvalue().split("\n"))

    def test_a_command_is_shown_once_as_it_will_run(self) -> None:
        command = (
            r"""grep -o "Stdio\|Http" a.js | grep "[/]" | sed 's:a:b:' && echo "it's \"done\"" -n"""
        )
        shown = self._shown("run_command", f"$ {command}", command=command)
        assert f"approval required: $ {command}\n" in shown
        assert shown.count(command) == 1, shown

    def test_what_a_terminal_would_act_on_is_shown_not_sent(self) -> None:
        command = "echo ok\x1b[1A\x1b[2Krm -rf ~ \u202eabc"
        shown = self._shown("run_command", f"$ {command}", command=command)
        assert "\x1b" not in shown
        assert "\u202e" not in shown
        assert r"echo ok\x1b[1A\x1b[2Krm -rf ~ \u202eabc" in shown

    def test_a_command_on_several_lines_is_shown_once_with_its_lines(self) -> None:
        """Its subject is those lines flattened into one, which reads as one command."""
        shown = self._shown(
            "run_command", "$ cd /tmp rm -rf build make", command="cd /tmp\nrm -rf build\nmake"
        )
        assert "approval required: run_command\n    cd /tmp\n    rm -rf build\n    make\n" in shown
        assert "cd /tmp rm -rf build make" not in shown

    def test_a_command_too_long_for_the_heading_is_shown_whole_under_it(self) -> None:
        command = " ".join(f"step{n}" for n in range(30))
        shown = self._shown("run_command", f"$ {command}", width=60, command=command)
        heading, *body = shown.strip("\n").split("\n")
        assert heading.endswith("approval required: run_command")
        assert all(line.startswith("    ") for line in body), shown
        assert " ".join(line.strip() for line in body) == command

    def test_an_edit_is_shown_as_a_diff(self) -> None:
        old = "def load(path):\n    return open(path).read()\n"
        new = "def load(path: str) -> str:\n    return open(path).read()\n"
        shown = self._shown("edit_file", "edit_file path=a.py", path="a.py", old=old, new=new)
        assert "approval required: edit_file path=a.py\n" in shown
        assert "    - def load(path):\n    + def load(path: str) -> str:\n" in shown
        assert "\n          return open(path).read()\n" in shown, "a kept line is unmarked"
        assert shown.count("a.py") == 1, shown
