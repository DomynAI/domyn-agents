"""What the session shows when a decision model changes something.

The line exists for one reason: when a model approves a call, the approval
panel never appears, and without a trace the tool simply runs. "It did not ask
me" has to be readable as a decision someone can inspect, not as the gate
having quietly stopped working.
"""

import io
from typing import ClassVar

import pytest
from rich.console import Console

from domyn.agents import BaseEvent, ExecutionEventType
from domyn.agents.interfaces.cli import Capabilities, Theme
from domyn.agents.interfaces.cli.renderer import EventRenderer


def render(metadata: dict, *, verbose: bool = False) -> str:
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False)
    renderer = EventRenderer(console, Theme.build(allow_animation=False), verbose=verbose)
    renderer.render(
        BaseEvent(event_type=ExecutionEventType.DECISION_MADE, author="A", metadata=metadata)
    )
    return buffer.getvalue()


def render_coloured(metadata: dict, *, verbose: bool = False) -> str:
    """The same line on a terminal that takes colour, ANSI codes and all.

    ``Theme.build`` reads ``TERM`` and ``NO_COLOR`` from the environment, and a
    test run is usually ``TERM=dumb``, where every style resolves to "". The
    capabilities are therefore forced here rather than inherited: what is under
    test is the hue, and a suite that goes green by having no colour at all
    would not be testing it.
    """
    base = Theme.build(allow_animation=False)
    theme = type(base)(
        capabilities=Capabilities(color=True, unicode=True, animation=False),
        symbols=type(base.symbols)(unicode=True),
        styles=base.styles,
    )
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=True, color_system="truecolor")
    renderer = EventRenderer(console, theme, verbose=verbose)
    renderer.render(
        BaseEvent(event_type=ExecutionEventType.DECISION_MADE, author="A", metadata=metadata)
    )
    return buffer.getvalue()


class TestItSpeaksWhenItActed:
    def test_an_approval_it_granted_is_on_screen(self):
        output = render(
            {
                "use_case": "approvals",
                "mode": "active",
                "acted": True,
                "known": True,
                "value": "allow",
                "confidence": 0.95,
                "latency_ms": 147,
            }
        )
        assert "decision[approvals]" in output
        assert "allow" in output
        assert "0.95" in output
        assert "147 ms" in output

    def test_the_confidence_is_always_there(self):
        """The number is the mechanism. A reader who cannot see it cannot tell
        a model that abstained from one that was never asked."""
        assert "0.79" in render(
            {
                "use_case": "approvals",
                "mode": "active",
                "acted": True,
                "known": True,
                "value": "allow",
                "confidence": 0.79,
            }
        )

    def test_a_summary_is_used_when_there_is_no_single_verdict(self):
        output = render(
            {
                "use_case": "tools",
                "mode": "active",
                "acted": True,
                "summary": "3 tool(s) hidden this turn",
                "latency_ms": 200,
            }
        )
        assert "3 tool(s) hidden this turn" in output
        assert "decision[tools]" in output


class TestItStaysQuietOtherwise:
    def test_a_decision_that_changed_nothing_is_not_narrated(self):
        """Otherwise the session reports a background service every turn, which
        is how a trace becomes wallpaper."""
        assert (
            render(
                {
                    "use_case": "approvals",
                    "mode": "active",
                    "acted": False,
                    "known": True,
                    "value": "ask",
                    "confidence": 0.9,
                }
            ).strip()
            == ""
        )

    def test_but_verbose_shows_it(self):
        output = render(
            {
                "use_case": "approvals",
                "mode": "active",
                "acted": False,
                "known": True,
                "value": "ask",
                "confidence": 0.9,
            },
            verbose=True,
        )
        assert "decision[approvals]" in output

    def test_shadow_says_it_applied_nothing(self):
        output = render(
            {
                "use_case": "approvals",
                "mode": "shadow",
                "acted": False,
                "known": True,
                "value": "allow",
                "confidence": 0.99,
            },
            verbose=True,
        )
        assert "shadow" in output
        assert "not applied" in output


class TestItDoesNotCrashOnWhatItIsGiven:
    def test_an_unanswered_decision_says_so(self):
        output = render(
            {
                "use_case": "skills",
                "mode": "active",
                "acted": True,
                "known": False,
                "reason": "decision provider timed out",
            },
            verbose=True,
        )
        assert "no answer" in output
        assert "timed out" in output

    def test_metadata_with_nothing_in_it(self):
        assert render({"acted": True}) is not None

    @pytest.mark.parametrize(
        "confidence", [None, "high", float("nan")], ids=["none", "text", "nan"]
    )
    def test_a_confidence_that_is_not_a_number(self, confidence):
        output = render(
            {
                "use_case": "x",
                "mode": "active",
                "acted": True,
                "known": True,
                "value": "allow",
                "confidence": confidence,
            }
        )
        assert "decision[x]" in output

    def test_a_summary_containing_markup_is_printed_as_written(self):
        """The trace is printed with markup off; a tool named `[x]` must not
        vanish or crash the console."""
        output = render(
            {
                "use_case": "tools",
                "mode": "active",
                "acted": True,
                "summary": "hid [bold]write_file[/]",
            }
        )
        assert "[bold]write_file[/]" in output


class TestConfigShowsIt:
    """A reader asking "why did it not ask me" needs the mode and the threshold
    together, so /config carries both or it is decorative."""

    def _described(self, raw: dict):
        from domyn.agents.interfaces.cli.config import describe_configuration_v1

        return describe_configuration_v1(raw)

    def _section(self, raw: dict):
        for section in self._described(raw).sections:
            if section.title == "Decision Models":
                return section
        return None

    def _base(self, **extra) -> dict:
        return {
            "version": "1.0",
            "entrypoint": "A",
            "llm_providers": [{"name": "p", "type": "vllm", "model_name": "m"}],
            "agents": [
                {
                    "name": "A",
                    "description": "d",
                    "instruction": "i",
                    "llm_provider": "p",
                    **extra,
                }
            ],
        }

    def test_nothing_configured_shows_no_section(self):
        assert self._section(self._base()) is None

    def test_the_provider_and_the_policy_are_both_shown(self):
        raw = self._base(
            decisions={
                "provider": "local_clef",
                "approvals": {"mode": "active", "min_confidence": 0.85},
            }
        )
        raw["decision_providers"] = [
            {
                "name": "local_clef",
                "type": "system_one",
                "model": "Cloudflare/clef",
                "base_url": "http://localhost:28080",
            }
        ]
        section = self._section(raw)
        assert section is not None
        rendered = str(section.items)
        assert "Cloudflare/clef" in rendered
        assert "active" in rendered
        assert "0.85" in rendered

    def test_a_key_is_reported_as_present_never_shown(self):
        raw = self._base(decisions={"provider": "jev", "approvals": {"mode": "off"}})
        raw["decision_providers"] = [
            {"name": "jev", "type": "system_one", "api_key": "sk-super-secret"}
        ]
        rendered = str(self._section(raw).items)
        assert "sk-super-secret" not in rendered
        assert "set" in rendered

    def test_on_unknown_is_visible_because_it_is_a_safety_choice(self):
        raw = self._base(
            decisions={
                "provider": "p",
                "sensitivity": {"mode": "active", "on_unknown": "allow"},
            }
        )
        rendered = str(self._section(raw).items)
        assert "on unknown: allow" in rendered


class TestTheVerdictIsVisibleWithoutReadingIt:
    """The line is the only record that the gate behaved differently.

    Reported as being unreadable: dim grey, in a trace where everything is dim
    grey. A decision that *acted* is not background — it is the answer to "why
    didn't it ask me" — so the two verdicts that act carry a hue, and the same
    green/red a diff uses for the same reason.
    """

    ALLOW: ClassVar[dict] = {
        "use_case": "approvals",
        "mode": "active",
        "acted": True,
        "known": True,
        "value": "allow",
        "confidence": 0.95,
        "latency_ms": 62,
    }
    DENY: ClassVar[dict] = {**ALLOW, "value": "deny", "confidence": 0.89}

    def test_an_allow_is_green(self):
        assert "\x1b[32m" in render_coloured(self.ALLOW)

    def test_a_deny_is_red(self):
        assert "\x1b[31m" in render_coloured(self.DENY)

    def test_the_two_verdicts_do_not_share_a_colour(self):
        assert render_coloured(self.ALLOW) != render_coloured(self.DENY)

    def test_a_shadow_run_stays_dim(self):
        """It changed nothing about the turn, so it must not look like it did."""
        output = render_coloured({**self.ALLOW, "mode": "shadow", "acted": False}, verbose=True)
        assert "\x1b[2m" in output
        assert "\x1b[32m" not in output

    def test_a_use_case_that_is_not_the_gate_is_not_coloured_by_verdict(self):
        """`tools` has no allow/deny; hiding a tool is not a verdict about a call."""
        output = render_coloured(
            {
                "use_case": "tools",
                "mode": "active",
                "acted": True,
                "summary": "17 tool(s) hidden this turn",
                "latency_ms": 411,
            }
        )
        assert "\x1b[32m" not in output
        assert "\x1b[31m" not in output
        assert "17 tool(s) hidden" in output
