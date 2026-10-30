"""Tests for the startup banner.

The banner is decoration, so the interesting cases are all the ways it must
gracefully not happen.
"""

import io
import os
import re

import pytest
from rich.console import Console

from domyn.agents.interfaces.cli import Capabilities, Theme, render_banner
from domyn.agents.interfaces.cli import banner as banner_mod


def _theme(*, color: bool = True, unicode: bool = True, animation: bool = False) -> Theme:
    base = Theme.build()
    return type(base)(
        capabilities=Capabilities(color=color, unicode=unicode, animation=animation),
        symbols=base.symbols,
        styles=base.styles,
    )


def _console(width: int, height: int = 60) -> tuple[Console, io.StringIO]:
    """A generously tall console by default, so width tests isolate width.

    ``color_system`` is pinned. ``force_terminal`` only makes rich *emit*
    colour; which colour system it uses is still detected from the environment,
    and a CI runner without ``COLORTERM`` gets 8- or 16-colour output. These
    tests are about what we ask for, not about what the machine running them can
    display — left unpinned they assert the capabilities of the terminal.
    """
    buffer = io.StringIO()
    return (
        Console(
            file=buffer,
            force_terminal=True,
            color_system="truecolor",
            width=width,
            height=height,
        ),
        buffer,
    )


_BAKED = banner_mod._LOGO_WIDTHS[0]
_SMALLEST = banner_mod._LOGO_WIDTHS[-1]


class TestArtSelection:
    @pytest.mark.parametrize("width", [_BAKED, _BAKED + 1, 400])
    def test_art_is_used_when_it_fits(self, width: int) -> None:
        console, _ = _console(width)
        assert banner_mod._pick_rows(console, _theme()) == banner_mod._load_logo(_BAKED)

    @pytest.mark.parametrize("baked", banner_mod._LOGO_WIDTHS)
    def test_the_widest_that_fits_wins(self, baked: int) -> None:
        """One column short of the next size up, so the choice is forced."""
        console, _ = _console(baked)
        assert banner_mod._pick_rows(console, _theme()) == banner_mod._load_logo(baked)

    def test_narrower_than_every_size_uses_the_wordmark(self) -> None:
        """The art cannot be re-rendered smaller than what was baked, so below
        the narrowest cut the wordmark is used rather than soft-wrapped noise."""
        console, _ = _console(_SMALLEST - 1)
        assert banner_mod._pick_rows(console, _theme()) == banner_mod._WORDMARK

    def test_very_narrow_terminal_falls_back_to_one_line(self) -> None:
        console, _ = _console(30)
        assert banner_mod._pick_rows(console, _theme()) == (banner_mod._COMPACT,)


class TestHeightAwareness:
    """The reason an earlier version scrolled: only width was considered."""

    def test_short_terminal_gives_up_on_art(self) -> None:
        """Derived from the asset rather than hardcoded, so re-baking the logo
        at different proportions does not turn this into a false failure."""
        # Measured against the narrowest cut, which is also the shortest: if even
        # that one does not fit vertically, none of them can.
        art = banner_mod._load_logo(_SMALLEST)
        assert art is not None
        # One row short of what the art needs, and wide enough that only the
        # height can be what rules it out.
        console, _ = _console(400, height=len(art) + banner_mod._RESERVED_ROWS - 1)
        assert banner_mod._pick_rows(console, _theme()) == banner_mod._WORDMARK

    def test_art_always_leaves_room_for_the_prompt(self) -> None:
        for height in range(20, 60, 4):
            console, _ = _console(400, height=height)
            rows = banner_mod._pick_rows(console, _theme())
            fits = len(rows) + banner_mod._RESERVED_ROWS <= height
            assert fits or rows in (banner_mod._WORDMARK, (banner_mod._COMPACT,))


class TestCapabilityGates:
    def test_no_colour_means_no_art(self) -> None:
        """The art is entirely colour: without it the blocks carry no shape."""
        console, _ = _console(400)
        assert banner_mod._pick_rows(console, _theme(color=False)) == banner_mod._WORDMARK

    def test_no_unicode_means_ascii_wordmark(self) -> None:
        console, _ = _console(400)
        assert banner_mod._pick_rows(console, _theme(unicode=False)) == banner_mod._WORDMARK_ASCII

    def test_pipe_emits_no_escape_codes(self) -> None:
        """Redirected output must stay clean so it can be parsed or logged."""
        buffer = io.StringIO()
        console = Console(file=buffer, width=400)  # not a terminal
        render_banner(console, Theme.build(), subtitle="v1", animate=False)
        assert "\x1b" not in buffer.getvalue()


class _VirtualClock:
    """``time`` for the animation loop, with the waiting taken out."""

    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class TestRendering:
    def test_art_is_emitted_verbatim(self) -> None:
        console, buffer = _console(400)
        render_banner(console, _theme(), animate=False)
        output = buffer.getvalue()
        assert "\x1b[38;2;" in output, "truecolor codes must pass through untouched"
        assert "▀" in output or "▄" in output

    def test_subtitle_is_printed(self) -> None:
        console, buffer = _console(400)
        render_banner(console, _theme(), subtitle="v1.3.2 · Assistant", animate=False)
        assert "Assistant" in buffer.getvalue()

    def test_missing_asset_does_not_crash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A stripped install must still start; a banner is not worth an exception."""
        banner_mod._load_logo.cache_clear()
        monkeypatch.setattr(banner_mod, "_ASSETS", banner_mod._ASSETS / "nonexistent")
        console, buffer = _console(400)
        render_banner(console, _theme(), animate=False)
        banner_mod._load_logo.cache_clear()
        assert buffer.getvalue().strip(), "something must still be printed"

    def test_animation_respects_its_ceiling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The ceiling is a backstop against a malformed archive, not the
        intended length — the frames carry their own durations.

        Played on a virtual clock: the real frames add up to well under the
        ceiling, so against them the assertion would hold with no ceiling at
        all. These durations are the malformed archive it exists for.
        """
        clock = _VirtualClock()
        monkeypatch.setattr(banner_mod, "time", clock)
        console, _ = _console(400)

        banner_mod._animate_art(console, tuple(("frame", 1.0) for _ in range(100)))

        assert clock.now <= banner_mod._ANIMATION_CEILING_S + 1.0


class TestSkipKey:
    """The reveal is skippable: a keypress ends it early.

    The interesting cases are all the ways it must *not* skip: a pipe has nobody
    at it, and a keypress the user never made must not be invented.
    """

    def test_skips_never_happen_on_a_pipe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Redirected stdin has no human at it; the banner must play out."""
        monkeypatch.setattr(banner_mod.sys, "stdin", io.StringIO(""))
        assert banner_mod._skip_key_pressed() is False

    def test_skips_never_happen_on_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The POSIX ``select`` path is not exercised on Windows."""
        monkeypatch.setattr(banner_mod.sys, "stdin", io.StringIO(""))
        monkeypatch.setattr(banner_mod.os, "name", "nt")
        assert banner_mod._skip_key_pressed() is False

    def test_a_ready_tty_key_is_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A key already buffered on a TTY counts as a skip."""
        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, b"x")
            with os.fdopen(read_fd, "rb", buffering=0) as stdin:
                stdin.isatty = lambda: True  # type: ignore[attr-defined]
                monkeypatch.setattr(banner_mod.sys, "stdin", stdin)
                monkeypatch.setattr(banner_mod.os, "name", "posix")
                assert banner_mod._skip_key_pressed() is True
        finally:
            os.close(write_fd)

    def test_an_idle_tty_is_not_a_skip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No key buffered means no skip, even on a TTY."""
        read_fd, write_fd = os.pipe()
        try:
            with os.fdopen(read_fd, "rb", buffering=0) as stdin:
                stdin.isatty = lambda: True  # type: ignore[attr-defined]
                monkeypatch.setattr(banner_mod.sys, "stdin", stdin)
                monkeypatch.setattr(banner_mod.os, "name", "posix")
                assert banner_mod._skip_key_pressed() is False
        finally:
            os.close(write_fd)

    def test_wait_returns_true_only_when_a_key_arrives(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``_wait_for_skip`` reports the skip so the caller can break early."""
        monkeypatch.setattr(banner_mod, "_skip_key_pressed", lambda: True)
        assert banner_mod._wait_for_skip(lambda: False) is True

    def test_wait_returns_false_when_the_stop_condition_wins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Frame duration ending first means no skip happened."""
        monkeypatch.setattr(banner_mod, "_skip_key_pressed", lambda: False)
        assert banner_mod._wait_for_skip(lambda: True) is False

    def test_animate_skips_when_a_key_is_pressed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A keypress mid-reveal ends the animation immediately."""
        clock = _VirtualClock()
        monkeypatch.setattr(banner_mod, "time", clock)
        monkeypatch.setattr(banner_mod, "_skip_key_pressed", lambda: True)
        console, _ = _console(400)
        banner_mod._animate_art(console, tuple(("frame", 1.0) for _ in range(3)))
        # Only the first frame played: the skip happened before any sleep.
        assert clock.now < 1.0


class TestKeyframes:
    """The reveal ships as baked keyframes; decoding a 1.8 MB GIF at startup
    would cost more than the banner is worth."""

    @pytest.mark.parametrize("width", banner_mod._LOGO_WIDTHS)
    def test_every_baked_width_has_its_keyframes(self, width: int) -> None:
        assert len(banner_mod._load_frames(width)) > 1

    @pytest.mark.parametrize("width", banner_mod._LOGO_WIDTHS)
    def test_last_keyframe_matches_the_static_frame(self, width: int) -> None:
        """The overlay is transient and the static frame is printed after it.
        If the two differ the logo visibly jumps at the end of the reveal."""
        logo = banner_mod._load_logo(width)
        assert logo is not None
        assert banner_mod._load_frames(width)[-1][0].rstrip("\n").split("\n") == list(logo)

    @pytest.mark.parametrize("width", banner_mod._LOGO_WIDTHS)
    def test_frames_play_at_the_source_cadence(self, width: int) -> None:
        """Durations ship with the frames so the reveal is not replayed at some
        rate that merely looked plausible."""
        total = sum(seconds for _frame, seconds in banner_mod._load_frames(width))
        assert 3.5 < total < 5.0, f"reveal lasts {total:.2f}s"

    def test_a_malformed_archive_is_rejected_rather_than_half_played(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: object
    ) -> None:
        """A duration list that does not match the frames means the archive was
        built by something this code does not understand."""
        import gzip
        from pathlib import Path

        assert isinstance(tmp_path, Path)
        (tmp_path / f"logo_{_BAKED}.frames.gz").write_bytes(
            gzip.compress(b"40,40\x0conly-one-frame")
        )
        banner_mod._load_frames.cache_clear()
        monkeypatch.setattr(banner_mod, "_ASSETS", tmp_path)
        assert banner_mod._load_frames(_BAKED) == ()
        banner_mod._load_frames.cache_clear()

    def test_missing_keyframes_still_render_the_logo(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An install with the archives stripped loses the animation, not the
        banner."""
        monkeypatch.setattr(banner_mod, "_load_frames", lambda width: ())
        console, buffer = _console(400)
        render_banner(console, _theme(animation=True), animate=True)
        assert "▀" in buffer.getvalue() or "▄" in buffer.getvalue()

    def test_a_corrupt_archive_is_not_fatal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        banner_mod._load_frames.cache_clear()
        monkeypatch.setattr(banner_mod, "_ASSETS", banner_mod._ASSETS / "nonexistent")
        assert banner_mod._load_frames(_BAKED) == ()
        banner_mod._load_frames.cache_clear()

    def test_animation_is_skipped_when_the_caller_opts_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        played = []
        monkeypatch.setattr(banner_mod, "_animate_art", lambda *a: played.append(a))
        console, _ = _console(400)
        render_banner(console, _theme(animation=True), animate=False)
        assert not played

    def test_animation_is_skipped_when_the_terminal_cannot_take_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        played = []
        monkeypatch.setattr(banner_mod, "_animate_art", lambda *a: played.append(a))
        console, _ = _console(400)
        render_banner(console, _theme(animation=False), animate=True)
        assert not played


class TestLightTerminals:
    """The assets are build output, so the property that makes them readable on a
    light background has to be asserted on the shipped bytes.

    A re-bake is the only way to lose it, and a re-bake is exactly the moment
    nobody is looking at a white terminal.
    """

    @staticmethod
    def _near_white(text: str) -> list[str]:
        """Truecolor codes bright enough to vanish on a white background."""
        found = []
        for kind, r, g, b in re.findall(r"\x1b\[(38|48);2;(\d+);(\d+);(\d+)m", text):
            if min(int(r), int(g), int(b)) >= 232:
                found.append(f"{'fg' if kind == '38' else 'bg'} {r};{g};{b}")
        return found

    @pytest.mark.parametrize("width", banner_mod._LOGO_WIDTHS)
    def test_the_static_logo_names_no_white(self, width: int) -> None:
        """The wordmark is emitted as SGR 39 — the terminal's own foreground —
        rather than as white, which is invisible on a light theme."""
        rows = banner_mod._load_logo(width)
        assert rows is not None
        offenders = self._near_white("\n".join(rows))
        assert not offenders, f"logo_{width}.ans hardcodes {offenders[:3]}"

    @pytest.mark.parametrize("width", banner_mod._LOGO_WIDTHS)
    def test_the_animation_names_no_white(self, width: int) -> None:
        frames = banner_mod._load_frames(width)
        assert frames
        offenders = self._near_white("".join(frame for frame, _ in frames))
        assert not offenders, f"logo_{width}.frames.gz hardcodes {offenders[:3]}"

    @pytest.mark.parametrize("width", banner_mod._LOGO_WIDTHS)
    def test_the_wordmark_is_actually_drawn(self, width: int) -> None:
        """Guards the other two from passing for the wrong reason: dropping the
        lettering altogether would also remove every white code."""
        rows = banner_mod._load_logo(width)
        assert rows is not None
        assert "█" in "\n".join(rows)
