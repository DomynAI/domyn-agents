"""Push-to-talk: the space bar read as held or tapped, and what reaches the editor.

The terminal never reports a key being let go, so the tests drive `on_space`
with a clock of their own (when each space "arrived") and let the real timers
decide the release, with the windows shortened to keep the suite fast.
"""

import asyncio
import io
import wave
from pathlib import Path
from typing import Any

import pytest
from pydantic import Field

from domyn.agents.audio import Transcriber, Transcript, TranscriptionError
from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli import prompt as prompt_mod
from domyn.agents.interfaces.cli import voice as voice_mod
from domyn.agents.interfaces.cli.prompt import InputSession
from domyn.agents.interfaces.cli.voice import VoiceError, VoiceInput


def wav(seconds: float) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16_000)
        f.writeframes(b"\x00\x00" * int(16_000 * seconds))
    return buffer.getvalue()


class FakeRecorder:
    def __init__(self, seconds: float = 2.0, fail: bool = False) -> None:
        self.recorded_seconds = seconds
        self.fail = fail
        self.started = False
        self.stopped = False

    def start(self) -> None:
        if self.fail:
            raise VoiceError("The microphone could not be opened: no device")
        self.started = True

    @property
    def seconds(self) -> float:
        return 3.0

    def stop(self) -> bytes:
        self.stopped = True
        return wav(self.recorded_seconds)


class FakeTranscriber(Transcriber):
    type: str = "fake-cli-voice"
    text: str = "Domani alle dieci."
    error: str | None = None
    heard: list[bytes] = Field(default_factory=list)

    async def transcribe(self, audio, *, filename=None):  # type: ignore[no-untyped-def]
        self.heard.append(audio)
        if self.error:
            raise TranscriptionError(self.error)
        return Transcript(text=self.text)


class FakeBuffer:
    def __init__(self) -> None:
        self.text = ""
        self.submitted: list[str] = []

    def validate_and_handle(self) -> None:
        self.submitted.append(self.text)


class FakeApp:
    def __init__(self, running: bool = True) -> None:
        self.is_running = running
        self.is_done = False
        self.current_buffer = FakeBuffer()

    def invalidate(self) -> None:
        pass


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def short_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(voice_mod, "TAP_WINDOW", 0.05)


def make(**kwargs: Any) -> tuple[VoiceInput, Clock, FakeRecorder, FakeTranscriber]:
    clock = Clock()
    recorder = FakeRecorder(**kwargs)
    transcriber = FakeTranscriber()
    voice = VoiceInput(transcriber, recorder=lambda: recorder, clock=clock)  # type: ignore[arg-type,return-value]
    voice.enabled = True
    return voice, clock, recorder, transcriber


async def settle(seconds: float = 0.5) -> None:
    await asyncio.sleep(seconds)


class TestHoldingTheKey:
    @pytest.mark.asyncio
    async def test_held_then_let_go_sends_what_was_said(self):
        voice, clock, recorder, transcriber = make()
        app = FakeApp()
        voice.on_space(app)  # the press
        assert voice.recording and recorder.started
        # The keyboard's delay until repeat, then repeats every 30 ms.
        clock.now += 0.375
        voice.on_space(app)
        for _ in range(5):
            clock.now += 0.03
            voice.on_space(app)
        assert voice.recording and voice._mode == "hold"
        await settle()  # no more repeats: let go
        assert not voice.recording and recorder.stopped
        assert app.current_buffer.submitted == ["Domani alle dieci."]
        assert len(transcriber.heard) == 1

    @pytest.mark.asyncio
    async def test_the_release_delay_follows_the_repeat_rate(self):
        voice, clock, _, _ = make()
        app = FakeApp()
        voice.on_space(app)
        clock.now += 0.03
        voice.on_space(app)
        assert voice.release_delay == pytest.approx(0.12)  # floored
        clock.now += 0.12
        voice.on_space(app)
        assert voice.release_delay == pytest.approx(0.3)


class TestTapping:
    @pytest.mark.asyncio
    async def test_a_tap_starts_and_the_next_tap_sends(self):
        voice, clock, _, _ = make()
        app = FakeApp()
        voice.on_space(app)
        await settle(0.1)  # no repeat came: a tap
        assert voice.recording and voice._mode == "tap"
        assert "press space to send" in "".join(text for _, text in voice.rows())
        clock.now += 4.0
        voice.on_space(app)  # the second tap
        assert voice.recording  # until it is clear no repeats follow
        await settle()
        assert app.current_buffer.submitted == ["Domani alle dieci."]

    @pytest.mark.asyncio
    async def test_a_hold_whose_repeats_start_late_is_still_a_hold(self):
        """A long delay until repeat makes a hold look like a tap first; the
        first repeat looks like a second tap, and the ones after it say otherwise."""
        voice, clock, _, _ = make()
        app = FakeApp()
        voice.on_space(app)
        await settle(0.1)
        assert voice._mode == "tap"
        clock.now += 1.0
        voice.on_space(app)  # the late first repeat
        for _ in range(3):
            clock.now += 0.05
            voice.on_space(app)
        await asyncio.sleep(0.05)
        assert voice.recording and voice._mode == "hold"
        assert app.current_buffer.submitted == []
        await settle()
        assert app.current_buffer.submitted == ["Domani alle dieci."]


class TestWhatIsNotSent:
    @pytest.mark.asyncio
    async def test_a_stray_press_records_nothing_worth_sending(self):
        voice, clock, _, transcriber = make(seconds=0.1)
        app = FakeApp()
        voice.on_space(app)
        clock.now += 0.02
        voice.on_space(app)
        await settle()
        assert app.current_buffer.submitted == [] and transcriber.heard == []

    @pytest.mark.asyncio
    async def test_a_failed_transcription_is_shown_not_sent(self):
        voice, clock, _, transcriber = make()
        transcriber.error = "engine broke"
        app = FakeApp()
        voice.on_space(app)
        clock.now += 0.03
        voice.on_space(app)
        await settle()
        assert app.current_buffer.submitted == []
        assert "Not transcribed: engine broke" in "".join(t for _, t in voice.rows())

    @pytest.mark.asyncio
    async def test_silence_is_shown_not_sent(self):
        voice, clock, _, transcriber = make()
        transcriber.text = ""
        app = FakeApp()
        voice.on_space(app)
        clock.now += 0.03
        voice.on_space(app)
        await settle()
        assert app.current_buffer.submitted == []
        assert "Nothing heard" in "".join(t for _, t in voice.rows())

    @pytest.mark.asyncio
    async def test_no_microphone_is_shown_and_nothing_records(self):
        voice, _, _, _ = make(fail=True)
        voice.on_space(FakeApp())
        assert not voice.recording
        assert "no device" in "".join(t for _, t in voice.rows())

    @pytest.mark.asyncio
    async def test_cancel_drops_a_recording(self):
        voice, _, recorder, _ = make()
        voice.on_space(FakeApp())
        voice.cancel()
        assert not voice.recording and recorder.stopped
        await settle(0.1)


class TestBetweenReads:
    @pytest.mark.asyncio
    async def test_a_transcription_finished_with_no_read_open_goes_to_the_next(self):
        voice, clock, _, _ = make()
        first = FakeApp()
        voice.on_space(first)
        clock.now += 0.03
        voice.on_space(first)
        first.is_running = False  # the read ended (the turn was interrupted)
        await settle()
        assert first.current_buffer.submitted == []

        second = FakeApp()
        voice.attach(second)
        await asyncio.sleep(0.01)
        assert second.current_buffer.submitted == ["Domani alle dieci."]


class TestRows:
    def test_nothing_when_off(self):
        voice, _, _, _ = make()
        voice.enabled = False
        assert voice.rows() == []

    def test_how_to_start_when_on(self):
        voice, _, _, _ = make()
        assert "hold space to talk" in voice.rows()[0][1]

    @pytest.mark.asyncio
    async def test_recording_shows_the_clock_and_how_to_send(self):
        voice, _, _, _ = make()
        voice.on_space(FakeApp())
        text = "".join(t for _, t in voice.rows())
        assert "● recording 0:03" in text and "release space to send" in text
        voice.cancel()


class TestTheEditor:
    """The space bar inside the real prompt_toolkit editor."""

    async def _drive(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, keys: str, *, enabled: bool = True
    ) -> tuple[InputSession, list[str], asyncio.Future[str]]:
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        pressed: list[str] = []

        class Spy:
            enabled = True
            busy = False
            recording = False

            def on_space(self, app: Any) -> None:
                pressed.append("space")

            def attach(self, app: Any) -> None:
                pass

            def rows(self) -> list[tuple[str, str]]:
                return []

        spy = Spy()
        spy.enabled = enabled
        monkeypatch.setattr(prompt_mod, "interactive_terminal", lambda: True)
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            from rich.console import Console

            session = InputSession(
                Console(file=io.StringIO(), width=80),
                Theme.build(),
                history_path=tmp_path / "history",
                voice=spy,  # type: ignore[arg-type]
            )
            read = asyncio.ensure_future(session.ask())
            await asyncio.sleep(0.05)
            pipe.send_text(keys)
            await asyncio.sleep(0.1)
            draft = session.draft()
            read.cancel()
            with pytest.raises(asyncio.CancelledError):
                await read
        return session, [draft, *pressed], read

    @pytest.mark.asyncio
    async def test_a_space_on_an_empty_line_goes_to_voice(self, monkeypatch, tmp_path):
        _, seen, _ = await self._drive(monkeypatch, tmp_path, " ")
        assert seen == ["", "space"]

    @pytest.mark.asyncio
    async def test_a_space_in_a_sentence_is_a_space(self, monkeypatch, tmp_path):
        _, seen, _ = await self._drive(monkeypatch, tmp_path, "ciao a")
        assert seen == ["ciao a"]

    @pytest.mark.asyncio
    async def test_with_voice_off_a_space_is_a_space(self, monkeypatch, tmp_path):
        _, seen, _ = await self._drive(monkeypatch, tmp_path, " x", enabled=False)
        assert seen == [" x"]


class TestTalkingOverTheAgent:
    @pytest.mark.asyncio
    async def test_a_space_during_a_turn_stops_it_and_keeps_recording(self, monkeypatch, tmp_path):
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        from rich.console import Console

        voice, _, recorder, _ = make()
        monkeypatch.setattr(prompt_mod, "interactive_terminal", lambda: True)
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            session = InputSession(
                Console(file=io.StringIO(), width=80),
                Theme.build(),
                history_path=tmp_path / "history",
                voice=voice,
            )

            async def read_alongside() -> str:
                # As the REPL's reader does: a KeyboardInterrupt ending a task
                # would leave the event loop itself.
                try:
                    return await session.ask(concurrent=True)
                except KeyboardInterrupt:
                    return "interrupted"

            read = asyncio.ensure_future(read_alongside())
            await asyncio.sleep(0.05)
            pipe.send_text(" ")
            assert await asyncio.wait_for(read, timeout=2) == "interrupted"
        assert voice.recording and recorder.started
        # Not armed: a Ctrl-C right after must not read as the second of a pair.
        assert not session._armed()
        voice.cancel()


class TestTheVoiceCommand:
    def _repl(self, config: dict[str, Any] | None = None) -> tuple[Any, io.StringIO]:
        from tests.cli.test_repl_session import _repl

        return _repl(config=config)

    @pytest.mark.asyncio
    async def test_it_needs_an_interactive_terminal(self):
        repl, out = self._repl()
        repl._input._session = None  # no editor: a pipe, a redirected terminal
        assert await repl.toggle_voice() is False
        assert "interactive terminal" in out.getvalue()

    @pytest.mark.asyncio
    async def test_it_loads_the_configured_transcriber_and_turns_on_and_off(self, monkeypatch):
        repl, out = self._repl(config={"voice": {"transcriber": {"type": "fake-cli-voice"}}})
        monkeypatch.setattr(type(repl._input), "enhanced", property(lambda self: True))
        monkeypatch.setattr(voice_mod.Recorder, "check", staticmethod(lambda: None))
        monkeypatch.setattr(voice_mod.Recorder, "probe", staticmethod(lambda: None))

        assert await repl.toggle_voice() is True
        assert isinstance(repl._input.voice.transcriber, FakeTranscriber)
        assert repl._input.voice.enabled
        assert "hold space on an empty line" in out.getvalue()

        assert await repl.toggle_voice() is False
        assert not repl._input.voice.enabled
        assert "Voice off." in out.getvalue()

    @pytest.mark.asyncio
    async def test_without_configuration_it_is_the_local_transcriber(self, monkeypatch):
        from domyn.agents.audio import LocalTranscriber

        repl, _ = self._repl()
        assert isinstance(repl._voice_transcriber(), LocalTranscriber)

    @pytest.mark.asyncio
    async def test_a_missing_microphone_library_says_how_to_install_it(self, monkeypatch):
        repl, out = self._repl()
        monkeypatch.setattr(type(repl._input), "enhanced", property(lambda self: True))

        def missing() -> None:
            raise VoiceError(voice_mod._INSTALL_HINT)

        monkeypatch.setattr(voice_mod.Recorder, "check", staticmethod(missing))
        assert await repl.toggle_voice() is False
        assert "domyn[cli,audio]" in out.getvalue()

    @pytest.mark.asyncio
    async def test_a_model_that_cannot_load_leaves_voice_off(self, monkeypatch):
        repl, out = self._repl(config={"voice": {"transcriber": {"type": "fake-cli-voice"}}})
        monkeypatch.setattr(type(repl._input), "enhanced", property(lambda self: True))
        monkeypatch.setattr(voice_mod.Recorder, "check", staticmethod(lambda: None))
        monkeypatch.setattr(voice_mod.Recorder, "probe", staticmethod(lambda: None))

        async def broken(self, on_download=None):  # type: ignore[no-untyped-def]
            raise TranscriptionError("Could not load the speech model")

        monkeypatch.setattr(FakeTranscriber, "prepare", broken)
        assert await repl.toggle_voice() is False
        assert repl._input.voice is None
        assert "Could not load the speech model" in out.getvalue()

    @pytest.mark.asyncio
    async def test_a_microphone_that_does_not_answer_points_to_the_permission(self, monkeypatch):
        import time

        from domyn.agents.interfaces.cli import repl as repl_mod

        repl, out = self._repl(config={"voice": {"transcriber": {"type": "fake-cli-voice"}}})
        monkeypatch.setattr(type(repl._input), "enhanced", property(lambda self: True))
        monkeypatch.setattr(voice_mod.Recorder, "check", staticmethod(lambda: None))
        monkeypatch.setattr(voice_mod.Recorder, "probe", staticmethod(lambda: time.sleep(0.3)))
        monkeypatch.setattr(repl_mod, "PROBE_TIMEOUT", 0.05)
        assert await repl.toggle_voice() is False
        assert "Microphone, then /voice again" in " ".join(out.getvalue().split())

    def test_it_is_listed_in_help(self):
        from domyn.agents.interfaces.cli.commands import help_lines

        assert any(name.startswith("/voice") for name, _ in help_lines())
