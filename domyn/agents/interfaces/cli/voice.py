"""Talking to the agent: push-to-talk on the space bar, transcribed into the editor.

With voice on (``/voice``, or ``domyn run --voice``), holding the space bar on an
empty line records from the microphone and letting go sends what was said, as
if it had been typed: the transcription is put in the editor and submitted, so
the transcript, the history and everything after it are those of a typed line.
Pressed during a turn, the space bar interrupts it first, as Ctrl-C would, and
what is said then becomes the next message.

**A terminal reports key presses, never releases.** Holding a key arrives as one
press, a pause (the keyboard's delay until repeat), then the same key repeated
every few tens of milliseconds. So "let go" is read as "the repeats stopped":

- a space while idle starts recording;
- spaces less than ``REPEAT_GAP`` apart are repeats: the key is held, and the
  recording ends ``release_delay`` after the last one (a few repeat intervals);
- no repeat within ``TAP_WINDOW`` of the first press means a tap: recording
  goes on until the next press, so a keyboard with key repeat turned off, or a
  long delay until repeat, still works, as press-to-start / press-to-send;
- a later press is confirmed as a press, and not the first repeat of a late
  hold, by no repeat following it within ``REPEAT_GAP``.

The kitty keyboard protocol reports releases, but only with every key sent as
an escape sequence, which prompt_toolkit's input parser does not read: the
editor would stop understanding typing. Inferring the release works in every
terminal, for the cost of a fraction of a second after letting go.
"""

import asyncio
import io
import time
import wave
from collections.abc import Callable
from typing import Any

from domyn.agents.audio import Transcriber, TranscriptionError
from domyn.agents.logger import get_logger

SAMPLE_RATE = 16_000

TAP_WINDOW = 0.6
"""Seconds after a first press within which a repeat makes it a hold. Longer
than the usual delay until repeat (macOS: 225-375 ms by default)."""

REPEAT_GAP = 0.25
"""Two spaces closer than this are one key held down."""

MIN_SECONDS = 0.3
"""Shorter than this is a stray press, not speech: nothing is sent."""

MAX_SECONDS = 300.0
"""A recording is sent on its own after five minutes."""

PERMISSION_HINT = (
    "The microphone did not answer. On macOS, allow it for this terminal in System "
    "Settings > Privacy & Security > Microphone, then /voice again."
)
PROBE_TIMEOUT = 20.0
"""Seconds to wait for the microphone to open at ``/voice``: long enough to
answer the permission question macOS asks the first time."""

_INSTALL_HINT = (
    "Voice needs the audio extra: `uv tool install 'domyn[cli,audio]'` "
    "or `pip install 'domyn[audio]'` (Python 3.11 to 3.14)."
)


class VoiceError(RuntimeError):
    """Voice cannot be used: no microphone library, no microphone."""


class Recorder:
    """The microphone, as 16 kHz mono 16-bit PCM, until stopped."""

    def __init__(self) -> None:
        self._chunks: list[bytes] = []
        self._stream: Any = None
        self._started = 0.0

    @staticmethod
    def check() -> None:
        """Raise ``VoiceError`` when the microphone library cannot be loaded."""
        try:
            import sounddevice  # noqa: F401
        except ImportError as exc:
            raise VoiceError(_INSTALL_HINT) from exc
        except OSError as exc:
            # sounddevice imports, then fails to find PortAudio (Linux without
            # libportaudio2).
            raise VoiceError(
                f"The microphone library could not start ({exc}). On Debian or Ubuntu: "
                "`apt install libportaudio2`."
            ) from exc

    @staticmethod
    def probe() -> None:
        """Open the microphone and close it again: what makes macOS ask for
        permission. Blocking while the question is open, so called from a thread
        at ``/voice``, never from a key press, where it would freeze the editor."""
        import sounddevice

        try:
            with sounddevice.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16"):
                pass
        except Exception as exc:
            raise VoiceError(f"The microphone could not be opened: {exc}") from exc

    def start(self) -> None:
        import sounddevice

        self._chunks = []

        def keep(data: Any, frames: int, when: Any, status: Any) -> None:
            self._chunks.append(bytes(data))

        try:
            self._stream = sounddevice.RawInputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=keep
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            raise VoiceError(f"The microphone could not be opened: {exc}") from exc
        self._started = time.monotonic()

    @property
    def seconds(self) -> float:
        return time.monotonic() - self._started if self._stream is not None else 0.0

    def stop(self) -> bytes:
        """Stop and return what was recorded, as a WAV file."""
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(SAMPLE_RATE)
            f.writeframes(b"".join(self._chunks))
        self._chunks = []
        return buffer.getvalue()


def _duration(wav: bytes) -> float:
    with wave.open(io.BytesIO(wav), "rb") as f:
        return f.getnframes() / f.getframerate()


class VoiceInput:
    """Push-to-talk for one editor. See the module docstring for the key logic.

    The editor calls ``on_space`` for every space typed on an empty line (and
    every space while recording), ``attach`` when a read starts, and draws
    ``rows()`` above the prompt. A finished transcription is put into the
    editor running at that moment and submitted; when none is, it waits for the
    next one to attach.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        *,
        recorder: Callable[[], Recorder] = Recorder,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.transcriber = transcriber
        self.enabled = False
        self._new_recorder = recorder
        self._clock = clock
        self._recorder: Recorder | None = None
        # "undecided" (first press, waiting for repeats), "hold" or "tap".
        self._mode: str | None = None
        self._last_press = 0.0
        self._repeat_interval: float | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._transcribing: asyncio.Task[None] | None = None
        self._ready: str | None = None
        self._notice: str | None = None
        self._app: Any = None

    # --- state, for the editor ---------------------------------------------

    @property
    def recording(self) -> bool:
        return self._recorder is not None

    @property
    def busy(self) -> bool:
        """Recording or transcribing: spaces belong to voice, not to the line."""
        return self.recording or self._transcribing is not None

    def rows(self) -> list[tuple[str, str]]:
        """The voice line above the prompt: what is happening, or the last
        problem, or how to start; nothing when voice is off."""
        if not self.enabled:
            return []
        if self._recorder is not None:
            elapsed = int(self._recorder.seconds)
            send = "press space to send" if self._mode == "tap" else "release space to send"
            return [
                ("class:voice-recording", f"● recording {elapsed // 60}:{elapsed % 60:02d}"),
                ("class:voice-hint", f" · {send}\n"),
            ]
        if self._transcribing is not None:
            return [("class:voice-hint", "◌ transcribing…\n")]
        if self._notice:
            return [("class:voice-notice", f"{self._notice}\n")]
        return [("class:voice-hint", "voice on · hold space to talk\n")]

    def attach(self, app: Any) -> None:
        """A read started on ``app``: the one a transcription goes to."""
        self._app = app
        if self._ready is not None:
            text, self._ready = self._ready, None
            # After the read has drawn itself: submitting from inside pre_run
            # would end it before it began.
            asyncio.get_running_loop().call_soon(self._submit, text)

    # --- the space bar -----------------------------------------------------

    def on_space(self, app: Any) -> None:
        self._app = app
        now = self._clock()
        if self._transcribing is not None:
            return
        if self._recorder is None:
            self._start(now)
            return
        gap = now - self._last_press
        self._last_press = now
        if gap < REPEAT_GAP:
            # A repeat: the key is held.
            self._mode = "hold"
            self._repeat_interval = gap
            self._schedule(self.release_delay, self._released)
        else:
            # A press after a pause: a second tap, or the first repeat of a hold
            # whose delay until repeat was long. A repeat within REPEAT_GAP says
            # which.
            self._schedule(REPEAT_GAP, self._released)

    @property
    def release_delay(self) -> float:
        """How long after the last repeat the key counts as released."""
        interval = self._repeat_interval or 0.1
        return min(0.35, max(0.12, 2.5 * interval))

    def _start(self, now: float) -> None:
        self._notice = None
        recorder = self._new_recorder()
        try:
            recorder.start()
        except VoiceError as exc:
            self._notice = str(exc)
            self._invalidate()
            return
        self._recorder = recorder
        self._mode = "undecided"
        self._last_press = now
        self._repeat_interval = None
        self._schedule(TAP_WINDOW, self._decided_tap)
        self._invalidate()

    def _decided_tap(self) -> None:
        if self._mode == "undecided":
            self._mode = "tap"
            self._schedule(MAX_SECONDS, self._released)
            self._invalidate()

    def _released(self) -> None:
        self._timer = None
        if self._recorder is None:
            return
        recorder, self._recorder = self._recorder, None
        self._mode = None
        audio = recorder.stop()
        if _duration(audio) < MIN_SECONDS:
            self._invalidate()
            return
        self._transcribing = asyncio.ensure_future(self._transcribe(audio))
        self._invalidate()

    def _schedule(self, delay: float, callback: Callable[[], None]) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = asyncio.get_running_loop().call_later(delay, callback)

    # --- transcription -----------------------------------------------------

    async def _transcribe(self, audio: bytes) -> None:
        try:
            transcript = await self.transcriber.transcribe(audio, filename="voice.wav")
        except TranscriptionError as exc:
            self._notice = f"Not transcribed: {exc}"
            return
        except Exception as exc:
            get_logger().warning(f"voice: transcription failed: {type(exc).__name__}")
            get_logger().debug(f"voice transcription detail: {exc}")
            self._notice = f"Not transcribed ({type(exc).__name__})"
            return
        finally:
            self._transcribing = None
            self._invalidate()
        if not transcript.text:
            self._notice = "Nothing heard. Hold space while you speak."
            self._invalidate()
            return
        self._submit(transcript.text)

    def _submit(self, text: str) -> None:
        app = self._app
        if app is None or not getattr(app, "is_running", False) or app.is_done:
            self._ready = text
            return
        buffer = app.current_buffer
        buffer.text = text
        buffer.validate_and_handle()

    def cancel(self) -> None:
        """Drop whatever is in progress: voice turned off, or the session ending."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._recorder is not None:
            self._recorder.stop()
            self._recorder = None
        if self._transcribing is not None:
            self._transcribing.cancel()
            self._transcribing = None
        self._mode = None
        self._ready = None
        self._notice = None

    def _invalidate(self) -> None:
        app = self._app
        if app is not None and getattr(app, "is_running", False):
            app.invalidate()
