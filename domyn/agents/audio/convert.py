"""Audio in whatever format it came, as a WAV file a local model can read.

The local engine reads PCM WAV only. Anything else (a Telegram voice note in
Ogg/Opus, an m4a from a phone, an mp3) is converted with ``ffmpeg``, the one tool
that reads all of them; it is looked for on ``PATH`` and only when needed, so WAV
works without it.
"""

import asyncio
import shutil
import tempfile
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from domyn.agents.audio.models import TranscriptionError
from domyn.agents.logger import get_logger

# The sample rates onnx-asr resamples from by itself; any other is converted.
_READABLE_RATES = frozenset({8_000, 11_025, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000})


def readable_wav(path: Path) -> bool:
    """Whether ``path`` is a WAV file the local engine reads as it is: PCM, at a
    sample rate it knows. ``wave`` reads PCM only, so anything it opens is PCM."""
    try:
        with wave.open(str(path), "rb") as f:
            return f.getsampwidth() in (1, 2, 3, 4) and f.getframerate() in _READABLE_RATES
    except (wave.Error, EOFError, OSError):
        return False


async def _ffmpeg(source: Path, target: Path) -> None:
    binary = shutil.which("ffmpeg")
    if binary is None:
        raise TranscriptionError(
            f"Reading {source.suffix or 'this'} audio needs ffmpeg, which is not installed "
            "(macOS: `brew install ffmpeg`; Debian/Ubuntu: `apt install ffmpeg`). "
            "WAV files work without it."
        )
    process = await asyncio.create_subprocess_exec(
        binary, "-nostdin", "-loglevel", "error", "-y", "-i", str(source),
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav", str(target),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )  # fmt: skip
    _, stderr = await process.communicate()
    if process.returncode != 0:
        get_logger().debug(f"ffmpeg detail: {stderr.decode(errors='replace').strip()}")
        raise TranscriptionError(
            f"ffmpeg could not read {source.name}: it is not audio, or not in a format ffmpeg knows"
        )


@asynccontextmanager
async def as_wav(audio: str | Path | bytes, filename: str | None = None) -> AsyncIterator[Path]:
    """A WAV file with ``audio`` in it, for the duration of the ``with``.

    A path to a readable WAV is used as it is; bytes are written to a temporary
    file first (named after ``filename``, so ffmpeg sees the right extension),
    and anything not readable as it is is converted. Temporary files are removed
    on the way out.
    """
    with tempfile.TemporaryDirectory(prefix="domyn-audio-") as tmp:
        if isinstance(audio, bytes):
            source = Path(tmp) / f"input{Path(filename or 'audio').suffix}"
            source.write_bytes(audio)
        else:
            source = Path(audio).expanduser()
            if not source.is_file():
                raise TranscriptionError(f"No audio file at {source}")
        if readable_wav(source):
            yield source
            return
        target = Path(tmp) / "converted.wav"
        await _ffmpeg(source, target)
        yield target
