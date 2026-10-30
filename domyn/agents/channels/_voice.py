"""Voice notes, for the chat channels: the audio becomes text the agent reads.

A model reads none of the audio a channel attaches, so without this a voice note
reaches the agent as a file it cannot open and gets no meaningful answer. Every
audio ``Part`` is followed by a text ``Part`` with its transcription, or, when
there is none, with a line saying why, so the agent can tell the user rather
than answer something else; the audio itself stays attached.

Each channel downloads its files its own way (Telegram puts the token in the
URL, WhatsApp and Slack want it as a header), so it passes the headers; the
rest is the same for all of them.
"""

from collections.abc import Mapping
from typing import Any

import httpx

from domyn.agents.audio import Transcriber, TranscriptionError, resolve_transcriber
from domyn.agents.core import FileType, Part
from domyn.agents.logger import get_logger

# Larger than any voice note (Telegram caps bot downloads at 20 MB, WhatsApp
# audio at 16 MB), small enough that a stray attachment cannot fill memory.
MAX_AUDIO_BYTES = 25 * 1024 * 1024

VOICE_PREFIX = "[Voice message, transcribed] "
NO_TRANSCRIBER = (
    "[The user sent a voice message. It was not transcribed: this channel has no "
    "transcriber configured, so tell the user to write instead.]"
)
FAILED = "[The user sent a voice message, which could not be transcribed.]"
SILENT = "[The user sent a voice message with no speech in it.]"


def as_transcriber(value: Transcriber | Mapping[str, Any] | None) -> Transcriber | None:
    """A channel's ``transcriber`` argument: an instance, its configuration, or none."""
    if value is None or isinstance(value, Transcriber):
        return value
    return resolve_transcriber(dict(value))


async def download(url: str, headers: Mapping[str, str] | None = None) -> bytes:
    """The bytes at ``url``, refusing anything over ``MAX_AUDIO_BYTES``."""
    async with (
        httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client,
        client.stream("GET", url, headers=dict(headers or {})) as response,
    ):
        response.raise_for_status()
        chunks: list[bytes] = []
        size = 0
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > MAX_AUDIO_BYTES:
                raise TranscriptionError(f"The audio is larger than {MAX_AUDIO_BYTES} bytes")
            chunks.append(chunk)
    return b"".join(chunks)


def _filename(url: str) -> str:
    """A name for the downloaded bytes; only its extension matters (ffmpeg reads
    the content anyway). WhatsApp's media URLs carry none."""
    name = url.split("?", 1)[0].rsplit("/", 1)[-1]
    return name if "." in name else "voice.ogg"


async def transcribe_voice(
    parts: list[Part],
    transcriber: Transcriber | None,
    headers: Mapping[str, str] | None = None,
) -> list[Part]:
    """``parts`` with a text ``Part`` after every audio one: its transcription,
    or why there is none. Other parts pass through unchanged."""
    out: list[Part] = []
    for part in parts:
        out.append(part)
        if part.file is None or part.file.type != FileType.AUDIO:
            continue
        out.append(await _transcribe_one(part.file.file_id, transcriber, headers))
    return out


async def _transcribe_one(
    url: str, transcriber: Transcriber | None, headers: Mapping[str, str] | None
) -> Part:
    if transcriber is None:
        return Part(text=NO_TRANSCRIBER)
    logger = get_logger()
    try:
        audio = await download(url, headers)
        transcript = await transcriber.transcribe(audio, filename=_filename(url))
    except (TranscriptionError, httpx.HTTPError) as exc:
        logger.warning(f"voice note: not transcribed: {type(exc).__name__}")
        logger.debug(f"voice note detail: {exc}")
        return Part(text=FAILED)
    if not transcript.text:
        return Part(text=SILENT)
    logger.info(
        f"voice note: transcribed {len(audio)} bytes into {len(transcript.text)} characters"
    )
    return Part(
        text=VOICE_PREFIX + transcript.text,
        metadata={
            "transcribed_from": "audio",
            "language": transcript.language,
            "duration": transcript.duration,
        },
    )
