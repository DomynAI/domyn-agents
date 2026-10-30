"""Transcription by an endpoint speaking OpenAI's ``/v1/audio/transcriptions``.

The same contract is served by OpenAI, by vLLM for a speech model (Whisper and
others), and by most gateways in front of either, so one class covers them: the
URL and the model say which. The audio is sent as it came; converting it is the
server's job, which is why this one needs no ffmpeg.
"""

import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import Field, SecretStr

from domyn.agents.audio.base import AudioSource, Transcriber
from domyn.agents.audio.models import Segment, Transcript, TranscriptionError
from domyn.agents.logger import get_logger


class OpenAICompatibleTranscriber(Transcriber):
    """Speech to text by an OpenAI-compatible endpoint: OpenAI, vLLM, a gateway."""

    type: str = "openai"
    url: str = Field(
        default="https://api.openai.com/v1/audio/transcriptions",
        description="The full transcriptions URL, `.../v1/audio/transcriptions`",
    )
    model: str = Field(default="whisper-1", description="The model name the endpoint expects")
    api_key: SecretStr | None = Field(default=None, description="Sent as a Bearer token")
    api_key_env: str | None = Field(
        default=None,
        description="Environment variable holding the key, read when a request is "
        "sent; used when `api_key` is not set. Neither: no Authorization header "
        "(a vLLM without --api-key)",
    )
    response_format: Literal["verbose_json", "json"] = Field(
        default="verbose_json",
        description="`verbose_json` also returns language, duration and segments; "
        "`json` for an endpoint or model that only knows that",
    )
    timeout: float = Field(default=300.0, description="Seconds to wait for the transcription")

    def describe(self) -> str:
        return f"openai ({self.model} at {self.url})"

    def _headers(self) -> dict[str, str]:
        key = self.api_key.get_secret_value() if self.api_key else None
        if key is None and self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise TranscriptionError(
                    f"The transcriber's key is read from ${self.api_key_env}, which is not set"
                )
        return {"Authorization": f"Bearer {key}"} if key else {}

    async def transcribe(self, audio: AudioSource, *, filename: str | None = None) -> Transcript:
        if isinstance(audio, bytes):
            content, name = audio, filename or "audio.wav"
        else:
            path = Path(audio).expanduser()
            try:
                content = path.read_bytes()
            except OSError as exc:
                raise TranscriptionError(f"No audio file at {path}") from exc
            name = filename or path.name
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        data: dict[str, str] = {"model": self.model, "response_format": self.response_format}
        if self.language:
            data["language"] = self.language

        logger = get_logger()
        logger.info(
            f"[Transcription Request]: {self.url}",
            payload={"model": self.model, "bytes": len(content), "format": mime},
        )
        started = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.url,
                    headers=self._headers(),
                    data=data,
                    files={"file": (name, content, mime)},
                )
                response.raise_for_status()
                body: dict[str, Any] = response.json()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                f"[Transcription Response]: {self.url} refused the audio: "
                f"{exc.response.status_code}"
            )
            logger.debug(f"transcription error detail: {exc.response.text}")
            raise TranscriptionError(
                f"The transcription endpoint answered {exc.response.status_code}"
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning(f"[Transcription Response]: {self.url} failed: {type(exc).__name__}")
            logger.debug(f"transcription error detail: {exc}")
            raise TranscriptionError(
                f"The transcription endpoint could not be reached or answered with "
                f"something else than JSON ({type(exc).__name__})"
            ) from exc

        transcript = _parse(body, self.language)
        logger.info(
            f"[Transcription Response]: {self.url}",
            payload={
                "execution_time": round(time.monotonic() - started, 3),
                "text_length": len(transcript.text),
                "segments": len(transcript.segments),
                "language": transcript.language,
            },
        )
        logger.debug(f"[Transcription Response] full payload: {self.url}", payload=body)
        return transcript


def _parse(body: dict[str, Any], language: str | None) -> Transcript:
    """``json`` (text only) or ``verbose_json`` (with language, duration, segments)."""
    if not isinstance(body.get("text"), str):
        raise TranscriptionError("The transcription endpoint answered without a text")
    segments = [
        Segment(start=float(s["start"]), end=float(s["end"]), text=str(s["text"]).strip())
        for s in body.get("segments") or []
        if isinstance(s, dict) and {"start", "end", "text"} <= s.keys()
    ]
    duration = body.get("duration")
    return Transcript(
        text=body["text"].strip(),
        # Whisper's verbose_json names the language in full ("italian"); kept as
        # the endpoint wrote it rather than guessed into a code.
        language=body.get("language") or language,
        duration=float(duration) if duration is not None else None,
        segments=segments,
    )
