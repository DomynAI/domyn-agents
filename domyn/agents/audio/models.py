"""What a transcription returns."""

from pydantic import BaseModel, Field


class TranscriptionError(RuntimeError):
    """Audio could not be transcribed: unreadable, an engine missing, or refused.

    The message is written for the person who has to fix it (install ffmpeg,
    install the ``audio`` extra, check the endpoint), never with the audio's
    content or a response body, which go to the debug log.
    """


class Segment(BaseModel):
    """A stretch of speech, with where it is in the audio."""

    start: float = Field(description="Start, in seconds from the beginning of the audio")
    end: float = Field(description="End, in seconds from the beginning of the audio")
    text: str


class Transcript(BaseModel):
    """The text of some audio.

    ``segments`` is empty when the engine reports none (a plain ``json`` answer
    from an endpoint, or a local model run without voice activity detection).
    ``language`` is what the engine detected or was told, when it says.
    """

    text: str
    language: str | None = None
    duration: float | None = Field(default=None, description="Length of the audio, in seconds")
    segments: list[Segment] = Field(default_factory=list)
