"""Speech to text: the ``Transcriber`` and its implementations.

Light to import: the speech model's libraries (``onnx-asr``, ``onnxruntime``) are
loaded the first time a ``LocalTranscriber`` is used, and installed with the
``audio`` extra, so nothing here costs a process that never transcribes.
"""

from domyn.agents.audio.base import Transcriber, resolve_transcriber
from domyn.agents.audio.local import LocalTranscriber
from domyn.agents.audio.models import Segment, Transcript, TranscriptionError
from domyn.agents.audio.openai_compatible import OpenAICompatibleTranscriber

__all__ = [
    "LocalTranscriber",
    "OpenAICompatibleTranscriber",
    "Segment",
    "Transcriber",
    "Transcript",
    "TranscriptionError",
    "resolve_transcriber",
]
