"""Transcription on this machine, with onnx-asr.

The default model is NVIDIA's Parakeet TDT 0.6B v3 (CC-BY-4.0), int8: 25 European
languages, Italian and English among them, detected by the model itself, with
punctuation, at about 670 MB. Silero splits the audio on its pauses first (voice
activity detection), which is what lets an hour of meeting through a model that
reads a few seconds at a time, and gives the segments their times.

Nothing is downloaded or loaded until the first ``transcribe`` (or ``prepare``):
installing the ``audio`` extra costs no download, and a session that never
transcribes loads nothing. A download goes into a temporary directory renamed
into place when complete, so an interrupted one is never taken for a model.
"""

import asyncio
import shutil
import threading
import uuid
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import Field, PrivateAttr

from domyn.agents.audio.base import AudioSource, Transcriber
from domyn.agents.audio.convert import as_wav
from domyn.agents.audio.models import Segment, Transcript, TranscriptionError
from domyn.agents.logger import get_logger

DEFAULT_MODEL = "nemo-parakeet-tdt-0.6b-v3"
VAD_MODEL = "silero"

# What a download weighs, for the line announcing it. Only the models we
# suggest: for another one the line names it without a size.
_APPROX_SIZE = {
    (DEFAULT_MODEL, "int8"): "670 MB",
    (DEFAULT_MODEL, None): "2.5 GB",
}

# CPU only. onnxruntime also offers CoreML on macOS, which takes a fraction of
# each graph and was slower in every measure on an Apple Silicon laptop
# (Parakeet int8: load 5.6 s against 1.1 s, 7 s of speech in 0.9 s against
# 0.3 s), besides printing a line per call on stderr.
_PROVIDERS = ["CPUExecutionProvider"]

_INSTALL_HINT = (
    "Local transcription needs the audio extra: `uv add 'domyn[audio]'` "
    "or `pip install 'domyn[audio]'` (Python 3.11 to 3.14)."
)


def _default_models_dir() -> Path:
    return Path.home() / ".domyn" / "models"


class LocalTranscriber(Transcriber):
    """Speech to text on this machine: no audio leaves it."""

    type: str = "local"
    model: str = Field(
        default=DEFAULT_MODEL,
        description="An onnx-asr model name (`nemo-parakeet-tdt-0.6b-v3`, "
        "`onnx-community/whisper-large-v3-turbo`...) or a Hugging Face repository",
    )
    quantization: str | None = Field(
        default="int8", description="`int8` (smaller, the default) or None for full precision"
    )
    vad: bool = Field(
        default=True,
        description="Split the audio on pauses before transcribing it: needed for "
        "anything longer than a short clip, and what gives the segments",
    )
    models_dir: Path = Field(
        # Looked up when an instance is built, not bound here, so the test suite
        # can point it away from the real home directory.
        default_factory=lambda: _default_models_dir(),
        description="Where models are downloaded to, one directory each",
    )

    _engine: Any = PrivateAttr(default=None)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def _dir_for(self, model: str, quantization: str | None) -> Path:
        name = model.replace("/", "--")
        return self.models_dir.expanduser() / (f"{name}-{quantization}" if quantization else name)

    @property
    def model_path(self) -> Path:
        return self._dir_for(self.model, self.quantization)

    @property
    def vad_path(self) -> Path:
        return self._dir_for(VAD_MODEL, None)

    def _missing(self) -> list[Path]:
        wanted = [self.model_path, *([self.vad_path] if self.vad else [])]
        return [path for path in wanted if not path.is_dir()]

    def is_ready(self) -> bool:
        return self._engine is not None or not self._missing()

    def describe(self) -> str:
        quantized = f", {self.quantization}" if self.quantization else ""
        return f"local ({self.model}{quantized})"

    async def prepare(self, on_download: Callable[[str], None] | None = None) -> None:
        if self._engine is not None:
            return
        if on_download is not None and self._missing():
            size = _APPROX_SIZE.get((self.model, self.quantization))
            weight = f" (about {size})" if size else ""
            on_download(
                f"Downloading the speech model {self.model}{weight} to "
                f"{self.models_dir.expanduser()}: only this once."
            )
        await asyncio.to_thread(self._load)

    def _load(self) -> None:
        with self._lock:
            if self._engine is not None:
                return
            try:
                import onnx_asr
                import onnxruntime
            except ImportError as exc:
                raise TranscriptionError(_INSTALL_HINT) from exc
            # onnxruntime warns on stderr about every graph node an accelerator
            # cannot take, which is most of them and nothing anyone can act on.
            onnxruntime.set_default_logger_severity(3)

            def load_asr(path: str) -> Any:
                return onnx_asr.load_model(
                    self.model, path, quantization=self.quantization, providers=_PROVIDERS
                )

            def load_vad(path: str) -> Any:
                return onnx_asr.load_vad(VAD_MODEL, path, providers=_PROVIDERS)

            try:
                engine = self._fetch(self.model_path, load_asr)
                if self.vad:
                    engine = engine.with_vad(self._fetch(self.vad_path, load_vad))
            except TranscriptionError:
                raise
            except Exception as exc:
                get_logger().warning(
                    f"audio: could not load the speech model {self.model}: {type(exc).__name__}"
                )
                get_logger().debug(f"audio model load detail: {exc}")
                raise TranscriptionError(
                    f"Could not load the speech model {self.model} ({type(exc).__name__}). "
                    "A download needs a connection to huggingface.co the first time."
                ) from exc
            self._engine = engine

    def _fetch(self, final: Path, load: Callable[[str], Any]) -> Any:
        """Load from ``final``, downloading into it first when it is not there.

        onnx-asr downloads into a directory that does not exist yet and reads
        one that does, so the download goes to a fresh temporary directory beside
        ``final`` and is renamed into place once loaded: a directory at ``final``
        is always a complete model.
        """
        if final.is_dir():
            return load(str(final))
        final.parent.mkdir(parents=True, exist_ok=True)
        partial = final.with_name(f".{final.name}.partial-{uuid.uuid4().hex[:8]}")
        get_logger().info(f"audio: downloading {final.name} to {final.parent}")
        try:
            loaded = load(str(partial))
            try:
                partial.rename(final)
            except OSError:
                # Another process finished the same download first: its copy is
                # as good as ours.
                if not final.is_dir():
                    raise
        finally:
            shutil.rmtree(partial, ignore_errors=True)
        return loaded

    async def transcribe(self, audio: AudioSource, *, filename: str | None = None) -> Transcript:
        await self.prepare()
        async with as_wav(audio, filename) as wav:
            return await asyncio.to_thread(self._recognize, wav)

    def _recognize(self, wav: Path) -> Transcript:
        options: dict[str, Any] = {"language": self.language} if self.language else {}
        try:
            with wave.open(str(wav), "rb") as f:
                duration = f.getnframes() / f.getframerate()
            result = self._engine.recognize(str(wav), channel="mean", **options)
            if not self.vad:
                return Transcript(
                    text=str(result).strip(), language=self.language, duration=duration
                )
            segments = [
                Segment(start=float(s.start), end=float(s.end), text=s.text.strip())
                for s in result
                if s.text.strip()
            ]
        except Exception as exc:
            get_logger().warning(f"audio: transcription failed: {type(exc).__name__}")
            get_logger().debug(f"audio transcription detail: {exc}")
            raise TranscriptionError(
                f"The audio could not be transcribed ({type(exc).__name__})"
            ) from exc
        return Transcript(
            text=" ".join(s.text for s in segments),
            language=self.language,
            duration=duration,
            segments=segments,
        )
