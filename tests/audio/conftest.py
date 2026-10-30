"""Audio fixtures: WAV files written on the spot, and a stand-in for onnx-asr.

The stand-in keeps the tests off the network and off the real model (670 MB):
it records how it was loaded, and "downloads" by creating the directory it was
given, which is the behaviour of onnx-asr that ``LocalTranscriber`` relies on.
"""

from __future__ import annotations

import sys
import types
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


def write_wav(path: Path, *, seconds: float = 0.5, rate: int = 16_000, channels: int = 1) -> Path:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(b"\x00\x00" * channels * int(rate * seconds))
    return path


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    return write_wav(tmp_path / "speech.wav")


@dataclass
class FakeSegment:
    start: float
    end: float
    text: str


class FakeAsr:
    def __init__(self, engine: FakeOnnxAsr, path: str, vad: Any = None) -> None:
        self.engine = engine
        self.path = path
        self.vad = vad

    def with_vad(self, vad: Any) -> FakeAsr:
        return FakeAsr(self.engine, self.path, vad)

    def recognize(self, wav: str, **kwargs: Any) -> Any:
        self.engine.recognized.append((wav, kwargs))
        if self.engine.fail:
            raise RuntimeError("model exploded")
        if self.vad is None:
            return " ".join(self.engine.segments)
        return iter(
            FakeSegment(start=i * 1.0, end=i * 1.0 + 0.8, text=text)
            for i, text in enumerate(self.engine.segments)
        )


class FakeOnnxAsr:
    def __init__(self) -> None:
        self.loads: list[tuple[str, str, dict[str, Any]]] = []
        self.recognized: list[tuple[str, dict[str, Any]]] = []
        self.segments = [" Ciao. ", "Domani alle dieci.", "  "]
        self.fail = False
        self.fail_load = False

    def _download(self, kind: str, name: str, path: str, kwargs: dict[str, Any]) -> None:
        self.loads.append((kind, path, kwargs))
        if self.fail_load:
            raise OSError("no connection")
        target = Path(path)
        if not target.exists():
            target.mkdir(parents=True)
            (target / "model.onnx").write_text(name)

    def load_model(self, model: str, path: str, **kwargs: Any) -> FakeAsr:
        self._download("asr", model, path, kwargs)
        return FakeAsr(self, path)

    def load_vad(self, model: str, path: str, **kwargs: Any) -> str:
        self._download("vad", model, path, kwargs)
        return "vad"


@pytest.fixture
def fake_onnx_asr(monkeypatch: pytest.MonkeyPatch) -> FakeOnnxAsr:
    fake = FakeOnnxAsr()
    asr_module = types.ModuleType("onnx_asr")
    asr_module.load_model = fake.load_model  # type: ignore[attr-defined]
    asr_module.load_vad = fake.load_vad  # type: ignore[attr-defined]
    runtime = types.ModuleType("onnxruntime")
    runtime.set_default_logger_severity = lambda level: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnx_asr", asr_module)
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    return fake
