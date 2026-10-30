"""LocalTranscriber: loaded only when used, downloaded once and whole, and what
it makes of the engine's output."""

import builtins
import shutil
import sys
from pathlib import Path

import pytest

from domyn.agents.audio import LocalTranscriber, TranscriptionError, local, resolve_transcriber
from domyn.agents.audio.local import _default_models_dir
from tests.audio.conftest import FakeOnnxAsr, write_wav

_REAL_DEFAULT = _default_models_dir  # before the suite-wide fixture replaces it


def transcriber(tmp_path: Path, **kwargs: object) -> LocalTranscriber:
    return LocalTranscriber(models_dir=tmp_path / "models", **kwargs)


class TestNothingHappensBeforeItIsUsed:
    def test_building_one_loads_and_downloads_nothing(self, tmp_path, fake_onnx_asr):
        t = transcriber(tmp_path)
        assert fake_onnx_asr.loads == []
        assert not (tmp_path / "models").exists()
        assert not t.is_ready()

    def test_the_default_is_parakeet_int8_with_vad_under_domyn(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr(local, "_default_models_dir", _REAL_DEFAULT)
        t = resolve_transcriber({"type": "local"})
        assert isinstance(t, LocalTranscriber)
        assert (t.model, t.quantization, t.vad) == ("nemo-parakeet-tdt-0.6b-v3", "int8", True)
        assert t.models_dir == tmp_path / ".domyn" / "models"
        assert t.model_path.name == "nemo-parakeet-tdt-0.6b-v3-int8"
        assert t.describe() == "local (nemo-parakeet-tdt-0.6b-v3, int8)"


class TestTheDownload:
    @pytest.mark.asyncio
    async def test_the_first_use_downloads_says_so_and_ends_in_place(self, tmp_path, fake_onnx_asr):
        t = transcriber(tmp_path)
        said: list[str] = []
        await t.prepare(said.append)

        assert len(said) == 1
        assert "nemo-parakeet-tdt-0.6b-v3" in said[0] and "670 MB" in said[0]
        assert t.model_path.is_dir() and t.vad_path.is_dir()
        assert t.is_ready()
        # Downloaded into a temporary directory beside the final one, never into it.
        kinds = {kind: Path(path) for kind, path, _ in fake_onnx_asr.loads}
        assert kinds["asr"].name.startswith(".nemo-parakeet-tdt-0.6b-v3-int8.partial-")
        assert kinds["vad"].name.startswith(".silero.partial-")
        assert sorted(p.name for p in (tmp_path / "models").iterdir()) == [
            "nemo-parakeet-tdt-0.6b-v3-int8",
            "silero",
        ]

    @pytest.mark.asyncio
    async def test_a_model_already_there_is_read_and_nothing_is_announced(
        self, tmp_path, fake_onnx_asr
    ):
        await transcriber(tmp_path).prepare()
        fake_onnx_asr.loads.clear()

        again = transcriber(tmp_path)
        assert again.is_ready()
        said: list[str] = []
        await again.prepare(said.append)
        assert said == []
        assert [Path(path) for _, path, _ in fake_onnx_asr.loads] == [
            again.model_path,
            again.vad_path,
        ]

    @pytest.mark.asyncio
    async def test_a_failed_download_leaves_nothing_that_looks_like_a_model(
        self, tmp_path, fake_onnx_asr
    ):
        fake_onnx_asr.fail_load = True
        t = transcriber(tmp_path)
        with pytest.raises(TranscriptionError, match=r"huggingface\.co"):
            await t.prepare()
        assert not t.is_ready()
        assert list((tmp_path / "models").iterdir()) == []

    @pytest.mark.asyncio
    async def test_it_is_loaded_once_per_transcriber(self, tmp_path, fake_onnx_asr, wav):
        t = transcriber(tmp_path)
        await t.transcribe(wav)
        await t.transcribe(wav)
        assert len(fake_onnx_asr.loads) == 2  # the model and the VAD, once each

    @pytest.mark.asyncio
    async def test_without_vad_only_the_model_is_fetched(self, tmp_path, fake_onnx_asr):
        t = transcriber(tmp_path, vad=False)
        await t.prepare()
        assert [kind for kind, _, _ in fake_onnx_asr.loads] == ["asr"]
        assert not t.vad_path.exists()

    @pytest.mark.asyncio
    async def test_it_runs_on_the_cpu(self, tmp_path, fake_onnx_asr):
        await transcriber(tmp_path).prepare()
        assert all(kw["providers"] == ["CPUExecutionProvider"] for _, _, kw in fake_onnx_asr.loads)

    @pytest.mark.asyncio
    async def test_a_missing_extra_says_how_to_install_it(self, tmp_path, monkeypatch, wav):
        monkeypatch.delitem(sys.modules, "onnx_asr", raising=False)
        real_import = builtins.__import__

        def no_onnx(name, *args, **kwargs):
            if name == "onnx_asr":
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_onnx)
        with pytest.raises(TranscriptionError, match=r"domyn\[audio\]"):
            await transcriber(tmp_path).transcribe(wav)


class TestTranscribing:
    @pytest.mark.asyncio
    async def test_segments_carry_their_times_and_empty_ones_are_dropped(
        self, tmp_path, fake_onnx_asr, wav
    ):
        result = await transcriber(tmp_path).transcribe(wav)
        assert result.text == "Ciao. Domani alle dieci."
        assert [(s.start, s.end, s.text) for s in result.segments] == [
            (0.0, 0.8, "Ciao."),
            (1.0, 1.8, "Domani alle dieci."),
        ]
        assert result.duration == pytest.approx(0.5)  # the audio's, not the last segment's

    @pytest.mark.asyncio
    async def test_without_vad_there_is_text_and_no_segments(self, tmp_path, fake_onnx_asr, wav):
        result = await transcriber(tmp_path, vad=False).transcribe(wav)
        assert result.segments == []
        assert result.text == "Ciao.  Domani alle dieci."

    @pytest.mark.asyncio
    async def test_a_language_is_passed_on_and_reported(self, tmp_path, fake_onnx_asr, wav):
        result = await transcriber(tmp_path, language="it").transcribe(wav)
        assert fake_onnx_asr.recognized[-1][1] == {"channel": "mean", "language": "it"}
        assert result.language == "it"

    @pytest.mark.asyncio
    async def test_no_language_means_the_model_detects_it(self, tmp_path, fake_onnx_asr, wav):
        await transcriber(tmp_path).transcribe(wav)
        assert fake_onnx_asr.recognized[-1][1] == {"channel": "mean"}

    @pytest.mark.asyncio
    async def test_bytes_are_transcribed_like_a_file(self, tmp_path, fake_onnx_asr, wav):
        result = await transcriber(tmp_path).transcribe(wav.read_bytes(), filename="note.wav")
        assert result.text == "Ciao. Domani alle dieci."

    @pytest.mark.asyncio
    async def test_a_stereo_file_is_read_as_one_channel(self, tmp_path, fake_onnx_asr):
        stereo = write_wav(tmp_path / "stereo.wav", channels=2, rate=44_100)
        await transcriber(tmp_path).transcribe(stereo)
        # Read as it is (onnx-asr resamples 44.1 kHz itself), channels averaged.
        assert fake_onnx_asr.recognized[-1] == (str(stereo), {"channel": "mean"})

    @pytest.mark.asyncio
    async def test_an_engine_failure_is_a_transcription_error(self, tmp_path, fake_onnx_asr, wav):
        fake_onnx_asr.fail = True
        with pytest.raises(TranscriptionError, match="RuntimeError"):
            await transcriber(tmp_path).transcribe(wav)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
class TestOtherFormats:
    @pytest.mark.asyncio
    async def test_an_mp3_is_converted_to_wav_first(self, tmp_path, fake_onnx_asr, wav):
        import subprocess

        mp3 = tmp_path / "note.mp3"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(wav), str(mp3)], check=True)
        await transcriber(tmp_path).transcribe(mp3)
        converted = Path(fake_onnx_asr.recognized[-1][0])
        assert converted.suffix == ".wav" and converted != wav
        assert not converted.exists()  # the temporary copy is gone afterwards


def test_the_suite_never_points_at_the_real_models_directory() -> None:
    assert LocalTranscriber().models_dir != Path.home() / ".domyn" / "models"
