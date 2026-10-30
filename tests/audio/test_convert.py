"""as_wav: what goes to the local engine as it is, and what needs ffmpeg."""

from pathlib import Path

import pytest

from domyn.agents.audio import TranscriptionError
from domyn.agents.audio.convert import as_wav, readable_wav
from tests.audio.conftest import write_wav


class TestReadableWav:
    def test_pcm_at_a_known_rate_is_read_as_it_is(self, wav):
        assert readable_wav(wav)

    def test_an_unknown_rate_is_not(self, tmp_path):
        assert not readable_wav(write_wav(tmp_path / "odd.wav", rate=12_345))

    def test_something_else_named_wav_is_not(self, tmp_path):
        fake = tmp_path / "fake.wav"
        fake.write_bytes(b"OggS not a wav at all")
        assert not readable_wav(fake)


class TestAsWav:
    @pytest.mark.asyncio
    async def test_a_readable_wav_path_is_used_directly(self, wav):
        async with as_wav(wav) as path:
            assert path == wav

    @pytest.mark.asyncio
    async def test_bytes_go_through_a_temporary_file_removed_afterwards(self, wav):
        async with as_wav(wav.read_bytes(), "voice.wav") as path:
            assert path.suffix == ".wav" and path.read_bytes() == wav.read_bytes()
        assert not path.exists()

    @pytest.mark.asyncio
    async def test_a_missing_file_is_named(self, tmp_path):
        with pytest.raises(TranscriptionError, match="No audio file"):
            async with as_wav(tmp_path / "nowhere.ogg"):
                pass

    @pytest.mark.asyncio
    async def test_without_ffmpeg_another_format_says_what_to_install(self, tmp_path, monkeypatch):
        monkeypatch.setattr("domyn.agents.audio.convert.shutil.which", lambda name: None)
        note = tmp_path / "voice.ogg"
        note.write_bytes(b"OggS...")
        with pytest.raises(TranscriptionError, match="needs ffmpeg") as raised:
            async with as_wav(note):
                pass
        assert ".ogg" in str(raised.value)

    @pytest.mark.asyncio
    async def test_something_ffmpeg_cannot_read_is_reported(self, tmp_path):
        import shutil

        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not installed")
        junk = tmp_path / "junk.mp3"
        junk.write_bytes(b"\x00" * 64)
        with pytest.raises(TranscriptionError, match=r"ffmpeg could not read junk\.mp3"):
            async with as_wav(junk):
                pass


def test_path_strings_are_accepted(wav: Path) -> None:
    import asyncio

    async def check() -> Path:
        async with as_wav(str(wav)) as path:
            return path

    assert asyncio.run(check()) == wav
