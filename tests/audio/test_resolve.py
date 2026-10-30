"""resolve_transcriber, and a transcriber of one's own."""

import pytest

from domyn.agents.audio import Transcriber, Transcript, resolve_transcriber


class EchoTranscriber(Transcriber):
    type: str = "echo-test"

    async def transcribe(self, audio, *, filename=None):  # type: ignore[no-untyped-def]
        return Transcript(text=f"{filename}:{len(audio)}")


def test_a_subclass_is_found_by_its_type() -> None:
    t = resolve_transcriber({"type": "echo-test", "language": "it"})
    assert isinstance(t, EchoTranscriber) and t.language == "it"


def test_an_instance_is_returned_as_it_is() -> None:
    t = EchoTranscriber()
    assert resolve_transcriber(t) is t


def test_an_unknown_type_lists_the_known_ones() -> None:
    with pytest.raises(ValueError, match="unknown transcriber type 'whisperx'") as raised:
        resolve_transcriber({"type": "whisperx"})
    assert "local" in str(raised.value) and "openai" in str(raised.value)


def test_a_missing_type_is_refused() -> None:
    with pytest.raises(ValueError, match="missing 'type'"):
        resolve_transcriber({"model": "x"})


@pytest.mark.asyncio
async def test_the_defaults_of_the_base_class() -> None:
    t = EchoTranscriber()
    assert t.is_ready() and t.describe() == "echo-test" and t.language is None
    await t.prepare(lambda line: pytest.fail("nothing to download"))
    assert (await t.transcribe(b"abc", filename="x.wav")).text == "x.wav:3"


def test_importing_the_package_loads_no_speech_engine() -> None:
    import subprocess
    import sys

    code = (
        "import sys, domyn.agents.audio; "
        "print(any(m.split('.')[0] in ('onnx_asr', 'onnxruntime') for m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
