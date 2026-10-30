"""OpenAICompatibleTranscriber against a mocked /v1/audio/transcriptions."""

import httpx
import pytest
import respx

from domyn.agents.audio import OpenAICompatibleTranscriber, TranscriptionError, resolve_transcriber

URL = "http://vllm.test/v1/audio/transcriptions"

VERBOSE = {
    "text": " Ciao. Domani alle dieci. ",
    "language": "italian",
    "duration": 3.2,
    "segments": [
        {"id": 0, "start": 0.0, "end": 0.9, "text": " Ciao."},
        {"id": 1, "start": 1.0, "end": 3.1, "text": " Domani alle dieci."},
    ],
}


def transcriber(**kwargs: object) -> OpenAICompatibleTranscriber:
    return OpenAICompatibleTranscriber(url=URL, model="openai/whisper-large-v3", **kwargs)


def form(request: httpx.Request) -> bytes:
    return request.read()


class TestTheRequest:
    @pytest.mark.asyncio
    @respx.mock
    async def test_it_posts_the_file_and_model_as_multipart(self, wav):
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=VERBOSE))
        await transcriber().transcribe(wav)

        request = route.calls.last.request
        assert request.headers["content-type"].startswith("multipart/form-data")
        body = form(request)
        assert b'name="model"\r\n\r\nopenai/whisper-large-v3' in body
        assert b'name="response_format"\r\n\r\nverbose_json' in body
        assert b'filename="speech.wav"' in body and b"Content-Type: audio/x-wav" in body
        assert wav.read_bytes() in body
        assert b'name="language"' not in body
        assert "authorization" not in request.headers

    @pytest.mark.asyncio
    @respx.mock
    async def test_bytes_take_their_name_and_type_from_filename(self):
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=VERBOSE))
        await transcriber().transcribe(b"OggS...", filename="voice.ogg")
        body = form(route.calls.last.request)
        assert b'filename="voice.ogg"' in body and b"Content-Type: audio/ogg" in body

    @pytest.mark.asyncio
    @respx.mock
    async def test_a_fixed_language_is_sent(self, wav):
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=VERBOSE))
        await transcriber(language="it").transcribe(wav)
        assert b'name="language"\r\n\r\nit' in form(route.calls.last.request)

    @pytest.mark.asyncio
    @respx.mock
    async def test_the_key_is_a_bearer_token(self, wav):
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=VERBOSE))
        await transcriber(api_key="sk-test").transcribe(wav)
        assert route.calls.last.request.headers["authorization"] == "Bearer sk-test"

    @pytest.mark.asyncio
    @respx.mock
    async def test_the_key_can_come_from_the_environment(self, wav, monkeypatch):
        monkeypatch.setenv("ASR_KEY", "from-env")
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=VERBOSE))
        await transcriber(api_key_env="ASR_KEY").transcribe(wav)
        assert route.calls.last.request.headers["authorization"] == "Bearer from-env"

    @pytest.mark.asyncio
    async def test_a_named_variable_that_is_not_set_is_an_error(self, wav, monkeypatch):
        monkeypatch.delenv("ASR_KEY", raising=False)
        with pytest.raises(TranscriptionError, match=r"\$ASR_KEY"):
            await transcriber(api_key_env="ASR_KEY").transcribe(wav)


class TestTheAnswer:
    @pytest.mark.asyncio
    @respx.mock
    async def test_verbose_json_gives_language_duration_and_segments(self, wav):
        respx.post(URL).mock(return_value=httpx.Response(200, json=VERBOSE))
        result = await transcriber().transcribe(wav)
        assert result.text == "Ciao. Domani alle dieci."
        assert result.language == "italian"
        assert result.duration == 3.2
        assert [(s.start, s.end, s.text) for s in result.segments] == [
            (0.0, 0.9, "Ciao."),
            (1.0, 3.1, "Domani alle dieci."),
        ]

    @pytest.mark.asyncio
    @respx.mock
    async def test_plain_json_gives_the_text(self, wav):
        route = respx.post(URL).mock(return_value=httpx.Response(200, json={"text": "Ciao."}))
        result = await transcriber(response_format="json", language="it").transcribe(wav)
        assert b'name="response_format"\r\n\r\njson' in form(route.calls.last.request)
        assert (result.text, result.segments, result.language) == ("Ciao.", [], "it")

    @pytest.mark.asyncio
    @respx.mock
    async def test_a_refusal_names_the_status_and_not_the_body(self, wav):
        respx.post(URL).mock(
            return_value=httpx.Response(400, json={"error": "secret detail about the audio"})
        )
        with pytest.raises(TranscriptionError, match="answered 400") as raised:
            await transcriber().transcribe(wav)
        assert "secret" not in str(raised.value)

    @pytest.mark.asyncio
    @respx.mock
    async def test_an_unreachable_endpoint_is_a_transcription_error(self, wav):
        respx.post(URL).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(TranscriptionError, match="ConnectError"):
            await transcriber().transcribe(wav)

    @pytest.mark.asyncio
    @respx.mock
    async def test_an_answer_without_text_is_an_error(self, wav):
        respx.post(URL).mock(return_value=httpx.Response(200, json={"result": "?"}))
        with pytest.raises(TranscriptionError, match="without a text"):
            await transcriber().transcribe(wav)

    @pytest.mark.asyncio
    async def test_a_missing_file_is_named(self, tmp_path):
        with pytest.raises(TranscriptionError, match="No audio file"):
            await transcriber().transcribe(tmp_path / "nowhere.wav")


def test_it_is_declared_as_type_openai() -> None:
    t = resolve_transcriber({"type": "openai", "url": URL, "model": "m", "language": "en"})
    assert isinstance(t, OpenAICompatibleTranscriber)
    assert t.describe() == f"openai (m at {URL})"
    assert t.is_ready()
