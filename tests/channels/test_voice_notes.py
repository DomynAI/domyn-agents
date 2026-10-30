"""Voice notes in the chat channels: transcribed into text the agent reads.

Also pins the attachments themselves, which were built with a field `File` no
longer has (`uri`, renamed `file_id`) and so never reached the agent at all.
"""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx
from pydantic import Field

from domyn.agents.audio import Transcriber, Transcript, TranscriptionError
from domyn.agents.channels import _voice
from domyn.agents.channels._voice import transcribe_voice
from domyn.agents.core import BaseEvent, ExecutionEventType, File, FileType, Part

AUDIO_URL = "https://files.test/voice/file_7.oga"


class FakeTranscriber(Transcriber):
    type: str = "fake-voice-test"
    text: str = "Domani alle dieci."
    fail: bool = False
    calls: list[tuple[bytes, str | None]] = Field(default_factory=list)

    async def transcribe(self, audio, *, filename=None):  # type: ignore[no-untyped-def]
        self.calls.append((audio, filename))
        if self.fail:
            raise TranscriptionError("engine broke")
        return Transcript(text=self.text, language="it", duration=1.5)


def audio_part(url: str = AUDIO_URL) -> Part:
    return Part(file=File(file_id=url, type=FileType.AUDIO))


class TestTranscribeVoice:
    @pytest.mark.asyncio
    @respx.mock
    async def test_the_transcription_follows_the_audio_which_stays(self):
        respx.get(AUDIO_URL).mock(return_value=httpx.Response(200, content=b"OggS-audio"))
        transcriber = FakeTranscriber()
        parts = await transcribe_voice([audio_part(), Part(text="caption")], transcriber)

        assert parts[0].file is not None and parts[0].file.file_id == AUDIO_URL
        assert parts[1].text == "[Voice message, transcribed] Domani alle dieci."
        assert parts[1].metadata == {
            "transcribed_from": "audio",
            "language": "it",
            "duration": 1.5,
        }
        assert parts[2].text == "caption"
        assert transcriber.calls == [(b"OggS-audio", "file_7.oga")]

    @pytest.mark.asyncio
    @respx.mock
    async def test_the_channel_headers_are_sent_with_the_download(self):
        route = respx.get(AUDIO_URL).mock(return_value=httpx.Response(200, content=b"x"))
        await transcribe_voice([audio_part()], FakeTranscriber(), {"Authorization": "Bearer tok"})
        assert route.calls.last.request.headers["authorization"] == "Bearer tok"

    @pytest.mark.asyncio
    async def test_without_a_transcriber_the_agent_is_told_why(self):
        parts = await transcribe_voice([audio_part()], None)
        assert parts[1].text == _voice.NO_TRANSCRIBER
        assert "write instead" in parts[1].text

    @pytest.mark.asyncio
    @respx.mock
    async def test_a_failed_transcription_is_said_not_swallowed(self):
        respx.get(AUDIO_URL).mock(return_value=httpx.Response(200, content=b"x"))
        parts = await transcribe_voice([audio_part()], FakeTranscriber(fail=True))
        assert parts[1].text == _voice.FAILED

    @pytest.mark.asyncio
    @respx.mock
    async def test_a_failed_download_is_said_too(self):
        respx.get(AUDIO_URL).mock(return_value=httpx.Response(403))
        parts = await transcribe_voice([audio_part()], FakeTranscriber())
        assert parts[1].text == _voice.FAILED

    @pytest.mark.asyncio
    @respx.mock
    async def test_silence_is_named(self):
        respx.get(AUDIO_URL).mock(return_value=httpx.Response(200, content=b"x"))
        parts = await transcribe_voice([audio_part()], FakeTranscriber(text=""))
        assert parts[1].text == _voice.SILENT

    @pytest.mark.asyncio
    @respx.mock
    async def test_an_oversized_file_is_not_downloaded_whole(self, monkeypatch):
        monkeypatch.setattr(_voice, "MAX_AUDIO_BYTES", 10)
        respx.get(AUDIO_URL).mock(return_value=httpx.Response(200, content=b"x" * 50))
        transcriber = FakeTranscriber()
        parts = await transcribe_voice([audio_part()], transcriber)
        assert parts[1].text == _voice.FAILED and transcriber.calls == []

    @pytest.mark.asyncio
    async def test_other_parts_pass_through(self):
        image = Part(file=File(file_id="https://files.test/a.png", type=FileType.IMAGE))
        parts = await transcribe_voice([Part(text="hi"), image], None)
        assert parts == [Part(text="hi"), image]

    def test_a_url_without_extension_gets_one(self):
        assert _voice._filename("https://lookaside.fbsbx.com/whatsapp?mid=1") == "voice.ogg"
        assert _voice._filename("https://x/y/clip.m4a?t=1") == "clip.m4a"

    def test_a_configuration_is_built_into_a_transcriber(self):
        built = _voice.as_transcriber({"type": "fake-voice-test", "text": "x"})
        assert isinstance(built, FakeTranscriber) and built.text == "x"
        assert _voice.as_transcriber(None) is None


class TestTelegram:
    @pytest.mark.asyncio
    async def test_a_voice_note_becomes_a_file_part(self):
        from domyn.agents.channels.telegram import _message_to_parts

        async def get_file(file_id: str) -> Any:
            return SimpleNamespace(file_path=f"https://api.telegram.test/file/bot/{file_id}.oga")

        bot = SimpleNamespace(get_file=get_file)
        message = SimpleNamespace(
            caption=None, text=None, photo=None, voice=SimpleNamespace(file_id="v1"),
            audio=None, document=None,
        )  # fmt: skip
        parts = await _message_to_parts(bot, message)  # type: ignore[arg-type]
        assert parts == [
            Part(
                file=File(file_id="https://api.telegram.test/file/bot/v1.oga", type=FileType.AUDIO)
            )
        ]

    @pytest.mark.asyncio
    async def test_a_file_telegram_will_not_serve_is_skipped(self):
        from domyn.agents.channels.telegram import _message_to_parts

        async def get_file(file_id: str) -> Any:
            return SimpleNamespace(file_path=None)

        message = SimpleNamespace(
            caption="hello", text=None, photo=None, voice=SimpleNamespace(file_id="v1"),
            audio=None, document=None,
        )  # fmt: skip
        parts = await _message_to_parts(SimpleNamespace(get_file=get_file), message)  # type: ignore[arg-type]
        assert parts == [Part(text="hello")]

    @pytest.mark.asyncio
    @respx.mock
    async def test_receive_yields_the_transcription(self):
        from domyn.agents.channels import TelegramChannel

        url = "https://api.telegram.test/file/botTOKEN/voice.oga"
        respx.get(url).mock(return_value=httpx.Response(200, content=b"OggS"))
        channel = TelegramChannel(token="TOKEN", transcriber=FakeTranscriber())

        class FakeBot:
            def __init__(self, token: str) -> None:
                self.calls = 0

            async def get_updates(self, **kwargs: Any) -> list[Any]:
                self.calls += 1
                if self.calls > 1:
                    await channel.close()
                    return []
                message = SimpleNamespace(
                    chat_id=42, caption=None, text=None, photo=None,
                    voice=SimpleNamespace(file_id="v1"), audio=None, document=None,
                )  # fmt: skip
                return [SimpleNamespace(update_id=1, message=message)]

            async def get_file(self, file_id: str) -> Any:
                return SimpleNamespace(file_path=url)

        with patch("telegram.Bot", FakeBot):
            received = [item async for item in channel.receive()]

        assert len(received) == 1
        conversation_id, event = received[0]
        assert conversation_id == "42"
        assert event.content[1].text == "[Voice message, transcribed] Domani alle dieci."


class TestWhatsApp:
    @pytest.mark.asyncio
    @respx.mock
    async def test_a_voice_note_is_resolved_downloaded_with_the_token_and_transcribed(self):
        from domyn.agents.channels.whatsapp import _GRAPH_API, WhatsAppChannel

        media_url = "https://lookaside.fbsbx.test/whatsapp_business/attachments/?mid=9"
        respx.get(f"{_GRAPH_API}/media-9").mock(
            return_value=httpx.Response(200, json={"url": media_url})
        )
        download = respx.get(media_url).mock(return_value=httpx.Response(200, content=b"OggS"))
        transcriber = FakeTranscriber()
        channel = WhatsAppChannel(
            token="wa-tok", phone_number_id="p", verify_token="v", transcriber=transcriber
        )

        parts = await channel._message_to_parts({"type": "audio", "audio": {"id": "media-9"}})

        assert parts[0] == Part(file=File(file_id=media_url, type=FileType.AUDIO))
        assert parts[1].text == "[Voice message, transcribed] Domani alle dieci."
        assert download.calls.last.request.headers["authorization"] == "Bearer wa-tok"
        assert transcriber.calls == [(b"OggS", "voice.ogg")]

    @pytest.mark.asyncio
    async def test_text_is_untouched(self):
        from domyn.agents.channels import WhatsAppChannel

        channel = WhatsAppChannel(token="t", phone_number_id="p", verify_token="v")
        parts = await channel._message_to_parts({"type": "text", "text": {"body": "ciao"}})
        assert parts == [Part(text="ciao")]


class TestSlack:
    def test_an_audio_clip_becomes_a_file_part(self):
        from domyn.agents.channels.slack import _message_to_parts

        event = {
            "files": [
                {"mimetype": "audio/webm", "url_private": "https://files.slack.test/clip.webm"}
            ]
        }
        parts = _message_to_parts(event, "")
        assert parts == [
            Part(file=File(file_id="https://files.slack.test/clip.webm", type=FileType.AUDIO))
        ]

    def test_the_transcriber_is_kept(self):
        from domyn.agents.channels import SlackChannel

        transcriber = FakeTranscriber()
        channel = SlackChannel(bot_token="b", app_token="a", transcriber=transcriber)
        assert channel.transcriber is transcriber


def test_an_outgoing_file_is_read_from_file_id() -> None:
    """The send side read `File.uri` too: an agent's file reply raised."""
    from domyn.agents.channels.telegram import _send_file

    sent: dict[str, Any] = {}

    async def send_audio(**kwargs: Any) -> None:
        sent.update(kwargs)

    bot = SimpleNamespace(send_audio=send_audio)
    asyncio.run(
        _send_file(bot, 1, File(file_id="https://x/a.mp3", type=FileType.AUDIO), caption=None)  # type: ignore[arg-type]
    )
    assert sent["audio"] == "https://x/a.mp3"


def test_events_still_carry_the_parts() -> None:
    event = BaseEvent(
        event_type=ExecutionEventType.AGENT_START, author="user", content=[audio_part()]
    )
    assert event.content[0].file is not None
