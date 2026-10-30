"""TelegramChannel — InputChannel implementation for Telegram Bot API.

Setup
-----
1. Create a bot via @BotFather and obtain ``TELEGRAM_BOT_TOKEN``.
2. Install the dependency::

       uv add "domyn[telegram]"

3. Use as the channel in AgentServer::

       from domyn.agents.channels.telegram import TelegramChannel
       from domyn.agents.server import AgentServer

       channel = TelegramChannel(token=os.environ["TELEGRAM_BOT_TOKEN"])
       server = AgentServer(channel=channel, adapter=adapter)
       await server.serve()

How it works
------------
* ``receive()`` polls ``getUpdates`` (long-poll) and yields one ``AGENT_START``
  per incoming message.  Text messages become a text ``Part``; photos, voice
  notes, audio and documents are resolved via ``getFile`` and attached as a
  ``Part(file=File(file_id=..., type=...))`` (caption, if any, is kept as text).
  A voice note or audio file is also transcribed into a text ``Part`` after it,
  when a ``transcriber`` is given (see ``channels/_voice.py``).
* ``send_event()`` / ``send_event_priority()`` call ``sendMessage`` /
  ``sendPhoto`` / ``sendAudio`` / ``sendDocument`` depending on whether the
  RESPONSE event's content contains a ``Part.file``, when the event type is
  ``RESPONSE`` (non-partial).  All other event types are silently ignored —
  Telegram has no concept of intermediate trace events.
* ``conversation_id`` is the Telegram ``chat_id`` (as a string).

Limitations
-----------
* Only one attachment per message is forwarded (Telegram media groups /
  albums are not reassembled).
* Outbound files are sent by URL (``File.file_id``); Telegram will fetch it
  server-side, so the URI must be publicly reachable.
* No webhook support: long polling only.
* No rate limiting, and no retry on send (a failed getUpdates is retried after 5 s).
"""

import asyncio
from collections.abc import AsyncGenerator, Mapping
from typing import TYPE_CHECKING, Any

from domyn.agents.audio import Transcriber
from domyn.agents.channels._parts import extract_file, extract_text
from domyn.agents.channels._voice import as_transcriber, transcribe_voice
from domyn.agents.core import BaseEvent, ExecutionEventType, File, FileType, Part
from domyn.agents.logger import get_logger

logger = get_logger()

if TYPE_CHECKING:
    import telegram


class TelegramChannel:
    """InputChannel that bridges Telegram chats to AgentServer.

    Parameters
    ----------
    token:
        Telegram Bot API token (from @BotFather).
    poll_timeout:
        Long-poll timeout in seconds for ``getUpdates``.  Default: 30.
    transcriber:
        Turns voice notes and audio into text for the agent: a ``Transcriber``
        or its configuration (``{"type": "local"}``). Without one, the agent is
        told a voice message arrived that could not be transcribed.
    """

    def __init__(
        self,
        token: str,
        poll_timeout: int = 30,
        transcriber: Transcriber | Mapping[str, Any] | None = None,
    ) -> None:
        self._token = token
        self._poll_timeout = poll_timeout
        self.transcriber = as_transcriber(transcriber)
        self._stop = asyncio.Event()
        # offset tracks the last processed update_id to avoid duplicates
        self._offset: int = 0

        # Lazy import: only required when this channel is actually used.
        try:
            import telegram  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                'python-telegram-bot is required for TelegramChannel. Run: uv add "domyn[telegram]"'
            ) from exc

    # ------------------------------------------------------------------
    # InputChannel contract
    # ------------------------------------------------------------------

    async def send_event(self, conversation_id: str, event: BaseEvent) -> None:
        """Send the agent response back to the Telegram chat (best-effort)."""
        await self._maybe_send(conversation_id, event)

    async def send_event_priority(self, conversation_id: str, event: BaseEvent) -> None:
        """Send the agent response back to the Telegram chat (guaranteed)."""
        await self._maybe_send(conversation_id, event)

    async def receive(self) -> AsyncGenerator[tuple[str, BaseEvent], None]:
        """Long-poll Telegram and yield one AGENT_START per incoming message."""
        import telegram

        bot = telegram.Bot(token=self._token)

        logger.info("TelegramChannel: starting long-poll (timeout=%ds)", self._poll_timeout)

        while not self._stop.is_set():
            try:
                updates = await bot.get_updates(
                    offset=self._offset,
                    timeout=self._poll_timeout,
                    allowed_updates=["message"],
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("TelegramChannel: getUpdates failed: %s", exc)
                await asyncio.sleep(5)
                continue

            for update in updates:
                self._offset = update.update_id + 1

                message = getattr(update, "message", None)
                if message is None:
                    continue

                parts = await _message_to_parts(bot, message)
                if not parts:
                    continue
                # The file URL carries the bot token: no header needed.
                parts = await transcribe_voice(parts, self.transcriber)

                chat_id = str(message.chat_id)
                conversation_id = chat_id  # one conversation per chat

                event = BaseEvent(
                    event_type=ExecutionEventType.AGENT_START,
                    author="user",
                    content=parts,
                    conversation_id=conversation_id,
                )
                logger.debug(
                    "TelegramChannel: AGENT_START chat_id=%s parts=%d", chat_id, len(parts)
                )
                yield conversation_id, event

    async def close(self) -> None:
        """Stop the long-poll loop."""
        self._stop.set()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _maybe_send(self, conversation_id: str, event: BaseEvent) -> None:
        """Send a text/media reply only for final (non-partial) RESPONSE events."""
        if event.event_type != ExecutionEventType.RESPONSE:
            return
        if getattr(event, "is_partial", False):
            return

        text = extract_text(event)
        file_part = extract_file(event)
        if not text and file_part is None:
            return

        import telegram

        bot = telegram.Bot(token=self._token)
        chat_id = int(conversation_id)
        try:
            if file_part is not None and file_part.file is not None:
                await _send_file(bot, chat_id, file_part.file, caption=text or None)
            else:
                await bot.send_message(chat_id=chat_id, text=text)
            logger.debug("TelegramChannel: sent reply to chat_id=%s", conversation_id)
        except Exception as exc:
            logger.warning(
                "TelegramChannel: failed to send message to chat_id=%s: %s",
                conversation_id,
                exc,
            )


async def _message_to_parts(bot: "telegram.Bot", message: "telegram.Message") -> list[Part]:
    """Convert a Telegram message into a list of Parts (text + at most one file).

    Handles photos (largest size), voice notes, audio and documents by
    resolving the ``file_id`` via ``getFile`` to obtain a downloadable URL.
    """
    parts: list[Part] = []
    caption = getattr(message, "caption", None) or getattr(message, "text", None)

    file_id: str | None = None
    file_type: FileType | None = None

    photos = getattr(message, "photo", None)
    voice = getattr(message, "voice", None)
    audio = getattr(message, "audio", None)
    document = getattr(message, "document", None)

    if photos:
        file_id = photos[-1].file_id  # largest resolution is last
        file_type = FileType.IMAGE
    elif voice is not None:
        file_id = voice.file_id
        file_type = FileType.AUDIO
    elif audio is not None:
        file_id = audio.file_id
        file_type = FileType.AUDIO
    elif document is not None:
        file_id = document.file_id
        file_type = FileType.DOCUMENT

    if file_id is not None and file_type is not None:
        try:
            tg_file = await bot.get_file(file_id)
            # Telegram leaves file_path empty for a file it will not serve
            # (over the 20 MB bot download limit).
            if tg_file.file_path:
                parts.append(Part(file=File(file_id=tg_file.file_path, type=file_type)))
        except Exception as exc:
            logger.warning("TelegramChannel: failed to resolve file_id=%s: %s", file_id, exc)

    if caption:
        parts.append(Part(text=caption))

    return parts


async def _send_file(
    bot: "telegram.Bot",
    chat_id: int,
    file: File,
    caption: str | None,
) -> None:
    """Send a File Part to Telegram using the API method matching its type."""
    if file.type == FileType.IMAGE:
        await bot.send_photo(chat_id=chat_id, photo=file.file_id, caption=caption)
    elif file.type == FileType.AUDIO:
        await bot.send_audio(chat_id=chat_id, audio=file.file_id, caption=caption)
    else:
        await bot.send_document(chat_id=chat_id, document=file.file_id, caption=caption)
