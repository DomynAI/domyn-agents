"""SlackChannel — InputChannel implementation via Slack Bolt (Socket Mode).

Setup
-----
1. Create a Slack App at https://api.slack.com/apps:
   - Enable Socket Mode and generate an ``SLACK_APP_TOKEN`` (starts with ``xapp-``).
   - Add a Bot Token OAuth scope: ``chat:write``.
   - Subscribe to the ``message.im`` (or ``message.channels``) bot event.
   - Install the app to your workspace and copy the ``SLACK_BOT_TOKEN`` (``xoxb-``).
2. Install the dependency::

       uv add "domyn[slack]"

3. Use as the channel in AgentServer::

       from domyn.agents.channels.slack import SlackChannel
       from domyn.agents.server import AgentServer

       channel = SlackChannel(
           bot_token=os.environ["SLACK_BOT_TOKEN"],
           app_token=os.environ["SLACK_APP_TOKEN"],
       )
       server = AgentServer(channel=channel, adapter=adapter)
       await server.serve()

How it works
------------
* ``receive()`` starts a Slack Bolt app in Socket Mode (no public endpoint needed).
  Each incoming DM or channel mention is converted to an ``AGENT_START`` event:
  the message text becomes a text ``Part``; the first attachment in
  ``event["files"]`` (image/audio/video/other) becomes a
  ``Part(file=File(file_id=url_private, type=...))``; an audio clip is also
  transcribed into a text ``Part`` when a ``transcriber`` is given (see
  ``channels/_voice.py``).
* ``send_event()`` / ``send_event_priority()`` call ``chat.postMessage`` for final
  (non-partial) ``RESPONSE`` events with only text; when the content contains a
  ``Part.file`` the file is downloaded (via ``httpx``) and re-uploaded with
  ``files.upload_v2``.
* ``conversation_id`` is the Slack ``channel`` ID (DM or channel) combined with
  the ``thread_ts`` when available, so parallel threads stay isolated.

Limitations
-----------
* Only the first attachment per message is forwarded.
* Downloading ``url_private`` requires the bot token as a Bearer header —
  handled internally, but any code reusing ``File.file_id`` directly must do the
  same.
* Block Kit and reactions are not implemented.
* The bot responds to every message in a DM; add a mention filter for public channels.
* No rate-limit handling beyond what Slack Bolt provides by default.
"""

import asyncio
from collections.abc import AsyncGenerator, Mapping
from typing import TYPE_CHECKING, Any

import httpx

from domyn.agents.audio import Transcriber
from domyn.agents.channels._parts import extract_file, extract_text
from domyn.agents.channels._voice import as_transcriber, transcribe_voice
from domyn.agents.core import BaseEvent, ExecutionEventType, File, FileType, Part
from domyn.agents.logger import get_logger

logger = get_logger()

if TYPE_CHECKING:
    from slack_sdk.web.async_client import AsyncWebClient

_SLACK_MIMETYPE_PREFIX: dict[str, FileType] = {
    "image": FileType.IMAGE,
    "audio": FileType.AUDIO,
    "video": FileType.VIDEO,
}


class SlackChannel:
    """InputChannel that bridges Slack (Socket Mode) to AgentServer.

    Parameters
    ----------
    bot_token:
        Slack Bot User OAuth Token (``xoxb-…``).
    app_token:
        Slack App-Level Token for Socket Mode (``xapp-…``).
    transcriber:
        Turns audio clips into text for the agent: a ``Transcriber`` or its
        configuration (``{"type": "local"}``). Without one, the agent is told a
        voice message arrived that could not be transcribed.
    """

    def __init__(
        self,
        bot_token: str,
        app_token: str,
        transcriber: Transcriber | Mapping[str, Any] | None = None,
    ) -> None:
        self._bot_token = bot_token
        self._app_token = app_token
        self.transcriber = as_transcriber(transcriber)
        self._queue: asyncio.Queue[tuple[str, BaseEvent]] = asyncio.Queue()
        self._stop = asyncio.Event()

        try:
            import slack_bolt  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                'slack-bolt is required for SlackChannel. Run: uv add "domyn[slack]"'
            ) from exc

    # ------------------------------------------------------------------
    # InputChannel contract
    # ------------------------------------------------------------------

    async def send_event(self, conversation_id: str, event: BaseEvent) -> None:
        """Send agent response to Slack (best-effort)."""
        await self._maybe_send(conversation_id, event)

    async def send_event_priority(self, conversation_id: str, event: BaseEvent) -> None:
        """Send agent response to Slack (guaranteed)."""
        await self._maybe_send(conversation_id, event)

    async def receive(self) -> AsyncGenerator[tuple[str, BaseEvent], None]:
        """Start Slack Socket Mode handler and yield AGENT_START per message."""
        from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
        from slack_bolt.async_app import AsyncApp

        app = AsyncApp(token=self._bot_token)

        @app.event("message")  # type: ignore[untyped-decorator]
        async def handle_message(event: dict[str, Any], say: object) -> None:
            # Ignore bot messages (including our own replies) to avoid loops.
            if event.get("bot_id"):
                return
            text: str = event.get("text", "").strip()
            parts = _message_to_parts(event, text)
            if not parts:
                return
            # url_private is downloaded with the bot token.
            parts = await transcribe_voice(
                parts, self.transcriber, {"Authorization": f"Bearer {self._bot_token}"}
            )

            channel_id: str = event.get("channel", "")
            thread_ts: str = event.get("thread_ts", event.get("ts", ""))
            # Use channel+thread as conversation_id so threads stay isolated.
            conversation_id = f"{channel_id}:{thread_ts}"

            start_event = BaseEvent(
                event_type=ExecutionEventType.AGENT_START,
                author="user",
                content=parts,
                conversation_id=conversation_id,
            )
            await self._queue.put((conversation_id, start_event))
            logger.debug("SlackChannel: AGENT_START channel=%s parts=%d", channel_id, len(parts))

        handler = AsyncSocketModeHandler(app, self._app_token)
        await handler.connect_async()
        logger.info("SlackChannel: Socket Mode connected")

        try:
            while not self._stop.is_set():
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                    yield item
                except TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
        finally:
            await handler.close_async()

    async def close(self) -> None:
        """Disconnect from Slack Socket Mode."""
        self._stop.set()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _maybe_send(self, conversation_id: str, event: BaseEvent) -> None:
        """Post a text/file message to Slack for final (non-partial) RESPONSE events."""
        if event.event_type != ExecutionEventType.RESPONSE:
            return
        if getattr(event, "is_partial", False):
            return

        text = extract_text(event)
        file_part = extract_file(event)
        if not text and file_part is None:
            return

        # conversation_id is "channel_id:thread_ts"
        parts = conversation_id.split(":", 1)
        channel_id = parts[0]
        thread_ts = parts[1] if len(parts) > 1 else None

        from slack_sdk.web.async_client import AsyncWebClient

        client = AsyncWebClient(token=self._bot_token)
        try:
            if file_part is not None and file_part.file is not None:
                await self._send_file(client, channel_id, thread_ts, file_part.file, text)
            else:
                kwargs: dict[str, Any] = {"channel": channel_id, "text": text}
                if thread_ts:
                    kwargs["thread_ts"] = thread_ts
                await client.chat_postMessage(**kwargs)
            logger.debug("SlackChannel: sent reply to channel=%s", channel_id)
        except Exception as exc:
            logger.warning("SlackChannel: failed to send message to %s: %s", channel_id, exc)

    async def _send_file(
        self,
        client: "AsyncWebClient",
        channel_id: str,
        thread_ts: str | None,
        file: File,
        caption: str,
    ) -> None:
        """Download the File.file_id and re-upload it to Slack via files.upload_v2."""
        async with httpx.AsyncClient() as http_client:
            resp = await http_client.get(file.file_id)
            resp.raise_for_status()
            content = resp.content

        kwargs: dict[str, Any] = {
            "channel": channel_id,
            "file": content,
            "filename": file.file_id.rsplit("/", 1)[-1] or "file",
        }
        if caption:
            kwargs["initial_comment"] = caption
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        await client.files_upload_v2(**kwargs)


def _message_to_parts(event: dict[str, Any], text: str) -> list[Part]:
    """Convert a Slack message event into text + file Parts.

    The first entry in ``event["files"]`` (if any) becomes a
    ``Part(file=File(file_id=url_private, type=...))``; ``url_private`` requires
    the bot token as a Bearer header to download.
    """
    parts: list[Part] = []
    files = event.get("files") or []
    if files:
        first = files[0]
        mimetype: str = first.get("mimetype", "")
        file_type = FileType.DOCUMENT
        for prefix, ftype in _SLACK_MIMETYPE_PREFIX.items():
            if mimetype.startswith(prefix):
                file_type = ftype
                break
        url = first.get("url_private")
        if url:
            parts.append(Part(file=File(file_id=url, type=file_type)))
    if text:
        parts.append(Part(text=text))
    return parts
