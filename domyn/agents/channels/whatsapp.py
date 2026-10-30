"""WhatsAppChannel — InputChannel implementation via WhatsApp Cloud API (Meta).

Setup
-----
1. Create a Meta App, enable WhatsApp Business and get:
   - ``WHATSAPP_TOKEN``      — permanent system user token
   - ``WHATSAPP_PHONE_ID``   — phone number ID (sender)
   - ``WHATSAPP_VERIFY_TOKEN`` — arbitrary string you set for webhook verification
2. No extra dependencies needed — uses ``httpx`` (already a core dependency) for
   outbound HTTP and the stdlib ``asyncio`` HTTP server for the inbound webhook.

3. Expose a public webhook endpoint (e.g. with ngrok during development) and
   register it in the Meta App dashboard pointing to ``/webhook``.

4. Use as the channel in AgentServer::

       from domyn.agents.channels.whatsapp import WhatsAppChannel
       from domyn.agents.server import AgentServer

       channel = WhatsAppChannel(
           token=os.environ["WHATSAPP_TOKEN"],
           phone_number_id=os.environ["WHATSAPP_PHONE_ID"],
           verify_token=os.environ["WHATSAPP_VERIFY_TOKEN"],
           webhook_host="0.0.0.0",
           webhook_port=8080,
       )
       server = AgentServer(channel=channel, adapter=adapter)
       await server.serve()

How it works
------------
* ``receive()`` starts a minimal ``asyncio`` HTTP server that handles the Meta
  webhook.  Each incoming message is converted to an ``AGENT_START`` event and
  yielded: text messages become a text ``Part``; image/audio/video/document
  messages have their ``media_id`` resolved (one extra Graph API GET) to a
  temporary downloadable URL and attached as ``Part(file=File(file_id=..., type=...))``;
  a voice note is also transcribed into a text ``Part`` when a ``transcriber``
  is given (see ``channels/_voice.py``)
  (caption, if any, is kept as text).  The GET used by Meta for webhook
  verification is handled automatically.
* ``send_event()`` / ``send_event_priority()`` use ``httpx.AsyncClient`` to POST
  to the WhatsApp Cloud API ``/messages`` endpoint for final (non-partial)
  ``RESPONSE`` events — as a text message, or as an image/audio/document message
  (by ``link``) when the content contains a ``Part.file``.
* ``conversation_id`` is the sender's WhatsApp phone number (wa_id).

Limitations
-----------
* Outbound files are sent by ``link`` (``File.file_id`` must be a publicly
  reachable URL); uploading raw bytes via the ``/media`` endpoint is not
  implemented.
* Downloading the resolved media URL requires the same Bearer token used for
  the Graph API — the URL alone is not enough (Meta signs it to the app).
* No message deduplication (Meta may deliver the same message more than once).
* No signature verification of incoming webhooks (add ``X-Hub-Signature-256``
  check for production).
"""

import asyncio
import json
import urllib.parse
from asyncio import StreamReader, StreamWriter
from collections.abc import AsyncGenerator, Mapping
from typing import Any

import httpx

from domyn.agents.audio import Transcriber
from domyn.agents.channels._parts import extract_file, extract_text
from domyn.agents.channels._voice import as_transcriber, transcribe_voice
from domyn.agents.core import BaseEvent, ExecutionEventType, File, FileType, Part
from domyn.agents.logger import get_logger

logger = get_logger()

_GRAPH_API = "https://graph.facebook.com/v19.0"

_WHATSAPP_MEDIA_TYPES: dict[str, FileType] = {
    "image": FileType.IMAGE,
    "audio": FileType.AUDIO,
    "video": FileType.VIDEO,
    "document": FileType.DOCUMENT,
}


class WhatsAppChannel:
    """InputChannel that bridges WhatsApp Cloud API to AgentServer.

    Parameters
    ----------
    token:
        Meta permanent system user access token.
    phone_number_id:
        The WhatsApp sender phone number ID (from Meta App dashboard).
    verify_token:
        The verification token you registered in the Meta webhook settings.
    webhook_host:
        Host to bind the local webhook server.  Default: ``"0.0.0.0"``.
    webhook_port:
        Port to bind the local webhook server.  Default: ``8080``.
    transcriber:
        Turns voice notes into text for the agent: a ``Transcriber`` or its
        configuration (``{"type": "local"}``). Without one, the agent is told a
        voice message arrived that could not be transcribed.
    """

    def __init__(
        self,
        token: str,
        phone_number_id: str,
        verify_token: str,
        webhook_host: str = "0.0.0.0",
        webhook_port: int = 8080,
        transcriber: Transcriber | Mapping[str, Any] | None = None,
    ) -> None:
        self._token = token
        self._phone_number_id = phone_number_id
        self._verify_token = verify_token
        self._webhook_host = webhook_host
        self._webhook_port = webhook_port
        self.transcriber = as_transcriber(transcriber)
        self._queue: asyncio.Queue[tuple[str, BaseEvent]] = asyncio.Queue()
        self._stop = asyncio.Event()
        self._server: asyncio.Server | None = None

    # ------------------------------------------------------------------
    # InputChannel contract
    # ------------------------------------------------------------------

    async def send_event(self, conversation_id: str, event: BaseEvent) -> None:
        """Send agent response to WhatsApp (best-effort)."""
        await self._maybe_send(conversation_id, event)

    async def send_event_priority(self, conversation_id: str, event: BaseEvent) -> None:
        """Send agent response to WhatsApp (guaranteed)."""
        await self._maybe_send(conversation_id, event)

    async def receive(self) -> AsyncGenerator[tuple[str, BaseEvent], None]:
        """Start webhook server and yield AGENT_START for each incoming message."""
        self._server = await asyncio.start_server(
            self._handle_connection,
            self._webhook_host,
            self._webhook_port,
        )
        logger.info(
            "WhatsAppChannel: webhook listening on %s:%d",
            self._webhook_host,
            self._webhook_port,
        )

        async with self._server:
            await self._server.start_serving()
            while not self._stop.is_set():
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                    yield item
                except TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break

    async def close(self) -> None:
        """Stop the webhook server."""
        self._stop.set()
        if self._server is not None:
            self._server.close()

    # ------------------------------------------------------------------
    # Webhook HTTP server (stdlib asyncio, no extra deps)
    # ------------------------------------------------------------------

    async def _handle_connection(self, reader: StreamReader, writer: StreamWriter) -> None:
        """Handle one HTTP connection from Meta's webhook."""
        try:
            raw = await reader.read(65536)
            if not raw:
                return
            request_line, rest = raw.split(b"\r\n", 1)
            method, path, _ = request_line.decode().split(" ", 2)

            # Split headers from body
            _header_block, _, body = rest.partition(b"\r\n\r\n")

            if method == "GET":
                response = self._handle_verification(path)
            elif method == "POST":
                response = await self._handle_webhook(body)
            else:
                response = b"HTTP/1.1 405 Method Not Allowed\r\n\r\n"

            writer.write(response)
            await writer.drain()
        except Exception as exc:
            logger.debug("WhatsAppChannel: connection handler error: %s", exc)
        finally:
            writer.close()

    def _handle_verification(self, path: str) -> bytes:
        """Respond to Meta's hub.challenge verification request."""
        parsed = urllib.parse.urlparse(path)
        params = urllib.parse.parse_qs(parsed.query)
        mode = params.get("hub.mode", [""])[0]
        token = params.get("hub.verify_token", [""])[0]
        challenge = params.get("hub.challenge", [""])[0]

        if mode == "subscribe" and token == self._verify_token and challenge:
            logger.info("WhatsAppChannel: webhook verified")
            body = challenge.encode()
            return (
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/plain\r\n"
                b"Content-Length: " + str(len(body)).encode() + b"\r\n"
                b"\r\n" + body
            )
        return b"HTTP/1.1 403 Forbidden\r\n\r\n"

    async def _handle_webhook(self, body: bytes) -> bytes:
        """Parse Meta webhook payload and enqueue AGENT_START events."""
        try:
            data = json.loads(body)
        except Exception:
            return b"HTTP/1.1 400 Bad Request\r\n\r\n"

        try:
            for entry in data.get("entry", []):
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    for message in value.get("messages", []):
                        wa_id = message.get("from", "")
                        if not wa_id:
                            continue
                        parts = await self._message_to_parts(message)
                        if not parts:
                            continue
                        event = BaseEvent(
                            event_type=ExecutionEventType.AGENT_START,
                            author="user",
                            content=parts,
                            conversation_id=wa_id,
                        )
                        await self._queue.put((wa_id, event))
                        logger.debug(
                            "WhatsAppChannel: AGENT_START wa_id=%s parts=%d",
                            wa_id,
                            len(parts),
                        )
        except Exception as exc:
            logger.warning("WhatsAppChannel: error parsing webhook payload: %s", exc)

        return b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK"

    async def _message_to_parts(self, message: dict[str, Any]) -> list[Part]:
        """Convert a WhatsApp message payload into text + file Parts.

        Text messages yield a single text ``Part``.  Media messages (image,
        audio, video, document) have their ``media_id`` resolved to a
        temporary downloadable URL via one extra Graph API GET.
        """
        msg_type = message.get("type") or ""
        parts: list[Part] = []

        if msg_type == "text":
            text = message.get("text", {}).get("body", "")
            if text:
                parts.append(Part(text=text))
            return parts

        file_type = _WHATSAPP_MEDIA_TYPES.get(msg_type)
        if file_type is None:
            return parts

        media = message.get(msg_type, {})
        media_id = media.get("id")
        caption = media.get("caption")
        if media_id:
            uri = await self._resolve_media_url(media_id)
            if uri:
                parts.append(Part(file=File(file_id=uri, type=file_type)))
        if caption:
            parts.append(Part(text=caption))
        # Media URLs are downloaded with the same token as the Graph API.
        return await transcribe_voice(
            parts, self.transcriber, {"Authorization": f"Bearer {self._token}"}
        )

    async def _resolve_media_url(self, media_id: str) -> str | None:
        """Resolve a WhatsApp media_id to a temporary downloadable URL."""
        url = f"{_GRAPH_API}/{media_id}"
        headers = {"Authorization": f"Bearer {self._token}"}
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                result = resp.json().get("url")
                return str(result) if result is not None else None
        except Exception as exc:
            logger.warning("WhatsAppChannel: failed to resolve media_id=%s: %s", media_id, exc)
            return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _maybe_send(self, conversation_id: str, event: BaseEvent) -> None:
        """POST a text/media message to WhatsApp Cloud API for final RESPONSE events."""
        if event.event_type != ExecutionEventType.RESPONSE:
            return
        if getattr(event, "is_partial", False):
            return

        text = extract_text(event)
        file_part = extract_file(event)
        if not text and file_part is None:
            return

        url = f"{_GRAPH_API}/{self._phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        payload = _build_send_payload(conversation_id, text, file_part)
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, headers=headers, content=json.dumps(payload))
                if resp.status_code not in (200, 201):
                    logger.warning(
                        "WhatsAppChannel: send failed status=%d body=%s",
                        resp.status_code,
                        resp.text,
                    )
                else:
                    logger.debug("WhatsAppChannel: sent reply to wa_id=%s", conversation_id)
        except Exception as exc:
            logger.warning(
                "WhatsAppChannel: failed to send message to %s: %s", conversation_id, exc
            )


_FILE_TYPE_TO_WHATSAPP = {
    FileType.IMAGE: "image",
    FileType.AUDIO: "audio",
    FileType.VIDEO: "video",
    FileType.DOCUMENT: "document",
}


def _build_send_payload(conversation_id: str, text: str, file_part: Part | None) -> dict[str, Any]:
    """Build the WhatsApp Cloud API /messages payload for text or media."""
    if file_part is not None and file_part.file is not None:
        media_kind = _FILE_TYPE_TO_WHATSAPP.get(
            file_part.file.type or FileType.DOCUMENT, "document"
        )
        media_payload: dict[str, Any] = {"link": file_part.file.file_id}
        if text and media_kind != "audio":
            media_payload["caption"] = text
        return {
            "messaging_product": "whatsapp",
            "to": conversation_id,
            "type": media_kind,
            media_kind: media_payload,
        }
    return {
        "messaging_product": "whatsapp",
        "to": conversation_id,
        "type": "text",
        "text": {"body": text},
    }
