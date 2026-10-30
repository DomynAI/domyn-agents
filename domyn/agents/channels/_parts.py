"""Shared helpers for extracting content from BaseEvent.content Parts.

Used by the chat-style InputChannel implementations (Telegram, WhatsApp,
Slack) to pull the text / file payload out of an outbound RESPONSE event
before translating it into the target platform's native send call.
"""

from domyn.agents.core import BaseEvent, Part


def extract_text(event: BaseEvent) -> str:
    """Return the first non-empty text Part in the event content, or ``""``."""
    for part in event.content:
        if part.text:
            return part.text
    return ""


def extract_file(event: BaseEvent) -> Part | None:
    """Return the first Part with a non-null ``file``, if any."""
    for part in event.content:
        if part.file is not None:
            return part
    return None
