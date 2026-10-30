"""A guardrail served over HTTP: the two endpoints the hooks of ``config.yaml`` call.

Each is sent ``{"current_event": <the event>}`` and answers ``{"modified_event": ...}``, the
event the runner goes on with, in one of three ways:

- the event unchanged: it passes;
- the event with its text changed: it goes on, redacted;
- a ``response`` event in its place: the turn ends there, with that text as the answer.

``emitted_content`` says what the check found; it is kept in the ``hook_end`` event. The
policy is deliberately trivial (no email addresses either way, and one phrase refused) so
the shape of the exchange stands out. See README.md to run it.
"""

import re
import uuid
from datetime import UTC, datetime
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

EMAIL = re.compile(r"[\w.%+-]+@[\w.-]+\.[a-zA-Z]{2,}")
BLOCKED_PHRASE = "ignore all previous instructions"

app = FastAPI()


class HookRequest(BaseModel):
    """The `input_fields` the hooks declare in config.yaml."""

    current_event: dict[str, Any]


def _text(event: dict[str, Any]) -> str:
    return " ".join(part.get("text") or "" for part in event.get("content") or [])


def _result(
    received: dict[str, Any], event: dict[str, Any], passed: bool, reason: str
) -> dict[str, Any]:
    print(f"[guardrail] {received['event_type']}: {reason}")
    return {
        "modified_event": event,
        "emitted_content": {"name": "Shop guardrail", "passed": passed, "reason": reason},
    }


def _redact(event: dict[str, Any]) -> dict[str, Any]:
    text = _text(event)
    if not EMAIL.search(text):
        return _result(event, event, True, "nothing to change")
    redacted = {**event, "content": [{"text": EMAIL.sub("[email removed]", text)}]}
    return _result(event, redacted, False, "email address removed")


@app.post("/input-guardrail")
async def input_guardrail(body: HookRequest) -> dict[str, Any]:
    """Refuse the blocked phrase; remove email addresses before the model reads them."""
    event = body.current_event
    if BLOCKED_PHRASE in _text(event).lower():
        answer = {
            "event_type": "response",
            "author": "guardrail",
            "content": [{"text": "I'm sorry, I cannot help with that request."}],
            "event_id": f"event_{uuid.uuid4()}",
            "timestamp": datetime.now(UTC).isoformat(),
        }
        return _result(event, answer, False, "request refused")
    return _redact(event)


@app.post("/output-guardrail")
async def output_guardrail(body: HookRequest) -> dict[str, Any]:
    """Remove email addresses from the reply before the user reads it."""
    return _redact(body.current_event)


if __name__ == "__main__":
    print("Guardrail on http://localhost:8766. Ctrl+C stops.")
    uvicorn.run(app, host="localhost", port=8766, log_level="warning")
