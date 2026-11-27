"""The analyst's console, in the browser: follows the cases and answers what the agent may not do alone.

It reads three topics from their start: the alerts (``fraud.alerts``), what the agent does
(``fraud.events``, written by ``domyn serve``) and the analyst's answers
(``fraud.decisions``), and keeps each case's story in memory. Nothing else is stored: a
restarted console reads the topics again and shows the same cases.

When an action waits for an analyst, the page shows it with the agent's reason. The
analyst approves, denies or, for a payment release, changes the amount; the answer goes
to ``fraud.decisions`` as a ``TOOL_FEEDBACK`` event, under the case's key, and the agent
resumes the case with it.

It also raises new alerts, as the fraud engine would: the bank's services write the
scenario's data, and the console publishes the alert to ``fraud.alerts``.

The console never calls the model: it needs Kafka and the services, not an API key. Open
http://localhost:8080 once it runs. See README.md.
"""

import asyncio
import contextlib
import json
import logging
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from domyn.agents import ApprovalType, BaseEvent, ExecutionEventType

KAFKA = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
ALERTS_TOPIC, EVENTS_TOPIC, DECISIONS_TOPIC = "fraud.alerts", "fraud.events", "fraud.decisions"
HOST = os.getenv("LISTEN_HOST", "127.0.0.1")  # 0.0.0.0 in Docker, to be reachable from the browser
PORT = 8080
SERVICES_URL = os.getenv("SERVICES_URL", "http://localhost:8001")
PAGE = Path(__file__).with_name("console.html")
ACTIONS = ("block_card", "close_alert", "release_payment")


class Cases:
    """Each case's story, built from the messages of the three topics."""

    def __init__(self) -> None:
        self.cases: dict[str, dict[str, Any]] = {}
        self.answered: set[str] = set()  # the calls an analyst has answered
        self.reasons: dict[str, str] = {}  # the agent's reason for each call
        self.seen: set[str] = set()

    def read(self, topic: str, key: bytes, value: bytes, at_ms: float | None = None) -> list[str]:
        """Apply one message; return what it adds, as log lines.

        ``at_ms`` is when the message was sent, as Kafka stamps it: an instant, unlike an
        event's own ``timestamp``, which is the local time of whatever machine made it.
        """
        sent = at_ms / 1000 if at_ms is not None else time.time()
        at = datetime.fromtimestamp(sent, UTC).isoformat()
        case_id = key.decode()
        case = self.cases.setdefault(
            case_id, {"id": case_id, "alert": None, "steps": [], "waiting": None}
        )
        if topic == ALERTS_TOPIC:
            new = case["alert"] is None
            alert = case["alert"] = json.loads(value)
            if resent := alert.pop("resent", None):
                if resent in self.seen:
                    return []  # a resend this console sent, already applied
                self.seen.add(resent)
                text = "Alert sent again"
                case["steps"].append({"at": at, "kind": "resent", "text": text})
                case["steps"].sort(key=lambda s: s["at"])
                return [f"{case_id}  {text}"]
            words = " · ".join(
                str(part)
                for part in (
                    alert.get("type"),
                    f"risk {alert.get('risk_score')}",
                    alert.get("summary"),
                )
                if part
            )
            return [f"{case_id}  new alert: {words}"] if new else []

        event = BaseEvent.model_validate_json(value)
        if event.event_id in self.seen:
            return []  # an answer this console sent, already applied
        self.seen.add(event.event_id)
        action = event.action
        lines: list[str] = []

        def step(kind: str, text: str, **more: Any) -> None:
            case["steps"].append({"at": at, "kind": kind, "text": text})
            case["steps"][-1].update(more)
            if line := _log_line(kind, text, more):
                lines.append(f"{case_id}  {line}")

        if (
            event.event_type
            in (
                ExecutionEventType.TOOL_START,
                ExecutionEventType.RESPONSE,
            )
            and case["waiting"]
        ):
            # The agent moved on (a new alert for the case closes the call that was
            # waiting): the request can no longer be answered.
            case["waiting"] = None

        match event.event_type:
            case ExecutionEventType.TOOL_START:
                arguments = {k: v for k, v in action.parameters.items() if k != "reasoning"}
                reason = action.parameters.get("reasoning") or action.thought
                if reason:
                    self.reasons[action.call_id] = reason
                step("call", action.name, arguments=arguments, action=action.name in ACTIONS)
            case ExecutionEventType.TOOL_END if action.name in ACTIONS:
                result = action.observation
                if isinstance(result, dict):  # what the service answered
                    step("result", "", name=action.name, arguments=result)
                else:  # refused, by the analyst or the policy
                    step("result", str(result), name=action.name)
            case ExecutionEventType.TOOL_NEEDS_FEEDBACK:
                if action.call_id not in self.answered:  # its answer may be read first
                    case["waiting"] = {
                        "call_id": action.call_id,
                        "name": action.name,
                        "editable": action.approval_type == ApprovalType.PARAMS_UPDATE,
                        "parameters": [
                            {"name": p.name, "value": p.value} for p in action.parameters
                        ],
                        "reason": self.reasons.get(action.call_id),
                        "request": event.model_dump(mode="json"),
                    }
                step("asked", f"{action.name} waits for an analyst")
            case ExecutionEventType.TOOL_FEEDBACK:
                self.answered.add(action.call_id)
                if case["waiting"] and case["waiting"]["call_id"] == action.call_id:
                    case["waiting"] = None
                values = {p.name: p.value for p in action.parameters}
                verdict = "approved" if action.approve else "denied"
                step("decision", f"Analyst {verdict} {action.name}", arguments=values)
            case ExecutionEventType.RESPONSE:
                step("note", "".join(part.text or "" for part in event.content or []))
            case ExecutionEventType.ERROR_OCCURRED:
                step("error", event.error_message or "error")
            case ExecutionEventType.AGENT_END if event.error_message:
                step("error", event.error_message)
            case ExecutionEventType.AGENT_END:
                step("end", "")  # not shown: tells an answer the agent had nothing to resume
        # The topics are read side by side, so their messages may come out of order.
        case["steps"].sort(key=lambda s: s["at"])
        return lines

    def verdict(self, verdict: dict[str, Any]) -> list[str]:
        """Apply what a bank rule decided, as the services recorded it; return its log line."""
        case_id = verdict["alert_id"]
        case = self.cases.setdefault(
            case_id, {"id": case_id, "alert": None, "steps": [], "waiting": None}
        )
        more = {k: verdict[k] for k in ("action", "decision", "rule")}
        case["steps"].append(
            {"at": verdict["at"], "kind": "policy", "text": verdict["reason"], **more}
        )
        case["steps"].sort(key=lambda s: s["at"])
        return [f"{case_id}  {_log_line('policy', verdict['reason'], more)}"]

    def view(self) -> list[dict[str, Any]]:
        def placed(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """A rule's verdict right after the call it ruled on. The two come from two
            places (the verdict from the services, the call from Kafka) and their
            clocks can put the verdict a moment first."""
            out: list[dict[str, Any]] = []
            moved: set[int] = set()
            for i, s in enumerate(steps):
                if i in moved:
                    continue
                if s["kind"] == "policy":
                    at = datetime.fromisoformat(s["at"])
                    j = next(
                        (
                            k
                            for k in range(i + 1, len(steps))
                            if steps[k]["kind"] == "call"
                            and steps[k]["text"] == s["action"]
                            and (datetime.fromisoformat(steps[k]["at"]) - at).total_seconds() < 3
                        ),
                        None,
                    )
                    if j is not None:
                        out.append(steps[j])
                        moved.add(j)
                out.append(s)
            return out

        def shown(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for s in placed(steps):
                if s["kind"] != "end":
                    out.append(s)
                elif out and out[-1]["kind"] == "decision":
                    # A turn that ended on the answer alone: nothing waited for it.
                    text = "The request was no longer waiting: nothing was resumed."
                    out.append({**s, "kind": "stale", "text": text})
            return out

        def status(case: dict[str, Any]) -> str:
            if case["waiting"]:
                return "waiting"
            last = case["steps"][-1]["kind"] if case["steps"] else None
            return {None: "queued", "note": "done", "stale": "done", "error": "error"}.get(
                last, "working"
            )

        cases = []
        for case in self.cases.values():
            view = {**case, "steps": shown(case["steps"])}
            view["status"] = status(view)
            if view["waiting"]:
                waiting = {k: v for k, v in view["waiting"].items() if k != "request"}
                # Why it waits: the bank rule that asked.
                waiting["policy"] = next(
                    (
                        {"rule": s["rule"], "reason": s["text"]}
                        for s in reversed(case["steps"])
                        if s["kind"] == "policy"
                        and s["action"] == waiting["name"]
                        and s["decision"] == "ask"
                    ),
                    None,
                )
                view["waiting"] = waiting
            cases.append(view)
        return sorted(cases, key=lambda c: c["id"])


VERDICT_WORDS = {"allow": "allows", "deny": "refuses", "ask": "asks the analyst for"}


def _log_line(kind: str, text: str, more: dict[str, Any]) -> str | None:
    """One step as one line of the console's log."""

    def short(value: Any, limit: int = 160) -> str:
        flat = " ".join(str(value).split())
        return flat if len(flat) <= limit else flat[: limit - 1] + "…"

    arguments = ", ".join(f"{k}={v}" for k, v in (more.get("arguments") or {}).items())
    return {
        "call": f"-> {text}({short(arguments)})",
        "policy": f"bank rule {VERDICT_WORDS.get(more.get('decision', ''), '')} "
        f"{more.get('action')}: {more.get('rule')} ({short(text)})",
        "result": f"<- {more.get('name')}: {short(text or arguments)}",
        "asked": f"!! {text}",
        "decision": f"{text} ({short(arguments)})" if arguments else text,
        "note": f"case note: {short(text, 300)}",
        "error": f"error: {short(text)}",
    }.get(kind)


def log(lines: list[str]) -> None:
    for line in lines:
        print(line, flush=True)


CASES = Cases()

# Kafka's client reports the broker settling in at startup as errors; its log is not ours.
logging.getLogger("aiokafka").setLevel(logging.CRITICAL)


async def create_topics() -> None:
    """Created here so the console can read them before anything is published."""
    from aiokafka.admin import AIOKafkaAdminClient, NewTopic

    admin = AIOKafkaAdminClient(bootstrap_servers=KAFKA)
    await admin.start()
    try:
        topics = [ALERTS_TOPIC, EVENTS_TOPIC, DECISIONS_TOPIC]
        with contextlib.suppress(Exception):  # those that exist already
            await admin.create_topics(
                [NewTopic(t, num_partitions=1, replication_factor=1) for t in topics]
            )
    finally:
        await admin.close()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

    await create_topics()
    # No consumer group: every start reads the three topics from their beginning.
    consumer = AIOKafkaConsumer(
        ALERTS_TOPIC,
        EVENTS_TOPIC,
        DECISIONS_TOPIC,
        bootstrap_servers=KAFKA,
        auto_offset_reset="earliest",
    )
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA)
    await consumer.start()
    await producer.start()

    # Every start reads the topics from their beginning: only what arrives from now on
    # is logged, not the cases already told.
    started_ms = (time.time() - 5) * 1000

    async def follow() -> None:
        async for message in consumer:
            lines = CASES.read(message.topic, message.key, message.value, message.timestamp)
            if message.timestamp >= started_ms:
                log(lines)

    async def follow_verdicts() -> None:
        """The bank rules' verdicts, from the services, as they are recorded."""
        count = 0
        async with httpx.AsyncClient(base_url=SERVICES_URL, timeout=5) as client:
            while True:
                with contextlib.suppress(httpx.HTTPError, ValueError):
                    new = (await client.get("/engine/verdicts", params={"after": count})).json()
                    count += len(new)
                    for verdict in new:
                        lines = CASES.verdict(verdict)
                        if datetime.fromisoformat(verdict["at"]).timestamp() * 1000 >= started_ms:
                            log(lines)
                await asyncio.sleep(1)

    task = asyncio.create_task(follow())
    verdicts_task = asyncio.create_task(follow_verdicts())
    app.state.producer = producer
    print(f"Analyst console on http://localhost:{PORT}", flush=True)
    try:
        yield
    finally:
        task.cancel()
        verdicts_task.cancel()
        await consumer.stop()
        await producer.stop()


app = FastAPI(title="Analyst console", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def page() -> FileResponse:
    return FileResponse(PAGE)


@app.get("/api/cases")
def cases() -> list[dict[str, Any]]:
    return CASES.view()


class Decision(BaseModel):
    approve: bool
    parameters: dict[str, Any] = {}  # changed values, for an action the analyst may edit


@app.post("/api/cases/{case_id}/decision")
async def decide(case_id: str, decision: Decision) -> dict[str, str]:
    case = CASES.cases.get(case_id)
    if not case or not case["waiting"]:
        raise HTTPException(409, "Nothing waits for an answer in this case.")
    waiting = case["waiting"]
    action = BaseEvent.model_validate(waiting["request"]).action

    update: dict[str, Any] = {"approve": decision.approve}
    if decision.approve and waiting["editable"] and decision.parameters:
        try:
            update["parameters"] = [
                p.model_copy(
                    update={"value": type(p.value)(decision.parameters.get(p.name, p.value))}
                )
                for p in action.parameters
            ]
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, f"Not a valid value: {exc}") from exc
    answer = BaseEvent(
        event_type=ExecutionEventType.TOOL_FEEDBACK,
        author="analyst",
        action=action.model_copy(update=update),
    )
    value = answer.model_dump_json().encode()
    await app.state.producer.send_and_wait(DECISIONS_TOPIC, value, key=case_id.encode())
    log(CASES.read(DECISIONS_TOPIC, case_id.encode(), value))
    return {"sent": answer.event_id}


@app.get("/api/policy")
async def get_policy() -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        return (await client.get(f"{SERVICES_URL}/engine/policy")).json()


class Policy(BaseModel):
    high_risk: int


@app.put("/api/policy")
async def set_policy(policy: Policy) -> dict[str, Any]:
    """Change the bank's threshold: the next verdicts follow it, for every alert."""
    async with httpx.AsyncClient() as client:
        response = await client.put(f"{SERVICES_URL}/engine/policy", json=policy.model_dump())
    if response.status_code != 200:
        raise HTTPException(response.status_code, response.json().get("detail"))
    log([f"bank rules  closing an alert now asks the analyst from risk {policy.high_risk}"])
    return response.json()


@app.get("/api/scenarios")
async def scenarios() -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        return (await client.get(f"{SERVICES_URL}/engine/scenarios")).json()


class NewAlert(BaseModel):
    scenario: str
    customer_id: str
    risk_score: int
    amount: float | None = None


@app.post("/api/alerts")
async def raise_alert(new: NewAlert) -> dict[str, Any]:
    """Have the services write the scenario and the alert, then publish it."""
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{SERVICES_URL}/engine/alerts", json=new.model_dump())
    if response.status_code != 200:
        raise HTTPException(response.status_code, response.json().get("detail"))
    alert = response.json()
    del alert["status"]  # the services' state, not part of the alert
    key, value = alert["alert_id"].encode(), json.dumps(alert).encode()
    await app.state.producer.send_and_wait(ALERTS_TOPIC, value, key=key)
    log(CASES.read(ALERTS_TOPIC, key, value))
    return alert


@app.post("/api/cases/{case_id}/resend")
async def resend(case_id: str) -> dict[str, str]:
    """Send a case's alert again, after an error: the agent takes the case up where it is."""
    view = next((c for c in CASES.view() if c["id"] == case_id), None)
    if view is None or view["alert"] is None:
        raise HTTPException(404, "No alert for this case.")
    if view["status"] != "error":
        raise HTTPException(409, "Only a case that ended in an error is sent again.")
    resend_id = f"resend-{uuid.uuid4()}"
    key, value = case_id.encode(), json.dumps({**view["alert"], "resent": resend_id}).encode()
    await app.state.producer.send_and_wait(ALERTS_TOPIC, value, key=key)
    log(CASES.read(ALERTS_TOPIC, key, value))
    return {"sent": case_id}


@app.post("/api/samples")
async def send_samples() -> dict[str, int]:
    """Publish the five sample alerts of services.py, unless the topic holds alerts already."""
    import alerts

    published = await alerts.main(list(alerts.ALERTS), if_empty=True, verbose=False)
    if not published:
        raise HTTPException(409, "The alerts have been sent already.")
    return {"published": published}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
