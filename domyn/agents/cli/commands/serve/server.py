"""FastAPI server for local agent serving."""

from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from domyn.agents.agents import Agent
from domyn.agents.core import BaseEvent, ExecutionEventType, Part
from domyn.agents.runner import Runner


class RunRequest(BaseModel):
    """Request model for agent execution."""

    text_input: str
    conversation_id: str = Field(default_factory=lambda: f"conv_{uuid4().hex}")
    user_id: str = "api_user"


class RunResponse(BaseModel):
    """Response model for synchronous agent execution."""

    response: str
    conversation_id: str
    agent_name: str


def create_app(main_agent: Agent, all_agents: dict[str, Agent]) -> FastAPI:
    """Create FastAPI app for local agent serving."""
    app = FastAPI(
        title="Domyn Agents API",
        description="Local REST API for Domyn Agents",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        return {"status": "healthy", "agent": main_agent.name}

    @app.get("/agents")
    async def list_agents() -> list[dict[str, Any]]:
        return [{"name": a.name, "description": a.description} for a in all_agents.values()]

    @app.get("/agents/{name}")
    async def get_agent_info(name: str) -> dict[str, Any]:
        agent = next((a for a in all_agents.values() if a.name == name), None)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
        return {
            "name": agent.name,
            "description": agent.description,
            "tools": [t.name for t in agent.tools],
        }

    @app.post("/run", response_model=RunResponse)
    async def run_agent(request: RunRequest) -> RunResponse:
        """Run agent and collect full response."""
        response_text = await _collect_response(main_agent, request)
        return RunResponse(
            response=response_text,
            conversation_id=request.conversation_id,
            agent_name=main_agent.name,
        )

    @app.post("/execute")
    async def stream_agent(request: RunRequest) -> StreamingResponse:
        """Stream all agent events via SSE."""
        return StreamingResponse(
            _stream_events(main_agent, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


async def _collect_response(agent: Agent, request: RunRequest) -> str:
    """Run agent and return the final response text."""
    runner = Runner()
    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text=request.text_input)],
    )

    response_parts: list[str] = []

    async for event in runner.run(
        application_name="serve",
        user_id=request.user_id,
        user_input=input_event,
        root=agent,
        entrypoint_name=agent.name,
        conversation_id=request.conversation_id,
    ):
        if event.event_type == ExecutionEventType.RESPONSE and not event.is_partial:
            for part in event.content:
                if part.text:
                    response_parts.append(part.text)

    return "\n".join(response_parts) if response_parts else ""


async def _stream_events(agent: Agent, request: RunRequest) -> AsyncGenerator[str, None]:
    """Yield all Runner events as SSE data frames."""
    runner = Runner()
    input_event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text=request.text_input)],
    )

    try:
        async for event in runner.run(
            application_name="serve",
            user_id=request.user_id,
            user_input=input_event,
            root=agent,
            entrypoint_name=agent.name,
            conversation_id=request.conversation_id,
        ):
            yield f"data: {event.model_dump_json()}\n\n"
    except Exception as e:
        error_event = BaseEvent(
            event_type=ExecutionEventType.ERROR_OCCURRED,
            author="server",
            error_message=str(e),
        )
        yield f"data: {error_event.model_dump_json()}\n\n"
    finally:
        yield "event: end\ndata: {}\n\n"


class ServerInterface:
    """Wraps FastAPI app with uvicorn lifecycle."""

    def __init__(self, main_agent: Agent, all_agents: dict[str, Agent]):
        self.main_agent = main_agent
        self.app = create_app(main_agent, all_agents)

    async def start_async(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        config = uvicorn.Config(self.app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
