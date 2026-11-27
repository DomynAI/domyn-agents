"""The model request's HTTP client: split timeouts and TCP keepalive.

The fake-server tests scale the timeouts down to fractions of a second; what
they check is which limit applies to which kind of call.
"""

import asyncio
import json
import socket
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from domyn.agents.llm import http_client
from domyn.agents.llm.base import LLMMessage
from domyn.agents.llm.custom import CustomLLMProvider
from domyn.agents.llm.http_client import (
    CONNECT_TIMEOUT_S,
    keepalive_socket_options,
    llm_client,
    request_timeout,
)
from domyn.agents.llm.openai import OpenAIProvider
from domyn.agents.llm.vllm import VLLMProvider

PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")

MESSAGES = [LLMMessage(role="user", content="hi")]

COMPLETION = {
    "choices": [{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


@pytest.fixture(autouse=True)
def no_proxy_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """A proxy in the developer's environment would carry the calls to the fake server."""
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)


class TestTheTimeouts:
    def test_only_the_read_timeout_varies(self) -> None:
        timeout = request_timeout(None)
        assert timeout.read is None
        assert timeout.connect == CONNECT_TIMEOUT_S == 10
        assert timeout.write == 60
        assert timeout.pool == 60

    def test_a_read_timeout_is_kept(self) -> None:
        assert request_timeout(300).read == 300


class TestKeepalive:
    def test_keepalive_is_on(self) -> None:
        assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in keepalive_socket_options()

    def test_the_idle_time_is_set_where_the_platform_names_it(self) -> None:
        idle = getattr(socket, "TCP_KEEPIDLE", None) or getattr(socket, "TCP_KEEPALIVE", None)
        if idle is None:
            pytest.skip("this platform sets no keepalive idle time")
        assert (socket.IPPROTO_TCP, idle, 60) in keepalive_socket_options()

    def test_every_option_is_accepted_by_a_real_socket(self) -> None:
        """A socket option the platform refuses would fail every connection."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            for level, name, value in keepalive_socket_options():
                sock.setsockopt(level, name, value)
                if name == socket.SO_KEEPALIVE:
                    # A flag: macOS reads it back as its bit (8), not as 1.
                    assert sock.getsockopt(level, name)
                else:
                    assert sock.getsockopt(level, name) == value

    @pytest.mark.asyncio
    async def test_the_client_sets_them_on_its_connections(self) -> None:
        async with llm_client(None) as client:
            transport = client._transport_for_url(httpx.URL("https://example.com"))
            assert isinstance(transport, httpx.AsyncHTTPTransport)
            assert transport._pool._socket_options == keepalive_socket_options()

    @pytest.mark.asyncio
    async def test_a_proxy_from_the_environment_is_still_used(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A client given its own transport ignores the proxy variables unless they are mounted."""
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:3128")
        monkeypatch.setenv("NO_PROXY", "models.internal")
        async with llm_client(None) as client:
            proxied = client._transport_for_url(httpx.URL("https://api.example.com"))
            direct = client._transport_for_url(httpx.URL("https://models.internal"))
            assert proxied is not client._transport
            assert proxied._pool._proxy_url.host == b"proxy.internal"  # type: ignore[attr-defined]
            assert proxied._pool._socket_options == keepalive_socket_options()  # type: ignore[attr-defined]
            assert direct is client._transport


class TestEachProviderPicksTheReadTimeout:
    """A call that is not streamed waits for the reply; a streamed one keeps its gap limit."""

    @pytest.mark.parametrize(
        ("provider", "streamed"),
        [
            (VLLMProvider(url="http://x/v1/chat/completions", model_name="m"), 300.0),
            (CustomLLMProvider(url="http://x/v1/chat/completions", model_name="m"), 300.0),
            (OpenAIProvider(model_name="gpt-5", api_key="k"), 60.0),
        ],
        ids=["vllm", "custom", "openai"],
    )
    def test_the_defaults(self, provider: Any, streamed: float) -> None:
        assert provider.timeout == streamed
        assert provider.non_stream_read_timeout is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("module", ["vllm", "custom", "openai"])
    async def test_the_client_is_built_with_them(
        self, module: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reads: list[float | None] = []

        def spy(read_timeout: float | None) -> httpx.AsyncClient:
            reads.append(read_timeout)
            raise _StoppedError

        monkeypatch.setattr(f"domyn.agents.llm.{module}.llm_client", spy)
        provider = {
            "vllm": lambda: VLLMProvider(
                url="http://x", model_name="m", timeout=7, non_stream_read_timeout=900
            ),
            "custom": lambda: CustomLLMProvider(
                url="http://x", model_name="m", timeout=7, non_stream_read_timeout=900
            ),
            "openai": lambda: OpenAIProvider(
                model_name="gpt-5", api_key="k", timeout=7, non_stream_read_timeout=900
            ),
        }[module]()

        with pytest.raises(_StoppedError):
            await provider.generate(MESSAGES)
        with pytest.raises(_StoppedError):
            async for _ in provider.generate_streaming(MESSAGES):
                pass
        assert reads == [900, 7]


class _StoppedError(Exception):
    pass


Handler = Callable[[asyncio.StreamWriter], Awaitable[None]]


@asynccontextmanager
async def fake_server(handler: Handler) -> AsyncIterator[str]:
    """An HTTP server on loopback that reads one request and leaves the answer to ``handler``."""

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        length = next(
            (
                int(line.split(b":", 1)[1])
                for line in head.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            ),
            0,
        )
        await reader.readexactly(length)
        try:
            await handler(writer)
        finally:
            writer.close()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/v1/chat/completions"
    finally:
        server.close()


def _response(body: bytes, content_type: str = "application/json") -> bytes:
    return (
        f"HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
    ).encode() + body


class TestAgainstAServer:
    @pytest.mark.asyncio
    async def test_a_reply_slower_than_the_streamed_limit_succeeds_when_not_streamed(
        self,
    ) -> None:
        async def slow_reply(writer: asyncio.StreamWriter) -> None:
            await asyncio.sleep(0.6)
            writer.write(_response(json.dumps(COMPLETION).encode()))
            await writer.drain()

        async with fake_server(slow_reply) as url:
            provider = VLLMProvider(url=url, model_name="m", timeout=0.2)
            response = await provider.generate(MESSAGES)
        assert response.content == "done"

    @pytest.mark.asyncio
    async def test_a_streamed_reply_that_goes_silent_fails_at_the_read_timeout(self) -> None:
        async def stall(writer: asyncio.StreamWriter) -> None:
            chunk = {"choices": [{"delta": {"content": "hel"}, "finish_reason": None}]}
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\n\r\n"
            )
            event = f"data: {json.dumps(chunk)}\n\n".encode()
            writer.write(f"{len(event):x}\r\n".encode() + event + b"\r\n")
            await writer.drain()
            await asyncio.sleep(5)

        async with fake_server(stall) as url:
            provider = VLLMProvider(url=url, model_name="m", timeout=0.3)
            start = time.monotonic()
            with pytest.raises(httpx.ReadTimeout):
                async for _ in provider.generate_streaming(MESSAGES):
                    pass
        assert time.monotonic() - start < 3

    @pytest.mark.asyncio
    async def test_a_capped_reply_that_is_not_streamed_fails_at_its_cap(self) -> None:
        async def never(writer: asyncio.StreamWriter) -> None:
            await asyncio.sleep(5)

        async with fake_server(never) as url:
            provider = VLLMProvider(url=url, model_name="m", non_stream_read_timeout=0.3)
            with pytest.raises(httpx.ReadTimeout):
                await provider.generate(MESSAGES)

    @pytest.mark.asyncio
    async def test_an_address_nobody_answers_fails_at_the_connect_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """10.255.255.1 drops the connection attempt; a network without a route refuses it."""
        monkeypatch.setattr(http_client, "CONNECT_TIMEOUT_S", 0.3)
        provider = VLLMProvider(url="http://10.255.255.1:81/v1/chat/completions", model_name="m")
        start = time.monotonic()
        with pytest.raises((httpx.ConnectTimeout, httpx.ConnectError)):
            await provider.generate(MESSAGES)
        assert time.monotonic() - start < 3
