"""The HTTP client a provider sends a model request with.

A single timeout number means a different thing for each kind of call. httpx's
``read`` timeout is the longest gap between two pieces of data: on a streamed
call that is the gap between two chunks, so it only catches a stalled server;
on a call that is not streamed the server sends nothing until the reply is
whole, so the same number caps the whole generation, and a reasoning model that
thinks for longer than it fails while the server is still working (300 s on
vLLM: 27 of 596 trials of one run, the longest reply measured at 1,078 s).

So the timeout is split (``request_timeout``): connecting, writing the request
and waiting for a pooled connection keep short limits, so an address nobody
answers fails fast, and only ``read`` follows the kind of call — a provider's
own streamed value, and by default none at all for a call that is not streamed.

With no read limit a dead peer is found by TCP keepalive instead
(``keepalive_socket_options``): the kernel probes a silent connection, so a host
that went away or a broken link fails the call in about two minutes, and the
probes keep load balancers that drop idle connections from cutting a long
reply. Keepalive reaches the first hop only: a server dying behind a proxy that
still answers the probes is not noticed, and is left to that proxy's timeouts.
"""

import socket
from typing import Final

import httpx

CONNECT_TIMEOUT_S: Final = 10.0
WRITE_TIMEOUT_S: Final = 60.0
POOL_TIMEOUT_S: Final = 60.0

# A silent connection is probed after a minute, then every 15 s; four
# unanswered probes close it: a dead peer is found in about two minutes.
KEEPALIVE_IDLE_S: Final = 60
KEEPALIVE_INTERVAL_S: Final = 15
KEEPALIVE_PROBES: Final = 4


def request_timeout(read: float | None) -> httpx.Timeout:
    """The timeouts of one model request: ``read`` is the only one that varies.

    ``None`` waits for the reply however long the model takes.
    """
    return httpx.Timeout(
        connect=CONNECT_TIMEOUT_S, read=read, write=WRITE_TIMEOUT_S, pool=POOL_TIMEOUT_S
    )


def keepalive_socket_options() -> list[tuple[int, int, int]]:
    """TCP keepalive on, with the probe timings this platform lets a socket set.

    Only the constants the platform defines are used: macOS names the idle time
    ``TCP_KEEPALIVE`` rather than ``TCP_KEEPIDLE``, and a platform with none of
    them still gets keepalive with the system's timings.
    """
    options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    idle = getattr(socket, "TCP_KEEPIDLE", None)
    if idle is None:
        idle = getattr(socket, "TCP_KEEPALIVE", None)
    for name, value in (
        (idle, KEEPALIVE_IDLE_S),
        (getattr(socket, "TCP_KEEPINTVL", None), KEEPALIVE_INTERVAL_S),
        (getattr(socket, "TCP_KEEPCNT", None), KEEPALIVE_PROBES),
    ):
        if name is not None:
            options.append((socket.IPPROTO_TCP, name, value))
    return options


def _environment_proxies() -> dict[str, str | None] | None:
    """The proxies the environment names (``HTTPS_PROXY``, ``NO_PROXY``…), as httpx reads them.

    ``None`` when this httpx no longer has the function, which is not public.
    """
    try:
        from httpx._utils import get_environment_proxies
    except ImportError:  # pragma: no cover - depends on the installed httpx
        return None
    return get_environment_proxies()


def llm_client(read_timeout: float | None) -> httpx.AsyncClient:
    """A client for one model request: the split timeouts, and keepalive on every connection.

    A client given its own transport ignores the proxy variables of the
    environment, so they are read here and mounted with the same socket
    options: keepalive then probes the connection to the proxy. When they cannot
    be read, the client is httpx's default one, without keepalive, rather than
    one that skips the user's proxy.
    """
    timeout = request_timeout(read_timeout)
    proxies = _environment_proxies()
    if proxies is None:  # pragma: no cover - depends on the installed httpx
        return httpx.AsyncClient(timeout=timeout)
    options = keepalive_socket_options()
    mounts: dict[str, httpx.AsyncBaseTransport | None] = {
        pattern: None
        if proxy is None
        else httpx.AsyncHTTPTransport(proxy=proxy, socket_options=options)
        for pattern, proxy in proxies.items()
    }
    return httpx.AsyncClient(
        timeout=timeout,
        transport=httpx.AsyncHTTPTransport(socket_options=options),
        mounts=mounts,
    )
