"""An allowlisting HTTPS proxy, run inside a container of its own.

The Docker sandbox puts the command container on an ``--internal`` network, which
has no route out. This proxy is the only thing attached to both that network and
the bridge, so everything a sandboxed command can reach is whatever this lets
through. It is passed to the proxy container as ``python -c "<this file>"``, which
is why it is standard library only and one file.

What it does, and deliberately nothing more:

* ``CONNECT host:443`` to a host on the allowlist — a tunnel, no TLS interception,
  so nothing here sees or alters what pip or npm download.
* Everything else is refused with a ``403``: other hosts, other ports, plain
  ``http://`` requests.
* A host is also refused when it *resolves* to anything that is not a public
  address (loopback, private ranges, link-local, cloud metadata). An allowlisted
  name that points inward is how a proxy becomes a way into the network it sits on.

Each refusal is one ``DENY host:port reason`` line on stderr, which the sandbox
backend reads back to tell the user what was blocked and how to allow it.
"""

import argparse
import asyncio
import contextlib
import ipaddress
import sys

ALLOWED_PORT = 443
HEADER_LIMIT = 16 * 1024
CONNECT_TIMEOUT_S = 15.0
IDLE_TIMEOUT_S = 600.0


def host_allowed(host: str, allow: list[str]) -> bool:
    """Whether ``host`` is on the list. ``*.example.com`` matches subdomains only."""
    host = host.lower().rstrip(".")
    for entry in allow:
        if entry.startswith("*."):
            if host.endswith(entry[1:]) and len(host) > len(entry) - 1:
                return True
        elif host == entry:
            return True
    return False


def public_address(address: str) -> bool:
    try:
        return ipaddress.ip_address(address.split("%")[0]).is_global
    except ValueError:
        return False


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await asyncio.wait_for(reader.read(65536), IDLE_TIMEOUT_S)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (OSError, TimeoutError, asyncio.IncompleteReadError):
        pass
    finally:
        with contextlib.suppress(OSError):
            writer.close()


def _deny(target: str, reason: str) -> None:
    print(f"DENY {target} {reason}", file=sys.stderr, flush=True)


async def _refuse(writer: asyncio.StreamWriter, message: str) -> None:
    body = f"domyn sandbox: {message}\n".encode()
    writer.write(
        b"HTTP/1.1 403 Forbidden\r\nContent-Type: text/plain\r\nConnection: close\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body
    )
    try:
        await writer.drain()
    finally:
        writer.close()


async def handle(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, allow: list[str]
) -> None:
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), CONNECT_TIMEOUT_S)
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, TimeoutError, OSError):
        writer.close()
        return
    if len(head) > HEADER_LIMIT:
        writer.close()
        return

    request = head.split(b"\r\n", 1)[0].decode("latin-1", "replace")
    parts = request.split()
    if len(parts) != 3 or parts[0].upper() != "CONNECT":
        _deny(parts[1] if len(parts) > 1 else "?", "only HTTPS through CONNECT is allowed")
        await _refuse(writer, "only HTTPS (CONNECT) is allowed from the sandbox")
        return

    target = parts[1]
    host, _, port_text = target.rpartition(":")
    host = host.strip("[]")
    try:
        port = int(port_text)
    except ValueError:
        port = -1

    if not host_allowed(host, allow):
        _deny(target, "not in the allowlist")
        await _refuse(writer, f"{host} is not in the allowlist ({', '.join(allow)})")
        return
    if port != ALLOWED_PORT:
        _deny(target, f"port {port} is not allowed")
        await _refuse(writer, f"only port {ALLOWED_PORT} is allowed")
        return

    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, type=1)
    except OSError:
        _deny(target, "does not resolve")
        await _refuse(writer, f"{host} does not resolve")
        return
    addresses = [str(info[4][0]) for info in infos]
    if not addresses or not all(public_address(a) for a in addresses):
        _deny(target, "resolves to a non-public address")
        await _refuse(writer, f"{host} resolves to a non-public address")
        return

    upstream_reader: asyncio.StreamReader | None = None
    upstream_writer: asyncio.StreamWriter | None = None
    for address in addresses:
        try:
            upstream_reader, upstream_writer = await asyncio.wait_for(
                asyncio.open_connection(address, port), CONNECT_TIMEOUT_S
            )
            break
        except (OSError, TimeoutError):
            continue
    if upstream_reader is None or upstream_writer is None:
        _deny(target, "unreachable")
        await _refuse(writer, f"{host} is unreachable")
        return

    writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
    await writer.drain()
    await asyncio.gather(
        _pipe(reader, upstream_writer),
        _pipe(upstream_reader, writer),
    )


async def serve(port: int, allow: list[str]) -> None:
    server = await asyncio.start_server(
        lambda r, w: handle(r, w, allow), host="0.0.0.0", port=port, limit=HEADER_LIMIT
    )
    print(f"READY {port}", flush=True)
    async with server:
        await server.serve_forever()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3128)
    parser.add_argument("--allow", default="")
    args = parser.parse_args(argv)
    allow = [h.strip().lower() for h in args.allow.split(",") if h.strip()]
    asyncio.run(serve(args.port, allow))


if __name__ == "__main__":
    main()
