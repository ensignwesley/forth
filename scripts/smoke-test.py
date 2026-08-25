#!/usr/bin/env python3
"""Deployed smoke test for Wesley's Forth.

Checks the public HTTP health endpoint and evaluates one expression through the
WebSocket REPL. Uses only the Python standard library.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import socket
import ssl
import struct
import sys
import urllib.parse
import urllib.request


DEFAULT_BASE_URL = "https://wesley.thesisko.com/forth"
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def fail(message: str) -> None:
    print(f"not ok forth smoke: {message}", file=sys.stderr)
    raise SystemExit(1)


def base_url() -> str:
    raw = None
    for index, arg in enumerate(sys.argv[1:], start=1):
        if arg == "--url" and index + 1 < len(sys.argv):
            raw = sys.argv[index + 1]
            break
        if arg.startswith("--url="):
            raw = arg.split("=", 1)[1]
            break
        if not arg.startswith("-"):
            raw = arg
            break
    raw = raw or os.environ.get("FORTH_BASE_URL", DEFAULT_BASE_URL)
    parsed = urllib.parse.urlparse(raw.rstrip("/"))
    if parsed.scheme in {"ws", "wss"}:
        scheme = "https" if parsed.scheme == "wss" else "http"
        path = parsed.path.removesuffix("/ws") or "/forth"
        parsed = parsed._replace(scheme=scheme, path=path, params="", query="", fragment="")
        return urllib.parse.urlunparse(parsed).rstrip("/")
    return raw.rstrip("/")


def check_health(base: str) -> dict:
    url = f"{base}/health"
    req = urllib.request.Request(url, headers={"User-Agent": "forth-smoke/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            if res.status != 200:
                fail(f"health returned HTTP {res.status}")
            if "application/json" not in (res.headers.get("content-type") or ""):
                fail(f"health content-type is {res.headers.get('content-type')!r}")
            if (res.headers.get("x-content-type-options") or "").lower() != "nosniff":
                fail("health missing X-Content-Type-Options: nosniff")
            if (res.headers.get("referrer-policy") or "").lower() != "no-referrer":
                fail("health missing Referrer-Policy: no-referrer")
            if "default-src 'self'" not in (res.headers.get("content-security-policy") or ""):
                fail("health missing baseline Content-Security-Policy")
            body = json.loads(res.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - this is a CLI smoke test
        fail(f"health request failed: {exc}")

    if body.get("ok") is not True:
        fail("health ok is not true")
    if body.get("service") != "forth":
        fail(f"unexpected service {body.get('service')!r}")
    return body


def ws_url(base: str) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(base)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = parsed.path.rstrip("/") + "/ws"
    return parsed._replace(scheme=scheme, path=path, params="", query="", fragment="")


def send_frame(sock: socket.socket, text: str) -> None:
    payload = text.encode("utf-8")
    mask = secrets.token_bytes(4)
    header = bytearray([0x81])
    n = len(payload)
    if n < 126:
        header.append(0x80 | n)
    elif n < 65536:
        header.append(0x80 | 126)
        header.extend(struct.pack("!H", n))
    else:
        header.append(0x80 | 127)
        header.extend(struct.pack("!Q", n))
    masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
    sock.sendall(bytes(header) + mask + masked)


def recv_frame(sock: socket.socket) -> str:
    first = sock.recv(2)
    if len(first) < 2:
        fail("websocket closed before frame")
    opcode = first[0] & 0x0F
    length = first[1] & 0x7F
    if length == 126:
        length = struct.unpack("!H", sock.recv(2))[0]
    elif length == 127:
        length = struct.unpack("!Q", sock.recv(8))[0]
    payload = b""
    while len(payload) < length:
        chunk = sock.recv(length - len(payload))
        if not chunk:
            fail("websocket closed mid-frame")
        payload += chunk
    if opcode == 0x8:
        fail("websocket closed by server")
    return payload.decode("utf-8", errors="replace")


def check_websocket(base: str) -> str:
    url = ws_url(base)
    port = url.port or (443 if url.scheme == "wss" else 80)
    host = url.hostname
    if not host:
        fail("websocket URL has no host")

    raw = socket.create_connection((host, port), timeout=10)
    raw.settimeout(10)
    sock: socket.socket
    if url.scheme == "wss":
        sock = ssl.create_default_context().wrap_socket(raw, server_hostname=host)
    else:
        sock = raw

    key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
    path = urllib.parse.urlunparse(("", "", url.path or "/", "", url.query, ""))
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "User-Agent: forth-smoke/1.0\r\n"
        "\r\n"
    )
    sock.sendall(request.encode("ascii"))
    response = b""
    while b"\r\n\r\n" not in response:
        response += sock.recv(4096)
        if len(response) > 16384:
            fail("websocket handshake response too large")
    head = response.decode("iso-8859-1", errors="replace")
    if " 101 " not in head.split("\r\n", 1)[0]:
        fail(f"websocket handshake failed: {head.splitlines()[0] if head else 'empty response'}")
    expected = base64.b64encode(hashlib.sha1((key + WS_GUID).encode()).digest()).decode("ascii")
    if f"sec-websocket-accept: {expected.lower()}" not in head.lower():
        fail("websocket accept header mismatch")

    banner = recv_frame(sock)
    if "Wesley's Forth" not in banner:
        fail("missing Forth welcome banner")
    send_frame(sock, "2 3 + .")
    result = recv_frame(sock)
    sock.close()
    if "5" not in result or "ok" not in result.lower():
        fail(f"unexpected eval result: {result!r}")
    return result.strip()


def main() -> None:
    base = base_url()
    health = check_health(base)
    result = check_websocket(base)
    print(f"ok forth smoke {base} version={health.get('version')} eval={result!r}")


if __name__ == "__main__":
    main()
