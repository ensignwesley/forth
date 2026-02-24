#!/usr/bin/env python3
"""
Forth WebSocket server.

Each WebSocket connection gets its own isolated ForthFull interpreter.
HTTP GET /forth/ serves the HTML REPL page.
HTTP GET /forth/ws upgrades to WebSocket.

Pure stdlib: socket, threading, hashlib, base64, struct, http.server
No external dependencies.
"""

import base64
import hashlib
import http.server
import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path

# Ensure this script's directory is on the path so we can import forth.py
sys.path.insert(0, str(Path(__file__).parent))
from forth import ForthFull

PORT     = 3005
WS_GUID  = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
HTML_PATH = Path(__file__).parent / 'forth.html'


# ── WebSocket frame helpers ───────────────────────────────────────────────────

def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    """Receive exactly n bytes, or None on disconnect."""
    buf = b''
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except OSError:
            return None
        if not chunk:
            return None
        buf += chunk
    return buf


def ws_recv(sock: socket.socket):
    """
    Read one WebSocket frame. Returns (opcode, text) or (None, None) on close.
    Handles fragmented text frames (FIN=0) by concatenating.
    """
    payload_acc = b''
    while True:
        header = _recv_exact(sock, 2)
        if not header:
            return None, None

        fin    = (header[0] >> 7) & 1
        opcode = header[0] & 0x0F
        masked = (header[1] >> 7) & 1
        length = header[1] & 0x7F

        if length == 126:
            ext = _recv_exact(sock, 2)
            if not ext: return None, None
            length = struct.unpack('!H', ext)[0]
        elif length == 127:
            ext = _recv_exact(sock, 8)
            if not ext: return None, None
            length = struct.unpack('!Q', ext)[0]

        mask_key = b''
        if masked:
            mask_key = _recv_exact(sock, 4)
            if not mask_key: return None, None

        raw = _recv_exact(sock, length) if length else b''
        if raw is None: return None, None

        if masked and mask_key:
            raw = bytes(b ^ mask_key[i % 4] for i, b in enumerate(raw))

        if opcode == 0x8:   # Close
            return None, None
        if opcode == 0x9:   # Ping → Pong
            ws_send(sock, raw.decode('utf-8', errors='replace'), opcode=0xA)
            continue
        if opcode == 0xA:   # Pong (ignore)
            continue

        payload_acc += raw

        if fin:
            return opcode, payload_acc.decode('utf-8', errors='replace')
        # else: continuation, keep reading


def ws_send(sock: socket.socket, text: str, opcode: int = 0x1):
    """Send a WebSocket text (or other opcode) frame. No masking (server→client)."""
    payload = text.encode('utf-8')
    n = len(payload)
    if n < 126:
        header = bytes([0x80 | opcode, n])
    elif n < 65536:
        header = bytes([0x80 | opcode, 126]) + struct.pack('!H', n)
    else:
        header = bytes([0x80 | opcode, 127]) + struct.pack('!Q', n)
    try:
        sock.sendall(header + payload)
    except OSError:
        pass


def ws_handshake(sock: socket.socket, key: str) -> bool:
    """Send WebSocket upgrade response. Returns True on success."""
    accept = base64.b64encode(
        hashlib.sha1((key + WS_GUID).encode()).digest()
    ).decode()
    response = (
        'HTTP/1.1 101 Switching Protocols\r\n'
        'Upgrade: websocket\r\n'
        'Connection: Upgrade\r\n'
        f'Sec-WebSocket-Accept: {accept}\r\n'
        '\r\n'
    )
    try:
        sock.sendall(response.encode())
        return True
    except OSError:
        return False


# ── HTTP request parsing ─────────────────────────────────────────────────────

def read_http_request(sock: socket.socket):
    """
    Read an HTTP request from the socket.
    Returns (method, path, headers) or (None, None, None) on error.
    """
    data = b''
    try:
        while b'\r\n\r\n' not in data:
            chunk = sock.recv(4096)
            if not chunk:
                return None, None, None
            data += chunk
            if len(data) > 16384:
                return None, None, None
    except OSError:
        return None, None, None

    head = data.split(b'\r\n\r\n', 1)[0].decode('utf-8', errors='replace')
    lines = head.split('\r\n')
    if not lines:
        return None, None, None

    parts = lines[0].split(' ')
    if len(parts) < 2:
        return None, None, None
    method, path = parts[0], parts[1]

    headers = {}
    for line in lines[1:]:
        if ':' in line:
            k, _, v = line.partition(':')
            headers[k.strip().lower()] = v.strip()

    return method, path, headers


def http_send(sock: socket.socket, status: int, content_type: str, body: bytes):
    response = (
        f'HTTP/1.1 {status} {"OK" if status==200 else "Not Found"}\r\n'
        f'Content-Type: {content_type}\r\n'
        f'Content-Length: {len(body)}\r\n'
        'Connection: close\r\n'
        '\r\n'
    ).encode() + body
    try:
        sock.sendall(response)
    except OSError:
        pass


# ── Connection handler ────────────────────────────────────────────────────────

def handle_connection(sock: socket.socket, addr):
    sock.settimeout(30)
    method, path, headers = read_http_request(sock)
    if method is None:
        sock.close()
        return

    # Normalize path (strip /forth prefix if proxied)
    norm = path.rstrip('/')
    if norm in ('', '/forth'):
        norm = '/forth'

    # WebSocket upgrade
    if headers.get('upgrade', '').lower() == 'websocket' and norm in ('/forth/ws', '/forth/ws/'):
        key = headers.get('sec-websocket-key', '')
        if not ws_handshake(sock, key):
            sock.close()
            return
        handle_ws_session(sock, addr)
        sock.close()
        return

    # Serve HTML page
    if method == 'GET' and norm in ('/forth', '/forth/', '/forth/index.html'):
        if HTML_PATH.exists():
            body = HTML_PATH.read_bytes()
        else:
            body = b'<html><body>forth.html not found</body></html>'
        http_send(sock, 200, 'text/html; charset=utf-8', body)
        sock.close()
        return

    http_send(sock, 404, 'text/plain', b'Not found')
    sock.close()


def handle_ws_session(sock: socket.socket, addr):
    """Run a Forth WebSocket session — one ForthFull per connection."""
    forth = ForthFull()
    sock.settimeout(300)  # 5-minute idle timeout

    # Send welcome banner
    ws_send(sock, '=== Wesley\'s Forth ===\nType Forth expressions. Multi-line: end with a blank line.\nType WORDS to see vocabulary. Type HELP for examples.\n')

    while True:
        opcode, text = ws_recv(sock)
        if opcode is None:
            break
        if opcode != 0x1:   # only text frames
            continue

        text = text.strip()
        if not text:
            continue

        if text.upper() == 'HELP':
            ws_send(sock, HELP_TEXT)
            continue

        if text.upper() == 'RESET':
            forth = ForthFull()
            ws_send(sock, '( Environment reset )\n ok')
            continue

        result = forth.interpret(text)
        ws_send(sock, result)


HELP_TEXT = """\
Examples:
  2 3 +  .                        → 5
  : SQUARE DUP * ;  7 SQUARE .    → 49
  : FACT DUP 1 > IF DUP 1- RECURSE * THEN ;  10 FACT .  → 3628800
  5 0 DO  I .  LOOP               → 0 1 2 3 4
  VARIABLE X  42 X !  X @ .       → 42
  ." Hello, World!"               → Hello, World!
  : FIZZ 15 MOD 0= ;  15 FIZZ .   → -1

Control flow:
  IF ... ELSE ... THEN
  BEGIN ... UNTIL
  BEGIN ... WHILE ... REPEAT
  n limit DO ... LOOP
  n limit DO ... +LOOP

Stack words:  DUP DROP SWAP OVER ROT .S DEPTH
Return stack: >R  R>  R@
Commands:     WORDS  SEE wordname  RESET  HELP
"""


# ── Main server ───────────────────────────────────────────────────────────────

def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', PORT))
    srv.listen(32)
    print(f'Forth WebSocket server on 127.0.0.1:{PORT}', flush=True)

    while True:
        try:
            conn, addr = srv.accept()
            t = threading.Thread(target=handle_connection, args=(conn, addr), daemon=True)
            t.start()
        except KeyboardInterrupt:
            break
        except OSError as e:
            print(f'Accept error: {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
