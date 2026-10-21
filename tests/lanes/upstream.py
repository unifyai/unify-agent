"""A stand-in for the provider behind the lane relay: an OpenAI-compatible chat-completions endpoint on this machine.

It keeps every request it gets (headers and body), counts the calls in flight at once, and answers by its ``mode``: a
JSON reply with usage and a charge, a server-sent event stream, an error status, or a held reply that waits for
:meth:`Upstream.release`. Nothing leaves the machine and no key is real.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

#: Key-shaped, so the redaction patterns would match it, and fake.
FAKE_KEY = "sk-or-v1-" + "0123456789abcdef" * 4  # pragma: allowlist secret
MODEL = "openai/gpt-6-luna"


def reply(
    content: str = "ok", *, cost=0.00125, gid: str = "gen-stub-1", usage: bool = True
) -> dict:
    doc = {
        "id": gid,
        "object": "chat.completion",
        "created": 0,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            },
        ],
    }
    if usage:
        doc["usage"] = {
            "prompt_tokens": 120,
            "completion_tokens": 7,
            "total_tokens": 127,
            "prompt_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 20},
            "completion_tokens_details": {"reasoning_tokens": 3},
        }
        if cost is not None:
            doc["usage"]["cost"] = cost
    return doc


class Upstream:
    """``mode``: ``json`` (default), ``stream``, ``stream_status`` (a stream at ``status``), ``error`` (``status``, a
    JSON error without usage), ``hold`` (wait for :meth:`release`), or a
    callable taking the request body and returning ``(status, doc)``. ``echo`` (bytes) is put into every reply, to
    show what the relay redacts. ``delay`` holds each reply that long."""

    def __init__(
        self,
        mode="json",
        *,
        status: int = 429,
        cost=0.00125,
        echo: bytes = b"",
        delay: float = 0.0,
    ):
        self.mode, self.status, self.cost, self.echo, self.delay = (
            mode,
            status,
            cost,
            echo,
            delay,
        )
        self.requests: list[dict] = []
        self.in_flight = 0
        self.most_in_flight = 0
        self.lock = threading.Lock()
        self.gate = threading.Event()
        owner = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                owner._serve(self)

            def do_GET(self):
                owner.requests.append(
                    {"method": "GET", "path": self.path, "headers": dict(self.headers)}
                )
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/api/v1"

    def release(self) -> None:
        self.gate.set()

    def close(self) -> None:
        self.gate.set()
        self.server.shutdown()
        self.server.server_close()

    def _serve(self, h: BaseHTTPRequestHandler) -> None:
        raw = h.rfile.read(int(h.headers["Content-Length"]))
        body = json.loads(raw)
        with self.lock:
            self.requests.append(
                {
                    "method": "POST",
                    "path": h.path,
                    "headers": {k.lower(): v for k, v in h.headers.items()},
                    "body": body,
                    "raw": raw,
                }
            )
            self.in_flight += 1
            self.most_in_flight = max(self.most_in_flight, self.in_flight)
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.mode == "hold":
                self.gate.wait(120)
            if callable(self.mode):
                status, doc = self.mode(body)
                self._json(h, status, doc)
            elif self.mode == "error":
                self._json(
                    h,
                    self.status,
                    {
                        "error": {
                            "code": self.status,
                            "message": "rate limited " + self.echo.decode(),
                        }
                    },
                )
            elif self.mode in ("stream", "stream_status") or body.get("stream"):
                self._stream(h)
            else:
                self._json(h, 200, reply("ok " + self.echo.decode(), cost=self.cost))
        finally:
            with self.lock:
                self.in_flight -= 1

    @staticmethod
    def _json(h, status: int, doc: dict) -> None:
        data = json.dumps(doc).encode()
        h.send_response(status)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(data)))
        h.end_headers()
        h.wfile.write(data)

    def _stream(self, h) -> None:
        h.send_response(self.status if self.mode == "stream_status" else 200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Transfer-Encoding", "chunked")
        h.end_headers()
        events = [
            b": OPENROUTER PROCESSING\n\n",
            b'data: {"id":"gen-stream-1","choices":[{"index":0,"delta":{"role":"assistant","content":"hel'
            + self.echo
            + b'"}}]}\n\n',
            b'data: {"id":"gen-stream-1","choices":[{"index":0,"finish_reason":"stop","delta":{"content":"lo"}}]'
            + (
                b',"usage":{"prompt_tokens":40,"completion_tokens":2,"cost":0.0004}'
                if self.cost is not None
                else b""
            )
            + b"}\n\n",
            b"data: [DONE]\n\n",
        ]
        for event in events:
            h.wfile.write(b"%x\r\n%s\r\n" % (len(event), event))
            h.wfile.flush()
        h.wfile.write(b"0\r\n\r\n")
