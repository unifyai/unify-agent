"""A scripted model behind the memory lanes' loopback routes, for the conformance suite.

The actor's own calls are scripted in process (:mod:`tests.scripted_model`). The memory agents run elsewhere: the
episode-end fork in its own worker process, the writer service detached, helpers inside the service's passes. Each
reaches its model through a loopback route (``http://127.0.0.1:<port>/<lane>/v1``, the form the build accepts), so
this module serves that route on 127.0.0.1: an OpenAI-compatible ``/chat/completions`` per lane, answered from the
lane's script, in plain JSON or as an event stream with a final usage frame (``stream: true``); and ``.../embeddings``
on any path, answered with fixed word-hash vectors (the run's embedding route, so no process fetches a model).

A script entry is a callable ``(body) -> reply`` or a reply; a reply is an assistant message dict (``content``,
``tool_calls``) or a plain text. Every call is recorded (lane, path, the request body, the request's ``x-unify-*``
metadata headers), so a test can assert what each agent was sent and which ids each call carried. A lane whose script
runs out answers HTTP 500 with a fixed text, and the test fails on ``unexpected``. No key is read or needed: any
bearer token is accepted and none is recorded.
"""

from __future__ import annotations

import itertools
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

#: The cost each scripted call reports (a decimal string, as OpenRouter reports ``usage.cost``).
CALL_USD = "0.0001"
_IDS = itertools.count()


@dataclass
class Call:
    lane: str
    path: str
    body: dict
    meta: dict


@dataclass
class Lanes:
    """The lanes' scripts and every call they answered."""

    scripts: dict[str, list] = field(default_factory=dict)
    calls: list[Call] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    usd: str = CALL_USD
    embed_calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def script(self, lane: str, *replies: Any) -> None:
        with self._lock:
            self.scripts.setdefault(lane, []).extend(replies)

    def of(self, lane: str) -> list[Call]:
        return [c for c in self.calls if c.lane == lane]

    def booked_usd(self, lane: str | None = None) -> Decimal:
        return sum((Decimal(self.usd) for c in self.calls if lane is None or c.lane == lane), Decimal(0))

    def answer(self, lane: str, path: str, body: dict, meta: dict) -> dict | None:
        with self._lock:
            self.calls.append(Call(lane, path, body, meta))
            queue = self.scripts.get(lane) or []
            if not queue:
                self.unexpected.append(f"{lane}: no scripted reply left ({len(self.of(lane))} calls)")
                return None
            entry = queue.pop(0)
        reply = entry(body) if callable(entry) else entry
        if isinstance(reply, str):
            reply = {"role": "assistant", "content": reply}
        return {"role": "assistant", **reply}


EMBED_DIM = 16


def _vector(text: str) -> list[float]:
    """A fixed unit vector per text (word hashes folded into EMBED_DIM slots), so equal texts embed equally and texts
    that share words are near."""
    import hashlib
    import math

    v = [0.0] * EMBED_DIM
    for word in str(text).lower().split():
        h = hashlib.sha256(word.encode()).digest()
        v[h[0] % EMBED_DIM] += 1.0 + h[1] / 255.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _embeddings(body: dict) -> dict:
    texts = body.get("input")
    texts = [texts] if isinstance(texts, str) else list(texts or [])
    return {"object": "list", "data": [{"object": "embedding", "index": i, "embedding": _vector(t)}
                                       for i, t in enumerate(texts)],
            "model": str(body.get("model") or "fake/embed"), "usage": {"prompt_tokens": 1, "total_tokens": 1}}


def _completion(message: dict, usd: str, model: str) -> dict:
    finish = "tool_calls" if message.get("tool_calls") else "stop"
    return {
        "id": f"chatcmpl-fake-{next(_IDS)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": usd},
    }


def _stream(message: dict, usd: str, model: str) -> list[bytes]:
    cid = f"chatcmpl-fake-{next(_IDS)}"
    delta: dict[str, Any] = {"role": "assistant"}
    if message.get("content"):
        delta["content"] = message["content"]
    if message.get("tool_calls"):
        delta["tool_calls"] = [{**c, "index": i} for i, c in enumerate(message["tool_calls"])]
    finish = "tool_calls" if message.get("tool_calls") else "stop"
    # every chunk carries ``created``, as a provider's does (LiteLLM's OpenRouter stream reader requires it)
    head = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model}
    frames = [
        {**head, "choices": [{"index": 0, "delta": delta}]},
        {**head, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
        {**head, "choices": [],
         "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": usd}},
    ]
    return [f"data: {json.dumps(f)}\n\n".encode() for f in frames] + [b"data: [DONE]\n\n"]


def serve(lanes: Lanes) -> tuple[ThreadingHTTPServer, int]:
    """Serve *lanes* on 127.0.0.1 (a free port); returns the server (``shutdown()`` it) and its port. A route is
    ``/<lane>/v1/chat/completions``; the lane is the path's first part."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a: Any) -> None:  # quiet
            return

        def do_POST(self) -> None:  # noqa: N802 - the stdlib's name
            parts = self.path.strip("/").split("/")
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                self.send_error(400)
                return
            if parts and parts[-1] == "embeddings":
                data = json.dumps(_embeddings(body)).encode()
                lanes.embed_calls += 1
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if len(parts) < 3 or parts[-1] != "completions":
                self.send_error(404)
                return
            lane = parts[0]
            meta = {k.lower(): v for k, v in self.headers.items() if k.lower().startswith("x-unify-")}
            message = lanes.answer(lane, self.path, body, meta)
            if message is None:
                self.send_error(500, "no scripted reply")
                return
            model = str(body.get("model") or "fake/model")
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for frame in _stream(message, lanes.usd, model):
                    self.wfile.write(frame)
                self.wfile.flush()
                return
            data = json.dumps(_completion(message, lanes.usd, model)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, name="fake-lanes", daemon=True).start()
    return server, server.server_address[1]


def route(port: int, lane: str) -> str:
    """A lane's base URL in the form the build accepts."""
    return f"http://127.0.0.1:{port}/{lane}/v1"


def tool_call(name: str, arguments: dict, call_id: str | None = None) -> dict:
    return {
        "id": call_id or f"call_{next(_IDS)}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


Reply = Callable[[dict], Any]
