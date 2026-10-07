"""API worlds: record every call, replay without the live server, keep secrets out of memory.

The workspace reaches an API world (AppWorld's relay, or any JSON-lines service of the same
shape) only through one Unix socket that the harness binds into the sandbox: a *gateway* the
harness runs outside the sandbox. Wire format (AppWorld relay's, docs/appworld-adapter.md):
    {"op": "describe"} -> {"ok": true, "apps": {app: {api: {"method": "GET", ...}}}}
    {"op": "call", "app", "api", "kwargs"} -> {"ok": true, "status": "ok", "response": ...}
                                           | {"ok": true, "status": "exception", "exception": {...}}
                                           | {"ok": false, "error": {"type", "message"}}

Gateway modes:
  live      forwards to the upstream socket and records every call: request, answer, the cell
            that made it, and whether it changes state (any method but GET/HEAD).
  readonly  forwards only calls that do not change state; refuses the rest; the completion call
            is captured, never forwarded. Used to pre-run a procedure for an offer.
  replay    answers only from a recording; it has no upstream at all, so nothing can reach the
            live world. Calls not in the recording are refused.

Secrets. Values under keys that look like credentials (password, token, secret, api_key, ...),
in requests or responses, are replaced in recordings by labels (<secret:1>, ...), consistently,
so a replay that passes a label back gets the recorded answer. A stored procedure never holds a
secret: a literal secret in the captured code is replaced by a credential parameter whose value
is fetched at call time by a recipe, the recorded call that first returned it plus the path to
it, for example
    cred1 = _cred(apis.supervisor.show_account_passwords(), [{"match": {"account_name": "spotify"}}, "password"])
Live, the recipe fetches today's value; in replay it gets the label back from the recording.

A call whose response carried a new secret (a login) is marked `issues_credentials`: it is not an
effect on the world's task state, so it is left out of a procedure's effects, a replay may answer
it more than once, and it alone does not make a procedure "state-changing".
"""
from __future__ import annotations

import ast
import copy
import json
import os
import re
import socket
import socketserver
import tempfile
import threading
from pathlib import Path

SECRET_KEY = re.compile(r"pass(word)?|token|secret|api_?key|otp|cvv|pin_?code", re.I)
READ_METHODS = {"GET", "HEAD"}
COMPLETION = ("supervisor", "complete_task")
INSIDE_DIR = "/run/world"
INSIDE_SOCKET = f"{INSIDE_DIR}/gw.sock"

# The client the workspace gets: AppWorld's calling convention, standard library only.
CLIENT = r'''
import json as _json, os as _os, socket as _socket
class _WorldError(Exception):
    pass
class _Conn:
    def __init__(self):
        self.s = None
    def ask(self, payload):
        if self.s is None:
            self.s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
            self.s.connect(_os.environ["WORLD_SOCKET"])
            self.f = self.s.makefile("rb")
        self.s.sendall((_json.dumps(payload) + "\n").encode())
        line = self.f.readline()
        if not line:
            self.s = None
            raise _WorldError("the API gateway closed the connection")
        return _json.loads(line)
_conn = _Conn()
class _Api:
    def __init__(self, app, api):
        self._app, self._api = app, api
    def __call__(self, *args, **kwargs):
        if args:
            raise TypeError("API calls take keyword arguments only")
        a = _conn.ask({"op": "call", "app": self._app, "api": self._api, "kwargs": kwargs})
        if not a.get("ok"):
            raise _WorldError((a.get("error") or {}).get("message"))
        if a.get("status") == "ok":
            return a.get("response")
        e = a.get("exception") or {}
        raise Exception(f"{e.get('type', 'Exception')}: {e.get('message')}")
    def __repr__(self):
        return f"<API {self._app}.{self._api}>"
class _App:
    def __init__(self, app):
        self._app = app
    def __getattr__(self, api):
        if api.startswith("_"):
            raise AttributeError(api)
        return _Api(self._app, api)
class _Apis:
    def __getattr__(self, app):
        if app.startswith("_"):
            raise AttributeError(app)
        return _App(app)
    def __getitem__(self, app):
        return _App(app)
apis = _Apis()
def _cred(value, path):
    for step in path:
        if isinstance(step, dict):
            value = next(v for v in value if all(v.get(k) == x for k, x in step["match"].items()))
        else:
            value = value[step]
    return value
__hidden__ = {"apis", "_cred"}
'''


class SecretBook:
    """Known secret values -> labels, and where each was first returned (the recipe source)."""

    def __init__(self):
        self.labels: dict[str, str] = {}
        self.sources: dict[str, dict] = {}

    def _add(self, value, source=None):
        if isinstance(value, str) and len(value) >= 4 and value not in self.labels:
            self.labels[value] = f"<secret:{len(self.labels) + 1}>"
            if source:
                self.sources[value] = source

    def scan_request(self, kwargs: dict):
        for k, v in kwargs.items():
            if SECRET_KEY.search(str(k)):
                self._add(v)

    def scan_response(self, response, call: dict):
        def walk(node, path, parent_list=None):
            if isinstance(node, dict):
                for k, v in node.items():
                    if isinstance(v, str) and SECRET_KEY.search(str(k)):
                        self._add(v, {**call, "path": path + [k]})
                    else:
                        walk(v, path + [k])
            elif isinstance(node, list):
                for i, item in enumerate(node):
                    walk(item, path + [_selector(node, i)])
        walk(response, [])

    def redact(self, value):
        if isinstance(value, str):
            for secret, label in sorted(self.labels.items(), key=lambda kv: -len(kv[0])):
                value = value.replace(secret, label)
            return value
        if isinstance(value, list):
            return [self.redact(v) for v in value]
        if isinstance(value, dict):
            return {k: self.redact(v) for k, v in value.items()}
        return value


def _selector(items: list, i: int):
    """A stable way to find items[i] again: match on a string field whose value is unique."""
    item = items[i]
    if isinstance(item, dict):
        for k, v in item.items():
            if isinstance(v, str) and not SECRET_KEY.search(k) and \
                    sum(isinstance(o, dict) and o.get(k) == v for o in items) == 1:
                return {"match": {k: v}}
    return i


class Gateway:
    """A JSON-lines Unix-socket server outside the sandbox. One per workspace session."""

    def __init__(self, mode: str, upstream: str | None = None, recording: dict | None = None,
                 on_complete=None):
        assert mode in ("live", "readonly", "replay")
        if mode == "replay":
            assert upstream is None, "a replay gateway has no upstream"
            assert recording is not None
        else:
            assert upstream
        self.mode, self._upstream_path = mode, upstream
        self.recording = recording
        self.on_complete = on_complete  # live mode: may hold the completion call (returns a message) or None
        self.book = SecretBook()
        self.calls: list[dict] = []  # this session's calls, redacted
        self.completion: dict | None = None
        self.cell = 0
        self.dir = tempfile.mkdtemp(prefix="cs-world-")
        os.chmod(self.dir, 0o700)
        self.path = os.path.join(self.dir, "gw.sock")
        self._up = None
        self._lock = threading.Lock()
        self._describe = None
        self._replay_queues: dict[str, list] = {}
        if mode == "replay":
            for c in recording["calls"]:
                self._replay_queues.setdefault(_key(c["app"], c["api"], c["kwargs"]), []).append(c)
        gw = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                for line in self.rfile:
                    try:
                        answer = gw.handle(json.loads(line))
                    except Exception as exc:  # never let the workspace crash the gateway
                        answer = _refusal(f"gateway error: {type(exc).__name__}")
                    self.wfile.write((json.dumps(answer) + "\n").encode())
                    self.wfile.flush()

        class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
            daemon_threads = True

        self.server = Server(self.path, Handler)
        os.chmod(self.path, 0o600)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    # ---------------------------------------------------------------- upstream (live, readonly)

    def _upstream(self, payload: dict) -> dict:
        with self._lock:
            if self._up is None:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(self._upstream_path)
                self._up = (s, s.makefile("rb"))
            s, f = self._up
            s.sendall((json.dumps(payload) + "\n").encode())
            line = f.readline()
            if not line:
                self._up = None
                return _refusal("the API world closed the connection")
            return json.loads(line)

    def describe(self) -> dict:
        if self._describe is None:
            if self.mode == "replay":
                self._describe = {"ok": True, "apps": self.recording["describe"]}
            else:
                self._describe = self._upstream({"op": "describe"})
        return self._describe

    def method(self, app: str, api: str) -> str:
        return (((self.describe().get("apps") or {}).get(app) or {}).get(api) or {}).get("method", "POST").upper()

    # ---------------------------------------------------------------- calls

    def handle(self, req: dict) -> dict:
        op = req.get("op")
        if op == "describe":
            return self.describe()
        if op != "call":
            return _refusal(f"unknown operation {op!r}")
        app, api, kwargs = req.get("app"), req.get("api"), req.get("kwargs") or {}
        if not isinstance(app, str) or not isinstance(api, str) or not isinstance(kwargs, dict):
            return _refusal("a call needs app, api and kwargs")
        changes = self.method(app, api) not in READ_METHODS
        completion = (app, api) == COMPLETION
        if self.mode == "replay":
            queue = self._replay_queues.get(_key(app, api, kwargs))
            if not queue:
                return _refusal(f"replay: {app}.{api} with these arguments is not in the recording")
            # a credential-issuing call (a login) may be repeated, as a live world would allow; any other
            # state-changing call is answered once per recorded occurrence, in order
            once = changes and not queue[0].get("issues_credentials")
            rec = queue.pop(0) if (once or len(queue) > 1) else queue[0]
            answer = copy.deepcopy(rec["answer"])
            self._note(app, api, kwargs, answer, changes, completion, redacted=True)
            return answer
        if completion and self.mode == "readonly":
            self._note(app, api, kwargs, {"ok": True, "status": "ok", "response": None}, changes, completion)
            return {"ok": True, "status": "ok", "response": {"message": "captured by the harness (not sent)"}}
        if changes and self.mode == "readonly":
            return _refusal(f"{app}.{api} changes state; this gateway forwards only read-only calls")
        if completion and self.on_complete:
            held = self.on_complete(kwargs)
            if held:
                return {"ok": True, "status": "exception",
                        "exception": {"type": "Exception", "message": held}}
        self.book.scan_request(kwargs)
        answer = self._upstream({"op": "call", "app": app, "api": api, "kwargs": kwargs})
        known = len(self.book.labels)
        if answer.get("ok") and answer.get("status") == "ok":
            self.book.scan_response(answer.get("response"), {"app": app, "api": api,
                                                             "kwargs": self.book.redact(kwargs)})
        self._note(app, api, kwargs, answer, changes, completion, issues=len(self.book.labels) > known)
        return answer

    def _note(self, app, api, kwargs, answer, changes, completion, redacted=False, issues=None):
        r = (lambda v: v) if redacted else self.book.redact
        if issues is None:  # replay: as recorded
            issues = any(c.get("issues_credentials") for c in self.recording["calls"]
                         if c["app"] == app and c["api"] == api) if self.recording else False
        row = {"i": len(self.calls), "cell": self.cell, "app": app, "api": api, "method": self.method(app, api),
               "changes_state": changes, "issues_credentials": issues, "kwargs": r(kwargs), "answer": r(answer)}
        self.calls.append(row)
        if completion and answer.get("ok") and answer.get("status") == "ok":
            self.completion = {"answer": r(kwargs.get("answer")), "status": kwargs.get("status", "success"),
                               "cell": self.cell}

    # ---------------------------------------------------------------- results

    def record(self) -> dict:
        """The redacted recording of this session (what a stored case keeps)."""
        return {"describe": self.describe().get("apps") or {}, "calls": copy.deepcopy(self.calls)}

    def effects(self) -> dict:
        """What a session did to the world: its state-changing calls in order, and the completion answer.
        This is the 'answer' a world procedure must reproduce in replay."""
        return {"__effects__": [[c["app"], c["api"], c["kwargs"]] for c in self.calls
                                if c["changes_state"] and (c["app"], c["api"]) != COMPLETION
                                and not c.get("issues_credentials") and c["answer"].get("status") == "ok"],
                "__answer__": (self.completion or {}).get("answer")}

    def changing_cells(self) -> set[int]:
        return {c["cell"] for c in self.calls if c["changes_state"] and not c.get("issues_credentials")
                and c["answer"].get("status") == "ok"}

    def response_text(self) -> str:
        """Every response this session saw, as text (the analogue of 'files the code read')."""
        return "\n".join(json.dumps(c["answer"].get("response"), default=str) for c in self.calls)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        if self._up:
            self._up[0].close()
        try:
            os.unlink(self.path)
        except OSError:
            pass
        os.rmdir(self.dir)


def _key(app, api, kwargs) -> str:
    return json.dumps([app, api, kwargs], sort_keys=True, default=str)


def _refusal(message: str) -> dict:
    return {"ok": False, "error": {"type": "WorldError", "message": message}}


# ---------------------------------------------------------------- secrets out of stored code


def credential_params(code: str, book: SecretBook) -> tuple[str, list[str]]:
    """Replace every string literal holding a known secret by a credential variable and prepend the
    recipes that fetch them at call time. Returns (code, recipe lines). Raises ValueError when a
    secret has no recipe (it came from the request or nowhere we saw): such code is not storable."""
    names: dict[str, str] = {}
    lines: list[str] = []

    def var_for(secret: str) -> str:
        if secret in names:
            return names[secret]
        src = book.sources.get(secret)
        if src is None:
            raise ValueError("a secret in the code has no recorded source; refusing to store it")
        args = []
        for k, v in src["kwargs"].items():
            raw = next((s for s, lab in book.labels.items() if lab == v), None)
            args.append(f"{k}={var_for(raw) if raw is not None else repr(v)}")
        name = f"cred{len(names) + 1}"
        names[secret] = name
        lines.append(f"{name} = _cred(apis.{src['app']}.{src['api']}({', '.join(args)}), {src['path']!r})")
        return name

    class Sub(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in book.labels:
                return ast.copy_location(ast.Name(var_for(node.value), ast.Load()), node)
            return node

    tree = Sub().visit(ast.parse(code))
    body = ast.unparse(ast.fix_missing_locations(tree))
    return ("\n".join(lines + [body]) if lines else body), lines


def no_secrets_in(text: str, book: SecretBook) -> bool:
    return not any(s in text for s in book.labels)


def write_recording(path: Path, recording: dict):
    path.write_text(json.dumps(recording, indent=1))
