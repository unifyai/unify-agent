"""The box side of the tool channel: the code every generated ``tools`` module starts with.

Standard library only and self-contained (the box imports nothing from the harness). ``TOOLS_CHANNEL`` names two
inherited pipe descriptors, ``fd:<request>,<reply>``; it is read once, at the first call. Each call writes one frame
(a 4-byte big-endian length, then UTF-8 JSON ``{"v": 1, "id": <n>, "tool": <name>, "args": <object>}``) and reads
one reply frame. A lock serialises the calls of a cell's threads, so frames never interleave.

A call returns a :class:`Result` (a ``str``: the tool's reply text, with ``.paths`` and ``.next``) or raises:
:class:`Refused` (the text starts ``refused: ``, ``not finished: `` or ``not answered: ``), :class:`NotRun` (a call
that was not run and can be sent again; a subclass of :class:`Refused`), :class:`ToolError` (the tool failed) or
:class:`ChannelClosed` (the channel is gone; its reason code). Nothing here is trusted by the host.
"""

import json as _json
import os as _os
import threading as _threading

_MAX_REQUEST_BYTES = 1114112
_MAX_REPLY_BYTES = 1048576
_LOCK = _threading.Lock()
_STATE = {"fds": None, "id": 0, "closed": None}


class Result(str):
    """A tool's reply text; ``.paths`` lists the box paths it touched, ``.next`` a paged reply's cursor (or None)."""

    def __new__(cls, text, paths=(), next=None):
        obj = super().__new__(cls, text)
        obj.paths = list(paths or ())
        obj.next = next
        return obj


class Refused(Exception):
    """The host refused the call; ``str()`` is the refusal text, ``.form`` its stable code, ``.tool`` the tool."""

    def __init__(self, text, tool=None, form=None):
        super().__init__(text)
        self.text, self.tool, self.form = text, tool, form

    def __str__(self):
        return self.text


class NotRun(Refused):
    """The call was not run (a limit for now, or the time ran out); it can be made again later."""


class ToolError(Exception):
    """The tool failed on the host; ``str()`` says how."""

    def __init__(self, text, tool=None, form=None):
        super().__init__(text)
        self.text, self.tool, self.form = text, tool, form

    def __str__(self):
        return self.text


class ChannelClosed(Exception):
    """The tool channel is closed; ``str()`` is the reason code. The cell goes on without tools."""

    def __init__(self, reason, tool=None):
        super().__init__(reason)
        self.text, self.tool, self.form = reason, tool, reason

    def __str__(self):
        return self.text


def _fds():
    if _STATE["fds"] is None:
        value = _os.environ.get("TOOLS_CHANNEL", "")
        try:
            kind, _, rest = value.partition(":")
            req, rep = (int(x) for x in rest.split(","))
            if kind != "fd":
                raise ValueError(value)
        except ValueError:
            _STATE["closed"] = "no_channel"
            return None
        _STATE["fds"] = (req, rep)
    return _STATE["fds"]


def _write_all(fd, data):
    view = memoryview(data)
    while view:
        view = view[_os.write(fd, view):]


def _read_exact(fd, n):
    out = bytearray()
    while len(out) < n:
        chunk = _os.read(fd, n - len(out))
        if not chunk:
            return None
        out += chunk
    return bytes(out)


def _exchange(tool, args):
    """One request and its reply, under the lock: (reply object, None) or (None, the reason the channel closed)."""
    with _LOCK:
        if _STATE["closed"] is not None:
            return None, _STATE["closed"]
        fds = _fds()
        if fds is None:
            return None, _STATE["closed"]
        req, rep = fds
        _STATE["id"] += 1
        call_id = _STATE["id"]
        data = _json.dumps(
            {"v": 1, "id": call_id, "tool": tool, "args": args},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(data) > _MAX_REQUEST_BYTES:
            return {"status": "error", "text": "error: the arguments are larger than %d bytes" % _MAX_REQUEST_BYTES,
                    "form": "too_large"}, None
        try:
            _write_all(req, len(data).to_bytes(4, "big") + data)
            header = _read_exact(rep, 4)
            if header is None:
                _STATE["closed"] = "closed"
                return None, "closed"
            length = int.from_bytes(header, "big")
            if length > _MAX_REPLY_BYTES:
                _STATE["closed"] = "bad_reply"
                return None, "bad_reply"
            body = _read_exact(rep, length)
        except OSError:
            _STATE["closed"] = "closed"
            return None, "closed"
        if body is None:
            _STATE["closed"] = "closed"
            return None, "closed"
        try:
            reply = _json.loads(body.decode("utf-8"))
        except ValueError:
            reply = None
        if not isinstance(reply, dict):
            _STATE["closed"] = "bad_reply"
            return None, "bad_reply"
        if reply.get("status") == "closed":
            _STATE["closed"] = str(reply.get("form") or reply.get("text") or "closed")
            return None, _STATE["closed"]
        if reply.get("id") != call_id:
            _STATE["closed"] = "bad_reply"
            return None, "bad_reply"
        return reply, None


def _call(tool, args):
    """Call the host tool *tool* with the arguments object *args*."""
    try:
        reply, closed = _exchange(tool, args)
    except (TypeError, ValueError) as exc:
        raise ToolError("error: arguments are not JSON: %s" % exc, tool, "bad_arguments") from None
    if closed is not None:
        raise ChannelClosed(closed, tool) from None
    status = reply.get("status")
    text = str(reply.get("text") or "")
    form = reply.get("form")
    if status == "ok":
        return Result(text, reply.get("paths") or (), reply.get("next"))
    if status == "not_run":
        raise NotRun(text, tool, form) from None
    if status == "refused":
        raise Refused(text, tool, form) from None
    raise ToolError(text, tool, form) from None
