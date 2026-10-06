"""``UNIFY_BIND_REQUEST=on``: the current request is ``request`` in a cell.

As shipped, model code sees the request only through the model: data the
request holds is typed into a cell again. With the switch on, a cell reads
the request itself:

* ``request.text`` is the requester's latest message as the model reads it:
  the request the loop started with, then each later requester message
  (the next request of a persistent session, or one sent while a request
  runs). The session context the harness opens the first message with
  (``first_message_context``: the clock, the library's size and shortlist)
  and the time prefix are not part of it; a loop-authored notice is not a
  requester message.
* ``request.data`` is the list of the JSON objects and arrays found in that
  text, in order of appearance, parsed with the standard json module
  (``worker_child.json_values``). Nothing else is parsed.

Pieces, in the order a request travels:

* The loop that answers a requester (the actor's task loop, which passes
  ``bind_request=True``) sets a :class:`RequestSlot` in its context and
  records each requester message in it (``loop.py``). Its handle makes the
  slot, so the loop restarted after context compression keeps the request.
  Any other loop (the storage review) sets ``None``, so a nested loop never
  sees its parent's request. Nothing is set while the switch is off.
* Before each cell the session executor calls :func:`install`, which binds a
  fresh ``worker_child.Request`` as the sandbox global ``request``, or
  removes it when the running loop has no request.
* Under ``UNIFY_WORKSPACE_PYTHON=worker`` the request's text crosses the
  boundary (``worker.py``) and the worker builds its own ``Request`` from it,
  a fresh one for each cell (``worker_child.py``).
"""

from __future__ import annotations

import contextvars
import dataclasses
from typing import Any, Optional

#: The sandbox global a cell reads the request from.
GLOBAL = "request"


def enabled() -> bool:
    """Whether ``UNIFY_BIND_REQUEST=on``."""
    from unify.settings import SETTINGS

    return getattr(SETTINGS, "UNIFY_BIND_REQUEST", "") == "on"


@dataclasses.dataclass
class RequestSlot:
    """One loop's current request text; ``None`` until the first arrives.

    A loop restarted after context compression is handed its predecessor's
    slot (``started``), so its first message, a loop-authored "continue",
    does not replace the request.
    """

    text: Optional[str] = None
    started: bool = False


_SLOT: contextvars.ContextVar[Optional[RequestSlot]] = contextvars.ContextVar(
    "unify_bound_request_slot",
    default=None,
)


def new_slot(bind_request: Any) -> Any:
    """What a loop handle passes its inner loop (and a restart of it) as
    ``bind_request``: a slot both share, or *bind_request* while off."""
    if enabled() and bind_request is True:
        return RequestSlot()
    return bind_request


def bind(bind_request: Any) -> Optional[contextvars.Token]:
    """Set the running loop's slot; ``None`` (nothing set) while the switch is off.

    *bind_request* is ``True`` (a new slot), a :class:`RequestSlot` (the
    handle's, shared with a restart) or false (no request).
    """
    if not enabled():
        return None
    if isinstance(bind_request, RequestSlot):
        return _SLOT.set(bind_request)
    return _SLOT.set(RequestSlot() if bind_request else None)


def unbind(token: Optional[contextvars.Token]) -> None:
    if token is not None:
        _SLOT.reset(token)


def current() -> Optional[RequestSlot]:
    return _SLOT.get()


def _content_text(content: Any) -> Optional[str]:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [
            block.get("text")
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        return "\n".join(texts)
    return None


def text_of(message: Any) -> Optional[str]:
    """The requester's text in *message*, as the loop is given it.

    A string is the text; a user message dict gives its content (text, or
    its text blocks joined by newlines); a list of content blocks is one
    message; a list of messages gives its last user message.
    """
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        if message.get("role", "user") != "user":
            return None
        return _content_text(message.get("content"))
    if isinstance(message, list):
        if all(isinstance(m, dict) and "role" not in m for m in message):
            return _content_text(message)
        for item in reversed(message):
            text = text_of(item)
            if text is not None:
                return text
    return None


def record_first(slot: Optional[RequestSlot], message: Any) -> None:
    """The message a loop starts with is its request, unless the loop is a
    restart after compression, which starts with a loop-authored message."""
    if slot is None or slot.started:
        return
    slot.started = True
    record(slot, message)


def record(slot: Optional[RequestSlot], message: Any) -> None:
    """Record *message* as the current request of the loop owning *slot*."""
    if slot is None:
        return
    text = text_of(message)
    if text is not None:
        slot.text = text


def install(namespace: dict) -> None:
    """Before a cell: bind a fresh ``request``, or remove one, in *namespace*."""
    if not enabled():
        return
    from unify.actor.execution.worker_child import Request

    slot = _SLOT.get()
    if slot is None or slot.text is None:
        if isinstance(namespace.get(GLOBAL), Request):
            del namespace[GLOBAL]
        return
    namespace[GLOBAL] = Request(slot.text)
