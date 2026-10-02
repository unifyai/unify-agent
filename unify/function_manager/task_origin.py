"""Which task a stored function came from (``UNIFY_TRY_FIRST``).

A top-level ``act()`` keys its task by a short hash of its request (the
session's first user message, whitespace collapsed); the task loop and its
storage review inherit the key through the task context. With the switch on, a
function stored during that task records the key in its ``metadata`` under
``origin_tasks``, and a search from a later task whose request has the same
key marks the function ``same_task: true``. Sub-agents inherit the key of the
task they work for. The key is only a hash: the request text is not stored.
With the switch off nothing is recorded or marked.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import re
from typing import Any, Dict, Optional

FIELD = "origin_tasks"

_WS = re.compile(r"\s+")

_CURRENT: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "unify_task_origin",
    default=None,
)


def enabled() -> bool:
    from unify.settings import SETTINGS

    return bool(getattr(SETTINGS, "UNIFY_TRY_FIRST", False))


def task_key(request: Any) -> Optional[str]:
    """The key of a request: 16 hex digits of the sha256 of its normalised text."""
    if request is None:
        return None
    if isinstance(request, str):
        text = request
    else:
        text = json.dumps(request, sort_keys=True, default=str)
    text = _WS.sub(" ", text).strip()
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def enter(request: Any) -> Optional[contextvars.Token]:
    """Key the current context's task by *request*, unless it is keyed already.

    A sub-agent is started inside the task it works for, whose key it
    inherits and keeps. The caller resets the token with :func:`leave` in
    the same context once the task's handle is built.
    """
    if not enabled() or _CURRENT.get() is not None:
        return None
    key = task_key(request)
    if key is None:
        return None
    return _CURRENT.set(key)


def leave(token: Optional[contextvars.Token]) -> None:
    if token is None:
        return
    try:
        _CURRENT.reset(token)
    except ValueError:
        # Reset from another context (a handle cleaned up elsewhere).
        pass


def current() -> Optional[str]:
    return _CURRENT.get()


def stamped(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """*metadata* with the current task added to ``origin_tasks``, or ``None`` when off."""
    key = current()
    if not enabled() or key is None:
        return None
    out = dict(metadata or {})
    tasks = [t for t in (out.get(FIELD) or []) if isinstance(t, str)]
    if key not in tasks:
        tasks.append(key)
    out[FIELD] = tasks
    return out


def annotate(row: Dict[str, Any]) -> None:
    """Replace a search row's ``origin_tasks`` with ``same_task: true`` when it matches.

    The hashes are never shown; a row stored with the switch on and read
    with it off just loses them.
    """
    metadata = row.get("metadata")
    if not isinstance(metadata, dict) or FIELD not in metadata:
        return
    tasks = metadata.get(FIELD) or []
    metadata = {k: v for k, v in metadata.items() if k != FIELD}
    row["metadata"] = metadata
    key = current()
    if enabled() and key is not None and key in tasks:
        row["same_task"] = True
