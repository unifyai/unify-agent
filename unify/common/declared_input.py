"""The request's input as the world declared it: a value a world adapter sends beside a message, as data.

A driver may send, with a message, the input the request is about (``{"message": ..., "input": ...}`` on the
CLI's stdin): the CLI keeps it here, the latest declaration replacing the one before, for the rest of the process.
``tools.answer_from`` answers on it, and the memory harness tells a use on the request's real input from a use on any
other value by it. The harness never parses a message's text for it: a world that declares nothing has none.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

#: The most bytes of JSON a declared input may take (as a channel reply carries).
MAX_BYTES = 1 << 20

_CURRENT: dict | None = None


def canonical(value: Any) -> str:
    """The value's canonical JSON: sorted keys, separators ``(",", ":")``, non-ASCII kept, anything else as its ``str``
    (the same text as ``unify.memory.fixtures.canonical``)."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def digest(text: str) -> str:
    """The 16-hex-digit sha256 of a canonical JSON text (the library package's ``result_digest`` of its value)."""
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def declare(value: Any) -> bool:
    """Keep *value* as the declared input; False, and the one before kept, when it has no JSON form or its JSON is
    larger than :data:`MAX_BYTES`."""
    global _CURRENT
    try:
        text = canonical(value)
    except (TypeError, ValueError, RecursionError):
        return False
    if len(text.encode("utf-8", "surrogatepass")) > MAX_BYTES:
        return False
    _CURRENT = {"value": json.loads(text), "digest": digest(text)}
    return True


def current() -> dict | None:
    """The declared input, ``{"value", "digest"}``, or None when none was declared."""
    return dict(_CURRENT) if _CURRENT is not None else None


def clear() -> None:
    """Forget the declared input (tests, and a driver that starts over)."""
    global _CURRENT
    _CURRENT = None
