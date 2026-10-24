"""The request's input and examples as the world declared them: values a world adapter sends beside a message, as data.

A driver may send, with a message, the input the request is about (``{"message": ..., "input": ...}`` on the
CLI's stdin): the CLI keeps it here, the latest declaration replacing the one before, for the rest of the process.
``tools.answer_from`` answers on it, and the memory harness tells a use on the request's real input from a use on any
other value by it. A driver may also send example pairs (``"examples": [{"input": ..., "output": ...}, ...]``): they
are kept in the order received, each distinct pair once, and ``tools.examples()`` returns them; ``tools.answer_from``
checks the function on them first. The harness never parses a message's text for either: a world that declares
nothing has none.
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


_EXAMPLES: list[dict] = []
_EXAMPLE_KEYS: set[str] = set()


def add_examples(pairs: Any) -> bool:
    """Keep each pair of *pairs* (``{"input": ..., "output": ...}``) not already kept, in order; False, and nothing
    kept, when *pairs* is not a list of such pairs with JSON forms, or when all the pairs together would take more than
    :data:`MAX_BYTES` of JSON."""
    if not isinstance(pairs, list) or not all(isinstance(p, dict) and set(p) == {"input", "output"} for p in pairs):
        return False
    try:
        texts = [canonical(p) for p in pairs]
    except (TypeError, ValueError, RecursionError):
        return False
    new = [(t, json.loads(t)) for t in dict.fromkeys(texts) if t not in _EXAMPLE_KEYS]
    size = sum(len(k.encode("utf-8", "surrogatepass")) for k in _EXAMPLE_KEYS)
    if size + sum(len(t.encode("utf-8", "surrogatepass")) for t, _ in new) > MAX_BYTES:
        return False
    for text, pair in new:
        _EXAMPLE_KEYS.add(text)
        _EXAMPLES.append(pair)
    return True


def examples() -> list[dict]:
    """The example pairs declared so far, in the order received ([] when none)."""
    return [dict(p) for p in _EXAMPLES]


def current() -> dict | None:
    """The declared input, ``{"value", "digest"}``, or None when none was declared."""
    return dict(_CURRENT) if _CURRENT is not None else None


def clear() -> None:
    """Forget the declared input and examples (tests, and a driver that starts over)."""
    global _CURRENT
    _CURRENT = None
    _EXAMPLES.clear()
    _EXAMPLE_KEYS.clear()
