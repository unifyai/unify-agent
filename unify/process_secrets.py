"""Credentials this process holds outside its environment, for the value-based redactors.

A credential removed from ``os.environ`` once read (so that no subprocess inherits it, as memory v2's Sol
token is) also drops out of every redactor that finds secrets by scanning the environment. It is registered
here instead, and those redactors (``unify.transcripts.scrub``, ``unify.memory_v2.redact``) consult this as
well, for the life of the process. Values are never removed or shown. Standard library only, so any module
may import it.
"""

from __future__ import annotations

import threading
from typing import Any

_SECRETS: dict[str, str] = {}  # value -> label
_LOCK = threading.Lock()


def register_secret(label: str, value: Any) -> None:
    """Keep *value* (and its stripped form) redacted by value for the life of this process; empty is ignored."""
    raw = str(value or "")
    with _LOCK:
        for v in (raw, raw.strip()):
            if v:
                _SECRETS.setdefault(v, label)


def registered_secrets() -> list[tuple[str, str]]:
    """``(label, value)`` for every registered credential, longest first."""
    with _LOCK:
        pairs = [(label, value) for value, label in _SECRETS.items()]
    return sorted(pairs, key=lambda kv: -len(kv[1]))
