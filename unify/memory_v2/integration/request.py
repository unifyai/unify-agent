"""The memory run of the request in progress (one request per CLI process).

Integration Task 25 adds ``RequestRun``; until then this module only holds the current run, which the
harness hooks read. A run exposes ``index`` (the rendered memory index) and ``paths`` (``Paths``).
"""

from __future__ import annotations

from typing import Any

_CURRENT: Any = None


def current() -> Any:
    return _CURRENT
