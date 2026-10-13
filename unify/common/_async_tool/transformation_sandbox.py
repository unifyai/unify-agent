"""Context compression's transformations, run in a box of their own.

The compression model edits transcript entries by writing Python
(``update(index, transformation)`` in context_compression.py). The
transcript holds untrusted content (tool output, files, web pages) that can
steer that model, so its code never runs in the harness: each
transformation runs in a fresh ``python -I`` process
(transformation_child.py, passed as ``-c`` text) in the bubblewrap box that
runs the memory agents' code (:func:`unify.memory.sandbox_run.start_confined`):

* a read-only root of the system directories, a few files of ``/etc`` and the
  interpreter's own install, nothing else of the host (no home directory, no
  checkout, no state directory, no transcripts), with private ``/tmp`` and
  ``/home`` that vanish with the box; no paths are bound for it;
* no network (``--unshare-all``) and a seccomp filter;
* a cleared environment: only the box's ``PATH``, ``HOME`` and
  ``PYTHONDONTWRITEBYTECODE``, nothing of the harness's;
* the box's resource limits, a lower address-space limit
  (:data:`MEMORY_BYTES`), a bound on the reply's size, and a time limit
  (:data:`TIMEOUT_S`) after which every process of the box is killed and
  verified gone; the box is killed and verified gone after every run.

The same globals as before (``create_base_globals()`` plus ``re`` and
``json``) are rebuilt in the child by import, so a transformation's result,
and an error's type name and message, are what in-process ``exec`` gave.
Code, content and result cross as JSON over the box's pipes. Without
bubblewrap the transformation is refused, never run here; the ``update`` tool
reports that, as it reports any failed transformation.
"""

from __future__ import annotations

import builtins
import json
import os
import selectors
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Optional

_CHILD = Path(__file__).with_name("transformation_child.py")

TIMEOUT_S = 10.0
"""Wall-clock limit of one transformation, the box's start included."""

MAX_REPLY_BYTES = 32 * 1024 * 1024
"""The reply may be this large, or twice the request, whichever is larger."""

MEMORY_BYTES = 2 * 1024 * 1024 * 1024
"""The child's address-space limit (``RLIMIT_AS``), below the box's own."""

_BOX_CWD = "/tmp"


def _spec(name: str, value: Any) -> Optional[list]:
    """How the child imports the global *name* = *value*, if it can."""
    if isinstance(value, types.ModuleType):
        return ["module", value.__name__]
    module = sys.modules.get(getattr(value, "__module__", None) or "")
    if module is not None and getattr(module, name, None) is value:
        return ["attr", module.__name__, name]
    return None


def _globals_spec(base_globals: dict) -> tuple[list[str], dict[str, list]]:
    names = sorted((base_globals.get("__builtins__") or {}).keys())
    specs = {}
    for name, value in base_globals.items():
        if name == "__builtins__":
            continue
        spec = _spec(name, value)
        if spec is not None:
            specs[name] = spec
    return names, specs


def _error(name: str, message: str) -> Exception:
    """The child's exception, as *name*: *message* (a builtin type when it is one). The name comes from the box, so
    anything but a plain identifier of at most 100 characters is shown as ``Exception``."""
    if not isinstance(name, str) or len(name) > 100 or not name.isidentifier():
        name = "Exception"
    base = getattr(builtins, name, None)
    if not (isinstance(base, type) and issubclass(base, Exception)):
        base = Exception

    def __init__(self, text: str) -> None:
        Exception.__init__(self, text)

    def __str__(self) -> str:
        return self.args[0]

    cls = type(name, (base,), {"__init__": __init__, "__str__": __str__})
    return cls(message)


def _well_formed(reply: Any) -> bool:
    if not isinstance(reply, dict):
        return False
    if "error" in reply:
        error = reply["error"]
        return (
            isinstance(error, list)
            and len(error) == 2
            and all(isinstance(e, str) for e in error)
            and error[0].isidentifier()
        )
    return isinstance(reply.get("x"), str)


def run_transformation(code: str, content: str, base_globals: dict) -> str:
    """*content* after the transformation *code*, run in a fresh box.

    Raises what the transformation raised (same type name and message),
    ``TimeoutError`` past :data:`TIMEOUT_S`, ``RuntimeError`` without
    bubblewrap or when the box gives no usable reply.
    """
    from unify import sandbox
    from unify.memory import sandbox_run

    names, specs = _globals_spec(base_globals)
    request = json.dumps(
        {
            "code": code,
            "x": content,
            "builtins": names,
            "globals": specs,
            "memory_bytes": MEMORY_BYTES,
        },
    ).encode("ascii")
    limit = max(MAX_REPLY_BYTES, 2 * len(request))
    argv = [str(sandbox_run.PYTHON), "-I", "-c", _CHILD.read_text(encoding="utf-8")]
    # The command is a box already: a cell's subprocess confinement, when
    # this runs inside one, must not wrap it a second time.
    with sandbox.unconfined():
        box = sandbox_run.start_confined(argv, cwd=_BOX_CWD)
    try:
        out = _exchange(box, request, limit)
    finally:
        # Every run ends with the box's verified kill (raises if anything
        # survives), the reply read or not.
        box.kill()
    try:
        reply = json.loads(out)
        if not _well_formed(reply):
            raise ValueError(reply)
    except ValueError:
        tail = box.stderr_tail(1.0).strip()[-2000:]
        raise RuntimeError(
            f"the transformation's box gave no result (exit code "
            f"{box.proc.returncode})" + (f": {tail}" if tail else ""),
        ) from None
    if "error" in reply:
        raise _error(*reply["error"])
    return reply["x"]


def _exchange(box: Any, request: bytes, limit: int) -> bytes:
    """Send *request* and read the reply until the box closes its stdout.

    Raises ``TimeoutError`` past :data:`TIMEOUT_S` and ``RuntimeError`` past
    *limit* bytes; the caller kills the box either way.
    """

    def write() -> None:
        try:
            box.stdin.write(request)
            box.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass

    writer = threading.Thread(
        target=write,
        name="compression-transformation-writer",
        daemon=True,
    )
    writer.start()
    deadline = time.monotonic() + TIMEOUT_S
    out = bytearray()
    fd = box.stdout.fileno()
    with selectors.DefaultSelector() as sel:
        sel.register(fd, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"the transformation ran longer than {TIMEOUT_S:g} s and "
                    "was stopped",
                )
            if not sel.select(remaining):
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            out += chunk
            if len(out) > limit:
                raise RuntimeError(
                    f"the transformation's result is larger than {limit} bytes "
                    "and was discarded",
                )
    writer.join(timeout=1.0)
    return bytes(out)
