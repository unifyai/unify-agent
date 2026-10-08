"""Recording the worker's calls of stored functions off the core surface.

Under worker Python the worker runs its own copy of a stored function, which
otherwise only notes that it was used. While an ``execute_code`` call runs
off the core surface (which records through its own ``functions`` library),
the worker asks the harness to record each call by name (helpers, and
functions a read bound) as ``functions.run``'s helpers are recorded under
core, through a recorder that serves recording only (:func:`recording`).
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Any, Iterator

_RECORDER: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "unify_function_helpers_recorder",
    default=None,
)


def _call_recorder(actor: Any) -> Any:
    from unify.actor.core_surface import FunctionLibrary, WritePolicy

    class _CallRecorder(FunctionLibrary):
        """Records the calls of stored functions the worker runs, and nothing
        else: ``functions.run`` exists only under ``UNIFY_TOOL_SURFACE=core``."""

        async def _begin(self, *, name: str, mode: str, args: Any, kwargs: Any) -> Any:
            if mode != "call":
                return {"found": False}
            return await super()._begin(name=name, mode=mode, args=args, kwargs=kwargs)

    return _CallRecorder(actor, WritePolicy(can_store=False))


@contextlib.contextmanager
def recording(actor: Any) -> Iterator[None]:
    """Unless the core surface records through its own ``functions``
    library, the worker's calls of stored
    functions during the block are recorded through ``actor``'s function
    manager (:func:`recorder`)."""
    from unify.actor import core_surface

    if core_surface.enabled() or getattr(actor, "function_manager", None) is None:
        yield
        return
    token = _RECORDER.set(_call_recorder(actor))
    try:
        yield
    finally:
        _RECORDER.reset(token)


def recorder() -> Any:
    """The recorder :func:`recording` set for this call, if any."""
    return _RECORDER.get()
