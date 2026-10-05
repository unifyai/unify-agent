"""``UNIFY_FUNCTION_HELPERS``: a stored entry point runs with its stored helpers.

The storage doctrine composes functions: an entry point that does a whole
job by calling smaller stored functions. ``execute_function`` runs a stored
function by prepending its source to the call, so in a session that has not
read it (a search, filter or list injects a function's ``depends_on``) the
first helper it calls is a ``NameError``, in process and under worker
Python alike. With the switch on, ``execute_function`` defines the stored
functions the entry point calls, transitively and each once (the resolution
``functions.run`` uses under ``UNIFY_TOOL_SURFACE=core``,
:func:`unify.actor.core_surface.stored_helpers`), in the namespace the call
runs in, each wrapped as a read wraps it, and installs their declared
dependencies with the entry point's. A helper the library no longer holds is
left out and named in the error the call then raises.

Recording matches the core path. In process a helper is the function
manager's boundary wrapper, which records usage, trust and a case. Under
worker Python the worker runs its own copy of a stored function, which
otherwise only notes that it was used; with the switch on the worker asks
the harness to record each call by name (helpers, and functions a read
bound) as ``functions.run``'s helpers are recorded under core, through a
recorder that serves recording only (:func:`recording`).

Off: nothing here runs, and ``execute_function`` defines only the entry
point, as shipped.
"""

from __future__ import annotations

import contextlib
import contextvars
import re
from typing import Any, Dict, Iterator, List, Optional

_RECORDER: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "unify_function_helpers_recorder",
    default=None,
)


def enabled() -> bool:
    """Whether ``UNIFY_FUNCTION_HELPERS`` is on."""
    from unify.settings import SETTINGS

    return bool(getattr(SETTINGS, "UNIFY_FUNCTION_HELPERS", False))


class Helpers:
    """The stored functions an entry point calls, and how to define them."""

    def __init__(self, function_manager: Any, func_data: Dict[str, Any]) -> None:
        from unify.actor import core_surface

        self._fm = function_manager
        self.entry = str(func_data.get("name"))
        self.missing: List[str] = []
        self.helpers = core_surface.stored_helpers(
            func_data,
            lambda name: core_surface.lookup_stored(function_manager, name),
            self.missing,
        )
        #: The entry point's declared dependencies and its helpers', each once.
        self.dependencies: List[str] = list(
            dict.fromkeys(
                spec
                for data in (func_data, *self.helpers)
                for spec in (data.get("dependencies") or [])
            ),
        )

    def define(self, namespace: Dict[str, Any]) -> None:
        """Define the helpers in ``namespace``, the one the call runs in, as a
        read that loads the entry point injects them: each executed there,
        then replaced by the function manager's boundary wrapper, so a call
        is recorded. All are defined before any is wrapped, so they find one
        another (and the entry point) by name."""
        fm = self._fm
        for data in self.helpers:
            fm._create_in_process_callable(data, namespace=namespace)
        for data in self.helpers:
            raw = namespace.get(data["name"])
            if callable(raw):
                namespace[data["name"]] = fm._boundary(raw, data)

    def missing_note(self, error: Any) -> Optional[str]:
        """A note naming the helpers the library no longer holds, when the
        call's ``error`` is the ``NameError`` one of them caused."""
        text = str(error or "")
        gone = [
            name
            for name in self.missing
            if re.search(rf"NameError: name '{re.escape(name)}' is not defined", text)
        ]
        if not gone:
            return None
        names = ", ".join(f"`{name}`" for name in gone)
        return (
            f"`{self.entry}` calls the stored function{'s' if len(gone) > 1 else ''} "
            f"{names}, which the function library no longer holds (deleted or "
            f"renamed): store {'them' if len(gone) > 1 else 'it'} again, or "
            f"change `{self.entry}`."
        )


def plan(function_manager: Any, func_data: Any) -> Optional[Helpers]:
    """The helpers of the stored function ``func_data``; ``None`` while off or
    when there is no stored source to run."""
    if not enabled() or function_manager is None or not isinstance(func_data, dict):
        return None
    impl = func_data.get("implementation")
    if func_data.get("is_primitive") or not (isinstance(impl, str) and impl.strip()):
        return None
    return Helpers(function_manager, func_data)


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
    """While the switch is on (and the core surface, which records through its
    own ``functions`` library, is not), the worker's calls of stored
    functions during the block are recorded through ``actor``'s function
    manager (:func:`recorder`)."""
    from unify.actor import core_surface

    if (
        not enabled()
        or core_surface.enabled()
        or getattr(actor, "function_manager", None) is None
    ):
        yield
        return
    token = _RECORDER.set(_call_recorder(actor))
    try:
        yield
    finally:
        _RECORDER.reset(token)


def recorder() -> Any:
    """The recorder :func:`recording` set for this call, if any."""
    return _RECORDER.get() if enabled() else None
