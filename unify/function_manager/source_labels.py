"""Filenames for compiled stored-function sources.

Every stored function is compiled under ``<function:NAME>`` so a Python stack
frame or traceback entry names the stored function it belongs to, and
``linecache`` can hand back the exact source that ran.
"""

from __future__ import annotations

import linecache
from types import CodeType

_PREFIX = "<function:"
_SUFFIX = ">"


def function_source_filename(name: str) -> str:
    return f"{_PREFIX}{name}{_SUFFIX}"


def compile_function_source(name: str, source: str) -> CodeType:
    """Compile ``source`` under the function's label and register it with ``linecache``.

    Registering the text lets ``traceback`` read the exact executed lines back
    from a frame, including sources that were rewritten (decorators stripped,
    steering probes inserted) before compiling.
    """
    filename = function_source_filename(name)
    lines = source.splitlines(keepends=True)
    linecache.cache[filename] = (len(source), None, lines, filename)
    return compile(source, filename, "exec")


class StoredSource:
    """A stored function bound in a session by its source, never executed here.

    With Python in the sandboxed worker (``UNIFY_WORKSPACE_PYTHON=worker``) a
    read that binds the functions it returns (``functions.get``, ``search``,
    ``filter``, ``list``) binds each as one of these: the worker defines the
    function from ``source`` and runs it there, confined
    (unify/actor/execution/worker.py). In this process, which holds the
    provider credentials, it only describes the function, and calling it
    raises.
    """

    def __init__(
        self,
        name: str,
        source: str,
        *,
        docstring: str = "",
        is_async: bool = False,
    ) -> None:
        self.__name__ = name
        self.__qualname__ = name
        self.__doc__ = docstring or None
        self.source = source
        self.filename = function_source_filename(name)
        self.is_async = is_async

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise RuntimeError(
            f"The stored function {self.__name__!r} runs only in the sandboxed "
            "worker, never in the harness's process: call it from a cell, or "
            "with functions.run",
        )

    def __repr__(self) -> str:
        return f"<stored function {self.__name__} (defined in the worker)>"
