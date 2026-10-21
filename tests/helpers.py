"""Shared test helpers."""

import functools
import inspect
import sys
from typing import Callable

# Cross-platform non-blocking file locking (tests/conftest.py ``unify_home``).
if sys.platform == "win32":
    import msvcrt

    def _lock_file_nb(file_obj):
        """Acquire an exclusive non-blocking lock on the file (Windows)."""
        msvcrt.locking(file_obj.fileno(), msvcrt.LK_NBLCK, 1)

else:
    import fcntl

    def _lock_file_nb(file_obj):
        """Acquire an exclusive non-blocking lock on the file (Unix)."""
        fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _handle_project(
    test_fn: Callable | None = None,
    *,
    try_reuse_prev_ctx: bool = False,
    delete_ctx_on_exit: bool = False,
):
    """No-op kept while the memory and agent tests still decorate with it (merge-time; see CLEAN-CORE-LOG.md)."""
    if test_fn is None:
        return lambda f: _handle_project(f)
    if inspect.iscoroutinefunction(test_fn):

        @functools.wraps(test_fn)
        async def wrapper(*args, **kwargs):
            return await test_fn(*args, **kwargs)

    else:

        @functools.wraps(test_fn)
        def wrapper(*args, **kwargs):
            return test_fn(*args, **kwargs)

    return wrapper
