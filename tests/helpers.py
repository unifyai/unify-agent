"""Shared test helpers."""

import sys

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


