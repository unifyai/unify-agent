"""Shared helpers of the library export's generated files: the key of a function's recorded input shapes, the token
estimate the index budgets use, and the writer of generated files into a copy of the library.

The files themselves are :mod:`.library_export`'s; the writer never edits them: the gate refuses any commit that
touches a path the harness generates (:func:`.layout.misplaced`), and the writer's copy of the library never holds
them.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
from collections.abc import Callable
from pathlib import Path

# (item id, body digest) -> (input shapes, whether they were backfilled from covers), or None
ShapeLookup = Callable[[str, str], "tuple[list[dict], bool] | None"]


def body_digest(body: str) -> str:
    """The key of a function's recorded input shapes: SHA-256 of its body (:func:`.layout.function_bodies`)."""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def write_files(checkout: Path, files: dict[str, bytes]) -> dict[str, bytes]:
    """Write *files* (relative path -> bytes) into *checkout*, replacing whatever is there; returns them.
    Nothing is followed: an existing link or directory at a path, or at its parent, is removed first.
    """
    checkout = Path(checkout)
    for rel, data in files.items():
        dest = checkout / rel
        parent = dest.parent
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            parent.unlink()
        parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        elif dest.is_dir():
            shutil.rmtree(dest)
        fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
    return files


