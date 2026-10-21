"""Where the harness keeps its state: ``UNIFY_HOME``, else ``~/.unify``."""

from __future__ import annotations

import os
from pathlib import Path


def store_home() -> Path:
    """The state directory: memory, transcripts, records, the workspace and the logs live under it."""
    raw = os.environ.get("UNIFY_HOME", "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".unify"


__all__ = ["store_home"]
