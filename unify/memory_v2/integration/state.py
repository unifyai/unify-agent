"""Harness-side state between requests (``<UNIFY_HOME>/memory-v2``): generations, drift, suspect, and the request lock."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..fingerprint import Generations


@dataclass
class State:
    path: Path
    generations: Generations = field(default_factory=Generations)
    drift: set[str] = field(default_factory=set)
    suspect: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, path: Path) -> "State":
        path = Path(path)
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            return cls(path)
        return cls(
            path,
            Generations.from_json(data.get("generations") or {}),
            set(data.get("drift") or []),
            set(data.get("suspect") or []),
        )

    def save(self) -> None:
        data = {
            "generations": self.generations.to_json(),
            "drift": sorted(self.drift),
            "suspect": sorted(self.suspect),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".state-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, sort_keys=True, indent=1)
                f.write("\n")
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


def acquire_lock(path: Path, timeout_s: float = 60.0) -> int:
    """An exclusive ``flock`` on *path*: one memory request per ``UNIFY_HOME`` at a time."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            if time.monotonic() >= deadline:
                os.close(fd)
                raise TimeoutError(f"memory request lock {path} held for {timeout_s}s")
            time.sleep(0.05)


def release_lock(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
