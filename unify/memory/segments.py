"""The record of one segment of an episode (``UNIFY_MEMORY=in_episode``): what the episode had done at a checkpoint.

A checkpoint inside an episode (:mod:`.integration.in_episode`) starts a segment fork for the cells since the last
checkpoint. Its staging dir is ``<state>/memory-staging/<episode>.s<n>/``, and the harness keeps, under its own state
dir (never a box's), the episode as it stood at the checkpoint: ``<state>/memory-segments/<episode>.s<n>/episode.json``,
redacted as the episode record is, written once and atomically. The segment's capture (its own check runs on that
record's messages and actions) and the writer (``read_episode`` of ``<episode>.s<n>``) read it there; once the episode
is recorded, its whole record is the episode's own.

Its id is the staging name, ``<episode>.s<n>``; its parts are numbered from the episode's start, so the pointers of
the segment's summary name the same parts the episode's record will have for its request, observations and cells.
"""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

SEGMENTS_DIR = "memory-segments"
EPISODE_FILE = "episode.json"
#: The most bytes one segment's record keeps; past it the actions' responses are left out, marked so.
MAX_SNAPSHOT_BYTES = 64 * 1024**2
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,120}\.s[0-9]{1,6}\Z")
TRIMMED = {"__trimmed__": "this response was left out of the segment's record (over its bound)"}


def split_staging_name(name: str) -> tuple[str, int | None]:
    """``(episode, segment)`` of a staging dir name (``<episode>`` or ``<episode>.s<n>``)."""
    head, sep, tail = name.rpartition(".s")
    if sep and tail.isdigit() and head:
        return head, int(tail)
    return name, None


def is_segment_name(name: Any) -> bool:
    return isinstance(name, str) and _NAME.fullmatch(name) is not None


def snapshot_path(state_dir: Path | str, name: str) -> Path:
    return Path(state_dir) / SEGMENTS_DIR / name / EPISODE_FILE


def _doc(ep: Any, redactor: Any) -> dict:
    from .episodes import Episode

    assert isinstance(ep, Episode)
    doc = asdict(ep)
    return redactor.obj(doc) if redactor is not None else doc


def write_snapshot(state_dir: Path | str, name: str, ep: Any, redactor: Any = None) -> Path:
    """Write the segment's record once (a second write of the same segment is refused: ``FileExistsError``), redacted
    with *redactor*, atomically, under the harness's state dir."""
    if not is_segment_name(name):
        raise ValueError(f"not a segment name: {str(name)[:60]!r}")
    path = snapshot_path(state_dir, name)
    if path.exists():
        raise FileExistsError(f"segment {name} is already recorded")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = _doc(ep, redactor)
    data = json.dumps(doc, sort_keys=True, default=str).encode("utf-8")
    if len(data) > MAX_SNAPSHOT_BYTES:
        for a in doc.get("actions") or []:
            a["response"] = dict(TRIMMED)
        doc["transcript"] = []
        data = json.dumps(doc, sort_keys=True, default=str).encode("utf-8")
    fd, tmp = tempfile.mkstemp(prefix=".seg-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.link(tmp, path)  # fails if another writer got there first: written once
    finally:
        os.unlink(tmp)
    return path


def load_snapshot(state_dir: Path | str, name: str) -> Any | None:
    """The segment's record as an :class:`.episodes.Episode` (its id the segment's name), or None when it has none."""
    from .episodes import Action, Cell, CostRow, Episode

    if not is_segment_name(name):
        return None
    path = snapshot_path(state_dir, name)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        return None
    with os.fdopen(fd, "rb") as fh:
        if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
            return None
        try:
            doc = json.loads(fh.read(MAX_SNAPSHOT_BYTES + 1).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
    if not isinstance(doc, dict):
        return None
    try:
        doc["cells"] = [Cell(**c) for c in doc.get("cells") or []]
        doc["actions"] = [Action(**a) for a in doc.get("actions") or []]
        doc["costs"] = [CostRow(**c) for c in doc.get("costs") or []]
        doc["episode_id"] = name
        return Episode(**doc)
    except (TypeError, ValueError):
        return None


def loader(state_dir: Path | str) -> Callable[[str], Any | None]:
    """A reader of segment records by name (:func:`load_snapshot`) for the capture step and the writer's lookup."""
    return lambda name: load_snapshot(state_dir, name)
