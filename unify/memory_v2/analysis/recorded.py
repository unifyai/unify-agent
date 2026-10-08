"""Per-kind queries over exported episodes and the recorded file blobs (spec §F2).

Inside a consolidation pass, ``/inputs/episodes/<id>.json`` holds each episode and ``/inputs/blobs/``
the file blobs its worktree actions recorded, as one file per blob id (the content's SHA-256), plus
``index.json``: ``{"exported": [ids], "skipped": {id: reason}}``. Blobs over the export caps are skipped
and listed there.

Tests in the memory library run in the gate with only ``/memory`` mounted, so a reader's fixtures must be
copied into ``env/<channel>/tests/`` (as files) before its tests can use them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

BLOB_DIR = Path("/inputs/blobs")
BLOB_ID = re.compile(r"^[0-9a-f]{64}\Z")


def _get(a: Any, name: str, default: Any = None) -> Any:
    return a.get(name, default) if isinstance(a, dict) else getattr(a, name, default)


def kind_of(a: Any) -> str:
    """An action's kind; rows recorded before kinds existed are ``tool``."""
    return _get(a, "kind") or "tool"


def actions_of_kind(row: dict, kind: str) -> list[dict]:
    """The exported actions of one episode row with the given kind (each keeps its ``index``)."""
    return [a for a in row.get("actions") or [] if kind_of(a) == kind]


def blob_ids(a: Any) -> list[str]:
    """The recorded blob ids of a worktree action (before, then after), well-formed ones only."""
    resp = _get(a, "response")
    if kind_of(a) != "worktree" or not isinstance(resp, dict):
        return []
    out = []
    for k in ("blob_before", "blob_after"):
        v = resp.get(k)
        if isinstance(v, str) and BLOB_ID.match(v) and v not in out:
            out.append(v)
    return out


def worktree_files(row: dict) -> list[dict]:
    """One dict per worktree action with a blob: index, method, path, blob ids and recorded shape."""
    out = []
    for a in actions_of_kind(row, "worktree"):
        resp = a.get("response") if isinstance(a.get("response"), dict) else {}
        args = a.get("args") or []
        out.append(
            {
                "episode": row["episode_id"],
                "index": a.get("index"),
                "method": a.get("method"),
                "path": args[0] if args and isinstance(args[0], str) else None,
                "blob_before": resp.get("blob_before"),
                "blob_after": resp.get("blob_after"),
                "shape": resp.get("shape"),
            },
        )
    return out


def exported_blobs(root: Path | str = BLOB_DIR) -> dict:
    """The export's ``index.json`` (``{"exported": [...], "skipped": {...}}``); empty if absent."""
    p = Path(root) / "index.json"
    if not p.is_file():
        return {"exported": [], "skipped": {}}
    return json.loads(p.read_text())


def load_blob(sha: str, root: Path | str = BLOB_DIR) -> bytes:
    """The bytes of a recorded blob. KeyError if it was not exported (the reason, if it was skipped)."""
    if not isinstance(sha, str) or not BLOB_ID.match(sha):
        raise ValueError(f"not a blob id: {sha!r}"[:200])
    p = Path(root) / sha
    if not p.is_file():
        why = (
            exported_blobs(root)
            .get("skipped", {})
            .get(sha, "not recorded by this pass's episodes")
        )
        raise KeyError(f"blob {sha} is not exported: {why}")
    return p.read_bytes()


def iter_episodes(root: Path | str = "/inputs/episodes") -> Iterable[dict]:
    """The exported episode rows, in file-name order."""
    for p in sorted(Path(root).glob("*.json")):
        yield json.loads(p.read_text())
