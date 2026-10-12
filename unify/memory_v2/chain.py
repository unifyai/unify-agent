"""Counts at each hand-off of staged candidates, from the fork's reply to the writer's adoption.

A store pass records, in its end event, how many candidate files (``candidates/<name>.py``) survived each hop:

- ``parsed``: candidate blocks the fork worker parsed from its replies (``fork-blocks.json`` beside each batch
  episode's staging; None when a staged episode has no such record, since that count is unknown);
- ``staged``: candidate files the pass read from staging (only allowed regular files are taken);
- ``pooled_at_start``: staged candidates held by a pool entry when the writer started;
- ``batch_map_rows``: candidate rows the batch map lists;
- ``read_by_writer``: staged candidate files the writer read (a ``read`` of the file, a box open of it, or a
  tool event that reads it);
- ``adopted``: staged candidates a landed item adopts (by its staged path or by the pool entry holding it).

On a clean chain every count equals the one before it; a drop shows where candidates were lost. Nothing here reads
transcript text: reads come from the tool calls' own arguments and from structured tool or open events.
"""

from __future__ import annotations

import json
import posixpath
import re
from pathlib import Path
from typing import Iterable

STAGING_PREFIX = "/inputs/staging/"
_CANDIDATE = re.compile(r"candidates/[^/]+\.py")
KEYS = (
    "parsed",
    "staged",
    "pooled_at_start",
    "batch_map_rows",
    "read_by_writer",
    "adopted",
)


def staged_candidates(views: dict) -> list[str]:
    """``<episode>/candidates/<name>.py`` for each candidate file in *views* (staging by episode)."""
    return sorted(
        f"{eid}/{rel}"
        for eid, view in views.items()
        for rel in view.files
        if _CANDIDATE.fullmatch(rel)
    )


def parsed_candidates(root: Path | None, eids: Iterable[str]) -> int | None:
    """Candidate blocks parsed for *eids*, from each episode's ``fork-blocks.json`` under *root*; None when any of
    them has no readable record (the parse count is then unknown), 0 for no episodes."""
    total = 0
    for eid in eids:
        try:
            doc = json.loads((Path(root) / eid / "fork-blocks.json").read_text())
        except (OSError, ValueError, TypeError):
            return None
        blocks = doc.get("blocks") if isinstance(doc, dict) else None
        if not isinstance(blocks, list):
            return None
        total += sum(
            1
            for b in blocks
            if isinstance(b, dict)
            and b.get("status") == "parsed"
            and (
                b.get("kind") == "candidate"
                or _CANDIDATE.fullmatch(str(b.get("path") or ""))
            )
        )
    return total


def _args(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        out = json.loads(raw) if isinstance(raw, str) else {}
    except ValueError:
        return {}
    return out if isinstance(out, dict) else {}


def read_paths(
    messages: Iterable[dict] = (),
    events: Iterable[dict] = (),
) -> set[str]:
    """Absolute paths the writer read: each ``read`` tool call's ``path`` (from the call's own arguments), each
    tool event path with mode ``read`` (``tool-events.jsonl`` rows) and each open event (``box-opens.jsonl`` rows,
    read modes only)."""
    out: set[str] = set()
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = (tc or {}).get("function") or {}
            if fn.get("name") == "read":
                path = _args(fn.get("arguments")).get("path")
                if isinstance(path, str) and path.startswith("/"):
                    out.add(posixpath.normpath(path))
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("event") == "open":
            mode = str(ev.get("mode") or "r")
            if isinstance(ev.get("path"), str) and "r" in mode:
                out.add(posixpath.normpath(ev["path"]))
            continue
        if ev.get("status") not in (None, "ok"):
            continue
        for p in ev.get("paths") or []:
            if isinstance(p, dict) and p.get("mode") == "read" and isinstance(p.get("path"), str):
                out.add(posixpath.normpath(p["path"]))
    return out


def staged_reads(paths: Iterable[str], staged: Iterable[str]) -> int:
    """How many of the *staged* candidates (``<episode>/candidates/<name>.py``) were read at their /inputs path."""
    seen = {p[len(STAGING_PREFIX) :] for p in paths if p.startswith(STAGING_PREFIX)}
    return sum(1 for s in staged if s in seen)


def adopted_candidates(
    provenance: Iterable[dict],
    staged: Iterable[str],
    staged_entry: dict[str, str],
) -> int:
    """How many *staged* candidates a landed item adopts: by its ``staging:`` ref, or by ``pool:<id>`` of the
    entry holding it (*staged_entry*: staging ref -> pool entry id)."""
    refs: set[str] = set()
    for row in provenance:
        refs.update(str(r) for r in (row.get("adopts") or []))
    n = 0
    for s in staged:
        ref = f"staging:{s}"
        eid = staged_entry.get(ref)
        if ref in refs or (eid is not None and f"pool:{eid}" in refs):
            n += 1
    return n


def counts(
    *,
    views: dict,
    root: Path | None,
    staged_entry: dict[str, str],
    pool_ids: set[str],
    batch_rows: Iterable[dict],
    read: Iterable[str],
    provenance: Iterable[dict],
) -> dict:
    """The chain counts (:data:`KEYS`) of one store pass. *views*: its staging by episode; *root*: where staging
    was read (for the fork's parse records); *staged_entry*: staging ref -> pool entry id at the pass's start;
    *pool_ids*: the entries the pool held at the start; *batch_rows*: the batch map's rows; *read*: the paths the
    writer read (:func:`read_paths`); *provenance*: the gate's rows for the landed items."""
    staged = staged_candidates(views)
    rows = sum(
        len((r.get("staging") or {}).get("candidates") or [])
        for r in batch_rows
        if isinstance(r, dict) and isinstance(r.get("staging"), dict)
    )
    return {
        "parsed": parsed_candidates(root, sorted(views)) if views else 0,
        "staged": len(staged),
        "pooled_at_start": sum(
            1 for s in staged if staged_entry.get(f"staging:{s}") in pool_ids
        ),
        "batch_map_rows": rows,
        "read_by_writer": staged_reads(read, staged),
        "adopted": adopted_candidates(provenance, staged, staged_entry),
    }
