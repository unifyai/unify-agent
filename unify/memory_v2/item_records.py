"""Item records of a memory v2.1 library (spec §4.4, D40): harness-written git notes, one map per library commit.

The map of library commit C is C's note on ``refs/notes/items``: a header line, then one JSON line per item
(sorted by id), each a whole record. The harness writes it once, when a consolidation first records C, and never
changes it, so every copy of C (the actor's export, the writer's box) renders the same bytes (cache rule).
``main`` is never rebased (D13), so the notes stay attached. Each record names the commit that last changed its
item (``changed_at``).

A commit no consolidation recorded (a crash between a merge and its record, or history from before v2.1) reads
the map of its nearest recorded first-parent ancestor (:func:`records_at`): statuses carry forward, and an item
the map does not hold is ``experimental``.

Models never write these records. The writer and the actor read them through ``memory.show``
(``.memory/items.json``, P3's generated files).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping

from .gitio import GitError, Repo
from .library_index import STATUSES

NOTES_REF = "items"
RECORD_VERSION = 1
SCAN = 200
_HEADER = "items_map"
_USE_TOTAL_KEYS = ("episodes", "uses", "errors", "negative_signals", "positive_signals")


def empty_record(item: str, kind: str) -> dict:
    """A record (spec §4.4) with the harness's own fields: why a status was given (``status_rule``,
    ``status_reason``, ``status_evidence``), the commit that last changed the item, its input form and source
    channels (D42), and the bisect result with the rollback proposal (§10.2)."""
    return {
        "item": item,
        "kind": kind,
        "status": "experimental",
        "status_rule": None,
        "status_reason": None,
        "status_evidence": [],
        "changed_at": None,
        "provenance": {"episodes": [], "pass": None, "failure_only": False},
        "verification": None,
        "use": {},
        "alias_of": None,
        "input": None,
        "source_channels": [],
        "bisect": None,
        "rollback": None,
    }


def _line(obj: dict) -> str:
    # ASCII only: a line separator inside a value (U+2028) stays escaped, so git notes' lines are the records
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    )


def write_records(repo: Repo, sha: str, records: Mapping[str, dict]) -> bool:
    """Record *records* as the map of library commit *sha*. Returns False, writing nothing, when *sha* already
    has a map: a commit's map is never rewritten. ``ValueError`` (nothing written) for a record whose ``item``
    is not its key or whose status is unknown."""
    for item, rec in records.items():
        if rec.get("item") != item:
            raise ValueError(f"the record under {item!r} names {rec.get('item')!r}")
        if rec.get("status") not in STATUSES:
            raise ValueError(f"unknown status {rec.get('status')!r} for {item}")
    if repo.notes(sha, ref=NOTES_REF):
        return False
    lines = [_line({_HEADER: RECORD_VERSION, "commit": sha, "count": len(records)})]
    lines += [_line(records[i]) for i in sorted(records)]
    repo.append_note_lines(sha, lines, NOTES_REF)
    return True


def read_records(repo: Repo, sha: str) -> dict[str, dict] | None:
    """The map recorded on *sha*, by item id, or None when *sha* has none (or its header is not one)."""
    lines = repo.notes(sha, ref=NOTES_REF)
    if not lines:
        return None
    try:
        head = json.loads(lines[0])
    except ValueError:
        return None
    if not isinstance(head, dict) or head.get(_HEADER) != RECORD_VERSION:
        return None
    out: dict[str, dict] = {}
    for ln in lines[1:]:
        try:
            rec = json.loads(ln)
        except ValueError:
            continue
        if isinstance(rec, dict) and isinstance(rec.get("item"), str):
            out[rec["item"]] = rec
    return dict(sorted(out.items()))


def records_at(
    repo: Repo,
    sha: str,
    *,
    scan: int = SCAN,
) -> tuple[dict[str, dict], str | None]:
    """The map that holds at library commit *sha*: its own, else its nearest recorded first-parent ancestor's
    within *scan* commits, with the commit it came from; ``({}, None)`` when none is recorded.
    """
    try:
        shas = repo.run(
            "rev-list",
            "--first-parent",
            f"--max-count={scan}",
            sha,
        ).split()
    except GitError:
        return {}, None
    for c in shas:
        recs = read_records(repo, c)
        if recs is not None:
            return recs, c
    return {}, None


def status_of(records: Mapping[str, dict]) -> Callable[[str], str]:
    """P3's ``status_of``: an item's recorded status, ``experimental`` without a (valid) one."""

    def get(item: str) -> str:
        rec = records.get(item)
        s = rec.get("status") if isinstance(rec, dict) else None
        return s if s in STATUSES else "experimental"

    return get


def verification_summary(record: Mapping | None) -> dict | None:
    """What ``memory.show`` prints of the verification record: flat, one value per key."""
    v = record.get("verification") if isinstance(record, Mapping) else None
    if not isinstance(v, dict):
        return None
    out: dict = {}
    if "tests" in v:
        out["tests"] = len(v["tests"]) if isinstance(v["tests"], list) else v["tests"]
    for key in ("drawn_inputs_read", "exact_assertions"):
        if key in v:
            out[key] = v[key]
    ce = v.get("cross_episode")
    if isinstance(ce, dict):
        out["cross_episode"] = f"{ce.get('ok', 0)}/{ce.get('ran', 0)} ok"
    m = v.get("mutation")
    out["mutation"] = (
        f"{m.get('killed', 0)}/{m.get('total', 0)} killed"
        if isinstance(m, dict)
        else "not run"
    )
    return out


def use_totals(record: Mapping | None) -> dict:
    """What ``memory.show`` prints of the use record: the totals over every library commit."""
    use = record.get("use") if isinstance(record, Mapping) else None
    use = use if isinstance(use, dict) else {}
    out = {"commits": len(use), **dict.fromkeys(_USE_TOTAL_KEYS, 0)}
    for row in use.values():
        if isinstance(row, dict):
            for key in _USE_TOTAL_KEYS:
                value = row.get(key, 0)
                out[key] += (
                    value
                    if isinstance(value, int) and not isinstance(value, bool)
                    else 0
                )
    return out
