"""Item records of a library-mode memory library: harness-written git notes, one map per library commit.

The map of library commit C is C's note on ``refs/notes/items``: a header line, then one JSON line per item
(sorted by id), each a whole record. The harness writes it once, when a consolidation first records C, and never
changes it, so every copy of C (the actor's export, the writer's box) renders the same bytes (cache rule).
``main`` is never rebased, so the notes stay attached. Each record names the commit that last changed its
item (``changed_at``).

A commit no consolidation recorded (a crash between a merge and its record, or history from before library mode) reads
the map of its nearest recorded first-parent ancestor (:func:`records_at`): statuses carry forward, and an item
the map does not hold is ``on-trial``.

Statuses and evidence levels use :mod:`.item_states`'s names. A map written before those names existed holds the old
status names: :func:`read_records` maps them as it reads (``experimental`` to ``on-trial``, ``stable`` to
``adopted``, ``suspect`` and ``deprecated`` to ``retired``; a ``deprecated`` item was the curator's, so it reads as a
retirement with the rule ``curate``), so an existing library loads unchanged.

Models never write these records. The writer and the actor read them through ``memory.show``
(``.memory/items.json``, one of the generated files).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping

from .gitio import GitError, Repo
from .item_states import (
    CURATE_RULE,
    LEVELS,
    ON_TRIAL,
    RETIRED,
    STATUSES,
    level_from_stored,
    level_of_record,
    status_from_stored,
)
from .library_index import EFFECTS

NOTES_REF = "items"
RECORD_VERSION = 1
SCAN = 200
_HEADER = "items_map"
#: the old stored status of a curator's retirement (an alias, or an item a curate pass retired)
OLD_CURATOR_STATUS = "deprecated"
_USE_TOTAL_KEYS = (
    "episodes",
    "uses",
    "seen",
    "errors",
    "refused",
    "negative_signals",
    "positive_signals",
)


def empty_record(item: str, kind: str) -> dict:
    """A record with the harness's own fields: why a status was given (``status_rule``, ``status_reason``,
    ``status_evidence``), the commit that last changed the item, its example call (``{"test", "line", "call"}``:
    a call quoted from its tests that the gate picked, None: none) and source channels, its
    effect as the gate derived it from its covers (``read``, ``write``, ``unknown``; None: not derived), and
    the bisect result with the rollback proposal. ``uid`` is the item's persistent identity, assigned once at its
    first record and carried across renames and aliases (:func:`new_uid`); ``renamed_from`` lists the ids it had
    before, oldest first.

    The evidence fields: ``level`` is the evidence level of a function's current version
    (:data:`.item_states.LEVELS`, from its gate run: :func:`.item_states.level_of_record`; None for a note),
    and ``level_reasons`` the findings that set it, one line each. ``role`` is ``"reader"`` for a function whose
    tests produced parsed fixtures other tests load, ``"procedure"`` for a recorded procedure, None otherwise.
    ``reader`` lists the ids of the readers whose parsed fixtures its tests load, and ``reads_for`` the fixture
    paths a reader produced.

    A procedure landed from a capture (``land_capture``) carries from that capture, in the harness's own store
    (:func:`.captures.landed_items`): ``landed_from`` (``{"capture", "reader"}``: the capture's id and the reader it
    was landed with), ``signature`` (the capture's applicability signature) and ``applies`` (the item that is its
    stated ``applies``). Each is None for any other item."""
    return {
        "item": item,
        "uid": None,
        "renamed_from": [],
        "kind": kind,
        "status": ON_TRIAL,
        "status_rule": None,
        "status_reason": None,
        "status_evidence": [],
        "changed_at": None,
        "provenance": {"episodes": [], "pass": None, "failure_only": False},
        "verification": None,
        "use": {},
        "alias_of": None,
        "example": None,
        "effect": None,
        "source_channels": [],
        "bisect": None,
        "rollback": None,
        # outcome-driven trust (trust.py): the record per kind of signal, and what an adoption rests on
        # (``verdicts`` or ``implicit``; None unless adopted)
        "track": None,
        "status_basis": None,
        "level": None,
        "level_reasons": [],
        "role": None,
        "reader": [],
        "reads_for": [],
        "landed_from": None,
        "signature": None,
        "applies": None,
    }


def new_uid(item: str, changed_at: str | None) -> str:
    """A fresh persistent identity for *item* first recorded at library commit *changed_at*. Derived, not random,
    so that recomputing a map gives the same bytes; after that it is copied, never recomputed.
    """
    return hashlib.sha256(f"{item}\0{changed_at or ''}".encode()).hexdigest()[:16]


def _line(obj: dict) -> str:
    # ASCII only: a line separator inside a value (U+2028) stays escaped, so git notes' lines are the records
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    )


def record_from_stored(rec: dict) -> dict:
    """*rec* as stored, with an old status name mapped to today's (:func:`.item_states.status_from_stored`) and an
    old level name mapped (:func:`.item_states.level_from_stored`; a record without a level keeps none). A
    ``deprecated`` record was the curator's (an alias, or an item a curate pass retired): it reads as ``retired``
    with the rule ``curate``, so it stays a curator's retirement. A record already in today's names is returned
    as it is."""
    stored = rec.get("status")
    level = rec.get("level")
    if stored in STATUSES and (level is None or level in LEVELS):
        return rec
    out = dict(rec)
    out["status"] = status_from_stored(stored)
    if stored == OLD_CURATOR_STATUS:
        out["status_rule"] = CURATE_RULE
    if level is not None:
        out["level"] = level_from_stored(level, default=None)
    return out


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
    """The map recorded on *sha*, by item id, or None when *sha* has none (or its header is not one). Old status
    and level names are read in today's names (:func:`record_from_stored`)."""
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
            out[rec["item"]] = record_from_stored(rec)
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
    """A ``status_of`` function: an item's recorded status (an old name mapped), ``on-trial`` without a (valid)
    one."""

    def get(item: str) -> str:
        rec = records.get(item)
        s = rec.get("status") if isinstance(rec, dict) else None
        return status_from_stored(s)

    return get


def effect_of(records: Mapping[str, dict]) -> Callable[[str], str | None]:
    """An item's recorded effect (``read``, ``write`` or ``unknown``), None when none was derived."""

    def get(item: str) -> str | None:
        rec = records.get(item)
        e = rec.get("effect") if isinstance(rec, dict) else None
        return e if e in EFFECTS else None

    return get


def level_of(records: Mapping[str, dict]) -> Callable[[str], str | None]:
    """A function's recorded evidence level (an old name mapped), None for a note, an item with no record, or a
    record whose level is None. A function record written before levels existed (no ``level`` key) fails closed:
    it reads as its verification record gives it (:func:`.item_states.level_of_record`: ``unchecked`` unless its
    example calls ran on recorded values), the level the next map writes for it."""

    def get(item: str) -> str | None:
        rec = records.get(item)
        if not isinstance(rec, dict):
            return None
        if "level" not in rec:
            return level_of_record(rec.get("verification")) if rec.get("kind") == "function" else None
        return level_from_stored(rec.get("level"), default=None)

    return get


def level_reasons_of(record: Mapping | None) -> list[str]:
    """The findings that set an item's evidence level, as ``memory.show`` prints them (strings only)."""
    reasons = record.get("level_reasons") if isinstance(record, Mapping) else None
    return [str(r) for r in reasons] if isinstance(reasons, list) else []


def is_outcome_retirement(record: Mapping | None) -> bool:
    """Whether *record* is retired by its outcomes (errors, negative signals, failure-only provenance, or a
    dependency so retired), not by the curator (rule ``curate``)."""
    if not isinstance(record, Mapping):
        return False
    rec = record_from_stored(dict(record))
    return rec.get("status") == RETIRED and rec.get("status_rule") != CURATE_RULE


def episodes_of(records: Mapping[str, dict]) -> Callable[[str], int | None]:
    """How many recorded episodes a function's returning example calls ran on (its verification record's
    ``examples.episodes``), None when none was recorded: evidence shown beside the function, never a gate.
    """

    def get(item: str) -> int | None:
        rec = records.get(item)
        v = rec.get("verification") if isinstance(rec, dict) else None
        ex = v.get("examples") if isinstance(v, dict) else None
        eps = ex.get("episodes") if isinstance(ex, dict) else None
        return len(eps) if isinstance(eps, list) else None

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
    ce = v.get("cross_episode")  # records written before examples replaced it
    if isinstance(ce, dict):
        out["cross_episode"] = f"{ce.get('ok', 0)}/{ce.get('ran', 0)} ok"
    ex = v.get("examples")
    if isinstance(ex, dict):
        episodes = ex.get("episodes") if isinstance(ex.get("episodes"), list) else []
        out["examples"] = (
            f"{ex.get('calls', 0)} calls, {ex.get('grounded_calls', 0)} on recorded values from "
            f"{len(episodes)} episodes"
        )
        pert = ex.get("perturbation")
        if isinstance(pert, dict):
            out["unseen_values"] = (
                f"{pert.get('judged', 0)} calls judged on {pert.get('fields', 0)} fields"
            )
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
