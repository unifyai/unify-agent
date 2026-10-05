"""``UNIFY_ENTRY_RECORD``: links between functions and guidance, many to many, in a table of their own.

Functions are procedures and guidance entries are guidelines for using them.
A note may guide several functions and a function may be guided by several
notes; a note may also stand alone. The store has kept the links on one side
only, as each guidance entry's ``function_ids`` list. With the switch on they
are also kept as rows of ``entry_links`` (``function_id``, ``guidance_id``;
created in the store only by this switch), read from either side:
:func:`notes_of` gives a function's notes, :func:`functions_of` a note's
functions. The table is filled from the existing ``function_ids`` lists the
first time it is created, and kept equal to them when a guidance entry is
added, updated or deleted, or a function is deleted. Listings render cards
from it (:mod:`unify.actor.evidence_list`).
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Dict, Iterable, List, Set, Tuple

from unify import db

logger = logging.getLogger(__name__)

TABLE = "entry_links"


def enabled() -> bool:
    from . import entry_record

    return entry_record.enabled()


def _exists() -> bool:
    return (
        db.query_one(
            "SELECT 1 AS found FROM sqlite_master WHERE type = 'table' AND name = ?",
            (TABLE,),
        )
        is not None
    )


def ensure() -> None:
    """Create the table, filled from every guidance entry's ``function_ids``, if it does not exist."""
    if _exists():
        return
    with db.transaction():
        db.execute(
            f"CREATE TABLE IF NOT EXISTS {TABLE} (function_id INTEGER NOT NULL,"
            " guidance_id INTEGER NOT NULL, PRIMARY KEY (function_id, guidance_id))",
        )
        for row in db.query("SELECT guidance_id, function_ids FROM guidance"):
            ids = db.loads(row["function_ids"]) or []
            for fid in ids:
                try:
                    db.execute(
                        f"INSERT OR IGNORE INTO {TABLE} (function_id, guidance_id)"
                        " VALUES (?, ?)",
                        (int(fid), int(row["guidance_id"])),
                    )
                except (TypeError, ValueError):
                    continue


def set_guidance_links(guidance_id: int, function_ids: Iterable[int]) -> None:
    """Make the links of note *guidance_id* exactly *function_ids* (no-op while off)."""
    if not enabled():
        return
    try:
        ensure()
        with db.transaction():
            db.execute(
                f"DELETE FROM {TABLE} WHERE guidance_id = ?",
                (int(guidance_id),),
            )
            for fid in dict.fromkeys(int(f) for f in function_ids):
                db.execute(
                    f"INSERT OR IGNORE INTO {TABLE} (function_id, guidance_id) VALUES (?, ?)",
                    (fid, int(guidance_id)),
                )
    except sqlite3.Error as exc:
        logger.warning(f"entry links not written: {type(exc).__name__}: {exc}")


def drop(*, function_id: int | None = None, guidance_id: int | None = None) -> None:
    """Remove every link of a deleted function or note (no-op while off or before the table exists)."""
    if not enabled() or not _exists():
        return
    try:
        if function_id is not None:
            db.execute(
                f"DELETE FROM {TABLE} WHERE function_id = ?",
                (int(function_id),),
            )
        if guidance_id is not None:
            db.execute(
                f"DELETE FROM {TABLE} WHERE guidance_id = ?",
                (int(guidance_id),),
            )
    except sqlite3.Error as exc:
        logger.warning(f"entry links not removed: {type(exc).__name__}: {exc}")


def links() -> Set[Tuple[int, int]]:
    """Every ``(function_id, guidance_id)`` link (the table is created and filled first)."""
    ensure()
    return {
        (int(row["function_id"]), int(row["guidance_id"]))
        for row in db.query(f"SELECT function_id, guidance_id FROM {TABLE}")
    }


def sides(
    pairs: Iterable[Tuple[int, int]],
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """``(notes by function, functions by note)`` from *pairs*, each list in id order."""
    notes: Dict[int, List[int]] = {}
    functions: Dict[int, List[int]] = {}
    for fid, gid in sorted(pairs):
        notes.setdefault(fid, []).append(gid)
        functions.setdefault(gid, []).append(fid)
    for value in notes.values():
        value.sort()
    return notes, functions


def notes_of(function_id: int) -> List[int]:
    """The guidance ids linked to function *function_id*."""
    return sides(links())[0].get(int(function_id), [])


def functions_of(guidance_id: int) -> List[int]:
    """The function ids linked to guidance entry *guidance_id*."""
    return sides(links())[1].get(int(guidance_id), [])


__all__ = [
    "TABLE",
    "drop",
    "enabled",
    "ensure",
    "functions_of",
    "links",
    "notes_of",
    "set_guidance_links",
    "sides",
]
