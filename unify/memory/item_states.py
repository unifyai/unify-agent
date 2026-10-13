"""The two axes a library item is described on, and the one vocabulary every module imports for them.

**Evidence level**: how much the evidence behind an item's current version shows. It rises in this order:

- ``unchecked``: the item passed every integrity and safety check, and at least one example call of it returns on a
  value from a recorded episode (in any recorded form), but some evidence check did not hold; the findings that set
  the level are kept with the item (``level_reasons``). A capture or a pool draft is also ``unchecked``.
- ``reproduces-its-case``: its example calls reproduce their recorded cases under every evidence check of the gate
  (structured arguments from fixtures or through a stored reader, the check with unseen values, the literal and
  rule checks, an exact assertion, and, for a procedure, its recorded result reproduced).
- ``confirmed-by-outcomes``: outcomes after use confirm it. The gate never gives this level; outcome tracking does,
  and outcome tracking alone raises an item to it, lowers it again and changes statuses.

**Status**: where the item is in its life, from what happens after it is stored. ``on-trial`` (every new or changed
item), ``adopted`` (used cleanly), ``retired`` (failed in use, depends on a failed item, or retired by the curator).
An item can be ``on-trial`` at any evidence level. A ``retired`` item is hidden from the index and stays reachable
through ``memory.show``.

Records written before these names existed carry the old status names; :func:`status_from_stored` maps them on
read (``experimental`` to ``on-trial``, ``stable`` to ``adopted``, ``suspect`` and ``deprecated`` to ``retired``). A
record without a level reads as ``unchecked`` (it fails closed); :func:`level_of_record` gives an item admitted before
levels existed ``reproduces-its-case`` only where its verification record shows its example calls ran on recorded
values.
"""

from __future__ import annotations

from typing import Any

# --- evidence levels ----------------------------------------------------------------------------------------------

UNCHECKED = "unchecked"
REPRODUCES_ITS_CASE = "reproduces-its-case"
CONFIRMED_BY_OUTCOMES = "confirmed-by-outcomes"
#: lowest first
LEVELS = (UNCHECKED, REPRODUCES_ITS_CASE, CONFIRMED_BY_OUTCOMES)
#: the levels the gate gives; the third comes from outcomes only
GATE_LEVELS = (UNCHECKED, REPRODUCES_ITS_CASE)
#: the level of a stored record that has none: the lowest, so a missing level fails closed (:func:`level_of_record`
#: gives an older record ``reproduces-its-case`` only where its verification shows its examples ran)
STORED_DEFAULT_LEVEL = UNCHECKED
#: level names of the design draft that a record may still hold
OLD_LEVELS = {
    "unverified": UNCHECKED,
    "experimental": UNCHECKED,
    "verified": REPRODUCES_ITS_CASE,
}

# --- statuses -----------------------------------------------------------------------------------------------------

ON_TRIAL = "on-trial"
ADOPTED = "adopted"
RETIRED = "retired"
STATUSES = (ON_TRIAL, ADOPTED, RETIRED)
#: statuses the index leaves out (``memory.show`` still shows them)
HIDDEN = frozenset({RETIRED})
#: the status rule of a curator's retirement (an alias, or an item a curate pass retired): it is kept as it is,
#: while a retirement by outcomes is decided again from the uses of the item's current version
CURATE_RULE = "curate"
#: stored status names from before the rename, and the status each now means
OLD_STATUSES = {
    "experimental": ON_TRIAL,
    "stable": ADOPTED,
    "suspect": RETIRED,
    "deprecated": RETIRED,
}


def status_from_stored(value: Any, default: str = ON_TRIAL) -> str:
    """A stored status in today's names: a current name as it is, an old name mapped, anything else *default*."""
    if value in STATUSES:
        return value
    if isinstance(value, str) and value in OLD_STATUSES:
        return OLD_STATUSES[value]
    return default


def level_from_stored(value: Any, default: str | None = STORED_DEFAULT_LEVEL) -> str | None:
    """A stored evidence level in today's names; a record without one (None) gets *default*."""
    if value in LEVELS:
        return value
    if isinstance(value, str) and value in OLD_LEVELS:
        return OLD_LEVELS[value]
    return default


def level_rank(level: str) -> int:
    """The position of *level* in :data:`LEVELS` (lowest 0); an unknown level ranks lowest."""
    return LEVELS.index(level) if level in LEVELS else 0


def lowest(*levels: str) -> str:
    """The lowest of *levels* (:data:`UNCHECKED` for none)."""
    return min(levels, key=level_rank) if levels else UNCHECKED


def level_of_record(verification: Any) -> str:
    """The evidence level of an item from its verification record (the gate's, as the item record keeps it).

    Its ``level`` when it has one (old names mapped). A record from before levels existed has none: it reads as
    ``reproduces-its-case`` only when it shows the gate ran its example calls on recorded values (``examples`` with at
    least one grounded call), else ``unchecked``. No record at all is ``unchecked``.
    """
    if not isinstance(verification, dict):
        return UNCHECKED
    stored = level_from_stored(verification.get("level"), default=None)
    if stored is not None:
        return stored
    examples = verification.get("examples")
    grounded = examples.get("grounded_calls") if isinstance(examples, dict) else None
    if isinstance(grounded, int) and not isinstance(grounded, bool) and grounded > 0:
        return REPRODUCES_ITS_CASE
    return UNCHECKED
