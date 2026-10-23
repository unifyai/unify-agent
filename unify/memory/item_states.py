"""The two axes a library item is described on, and the one vocabulary every module imports for them.

**Evidence level**: how much the evidence behind an item's current version shows. It rises in this order:

- ``unchecked``: the item passed every integrity and safety check, and at least one example call of it returns on a
  value from a recorded episode (in any recorded form), but some evidence check did not hold; the findings that set
  the level are kept with the item (``level_reasons``). A capture or a pool draft is also ``unchecked``.
- ``reproduces-its-case``: a level a run of an item's examples or tests once gave. Nothing gives it now, and a stored
  record that holds it reads as ``unchecked``: no test result is evidence for an item.
- ``confirmed-by-outcomes``: outcomes after use confirm it. The gate never gives this level; outcome tracking does,
  and outcome tracking alone raises an item to it, lowers it again (to ``unchecked``) and changes statuses.

**Status**: where the item is in its life, from what happens after it is stored. ``on-trial`` (every new or changed
item), ``adopted`` (used cleanly), ``retired`` (failed in use, depends on a failed item, or retired by the curator).
An item can be ``on-trial`` at any evidence level. A ``retired`` item is hidden from the index and stays reachable
through ``memory.get``.

Records written before these names existed carry the old status names; :func:`status_from_stored` maps them on
read (``experimental`` to ``on-trial``, ``stable`` to ``adopted``, ``suspect`` and ``deprecated`` to ``retired``). A
record without a level reads as ``unchecked`` (it fails closed), as does one whose level a test run gave.
"""

from __future__ import annotations

from typing import Any

# --- evidence levels ----------------------------------------------------------------------------------------------

UNCHECKED = "unchecked"
REPRODUCES_ITS_CASE = "reproduces-its-case"
CONFIRMED_BY_OUTCOMES = "confirmed-by-outcomes"
#: lowest first
LEVELS = (UNCHECKED, REPRODUCES_ITS_CASE, CONFIRMED_BY_OUTCOMES)
#: the level of a stored record that has none: the lowest, so a missing level fails closed
STORED_DEFAULT_LEVEL = UNCHECKED
#: level names a stored record may still hold, read as today's: a level a run of tests or examples gave (``verified``,
#: ``reproduces-its-case``) reads as ``unchecked``, since no test result is ever evidence for an item
OLD_LEVELS = {
    "unverified": UNCHECKED,
    "experimental": UNCHECKED,
    "verified": UNCHECKED,
    REPRODUCES_ITS_CASE: UNCHECKED,
}

# --- statuses -----------------------------------------------------------------------------------------------------

ON_TRIAL = "on-trial"
ADOPTED = "adopted"
RETIRED = "retired"
STATUSES = (ON_TRIAL, ADOPTED, RETIRED)
#: statuses the index leaves out (``memory.get`` still opens them)
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


#: the same map under the name the outcome-trust module uses for it (:mod:`.trust` imports its names from here)
FROM_LIFECYCLE = OLD_STATUSES


def status_from_stored(value: Any, default: str = ON_TRIAL) -> str:
    """A stored status in today's names: a current name as it is, an old name mapped, anything else *default*."""
    if value in STATUSES:
        return value
    if isinstance(value, str) and value in OLD_STATUSES:
        return OLD_STATUSES[value]
    return default


def level_from_stored(value: Any, default: str | None = STORED_DEFAULT_LEVEL) -> str | None:
    """A stored evidence level in today's names; a record without one (None) gets *default*."""
    if isinstance(value, str) and value in OLD_LEVELS:
        return OLD_LEVELS[value]
    if value in LEVELS:
        return value
    return default


def level_of_record(verification: Any) -> str:
    """The evidence level of an item from its verification record (the gate's, as the item record keeps it).

    Its ``level`` when it has one (old names mapped, a test-given level read as ``unchecked``); otherwise
    ``unchecked``, whatever examples or tests its record shows ran: no test result is evidence for an item.
    """
    if not isinstance(verification, dict):
        return UNCHECKED
    stored = level_from_stored(verification.get("level"), default=None)
    return stored if stored is not None else UNCHECKED
