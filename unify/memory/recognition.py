"""Whether a landed procedure is listed: by its structure, never by running it (rev 2).

Before rev 2 a landed procedure's reader, ``applies``, procedure and ``holds`` ran, confined, on the store's other
recorded requests when it landed, and the counts of true and false holds (and of the recorded variants its reader
failed on) decided whether the package's structural check and the librarian listed it. That was code run to classify, which the
harness no longer does: no capture or landed procedure is executed to classify, dedupe, rank or list it. The actor's
own call of an item is the one place stored code runs.

Now every landed procedure is listed, and the librarian judges fit at ask time from its catalogue row
(``library_export.dispatch_data``: the writer's declared ``applies`` as text, its form, the tool calls its code names,
the static constant label, and its capture's occurrences and outcomes from signals). A landing records
:data:`NOT_RUN` where the old counts were (its ``recognition``), so a reader sees an explicit absence, never a silent
default. A record from before rev 2 that carries counts is not used: the procedure is listed, and its line says why.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: What a landing records as its ``recognition`` (rev 2): nothing was run, and why.
NOT_RUN: dict = {"ran": False, "why": "rev2: a landed procedure is never run to decide its listing"}
#: The line a listed procedure's card carries.
LISTED_BY_STRUCTURE = "listed by its structure: it is never run to decide its listing"
EARLIER_TEST_UNUSED = (
    "listed by its structure: an earlier landing test that ran it is not used (rev 2 never runs a procedure to "
    "decide its listing)"
)


def not_run() -> dict:
    """A fresh copy of :data:`NOT_RUN`, for a landing's record."""
    return dict(NOT_RUN)


def listing(rec: Mapping[str, Any] | None) -> tuple[bool, str]:
    """Whether a procedure with recognition record *rec* is listed by ``Item.fits`` and in the librarian's
    catalogue, and the line that says why: always listed (module docstring); ``""`` for a procedure with no record."""
    if not isinstance(rec, Mapping):
        return True, ""
    if rec.get("ran") is False:
        return True, LISTED_BY_STRUCTURE
    return True, EARLIER_TEST_UNUSED


def listed(record: Mapping[str, Any] | None) -> bool:
    """Whether the item record *record* is listed (:func:`listing`): always, under rev 2."""
    rec = record.get("recognition") if isinstance(record, Mapping) else None
    return listing(rec)[0]


