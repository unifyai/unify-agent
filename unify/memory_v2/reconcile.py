"""Pricing a v2.1 pass's model calls after the fact, by lane and window (spec v2.1 §6; P7 Amendment D).

Passes run one at a time (:mod:`.integration.async_pass`) and Sol has its own proxy lane, so a pass's calls are the
Sol-lane requests of the proxy journal (``UNIFY_MEMORY_V2_SOL_JOURNAL``) that *started* inside the pass's window.
The journal is append-only and carries no clock, so the window is a byte range: the journal's size when the pass
started and when its supervisor recorded its end (:func:`journal_offset`). A request belongs to the pass when its
``request_started`` row lies in that range; its terminal row is read wherever it was appended, so a call cancelled
at the wall bound, whose terminal row the proxy writes late, is still found.

A request is priced by its last row's ``account_charge``. One with no charge (cancelled in flight, a transport
error, a terminal row not yet written) stays ``unknown``, never zero, and the run guard books it at
:data:`SOL_CALL_WORST_CASE_USD` (the guards' tier-1 catalogue maximum for one Sol request; the 8 Oct rule for
never-priceable requests). No header is sent and no generation lookup is made: the proxy mints its own attempt id
and the Sol lane refuses ``GET /generation`` (Amendment D). Money is a decimal string.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

UNKNOWN = "unknown"
#: How long a reconciliation keeps re-reading the journal for terminal rows the proxy has not written yet.
RECONCILE_S = 120.0
RETRY_S = 5.0
#: One Sol request's worst case, as the programme guard bounds it (tier 1, catalogue cbb212a9: context x the highest
#: input-side rate + context x the completion rate, for openai/gpt-6-sol). An unpriced call is booked at this.
SOL_CALL_WORST_CASE_USD = Decimal("13.125")
_TERMINAL = frozenset({"completed", "failed", "cancelled", "error", "ok"})


def money(value: Any) -> str:
    """A non-negative finite decimal string, or ``unknown``."""
    if value is None or isinstance(value, bool):
        return UNKNOWN
    try:
        amount = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return UNKNOWN
    if not amount.is_finite() or amount < 0:
        return UNKNOWN
    return format(abs(amount) if amount.is_zero() else amount, "f")


def journal_offset(path: str | None) -> int | None:
    """The journal's size in bytes now (a window edge), or None when there is no readable journal."""
    if not path:
        return None
    try:
        return Path(path).stat().st_size
    except OSError:
        return None


@dataclass
class LaneCall:
    """One Sol-lane request the pass started: its attempt id (the proxy's), its last status, its charge."""

    attempt_id: str
    status: str = "started"
    usd: str = UNKNOWN
    generation_id: str | None = None

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL


def _rows(path: str, offset: int = 0) -> Iterable[tuple[int, dict]]:
    """``(byte offset of the line, row)`` of every complete, parseable JSON line of the journal from byte *offset*
    (a line boundary: a window edge is the journal's size when it was taken)."""
    with open(path, "rb") as fh:
        fh.seek(offset)
        data = fh.read()
    complete = data[
        : data.rfind(b"\n") + 1
    ]  # a last line without its newline is still being written
    at = offset
    for raw in complete.split(b"\n")[:-1]:
        try:
            row = json.loads(raw)
        except ValueError:
            row = None
        if isinstance(row, dict):
            yield at, row
        at += len(raw) + 1


def _on_lane(row: dict, models: frozenset[str]) -> bool:
    return any(str(row.get(k) or "") in models for k in ("requested_model", "model"))


def lane_calls(
    path: str,
    start: int,
    end: int,
    models: Iterable[str],
) -> dict[str, LaneCall]:
    """The Sol-lane requests whose ``request_started`` row lies in bytes ``[start, end)``, each with its LAST row's
    status, charge and generation id from anywhere in the journal (terminal rows may come later).
    """
    lane = frozenset(m for m in models if m)
    calls: dict[str, LaneCall] = {}
    # from the window's start: a request's later rows always follow its request_started row (append-only)
    for at, row in _rows(path, start):
        rid = row.get("request_attempt_id") or row.get("call_id")
        if not isinstance(rid, str) or not rid:
            continue
        if rid not in calls:
            if not (
                start <= at < end
                and row.get("origin") == "request_started"
                and _on_lane(row, lane)
            ):
                continue
            calls[rid] = LaneCall(rid)
        c = calls[rid]
        c.status = str(row.get("status") or c.status)
        charge = money(row.get("account_charge"))
        if charge != UNKNOWN:
            c.usd = charge
        gid = row.get("generation_id")
        if isinstance(gid, str) and gid:
            c.generation_id = gid
    return calls


def unavailable(worst_case_usd: Decimal = SOL_CALL_WORST_CASE_USD) -> dict:
    """The summary when the journal cannot be read: nothing priced, nothing known, the whole cap kept."""
    return {
        "journal": "unavailable",
        "calls": None,
        "priced": 0,
        "unknown": None,
        "usd": "0",
        "worst_case_usd": money(worst_case_usd),
        "booked_usd": UNKNOWN,
    }


async def reconcile_window(
    path: str | None,
    start: int | None,
    end: int | None,
    *,
    models: Iterable[str],
    worst_case_usd: Decimal = SOL_CALL_WORST_CASE_USD,
    budget_s: float = RECONCILE_S,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> dict:
    """The pass's Sol calls priced from the journal, re-read until each started call has a terminal row or
    *budget_s* has passed. Returns the summary the pass's end event and the run guard's settle line carry:

    ``journal`` (``read`` or ``unavailable``: no path, no window, or unreadable; then nothing is priced and the run
    guard keeps the pass's whole cap), ``calls``, ``priced``, ``unknown``, ``usd`` (the priced sum), and
    ``booked_usd`` = ``usd`` + ``unknown`` x *worst_case_usd* (the worst case the run guard books).
    """
    models = tuple(models)
    if not path or start is None or end is None or end < start:
        return unavailable(worst_case_usd)
    deadline = clock() + float(budget_s)
    while True:
        try:
            calls = lane_calls(path, start, end, models)
        except OSError:
            return unavailable(worst_case_usd)
        if all(c.terminal for c in calls.values()) or clock() >= deadline:
            break
        await sleep(min(RETRY_S, max(0.0, deadline - clock())))
    priced = [c for c in calls.values() if c.usd != UNKNOWN]
    unknown = len(calls) - len(priced)
    usd = sum((Decimal(c.usd) for c in priced), Decimal(0))
    return {
        "journal": "read",
        "calls": len(calls),
        "priced": len(priced),
        "unknown": unknown,
        "usd": money(usd),
        "worst_case_usd": money(worst_case_usd),
        "booked_usd": money(usd + unknown * Decimal(worst_case_usd)),
    }
