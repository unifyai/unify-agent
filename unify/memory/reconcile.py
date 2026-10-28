"""Pricing a library-mode pass's model calls after the fact, by lane and window.

Passes run one at a time (:mod:`.integration.async_pass`) and the writer has its own proxy lane, so a pass's calls
are the writer-lane requests of the proxy journal (``UNIFY_MEMORY_WRITER_JOURNAL``) that *started* inside the
pass's window.
The journal is append-only and carries no clock, so the window is a byte range: the journal's size when the pass
started and when its supervisor recorded its end (:func:`journal_offset`). A request belongs to the pass when its
``request_started`` row lies in that range; its terminal row is read wherever it was appended, so a call cancelled
at the wall bound, whose terminal row the proxy writes late, is still found.

A request is priced as the key proxy books a writer call: by its last ``account_charge``; else, when the run pins
the writer's prices (``UNIFY_MEMORY_WRITER_PRICES_USD_PER_MILLION``) and the journal shows the request's tokens,
``input x p_input + output x p_output (+ cache_write x p_cache_write)`` per million (:func:`estimate`, counted as
``estimated``); else it stays ``unknown``, never zero, and is booked at the worst case the caller gives for the
writer's model (``worst_case_usd``, required: the pinned catalogue's tier-1 bound for one request of that model,
:mod:`.call_bounds`; there is no default, so no model is ever booked at another's bound). No header is sent and no generation lookup is made: the proxy
mints its own attempt id and the writer lane refuses ``GET /generation``. Money is a decimal string.
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
    """One writer-lane request the pass started: its attempt id (the proxy's), its last status, its charge."""

    attempt_id: str
    status: str = "started"
    usd: str = UNKNOWN
    generation_id: str | None = None
    #: the request's token counts from its last row that has any (input, output, cache_write …), else None
    tokens: dict | None = None

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
    """The writer-lane requests whose ``request_started`` row lies in bytes ``[start, end)``, each with its LAST row's
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
        tokens = row.get("tokens")
        if isinstance(tokens, dict) and any(v is not None for v in tokens.values()):
            c.tokens = tokens
    return calls


def estimate(tokens: Any, prices: dict[str, Decimal] | None) -> Decimal | None:
    """A call's USD from its tokens at the pinned *prices* (per million tokens), as the key proxy estimates an
    unpriced writer call; None without prices, without input and output counts, or with an unreadable count.
    """
    if not prices or not isinstance(tokens, dict):
        return None
    t_in, t_out = tokens.get("input"), tokens.get("output")
    if t_in is None and t_out is None:
        return None
    try:
        counts = {
            k: int(tokens.get(k) or 0) for k in ("input", "output", "cache_write")
        }
    except (TypeError, ValueError):
        return None
    if any(n < 0 for n in counts.values()):
        return None
    usd = prices["input"] * counts["input"] + prices["output"] * counts["output"]
    if "cache_write" in prices:
        usd += prices["cache_write"] * counts["cache_write"]
    return usd / Decimal(1000000)


def unavailable(worst_case_usd: Decimal | None) -> dict:
    """The summary when the journal cannot be read: nothing priced, nothing known, the whole cap kept (*worst_case_usd*
    as the caller has it for the model, None when it has none)."""
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
    worst_case_usd: Decimal,
    budget_s: float = RECONCILE_S,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    prices: dict[str, Decimal] | None = None,
) -> dict:
    """The pass's writer calls priced from the journal, re-read until each started call has a terminal row or
    *budget_s* has passed. Returns the summary the pass's end event and the run guard's settle line carry:

    ``journal`` (``read`` or ``unavailable``: no path, no window, or unreadable; then nothing is priced and the run
    guard keeps the pass's whole cap), ``calls``, ``priced`` (by charge), ``estimated`` (from tokens at *prices*),
    ``estimated_usd``, ``unknown``, ``usd`` (the charged and estimated sum), and ``booked_usd`` = ``usd`` +
    ``unknown`` x *worst_case_usd* (the worst case the run guard books).
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
    estimates = [
        e
        for c in calls.values()
        if c.usd == UNKNOWN and (e := estimate(c.tokens, prices)) is not None
    ]
    unknown = len(calls) - len(priced) - len(estimates)
    estimated_usd = sum(estimates, Decimal(0))
    usd = sum((Decimal(c.usd) for c in priced), Decimal(0)) + estimated_usd
    return {
        "journal": "read",
        "calls": len(calls),
        "priced": len(priced),
        "estimated": len(estimates),
        "estimated_usd": money(estimated_usd),
        "unknown": unknown,
        "usd": money(usd),
        "worst_case_usd": money(worst_case_usd),
        "booked_usd": money(usd + unknown * Decimal(worst_case_usd)),
    }
