"""``UNIFY_FUNCTION_SUMMARY``: a running summary of every recorded call of a stored function.

``UNIFY_FUNCTION_CASES`` keeps at most the latest three passing and three
failing calls of a function as replayable cases. Facts about how a function
behaves across many runs -- "accepted runs read 14-40 songs", "these
arguments varied, that one never did" -- need every accepted run, not the
latest three, or a job that ran twenty times reads as if it ran three.

With the switch on (and cases on), each recorded call of a stored function
also leaves one small row: for each environment endpoint the call reached,
how many times and how many items each answer held; for each argument, a
digest of its value; whether the trace is complete; and the request the call
ran under. Summaries are computed when read, over the calls whose session is
known to be accepted (the checker's outcome, else the storage review's
judgement), so an outcome that arrives after the call still counts.

Missing evidence reads "unknown", never "none": a summary over calls whose
traces are incomplete (an environment global, a cut trace) says so, and an
endpoint count is reported only over complete traces. Rows exist only from
this switch on, recorded under the rule that marks a caller incomplete when a
function it runs reads an environment global.

Nothing here is shown to the agent; readers of the summary (a later listing
fact) decide what to show. Off: no table, no row, no read.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from contextlib import closing
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

#: The tracing rule rows are recorded under: 2 marks a caller incomplete when
#: a function it runs reads an environment global.
TRACE_RULE = 2
#: Rows kept per function (the latest).
ROWS_KEPT = 200
#: Distinct argument values counted per parameter before "many".
DISTINCT_CAP = 50

_TABLE = """
CREATE TABLE IF NOT EXISTS function_runs (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    function_id INTEGER NOT NULL,
    source_hash TEXT,
    text_key TEXT,
    recorded_at TEXT,
    trace_rule INTEGER,
    trace_complete INTEGER,
    errored INTEGER,
    endpoints TEXT,
    args TEXT
)
"""


def enabled() -> bool:
    """``UNIFY_FUNCTION_SUMMARY`` with ``UNIFY_FUNCTION_CASES`` on."""
    from unify.settings import SETTINGS

    from . import store_cases

    return (
        bool(getattr(SETTINGS, "UNIFY_FUNCTION_SUMMARY", False))
        and store_cases.enabled()
    )


def _connect() -> sqlite3.Connection:
    from unify import db

    path = db.store_home() / "function_runs.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_TABLE)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS function_runs_fid ON function_runs(function_id)",
    )
    return conn


def item_count(value: Any) -> Optional[int]:
    """How many items an environment answer holds: a list's length, else the longest list among a dict's values."""
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, Mapping):
        lengths = [len(v) for v in value.values() if isinstance(v, (list, tuple))]
        return max(lengths) if lengths else None
    return None


def _digest(value: Any) -> str:
    try:
        text = json.dumps(value, sort_keys=True, default=repr)
    except (TypeError, ValueError):
        text = repr(value)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _endpoints(trace_calls: List[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for call in trace_calls:
        name = str(call.get("call") or "")
        if not name:
            continue
        entry = out.setdefault(name, {"calls": 0, "items": [], "uncounted": 0})
        entry["calls"] += 1
        if "result" in call:
            count = item_count(call.get("result"))
            if count is not None:
                entry["items"].append(count)
        elif "error" not in call:
            # The answer was too large or inexact to keep, so its size is unknown.
            entry["uncounted"] += 1
    return out


def _arguments(pending: Any) -> Optional[Dict[str, str]]:
    """``{parameter: digest}`` of the call's (redacted) arguments, positional ones keyed by position; ``None`` when they were not kept."""
    call = getattr(pending, "call", None)
    if not isinstance(call, Mapping):
        return None
    out: Dict[str, str] = {}
    for index, value in enumerate(call.get("args") or []):
        out[f"#{index}"] = _digest(value)
    for name, value in (call.get("kwargs") or {}).items():
        out[str(name)] = _digest(value)
    return out


def record(recorder: Any, pending: Any, *, error: Any = None) -> None:
    """Leave one row for a finished recorded call (no-op while off); never raises."""
    if not enabled() or pending is None:
        return
    try:
        from unify import db

        from . import task_origin

        trace = pending.trace
        calls = [
            c for c in (getattr(trace, "calls", None) or []) if isinstance(c, Mapping)
        ]
        text = task_origin.current_request()
        row = (
            int(recorder.function_id),
            hashlib.sha256(str(recorder.source).encode()).hexdigest(),
            task_origin.text_key(text) if text else None,
            db.now_iso(),
            TRACE_RULE,
            int(bool(trace.complete)),
            int(error is not None),
            json.dumps(_endpoints(calls), sort_keys=True),
            json.dumps(_arguments(pending), sort_keys=True),
        )
        with closing(_connect()) as conn, conn:
            conn.execute(
                "INSERT INTO function_runs (function_id, source_hash, text_key, recorded_at,"
                " trace_rule, trace_complete, errored, endpoints, args)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
            conn.execute(
                "DELETE FROM function_runs WHERE function_id = ? AND seq NOT IN ("
                "SELECT seq FROM function_runs WHERE function_id = ? ORDER BY seq DESC LIMIT ?)",
                (row[0], row[0], ROWS_KEPT),
            )
    except Exception as exc:  # noqa: BLE001 - a summary must never break a call
        logger.warning("run summary not recorded: %s: %s", type(exc).__name__, exc)


def _accepted(text_keys: List[str]) -> Dict[str, Optional[bool]]:
    """``{text_key: accepted?}`` from the request log's kept outcomes (the checker's over the review's); ``None`` when unknown."""
    from . import task_origin

    out: Dict[str, Optional[bool]] = {key: None for key in text_keys}
    path = task_origin.request_log_path()
    if not text_keys or not path.exists():
        return out
    try:
        with closing(task_origin._connect_outcomes(path)) as conn:
            for key in text_keys:
                rows = dict(
                    conn.execute(
                        "SELECT source, solved FROM request_outcomes WHERE text_key = ?",
                        (key,),
                    ).fetchall(),
                )
                for source in (task_origin.CHECKER, task_origin.REVIEW):
                    if source in rows:
                        out[key] = bool(rows[source])
                        break
    except sqlite3.Error as exc:
        logger.warning("request outcomes not read: %s", exc)
    return out


def summary(
    function_id: int,
    *,
    accepted_only: bool = True,
) -> Optional[Dict[str, Any]]:
    """The function's running summary, or ``None`` while off or with no rows.

    ``runs``: calls summarised; ``accepted``/``unknown``: their sessions'
    outcomes; ``complete``: whether every summarised trace is complete (if
    not, endpoint facts read "unknown"); ``endpoints``: per endpoint, calls
    per run (min, max) and items per answer (min, max) over complete traces;
    ``arguments``: per parameter, distinct values seen (capped) and runs,
    or "unknown" when some summarised call's arguments were not kept. Items
    read "unknown" for an endpoint when some answer was too large to keep.
    """
    if not enabled():
        return None
    try:
        with closing(_connect()) as conn:
            rows = conn.execute(
                "SELECT text_key, trace_complete, errored, endpoints, args FROM function_runs"
                " WHERE function_id = ? AND trace_rule >= ? ORDER BY seq",
                (int(function_id), TRACE_RULE),
            ).fetchall()
    except sqlite3.Error as exc:
        logger.warning("run summary not read: %s", exc)
        return None
    if not rows:
        return None
    outcomes = _accepted(sorted({r[0] for r in rows if r[0]}))
    kept = []
    unknown = 0
    for key, complete, errored, endpoints, args in rows:
        verdict = outcomes.get(key) if key else None
        if verdict is None:
            unknown += 1
        if accepted_only and verdict is not True:
            continue
        kept.append(
            (
                bool(complete),
                bool(errored),
                json.loads(endpoints),
                json.loads(args or "null"),
            ),
        )
    out: Dict[str, Any] = {
        "runs": len(kept),
        "unknown_outcome": unknown,
        "complete": bool(kept) and all(c for c, _, _, _ in kept),
        "endpoints": "unknown",
        "arguments": {},
    }
    if not kept:
        return out
    if out["complete"]:
        names = sorted({name for _, _, endpoints, _ in kept for name in endpoints})
        facts: Dict[str, Any] = {}
        for name in names:
            per_run = [
                endpoints.get(name, {}).get("calls", 0) for _, _, endpoints, _ in kept
            ]
            items = [
                n
                for _, _, endpoints, _ in kept
                for n in endpoints.get(name, {}).get("items", [])
            ]
            uncounted = sum(
                endpoints.get(name, {}).get("uncounted", 0)
                for _, _, endpoints, _ in kept
            )
            facts[name] = {
                "calls_per_run": [min(per_run), max(per_run)],
                "items": (
                    "unknown"
                    if uncounted
                    else ([min(items), max(items)] if items else None)
                ),
            }
        out["endpoints"] = facts
    if any(args is None for _, _, _, args in kept):
        out["arguments"] = "unknown"
        return out
    params = sorted({p for _, _, _, args in kept for p in args})
    for param in params:
        values = {args[param] for _, _, _, args in kept if param in args}
        out["arguments"][param] = {
            "distinct": (
                len(values) if len(values) < DISTINCT_CAP else f"{DISTINCT_CAP}+"
            ),
            "runs": sum(1 for _, _, _, args in kept if param in args),
        }
    return out


__all__ = ["ROWS_KEPT", "TRACE_RULE", "enabled", "item_count", "record", "summary"]
