"""``UNIFY_FUNCTION_VALUE_NOTICE``: say when a value a stored function matches on no longer occurs in what it read.

A reused function can run cleanly on a value that no longer matches its data:
the request says "meal" where the export says "meals", a function keeps last
month's error codes, a yes/no column is now Y/N. The answer comes out empty or
short and reads as plausible.

With the switch on (with ``UNIFY_FUNCTION_SUMMARY``), while a recorded call of
a stored function runs, an audit hook notes the files it opens for reading.
When it returns, a bounded scan of exactly those files counts, as whole values
(a table cell, a JSON string or key, a whole word in text; never a substring),
each value the call matches on: its short string arguments, and the string
literals its code compares against (``==``, ``!=``, ``in``, ``.isin``, and
literal lists it assigns). The running summary keeps, per value's key
(``arg:<parameter>`` or ``lit:<text>``), the count and the table columns it was
found in. Argument values themselves are never kept, and credential arguments
are never watched.

A value whose key matched in every earlier call of the same source whose
request was accepted (at least one), and that now occurs 0 times, gets one
plain line after the call, with the earlier counts and, for a table column of
at most ``WATCH_DISTINCT`` distinct values, the closest values in it now
(same after case, whitespace or a plural "s", or within two edits). The line
asks for nothing.

Unknown, never zero: no line when the scan was cut short, the call started a
process (a shell pipeline reads where the hook cannot see), or it read no
files (data from an API).

This checks after the call, on the files the call itself read: a recurring job
meets new workspaces and newly named exports, so the files read by earlier
calls say nothing reliable about which file this call will read. What is given
up is a warning before a write that used the stale value.

The hook and the scan run in the same process as the function, which can open
files where the hook does not look or alter what it records: the line informs
and nothing may enforce on it. Off: no hook, no scan, no row field, no line.
"""

from __future__ import annotations

import ast
import contextlib
import json
import logging
import os
from contextlib import closing
from typing import Any, Dict, Iterator, List, Mapping, Optional

logger = logging.getLogger(__name__)

#: Longest string argument or literal watched.
MAX_VALUE_CHARS = 64
#: Literals watched per function.
MAX_LITERALS = 20
#: Accepted earlier calls the value must have matched in.
MIN_ACCEPTED = 1


def enabled() -> bool:
    """``UNIFY_FUNCTION_VALUE_NOTICE`` with the running summary on."""
    from unify.settings import SETTINGS

    from . import run_summary

    return bool(getattr(SETTINGS, "UNIFY_FUNCTION_VALUE_NOTICE", False)) and (
        run_summary.enabled()
    )


def _watchable(value: Any) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value.strip()) <= MAX_VALUE_CHARS
        and "\n" not in value
        and os.sep not in value
    )


def literals(source: Any, name: str) -> List[str]:
    """The string literals the function ``name`` in ``source`` matches values against."""
    try:
        tree = ast.parse(str(source or ""))
    except SyntaxError:
        return []
    node = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
        ),
        None,
    )
    if node is None:
        return []
    found: List[str] = []

    def strings(expr: ast.AST) -> List[str]:
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return [expr.value]
        if isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
            return [
                e.value
                for e in expr.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            ]
        if isinstance(expr, ast.Dict):
            return [
                k.value
                for k in expr.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            ]
        return []

    for sub in ast.walk(node):
        if isinstance(sub, ast.Compare) and all(
            isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)) for op in sub.ops
        ):
            for operand in (sub.left, *sub.comparators):
                found.extend(strings(operand))
        elif (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "isin"
            and sub.args
        ):
            found.extend(strings(sub.args[0]))
        elif isinstance(sub, (ast.Assign, ast.AnnAssign)) and isinstance(
            sub.value,
            (ast.List, ast.Tuple, ast.Set, ast.Dict),
        ):
            items = strings(sub.value)
            if items and len(items) == len(
                getattr(sub.value, "elts", None) or getattr(sub.value, "keys", []),
            ):
                found.extend(items)
    return [v for v in dict.fromkeys(found) if _watchable(v)][:MAX_LITERALS]


class Plan:
    """What one recorded call watches: ``values`` by key, and the history of each key over accepted calls."""

    __slots__ = ("values", "history", "columns", "result")

    def __init__(self, values: Dict[str, str]) -> None:
        self.values = values
        #: key -> the counts in accepted earlier calls (all non-zero when the key carries matching).
        self.history: Dict[str, List[int]] = {}
        #: key -> table columns it matched in, in accepted earlier calls.
        self.columns: Dict[str, List[str]] = {}
        #: The scan of this call's reads: worker_child.count_values plus ``shelled``.
        self.result: Optional[dict] = None

    def wanted_columns(self) -> List[str]:
        return sorted({c for cols in self.columns.values() for c in cols})


def plan(
    recorder: Any,
    pending: Any,
    args: Any,
    kwargs: Mapping[str, Any],
) -> Optional[Plan]:
    """The values a call watches and their accepted history, or ``None`` (off, nothing to watch, any error)."""
    if not enabled() or pending is None:
        return None
    try:
        from .store_cases import _credential_name
        from .store_trust import source_signature

        values: Dict[str, str] = {}
        signature = source_signature(recorder.source, recorder.name)
        if signature is not None:
            try:
                named = dict(signature.bind_partial(*args, **kwargs).arguments)
            except TypeError:
                named = {}
            for param, value in named.items():
                if (
                    _watchable(value)
                    and not _credential_name(param)
                    and pending.redactor.scrub(value) == value
                ):
                    values[f"arg:{param}"] = value
        for literal in literals(recorder.source, recorder.name):
            values.setdefault(f"lit:{literal}", literal)
        if not values:
            return None
        out = Plan(values)
        _history(recorder, out)
        return out
    except Exception as exc:  # noqa: BLE001 - a notice must never break a call
        logger.warning("value watch not planned: %s: %s", type(exc).__name__, exc)
        return None


def _history(recorder: Any, out: Plan) -> None:
    from . import run_summary

    with closing(run_summary._connect()) as conn:
        rows = conn.execute(
            "SELECT text_key, inputs FROM function_runs WHERE function_id = ? AND source_hash = ?"
            " AND trace_rule >= ? AND errored = 0",
            (
                int(recorder.function_id),
                run_summary._source_hash(recorder),
                run_summary.TRACE_RULE,
            ),
        ).fetchall()
    outcomes = run_summary._accepted(sorted({r[0] for r in rows if r[0]}))
    accepted = [
        json.loads(r[1]) if r[1] else None
        for r in rows
        if r[0] and outcomes.get(r[0]) is True
    ]
    for key in out.values:
        counts: List[int] = []
        columns: set = set()
        for inputs in accepted:
            seen = (
                (inputs or {}).get("keys", {}).get(key)
                if isinstance(inputs, dict)
                else None
            )
            usable = (
                isinstance(inputs, dict)
                and inputs.get("files")
                and not inputs.get("sampled")
                and not inputs.get("shelled")
            )
            if not usable or not isinstance(seen, dict):
                counts = []
                break
            counts.append(int(seen.get("n") or 0))
            columns.update(seen.get("where") or [])
        if len(counts) >= MIN_ACCEPTED and all(n > 0 for n in counts):
            out.history[key] = counts
            out.columns[key] = sorted(columns)


@contextlib.contextmanager
def watching(pending: Any) -> Iterator[None]:
    """Watch the files the block reads and, when it returns, scan them for the planned values (in-process calls)."""
    watch_plan = getattr(pending, "value_plan", None) if pending is not None else None
    if watch_plan is None:
        yield
        return
    from unify.actor.execution import worker_child

    with worker_child.watching_inputs() as watch:
        yield
    scan(watch_plan, watch.files, watch.shelled)


def scan(watch_plan: Plan, files: List[str], shelled: bool) -> None:
    """Fill ``watch_plan.result`` from a call's reads; never raises."""
    try:
        from unify.actor.execution import worker_child

        result = worker_child.count_values(
            list(files),
            list(watch_plan.values.values()),
            tuple(watch_plan.wanted_columns()),
        )
        result["shelled"] = bool(shelled)
        watch_plan.result = result
    except Exception as exc:  # noqa: BLE001 - a notice must never break a call
        logger.warning("value watch not scanned: %s: %s", type(exc).__name__, exc)


def worker_request(pending: Any) -> Optional[dict]:
    """What the worker child needs to scan a call it runs: the values and the columns."""
    watch_plan = getattr(pending, "value_plan", None) if pending is not None else None
    if watch_plan is None:
        return None
    return {
        "values": list(watch_plan.values.values()),
        "columns": watch_plan.wanted_columns(),
    }


def accept_worker_result(pending: Any, result: Any) -> None:
    """Take the worker child's scan of a call it ran."""
    watch_plan = getattr(pending, "value_plan", None) if pending is not None else None
    if watch_plan is not None and isinstance(result, dict):
        watch_plan.result = dict(result)


def row_field(pending: Any) -> Optional[dict]:
    """The running summary's ``inputs`` field for this call: files, bounds and, per key, count and columns."""
    watch_plan = getattr(pending, "value_plan", None) if pending is not None else None
    result = watch_plan.result if watch_plan is not None else None
    if not isinstance(result, dict):
        return None
    counts = result.get("counts") or {}
    where = result.get("where") or {}
    return {
        "files": list(result.get("files") or []),
        "sampled": bool(result.get("sampled")),
        "shelled": bool(result.get("shelled")),
        "keys": {
            key: {"n": int(counts.get(value, 0)), "where": list(where.get(value) or [])}
            for key, value in watch_plan.values.items()
        },
    }


def _normal(text: str) -> str:
    text = text.strip().casefold()
    return text[:-1] if text.endswith("s") and len(text) > 3 else text


def _within_two_edits(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > 2:
        return False
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)),
            )
        if min(current) > 2:
            return False
        previous = current
    return previous[-1] <= 2


def closest(value: str, distinct: Mapping[str, int]) -> List[tuple]:
    """Values in ``distinct`` that read as ``value`` up to case, whitespace, a plural "s" or two edits."""
    target = _normal(value)
    near = [
        (other, n)
        for other, n in distinct.items()
        if other != value
        and other.strip()
        and (
            _normal(other) == target
            or _within_two_edits(other.casefold(), value.casefold())
        )
    ]
    return sorted(near, key=lambda item: (-item[1], item[0]))[:3]


def notice(recorder: Any, pending: Any) -> Optional[str]:
    """The line for a value that matched in every accepted earlier call and now occurs 0 times; else ``None``."""
    watch_plan = getattr(pending, "value_plan", None) if pending is not None else None
    result = watch_plan.result if watch_plan is not None else None
    if (
        not isinstance(result, dict)
        or not result.get("files")
        or result.get("sampled")
        or result.get("shelled")
    ):
        return None
    counts = result.get("counts") or {}
    distinct = result.get("distinct") or {}
    parts: List[str] = []
    for key, earlier in watch_plan.history.items():
        value = watch_plan.values[key]
        if counts.get(value, 1) != 0:
            continue
        what = (
            f"{key[4:]}={value!r}"
            if key.startswith("arg:")
            else f"the literal {value!r} in its code"
        )
        span = (
            f"{earlier[0]}"
            if len(set(earlier)) == 1
            else f"{min(earlier)}-{max(earlier)}"
        )
        calls = (
            "its one earlier accepted call"
            if len(earlier) == 1
            else f"its {len(earlier)} earlier accepted calls"
        )
        part = f"{what} occurs 0 times as a whole value in the files this call read; in {calls} it occurred {span} times."
        offers = []
        for column in watch_plan.columns.get(key, []):
            near = closest(value, distinct.get(column) or {})
            if near:
                offers.append(
                    f"column {column}: " + ", ".join(f"{v!r} ({n})" for v, n in near),
                )
        if offers:
            part += " Closest values now in " + "; ".join(offers) + "."
        parts.append(part)
    if not parts:
        return None
    return f"[{recorder.name}: " + " ".join(parts) + "]"


__all__ = [
    "MIN_ACCEPTED",
    "Plan",
    "accept_worker_result",
    "closest",
    "enabled",
    "literals",
    "notice",
    "plan",
    "row_field",
    "scan",
    "watching",
    "worker_request",
]
