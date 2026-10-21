"""Composing and reporting on the SQL ``WHERE`` clauses the libraries read with.

The skill libraries take their filters as SQL clauses: from the model (a
tool's ``filter`` argument, a nested actor's discovery scope) and from the
runtime (a manager's ``filter_scope``, id exclusions). Clauses are combined
with ``AND`` here, and a clause SQLite rejects becomes a tool error that
names the columns the clause may use.

A scope is only a bound if the clauses joined with it cannot step outside
their own parentheses. ``(scope) AND (1=1) OR (1=1)`` is what joining the
clause ``1=1) OR (1=1`` would give, and a ``--`` comment would drop
everything after it, so every clause must be one self-contained
expression: balanced parentheses, closed quotes, and no comment or
statement separator outside a quoted string. Any other clause is refused
with :class:`UnsafeClauseError`, which callers already report as an
invalid filter.
"""

from __future__ import annotations

import sqlite3
from typing import Optional, Sequence

from .tool_outcome import ToolErrorException


class UnsafeClauseError(sqlite3.ProgrammingError):
    """A clause that is not one self-contained SQL expression."""


_CLOSING_QUOTE = {"'": "'", '"': '"', "`": "`", "[": "]"}


def _unsafe_reason(clause: str) -> Optional[str]:
    """Why *clause* could reach outside its own parentheses; ``None`` if it cannot."""
    depth = 0
    i = 0
    n = len(clause)
    while i < n:
        ch = clause[i]
        if ch in _CLOSING_QUOTE:
            close = _CLOSING_QUOTE[ch]
            j = i + 1
            while True:
                end = clause.find(close, j)
                if end == -1:
                    return "it has an unclosed quote"
                # A doubled quote is an escaped quote inside the string.
                if close != "]" and clause[end + 1 : end + 2] == close:
                    j = end + 2
                    continue
                break
            i = end + 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return "it closes a parenthesis it did not open"
        elif ch == ";":
            return "it contains a statement separator"
        elif clause.startswith("--", i) or clause.startswith("/*", i):
            return "it contains a comment"
        i += 1
    if depth:
        return "it leaves a parenthesis open"
    return None


def require_self_contained(clause: Optional[str]) -> Optional[str]:
    """Return *clause*, or raise :class:`UnsafeClauseError` if it is not one
    self-contained expression."""
    if clause:
        reason = _unsafe_reason(clause)
        if reason is not None:
            raise UnsafeClauseError(
                f"the clause {clause!r} is not one self-contained SQL "
                f"expression: {reason}",
            )
    return clause


def _checked(clauses: Sequence[Optional[str]]) -> list[str]:
    return [require_self_contained(clause) for clause in clauses if clause]


def and_clauses(*clauses: Optional[str]) -> Optional[str]:
    """Join the non-empty clauses with ``AND``; ``None`` when there are none.

    Raises :class:`UnsafeClauseError` for a clause that is not one
    self-contained expression.
    """
    parts = _checked(clauses)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return " AND ".join(f"({part})" for part in parts)


def or_clauses(*clauses: Optional[str]) -> Optional[str]:
    parts = _checked(clauses)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return " OR ".join(f"({part})" for part in parts)


def not_in(column: str, ids: Optional[Sequence[int]]) -> Optional[str]:
    """``column NOT IN (...)`` for the given ids; ``None`` when there are none."""
    if not ids:
        return None
    return f"{column} NOT IN ({', '.join(str(int(value)) for value in sorted(ids))})"


def invalid_filter_error(
    exc: Exception,
    filter: Optional[str],
    columns: Sequence[str],
) -> ToolErrorException:
    """Translate a clause SQLite rejected into an actionable tool error."""
    return ToolErrorException(
        {
            "error_kind": "invalid_filter",
            "message": (
                f"filter {filter!r} was rejected: {exc}. The filter is a SQL "
                f"WHERE clause (without the WHERE keyword) over the columns "
                f"{', '.join(columns)}."
            ),
            "details": {"filter": filter, "columns": list(columns)},
        },
    )


__all__ = [
    "UnsafeClauseError",
    "and_clauses",
    "invalid_filter_error",
    "not_in",
    "or_clauses",
    "require_self_contained",
]
