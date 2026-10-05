"""The assistant's persistence layer: a few concrete tables in one SQLite file.

One user, one assistant, one store. The tables are the things the harness
keeps: stored ``functions`` and the read-only ``primitives`` catalogue seeded
from the code (read together through the ``all_functions`` view), the user's
``guidance`` and the ``builtin_guidance`` seeded from the committed snapshot
(read together through ``all_guidance``), and the chat ``messages``.
With ``UNIFY_FUNCTION_PATCH`` on, ``function_history`` and
``guidance_history`` keep each row as it was before an overwrite; nothing
removes their rows, :func:`clear` included. With ``UNIFY_STORE_TRUST``
on, ``function_trust`` holds one trust record per stored function; deleting
the function deletes it. With ``UNIFY_FUNCTION_CASES`` on,
``function_cases`` holds a few recorded calls per stored function; deleting
the function deletes them too.

Managers issue SQL through :func:`execute`, :func:`query` and
:func:`query_one`. Clauses written by the model run through
:func:`query_readonly`, which refuses anything but reads. List and dict
columns hold JSON text; :func:`dumps` / :func:`loads` convert.

The file lives at ``UNIFY_STORE_PATH``, else ``<UNIFY_HOME>/store.sqlite``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS functions (
    function_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    argspec TEXT NOT NULL DEFAULT '',
    docstring TEXT NOT NULL DEFAULT '',
    implementation TEXT NOT NULL,
    depends_on TEXT NOT NULL DEFAULT '[]',
    stale_reasons TEXT NOT NULL DEFAULT '[]',
    precondition TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    dependencies TEXT NOT NULL DEFAULT '[]',
    third_party_imports TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    usage_calls INTEGER NOT NULL DEFAULT 0,
    usage_last_called_at TEXT,
    usage_recent_calls TEXT NOT NULL DEFAULT '[]',
    usage_search_hits INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS primitives (
    function_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    argspec TEXT NOT NULL DEFAULT '',
    docstring TEXT NOT NULL DEFAULT '',
    primitive_class TEXT NOT NULL,
    primitive_method TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE VIEW IF NOT EXISTS all_functions AS
    SELECT function_id, name, argspec, docstring, implementation, depends_on,
           stale_reasons, precondition, metadata, dependencies,
           third_party_imports, created_at,
           usage_calls, usage_last_called_at, usage_recent_calls,
           usage_search_hits, 0 AS is_primitive,
           NULL AS primitive_class, NULL AS primitive_method
    FROM functions
    UNION ALL
    SELECT function_id, name, argspec, docstring, NULL, '[]', '[]', NULL,
           metadata, '[]', '[]', NULL, 0, NULL, '[]', 0, 1,
           primitive_class, primitive_method
    FROM primitives;
CREATE TABLE IF NOT EXISTS guidance (
    guidance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    function_ids TEXT NOT NULL DEFAULT '[]',
    stale_reasons TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS builtin_guidance (
    guidance_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL
);
CREATE VIEW IF NOT EXISTS all_guidance AS
    SELECT guidance_id, title, content, function_ids, stale_reasons,
           0 AS is_builtin
    FROM guidance
    UNION ALL
    SELECT guidance_id, title, content, '[]', '[]', 1 FROM builtin_guidance;
CREATE TABLE IF NOT EXISTS function_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    function_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    previous TEXT NOT NULL,
    reason TEXT NOT NULL,
    replaced_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS guidance_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    guidance_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    previous TEXT NOT NULL,
    reason TEXT NOT NULL,
    replaced_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS function_trust (
    function_id INTEGER PRIMARY KEY
        REFERENCES functions(function_id) ON DELETE CASCADE,
    state TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    dependency_hash TEXT NOT NULL,
    effect_class TEXT NOT NULL,
    passes INTEGER NOT NULL DEFAULT 0,
    failures INTEGER NOT NULL DEFAULT 0,
    input_hashes TEXT NOT NULL DEFAULT '[]',
    distinct_inputs INTEGER NOT NULL DEFAULT 0,
    clean_uses INTEGER NOT NULL DEFAULT 0,
    last_failure TEXT,
    updated_at TEXT NOT NULL,
    failure_hashes TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS function_cases (
    case_id INTEGER PRIMARY KEY AUTOINCREMENT,
    function_id INTEGER NOT NULL
        REFERENCES functions(function_id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    args_hash TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    call TEXT,
    args_shown TEXT NOT NULL DEFAULT '',
    result TEXT,
    error TEXT,
    trace TEXT NOT NULL DEFAULT '[]',
    trace_complete INTEGER NOT NULL DEFAULT 1,
    session TEXT,
    outcome TEXT,
    retired_why TEXT,
    retired_at TEXT,
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS function_cases_by_function
    ON function_cases (function_id, status, kind);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    attachments TEXT NOT NULL DEFAULT '[]'
);
"""

# Tables that hold the user's own rows; the seeded catalogues are not listed,
# nor the history tables, which are append-only.
USER_TABLES = ("functions", "guidance", "messages")

FUNCTION_COLUMNS = (
    "function_id",
    "name",
    "argspec",
    "docstring",
    "implementation",
    "depends_on",
    "stale_reasons",
    "precondition",
    "metadata",
    "dependencies",
    "third_party_imports",
    "created_at",
    "usage_calls",
    "usage_last_called_at",
    "usage_recent_calls",
    "usage_search_hits",
    "is_primitive",
    "primitive_class",
    "primitive_method",
)
FUNCTION_JSON_COLUMNS = (
    "depends_on",
    "stale_reasons",
    "precondition",
    "metadata",
    "dependencies",
    "third_party_imports",
    "usage_recent_calls",
)
GUIDANCE_COLUMNS = (
    "guidance_id",
    "title",
    "content",
    "function_ids",
    "stale_reasons",
    "is_builtin",
)
GUIDANCE_JSON_COLUMNS = ("function_ids", "stale_reasons")

_READ_ONLY_ACTIONS = frozenset(
    {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION},
)


class ReadOnlyViolation(sqlite3.Error):
    """A clause supplied for a read attempted something other than reading."""


def store_home() -> Path:
    """Directory holding the store file, the workspace and the logs."""
    raw = os.environ.get("UNIFY_HOME", "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".unify"


def store_path() -> str:
    explicit = os.environ.get("UNIFY_STORE_PATH", "").strip()
    if explicit:
        return explicit
    return str(store_home() / "store.sqlite")


def utc_now() -> datetime:
    """The store's clock: the current UTC time, timezone-aware.

    Every timestamp the store writes (function and guidance rows, usage
    traces, history, trust and case records) reads this one function, so a
    test can freeze it in one place.
    """
    return datetime.now(timezone.utc)


def now_iso() -> str:
    return utc_now().isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def dumps(value: Any) -> str:
    """Encode a list or dict column value."""
    return json.dumps(value, default=_json_default, sort_keys=True)


def loads(text: str | None) -> Any:
    """Decode a JSON column value; ``NULL`` stays ``None``."""
    return None if text is None else json.loads(text)


def decode(row: dict[str, Any], json_columns: Sequence[str]) -> dict[str, Any]:
    """Decode the JSON columns of one row in place and return it."""
    for column in json_columns:
        if column in row and isinstance(row[column], str):
            row[column] = json.loads(row[column])
    return row


class _Connection:
    """The process-wide connection plus the lock and transaction depth."""

    def __init__(self, path: str) -> None:
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            path,
            check_same_thread=False,
            isolation_level=None,
        )
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.depth = 0
        if path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        _add_missing_columns(self.conn)


_ADDED_COLUMNS = (
    ("function_trust", "failure_hashes", "TEXT NOT NULL DEFAULT '[]'"),
    # UNIFY_GUIDANCE_ORIGIN: the requests a guidance entry was written for
    # (JSON, NULL when none were recorded). Not in ``all_guidance``.
    ("guidance", "origin", "TEXT"),
)
"""Columns added to a table after it first shipped: a store created earlier gets them on open."""


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    for table, column, declaration in _ADDED_COLUMNS:
        present = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in present:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


_STATE: _Connection | None = None
_STATE_LOCK = threading.Lock()


def _state() -> _Connection:
    global _STATE
    with _STATE_LOCK:
        if _STATE is None or _STATE.path != store_path():
            if _STATE is not None:
                _STATE.conn.close()
            _STATE = _Connection(store_path())
        return _STATE


def connect() -> sqlite3.Connection:
    """The process-wide connection, opened on first use at :func:`store_path`."""
    return _state().conn


def reset_store() -> None:
    """Close the process-wide connection so the next call reopens it."""
    global _STATE
    with _STATE_LOCK:
        if _STATE is not None:
            _STATE.conn.close()
            _STATE = None


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    """Run statements atomically; nested uses join the outermost transaction."""
    state = _state()
    with state.lock:
        outermost = state.depth == 0
        if outermost:
            state.conn.execute("BEGIN IMMEDIATE")
        state.depth += 1
        try:
            yield state.conn
        except BaseException:
            state.depth -= 1
            if outermost:
                state.conn.execute("ROLLBACK")
            raise
        else:
            state.depth -= 1
            if outermost:
                state.conn.execute("COMMIT")


def execute(sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> sqlite3.Cursor:
    """Run one statement inside a transaction and return its cursor."""
    with transaction() as conn:
        return conn.execute(sql, params)


def executemany(sql: str, rows: Sequence[Sequence[Any]]) -> None:
    with transaction() as conn:
        conn.executemany(sql, rows)


def query(
    sql: str,
    params: Sequence[Any] | dict[str, Any] = (),
) -> list[dict[str, Any]]:
    """Run a read and return every row as a dict."""
    state = _state()
    with state.lock:
        return [dict(row) for row in state.conn.execute(sql, params).fetchall()]


def query_one(
    sql: str,
    params: Sequence[Any] | dict[str, Any] = (),
) -> dict[str, Any] | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def query_readonly(
    sql: str,
    params: Sequence[Any] | dict[str, Any] = (),
) -> list[dict[str, Any]]:
    """Run a read whose text may have been written by the model.

    An authorizer refuses every action other than reading for the duration
    of the statement, so a clause that tries to write, alter or attach
    raises :class:`ReadOnlyViolation` instead of touching the store.
    """
    state = _state()

    def authorize(action: int, *_: Any) -> int:
        return (
            sqlite3.SQLITE_OK if action in _READ_ONLY_ACTIONS else sqlite3.SQLITE_DENY
        )

    with state.lock:
        state.conn.set_authorizer(authorize)
        try:
            return [dict(row) for row in state.conn.execute(sql, params).fetchall()]
        except sqlite3.DatabaseError as exc:
            if "not authorized" in str(exc):
                raise ReadOnlyViolation(str(exc)) from exc
            raise
        finally:
            state.conn.set_authorizer(None)


def clear() -> None:
    """Delete every row the user owns and restart their id sequences.

    The seeded catalogues (``primitives``, ``builtin_guidance``) are kept.
    """
    with transaction() as conn:
        for table in USER_TABLES:
            conn.execute(f"DELETE FROM {table}")
            conn.execute("DELETE FROM sqlite_sequence WHERE name = ?", (table,))
        # UNIFY_ENTRY_RECORD: the link table, which only that switch creates.
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'entry_links'",
        ).fetchone():
            conn.execute("DELETE FROM entry_links")


__all__ = [
    "FUNCTION_COLUMNS",
    "FUNCTION_JSON_COLUMNS",
    "GUIDANCE_COLUMNS",
    "GUIDANCE_JSON_COLUMNS",
    "ReadOnlyViolation",
    "SCHEMA",
    "USER_TABLES",
    "clear",
    "connect",
    "decode",
    "dumps",
    "execute",
    "executemany",
    "loads",
    "now_iso",
    "query",
    "query_one",
    "query_readonly",
    "reset_store",
    "store_home",
    "store_path",
    "transaction",
    "utc_now",
]
