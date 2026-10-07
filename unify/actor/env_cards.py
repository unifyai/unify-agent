"""``UNIFY_ENV_CARDS``: verified facts about the environment a session will touch, shown when it starts.

Offline (research artifact memory-a-v1/env-memory-v1 and its amendment 1;
AppWorld everyday train-v1, 900 visits of lean-all and Unify @ main), about
80% of the agent's API calls are about the environment, not the job:
reading API documentation, repeating a call an earlier session had already
found to fail, working out how to log in. Listing what earlier sessions
verified at the start of a session would have removed 11% of turns and,
net of the listing's own cost, 11% of task USD -- on first visits to a job
as much as on returns. The same facts shown only when the agent first
touches an app saved nothing: the saving is in that first turn.

With the switch on, for a top-level task:

* **Recording.** An observer on the environment seam
  (:mod:`unify.function_manager.primitives.observers`) keeps one row per
  environment call -- ``primitives.<ns>.<method>`` or a raw environment
  global such as AppWorld's ``apis.spotify.login`` -- in
  ``<UNIFY_HOME>/env_cards.sqlite``: the call's dotted path, its argument
  *names*, whether it worked (or its error type and a masked message), and
  the *shape* of its response (type, and for a mapping up to
  :data:`MAX_KEYS` keys with their types). Argument values are never kept,
  except a string that is itself a name of the environment's surface (an
  app or method name, as a documentation lookup passes). A call whose
  method or argument names look like authentication keeps no response
  shape. Credentials and data are never stored.
* **Facts.** From earlier sessions only: per method, how it was called and
  what it returned when it worked (weighted by the number of sessions that
  called it successfully); per failed method, the call in the same group
  that then worked in the same session (weighted by the sessions it failed
  in).
* **Showing.** One section in the session's first message, for the groups
  (``apis.spotify``) used by the :data:`NEAREST` earlier sessions whose
  requests are closest to this one by embedding (the request's embedding
  is kept, never its text), and those used in at least half of the earlier
  sessions: at most
  :data:`K` facts per group, heaviest first. It informs; nothing is
  refused. Facts are keyed by the environment's own names, never by a task
  or a request; nothing is chosen by shared words. An embedding failure
  leaves only the share rule.

Off: no observer, no table, no section.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import logging
import re
import sqlite3
import uuid
from collections import defaultdict
from contextlib import closing
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

TABLE = "env_calls"
SESSIONS = "env_sessions"
K = 5
"""Facts shown per group, at most."""
MIN_SHARE = 0.5
"""A group used in at least this share of earlier sessions is shown without being named."""
MIN_SESSIONS = 2
"""Earlier sessions needed before the share rule applies."""
NEAREST = 3
"""The earlier sessions whose requests are closest (by embedding) to this one: the groups they used are shown."""
MAX_KEYS = 12
MAX_ERROR = 120
MAX_NAME = 40
KEEP_ROWS = 50_000
"""The table keeps this many calls, the latest."""

_AUTH = re.compile(
    r"login|logout|token|password|passwd|secret|credential|auth|api_?key",
    re.I,
)
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")
_DIGITS = re.compile(r"\d+")

HEADER = (
    "## Environment Notes\n\n"
    "How earlier sessions called the parts of the environment this request "
    "names or that most sessions use: argument names and the shape of what "
    "came back when a call worked, and calls that failed with the one that "
    "then worked. No values are kept.\n\n"
)


def enabled() -> bool:
    from unify.settings import SETTINGS

    return bool(getattr(SETTINGS, "UNIFY_ENV_CARDS", False))


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    from unify import db

    path = db.store_home() / "env_cards.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {SESSIONS} (session TEXT PRIMARY KEY,"
        " vec BLOB NOT NULL, created_at TEXT NOT NULL)",
    )
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {TABLE} (seq INTEGER PRIMARY KEY"
        " AUTOINCREMENT, session TEXT NOT NULL, grp TEXT NOT NULL,"
        " method TEXT NOT NULL, args TEXT NOT NULL, ok INTEGER NOT NULL,"
        " error TEXT, shape TEXT, created_at TEXT NOT NULL)",
    )
    return conn


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def split_path(namespace: str, method: str) -> Tuple[str, str]:
    """``(group, method)``: the dotted path minus its last name, and that name."""
    parts = [p for p in f"{namespace}.{method}".split(".") if p]
    if len(parts) < 2:
        return namespace, method
    return ".".join(parts[:-1]), parts[-1]


def _type(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float, str)):
        return type(value).__name__
    if isinstance(value, Mapping):
        return "dict"
    if isinstance(value, (list, tuple)):
        return "list"
    return type(value).__name__


def shape(value: Any) -> str:
    """The shape of a response: its type, and for a mapping (or a list of them) its keys with their types."""
    if isinstance(value, Mapping):
        keys = list(value.keys())[:MAX_KEYS]
        inner = ", ".join(f"{k}: {_type(value[k])}" for k in keys if isinstance(k, str))
        more = ", …" if len(value) > MAX_KEYS else ""
        return "{" + inner + more + "}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        return "[" + shape(value[0]) + ", …]"
    return _type(value)


def _masked_error(error: BaseException, values: Iterable[Any] = ()) -> str:
    """The error's type and message, with quoted text, numbers and the call's own values masked."""
    text = str(error)
    for value in values:
        if isinstance(value, str) and value.strip():
            text = text.replace(value, "…")
    text = _QUOTED.sub("'…'", text)
    text = _DIGITS.sub("#", text).strip().replace("\n", " ")
    return f"{type(error).__name__}: {text}"[:MAX_ERROR]


def _surface_name(value: Any, names: Iterable[str]) -> bool:
    return (
        isinstance(value, str)
        and len(value) <= MAX_NAME
        and bool(_IDENT.match(value))
        and value in names
    )


def describe_args(
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    names: Iterable[str],
) -> str:
    """Argument names (and surface names passed as values), never other values."""
    names = set(names)
    out: List[str] = []
    for value in args:
        out.append(repr(value) if _surface_name(value, names) else "…")
    for key, value in kwargs.items():
        out.append(f"{key}={value!r}" if _surface_name(value, names) else str(key))
    return ", ".join(out)


class Recorder:
    """The observer one session pushes: one row per environment call."""

    complete = False

    def __init__(self, session: str, names: Iterable[str] = ()) -> None:
        self.session = session
        self.names = set(names)

    def before(self, call: Any) -> None:
        return None

    def after(
        self,
        call: Any,
        *,
        result: Any,
        error: Optional[BaseException],
        intercepted: bool,
        started: float,
        elapsed_s: float,
    ) -> None:
        if intercepted:
            return  # not observed evidence
        group, method = split_path(str(call.namespace), str(call.method))
        self.names.update(p for p in group.split(".") + [method] if p)
        auth = bool(_AUTH.search(method)) or any(
            _AUTH.search(str(k)) for k in (call.kwargs or {})
        )
        args = describe_args(call.args or (), call.kwargs or {}, self.names)
        row = (
            self.session,
            group,
            method,
            args,
            0 if error is not None else 1,
            (
                _masked_error(
                    error,
                    list(call.args or ()) + list((call.kwargs or {}).values()),
                )
                if error is not None
                else None
            ),
            None if (error is not None or auth) else shape(result),
            _now(),
        )
        try:
            with closing(_connect()) as conn, conn:
                conn.execute(
                    f"INSERT INTO {TABLE} (session, grp, method, args, ok, error,"
                    " shape, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    row,
                )
                conn.execute(
                    f"DELETE FROM {TABLE} WHERE seq <= (SELECT MAX(seq) FROM {TABLE}) - ?",
                    (KEEP_ROWS,),
                )
        except Exception as exc:  # noqa: BLE001 - a note must never break a call
            logger.warning("env card call not kept: %s: %s", type(exc).__name__, exc)


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------


def _rows(exclude_session: str = "") -> List[Tuple]:
    try:
        with closing(_connect()) as conn:
            return conn.execute(
                f"SELECT seq, session, grp, method, args, ok, error, shape FROM {TABLE}"
                " WHERE session != ? ORDER BY seq",
                (exclude_session,),
            ).fetchall()
    except sqlite3.Error as exc:
        logger.warning("env cards not read: %s", exc)
        return []


def _embed(texts: List[str]) -> Any:
    from unify.common import embeddings

    return embeddings.embed(texts)


def _vector(text: str) -> Optional[bytes]:
    """The request's embedding as float32 bytes, or ``None`` (no text, or the embedder failed)."""
    if not text:
        return None
    try:
        import numpy as np

        vec = np.asarray(_embed([text]), dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        return (vec / norm).astype(np.float32).tobytes() if norm else None
    except Exception as exc:  # noqa: BLE001 - only the share rule, then
        logger.warning(
            "env cards: request not embedded: %s: %s",
            type(exc).__name__,
            exc,
        )
        return None


def _keep_session(session: str, vector: Optional[bytes]) -> None:
    if vector is None:
        return
    try:
        with closing(_connect()) as conn, conn:
            conn.execute(
                f"INSERT OR REPLACE INTO {SESSIONS} VALUES (?, ?, ?)",
                (session, vector, _now()),
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("env cards: session not kept: %s: %s", type(exc).__name__, exc)


def _nearest(vector: Optional[bytes], exclude_session: str) -> set:
    """The :data:`NEAREST` earlier sessions whose requests are closest to *vector* by cosine."""
    if vector is None:
        return set()
    try:
        import numpy as np

        with closing(_connect()) as conn:
            rows = conn.execute(
                f"SELECT session, vec FROM {SESSIONS} WHERE session != ?",
                (exclude_session,),
            ).fetchall()
        if not rows:
            return set()
        here = np.frombuffer(vector, dtype=np.float32)
        scored = []
        for session, blob in rows:
            other = np.frombuffer(blob, dtype=np.float32)
            if other.shape == here.shape:
                scored.append((float(here @ other), session))
        scored.sort(key=lambda item: -item[0])
        return {session for _, session in scored[:NEAREST]}
    except Exception as exc:  # noqa: BLE001
        logger.warning("env cards: nearest sessions not read: %s", exc)
        return set()


def facts(
    request: str,
    *,
    exclude_session: str = "",
    vector: Optional[bytes] = None,
) -> Dict[str, List[str]]:
    """Per shown group, its facts as lines, heaviest first (at most :data:`K`)."""
    rows = _rows(exclude_session)
    if not rows:
        return {}
    sessions = {r[1] for r in rows}
    used: Dict[str, set] = defaultdict(set)
    usage: Dict[Tuple[str, str], Dict[str, Any]] = {}
    usage_sessions: Dict[Tuple[str, str], set] = defaultdict(set)
    pitfalls: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    pitfall_sessions: Dict[Tuple[str, str, str], set] = defaultdict(set)
    open_failures: Dict[Tuple[str, str], List[Tuple]] = defaultdict(list)
    for seq, session, group, method, args, ok, error, shp in rows:
        used[group].add(session)
        if ok:
            usage[(group, method)] = {"args": args, "shape": shp, "seq": seq}
            usage_sessions[(group, method)].add(session)
            for f_method, f_args, f_error in open_failures.pop((session, group), []):
                if f_method == method and f_args == args:
                    continue
                key = (group, f_method, method)
                pitfalls[key] = {
                    "failed": f"{f_method}({f_args})",
                    "error": f_error,
                    "worked": f"{method}({args})",
                    "seq": seq,
                }
                pitfall_sessions[key].add(session)
        else:
            open_failures[(session, group)].append((method, args, error))
    near = _nearest(vector, exclude_session)
    shown = sorted(
        g
        for g in used
        if used[g] & near
        or (len(sessions) >= MIN_SESSIONS and len(used[g]) / len(sessions) >= MIN_SHARE)
    )
    out: Dict[str, List[str]] = {}
    for group in shown:
        candidates: List[Tuple[int, int, str]] = []
        for (g, method), info in usage.items():
            if g != group:
                continue
            line = f"`{group}.{method}({info['args']})`"
            if info["shape"]:
                line += f" → {info['shape']}"
            n = len(usage_sessions[(g, method)])
            candidates.append(
                (
                    n,
                    info["seq"],
                    f"{line} (worked in {n} session{'s' if n != 1 else ''})",
                ),
            )
        for (g, f_method, method), info in pitfalls.items():
            if g != group:
                continue
            n = len(pitfall_sessions[(g, f_method, method)])
            candidates.append(
                (
                    n,
                    info["seq"],
                    f"`{group}.{info['failed']}` failed ({info['error']}; "
                    f"{n} session{'s' if n != 1 else ''}), then "
                    f"`{group}.{info['worked']}` worked",
                ),
            )
        candidates.sort(key=lambda c: (-c[0], -c[1]))
        if candidates:
            out[group] = [c[2] for c in candidates[:K]]
    return out


def section(
    request: str,
    *,
    exclude_session: str = "",
    vector: Optional[bytes] = None,
) -> str:
    """The first-message section for *request*, or "" (off, or nothing verified yet)."""
    if not enabled() or not request:
        return ""
    found = facts(request, exclude_session=exclude_session, vector=vector)
    if not found:
        return ""
    blocks = [
        f"### {group}\n" + "\n".join(f"- {line}" for line in lines)
        for group, lines in found.items()
    ]
    return HEADER + "\n\n".join(blocks) + "\n"


# ---------------------------------------------------------------------------
# Session scope
# ---------------------------------------------------------------------------


def _surface_names() -> set:
    try:
        from unify.function_manager.primitives import environment

        names: set = set()
        for name, ns in environment.environment_namespaces().items():
            names.add(name)
            names.update(m.name for m in ns.methods)
        return names
    except Exception:  # noqa: BLE001 - names only widen what is kept
        return set()


def enter(request: str) -> Tuple[Optional[contextlib.ExitStack], str]:
    """Push this session's recorder; ``(scope, section)``. ``(None, "")`` when off.

    Called for a top-level task, before the task loop starts, so the loop's
    tasks inherit the recorder. The section is built from earlier sessions'
    calls only.
    """
    if not enabled():
        return None, ""
    from unify.function_manager.primitives import observers

    if any(isinstance(o, Recorder) for o in observers.current()):
        return None, ""  # a sub-agent: its calls are its caller's session's
    session = uuid.uuid4().hex
    vector = _vector(str(request or ""))
    text = section(request, exclude_session=session, vector=vector)
    _keep_session(session, vector)
    stack = contextlib.ExitStack()
    stack.enter_context(observers.observing(Recorder(session, _surface_names())))
    return stack, text


def leave(scope: Optional[contextlib.ExitStack]) -> None:
    if scope is None:
        return
    try:
        scope.close()
    except ValueError:  # entered in another context
        pass


__all__ = [
    "HEADER",
    "K",
    "Recorder",
    "describe_args",
    "enabled",
    "enter",
    "facts",
    "leave",
    "section",
    "shape",
    "split_path",
]
