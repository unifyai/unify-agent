"""The whole trajectory of one request, read from its session transcript (integration Task 23).

The unit of experience is the trajectory, not the environment calls. :func:`assemble` builds one
:class:`..episodes.Episode` from:

* the session transcript ``<UNIFY_HOME>/transcripts/<episode_id>.jsonl`` (the line format of
  :mod:`unify.transcripts`: ``{"seq", "ts", "session", "type", ...}``, where ``type`` is
  ``session_start``, ``system_prompt``, ``message``, ``message_update`` or ``compaction``, and a
  ``message``/``message_update`` line carries the chat message under ``"message"``): every user message
  (the request, then the observations), the replies, and every ``execute_code`` cell with its code,
  language and output;
* the tool actions the request's observer recorded (:mod:`.adapters.tool`), each attributed to the cell
  whose time span holds the call's start;
* actions other adapters made (``extra_actions``, e.g. the work-tree rows) and the work-tree snapshots.

A tool result that first went to disk as a placeholder is completed by a later ``message_update`` line
(same message, same ``tool_call_id``); :func:`fold` keeps the last version at the first position.
``Episode.transcript`` keeps the raw lines, so nothing is lost; the returned :class:`..redact.Redactor`
runs over everything when the episode is written.

Times are seconds since the epoch on the harness's wall clock: a transcript ``ts`` is
``datetime.now(UTC).isoformat()`` and :class:`TimedObserver` stamps calls with ``time.time()``, in the
same process, so a shifted clock (faketime) shifts both alike.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..analysis.cells import _raw, _stdout
from ..episodes import Action, Cell, Episode
from ..fingerprint import fingerprint
from ..redact import _MIN_SECRET_LEN, _SECRET_NAME_PARTS, Redactor
from .adapters.tool import RecordingObserver, action_fingerprints

__all__ = [
    "TimedCell",
    "TimedObserver",
    "assemble",
    "cell_at",
    "dialogue_text",
    "fold",
    "learned_secrets",
    "read_jsonl",
    "timed_cells",
]

_STDERR = "\n--- stderr ---\n"
_LEARNED_PARTS = _SECRET_NAME_PARTS  # KEY, TOKEN, SECRET, PASSWORD, CREDENTIAL
_MAX_DEPTH = 32


@dataclass
class TimedCell:
    """One ``execute_code`` cell and when it ran (epoch seconds; ``nan`` when a time was unreadable)."""

    cell: Cell
    call_id: str
    start: float
    end: float


def read_jsonl(path: Path) -> list[dict]:
    """The JSON object lines of *path*; an unreadable line is skipped, a missing file is ``[]``."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return []
    out = []
    for ln in text.splitlines():
        if not ln.strip():
            continue
        try:
            row = json.loads(ln)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _seconds(ts: Any) -> float:
    if not isinstance(ts, str):
        return math.nan
    try:
        when = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return math.nan
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.UTC)
    return when.timestamp()


def fold(lines: list[dict]) -> list[dict]:
    """The ``message`` lines in order, each tool result replaced by its final version.

    A later ``message_update`` (or a later ``message``) for the same ``tool_call_id`` replaces the tool
    result at its first position, with ``end_ts`` set to the later line's ``ts``. Other updates, and
    every non-message line, are dropped.
    """
    out: list[dict] = []
    tool_at: dict[str, int] = {}
    for ln in lines:
        if not isinstance(ln, dict):
            continue
        kind, msg = ln.get("type"), ln.get("message")
        if kind not in ("message", "message_update") or not isinstance(msg, dict):
            continue
        call_id = msg.get("tool_call_id") if msg.get("role") == "tool" else None
        key = str(call_id) if call_id is not None else None
        if key is not None and key in tool_at:
            i = tool_at[key]
            out[i] = {**out[i], "message": msg, "end_ts": ln.get("ts")}
            continue
        if kind != "message":
            continue  # an update of a message that is not a tool result
        out.append(dict(ln))
        if key is not None:
            tool_at[key] = len(out) - 1
    return out


def _arguments(fn: dict) -> dict:
    args = fn.get("arguments") or "{}"
    if isinstance(args, dict):
        return args
    try:
        parsed = json.loads(args)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _placeholder(content: Any) -> bool:
    if not isinstance(content, str):
        return False
    try:
        parsed = json.loads(content)
    except ValueError:
        return False
    return isinstance(parsed, dict) and "_placeholder" in parsed


def timed_cells(folded: list[dict]) -> list[TimedCell]:
    """Every ``execute_code`` call with a result, in result order, as in :func:`..analysis.cells.cells_from_transcript`.

    ``start`` is the ``ts`` of the assistant line that made the call; ``end`` that of the final result.
    A call whose result never left its placeholder keeps no output and says so in ``error``.
    """
    pending: dict[str, tuple[str, str, float]] = {}
    cells: list[TimedCell] = []
    for row in folded:
        msg = row.get("message")
        if row.get("type") != "message" or not isinstance(msg, dict):
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            if fn.get("name") != "execute_code":
                continue
            args = _arguments(fn)
            code = args.get("code")
            lang = args.get("language") or args.get("_language") or "python"
            pending[tc.get("id")] = (
                "" if code is None else str(code),
                str(lang),
                _seconds(row.get("ts")),
            )
        if msg.get("role") == "tool" and msg.get("tool_call_id") in pending:
            call_id = msg["tool_call_id"]
            code, lang, start = pending.pop(call_id)
            content = msg.get("content")
            if _placeholder(content):
                output, err = "", "(no final result was recorded)"
            else:
                raw = _raw(content)
                output = _stdout(raw)
                err = (raw.split(_STDERR, 1)[1] or None) if _STDERR in raw else None
            end = _seconds(row.get("end_ts") or row.get("ts"))
            cells.append(
                TimedCell(
                    Cell(len(cells), code, output, err, language=lang),
                    str(call_id),
                    start,
                    end,
                ),
            )
    return cells


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        )
    return ""


def dialogue_text(folded: list[dict]) -> tuple[list[str], list[str]]:
    """(every user message in order: the request, then the observations; the replies).

    A reply is an assistant message with text and no tool calls.
    """
    requests: list[str] = []
    replies: list[str] = []
    for row in folded:
        msg = row.get("message")
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        text = _text(msg.get("content"))
        if role == "user" and text:
            requests.append(text)
        elif role == "assistant" and text and not msg.get("tool_calls"):
            replies.append(text)
    return requests, replies


def cell_at(ts: float | None, cells: list[TimedCell]) -> int:
    """The index of the first cell with ``start <= ts <= end``, else -1."""
    if ts is None or not isinstance(ts, (int, float)) or math.isnan(ts):
        return -1
    for c in cells:
        if c.start <= ts <= c.end:
            return c.cell.index
    return -1


def learned_secrets(values: Iterable[tuple[str, Any]]) -> dict[str, str]:
    """Credentials seen in recorded values: ``{structural label: value}``.

    *values* are ``(label prefix, value)`` pairs (``("spotify.login", response)``). A string of at least
    8 characters under a dict key whose upper-cased name contains TOKEN, PASSWORD, SECRET, KEY or
    CREDENTIAL is learned under ``<prefix>.<key path>`` (``spotify.login.access_token``); list items share
    their list's path. The label is structure, never the value; two values on one path get ``#2``, ``#3``.
    """
    out: dict[str, str] = {}
    have: set[str] = set()

    def keep(label: str, value: str) -> None:
        if value in have:
            return
        name, n = label, 1
        while name in out:
            n += 1
            name = f"{label}#{n}"
        out[name] = value
        have.add(value)

    def walk(prefix: str, value: Any, depth: int) -> None:
        if depth > _MAX_DEPTH:
            return
        if isinstance(value, dict):
            for k, v in value.items():
                key = str(k)
                path = f"{prefix}.{key}" if prefix else key
                if (
                    isinstance(v, str)
                    and len(v) >= _MIN_SECRET_LEN
                    and any(p in key.upper() for p in _LEARNED_PARTS)
                ):
                    keep(path, v)
                else:
                    walk(path, v, depth + 1)
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(prefix, v, depth + 1)

    for prefix, value in values:
        walk(prefix, value, 0)
    return out


def _env_secrets() -> dict[str, str]:
    return {
        k: v
        for k, v in os.environ.items()
        if any(p in k.upper() for p in _SECRET_NAME_PARTS)
    }


class TimedObserver(RecordingObserver):
    """The tool adapter's recorder, also stamping each call's start on the harness clock (``time.time()``).

    :func:`assemble` attributes a stamped action to the cell whose span holds its stamp; without a
    stamp the recorder's own cell index stands.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._t: dict[int, float] = {}
        self._stamps: dict[int, tuple[Action, float]] = {}
        self._tlock = threading.Lock()

    def before(self, call: Any) -> None:
        with self._tlock:
            if len(self._t) < 4 * self.max_calls:
                self._t[id(call)] = time.time()
        return super().before(call)

    def _action(self, call: Any, cell: int, result: Any, error: Any) -> Action:
        action = super()._action(call, cell, result, error)
        with self._tlock:
            ts = self._t.pop(id(call), None)
            if ts is not None and len(self._stamps) < 4 * self.max_calls:
                self._stamps[id(action)] = (action, ts)
        return action

    def after(self, call: Any, **kwargs: Any) -> None:
        try:
            super().after(call, **kwargs)
        finally:
            with self._tlock:
                self._t.pop(
                    id(call),
                    None,
                )  # intercepted or dropped calls never reach _action

    def stamp_of(self, action: Action) -> float | None:
        with self._tlock:
            hit = self._stamps.get(id(action))
        return hit[1] if hit is not None and hit[0] is action else None


def _tool_actions(observer: Any, cells: list[TimedCell]) -> list[Action]:
    if observer is None:
        return []
    drained = observer.drain()
    actions = list(getattr(drained, "actions", drained) or [])
    stamp = getattr(observer, "stamp_of", None)
    if stamp is not None:
        for a in actions:
            ts = stamp(a)
            if ts is not None:
                a.cell = cell_at(ts, cells)
    return actions


def _fingerprints(actions: list[Action]) -> dict:
    out = fingerprint([a for a in actions if a.kind != "tool"])
    out.update(
        action_fingerprints(actions),
    )  # tool keys, with capped responses at their full shape
    return out


def assemble(
    run: Any,
    lines: list[dict],
    memory_diff: str,
    ended_at: str,
    *,
    extra_actions: Iterable[Action] = (),
    worktree_before: str | None = None,
    worktree_after: str | None = None,
    worktree_diff: str = "",
) -> tuple[Episode, Redactor]:
    """The request's episode and the redactor to write it with.

    *run* supplies ``episode_id``, ``started_at``, ``build``, ``model``, ``effort``, ``pin`` (memory
    ``main`` at the request's start), ``request`` (used when the transcript holds no user message),
    ``observer`` (the tool recorder, drained here; may be None), ``costs`` (a
    :class:`.cost.CostListener`) and optionally ``regime`` (default ``"dense"``). The tool actions come
    first, then *extra_actions* in their order; an action's index in the episode is what covers cite.
    """
    folded = fold(lines)
    cells = timed_cells(folded)
    requests, replies = dialogue_text(folded)
    actions = _tool_actions(getattr(run, "observer", None), cells) + list(extra_actions)
    secrets = {
        **_env_secrets(),
        **learned_secrets(
            [(f"{a.channel}.{a.method}", a.response) for a in actions]
            + [(f"{a.channel}.{a.method}", a.kwargs) for a in actions],
        ),
    }
    costs = getattr(run, "costs", None)
    ep = Episode(
        episode_id=run.episode_id,
        started_at=run.started_at,
        ended_at=ended_at,
        build=run.build,
        model=run.model,
        effort=run.effort,
        regime=getattr(run, "regime", None) or "dense",
        memory_main=run.pin,
        worktree_before=worktree_before,
        worktree_after=worktree_after,
        request=requests or [run.request],
        transcript=list(lines),
        cells=[c.cell for c in cells],
        actions=actions,
        memory_diff=memory_diff,
        worktree_diff=worktree_diff,
        costs=list(getattr(costs, "rows", None) or []),
        fingerprints=_fingerprints(actions),
        replies=replies,
    )
    return ep, Redactor(secrets)
