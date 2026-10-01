"""Append-only session transcripts on disk (``UNIFY_TRANSCRIPTS``).

Every agent conversation -- the actor's loop, a sub-agent's, the storage
review's, the context compressor's -- is one *session*, keyed by the LLM client
that carries it, so a loop restarted after context compression stays in the
session it started. Each session appends its messages, tool calls and results,
in the order the loop publishes them, as JSON lines to
``<UNIFY_HOME>/transcripts/<session-id>.jsonl``; no line is ever rewritten. A
tool result that fills a placeholder already on disk is appended again as a
``message_update`` line rather than edited in place.

When a session ends (its client is released, or the process exits) one line
goes to ``<UNIFY_HOME>/transcripts/index.jsonl`` naming the session, the session
that spawned it, where it came from, when it started and ended, the tools it
called and how many results were errors.

Nothing is redacted from what the model saw, except that the value of any
environment variable whose name contains KEY, TOKEN, SECRET or PASSWORD, and any
provider credential unillm holds, is replaced by ``[REDACTED:<name>]`` before a
line is written.

With the switch off nothing here touches the filesystem: :func:`attach`
returns ``None`` and every other entry point is a no-op for a loop it did not
attach.
"""

from __future__ import annotations

import contextvars
import datetime as dt
import json
import logging
import os
import re
import secrets
import threading
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

__all__ = [
    "SECRET_NAME_MARKERS",
    "TranscriptSession",
    "attach",
    "close",
    "enabled",
    "observe",
    "record_compaction",
    "resume_session",
    "session_for_messages",
    "transcripts_dir",
]

logger = logging.getLogger(__name__)

# An environment variable whose name contains one of these (case-insensitive)
# is treated as a credential: its value never reaches a transcript line.
SECRET_NAME_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")

# Shorter values are too likely to occur by chance ("1", "true") to replace.
_MIN_SECRET_LEN = 8

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# The session of the loop whose task is running; a loop started from inside
# it (a sub-agent, a nested manager call) inherits it through the task context
# and records it as its parent.
_CURRENT: contextvars.ContextVar[Optional["TranscriptSession"]] = (
    contextvars.ContextVar("unify_transcript_session", default=None)
)
# A session id the next root session in this context should continue.
_REQUESTED_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "unify_transcript_requested_id",
    default=None,
)

_LIVE: "weakref.WeakSet[TranscriptSession]" = weakref.WeakSet()
_BY_LABEL: "weakref.WeakValueDictionary[str, TranscriptSession]" = (
    weakref.WeakValueDictionary()
)
_REGISTRY_LOCK = threading.Lock()
_INDEX_LOCK = threading.Lock()


def enabled() -> bool:
    """Whether ``UNIFY_TRANSCRIPTS`` is on."""
    from unify.settings import SETTINGS

    return bool(getattr(SETTINGS, "UNIFY_TRANSCRIPTS", False))


def transcripts_dir() -> Path:
    """``<UNIFY_HOME>/transcripts``; not created by this call."""
    from unify.db import store_home

    return store_home() / "transcripts"


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _new_session_id() -> str:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{secrets.token_hex(4)}"


# ---------------------------------------------------------------------------
# Credential scrubbing
# ---------------------------------------------------------------------------


def _is_secret_name(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in SECRET_NAME_MARKERS)


def _secret_values() -> list[tuple[str, str]]:
    """``(name, value)`` for every credential a line must not carry."""
    found: dict[str, str] = {}
    for name, value in os.environ.items():
        if _is_secret_name(name) and value and len(value) >= _MIN_SECRET_LEN:
            found.setdefault(value, name)
    try:
        import unillm
        from pydantic import SecretStr

        settings = unillm.SETTINGS
        for name in type(settings).model_fields:
            value = getattr(settings, name, None)
            if isinstance(value, SecretStr):
                raw = value.get_secret_value()
                if raw and len(raw) >= _MIN_SECRET_LEN:
                    found.setdefault(raw, name)
    except Exception:
        pass
    # Longest first, so a secret that contains another is replaced whole.
    return sorted(
        ((name, value) for value, name in found.items()),
        key=lambda item: -len(item[1]),
    )


def scrub(line: str) -> str:
    """Replace every credential value in *line*, raw or JSON-escaped."""
    for name, value in _secret_values():
        marker = f"[REDACTED:{name}]"
        if value in line:
            line = line.replace(value, marker)
        escaped = json.dumps(value, ensure_ascii=False)[1:-1]
        if escaped != value and escaped in line:
            line = line.replace(escaped, marker)
    return line


# ---------------------------------------------------------------------------
# Result classification
# ---------------------------------------------------------------------------


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return "" if content is None else str(content)


def _is_placeholder(msg: dict) -> bool:
    try:
        from unify.common._async_tool.messages import is_non_final_tool_reply

        return is_non_final_tool_reply(msg)
    except Exception:
        return False


def _is_error_result(msg: dict) -> bool:
    """A final tool result that reports a failure."""
    text = _content_text(msg.get("content"))
    if "Traceback (most recent call last)" in text:
        return True
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return False
    return isinstance(parsed, dict) and bool(parsed.get("error"))


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class TranscriptSession:
    """One agent conversation's append-only JSONL file."""

    def __init__(
        self,
        client: Any,
        *,
        session_id: str,
        resumed: bool,
        parent: Optional["TranscriptSession"],
        origin: str,
        label: str,
    ) -> None:
        self.id = session_id
        self.parent_id = parent.id if parent is not None else None
        self.origin = origin
        self.label = label
        self.resumed = resumed
        self.path = transcripts_dir() / f"{session_id}.jsonl"
        self.started_at = _now()
        self.ended_at = self.started_at
        self.tools: set[str] = set()
        self.tool_calls = 0
        self.errors = 0
        self.messages = 0
        self.compactions = 0
        self._lock = threading.RLock()
        self._client_ref = _weak(client)
        # Identity of every message already on disk -> the JSON it was written
        # as. The strong reference in the value keeps the id from being reused
        # by a new dict while this session lives.
        self._seen: dict[int, tuple[Any, str]] = {}
        self._errored_calls: set[str] = set()
        self._system_prompt: Optional[str] = None
        self._seq = _count_lines(self.path) if resumed else 0
        self._finalizer: Optional[weakref.finalize] = None

    # -- writing ------------------------------------------------------------
    def _write(self, record: dict) -> None:
        with self._lock:
            now = _now()
            line = json.dumps(
                {"seq": self._seq, "ts": now, "session": self.id, **record},
                default=str,
                ensure_ascii=False,
            )
            line = scrub(line)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # O_APPEND: every write lands at the end, whoever else has the
            # file open, and nothing before it is touched.
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            self._seq += 1
            self.ended_at = now

    def _client_messages(self) -> Optional[list]:
        client = self._client_ref() if self._client_ref is not None else None
        if client is None:
            return None
        try:
            msgs = client.messages
        except Exception:
            return None
        return msgs if isinstance(msgs, list) else None

    def _note(self, msg: dict) -> None:
        """Update the index counters for a message just written."""
        role = msg.get("role")
        if role == "assistant":
            for call in msg.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                name = (call.get("function") or {}).get("name")
                if name:
                    self.tools.add(str(name))
                    self.tool_calls += 1
        elif role == "tool" and not _is_placeholder(msg) and _is_error_result(msg):
            call_id = str(msg.get("tool_call_id") or id(msg))
            if call_id not in self._errored_calls:
                self._errored_calls.add(call_id)
                self.errors += 1

    def _record(self, msg: Any, *, in_context: bool, loop: str) -> None:
        if not isinstance(msg, dict):
            msg = {"content": msg}
        dumped = json.dumps(msg, default=str, ensure_ascii=False, sort_keys=True)
        prior = self._seen.get(id(msg))
        if prior is not None and prior[0] is msg:
            if prior[1] == dumped:
                return
            self._seen[id(msg)] = (msg, dumped)
            self._write({"type": "message_update", "loop": loop, "message": msg})
        else:
            self._seen[id(msg)] = (msg, dumped)
            self.messages += 1
            self._write(
                {
                    "type": "message",
                    "loop": loop,
                    "in_context": in_context,
                    "message": msg,
                },
            )
        self._note(msg)

    def _record_system_prompt(self, loop: str) -> None:
        client = self._client_ref() if self._client_ref is not None else None
        prompt = getattr(client, "system_message", None) if client else None
        if isinstance(prompt, str) and prompt and prompt != self._system_prompt:
            self._system_prompt = prompt
            self._write({"type": "system_prompt", "loop": loop, "content": prompt})

    def sync(self, loop: str = "") -> None:
        """Append every message in the client's context not yet on disk."""
        with self._lock:
            self._record_system_prompt(loop)
            for msg in list(self._client_messages() or []):
                if id(msg) in self._seen and self._seen[id(msg)][0] is msg:
                    continue
                self._record(msg, in_context=True, loop=loop)

    def observe(self, published: Iterable[Any], loop: str = "") -> None:
        """Record *published* messages, after anything that preceded them."""
        with self._lock:
            self.sync(loop)
            context_ids = {id(m) for m in (self._client_messages() or [])}
            for msg in published:
                self._record(msg, in_context=id(msg) in context_ids, loop=loop)

    def start(self) -> None:
        client = self._client_ref() if self._client_ref is not None else None
        prompt = getattr(client, "system_message", None) if client else None
        self._system_prompt = prompt if isinstance(prompt, str) else None
        self._write(
            {
                "type": "session_start",
                "parent": self.parent_id,
                "origin": self.origin,
                "lineage": self.label,
                "resumed": self.resumed,
                "pid": os.getpid(),
                "system_prompt": self._system_prompt,
            },
        )

    def mark_seen(self, messages: Iterable[Any]) -> None:
        """Treat *messages* as on disk already (a compaction line holds them)."""
        with self._lock:
            for msg in messages:
                self._seen[id(msg)] = (
                    msg,
                    json.dumps(msg, default=str, ensure_ascii=False, sort_keys=True),
                )

    def pointer_line(self) -> str:
        return (
            f"The full history of this session is at {self.path}; "
            "search it with grep/rg if you need details."
        )

    # -- ending ---------------------------------------------------------------
    def finalize(self) -> None:
        """Write this session's index line (once)."""
        if self._finalizer is not None:
            self._finalizer()
        else:
            self._write_index()

    def _write_index(self) -> None:
        record = {
            "session": self.id,
            "parent": self.parent_id,
            "origin": self.origin,
            "lineage": self.label,
            "path": str(self.path),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "tools": sorted(self.tools),
            "tool_calls": self.tool_calls,
            "errors": self.errors,
            "messages": self.messages,
            "compactions": self.compactions,
            "resumed": self.resumed,
        }
        line = scrub(json.dumps(record, default=str, ensure_ascii=False))
        index = self.path.parent / "index.jsonl"
        with _INDEX_LOCK:
            if not self.path.parent.is_dir():
                # The home was removed under the session; do not recreate it.
                return
            with open(index, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def _weak(obj: Any):
    try:
        return weakref.ref(obj)
    except TypeError:
        return lambda: obj


def _count_lines(path: Path) -> int:
    try:
        with open(path, "rb") as fh:
            return sum(1 for _ in fh)
    except FileNotFoundError:
        return 0


def _finalize_session(session: TranscriptSession) -> None:
    try:
        session._write_index()
    except Exception:
        logger.debug("transcript index write failed", exc_info=True)


# ---------------------------------------------------------------------------
# Entry points used by the tool loop
# ---------------------------------------------------------------------------


@contextmanager
def resume_session(session_id: str) -> Iterator[None]:
    """Make the next root session started in this context continue *session_id*.

    A later process that passes the same id appends to the same file; any other
    id, or none, starts a file of its own.
    """
    if not _SESSION_ID_RE.match(session_id or ""):
        raise ValueError(f"not a valid transcript session id: {session_id!r}")
    token = _REQUESTED_ID.set(session_id)
    try:
        yield
    finally:
        _REQUESTED_ID.reset(token)


@contextmanager
def as_current(session: Optional[TranscriptSession]) -> Iterator[None]:
    """Run a block as if inside *session*'s loop (children link to it)."""
    if session is None:
        yield
        return
    token = _CURRENT.set(session)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def attach(client: Any, loop_cfg: Any) -> Optional[TranscriptSession]:
    """Bind the loop described by *loop_cfg* to its client's session.

    Called once per loop run, from inside the loop's task. The first loop on a
    client opens its session; a later loop on the same client (a restart after
    compression) joins it. Returns ``None``, touching nothing, when the switch
    is off.
    """
    if not enabled() or client is None:
        return None
    try:
        session = getattr(client, "_unify_transcript", None)
        label = str(getattr(loop_cfg, "label", "") or "")
        if not isinstance(session, TranscriptSession):
            session = _open_session(client, loop_cfg, label)
        with _REGISTRY_LOCK:
            if label:
                _BY_LABEL[label] = session
        with suppress_all():
            setattr(loop_cfg, "_unify_transcript", session)
        _CURRENT.set(session)
        session.sync(label)
        return session
    except Exception:
        logger.warning("transcript attach failed", exc_info=True)
        return None


def _open_session(client: Any, loop_cfg: Any, label: str) -> TranscriptSession:
    parent = _CURRENT.get()
    if parent is None:
        lineage = list(getattr(loop_cfg, "lineage", []) or [])
        if len(lineage) > 1:
            with _REGISTRY_LOCK:
                parent = _BY_LABEL.get("->".join(lineage[:-1]))
    requested = _REQUESTED_ID.get() if parent is None else None
    with _REGISTRY_LOCK:
        live_ids = {s.id for s in _LIVE}
    if requested in live_ids:
        # Another live root already continues this id; two writers must never
        # share one file, so this one starts its own.
        requested = None
    directory = transcripts_dir()
    if requested:
        session_id = requested
        resumed = (directory / f"{session_id}.jsonl").exists()
        _REQUESTED_ID.set(None)
    else:
        session_id = _new_session_id()
        while (directory / f"{session_id}.jsonl").exists():
            session_id = _new_session_id()
        resumed = False
    origin = str(getattr(loop_cfg, "loop_id", "") or getattr(client, "origin", ""))
    session = TranscriptSession(
        client,
        session_id=session_id,
        resumed=resumed,
        parent=parent,
        origin=origin,
        label=label,
    )
    session.start()
    try:
        session._finalizer = weakref.finalize(client, _finalize_session, session)
    except TypeError:
        import atexit

        atexit.register(_finalize_session, session)
    with suppress_all():
        setattr(client, "_unify_transcript", session)
    with _REGISTRY_LOCK:
        _LIVE.add(session)
    return session


def observe(messages: Any, loop_cfg: Any) -> None:
    """Record messages a loop publishes; a no-op for loops never attached."""
    session = getattr(loop_cfg, "_unify_transcript", None)
    if session is None:
        return
    try:
        batch = messages if isinstance(messages, list) else [messages]
        session.observe(batch, str(getattr(loop_cfg, "label", "") or ""))
    except Exception:
        logger.warning("transcript write failed", exc_info=True)


def session_for_messages(messages: Any) -> Optional[TranscriptSession]:
    """The live session whose client currently holds *messages* (by identity)."""
    if not enabled() or not isinstance(messages, list):
        return None
    with _REGISTRY_LOCK:
        live = list(_LIVE)
    for session in live:
        if session._client_messages() is messages:
            return session
    return None


def record_compaction(
    session: Optional[TranscriptSession],
    *,
    archived: int,
    pass_number: int,
    context: list[dict],
) -> None:
    """Append a ``compaction`` line holding the context that replaced history."""
    if session is None:
        return
    try:
        session.compactions += 1
        session._write(
            {
                "type": "compaction",
                "pass": pass_number,
                "archived_messages": archived,
                "context": context,
            },
        )
        session.mark_seen(context)
    except Exception:
        logger.warning("transcript compaction write failed", exc_info=True)


def close(client: Any) -> None:
    """End *client*'s session now, writing its index line."""
    session = getattr(client, "_unify_transcript", None)
    if isinstance(session, TranscriptSession):
        session.finalize()


@contextmanager
def suppress_all() -> Iterator[None]:
    try:
        yield
    except Exception:
        pass
