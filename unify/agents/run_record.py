"""The run record: one append-only record per memory home for the whole run, shared by
every agent of the run across processes: the actor of each request, the fork that reviews each episode, the
librarian, the writer service's passes and every helper any of them starts.

A session record (:mod:`unify.agents.record`) lives in one process and numbers its entries from its own list. Here
several processes append to one file, so numbering is done under a file lock: the appender takes ``flock`` on
``run-record.lock``, reads the lines other processes appended since it last looked, writes ``seq + 1`` and fsyncs, all
while holding the lock. A line cut by a crash is closed with a newline by the next appender and skipped by every
reader; numbering continues from the last whole entry.

Files, under ``<home>/memory-state/agents/`` (the harness-only state directory: no box binds it):

* ``run-record.jsonl``: the entries (:func:`row_of`), scrubbed of credentials on append;
* ``run-record.cursors.jsonl``: the delivery log, one row per delivery to an agent (a block at its turn boundary,
  what ``wait`` returned, what ``messages`` showed): which entries it was given, by which way, in which pass;
* ``roster.json``: who is on the record, rewritten whole (through a temporary file and a rename) under
  ``roster.lock`` at each join, leave and change of state; ``roster-ended.jsonl`` keeps every name a run has used.

Addressing: ``@<name>`` names one agent; ``@writer`` is the writer (the service's pass agent, or the service itself
while it is idle); ``@actor`` every running actor; ``@fork`` the fork of the sender's own episode; ``@helpers`` every
running helper; ``@all`` every running agent but the sender. Names that match nobody are returned as unresolved.

Roles that outlive their instances (:data:`INSTANCE_ROLES`: the librarian). ``@librarian`` names the role, which lives as
long as the run once it is seeded (its state under ``roles/librarian/``, :func:`seed_role`): a message to it is a
question for the role's next instance. The request's role dispatcher (``integration.request``) starts one instance,
``librarian-<hex>``, per such message, in order and one at a time across processes (:func:`role_lock`, the claims
file); the instance answers with one ``reply`` entry to that message, mentioning its author, and ends. ``@librarian``
in a run with no librarian is refused, with the addresses that do reach someone.

An instance's address, and every agent's, is valid only while it lives: a post that names an agent that has ended is
refused, with a pointer (for an instance of a role: address the role; for a fork: when it ended). The writer is the one
exception: the service holds ``writer`` across passes and restarts, so a message to it is kept for when it next runs.

Waiting across processes: ``wait`` returns the entries for the agent as soon as one is appended (in this process at
once; from another process within a poll interval, or at once where inotify is available), and returns with
:data:`ALL_WAITING` when every other agent that could answer is itself waiting with nothing unread, idle or ended.
:data:`MAX_WAIT_S` stays the outer backstop.

The record's outer backstops, stated, never steering: the byte backstop (:data:`MAX_RECORD_BYTES`), the per-agent
post interval, and the storm notice: a pair of agents whose last :data:`STORM_RUN` entries are only to each other,
with no tool call or file read by either in between, gets one harness notice (delivered to both), never a refusal.
"""

from __future__ import annotations

import asyncio
import ctypes
import ctypes.util
import fcntl
import json
import logging
import os
import re
import secrets
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from unify.agents.entry import KINDS, MAX_TEXT_BYTES, Entry, mention_tokens, valid_name
from unify.agents.options import Options
from unify.common import tool_registry as _registry
from unify.agents.record import (
    ALL,
    HARNESS,
    MAX_WAIT_S,
    USER,
    Participant,
    PostRefused,
    Record,
    _now,
    _wake,
)

logger = logging.getLogger(__name__)

#: Where the run record lives under the memory home (the memory state directory, ``Paths.state_dir``).
STATE_DIRNAME = "memory-state"
AGENTS_DIRNAME = "agents"
RECORD_FILE = "run-record.jsonl"
LOCK_FILE = "run-record.lock"
CURSORS_FILE = "run-record.cursors.jsonl"
ROSTER_FILE = "roster.json"
ROSTER_LOCK = "roster.lock"
ENDED_FILE = "roster-ended.jsonl"

#: The schema version every row carries.
VERSION = 1
#: Who an entry's author is (§2.1).
ROLES = ("actor", "fork", "librarian", "writer", "helper", "harness", "user")
#: The roles of agents (the others are the harness and the user).
AGENT_ROLES = ("actor", "fork", "librarian", "writer", "helper")
#: An agent's state on the roster (§2.2): ``idle`` is the writer service with no pass running; ``ended`` is gone.
STATES = ("running", "waiting", "idle", "ended")
#: The ways an entry reaches an agent, as the delivery log records them.
VIAS = ("block", "wait", "messages")
#: Addresses that name a role, and what each resolves to (module docstring).
ROLE_ADDRESSES = ("writer", "actor", "librarian", "fork", "helpers")
#: The writer's one name on the record.
WRITER = "writer"
#: Rev 2: the roles that live as long as the run (module docstring); each instance is ``<role>-<hex>``, short-lived.
INSTANCE_ROLES = ("librarian",)
#: A role's durable state, ``<agents>/roles/<role>/``: ``role.json`` (written once, when the run seeds the role),
#: ``answers.jsonl`` (the role's exact-match answer cache), ``notes.md`` (what instances keep for later ones),
#: ``claims.jsonl`` (which instance took which message, and how it ended), ``events.jsonl`` and the instance lock.
ROLES_DIRNAME = "roles"
ROLE_FILE = "role.json"
ROLE_ANSWERS = "answers.jsonl"
ROLE_NOTES = "notes.md"
ROLE_CLAIMS = "claims.jsonl"
ROLE_EVENTS = "events.jsonl"
ROLE_LOCK = "instance.lock"
#: How a claim on a role message ends: answered (one reply posted), failed (one reply saying why), withdrawn (its author
#: cancelled it: no reply), abandoned (the instance was stopped before its reply: the next dispatcher takes it again).
CLAIM_STATES = ("claimed", "answered", "failed", "withdrawn", "abandoned")


#: The byte backstop of a run-long record: past it, model posts are refused and harness entries are cut short.
MAX_RECORD_BYTES = 512 * 1024 * 1024
#: A harness entry's text past the byte backstop, at most.
CUT_TEXT_BYTES = 1024
#: The storm notice: this many entries of a pair only to each other, with no evidence read in between.
STORM_RUN = 20
STORM_MARK = "this exchange has gone back and forth"
#: How often a waiting agent looks at the file when inotify is not available (and the backstop interval when it is).
POLL_S = 0.1
INOTIFY_BACKSTOP_S = 1.0
#: What a row adds to its text's bytes, at most, short of the text's own escaping (the byte backstop's margin).
_ROW_OVERHEAD = 2048
#: A helper's line on the roster: the start of its task, at most this many characters.
HELP_CHARS = 120
#: The longest name tail a clash can grow an episode prefix to.
_MAX_HEX = 16

#: The run record's options: no count cap (a cap would refuse posts mid-run), the byte backstop, the per-agent post
#: interval of a session record, and the helper limits a session record gives each request or pass.
RUN_OPTIONS = Options(
    max_live=16, max_total=64, max_entries=None, max_record_bytes=MAX_RECORD_BYTES
)

#: What a running agent of each role is shown as helping with (a helper's line is the start of its task).
HELPS_WITH = {
    "actor": "is doing the current request",
    "fork": "reviews its episode and writes down what is worth keeping from it",
    "librarian": "answers which stored items fit a situation",
    "writer": "keeps the library: decides what it stores, changes and retires, and answers questions about stored items",
}

#: Whom each role should mention, and when (§2.2): the complete default set; general, naming no world.
SHOULD_MENTION: dict[str, tuple[dict[str, str], ...]] = {
    # the actor asks the library one way, memory.ask (its tool text says so); it is never told to mention @librarian
    "actor": (
        {
            "who": "writer",
            "when": "a served item was wrong, misleading, or something you needed was missing",
        },
    ),
    "writer": (
        {
            "who": "fork",
            "when": "to ask what an episode did, while that episode's fork is still running",
        },
        {
            "who": "actor",
            "when": "to ask the running actor a clarifying question about its own episode; never to give it "
            "instructions while it works, unless it asked",
        },
        {"who": "helpers", "when": "to message the helpers you started (tools.delegate starts one)"},
    ),
    "helper": ({"who": "<starter>", "when": "to report findings, with their sources"},),
    "librarian": (
        {
            "who": "writer",
            "when": "an item you were asked about looks stale or contradicted",
        },
    ),
}

#: The one common prompt section, the same bytes in every role's prompt (defined in the shared tool registry with the
#: comms tools it describes).
COMMON_SECTION = _registry.COMMON_SECTION
#: The common section of a run without memory (``UNIFY_MEMORY=off``): no writer, fork or librarian named.
COMMON_SECTION_NO_MEMORY = _registry.COMMON_SECTION_NO_MEMORY


def prompt_lines(name: str, role: str, spawner: Optional[str] = None, *, home: Optional[Path | str] = None) -> str:
    """What every agent on the record reads after its cache-preserving prefix: :data:`COMMON_SECTION` (the same bytes
    for every role), then its own roster lines: its name and whom its role should mention, and when. A role that
    outlives its instances (:data:`INSTANCE_ROLES`) is named only when the run has it (seeded in *home*; no *home*:
    not named), so no agent is told to mention someone the run does not have; a run without memory
    (``UNIFY_MEMORY=off``) reads :data:`COMMON_SECTION_NO_MEMORY` and is never told to mention a writer, fork or
    librarian.
    """
    memory = run_has_memory(home, role, spawner)
    lines = [_registry.common_section(memory), "", f"On the record you are `{name}` ({role})."]
    for m in SHOULD_MENTION.get(role, ()):
        who = (spawner or "") if m["who"] == "<starter>" else m["who"]
        if who in INSTANCE_ROLES and (home is None or not role_seeded(home, who)):
            continue
        if who in _registry.MEMORY_ROLES and not memory:   # a run without memory has no writer, fork or librarian
            continue
        if who:
            lines.append(mention_line(who, m["when"]))
    return "\n".join(lines)


def mention_line(who: str, when: str) -> str:
    """One "whom to mention, and when" line, as every text of the record writes it (:func:`prompt_lines` and
    ``tools.who()``): *when* is a purpose (``to report findings``) or a condition (``a served item was wrong``)."""
    return f"Mention @{who} {when}." if when.startswith("to ") else f"Mention @{who} when {when}."


def _address_hint(lead: str, addresses: str) -> str:
    """The end of a refusal that names the addresses that reach someone now (``RunRecordView.addresses``)."""
    return f"{lead} {addresses}" if addresses else "no other agent is on the record now"


def works_in_request(rows: list, name: str) -> bool:
    """Whether the agent *name* on the roster *rows* works inside a request: an actor, or a helper started by one,
    directly or through other helpers (a teammate). Such an agent never reaches an episode's fork, which starts only
    after the episode has ended."""
    by_name = {r.get("name"): r for r in rows if isinstance(r, dict)}
    row, seen = by_name.get(name), set()
    while row is not None and row.get("role") == "helper" and row.get("name") not in seen:
        seen.add(row.get("name"))
        row = by_name.get(row.get("spawner"))
    return row is not None and row.get("role") == "actor"


#: ``wait``'s end when no one who could answer is running.
ALL_WAITING = "every other agent that could answer is waiting, idle or ended, so no message can arrive"


# --- paths and stdlib readers (B2's ledger and C4's scorers read the record through these) -------------------------


def agents_dir(home: Path | str) -> Path:
    """``<home>/memory-state/agents``: the run record's folder under the memory home."""
    return Path(home) / STATE_DIRNAME / AGENTS_DIRNAME


def record_path(home: Path | str) -> Path:
    return agents_dir(home) / RECORD_FILE


def deliveries_path(home: Path | str) -> Path:
    return agents_dir(home) / CURSORS_FILE


# --- a role's state (rev 2: INSTANCE_ROLES) --------------------------------------------------------------------------


def role_dir(home: Path | str, role: str) -> Path:
    """``<home>/memory-state/agents/roles/<role>``: the role's durable state."""
    if role not in INSTANCE_ROLES:
        raise ValueError(f"{role!r} is not a role that outlives its instances")
    return agents_dir(home) / ROLES_DIRNAME / role


def role_seeded(home: Path | str, role: str) -> bool:
    """Whether the run has seeded *role* (its ``role.json`` names it)."""
    if role not in INSTANCE_ROLES:
        return False
    try:
        doc = json.loads((role_dir(home, role) / ROLE_FILE).read_text())
    except (OSError, ValueError):
        return False
    return isinstance(doc, dict) and doc.get("role") == role


def _memory_agent_name(name: Optional[str]) -> bool:
    """Whether *name* is a memory agent's on the record: the writer, a fork or a librarian instance."""
    n = str(name or "")
    return n == WRITER or n.startswith(("fork-", "librarian-"))


def run_has_memory(home: Optional[Path | str] = None, role: Optional[str] = None, spawner: Optional[str] = None) -> bool:
    """Whether the run an agent is in has memory, so a writer, forks and a librarian can be on its record: the agent
    is one of them (or a helper one of them started), the run's ``UNIFY_MEMORY`` is on or in_episode, or the run's
    record already shows one (a memory agent on its roster, or a seeded librarian). Only a run without memory by all
    three reads :data:`COMMON_SECTION_NO_MEMORY` and is never told to mention a writer, fork or librarian."""
    if role in _registry.MEMORY_ROLES or (role == "helper" and _memory_agent_name(spawner)):
        return True
    if _registry.memory_is_on():
        return True
    if home is None:
        return False
    try:
        rows = _load_roster(agents_dir(home) / ROSTER_FILE)["agents"]
    except Exception:  # noqa: BLE001 - an unreadable roster shows no memory agent
        rows = []
    return any(r.get("role") in _registry.MEMORY_ROLES for r in rows) or role_seeded(home, "librarian")


def seed_role(home: Path | str, role: str, *, by: str) -> bool:
    """Seed *role* once per run: its folder (private), ``role.json`` (created exclusively, so two processes starting
    together seed it once) and its empty answers, notes and claims files. True when this call seeded it."""
    folder = role_dir(home, role)
    folder.mkdir(parents=True, exist_ok=True)
    for d in (folder, folder.parent, agents_dir(home)):
        os.chmod(d, 0o700)
    for name in (ROLE_ANSWERS, ROLE_CLAIMS, ROLE_EVENTS):
        os.close(os.open(folder / name, os.O_WRONLY | os.O_CREAT | os.O_CLOEXEC, 0o600))
    notes = folder / ROLE_NOTES
    if not notes.exists():
        try:
            fd = os.open(notes, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(fd)
    doc = {"v": VERSION, "role": role, "seeded_at": _now(), "by": str(by)[:200]}
    try:
        fd = os.open(folder / ROLE_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    except FileExistsError:
        return False
    try:
        _write_all(fd, (json.dumps(doc) + "\n").encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    return True


class RoleBusy(RuntimeError):
    """Another process's instance of the role holds the instance lock now."""


@contextmanager
def role_lock(home: Path | str, role: str, *, wait: bool = True) -> Iterator[None]:
    """One instance of *role* at a time, across the run's processes (an exclusive lock on its instance lock). With
    ``wait=False`` a held lock raises :class:`RoleBusy` at once (an event loop never blocks on it)."""
    path = role_dir(home, role) / ROLE_LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            raise RoleBusy(f"an instance of {role} is running in another process") from None
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def role_claims(home: Path | str, role: str) -> dict[int, dict]:
    """The latest claim row per message seq of *role*."""
    rows, _ = _whole_lines(role_dir(home, role) / ROLE_CLAIMS)
    out: dict[int, dict] = {}
    for row in rows:
        seq = row.get("seq")
        if isinstance(seq, int) and not isinstance(seq, bool) and row.get("state") in CLAIM_STATES:
            out[seq] = row
    return out


def claim(home: Path | str, role: str, seq: int, state: str, **fields: Any) -> dict:
    """Append one claim row for message *seq* of *role* (``state`` one of :data:`CLAIM_STATES`)."""
    if state not in CLAIM_STATES:
        raise ValueError(f"unknown claim state {state!r}")
    row = {"v": VERSION, "seq": int(seq), "state": state, "at": _now(), **fields}
    folder = role_dir(home, role)
    with _flock(folder / (ROLE_CLAIMS + ".lock")):
        _append_line(folder / ROLE_CLAIMS, json.dumps(row, default=str))
    return row


def role_answers(home: Path | str, role: str) -> list[dict]:
    """The role's kept answers (``{"key", "answer", ...}`` rows), in the order they were kept."""
    rows, _ = _whole_lines(role_dir(home, role) / ROLE_ANSWERS)
    return [r for r in rows if isinstance(r.get("key"), str) and isinstance(r.get("answer"), dict)]


def keep_role_answer(home: Path | str, role: str, key: str, answer: dict, **fields: Any) -> None:
    """Keep one answer of *role* under its exact-match *key* (never a similarity key; module docstring)."""
    folder = role_dir(home, role)
    row = {"v": VERSION, "key": key, "answer": answer, "at": _now(), **fields}
    with _flock(folder / (ROLE_ANSWERS + ".lock")):
        _append_line(folder / ROLE_ANSWERS, json.dumps(row, default=str))


def role_notes(home: Path | str, role: str) -> str:
    """What earlier instances of *role* kept (empty when nothing was kept)."""
    try:
        return (role_dir(home, role) / ROLE_NOTES).read_text()
    except OSError:
        return ""


def role_event(home: Path | str, role: str, row: dict) -> None:
    """One event row of *role*'s instances (sizes, outcomes; never question or answer text)."""
    try:
        _append_line(role_dir(home, role) / ROLE_EVENTS, json.dumps({"v": VERSION, "at": _now(), **row}, default=str))
    except OSError:  # an event never stops an instance
        pass


def _whole_lines(path: Path, offset: int = 0) -> tuple[list[dict], int]:
    """The JSON object lines of *path* from byte *offset* that end with a newline (a line still being written
    waits), and the offset after them. A line that does not parse (cut by a crash) is skipped.
    """
    try:
        with open(path, "rb") as fh:
            fh.seek(offset)
            data = fh.read()
    except FileNotFoundError:
        return [], offset
    end = data.rfind(b"\n") + 1
    rows: list[dict] = []
    for line in data[:end].split(b"\n"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows, offset + end


def read_entries(home: Path | str, since: int = 0) -> list[dict]:
    """Every whole entry row of the run record with ``seq`` above *since*, in order (readers ignore unknown keys)."""
    rows, _ = _whole_lines(record_path(home))
    out: list[dict] = []
    last = 0
    for row in rows:
        seq = row.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq <= last:
            continue
        last = seq
        if seq > since:
            out.append(row)
    return out


def read_deliveries(home: Path | str) -> list[dict]:
    """Every delivery row (``agent``, ``role``, ``seqs``, ``via``, ``upto`` on block rows, ``pass_id``, ``episode``)."""
    return [
        r
        for r in _whole_lines(deliveries_path(home))[0]
        if isinstance(r.get("agent"), str)
    ]


# --- process identity -------------------------------------------------------------------------------------------


def proc_start(pid: int) -> Optional[str]:
    """The start time of a live, non-zombie process *pid* (``/proc/<pid>/stat`` field 22), else None."""
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text()
    except (OSError, ValueError):
        return None
    rest = stat.rsplit(")", 1)[-1].split()
    if len(rest) < 20 or rest[0] in ("Z", "X"):
        return None
    return rest[19]


def _process_alive(pid: Any, start: Any) -> bool:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return True  # no process named: nothing to check
    now = proc_start(pid)
    return now is not None and (not start or now == start)


def effective_state(row: dict) -> str:
    """The row's state, or ``ended`` when the process that holds it is gone (a crash never leaves an agent
    ``running`` for the others to wait on)."""
    state = row.get("state")
    if state not in STATES:
        return "ended"
    if state != "ended" and not _process_alive(row.get("pid"), row.get("proc_start")):
        return "ended"
    return state


# --- file locking and atomic writes --------------------------------------------------------------------------------


@contextmanager
def _flock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _write_all(fd: int, data: bytes) -> None:
    while data:
        data = data[os.write(fd, data) :]


def _append_line(path: Path, line: str) -> None:
    """Append one line (its newline added), fsynced, mode 0600."""
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        _write_all(fd, (line + "\n").encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(doc, ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load_roster(path: Path) -> dict:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        doc = None
    if not isinstance(doc, dict) or not isinstance(doc.get("agents"), list):
        doc = {"v": VERSION, "agents": []}
    doc["agents"] = [
        a
        for a in doc["agents"]
        if isinstance(a, dict) and isinstance(a.get("name"), str)
    ]
    return doc


# --- scrubbing ---------------------------------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """*text* as it may be written to the record: every credential this process holds, by value (the session record's
    scrub), then credential structures and key-shaped strings (:func:`unify.memory.redact.redact_error`).
    """
    from unify.memory.redact import redact_error
    from unify.transcripts import scrub

    return redact_error(scrub(str(text)))


def cut_to(text: str, limit: int, why: str) -> str:
    """*text* cut to *limit* UTF-8 bytes, saying why, when it is longer."""
    data = text.encode("utf-8")
    if len(data) <= limit:
        return text
    note = f"\n[cut: {why}]"
    room = max(0, limit - len(note.encode("utf-8")))
    return data[:room].decode("utf-8", errors="ignore") + note


def row_of(entry: Entry) -> dict:
    """The run record's line for *entry* (§2.1): every field, null where it does not apply."""
    return {
        "v": VERSION,
        "seq": entry.seq,
        "ts": entry.ts,
        "author": entry.author,
        "kind": entry.kind,
        "text": entry.text,
        "mentions": list(entry.mentions),
        "role": entry.role,
        "episode": entry.episode,
        "pass_id": entry.pass_id,
        "reply_to": entry.reply_to,
        "refs": list(entry.refs),
        "acts": entry.acts,
    }


#: A ref's form (§2.1): ``<episode>/<part>`` with the episode part grammar, or ``item:``, ``entry:``, ``capture:``.
_REF = re.compile(
    r"^(?:[A-Za-z0-9][A-Za-z0-9._-]{0,127}/(?:request|diff|(?:observation|cell|action):[0-9]+)"
    r"|(?:item|entry|capture):[^\s]{1,200})$",
)
#: A ref written in a message's text, in backticks or after a space: the parts a reader can open.
_REF_IN_TEXT = re.compile(
    r"(?<![\w/.:-])((?:[A-Za-z0-9][A-Za-z0-9._-]{0,127}/(?:request|diff|(?:observation|cell|action):[0-9]+))"
    r"|(?:item|entry|capture):[A-Za-z0-9_./:-]{1,200}[A-Za-z0-9_/])",
)


def refs_in(text: str, episode: Optional[str] = None) -> tuple[str, ...]:
    """The typed pointers a message names (§2.1 ``refs``), in first-seen order: ``<episode>/<part>``, ``item:<id>``,
    ``entry:<id>``, ``capture:<id>``; a bare ``action:<n>`` (or another part) is the author's own episode's.
    """
    out: list[str] = []
    for match in _REF_IN_TEXT.finditer(text or ""):
        ref = match.group(1)
        if _REF.match(ref) and ref not in out:
            out.append(ref)
    if episode:
        for match in re.finditer(
            r"(?<![\w/.:-])((?:observation|cell|action):[0-9]+)\b", text or ""
        ):
            ref = f"{episode}/{match.group(1)}"
            if ref not in out:
                out.append(ref)
    return tuple(out)


# --- waking a waiter in another process --------------------------------------------------------------------------

_IN_MODIFY, _IN_CLOSE_WRITE, _IN_CREATE, _IN_MOVED_TO = 0x2, 0x8, 0x100, 0x80
_IN_NONBLOCK, _IN_CLOEXEC = 0o4000, 0o2000000
_LIBC: Any = None
_LIBC_TRIED = False


def _libc() -> Any:
    global _LIBC, _LIBC_TRIED
    if not _LIBC_TRIED:
        _LIBC_TRIED = True
        try:
            name = ctypes.util.find_library("c")
            lib = ctypes.CDLL(name, use_errno=True) if name else None
            if (
                lib is not None
                and hasattr(lib, "inotify_init1")
                and hasattr(lib, "inotify_add_watch")
            ):
                _LIBC = lib
        except OSError:
            _LIBC = None
    return _LIBC


class _Watch:
    """An inotify watch on the record's folder, where the platform has it: the descriptor becomes readable when a file
    there changes. Without it (or if anything fails) :attr:`fd` is None and the waiter polls.
    """

    def __init__(self, folder: Path) -> None:
        self.fd: Optional[int] = None
        lib = _libc()
        if lib is None:
            return
        try:
            fd = lib.inotify_init1(_IN_NONBLOCK | _IN_CLOEXEC)
            if fd < 0:
                return
            mask = _IN_MODIFY | _IN_CLOSE_WRITE | _IN_CREATE | _IN_MOVED_TO
            if lib.inotify_add_watch(fd, os.fsencode(str(folder)), mask) < 0:
                os.close(fd)
                return
            self.fd = fd
        except (OSError, AttributeError, ValueError):
            self.fd = None

    def drain(self) -> None:
        if self.fd is None:
            return
        try:
            while os.read(self.fd, 65536):
                pass
        except (BlockingIOError, OSError):
            pass

    def close(self) -> None:
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None


# --- participants ------------------------------------------------------------------------------------------------

#: The session record's states an agent of this process can be in, and the roster state each is shown as.
_ROSTER_STATE = {
    "running": "running",
    "replied": "ended",
    "stopped": "ended",
    "failed": "ended",
    "lost": "ended",
}


@dataclass
class RunParticipant(Participant):
    """A participant of this process on the run record: its role, where it works, and its roster row kept in step
    with its state (a helper the pool marks ``replied``, ``stopped`` or ``failed`` has ended).
    """

    run_role: str = "helper"
    episode: Optional[str] = None
    pass_id: Optional[str] = None
    #: the tool calls and cells other than messages this agent has made (:meth:`RunRecord.note_act`)
    acts: int = 0
    #: the entries ``messages(all=True)`` has shown it up to
    all_cursor: int = 0
    record: Any = field(default=None, repr=False, compare=False)

    def __setattr__(self, name: str, value: Any) -> None:
        before = self.__dict__.get(name)
        super().__setattr__(name, value)
        record = self.__dict__.get("record")
        if name == "state" and record is not None and before != value:
            shown = _ROSTER_STATE.get(value)
            if shown is not None:
                try:
                    record.set_state(self.name, shown)
                except (
                    Exception
                ):  # noqa: BLE001 - the roster never stops an agent's work
                    logger.exception(
                        "run record: roster state of %s not written", self.name
                    )


# --- the record -----------------------------------------------------------------------------------------------------


class RunRecord(Record):
    """The run's record as one process sees it: the shared file read up to its end at every use, the agents this
    process hosts (joined here), and the roster every process shares. Several instances in one process (the actor's
    request, a writer pass, a fork relay) each read the same files; each wakes its own waiters at once and the others'
    through the file."""

    is_run_record = True

    def __init__(
        self,
        home: Path | str,
        *,
        create: bool = False,
        options: Options = RUN_OPTIONS,
        root: str = "",
        clock: Callable[[], str] = _now,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(
            None, options=options, root=root or "_", clock=clock, monotonic=monotonic
        )
        self.home = Path(home)
        self.folder = agents_dir(home)
        self.file = self.folder / RECORD_FILE
        self.cursor_file = self.folder / CURSORS_FILE
        self.lock_file = self.folder / LOCK_FILE
        self.roster_file = self.folder / ROSTER_FILE
        self.roster_lock = self.folder / ROSTER_LOCK
        self.ended_file = self.folder / ENDED_FILE
        self.folder.mkdir(parents=True, exist_ok=True)
        os.chmod(self.folder, 0o700)
        if create:
            os.close(
                os.open(
                    self.file,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                    0o600,
                )
            )
        elif not self.file.exists():
            raise FileNotFoundError(f"no run record at {self.file}")
        # the root is a name only a session record gives its main agent; here every agent joins by name
        self.participants.pop(self.root, None)
        self._offset = 0
        self._size = 0
        self._doff = 0
        #: name -> (block cursor, every seq delivered to it), from the delivery log
        self._delivered: dict[str, tuple[int, set[int]]] = {}
        self._roster_cache: Optional[tuple[Any, dict]] = None
        self.sync()

    # -- reading the shared files ------------------------------------------------------------------------------

    @property
    def path(self) -> Optional[Path]:
        return self.file

    def sync(self) -> int:
        """Read what other processes appended since the last look: entries and deliveries. Returns the new entries'
        count."""
        with self._lock:
            rows, offset = _whole_lines(self.file, self._offset)
            self._offset = offset
            try:
                self._size = os.path.getsize(self.file)
            except OSError:
                pass
            added = 0
            for row in rows:
                try:
                    entry = Entry.from_dict(row)
                except (ValueError, KeyError, TypeError):
                    continue
                if entry.seq <= self.last_seq:
                    continue  # a duplicate number is never taken twice
                self.entries.append(entry)
                added += 1
            drows, self._doff = _whole_lines(self.cursor_file, self._doff)
            for row in drows:
                self._note_delivery(row)
            return added

    def _note_delivery(self, row: dict) -> None:
        agent = row.get("agent")
        if not isinstance(agent, str):
            return
        upto, seqs = self._delivered.get(agent, (0, set()))
        got = row.get("upto")
        if isinstance(got, int) and not isinstance(got, bool):
            upto = max(upto, got)
        for s in row.get("seqs") or ():
            if isinstance(s, int) and not isinstance(s, bool):
                seqs.add(s)
        self._delivered[agent] = (upto, seqs)


    # -- the roster ---------------------------------------------------------------------------------------------

    def _ended_names(self) -> set[str]:
        rows, _ = _whole_lines(self.ended_file)
        return {r["name"] for r in rows if isinstance(r.get("name"), str)}

    def roster_doc(self) -> dict:
        """The roster with each agent's effective state (the file is parsed again only when it changed)."""
        try:
            st = os.stat(self.roster_file)
            key: Any = (st.st_ino, st.st_mtime_ns, st.st_size)
        except OSError:
            key = None
        cached = self._roster_cache
        if key is None or cached is None or cached[0] != key:
            cached = (key, _load_roster(self.roster_file))
            self._roster_cache = cached
        doc = json.loads(json.dumps(cached[1]))
        for row in doc["agents"]:
            row["state"] = effective_state(row)
        return doc

    def roster_rows(self, *, include_ended: bool = False) -> list[dict]:
        rows = self.roster_doc()["agents"]
        return rows if include_ended else [r for r in rows if r["state"] != "ended"]

    def _row(self, name: str) -> Optional[dict]:
        for row in self.roster_doc()["agents"]:
            if row["name"] == name:
                return row
        return None

    def _taken(self, name: str, doc: dict) -> bool:
        return (
            any(r["name"] == name for r in doc["agents"]) or name in self._ended_names()
        )

    def join(
        self,
        name: str,
        role: str,
        *,
        state: str = "running",
        episode: Optional[str] = None,
        pass_id: Optional[str] = None,
        spawner: Optional[str] = None,
        helps_with: Optional[str] = None,
        pid: Optional[int] = None,
        rejoin: bool = False,
    ) -> RunParticipant:
        """Put *name* on the roster and host it in this process. A name is never given to two agents of a run:
        one taken (by an agent on the roster or one that ended) is refused, unless *rejoin* (the writer service holds
        ``writer`` across passes and restarts, so it takes its own row back)."""
        if role not in AGENT_ROLES:
            raise ValueError(f"unknown role {role!r}; one of {', '.join(AGENT_ROLES)}")
        if state not in STATES or state == "ended":
            raise ValueError(f"an agent joins running, waiting or idle, not {state!r}")
        if (
            not valid_name(name)
            or name in (USER, HARNESS, ALL)
            or (name in ROLE_ADDRESSES and name != WRITER)
        ):
            raise PostRefused(f"{name!r} cannot be an agent's name on the run record")
        if name == WRITER and role != "writer":
            raise PostRefused("the name 'writer' is the writer's")
        pid = os.getpid() if pid is None else pid
        help_text = helps_with if helps_with is not None else HELPS_WITH.get(role, "")
        help_text = " ".join(clean_text(help_text).split())
        if len(help_text) > HELP_CHARS:
            help_text = help_text[:HELP_CHARS] + "…"
        mention = [dict(m) for m in SHOULD_MENTION.get(role, ())]   # who_text drops the roles the run does not have
        if role == "helper":
            mention = [
                {"who": spawner or "the agent that started you", "when": m["when"]}
                for m in mention
            ]
        row = {
            "name": name,
            "role": role,
            "state": state,
            "helps_with": help_text,
            "should_mention": mention,
            "can_mention": ["*"],
            "episode": episode,
            "pass_id": pass_id,
            "spawner": spawner,
            "pid": pid,
            "proc_start": proc_start(pid) or "",
            "joined_at": self._clock(),
        }
        with self._lock, _flock(self.roster_lock):
            doc = _load_roster(self.roster_file)
            if rejoin and name == WRITER:
                doc["agents"] = [r for r in doc["agents"] if r["name"] != name]
            elif self._taken(name, doc):
                raise PostRefused(f"the name {name!r} is taken in this run")
            doc["agents"].append(row)
            doc["v"] = VERSION
            _atomic_json(self.roster_file, doc)
            p = self.participants.get(name)
            if not isinstance(p, RunParticipant):
                upto, _seqs = self._delivered.get(name, (0, set()))
                p = RunParticipant(name, "agent", spawner=spawner, cursor=max(upto, 0))
                self.participants[name] = p
            p.__dict__.update(
                run_role=role, episode=episode, pass_id=pass_id, spawner=spawner
            )
            p.__dict__["state"] = "running"
            p.__dict__["record"] = self
            return p

    def attach(self, name: str) -> RunParticipant:
        """Host in this process an agent another process put on the roster for it (the fork's relay takes the fork
        the request named): its row now names this process, and its cursor is where the delivery log left it.
        """
        with self._lock, _flock(self.roster_lock):
            doc = _load_roster(self.roster_file)
            row = next((r for r in doc["agents"] if r["name"] == name), None)
            if row is None or row.get("state") == "ended":
                raise PostRefused(f"{name!r} is not on the run record")
            row["pid"] = os.getpid()
            row["proc_start"] = proc_start(os.getpid()) or ""
            _atomic_json(self.roster_file, doc)
            upto, _seqs = self._delivered.get(name, (0, set()))
            p = RunParticipant(
                name, "agent", spawner=row.get("spawner"), cursor=max(upto, 0)
            )
            p.__dict__.update(
                run_role=row.get("role") or "helper",
                episode=row.get("episode"),
                pass_id=row.get("pass_id"),
                record=self,
            )
            self.participants[name] = p
            return p

    def set_state(self, name: str, state: str, **fields: Any) -> None:
        """Change *name*'s roster row: its state (``ended`` moves it to the ended list's names too) and any of
        ``pass_id``, ``episode`` or ``note``."""
        if state not in STATES:
            raise ValueError(f"unknown state {state!r}")
        with self._lock, _flock(self.roster_lock):
            doc = _load_roster(self.roster_file)
            found = False
            for row in doc["agents"]:
                if row["name"] != name:
                    continue
                found = True
                if row.get("state") == "ended" and state != "ended":
                    break  # an agent that ended never comes back under its name
                row["state"] = state
                for key in ("pass_id", "episode", "note"):
                    if key in fields:
                        row[key] = fields[key]
                if state == "ended":
                    row["left_at"] = self._clock()
                    # an ended row keeps what reports need, not what an agent would act on
                    row.pop("should_mention", None)
                    row.pop("can_mention", None)
            if not found:
                return
            _atomic_json(self.roster_file, doc)
            if state == "ended":
                _append_line(
                    self.ended_file,
                    json.dumps({"v": VERSION, "name": name, "at": self._clock()}),
                )

    def leave(self, name: str) -> None:
        """*name* has ended: its roster row says so and its name stays taken."""
        p = self.participants.get(name)
        if isinstance(p, RunParticipant):
            p.__dict__["state"] = "stopped" if p.state == "running" else p.state
        self.set_state(name, "ended")

    def next_name(self, prefix: str) -> str:
        """The first ``<prefix><k>`` (k = 1, 2, ...) no agent of the run has had."""
        doc = _load_roster(self.roster_file)
        taken = (
            {r["name"] for r in doc["agents"]}
            | self._ended_names()
            | set(self.participants)
        )
        k = 1
        while f"{prefix}{k}" in taken:
            k += 1
        return f"{prefix}{k}"

    def unique_name(self, prefix: str, hexid: str, suffix: str = "") -> str:
        """``<prefix>-<4 hex of hexid><suffix>``, with more of its hex digits only when a clash in this run needs them
        (a segment's fork is ``fork-<hex>-s<n>``: *suffix* ``-s<n>``)."""
        # an episode id ``<stamp>-<hex>`` is named by its hex part; a bare hex id by itself
        digits = re.sub(
            r"[^0-9a-f]", "", (hexid or "").lower().rsplit("-", 1)[-1]
        ) or secrets.token_hex(8)
        doc = _load_roster(self.roster_file)
        taken = (
            {r["name"] for r in doc["agents"]}
            | self._ended_names()
            | set(self.participants)
        )
        for n in range(4, min(len(digits), _MAX_HEX) + 1, 2):
            name = f"{prefix}-{digits[:n]}{suffix}"
            if name not in taken:
                return name
        while True:
            name = f"{prefix}-{secrets.token_hex(4)}{suffix}"
            if name not in taken:
                return name

    def add_agent(self, name: str, *, spawner: str) -> Participant:
        """A helper this process starts (a pool's ``spawn``): on the roster under its name, with its starter's episode
        and pass."""
        starter = self.participants.get(spawner)
        episode = getattr(starter, "episode", None)
        pass_id = getattr(starter, "pass_id", None)
        return self.join(
            name, "helper", spawner=spawner, episode=episode, pass_id=pass_id
        )

    def live_agents(self) -> list[str]:
        """Every agent of the run that is running or waiting now."""
        return [
            r["name"]
            for r in self.roster_rows()
            if r["state"] in ("running", "waiting")
        ]

    # -- addressing -------------------------------------------------------------------------------------------

    def resolve_for(
        self, author: str, text: str
    ) -> tuple[list[str], list[str], list[str]]:
        """The agents *text* addresses (role addresses resolved to names, never the author), the addresses that match
        no agent of the run, and the named agents that have ended (no one reads for them now).
        """
        rows = self.roster_doc()["agents"]
        by_name = {r["name"]: r for r in rows}
        mine = by_name.get(author) or {}
        my_episode = mine.get("episode") or getattr(
            self.participants.get(author), "episode", None
        )
        live = [r for r in rows if r["state"] in ("running", "waiting")]
        resolved: list[str] = []
        unresolved: list[str] = []
        ended: list[str] = []

        def add(names: list[str]) -> None:
            for n in names:
                if n != author and n not in resolved:
                    resolved.append(n)

        for token in mention_tokens(text):
            if token == ALL:
                add([r["name"] for r in live])
            elif token == WRITER:
                if WRITER in by_name:
                    add([WRITER])
                    if by_name[WRITER]["state"] == "ended":
                        ended.append(WRITER)
                else:
                    unresolved.append(token)
            elif token == "actor":
                found = [r["name"] for r in live if r.get("role") == "actor"]
                add(found) if found else unresolved.append(token)
            elif token in INSTANCE_ROLES:
                # rev 2: the role itself, while the run has it: a question for its next instance (module docstring)
                add([token]) if role_seeded(self.home, token) else unresolved.append(token)
            elif token == "fork":
                found = [
                    r["name"]
                    for r in live
                    if r.get("role") == token
                    and my_episode
                    and r.get("episode") == my_episode
                ]
                add(found) if found else unresolved.append(token)
            elif token == "helpers":
                found = [r["name"] for r in live if r.get("role") == "helper"]
                add(found) if found else unresolved.append(token)
            elif token == USER and USER in self.participants:
                add([USER])
            elif token in by_name and token != HARNESS:
                add([token])
                if by_name[token]["state"] == "ended" and token not in ended:
                    ended.append(token)
            else:
                unresolved.append(token)
        return resolved, unresolved, ended

    # -- the roles' messages (rev 2) -----------------------------------------------------------------------------

    def role_messages(self, role: str) -> list[Entry]:
        """The entries addressed to *role* (the role in their mentions), never one an instance of the role wrote."""
        self.sync()
        mine = {r["name"] for r in self.roster_doc()["agents"] if r.get("role") == role}
        with self._lock:
            return [e for e in self.entries if role in e.mentions and e.author not in mine and e.role != role]

    def withdrawn(self, seq: int) -> bool:
        """Whether the author of entry *seq* withdrew it (a ``cancel`` entry of theirs that answers it)."""
        self.sync()
        with self._lock:
            asked = next((e for e in self.entries if e.seq == seq), None)
            return asked is not None and any(
                e.kind == "cancel" and e.reply_to == seq and e.author == asked.author for e in self.entries
            )

    def role_pending_for(self, name: str) -> bool:
        """Whether *name* has a question to a seeded role that is still to be answered (no reply to it, not withdrawn,
        no claim that ended it): something can still arrive for *name*."""
        roles = [r for r in INSTANCE_ROLES if role_seeded(self.home, r)]
        if not roles:
            return False
        self.sync()
        with self._lock:
            asked = [e for e in self.entries if e.author == name and any(r in e.mentions for r in roles)]
            if not asked:
                return False
            ended = {e.reply_to for e in self.entries if e.kind in ("reply", "cancel") and e.reply_to is not None}
        for role in roles:
            claims = role_claims(self.home, role)
            for e in asked:
                if role in e.mentions and e.seq not in ended and claims.get(e.seq, {}).get("state") not in (
                    "answered", "failed", "withdrawn"
                ):
                    return True
        return False

    def log_role_delivery(self, instance: str, seqs: list[int]) -> None:
        """Log that the role message(s) *seqs* were given to *instance* (the delivery log, as ``messages``)."""
        self._log_delivery(instance, list(seqs), "messages")

    def _resolve(self, text: str, author: str) -> tuple[list[str], list[str]]:
        resolved, unresolved, _ended = self.resolve_for(author, text)
        return resolved, unresolved

    def is_for(self, entry: Entry, name: str) -> bool:
        if entry.author == name:
            return False
        return self.options.delivery == "all" or name in entry.mentions

    # -- appending ----------------------------------------------------------------------------------------------

    def _accept(
        self,
        author: str,
        text: str,
        *,
        kind: str,
        mentions: Optional[list[str]],
        fit_at: Optional[str],
        reply_to: Optional[int] = None,
        refs: Optional[tuple[str, ...]] = None,
    ) -> tuple[Entry, list[str]]:
        if kind not in KINDS:
            raise PostRefused(f"unknown kind {kind!r}; use one of {', '.join(KINDS)}")
        text = clean_text(str(text))
        if fit_at is not None:
            from unify.agents.record import fit_text

            text = fit_text(text, fit_at)
        size = len(text.encode("utf-8"))
        if size > MAX_TEXT_BYTES:
            raise PostRefused(
                f"the entry is {size} bytes and the limit is {MAX_TEXT_BYTES}; "
                "write long content to a file and post its path",
            )
        if mentions is None:
            resolved, unresolved = self._resolve(text, author)
        else:
            resolved, unresolved = list(dict.fromkeys(mentions)), []
        p = self.participants.get(author)
        role = getattr(p, "run_role", None) or (
            "user" if author == USER else "harness" if author == HARNESS else "helper"
        )
        episode = getattr(p, "episode", None)
        pass_id = getattr(p, "pass_id", None)
        if refs is None:
            refs = refs_in(text, episode)
        acts = getattr(p, "acts", None) if isinstance(p, RunParticipant) else None
        with self._lock, _flock(self.lock_file):
            self._close_torn_line()
            self.sync()
            room = self.options.max_record_bytes
            if room is not None and self._size + size + _ROW_OVERHEAD > room:
                if fit_at is None:
                    raise PostRefused(
                        f"the run record has reached its size backstop ({room} bytes); "
                        "write what you would post to a file and keep it there",
                    )
                text = cut_to(
                    text,
                    CUT_TEXT_BYTES,
                    f"the run record reached its size backstop; the full text is in {fit_at}",
                )
            entry = Entry(
                seq=self.last_seq + 1,
                ts=self._clock(),
                author=author,
                kind=kind,
                text=text,
                mentions=tuple(resolved),
                role=role,
                episode=episode,
                pass_id=pass_id,
                reply_to=reply_to,
                refs=tuple(refs),
                acts=acts,
            )
            self._write_entry(entry)
            if author in self.participants:
                self.participants[author].last_seq = entry.seq
            notice = self._storm_notice(entry)
            waiters, self._waiters = self._waiters, []
            listeners = list(self._listeners)
        for future in waiters:
            _wake(future)
        for listener in listeners:
            for e in (entry, notice) if notice is not None else (entry,):
                try:
                    listener(e)
                except Exception:
                    logger.exception("record listener failed on entry %s", e.seq)
        return entry, unresolved

    def _close_torn_line(self) -> None:
        """Holding the file lock, before a number is taken: a last line a crash cut (no newline) is closed, so it
        becomes one whole line that every reader reads now (and skips, unless it happens to be a whole entry) and the
        next entry is numbered after whatever it was."""
        fd = os.open(self.file, os.O_RDWR | os.O_APPEND | os.O_CLOEXEC)
        try:
            end = os.lseek(fd, 0, os.SEEK_END)
            if end > 0 and os.pread(fd, 1, end - 1) != b"\n":
                _write_all(fd, b"\n")
                os.fsync(fd)
        finally:
            os.close(fd)

    def _write_entry(self, entry: Entry) -> None:
        """Write *entry* as one whole line at the end of the file (holding the file lock, the last line closed and
        read), then read it back into the entries."""
        line = json.dumps(row_of(entry), ensure_ascii=False)
        fd = os.open(self.file, os.O_RDWR | os.O_APPEND | os.O_CLOEXEC)
        try:
            _write_all(fd, line.encode("utf-8") + b"\n")
            os.fsync(fd)
        finally:
            os.close(fd)
        self.sync()
        if not self.entries or self.entries[-1].seq != entry.seq:
            raise PostRefused("the entry could not be read back from the run record")

    def append(
        self,
        author: str,
        text: str,
        *,
        kind: str = "post",
        mentions: Optional[list[str]] = None,
        reply_to: Optional[int] = None,
        refs: Optional[tuple[str, ...]] = None,
    ) -> tuple[Entry, list[str]]:
        return self._accept(
            author,
            text,
            kind=kind,
            mentions=mentions,
            fit_at=None,
            reply_to=reply_to,
            refs=refs,
        )

    # -- the storm notice -----------------------------------------------------------------------------------------

    def _storm_notice(self, entry: Entry) -> Optional[Entry]:
        """Called holding the file lock right after *entry* is written: when the last :data:`STORM_RUN` entries by its
        author and the one agent it addresses are only to each other, and neither made a tool call or a file read in
        between (their ``acts`` counts did not move), one harness notice to both. Once per run of that length.
        """
        if (
            entry.kind not in ("post", "reply")
            or len(entry.mentions) != 1
            or entry.acts is None
        ):
            return None
        a, b = entry.author, entry.mentions[0]
        pair = {a, b}
        chain: list[Entry] = []
        for e in reversed(self.entries):
            if (
                e.author == HARNESS
                and e.kind == "system"
                and set(e.mentions) == pair
                and STORM_MARK in e.text
            ):
                break  # the last notice to this pair: count from there
            if e.author not in pair:
                continue
            if e.kind not in ("post", "reply") or set(e.mentions) != pair - {e.author}:
                break
            chain.append(e)
        if len(chain) < STORM_RUN:
            return None
        recent = chain[:STORM_RUN]
        for who in pair:
            counts = {e.acts for e in recent if e.author == who}
            if not counts or None in counts or len(counts) > 1:
                return (
                    None  # one of them read or ran something in between: evidence moved
                )
        text = (
            f"@{a} @{b} {STORM_MARK} {len(chain)} times with no new evidence: neither of you made a tool call or "
            "read a file between these messages. Settle it from evidence (read the record or the files it names, "
            "run a test), or say what is still open and go on with your own work."
        )
        notice = Entry(
            seq=self.last_seq + 1,
            ts=self._clock(),
            author=HARNESS,
            kind="system",
            text=text,
            mentions=tuple(sorted(pair)),
            role="harness",
        )
        self._write_entry(notice)
        return notice

    def note_act(self, name: str) -> None:
        """*name* made a tool call or ran a cell that was not only messages: evidence for the storm notice."""
        with self._lock:
            p = self.participants.get(name)
            if isinstance(p, RunParticipant):
                p.__dict__["acts"] = p.acts + 1

    # -- delivery -------------------------------------------------------------------------------------------------

    def _log_delivery(
        self, name: str, seqs: list[int], via: str, upto: Optional[int] = None
    ) -> None:
        if via not in VIAS:
            raise ValueError(via)
        p = self.participants.get(name)
        row = {
            "v": VERSION,
            "agent": name,
            "role": getattr(p, "run_role", None),
            "seqs": sorted(set(seqs)),
            "via": via,
            "pass_id": getattr(p, "pass_id", None),
            "episode": getattr(p, "episode", None),
            "ts": self._clock(),
        }
        if upto is not None:
            row["upto"] = int(upto)
        with self._lock, _flock(self.lock_file):
            _append_line(self.cursor_file, json.dumps(row))
            self._note_delivery(row)

    def append_cursor(
        self, name: str, upto: int
    ) -> None:  # pragma: no cover - the session record's API
        self._log_delivery(name, [], "block", upto)

    def set_cursor(self, name: str, upto: int) -> None:
        """Mark entries up to *upto* as already seen by *name* (kept on disk, as a block row with no entries)."""
        with self._lock:
            self.participants[name].cursor = upto
        self._log_delivery(name, [], "block", upto)

    def read(
        self,
        *,
        since: int = 0,
        mentions: Optional[str] = None,
        author: Optional[str] = None,
        limit: int = 50,
    ) -> list[Entry]:
        self.sync()
        return super().read(since=since, mentions=mentions, author=author, limit=limit)

    def _unseen(self, name: str, mentions_only: bool) -> list[Entry]:
        self.sync()
        return super()._unseen(name, mentions_only)

    def take_unseen(self, name: str, via: str) -> list[Entry]:
        """The entries for *name* it has not been given, marked given and logged as delivered by *via*."""
        with self._lock:
            found = self._unseen(name, True)
            if not found:
                return []
            self.participants[name].returned.update(e.seq for e in found)
        self._log_delivery(name, [e.seq for e in found], via)
        return found

    def has_unseen(self, name: str) -> bool:
        """Whether an entry addressed to *name* has not been delivered to it (by the delivery log, whichever process
        hosts it)."""
        with self._lock:
            upto, seqs = self._delivered.get(name, (0, set()))
            local = self.participants.get(name)
            if isinstance(local, RunParticipant):
                upto = max(upto, local.cursor)
                seqs = seqs | local.returned
            return any(
                name in e.mentions
                and e.author != name
                and e.seq > upto
                and e.seq not in seqs
                for e in self.entries
            )

    def take_block(self, name: str) -> Optional[str]:
        """The boundary block for *name*: the entries for it since its cursor, then a line counting the others; the
        delivery log records which entries it was given."""
        self.sync()
        with self._lock:
            p = self.participants[name]
            new = [e for e in self.entries if e.seq > p.cursor and e.author != name]
            mine = [e for e in new if self.is_for(e, name)]
            if not mine:
                return None
            others = [e for e in new if not self.is_for(e, name)]
            block = render_block(
                mine, others, p.returned, cap_tokens=self.options.deliver_max_tokens
            )
            upto = self.last_seq
            shown = [e.seq for e in mine if e.seq not in p.returned]
            p.cursor = upto
            p.returned = {s for s in p.returned if s > upto}
        self._log_delivery(name, shown, "block", upto)
        return block

    def page_all(self, name: str, limit: int = 50) -> tuple[list[Entry], int]:
        """The next page of every entry of the record (any author but *name*) after what this has shown *name*,
        and how many more follow; entries for it on the page are logged as delivered."""
        self.sync()
        with self._lock:
            p = self.participants[name]
            after = [
                e for e in self.entries if e.seq > p.all_cursor and e.author != name
            ]
            page, more = after[:limit], max(0, len(after) - limit)
            if page:
                p.__dict__["all_cursor"] = page[-1].seq
                p.returned.update(e.seq for e in page if self.is_for(e, name))
        if page:
            self._log_delivery(name, [e.seq for e in page], "messages")
        return page, more

    # -- waiting --------------------------------------------------------------------------------------------------

    def nobody_can_answer(self, name: str) -> bool:
        """Whether every other agent that could answer *name* is waiting with nothing unread, idle or ended (rev 2: and no
        question of *name*'s to a role is still to be answered: its instance can still reply)."""
        self.sync()
        if self.role_pending_for(name):
            return False
        for row in self.roster_rows():
            other = row["name"]
            if other == name:
                continue
            if row["state"] == "running":
                return False
            if row["state"] == "waiting" and self.has_unseen(other):
                return False
        return True

    async def wait_outcome(
        self,
        name: str,
        *,
        timeout: float,
        deadline: Optional[float] = None,
        stop: Optional[Callable[[], Optional[str]]] = None,
    ) -> tuple[list[Entry], str]:
        """Entries for *name* it has not been given, and why the wait ended: ``message``, ``nothing_can_arrive``
        (:data:`ALL_WAITING`), ``asked_to_stop`` (*stop* gives a reason), ``wall_clock`` (*deadline*, a
        ``time.monotonic()`` time) or ``timeout`` (at most :data:`MAX_WAIT_S`)."""
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not timeout > 0
        ):
            raise ValueError(
                "timeout is required: a number of seconds above 0, for example timeout=300"
            )
        loop = asyncio.get_running_loop()
        end = time.monotonic() + min(float(timeout), MAX_WAIT_S)
        if deadline is not None:
            end = min(end, float(deadline))
        watch = _Watch(self.folder)
        poll = INOTIFY_BACKSTOP_S if watch.fd is not None else POLL_S
        p = self.participants[name]
        was = p.state
        if was == "running":
            self.set_state(name, "waiting")
        try:
            while True:
                found = self.take_unseen(name, "wait")
                if found:
                    return found, "message"
                why = stop() if stop is not None else None
                if why:
                    return [], "asked_to_stop"
                if self.nobody_can_answer(name):
                    return [], "nothing_can_arrive"
                left = end - time.monotonic()
                if left <= 0:
                    return [], (
                        "wall_clock"
                        if deadline is not None and end >= float(deadline)
                        else "timeout"
                    )
                future = loop.create_future()
                with self._lock:
                    self._waiters.append(future)
                if watch.fd is not None:
                    loop.add_reader(watch.fd, _wake, future)
                try:
                    await asyncio.wait_for(asyncio.shield(future), min(left, poll))
                except asyncio.TimeoutError:
                    pass
                finally:
                    if watch.fd is not None:
                        loop.remove_reader(watch.fd)
                        watch.drain()
                    with self._lock:
                        if future in self._waiters:
                            self._waiters.remove(future)
        finally:
            watch.close()
            if was == "running" and self.participants[name].state == "running":
                self.set_state(name, "running")

    async def wait_for(
        self, name: str, *, timeout: float, mentions_only: bool = True
    ) -> list[Entry]:
        found, _why = await self.wait_outcome(name, timeout=timeout)
        return found


def open_run_record(
    home: Path | str,
    *,
    create: bool,
    options: Options = RUN_OPTIONS,
    clock: Callable[[], str] = _now,
    monotonic: Callable[[], float] = time.monotonic,
) -> RunRecord:
    """The run record of the memory home *home*. ``create=True`` makes a new record and fails if one exists (two runs
    never share a record); ``create=False`` opens the run's record and fails if there is none.
    """
    return RunRecord(
        home, create=create, options=options, clock=clock, monotonic=monotonic
    )


def join_run_record(home: Path | str, *, options: Options = RUN_OPTIONS) -> RunRecord:
    """The run record of *home*, made by the first agent of the run that opens it (the file is created exclusively, so
    two processes starting together get one record)."""
    try:
        return open_run_record(home, create=True, options=options)
    except FileExistsError:
        return open_run_record(home, create=False, options=options)


# --- the block an agent is shown at its turn boundary --------------------------------------------------------------

_TAGS = {"post": "", "reply": " (reply)", "cancel": " (stop)", "system": " (notice)"}


def _span(entries: list[Entry]) -> str:
    lo, hi = entries[0].seq, entries[-1].seq
    return f"#{lo}" if lo == hi else f"#{lo}–#{hi}"


def render_block(
    mine: list[Entry], others: list[Entry], returned: set[int], *, cap_tokens: int
) -> str:
    """The session record's block (:func:`unify.agents.delivery.render_block`), pointing at the comms tools."""
    lines: list[str] = []
    overflow: list[Entry] = []
    used = 0
    for entry in mine:
        tag = _TAGS.get(entry.kind, "")
        priority = entry.author == USER or entry.kind in ("cancel", "system")
        if entry.seq in returned and not priority:
            line = f"#{entry.seq} {entry.author}{tag} — already returned to your code"
        else:
            line = f"#{entry.seq} {entry.author}{tag}: {entry.text}"
        cost = len(line) // 4 + 1
        if not priority and lines and used + cost > cap_tokens:
            overflow.append(entry)
            continue
        lines.append(line)
        used += cost
    shown = len(mine) - len(overflow)
    both = sorted(mine + others, key=lambda e: e.seq)
    out = [f"[record · {shown} new for you · {_span(both)}]", *lines]
    if overflow:
        out.append(
            f"(+{len(overflow)} more for you, {_span(overflow)}; tools.messages() shows them)"
        )
    if others:
        out.append(
            f"(+{len(others)} other entries, {_span(others)}; tools.messages(all=True) shows every entry)"
        )
    return "\n".join(out)


# --- one agent's handle -------------------------------------------------------------------------------------------


class RunRecordView:
    """One agent's handle on the run record, as the comms tools use it (:class:`unify.common.tool_registry.Record`):
    ``post``, ``unseen``, ``wait``, ``roster``, plus ``messages_all``, ``wait_outcome``, ``take_block`` and
    ``note_act``. ``start`` and ``stop`` go to the pool that starts this agent's helpers, when it has one.
    """

    def __init__(
        self,
        record: RunRecord,
        name: str,
        role: str,
        *,
        pool: Any = None,
        spawner: Optional[str] = None,
    ):
        self.record, self.name, self.role, self.pool, self.spawner = (
            record,
            name,
            role,
            pool,
            spawner,
        )

    # the protocol ---------------------------------------------------------------------------------------------------

    def post(
        self,
        sender: str,
        text: str,
        to: Optional[list] = None,
        *,
        reply_to: Optional[int] = None,
        default_to: Optional[str] = None,
    ) -> dict:
        """Post *text* from *sender* (``@`` names in *to* go first). A message that addresses no agent goes to
        *default_to* when given, and is otherwise refused (``no_recipient``): nobody would read it. Rev 2: a post that
        names an agent that has ended (the writer apart), or a role this run does not have, is refused with a pointer
        (module docstring); nothing is posted.
        """
        from unify.common.tool_registry import RecordRefused

        rec = self.record
        if to:
            text = " ".join(f"@{n}" for n in to) + " " + str(text)
        text = str(text)
        resolved, unresolved, ended = rec.resolve_for(sender, text)
        missing = [r for r in INSTANCE_ROLES if r in unresolved]
        if missing:
            role = missing[0]
            running = [r["name"] for r in rec.roster_doc()["agents"]
                       if r.get("role") == role and r.get("state") in ("running", "waiting")]  # fmt: skip
            # worded from the record as it is now: the role is not kept on it, though an instance may be running
            why = (
                f"@{role} reaches no one: this run's record keeps no {role} role, and a running {role} is "
                "addressed by its name"
                if running
                else f"there is no {role} in this run, so nobody would read @{role}"
            )
            raise RecordRefused("no_role", f"{why}; " + _address_hint("address one of", self.addresses(sender)))
        stale = [n for n in ended if n != WRITER]
        if stale:
            raise RecordRefused("ended", "ended: " + "; ".join(self._ended_pointer(n) for n in stale))
        if not resolved:
            if default_to:
                resolved = [default_to]
            else:
                names = self.addresses(sender)
                raise RecordRefused(
                    "no_recipient",
                    "the message addresses no agent on the record"
                    + (
                        f" ({', '.join('@' + n for n in unresolved)} match none)"
                        if unresolved
                        else ""
                    )
                    + "; " + _address_hint("address it with", names),
                )
        try:
            rec.check_rate(sender)
            entry, _ = rec.append(sender, text, mentions=resolved, reply_to=reply_to)
        except PostRefused as exc:
            from unify.common.record_adapters import _post_form

            raise RecordRefused(_post_form(exc), str(exc)) from None
        return {
            "seq": entry.seq,
            "to": list(entry.mentions),
            "unresolved": list(unresolved),
            "ended": list(ended),
        }

    def unseen(self, reader: str) -> list:
        return [row_of(e) for e in self.record.take_unseen(reader, "messages")]

    async def wait(self, reader: str, timeout: float) -> list:
        found, _why = await self.record.wait_outcome(
            reader, timeout=min(float(timeout), MAX_WAIT_S)
        )
        return [row_of(e) for e in found]

    def roster(self) -> list:
        return [
            {
                "name": r["name"],
                "role": r.get("role"),
                "state": r["state"],
                "help": r.get("helps_with", ""),
            }
            for r in self.record.roster_rows()
        ]

    async def start(self, spawner: str, task: str, **kw: Any) -> dict:
        from unify.common.tool_registry import RecordRefused

        if self.pool is None or not hasattr(self.pool, "spawn"):
            raise RecordRefused("spawn_off", "this agent starts no helpers")
        from unify.common.record_adapters import _spawn_form

        try:
            name = await self.pool.spawn(spawner, task, kw.get("name"))
        except Exception as exc:  # the pool's own refusals, with the reason
            raise RecordRefused(_spawn_form(exc), str(exc)) from None
        return {"name": name}

    async def stop(self, by: str, name: str) -> dict:
        from unify.common.record_adapters import AgentRecordAdapter
        from unify.common.tool_registry import RecordRefused

        if self.pool is None:
            raise RecordRefused("not_allowed", "this agent started no helpers")
        return await AgentRecordAdapter(self.pool).stop(by, name)

    # beyond the protocol ------------------------------------------------------------------------------------------

    def _ended_pointer(self, name: str) -> str:
        """Why a post to *name* (ended) is refused, and where to go instead: an instance of a role -> the role; a fork ->
        when it ended; any other agent -> when it ended."""
        row = self.record._row(name) or {}
        role, at = row.get("role"), row.get("left_at") or "an earlier time"
        if role in INSTANCE_ROLES:
            return f"@{name} ended at {at}; address the role: @{role}"
        if role == "fork":
            return f"the fork @{name} ended at {at}"
        return f"@{name} ended at {at}"

    def addresses(self, sender: Optional[str] = None) -> str:
        """The addresses that reach someone now, for a refusal's text: the running agents by name, and each role that
        resolves to an agent now (the record's own resolution), never one that would match no one. Without memory no
        writer, fork or librarian is offered; an agent inside a request (the actor, a teammate) is never offered its
        episode's fork, which starts after the episode."""
        sender = sender or self.name
        names = [f"@{n}" for n in self.record.live_agents() if n != sender]
        memory = run_has_memory(self.record.home, self.role if sender == self.name else None,
                                self.spawner if sender == self.name else None)
        inside = works_in_request(self.record.roster_doc()["agents"], sender)
        roles = []
        for r in ROLE_ADDRESSES:
            if (r in _registry.MEMORY_ROLES and not memory) or (r == "fork" and inside):
                continue
            reached, unmatched, _ended = self.record.resolve_for(sender, f"@{r}")
            if reached and not unmatched:
                roles.append(f"@{r}")
        return ", ".join(dict.fromkeys(names + roles))

    def messages_all(self, limit: int = 50) -> tuple[list[dict], int]:
        page, more = self.record.page_all(self.name, limit)
        return [row_of(e) for e in page], more

    async def wait_outcome(
        self,
        timeout: float,
        *,
        deadline: Optional[float] = None,
        stop: Optional[Callable[[], Optional[str]]] = None,
    ) -> tuple[list[dict], str]:
        found, why = await self.record.wait_outcome(
            self.name, timeout=timeout, deadline=deadline, stop=stop
        )
        return [row_of(e) for e in found], why

    def take_block(self) -> Optional[str]:
        return self.record.take_block(self.name)

    def note_act(self) -> None:
        self.record.note_act(self.name)

    def last_from(self, author: str) -> Optional[int]:
        """The ``seq`` of the latest entry *author* addressed to this agent (what a reply to it answers), or None."""
        self.record.sync()
        for e in reversed(self.record.entries):
            if e.author == author and self.name in e.mentions:
                return e.seq
        return None


__all__ = [
    "ALL_WAITING",
    "CLAIM_STATES",
    "INSTANCE_ROLES",
    "RoleBusy",
    "claim",
    "keep_role_answer",
    "role_answers",
    "role_claims",
    "role_dir",
    "role_event",
    "role_lock",
    "role_notes",
    "role_seeded",
    "seed_role",
    "COMMON_SECTION",
    "COMMON_SECTION_NO_MEMORY",
    "run_has_memory",
    "MAX_RECORD_BYTES",
    "ROLES",
    "RUN_OPTIONS",
    "RunParticipant",
    "RunRecord",
    "RunRecordView",
    "SHOULD_MENTION",
    "STATES",
    "STORM_RUN",
    "agents_dir",
    "deliveries_path",
    "join_run_record",
    "open_run_record",
    "prompt_lines",
    "read_deliveries",
    "read_entries",
    "record_path",
    "refs_in",
    "row_of",
    "mention_line",
    "works_in_request",
]
