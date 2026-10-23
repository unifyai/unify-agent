"""The attention ledger: everything that needs the writer's attention, read at least once.

**What it is.** ``<state>/memory-attention/ledger.jsonl`` is an append-only list of rows, written by the harness only
(every write holds ``flock(ledger.lock)``, so the writer service and the request processes may both write; no box binds
the directory). A row holds ids and kinds only, never content: the reason text of a decision stays in the pass's own
``decisions.jsonl``. The ledger's state is the fold of its rows (:meth:`Ledger.entries`).

**What needs attention** (an *entry*; each kind fires from a record that exists in any world where its event happens,
and from nothing else: no world name, label, id of a request, stream position or request word is read):

=====================  ==================================================================================================
kind                   created when
=====================  ==================================================================================================
``new_experience``     an episode's (or a segment's) fork settled with any output: the summary, plus the procedure
``own_check_failed``   never made under rev 2 (no capture is run to classify it); kept so an older ledger still folds
``use_went_badly``     a use of an item was failed (an error, ``holds`` false, a contrary signal after it) or refused (the
                       item refused the call, ``MemoryInputError``), or an actor wrote to the writer about it on the run
                       record (an actor entry to the writer with an ``item:`` ref, or naming by its library name an item
                       its episode used: :func:`named_items`)
``late_evidence``      a signal arrived after its episode ended
``repeated_use``       an item was called in 2 or more new episodes since the writer last read it
``deferred``           the writer deferred an entry, or the gate refused a draft
``message``            a run-record entry mentions the writer, or replies to a writer entry
=====================  ==================================================================================================

An episode whose capture is the same capture as an entry's (the same id: byte-identical, or folded into it by its
shape address, :mod:`.capture_shape`) is never made a new entry. Onto an active entry it is appended (``append_ref``);
onto a closed one it is counted only (an ``occurrence`` row: no new entry, nothing handed again), unless its outcome
is contrary (from its signals, never from running it), which reopens the entry (``append_ref``, then ``reopen`` by
``contrary_occurrence``).

**Lifecycle**: ``open`` -> ``handed`` (to a pass, :meth:`Ledger.hand`, or pulled in by the writer,
:meth:`Ledger.take`) -> ``read`` (verified from the pass's own read evidence, :meth:`Ledger.verify_reads`; never from
what the writer says) -> ``closed`` (with a decision, :meth:`Ledger.decide`). Handed but not read when the pass ends,
an entry goes back (``return``); after two such misses it is handed alone; after a third it is ``unreadable``, with
the evidence. A capture entry of a group's pass is never handed alone and never ends unreadable: at that bound the
harness closes it (:data:`GIVEN_UP`), its capture stays served, and it is history for any later related capture; a
group whose members came back yields to the other due groups. Read but undecided, it goes back without a miss. :meth:`Ledger.mark_not_reached` ends what the run could
not reach (``spend_cap``, ``run_ended``, ``service_stopped``, ``lifetime``). The guarantee: at run end every entry is
``closed``, ``unreadable`` or ``not_reached``; :meth:`Ledger.report` says so per kind, with the creation-to-read delay.
The end of one stage of a run that goes on (a ``suspend``) is not the run's end: nothing is marked then, and the next
start hands every entry still owed again (those a stopped pass held go back with ``suspend``, no miss; those a pass
the service did not end held, found at the next start, with ``service_restart``).

**Waiting for an episode.** An entry is handed only when every episode it names can be read (:func:`episode_resolver`):
an episode is readable once it is recorded (indexed), and a running one through its latest in-episode segment record
(``<episode>.s<n>``) when it has one. An entry naming a running episode with no segment record (a message posted
mid-episode cites its sender's episode) stays ``open``, with one ``wait`` row as its evidence: it is not handed, so no
pass can miss it, and it is handed once the episode is recorded. What still waits at the run's end is ``not_reached``.

**Hand order** (:data:`HAND_ORDER`): bad news first, then the time-sensitive messages, oldest first within each kind.
The age backstop: when the oldest open entry of any kind is older than the 90th-percentile age-at-hand of the entries
handed in the last :data:`STARVATION_PASSES` passes, it is handed first. :data:`K` entries per pass, the same
everywhere; the writer may take more.

**Capture groups** (:mod:`.capture_groups`): the writer service hands with a grouping, so a ``new_experience``
entry with a capture waits until a related capture exists (another new one, or its history: earlier related ones),
and a pass is then handed that group's new members with a ``group`` row naming its history. A capture with no related
capture is never handed alone; an entry whose episode left no capture is closed by the harness and runs no pass. A
judge's ``kin`` rows (:meth:`Ledger.record_kin`) are one of the relations. Every other kind is still handed at once.

**Reopening.** A ``late_evidence`` entry reopens every closed entry that names one of the items its episode used or
produced, when the signal is newer than that entry's last close; each reopen names its cause.

Standard library only, apart from the readers of the harness's own stores in :class:`HomeSources`.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import posixpath
import re
import stat
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, Mapping, Protocol, get_args

# --- vocabulary ------------------------------------------------------------------------------------------------------

Kind = Literal[
    "new_experience",
    "own_check_failed",
    "use_went_badly",
    "late_evidence",
    "repeated_use",
    "deferred",
    "message",
]
Decision = Literal["new_item", "changed_item", "merged_into", "note", "retire", "nothing_to_keep", "defer"]
NotReached = Literal["spend_cap", "run_ended", "service_stopped", "lifetime"]
ReturnWhy = Literal["handed_not_read", "read_not_decided", "suspend", "service_restart"]

KINDS: tuple[str, ...] = get_args(Kind)
DECISIONS: tuple[str, ...] = get_args(Decision)
NOT_REACHED: tuple[str, ...] = get_args(NotReached)
RETURN_WHYS: tuple[str, ...] = get_args(ReturnWhy)
#: The decisions that must name the item they rest on.
NEEDS_ITEMS = frozenset({"merged_into", "changed_item"})
#: Bad news first; a question from a running agent is time-sensitive, so ``message`` comes right after it.
HAND_ORDER: tuple[str, ...] = (
    "late_evidence",
    "use_went_badly",
    "message",
    "own_check_failed",
    "new_experience",
    "deferred",
    "repeated_use",
)
#: Entries handed to one pass; a run constant, the same in every world.
K = 4
#: The passes whose hands set the age backstop.
STARVATION_PASSES = 10
#: Misses (handed, pass ended, not read) after which an entry is handed alone, and after which it is unreadable.
ALONE_AFTER_MISSES = 2
UNREADABLE_AFTER_MISSES = 3
#: The reason the harness closes a capture entry of a group the writer could not finish in
#: :data:`UNREADABLE_AFTER_MISSES` passes (:meth:`Ledger.return_unread`): its capture stays served, and it becomes the
#: history of later related captures; it never ends unreadable and never holds later groups up.
GIVEN_UP = "group_given_up"  # the same as :data:`.capture_groups.GIVEN_UP`
#: Episodes calling an item since the writer last read it that make a ``repeated_use`` entry.
REPEATED_USE_EPISODES = 2

LEDGER_DIR = "memory-attention"
LEDGER_FILE = "ledger.jsonl"
LOCK_FILE = "ledger.lock"
#: Cached facts that are costly to read again (episode end times); the ledger never depends on it for correctness.
SYNC_FILE = "sync-state.json"
SCHEMA = 1

ATTENTION_INPUTS = "/inputs/attention"
STAGING_INPUTS = "/inputs/staging"
CAPTURES_INPUTS = "/inputs/captures"
DRAFTS_INPUTS = "/inputs/drafts"
MEMORY_ROOT = "/memory"
CAPTURE_PREFIX = "capture:"
#: The pass's read evidence, written at its end beside its other records (:func:`write_pass_reads`).
READ_LOG_FILE = "read-log.json"
DECISIONS_FILE = "decisions.jsonl"

_ID_CHARS = frozenset("0123456789abcdef")


def _group_reasons(group: Any) -> dict:
    """Why *group* formed (:func:`.capture_groups.reasons`): its related pairs and how many each relation kind joined."""
    from .capture_groups import reasons

    return reasons(getattr(group, "relations", None) or [])


class LedgerError(ValueError):
    """A ledger call with arguments it cannot take; the text says what was given and what is expected."""


# --- time, ids, rows -------------------------------------------------------------------------------------------------


def now_iso(at: dt.datetime | None = None) -> str:
    """UTC ISO-8601 with milliseconds and a ``Z``."""
    at = at or dt.datetime.now(dt.UTC)
    return at.astimezone(dt.UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_time(value: Any) -> dt.datetime | None:
    """An ISO time as an aware UTC datetime (``Z`` read as UTC; a naive time is UTC), else None."""
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.UTC)
    if not isinstance(value, str) or not value:
        return None
    try:
        at = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return at if at.tzinfo is not None else at.replace(tzinfo=dt.UTC)


def entry_id(kind: str, *source: Any) -> str:
    """The 16-hex id of the entry a source makes: deterministic, so a source never makes two entries."""
    text = "\0".join([kind, *(str(s) for s in source)])
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def is_entry_id(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 16 and set(value) <= _ID_CHARS


@dataclass(frozen=True)
class Refs:
    """What an entry points at, ids only."""

    episodes: tuple[str, ...] = ()
    segment: int | None = None
    captures: tuple[str, ...] = ()
    items: tuple[str, ...] = ()
    signals: tuple[str, ...] = ()
    record_seq: int | None = None
    passes: tuple[str, ...] = ()

    def row(self) -> dict:
        return {
            "episodes": list(self.episodes),
            "segment": self.segment,
            "captures": list(self.captures),
            "items": list(self.items),
            "signals": list(self.signals),
            "record_seq": self.record_seq,
            "passes": list(self.passes),
        }

    @classmethod
    def from_row(cls, row: Any) -> "Refs":
        row = row if isinstance(row, Mapping) else {}

        def ids(key: str) -> tuple[str, ...]:
            v = row.get(key)
            return tuple(dict.fromkeys(x for x in v if isinstance(x, str))) if isinstance(v, list) else ()

        seg, seq = row.get("segment"), row.get("record_seq")
        return cls(
            ids("episodes"),
            seg if isinstance(seg, int) and not isinstance(seg, bool) else None,
            ids("captures"),
            ids("items"),
            ids("signals"),
            seq if isinstance(seq, int) and not isinstance(seq, bool) else None,
            ids("passes"),
        )

    def merged(self, more: "Refs") -> "Refs":
        """These refs with *more*'s ids appended (no duplicates; the first segment and record seq kept)."""

        def join(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...]:
            return tuple(dict.fromkeys((*a, *b)))

        return Refs(
            join(self.episodes, more.episodes),
            self.segment if self.segment is not None else more.segment,
            join(self.captures, more.captures),
            join(self.items, more.items),
            join(self.signals, more.signals),
            self.record_seq if self.record_seq is not None else more.record_seq,
            join(self.passes, more.passes),
        )


@dataclass(frozen=True)
class AttentionEntry:
    """One entry, as the fold of its rows."""

    entry: str
    kind: str
    state: str
    created_at: str
    refs: Refs
    #: times handed to a pass (by the harness or taken by the writer)
    hand_count: int = 0
    #: passes that ended with it handed and not read since it was last opened (a reopen starts again at 0)
    misses: int = 0
    #: passes that ended with it read and not decided since it was last opened (a reopen starts again at 0)
    undecided: int = 0
    #: the pass it is handed to, while handed or read
    pass_id: str | None = None
    #: the parts of its material the pass's read evidence showed opened, and their share
    parts_read: tuple[str, ...] = ()
    share: str | None = None
    first_read_at: str | None = None
    decision: str | None = None
    decided_items: tuple[str, ...] = ()
    reason_ref: str | None = None
    closed_at: str | None = None
    reopen_count: int = 0
    #: the last return's or not-reached reason, or the unreadable evidence
    why: str | None = None
    #: the source ids appended to it (duplicates of its capture)
    sources: tuple[str, ...] = ()
    handed_at: str | None = None
    #: later occurrences of its capture counted while it was closed (``occurrence`` rows)
    occurrences: int = 0
    #: the entries a judge of behaviour named as doing the same kind of work (``kin`` rows): a grouping relation
    kin: tuple[str, ...] = ()
    #: whether a judge was asked about it (a ``kin`` row, with or without kin): it is asked once
    kin_judged: bool = False

    @property
    def active(self) -> bool:
        """Still owed something: open, handed or read (not closed, unreadable or not reached)."""
        return self.state in ("open", "handed", "read")


def _ledger_dir_of(home: Path) -> Path:
    from .integration.paths import Paths

    return Paths.under(Path(home)).state_dir / LEDGER_DIR


def _append_lines(path: Path, rows: list[dict]) -> None:
    """Append *rows* as JSON lines with ``O_APPEND`` and ``fsync``, never through a link."""
    if not rows:
        return
    data = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise OSError(f"{path.name} is not a regular file")
        while data:
            data = data[os.write(fd, data) :]
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_rows(path: Path) -> list[dict]:
    """Every whole JSON object line of *path* (a line cut by a crash, or not an object, is skipped)."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except FileNotFoundError:
        return []
    with os.fdopen(fd, "rb") as fh:
        if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
            return []
        data = fh.read()
    out = []
    for line in data.decode("utf-8", "replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def fold(rows: Iterable[Mapping[str, Any]]) -> dict[str, AttentionEntry]:
    """The entries the ledger's *rows* describe, by id, in creation order. Rows for an unknown entry, of an unknown
    op or with unknown values are ignored (readers ignore what they do not know)."""
    out: dict[str, AttentionEntry] = {}
    for row in rows:
        op, eid = row.get("op"), row.get("entry")
        at = row.get("at") if isinstance(row.get("at"), str) else None
        if op == "create":
            if not is_entry_id(eid) or eid in out or row.get("kind") not in KINDS or at is None:
                continue
            out[eid] = AttentionEntry(eid, row["kind"], "open", at, Refs.from_row(row.get("refs")))
            continue
        e = out.get(eid) if isinstance(eid, str) else None
        if e is None:
            continue
        if op == "append_ref":
            src = row.get("src")
            out[eid] = replace(
                e,
                refs=e.refs.merged(Refs.from_row(row.get("refs"))),
                sources=e.sources + ((src,) if isinstance(src, str) else ()),
            )
        elif op == "kin":
            named = row.get("kin")
            if isinstance(named, list):
                more = tuple(k for k in named if is_entry_id(k) and k != eid and k not in e.kin)
                out[eid] = replace(e, kin=e.kin + more, kin_judged=True)
        elif op == "occurrence":
            src = row.get("src")
            out[eid] = replace(
                e, sources=e.sources + ((src,) if isinstance(src, str) else ()), occurrences=e.occurrences + 1
            )
        elif op == "hand":
            if not e.active:
                continue
            n = row.get("hand_count")
            out[eid] = replace(
                e,
                state="handed" if e.state == "open" else e.state,
                pass_id=row.get("pass_id") if isinstance(row.get("pass_id"), str) else None,
                hand_count=n if isinstance(n, int) and not isinstance(n, bool) else e.hand_count + 1,
                handed_at=at,
            )
        elif op == "read":
            if not e.active:
                continue
            parts = row.get("parts_read")
            out[eid] = replace(
                e,
                state="read",
                parts_read=tuple(p for p in parts if isinstance(p, str)) if isinstance(parts, list) else (),
                share=row.get("share") if isinstance(row.get("share"), str) else None,
                first_read_at=e.first_read_at or at,
            )
        elif op == "return":
            if not e.active:
                continue
            why = row.get("why") if row.get("why") in RETURN_WHYS else "handed_not_read"
            out[eid] = replace(
                e,
                state="open",
                pass_id=None,
                misses=e.misses + (1 if why == "handed_not_read" else 0),
                undecided=e.undecided + (1 if why == "read_not_decided" else 0),
                why=why,
            )
        elif op == "close":
            if not e.active or row.get("decision") not in DECISIONS:
                continue
            items = row.get("items")
            out[eid] = replace(
                e,
                state="closed",
                decision=row["decision"],
                decided_items=tuple(i for i in items if isinstance(i, str)) if isinstance(items, list) else (),
                reason_ref=row.get("reason_ref") if isinstance(row.get("reason_ref"), str) else None,
                closed_at=at,
                pass_id=row.get("pass_id") if isinstance(row.get("pass_id"), str) else e.pass_id,
                why=row.get("why") if isinstance(row.get("why"), str) else e.why,
            )
        elif op == "reopen":
            if e.state not in ("closed", "unreadable", "not_reached"):
                continue
            out[eid] = replace(
                e,
                state="open",
                pass_id=None,
                misses=0,
                undecided=0,
                reopen_count=e.reopen_count + 1,
                why=row.get("by") if isinstance(row.get("by"), str) else None,
            )
        elif op == "unreadable":
            if not e.active:
                continue
            out[eid] = replace(e, state="unreadable", pass_id=None, why=str(row.get("evidence") or ""))
        elif op == "not_reached":
            if not e.active or row.get("why") not in NOT_REACHED:
                continue
            out[eid] = replace(e, state="not_reached", pass_id=None, why=row["why"])
    return out


# --- the material an entry owes a read, and what a pass read ---------------------------------------------------------


def item_box_paths(item: str) -> tuple[str, ...]:
    """Where an item is in the writer's box: a capture's procedure under /inputs/captures, a function's module or a
    note's file under /memory."""
    if item.startswith(CAPTURE_PREFIX):
        cid = item[len(CAPTURE_PREFIX) :]
        return (f"{CAPTURES_INPUTS}/{cid}/procedure.py",)
    from .layout import item_path

    return (posixpath.normpath(f"{MEMORY_ROOT}/{item_path(item)}"),)


@dataclass(frozen=True)
class Part:
    """One part of an entry's material: satisfied by opening any of *paths*, reading any part of any of
    *episodes* (through ``read_episode`` or ``overview``), or receiving any of *seqs* on the run record."""

    name: str
    paths: tuple[str, ...] = ()
    episodes: tuple[str, ...] = ()
    seqs: tuple[int, ...] = ()
    required: bool = True

    def row(self) -> dict:
        return {
            "name": self.name,
            "paths": list(self.paths),
            "episodes": list(self.episodes),
            "seqs": list(self.seqs),
            "required": self.required,
        }


def staging_name(episode: str, segment: int | None) -> str:
    """The staging directory of an episode, or of one of its segments (``<episode>.s<n>``)."""
    return episode if segment is None else f"{episode}.s{int(segment)}"


def _summary_paths(names: Iterable[str]) -> tuple[str, ...]:
    return tuple(f"{STAGING_INPUTS}/{n}/summary.md" for n in names)


def material(
    entry: AttentionEntry,
    *,
    has_summary: Callable[[str], bool] = lambda _n: True,
    has_procedure: Callable[[str], bool] = lambda _n: True,
) -> list[Part]:
    """The parts of *entry*'s material, required and optional. *has_summary* says whether a staging dir holds a
    summary (where it holds none, reading any part of the episode stands for it); *has_procedure* whether it holds a
    ``procedure.py``. The material names only files the pass's inputs hold: an episode's procedure reaches the writer
    by one route, its capture's ``procedure.py`` (the staging copy is left out of the inputs when the capture holds
    it), and a staged ``procedure.py`` is named only for an entry with no capture, where it exists."""
    r = entry.refs
    names = [staging_name(e, r.segment) for e in r.episodes]
    own = f"{ATTENTION_INPUTS}/{entry.entry}"
    proc_paths = tuple(f"{CAPTURES_INPUTS}/{c}/procedure.py" for c in r.captures) or tuple(
        f"{STAGING_INPUTS}/{n}/procedure.py" for n in names if has_procedure(n)
    )
    with_summary = [n for n in names if has_summary(n)]
    summary = Part(
        "summary",
        _summary_paths(with_summary) + tuple(f"{STAGING_INPUTS}/{n}/" for n in names if n not in with_summary),
        tuple(names) + tuple(r.episodes),
    )
    items = tuple(p for i in r.items for p in item_box_paths(i))
    kind = entry.kind
    if kind == "new_experience":
        parts = [summary]
        if r.captures:
            parts.append(Part("procedure.py", proc_paths))
        parts.append(Part("episode", (), tuple(names) + tuple(r.episodes), required=False))
        return parts
    if kind == "own_check_failed":
        return [
            Part("procedure.py", proc_paths),
            Part("capture record", tuple(f"{CAPTURES_INPUTS}/{c}/record.json" for c in r.captures)),
            Part("summary", summary.paths, summary.episodes, required=False),
        ]
    if kind == "use_went_badly":
        return [
            Part("use row", (f"{own}/use.json",)),
            Part("item", items),
            Part("summary", summary.paths, summary.episodes, required=False),
        ]
    if kind == "late_evidence":
        return [
            Part("signal", (f"{own}/signal.json",)),
            Part("summary", summary.paths, summary.episodes),
            Part("items", items, required=False),
        ]
    if kind == "repeated_use":
        return [Part("item", items), Part("episode", summary.paths, summary.episodes)]
    if kind == "deferred":
        return [Part("note", (f"{own}/note.json",) + tuple(f"{DRAFTS_INPUTS}/{p}/gate.md" for p in r.passes))]
    if kind == "message":
        return [Part("entry", (), (), (r.record_seq,) if r.record_seq is not None else ())]
    return []


def _norm(path: str) -> str:
    keep_dir = path.endswith("/")
    out = posixpath.normpath(path)
    return out + "/" if keep_dir and not out.endswith("/") else out


@dataclass(frozen=True)
class PassReads:
    """What one pass's agents read, from the harness's own evidence (never the writer's words): the box paths opened
    for reading (``read`` tool calls, other tool rows with mode ``read``, and the boxes' open rows), and the episode
    parts each agent was shown (``read_episode``, ``overview``)."""

    paths: frozenset[str] = frozenset()
    episodes: Mapping[str, frozenset[str]] = field(default_factory=dict)

    def opened(self, path: str) -> bool:
        if path.endswith("/"):
            return any(p.startswith(path) for p in self.paths)
        return _norm(path) in self.paths

    def read_episode(self, eid: str) -> bool:
        return bool(self.episodes.get(eid))

    def row(self) -> dict:
        return {
            "v": SCHEMA,
            "paths": sorted(self.paths),
            "episodes": {e: sorted(ps) for e, ps in sorted(self.episodes.items())},
        }

    @classmethod
    def from_row(cls, row: Any) -> "PassReads":
        row = row if isinstance(row, Mapping) else {}
        paths = row.get("paths") if isinstance(row.get("paths"), list) else []
        eps = row.get("episodes") if isinstance(row.get("episodes"), Mapping) else {}
        return cls(
            frozenset(_norm(p) for p in paths if isinstance(p, str)),
            {
                str(e): frozenset(p for p in ps if isinstance(p, str))
                for e, ps in eps.items()
                if isinstance(ps, list)
            },
        )

    @classmethod
    def collect(
        cls,
        *,
        read_logs: Mapping[str, Any] = {},
        tool_events: Iterable[Mapping[str, Any]] = (),
        box_opens: Iterable[Mapping[str, Any]] = (),
    ) -> "PassReads":
        """From a pass's live evidence: its agents' read logs (``writer_tools.ReadLog``: episode -> parts), its tool
        rows (paths with mode ``read``, from calls that succeeded) and its boxes' open rows (read modes)."""
        paths: set[str] = set()
        for ev in tool_events:
            if not isinstance(ev, Mapping) or ev.get("status") not in (None, "ok"):
                continue
            for p in ev.get("paths") or []:
                if isinstance(p, Mapping) and p.get("mode") == "read" and isinstance(p.get("path"), str):
                    if p["path"].startswith("/"):
                        paths.add(_norm(p["path"]))
        for o in box_opens:
            if not isinstance(o, Mapping):
                continue
            mode = str(o.get("mode") or "r")
            if isinstance(o.get("path"), str) and o["path"].startswith("/") and "r" in mode:
                paths.add(_norm(o["path"]))
        eps: dict[str, set[str]] = {}
        for log in read_logs.values():
            parts = getattr(log, "parts", None)
            if not isinstance(parts, Mapping):
                continue
            for eid, got in parts.items():
                if isinstance(got, Mapping) and any(int(n or 0) > 0 for n in got.values()):
                    eps.setdefault(str(eid), set()).update(str(k) for k in got)
        return cls(frozenset(paths), {e: frozenset(ps) for e, ps in eps.items()})


def satisfied(part: Part, reads: PassReads, delivered: Iterable[int] = ()) -> bool:
    if any(reads.opened(p) for p in part.paths):
        return True
    if any(reads.read_episode(e) for e in part.episodes):
        return True
    got = set(delivered)
    return any(s in got for s in part.seqs)


def verify(
    entry: AttentionEntry,
    reads: PassReads,
    delivered: Iterable[int] = (),
    *,
    has_summary: Callable[[str], bool] = lambda _n: True,
    has_procedure: Callable[[str], bool] = lambda _n: True,
) -> tuple[bool, list[str], Decimal]:
    """``(read, parts read, share)`` for *entry* under *reads* and the run-record *delivered* seqs: read when every
    required part of its material was opened (:func:`material`); the share counts every part, optional ones too."""
    delivered = list(delivered)
    parts = material(entry, has_summary=has_summary, has_procedure=has_procedure)
    got = [p for p in parts if satisfied(p, reads, delivered)]
    required = [p for p in parts if p.required]
    ok = bool(required) and all(p in got for p in required)
    share = (Decimal(len(got)) / Decimal(len(parts))).quantize(Decimal("0.01")) if parts else Decimal("0")
    return ok, [p.name for p in got], share


# --- results the writer's tools return ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenEntry:
    entry: str
    kind: str
    age_s: int
    goal: str
    state: str


@dataclass(frozen=True)
class TakeResult:
    entry: str
    kind: str
    taken: bool
    detail: str
    material: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecideResult:
    entries: tuple[str, ...]
    decision: str
    items: tuple[str, ...]
    reason_ref: str
    deferred: str | None = None


@dataclass(frozen=True)
class KindReport:
    created: int = 0
    closed: int = 0
    open: int = 0
    unreadable: int = 0
    not_reached: int = 0
    read_delay_median_s: str | None = None
    read_delay_p90_s: str | None = None


@dataclass(frozen=True)
class CoverageReport:
    kinds: Mapping[str, KindReport]
    #: entries per terminal state at the time of the report, and the reopen count per item
    not_reached_why: Mapping[str, int]
    reopens_by_item: Mapping[str, int]
    #: captures no pass was ever handed (:meth:`Ledger.never_grouped`)
    never_grouped: int = 0

    def lines(self) -> list[str]:
        out = ["kind: created / closed / open / unreadable / not reached; creation-to-read delay median, p90 (s)"]
        for kind in KINDS:
            k = self.kinds.get(kind) or KindReport()
            out.append(
                f"{kind}: {k.created} / {k.closed} / {k.open} / {k.unreadable} / {k.not_reached}; "
                f"{k.read_delay_median_s if k.read_delay_median_s is not None else '-'}, "
                f"{k.read_delay_p90_s if k.read_delay_p90_s is not None else '-'}",
            )
        out.append(f"captures never handed to a pass (never grouped): {self.never_grouped}")
        return out


def _p(values: list[float], q: float) -> float | None:
    """The nearest-rank *q* quantile of *values* (None when empty)."""
    if not values:
        return None
    xs = sorted(values)
    rank = max(1, -(-int(q * 100) * len(xs) // 100))
    return xs[min(len(xs), rank) - 1]


def _seconds(value: float | None) -> str | None:
    return None if value is None else str(Decimal(str(value)).quantize(Decimal("0.1")))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


# --- the sources sync reads -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SettledFork:
    """A settled fork's staging dir: whose it is, and whether it left any output and a procedure."""

    name: str
    episode: str
    segment: int | None
    output: bool
    procedure: bool
    settled_at: str | None


@dataclass(frozen=True)
class UseFailure:
    """A failed use of *item* in *episode*: why (``error``, ``holds_false``, a contrary signal's kind, or
    ``actor_reported``), and the signals or record seq it rests on."""

    episode: str
    item: str
    why: str
    signals: tuple[str, ...] = ()
    record_seq: int | None = None


@dataclass(frozen=True)
class LateSignal:
    signal_id: str
    episode: str
    arrived_at: str | None
    about: str | None = None


@dataclass(frozen=True)
class EpisodeCalls:
    """The items an episode called, and when it ended (its uses were complete by then)."""

    episode: str
    ended_at: str | None
    items: tuple[str, ...]


@dataclass(frozen=True)
class GateRefusal:
    pass_id: str
    episodes: tuple[str, ...]
    items: tuple[str, ...]
    whole_pass: bool


class Sources(Protocol):
    """The harness records :meth:`Ledger.sync` reads; each is a record that exists in any world where its event
    happens (:class:`HomeSources` reads them from a memory home)."""

    def settled_forks(self) -> list[SettledFork]: ...

    def capture_rows(self) -> dict[str, dict]: ...

    def use_failures(self) -> list[UseFailure]: ...

    def late_signals(self) -> list[LateSignal]: ...

    def episode_calls(self) -> list[EpisodeCalls]: ...

    def episode_items(self, episode: str) -> tuple[str, ...]: ...

    def gate_refusals(self) -> list[GateRefusal]: ...

    def record_entries(self) -> list[dict]: ...

    # optional: seq -> the pass the writer was delivered it in (record_deliveries); a source without it delivers nothing


# --- the ledger ---------------------------------------------------------------------------------------------------------


class Ledger:
    """The attention ledger of one memory home (module docstring). *clock* returns the current UTC datetime (tests
    inject one)."""

    def __init__(
        self,
        home: Path | str,
        *,
        clock: Callable[[], dt.datetime] | None = None,
        directory: Path | str | None = None,
    ) -> None:
        self.home = Path(home)
        self.dir = Path(directory) if directory is not None else _ledger_dir_of(self.home)
        self.path = self.dir / LEDGER_FILE
        self.lock_path = self.dir / LOCK_FILE
        self._clock = clock or (lambda: dt.datetime.now(dt.UTC))

    # -- files ------------------------------------------------------------------------------------------------------

    @classmethod
    def at_state(cls, state_dir: Path | str, **kw: Any) -> "Ledger":
        """The ledger under a state dir (``<state>/memory-attention``) when the caller has the state dir, not the home."""
        state = Path(state_dir)
        return cls(state.parent, directory=state / LEDGER_DIR, **kw)

    @property
    def state_dir(self) -> Path:
        return self.dir.parent

    def now(self) -> str:
        return now_iso(self._clock())

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        self.dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def rows(self) -> list[dict]:
        return _read_rows(self.path)

    def _write(self, rows: list[dict]) -> None:
        for r in rows:
            r.setdefault("v", SCHEMA)
        self.dir.mkdir(parents=True, exist_ok=True)
        _append_lines(self.path, rows)

    def entries(self) -> dict[str, AttentionEntry]:
        """Every entry, by id, in creation order."""
        return fold(self.rows())

    def get(self, eid: str) -> AttentionEntry | None:
        return self.entries().get(eid)

    def in_pass(self, pass_id: str) -> list[AttentionEntry]:
        """The entries handed to (or taken into) *pass_id* that are still owed something."""
        return [e for e in self.entries().values() if e.active and e.state != "open" and e.pass_id == pass_id]

    def _has_summary(self, name: str) -> bool:
        return self._staged(name, "summary.md")

    def _has_procedure(self, name: str) -> bool:
        return self._staged(name, "procedure.py")

    def _staged_files(self) -> dict:
        """What :func:`material` asks of the staging dirs: whether one holds a summary, and a procedure."""
        return {"has_summary": self._has_summary, "has_procedure": self._has_procedure}

    def _staged(self, name: str, file: str) -> bool:
        p = self.state_dir / "memory-staging" / name / file
        try:
            return stat.S_ISREG(os.lstat(p).st_mode)
        except OSError:
            return False

    # -- creation ---------------------------------------------------------------------------------------------------

    def _create_rows(
        self,
        state: dict[str, AttentionEntry],
        kind: str,
        refs: Refs,
        source: tuple,
        *,
        at: str | None = None,
        dup_capture: bool = False,
        contrary: bool = False,
    ) -> list[dict]:
        """The rows that record *source* once (none when it is already recorded): ``create``; or, for the same
        capture as an entry of the same kind (*dup_capture*), ``append_ref`` to an active one, else for a closed one an
        ``occurrence`` row only, or, when this occurrence's outcome is *contrary*, ``append_ref`` and a ``reopen``
        (module docstring). *state* is updated in place."""
        eid = entry_id(kind, *source)
        if eid in state or any(eid in e.sources for e in state.values()):
            return []
        at = at or self.now()
        if dup_capture and refs.captures:
            same = [e for e in state.values() if e.kind == kind and set(e.refs.captures) & set(refs.captures)]
            for e in same:
                if e.active:
                    row = {"op": "append_ref", "entry": e.entry, "refs": refs.row(), "src": eid, "at": at}
                    state[e.entry] = replace(e, refs=e.refs.merged(refs), sources=e.sources + (eid,))
                    return [row]
            closed = [e for e in same if e.state == "closed"]
            if closed:
                e = closed[-1]
                if contrary:
                    rows = [
                        {"op": "append_ref", "entry": e.entry, "refs": refs.row(), "src": eid, "at": at},
                        {"op": "reopen", "entry": e.entry, "by": "contrary_occurrence", "at": at},
                    ]
                    state[e.entry] = replace(e, refs=e.refs.merged(refs), sources=e.sources + (eid,), state="open",
                                             pass_id=None, misses=0, undecided=0, reopen_count=e.reopen_count + 1,
                                             why="contrary_occurrence")  # fmt: skip
                    return rows
                row = {"op": "occurrence", "entry": e.entry, "refs": refs.row(), "src": eid, "at": at}
                state[e.entry] = replace(e, sources=e.sources + (eid,), occurrences=e.occurrences + 1)
                return [row]
        row = {"op": "create", "entry": eid, "kind": kind, "at": at, "refs": refs.row()}
        state[eid] = AttentionEntry(eid, kind, "open", at, refs)
        return [row]

    def create(self, kind: str, refs: Refs, *source: Any) -> str | None:
        """Record one entry of *kind* for *source* (no duplicate: the same source never makes a second entry);
        returns its id, or None when the source was already recorded."""
        if kind not in KINDS:
            raise LedgerError(f"kind is one of {', '.join(KINDS)}, not {kind!r}")
        with self.locked():
            state = self.entries()
            rows = self._create_rows(state, kind, refs, tuple(source))
            self._write(rows)
        return rows[0]["entry"] if rows and rows[0]["op"] == "create" else None

    def sync(self, sources: Sources | None = None) -> list[AttentionEntry]:
        """Record the entries the harness's own records now call for (module docstring); returns the new ones. Never
        reads a world name, label, id of a request, stream position or request word. Late evidence reopens the closed
        entries it bears on."""
        sources = sources if sources is not None else HomeSources(self.home, state_dir=self.state_dir)
        with self.locked():
            rows = self.rows()
            state = fold(rows)
            before = set(state)
            out: list[dict] = []
            caps = sources.capture_rows()
            # new experience: every settled fork with any output; one with a procedure waits for its capture row
            for f in sources.settled_forks():
                if not f.output:
                    continue
                cap = caps.get(f.name)
                if f.procedure and cap is None:
                    continue  # the capture step has not run on it yet; the next sync records it
                cid = cap.get("capture_id") if isinstance(cap, Mapping) else None
                refs = Refs(
                    episodes=(f.episode,),
                    segment=f.segment,
                    captures=(cid,) if isinstance(cid, str) and cid else (),
                )
                contrary = isinstance(cap, Mapping) and cap.get("outcome") in ("contrary", "failed")
                out += self._create_rows(state, "new_experience", refs, (f.name,), dup_capture=True, contrary=contrary)
            # rev 2: no capture is run to classify it, so no own check fails and no ``own_check_failed`` entry
            # is made (the kind stays readable: an older ledger's entries still fold)
            # uses that went badly (an actor's report on the record comes through the same source)
            for u in sources.use_failures():
                refs = Refs(
                    episodes=(u.episode,),
                    items=(u.item,),
                    signals=u.signals,
                    record_seq=u.record_seq,
                    captures=(u.item[len(CAPTURE_PREFIX) :],) if u.item.startswith(CAPTURE_PREFIX) else (),
                )
                out += self._create_rows(state, "use_went_badly", refs, (u.episode, u.item))
            # late evidence, which reopens the closed entries naming what its episode used or produced
            for s in sources.late_signals():
                items = sources.episode_items(s.episode)
                refs = Refs(episodes=(s.episode,), signals=(s.signal_id,), items=tuple(items))
                made = self._create_rows(state, "late_evidence", refs, (s.signal_id,), at=None)
                out += made
                if made and made[0]["op"] == "create":
                    out += self._reopen_rows(state, made[0]["entry"], items, s.arrived_at)
            # repeated use since the writer last read the item
            out += self._repeated_use_rows(state, sources.episode_calls())
            # the gate's refusals of drafts: a landing the gate refused in UNREADABLE_AFTER_MISSES passes is not handed
            # again (its entry ends unreadable, with the passes as evidence) until new evidence reopens it
            refused_in: dict[str, list[str]] = {}
            for g in sources.gate_refusals():
                refs_base = Refs(episodes=g.episodes, passes=(g.pass_id,))
                for item in g.items:
                    refused_in.setdefault(item, []).append(g.pass_id)
                rounds = max((len(refused_in[i]) for i in g.items), default=1)
                made: list[dict] = []
                if g.whole_pass:
                    made += self._create_rows(state, "deferred", replace(refs_base, items=g.items), ("gate", g.pass_id, "*"))
                for item in g.items:
                    made += self._create_rows(state, "deferred", replace(refs_base, items=(item,)), ("gate", g.pass_id, item))
                out += made
                if rounds >= UNREADABLE_AFTER_MISSES:
                    passes = sorted({p for i in g.items for p in refused_in[i]})
                    for row in made:
                        if row["op"] != "create":
                            continue
                        evidence = f"the gate refused this landing in {rounds} passes: {', '.join(passes)}"
                        out.append({"op": "unreadable", "entry": row["entry"], "pass_id": None, "evidence": evidence,
                                    "at": row["at"]})  # fmt: skip
                        state[row["entry"]] = replace(state[row["entry"]], state="unreadable", why=evidence)
            # messages to the writer on the run record; one the writer was already delivered in a pass is read there
            delivered = getattr(sources, "record_deliveries", None)
            out += self._message_rows(state, sources.record_entries(), delivered() if callable(delivered) else {})
            self._write(out)
            new = [state[e] for e in state if e not in before]
        return new

    def _message_rows(
        self,
        state: dict[str, AttentionEntry],
        entries: list[dict],
        delivered: Mapping[int, str] | None = None,
    ) -> list[dict]:
        """``message`` entries for the record entries addressed to the writer. One the writer was already delivered in
        a pass (*delivered*: seq -> that pass's id, from the record's delivery log) is created read in that pass and
        closed there (``nothing_to_keep``, its reason the delivery): the writer took it as it arrived, and a later pass
        could not be delivered it again."""
        delivered = delivered or {}
        out: list[dict] = []
        writer_seqs = {
            r.get("seq")
            for r in entries
            if isinstance(r, Mapping) and (r.get("author") == "writer" or r.get("role") == "writer")
        }
        for r in entries:
            if not isinstance(r, Mapping) or not isinstance(r.get("seq"), int):
                continue
            if r.get("author") == "writer" or r.get("role") == "writer":
                continue
            mentions = [m for m in r.get("mentions") or [] if isinstance(m, str)]
            if "writer" not in mentions and r.get("reply_to") not in writer_seqs:
                continue
            seq = int(r["seq"])
            refs_raw = [x for x in r.get("refs") or [] if isinstance(x, str)]
            items = tuple(x[len("item:") :] for x in refs_raw if x.startswith("item:"))
            caps = tuple(x[len("capture:") :] for x in refs_raw if x.startswith("capture:"))
            eps = tuple(x.split("/", 1)[0] for x in refs_raw if not x.startswith(("item:", "capture:", "entry:")))
            episode = r.get("episode") if isinstance(r.get("episode"), str) else None
            refs = Refs(
                episodes=tuple(dict.fromkeys(((episode,) if episode else ()) + eps)),
                items=items,
                captures=caps,
                record_seq=seq,
            )
            made = self._create_rows(state, "message", refs, (seq,), at=r.get("ts") if isinstance(r.get("ts"), str) else None)
            out += made
            pid = delivered.get(seq)
            if made and made[0]["op"] == "create" and isinstance(pid, str) and pid:
                eid, at = made[0]["entry"], self.now()
                out.append({"op": "read", "entry": eid, "pass_id": pid, "parts_read": ["entry"], "share": "1.00", "at": at})
                out.append({"op": "close", "entry": eid, "pass_id": pid, "decision": "nothing_to_keep", "items": [],
                            "reason_ref": f"record:delivered#{seq}", "at": at})  # fmt: skip
                e = state[eid]
                state[eid] = replace(e, state="closed", pass_id=pid, parts_read=("entry",), share="1.00",
                                     first_read_at=at, decision="nothing_to_keep", reason_ref=f"record:delivered#{seq}",
                                     closed_at=at)  # fmt: skip
        return out

    def _last_read_of(self, state: dict[str, AttentionEntry]) -> dict[str, dt.datetime]:
        """When the writer last read each item: a verified read of an entry naming it, or a decision that changed or
        landed it."""
        out: dict[str, dt.datetime] = {}
        for e in state.values():
            times = []
            if e.first_read_at:
                times.append((e.first_read_at, e.refs.items))
            if e.closed_at and e.decision in ("new_item", "changed_item", "merged_into", "note", "retire"):
                times.append((e.closed_at, e.decided_items + e.refs.items))
            for at, items in times:
                when = parse_time(at)
                if when is None:
                    continue
                for i in items:
                    if i not in out or when > out[i]:
                        out[i] = when
        return out

    def _repeated_use_rows(self, state: dict[str, AttentionEntry], calls: list[EpisodeCalls]) -> list[dict]:
        last = self._last_read_of(state)
        by_item: dict[str, list[EpisodeCalls]] = {}
        for c in calls:
            for item in c.items:
                by_item.setdefault(item, []).append(c)
        out: list[dict] = []
        for item, eps in sorted(by_item.items()):
            seen = last.get(item)
            new = [
                c.episode
                for c in eps
                if seen is None or ((parse_time(c.ended_at) or dt.datetime.max.replace(tzinfo=dt.UTC)) > seen)
            ]
            new = list(dict.fromkeys(new))
            open_r = [e for e in state.values() if e.kind == "repeated_use" and e.active and item in e.refs.items]
            if open_r:
                e = open_r[0]
                more = tuple(x for x in new if x not in e.refs.episodes)
                if more:
                    row = {
                        "op": "append_ref",
                        "entry": e.entry,
                        "refs": Refs(episodes=more).row(),
                        "src": entry_id("repeated_use", item, *more),
                        "at": self.now(),
                    }
                    state[e.entry] = replace(e, refs=e.refs.merged(Refs(episodes=more)))
                    out.append(row)
                continue
            if len(new) >= REPEATED_USE_EPISODES:
                refs = Refs(episodes=tuple(new), items=(item,))
                out += self._create_rows(state, "repeated_use", refs, (item, new[0]))
        return out

    def _reopen_rows(
        self,
        state: dict[str, AttentionEntry],
        by_entry: str,
        items: Iterable[str],
        signal_at: str | None,
    ) -> list[dict]:
        """``reopen`` rows for each closed entry naming one of *items*, when the signal (*signal_at*, else now) is newer
        than its last close."""
        items = set(items)
        if not items:
            return []
        when = parse_time(signal_at) or self._clock()
        out = []
        for e in list(state.values()):
            if e.state != "closed" or e.entry == by_entry:
                continue
            if not (items & (set(e.refs.items) | set(e.decided_items))):
                continue
            closed = parse_time(e.closed_at)
            if closed is not None and not when > closed:
                continue
            row = {"op": "reopen", "entry": e.entry, "by": f"late_evidence:{by_entry}", "at": self.now()}
            state[e.entry] = replace(e, state="open", pass_id=None, misses=0, reopen_count=e.reopen_count + 1)
            out.append(row)
        return out

    def reopen(self, eid: str, by: str) -> bool:
        """Open a closed (or unreadable, or not reached) entry again, naming why (*by*); False when it is not one."""
        with self.locked():
            e = self.entries().get(eid)
            if e is None or e.state not in ("closed", "unreadable", "not_reached"):
                return False
            self._write([{"op": "reopen", "entry": eid, "by": str(by), "at": self.now()}])
        return True

    # -- handing out --------------------------------------------------------------------------------------------------

    def _age_backstop_s(self, state: dict[str, AttentionEntry], rows: list[dict]) -> float | None:
        """The 90th-percentile age at hand of the entries handed in the last :data:`STARVATION_PASSES` passes."""
        hands = [r for r in rows if r.get("op") == "hand" and isinstance(r.get("pass_id"), str)]
        passes = list(dict.fromkeys(r["pass_id"] for r in hands))[-STARVATION_PASSES:]
        recent = set(passes)
        ages = []
        for r in hands:
            if r["pass_id"] not in recent:
                continue
            e = state.get(r.get("entry"))
            a, c = parse_time(r.get("at")), parse_time(e.created_at) if e else None
            if a is not None and c is not None:
                ages.append((a - c).total_seconds())
        return _p(ages, 0.9)

    def _order(self, entries: Iterable[AttentionEntry]) -> list[AttentionEntry]:
        rank = {k: i for i, k in enumerate(HAND_ORDER)}
        return sorted(entries, key=lambda e: (rank.get(e.kind, len(rank)), e.created_at, e.entry))

    def hand(
        self,
        pass_id: str,
        k: int = K,
        *,
        ready: Callable[[AttentionEntry], bool] | None = None,
        grouping: Callable[[dict[str, AttentionEntry]], list] | None = None,
    ) -> list[AttentionEntry]:
        """Hand up to *k* open entries to *pass_id* (module docstring: order, the age backstop, alone after two
        misses); returns them in the order handed. With *ready* (:func:`entry_ready`), an open entry it refuses waits
        (module docstring: waiting for an episode): it is not handed, and its first wait is recorded (a ``wait`` row).

        With *grouping* (:func:`.capture_groups.groups` over the fold), the grouped kinds
        (:data:`.capture_groups.GROUPED_KINDS`) wait for their group: every other open entry is handed first, as
        above; else the first due group's open members, oldest first, up to *k*, with one ``group`` row naming the
        group, why it is due and its related closed entries (the pass reads it, :func:`.attention_tools.group_of`)."""
        if not isinstance(pass_id, str) or not pass_id:
            raise LedgerError("pass_id is the pass's id, a non-empty string")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise LedgerError(f"k is a positive whole number of entries, not {k!r}")
        with self.locked():
            rows = self.rows()
            state = fold(rows)
            pending = [e for e in state.values() if e.state == "open"]
            if ready is not None:
                waiting = [e for e in pending if not ready(e)]
                if waiting:
                    noted = {r.get("entry") for r in rows if r.get("op") == "wait"}
                    at = self.now()
                    self._write([{"op": "wait", "entry": e.entry, "why": "episode_not_recorded",
                                  "episodes": list(e.refs.episodes), "at": at}
                                 for e in waiting if e.entry not in noted])  # fmt: skip
                    pending = [e for e in pending if e not in waiting]
            if not pending:
                return []
            group_row: dict | None = None
            if grouping is not None:
                from .capture_groups import GROUPED_KINDS, NO_CAPTURE

                # an episode that left no capture has nothing to write up with others: the harness closes its entry,
                # and no pass runs on it
                bare = [e for e in pending if e.kind in GROUPED_KINDS and not e.refs.captures]
                if bare:
                    at = self.now()
                    self._write([{"op": "close", "entry": e.entry, "pass_id": None, "decision": "nothing_to_keep",
                                  "items": [], "reason_ref": NO_CAPTURE, "by": "harness", "at": at}
                                 for e in bare])  # fmt: skip
                    for e in bare:
                        state[e.entry] = replace(e, state="closed", decision="nothing_to_keep",
                                                 reason_ref=NO_CAPTURE, closed_at=at)  # fmt: skip
                    pending = [e for e in pending if e not in bare]
                    if not pending:
                        return []
                now_state = {e.entry: e for e in state.values()}
                waiting_ids = {e.entry for e in state.values() if e.state == "open"} - {e.entry for e in pending}
                for wid in waiting_ids:
                    now_state.pop(wid, None)  # an entry waiting for its episode is not part of a group's pass yet
                immediate = [e for e in pending if e.kind not in GROUPED_KINDS]
                if not immediate:
                    due = [g for g in grouping(now_state) if g.due]
                    if not due:
                        return []
                    # a group whose members came back from earlier passes yields to the others (stable otherwise), so
                    # a group the writer cannot finish never holds later groups up
                    due.sort(key=lambda g: max((e.misses + e.undecided for e in g.open), default=0))
                    g = due[0]
                    # a group's pass reads its members together: a member missed before is not handed alone
                    members = [e for e in g.open if e in pending][:k]
                    if not members:
                        return []
                    group_row = {"op": "group", "pass_id": pass_id, "group": g.id, "due": g.due,
                                 "members": [e.entry for e in members],
                                 "context": [e.entry for e in g.context],
                                 "reasons": _group_reasons(g), "at": self.now()}  # fmt: skip
                    pending = members
                else:
                    pending = immediate
            alone = [e for e in pending if e.misses >= ALONE_AFTER_MISSES or e.undecided >= ALONE_AFTER_MISSES]
            if group_row is not None:
                chosen = list(pending)
            elif alone:
                chosen = self._order(alone)[:1]
            else:
                chosen = []
                backstop = self._age_backstop_s(state, rows)
                now = self._clock()
                if backstop is not None:
                    oldest = min(pending, key=lambda e: (e.created_at, e.entry))
                    created = parse_time(oldest.created_at)
                    if created is not None and (now - created).total_seconds() > backstop:
                        chosen.append(oldest)
                for e in self._order(pending):
                    if len(chosen) >= k:
                        break
                    if e not in chosen:
                        chosen.append(e)
            at = self.now()
            handed = [
                {"op": "hand", "entry": e.entry, "pass_id": pass_id, "at": at, "hand_count": e.hand_count + 1}
                for e in chosen
            ]
            self._write(handed + ([group_row] if group_row is not None else []))
            got = fold([*rows, *handed])
        return [got[e.entry] for e in chosen]

    def group_of(self, pass_id: str) -> dict | None:
        """The ``group`` row written when *pass_id* was given its entries (its group, why it was due, its members and
        its related closed entries), or None when the pass was given no group."""
        found = None
        for r in self.rows():
            if r.get("op") == "group" and r.get("pass_id") == pass_id:
                found = r
        return found

    def take(self, pass_id: str, eid: str) -> TakeResult:
        """Pull an open entry into *pass_id*: it is then owed a read and a decision like the handed ones."""
        with self.locked():
            state = self.entries()
            e = state.get(eid) if isinstance(eid, str) else None
            if e is None:
                return TakeResult(str(eid)[:40], "", False, _unknown(eid, state))
            if e.state != "open":
                where = f"pass {e.pass_id}" if e.pass_id else e.state
                if e.pass_id == pass_id and e.active:
                    return TakeResult(eid, e.kind, True, f"{eid} is already in this pass", self._material_paths(e))
                return TakeResult(eid, e.kind, False, f"{eid} is not open (it is {e.state}, {where}); take an open one")
            self._write([{"op": "hand", "entry": eid, "pass_id": pass_id, "at": self.now(),
                          "hand_count": e.hand_count + 1, "by": "take"}])  # fmt: skip
        return TakeResult(eid, e.kind, True, f"{eid} is in this pass; read its material, then decide it", self._material_paths(e))

    def _material_paths(self, e: AttentionEntry) -> tuple[str, ...]:
        out: list[str] = []
        for p in material(e, **self._staged_files()):
            if p.required:
                out.extend(p.paths[:1] or [f"episode {x}" for x in p.episodes[:1]] or [f"record entry #{s}" for s in p.seqs[:1]])
        return tuple(out)

    def open_entries(self, *, goal: Callable[[AttentionEntry], str] | None = None) -> list[OpenEntry]:
        """Every open entry: id, kind, age in seconds, a one-line goal as written (:func:`goal_line`)."""
        now = self._clock()
        out = []
        for e in self._order(e for e in self.entries().values() if e.state == "open"):
            created = parse_time(e.created_at)
            age = int((now - created).total_seconds()) if created is not None else 0
            out.append(OpenEntry(e.entry, e.kind, max(0, age), (goal or self.goal_line)(e), e.state))
        return out

    def goal_line(self, e: AttentionEntry) -> str:
        """The ``Asked`` line of the entry's first episode's summary, as written; else the kind in words."""
        from .captures import asked

        for ep in e.refs.episodes:
            p = self.state_dir / "memory-staging" / staging_name(ep, e.refs.segment) / "summary.md"
            try:
                if stat.S_ISREG(os.lstat(p).st_mode):
                    text = asked(p.read_text(encoding="utf-8", errors="replace"))
                    if text:
                        return " ".join(text.split())[:200]
            except OSError:
                continue
        return KIND_WORDS.get(e.kind, e.kind)

    # -- reading and deciding -------------------------------------------------------------------------------------------

    def verify_reads(
        self,
        pass_id: str,
        read_log: PassReads,
        record_delivery: Iterable[int] = (),
    ) -> set[str]:
        """Record a ``read`` row for each entry of *pass_id* whose required material *read_log* (the pass's own read
        evidence) and *record_delivery* (the seqs the pass's writer was delivered on the run record) show opened;
        returns every entry of the pass now verified read. The writer's words are never read."""
        delivered = list(record_delivery)
        with self.locked():
            state = self.entries()
            rows, verified = [], set()
            at = self.now()
            for e in state.values():
                if not (e.active and e.state != "open" and e.pass_id == pass_id):
                    continue
                ok, parts, share = verify(e, read_log, delivered, **self._staged_files())
                if not ok:
                    continue
                verified.add(e.entry)
                if e.state != "read" or tuple(parts) != e.parts_read:
                    rows.append({"op": "read", "entry": e.entry, "pass_id": pass_id, "parts_read": parts,
                                 "share": str(share), "at": at})  # fmt: skip
            self._write(rows)
        return verified

    def close(
        self,
        pass_id: str,
        entry_ids: Iterable[str],
        decision: str,
        items: Iterable[str],
        reason_ref: str,
    ) -> None:
        """Close *entry_ids* (each verified read in *pass_id*) with *decision*; refuses (``LedgerError``) an entry
        that is not read in this pass."""
        ids = list(entry_ids)
        if decision not in DECISIONS:
            raise LedgerError(f"decision is one of {', '.join(DECISIONS)}, not {decision!r}")
        with self.locked():
            state = self.entries()
            bad = [i for i in ids if i not in state or state[i].state != "read" or state[i].pass_id != pass_id]
            if bad:
                raise LedgerError(f"not read in pass {pass_id}: {', '.join(map(str, bad))}")
            at = self.now()
            self._write([{"op": "close", "entry": i, "pass_id": pass_id, "decision": decision,
                          "items": list(items), "reason_ref": reason_ref, "at": at} for i in ids])  # fmt: skip

    def decide(
        self,
        pass_id: str,
        entry_ids: str | Iterable[str],
        decision: str,
        items: Iterable[str] = (),
        reason: str = "",
        *,
        reads: PassReads,
        delivered: Iterable[int] = (),
        decisions_path: Path | str,
    ) -> DecideResult:
        """The writer's decision on *entry_ids* (one or several: one decision closing several entries is how the
        writer records its own grouping). Refused (``LedgerError``, with what was given, what is expected and one
        correct call) for an unknown id, an entry not in this pass, an entry its material does not show read
        (*reads*, *delivered*), an unknown decision, or ``merged_into``/``changed_item`` with no item. The reason text
        goes to the pass's ``decisions.jsonl`` only; ``defer`` creates a ``deferred`` entry."""
        ids = [entry_ids] if isinstance(entry_ids, str) else list(entry_ids or [])
        items = [i for i in (items or []) if isinstance(i, str) and i.strip()]
        example = (
            "tools.decide(['<entry id>'], 'nothing_to_keep', reason='what the material shows and why nothing is kept')"
        )
        if decision not in DECISIONS:
            raise LedgerError(
                f"decision {decision!r} is not one of {', '.join(DECISIONS)}; for example {example}",
            )
        if not ids or not all(isinstance(i, str) for i in ids):
            raise LedgerError(f"entry_ids is one entry id or a list of them, given {entry_ids!r}; for example {example}")
        if decision in NEEDS_ITEMS and not items:
            raise LedgerError(
                f"{decision} names the item it rests on in items, given none; for example "
                f"tools.decide(['<entry id>'], '{decision}', items=['<item id>'], reason='...')",
            )
        if not str(reason or "").strip():
            raise LedgerError(f"reason says why, in a sentence, given nothing; for example {example}")
        delivered = list(delivered)
        with self.locked():
            state = self.entries()
            unknown = [i for i in ids if i not in state]
            if unknown:
                raise LedgerError(_unknown(unknown[0], state) + f"; for example {example}")
            not_here = [i for i in ids if not (state[i].active and state[i].state != "open" and state[i].pass_id == pass_id)]
            if not_here:
                i = not_here[0]
                raise LedgerError(
                    f"{i} is not in this pass (it is {state[i].state}"
                    + (f", in pass {state[i].pass_id}" if state[i].pass_id and state[i].pass_id != pass_id else "")
                    + f"); take an open entry first with tools.take('{i}')"
                    + ("" if state[i].state == "open" else ", or decide only the entries handed to this pass"),
                )
            unread = []
            parts_of: dict[str, tuple[list[str], Decimal]] = {}
            for i in ids:
                ok, parts, share = verify(state[i], reads, delivered, **self._staged_files())
                parts_of[i] = (parts, share)
                if not ok:
                    need = [p for p in material(state[i], **self._staged_files()) if p.required and p.name not in parts]
                    unread.append((i, need))
            if unread:
                i, need = unread[0]
                where = "; ".join(
                    f"{p.name}: " + (
                        ", ".join(p.paths[:2]) if p.paths
                        else f"read_episode('{p.episodes[0]}')" if p.episodes
                        else f"the record entry #{p.seqs[0]} (tools.messages())" if p.seqs else "not available"
                    )
                    for p in need
                )  # fmt: skip
                raise LedgerError(
                    f"{i} is not read yet: open its material first ({where}), then decide it; nothing was decided"
                    + (
                        f" ({len(unread)} of the {len(ids)} entries are unread: {', '.join(u for u, _ in unread)}; "
                        f"the read ones, {', '.join(x for x in ids if x not in {u for u, _ in unread}) or 'none'}, may be "
                        "decided on their own)"
                        if len(ids) > 1
                        else ""
                    ),
                )
            dpath = Path(decisions_path)
            dpath.parent.mkdir(parents=True, exist_ok=True)
            n = len(_read_rows(dpath))
            at = self.now()
            _append_lines(
                dpath,
                [{"v": SCHEMA, "n": n, "pass_id": pass_id, "entries": ids, "decision": decision, "items": items,
                  "reason": str(reason)[:4000], "at": at}],
            )  # fmt: skip
            ref = f"pass:{pass_id}/{DECISIONS_FILE}#{n}"
            rows: list[dict] = []
            for i in ids:
                parts, share = parts_of[i]
                if state[i].state != "read":
                    rows.append({"op": "read", "entry": i, "pass_id": pass_id, "parts_read": parts,
                                 "share": str(share), "at": at})  # fmt: skip
                rows.append({"op": "close", "entry": i, "pass_id": pass_id, "decision": decision, "items": items,
                             "reason_ref": ref, "at": at})  # fmt: skip
            deferred = None
            if decision == "defer":
                refs = Refs(
                    episodes=tuple(dict.fromkeys(x for i in ids for x in state[i].refs.episodes)),
                    items=tuple(dict.fromkeys(x for i in ids for x in state[i].refs.items)),
                    captures=tuple(dict.fromkeys(x for i in ids for x in state[i].refs.captures)),
                    passes=(pass_id,),
                )
                made = self._create_rows(state, "deferred", refs, ("decision", pass_id, n), at=at)
                rows += made
                deferred = made[0]["entry"] if made else None
            self._write(rows)
        return DecideResult(tuple(ids), decision, tuple(items), ref, deferred)

    # -- the end of a pass, a restart, the end of a run -----------------------------------------------------------------

    def return_unread(self, pass_id: str, *, stopped: bool = False) -> list[str]:
        """At *pass_id*'s end: each entry still owed something there goes back. Not read: a miss (alone after two,
        ``unreadable`` with the pass's read evidence named after the third). Read but undecided: the same bound on its
        own count (alone after two such passes, ``unreadable`` after the third, its evidence saying it was read and never
        decided), so an entry no pass decides costs at most three passes. A pass the service was asked to stop during
        (*stopped*: a stop marker was set while it ran) had no fair turn: its entries go back with ``suspend``, no miss.
        Returns the ids returned or made unreadable."""
        with self.locked():
            state = self.entries()
            rows, out = [], []
            at = self.now()
            # a group's pass (a ``group`` row): its capture entries are given up at the bound instead of unreadable
            grouped = any(r.get("op") == "group" and r.get("pass_id") == pass_id for r in self.rows())
            for e in state.values():
                if not (e.active and e.state != "open" and e.pass_id == pass_id):
                    continue
                out.append(e.entry)
                bound = (e.undecided if e.state == "read" else e.misses) + 1 >= UNREADABLE_AFTER_MISSES
                if stopped:
                    rows.append({"op": "return", "entry": e.entry, "pass_id": pass_id, "why": "suspend", "at": at})
                elif grouped and bound and e.refs.captures:
                    # the writer could not finish this group: the harness closes the entry so it holds no queue up; its
                    # capture stays served as it was, and it is the history of any related capture that comes later
                    how = "read and not decided" if e.state == "read" else "handed and not read"
                    n = (e.undecided if e.state == "read" else e.misses) + 1
                    rows.append({"op": "close", "entry": e.entry, "pass_id": pass_id, "decision": "nothing_to_keep",
                                 "items": [], "reason_ref": GIVEN_UP, "by": "harness",
                                 "why": f"{how} in {n} group passes, the last pass:{pass_id}", "at": at})  # fmt: skip
                elif e.state == "read" and e.undecided + 1 >= UNREADABLE_AFTER_MISSES:
                    rows.append({"op": "unreadable", "entry": e.entry, "pass_id": pass_id,
                                 "evidence": f"read and not decided in {e.undecided + 1} passes, the last pass:{pass_id}",
                                 "at": at})  # fmt: skip
                elif e.state == "read":
                    rows.append({"op": "return", "entry": e.entry, "pass_id": pass_id, "why": "read_not_decided", "at": at})
                elif e.misses + 1 >= UNREADABLE_AFTER_MISSES:
                    rows.append({"op": "unreadable", "entry": e.entry, "pass_id": pass_id,
                                 "evidence": f"pass:{pass_id}/{READ_LOG_FILE}", "at": at})  # fmt: skip
                else:
                    rows.append({"op": "return", "entry": e.entry, "pass_id": pass_id, "why": "handed_not_read", "at": at})
            self._write(rows)
        return out

    def return_orphans(self, live_pass_ids: Iterable[str] = ()) -> list[str]:
        """On a service start: entries handed to a pass that is not live (no live pass id, no resumable turns) go back
        with ``why: service_restart`` (no miss: the pass never ended)."""
        live = set(live_pass_ids)
        with self.locked():
            state = self.entries()
            at = self.now()
            rows = [
                {"op": "return", "entry": e.entry, "pass_id": e.pass_id, "why": "service_restart", "at": at}
                for e in state.values()
                if e.active and e.state != "open" and e.pass_id not in live
            ]
            self._write(rows)
        return [r["entry"] for r in rows]

    def record_kin(self, entry: str, kin: Iterable[str], why: str) -> bool:
        """Record that a judge of behaviour (:mod:`.kinship`) was asked about *entry* and named the entries *kin* as
        doing the same kind of work (a ``kin`` row: a grouping relation, :mod:`.capture_groups`; with no kin, the row
        records that it was asked); *why* is the judge's reason. Unknown ids are left out. Returns whether any entry
        was named; nothing is written for an unknown *entry*."""
        with self.locked():
            state = self.entries()
            if entry not in state:
                return False
            named = [k for k in dict.fromkeys(kin) if isinstance(k, str) and k in state and k != entry]
            self._write([{"op": "kin", "entry": entry, "kin": named, "why": str(why)[:2000], "at": self.now()}])
        return bool(named)

    def mark_not_reached(self, why: str, entry_ids: Iterable[str] | None = None) -> list[str]:
        """End what the run could not reach, with *why* (:data:`NOT_REACHED`): the given entries, or every entry still
        owed something."""
        if why not in NOT_REACHED:
            raise LedgerError(f"why is one of {', '.join(NOT_REACHED)}, not {why!r}")
        want = set(entry_ids) if entry_ids is not None else None
        with self.locked():
            state = self.entries()
            at = self.now()
            rows = [
                {"op": "not_reached", "entry": e.entry, "why": why, "at": at}
                for e in state.values()
                if e.active and (want is None or e.entry in want)
            ]
            self._write(rows)
        return [r["entry"] for r in rows]

    def never_grouped(self) -> list[dict]:
        """The captures no pass was ever handed: new-experience entries with a capture that is still open or was not
        reached, never handed (no related capture arrived, or the run ended first). Their captures stayed served;
        they were never written up. Each as ``{"entry", "captures", "state"}``."""
        return [
            {"entry": e.entry, "captures": list(e.refs.captures), "state": e.state}
            for e in self.entries().values()
            if e.kind == "new_experience" and e.refs.captures and e.hand_count == 0
            and e.state in ("open", "not_reached")
        ]

    # -- the report --------------------------------------------------------------------------------------------------

    def report(self) -> CoverageReport:
        """Per kind: created, closed, still owed (open, handed or read), unreadable, not reached, and the delay from
        creation to the first verified read (median, 90th percentile, seconds); the not-reached reasons and the
        reopen count per item."""
        state = self.entries()
        kinds: dict[str, KindReport] = {}
        why: dict[str, int] = {}
        reopens: dict[str, int] = {}
        for kind in KINDS:
            es = [e for e in state.values() if e.kind == kind]
            delays = []
            for e in es:
                a, b = parse_time(e.created_at), parse_time(e.first_read_at)
                if a is not None and b is not None:
                    delays.append(max(0.0, (b - a).total_seconds()))
            kinds[kind] = KindReport(
                created=len(es),
                closed=sum(e.state == "closed" for e in es),
                open=sum(e.active for e in es),
                unreadable=sum(e.state == "unreadable" for e in es),
                not_reached=sum(e.state == "not_reached" for e in es),
                read_delay_median_s=_seconds(_median(delays)),
                read_delay_p90_s=_seconds(_p(delays, 0.9)),
            )
        for e in state.values():
            if e.state == "not_reached" and e.why:
                why[e.why] = why.get(e.why, 0) + 1
            if e.reopen_count:
                for i in dict.fromkeys(e.refs.items + e.decided_items):
                    reopens[i] = reopens.get(i, 0) + e.reopen_count
        never = sum(
            1 for e in state.values()
            if e.kind == "new_experience" and e.refs.captures and e.hand_count == 0
            and e.state in ("open", "not_reached")
        )
        return CoverageReport(kinds, why, reopens, never)


KIND_WORDS = {
    "new_experience": "new experience: an episode's summary and procedure",
    "own_check_failed": "a capture's own check failed",
    "use_went_badly": "a use of an item went badly",
    "late_evidence": "a signal arrived after its episode ended",
    "repeated_use": "an item was used in new episodes",
    "deferred": "work left open earlier",
    "message": "a message on the run record",
}


def _unknown(eid: Any, state: Mapping[str, AttentionEntry]) -> str:
    near = sorted(state, key=lambda x: _distance(str(eid), x))[:1]
    return f"no entry {str(eid)[:40]!r}" + (
        f"; the nearest id is {near[0]} ({state[near[0]].kind}, {state[near[0]].state}); tools.open_entries() lists "
        "the open ones"
        if near
        else "; the ledger has no entries"
    )


def _distance(a: str, b: str) -> int:
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    return -common


# --- what a pass is given ------------------------------------------------------------------------------------------------


def pass_episodes(
    entries: Iterable[AttentionEntry], *, resolve: Callable[[str, int | None], str | None] | None = None
) -> list[str]:
    """The recorded episodes (and segments, as ``<episode>.s<n>``) whose records and staging an attention pass exports
    for *entries*, in hand order. With *resolve* (:func:`episode_resolver`) each is the name it can be read by now, and
    one that cannot be read yet is left out (it is never asked for, so no pass breaks on it)."""
    out: list[str] = []
    for e in entries:
        for ep in e.refs.episodes:
            name = resolve(ep, e.refs.segment) if resolve is not None else staging_name(ep, e.refs.segment)
            if name is not None:
                out.append(name)
    return list(dict.fromkeys(out))


def episode_resolver(
    state_dir: Path | str, indexed: Callable[[str], bool]
) -> Callable[[str, int | None], str | None]:
    """The name an episode (or one of its segments) can be read by now, or None while it cannot (module docstring:
    waiting for an episode). *indexed* says whether an episode is recorded (the evidence index). A segment
    (``<episode>.s<n>``) reads from its own record, which its checkpoint wrote; an episode with no segment named reads
    from its record once recorded, and while it runs from its latest segment record, if it has one. Its answers are
    kept for the resolver's life (one handing or one pass); both only ever become readable."""
    from . import segments

    root = Path(state_dir) / segments.SEGMENTS_DIR
    seen: dict[tuple[str, int | None], str | None] = {}

    def has_record(name: str) -> bool:
        try:
            return stat.S_ISREG(os.lstat(segments.snapshot_path(state_dir, name)).st_mode)
        except OSError:
            return False

    def latest(episode: str) -> str | None:
        try:
            names = os.listdir(root)
        except OSError:
            return None
        best: tuple[int, str] | None = None
        for name in names:
            head, n = segments.split_staging_name(name)
            if head == episode and n is not None and (best is None or n > best[0]) and has_record(name):
                best = (n, name)
        return best[1] if best else None

    def resolve(episode: str, segment: int | None) -> str | None:
        key = (episode, segment)
        if key not in seen:
            if segment is not None:
                name = staging_name(episode, segment)
                seen[key] = name if has_record(name) or indexed(episode) else None
            else:
                seen[key] = episode if indexed(episode) else latest(episode)
        return seen[key]

    return resolve


def entry_ready(entry: AttentionEntry, resolve: Callable[[str, int | None], str | None]) -> bool:
    """Whether every episode *entry* names can be read now (:func:`episode_resolver`); an entry naming none is ready."""
    return all(resolve(ep, entry.refs.segment) is not None for ep in entry.refs.episodes)


def indexed_by(evidence: Any) -> Callable[[str], bool]:
    """Whether an episode is recorded, from the evidence index; a read that fails says not yet (the entry waits and
    is asked about again; a wait costs no miss)."""

    def indexed(episode: str) -> bool:
        try:
            return bool(evidence.episode_exists(episode))
        except Exception:  # noqa: BLE001 - unknown now: wait
            return False

    return indexed


def export_material(
    entries: Iterable[AttentionEntry],
    dest: Path | str,
    *,
    ledger: Ledger,
    sources: Sources | None = None,
    open_listing: list[OpenEntry] | None = None,
) -> dict:
    """Write the pass's attention inputs under *dest* (the box's ``/inputs/attention``): ``index.json`` (the handed
    entries: id, kind, refs, and each part of the material that must be read, as box paths) and, per entry,
    ``<id>/entry.json`` beside the material the harness holds for it outside the episode records: ``use.json`` (the
    failed use), ``signal.json`` (the late signal) and ``note.json`` (a deferral's note or the gate's refusal). Ids and
    harness facts only, apart from a deferral's own reason text. Returns the index document."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in entries:
        d = dest / e.entry
        d.mkdir(parents=True, exist_ok=True)
        parts = material(e, **ledger._staged_files())
        doc = {
            "entry": e.entry,
            "kind": e.kind,
            "refs": e.refs.row(),
            "hand_count": e.hand_count,
            "material": [p.row() for p in parts],
        }
        (d / "entry.json").write_text(json.dumps(doc, sort_keys=True, indent=1) + "\n", encoding="utf-8")
        if e.kind == "use_went_badly":
            _write_json(d / "use.json", _use_doc(e, sources))
        elif e.kind == "late_evidence":
            _write_json(d / "signal.json", _signal_doc(e, sources))
        elif e.kind == "deferred":
            _write_json(d / "note.json", _note_doc(e, ledger))
        rows.append(doc)
    index = {"v": SCHEMA, "entries": rows}
    if open_listing is not None:
        index["open"] = [
            {"entry": o.entry, "kind": o.kind, "age_s": o.age_s, "goal": o.goal} for o in open_listing
        ]
    _write_json(dest / "index.json", index)
    return index


def _write_json(path: Path, doc: Any) -> None:
    path.write_text(json.dumps(doc, sort_keys=True, indent=1, default=str) + "\n", encoding="utf-8")


def _use_doc(e: AttentionEntry, sources: Sources | None) -> dict:
    rows = []
    if sources is not None:
        try:
            rows = [
                {"episode": u.episode, "item": u.item, "why": u.why, "signals": list(u.signals), "record_seq": u.record_seq}
                for u in sources.use_failures()
                if u.episode in e.refs.episodes and u.item in e.refs.items
            ]
        except Exception:  # noqa: BLE001 - the entry still names the use by its ids
            rows = []
    return {"entry": e.entry, "episodes": list(e.refs.episodes), "items": list(e.refs.items), "uses": rows}


def _signal_doc(e: AttentionEntry, sources: Sources | None) -> dict:
    rows = []
    if sources is not None:
        try:
            got = getattr(sources, "signal_rows", None)
            rows = list(got(e.refs.signals)) if callable(got) else []
        except Exception:  # noqa: BLE001
            rows = []
    return {"entry": e.entry, "episodes": list(e.refs.episodes), "signals": rows or list(e.refs.signals),
            "items": list(e.refs.items)}  # fmt: skip


def _note_doc(e: AttentionEntry, ledger: Ledger) -> dict:
    """A deferral's note: the decision row that deferred (its reason text), or the gate refusal it records."""
    out: dict = {"entry": e.entry, "passes": list(e.refs.passes), "items": list(e.refs.items)}
    for p in e.refs.passes:
        path = ledger.state_dir / "passes" / p / DECISIONS_FILE
        rows = [r for r in _read_rows(path) if r.get("decision") == "defer"]
        if rows:
            out["deferred_by"] = [{"entries": r.get("entries"), "reason": r.get("reason"), "at": r.get("at")} for r in rows]
    if "deferred_by" not in out:
        out["gate"] = "the gate refused this draft; its full result is /inputs/drafts/<pass>/gate.md"
    return out


def write_pass_reads(path: Path | str, reads: PassReads) -> None:
    """Keep a pass's read evidence (:class:`PassReads`) beside its records, atomically."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(reads.row(), sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def pass_reads(state_dir: Path | str, pass_id: str) -> PassReads:
    """A finished pass's read evidence (``<state>/passes/<pass id>/read-log.json``), or nothing read when it has none."""
    p = Path(state_dir) / "passes" / pass_id / READ_LOG_FILE
    try:
        return PassReads.from_row(json.loads(p.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return PassReads()


_KEEPS = ("new_item", "changed_item", "note")


def item_decisions(home: Path | str, state_dir: Path | str | None = None) -> dict[str, dict]:
    """What the writer's decisions say of each library item, the latest decision naming it winning: ``{"status",
    "by", "ref", "at"}`` with *status* ``retired`` (``retire`` named it), ``superseded`` (an entry about it was decided
    ``merged_into`` another item, ``by``), or ``kept`` (``new_item``, ``changed_item`` or ``note`` named it, or it was
    the item merged into: a later decision undoes an earlier retirement). Captures are left to the capture pool. From
    the ledger's own close rows (each with the entries' refs), in the order they were written."""
    led = Ledger.at_state(state_dir) if state_dir is not None else Ledger(home)
    rows = led.rows()
    state = fold(rows)
    out: dict[str, dict] = {}

    def put(item: Any, status: str, by: str | None, row: Mapping) -> None:
        if isinstance(item, str) and item and not item.startswith(CAPTURE_PREFIX):
            out[item] = {"status": status, "by": by, "ref": row.get("reason_ref"), "at": row.get("at")}

    for row in rows:
        if row.get("op") != "close" or row.get("decision") not in DECISIONS:
            continue
        e = state.get(row.get("entry"))
        named = [i for i in row.get("items") or [] if isinstance(i, str)]
        decision = row["decision"]
        if decision == "retire":
            for i in named:
                put(i, "retired", None, row)
        elif decision == "merged_into" and named:
            target = named[0]
            for i in (e.refs.items if e is not None else ()):
                if i not in named:
                    put(i, "superseded", target, row)
            put(target, "kept", None, row)
        elif decision in _KEEPS:
            for i in named:
                put(i, "kept", None, row)
    return out


def _delivery_rows(home: Path | str) -> list:
    try:
        from unify.agents import run_record  # the mesh's reader, where it is built

        return list(run_record.read_deliveries(Path(home)))
    except Exception:  # noqa: BLE001 - not built, or unreadable: read the rows directly
        from .integration.paths import Paths

        return _read_rows(Paths.under(Path(home)).state_dir / "agents" / "run-record.cursors.jsonl")


def delivered_seqs(home: Path | str, pass_id: str, agent: str = "writer") -> set[int]:
    """The run-record seqs delivered to *agent* in *pass_id* (in a block, a ``wait`` or a ``messages`` call), from the
    record's delivery log."""
    out: set[int] = set()
    for r in _delivery_rows(home):
        if not isinstance(r, Mapping) or r.get("agent") != agent or r.get("pass_id") != pass_id:
            continue
        for s in r.get("seqs") or []:
            if isinstance(s, int) and not isinstance(s, bool):
                out.add(s)
    return out


def writer_deliveries(home: Path | str, agent: str = "writer") -> dict[int, str]:
    """Each run-record seq delivered to *agent* in a pass, with the first pass it was delivered in (the record's
    delivery log, run-wide)."""
    out: dict[int, str] = {}
    for r in _delivery_rows(home):
        if not isinstance(r, Mapping) or r.get("agent") != agent or not isinstance(r.get("pass_id"), str):
            continue
        for s in r.get("seqs") or []:
            if isinstance(s, int) and not isinstance(s, bool) and s not in out:
                out[s] = r["pass_id"]
    return out


# --- the sources of a memory home ----------------------------------------------------------------------------------------


class HomeSources:
    """:class:`Sources` read from a memory home: the fork staging dirs, the capture store's events, the evidence store
    (use facts, signals, refused passes), the episode records (end times) and the run record. Opened lazily; nothing
    is written but the end-time cache (``sync-state.json``)."""

    def __init__(self, home: Path | str, *, state_dir: Path | str | None = None):
        from .integration.paths import Paths

        self.paths = Paths.under(Path(home))
        self.state_dir = Path(state_dir) if state_dir is not None else self.paths.state_dir
        self._stores: Any = None
        self._lookup: Any = None
        self._facts: list | None = None

    # the stores, opened once
    def stores(self) -> Any:
        if self._stores is None:
            from .integration.consolidate import open_stores

            self._stores = open_stores(self.paths, busy_timeout_s=60.0)
        return self._stores

    def resolver(self) -> Callable[[str, int | None], str | None]:
        """:func:`episode_resolver` for this home: recorded episodes from the evidence index, segment records under
        the state dir."""
        return episode_resolver(self.state_dir, indexed_by(self.stores().evidence))

    def settled_forks(self) -> list[SettledFork]:
        from . import staging as _staging

        root = self.state_dir / "memory-staging"
        try:
            names = sorted(os.listdir(root))
        except OSError:
            return []
        out = []
        for name in names:
            d = root / name
            try:
                if not stat.S_ISDIR(os.lstat(d).st_mode):
                    continue
            except OSError:
                continue
            view = _staging.read(root, name)
            if view is None:
                continue
            episode, segment = split_staging_name(name)
            out.append(
                SettledFork(
                    name,
                    episode,
                    segment,
                    bool(view.finished and view.files),
                    "procedure.py" in (view.files or {}),
                    None,
                ),
            )
        return out

    def capture_rows(self) -> dict[str, dict]:
        from .captures import episode_rows, store_dir

        return episode_rows(store_dir(self.state_dir))

    def _episode_facts(self) -> list:
        if self._facts is None:
            from .lifecycle import facts_from_evidence

            self._facts = facts_from_evidence(self.stores().evidence)
        return self._facts

    def use_failures(self) -> list[UseFailure]:
        from . import trust
        from .lifecycle import trust_uses
        from .signals import _CONTRARY  # the (source, label) pairs that contradict an episode

        out: list[UseFailure] = []
        for f in self._episode_facts():
            contrary = None
            for item, u in f.items.items():
                for use in trust_uses(f, u):
                    verdict, why, _ = trust.judge(use)
                    # a refusal is never a failure of the item's trust record (it was not built for that input), but
                    # it is a use that went badly: the writer reads it
                    if use.outcome == "refused":
                        verdict, why = trust.FAILED, "refused"
                    if verdict != trust.FAILED:
                        continue
                    sigs: tuple[str, ...] = ()
                    if why in ("rejected", "contrary_end", "negative", "corrected"):
                        if contrary is None:
                            contrary = tuple(
                                s.signal_id
                                for s in self.stores().evidence.signals_for(f.episode_id)
                                if (s.source, s.label) in _CONTRARY
                            )
                        sigs = contrary
                    out.append(UseFailure(f.episode_id, item, str(why), sigs))
                    break
        # an actor's report about an item, addressed to the writer on the run record: the items its ``item:`` refs
        # name, else the items its episode used that it names by their library names
        for r in self.record_entries():
            if not _actor_to_writer(r):
                continue
            ep = r.get("episode") if isinstance(r.get("episode"), str) else None
            if not ep:
                continue
            items = [ref[len("item:") :] for ref in r.get("refs") or [] if isinstance(ref, str) and ref.startswith("item:")]
            if not items:
                text = r.get("text") if isinstance(r.get("text"), str) else ""
                items = named_items(text, self.episode_items(ep))
            for item in items:
                out.append(UseFailure(ep, item, "actor_reported", (), int(r["seq"])))
        return out

    def _ended(self) -> dict[str, str]:
        """Each recorded episode's end time (cached in ``sync-state.json``; read from the episode's meta once)."""
        cache_path = self.state_dir / LEDGER_DIR / SYNC_FILE
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            ended = cache.get("ended_at") if isinstance(cache, dict) else None
            ended = dict(ended) if isinstance(ended, dict) else {}
        except (OSError, ValueError):
            ended = {}
        ev = self.stores().evidence
        missing = [
            eid for (eid,) in ev.db.execute("SELECT episode_id FROM episodes ORDER BY seq") if eid not in ended
        ]
        if missing:
            from .integration.consolidate import EpisodeLookup

            lookup = self._lookup or EpisodeLookup(self.stores())
            self._lookup = lookup
            for eid in missing:
                try:
                    ended[eid] = str(lookup.episode(eid).ended_at)
                except Exception:  # noqa: BLE001 - not readable now; read again next time
                    continue
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_path.with_name(cache_path.name + ".tmp")
                tmp.write_text(json.dumps({"v": SCHEMA, "ended_at": ended}, sort_keys=True) + "\n", encoding="utf-8")
                os.replace(tmp, cache_path)
            except OSError:
                pass
        return ended

    def late_signals(self) -> list[LateSignal]:
        ev = self.stores().evidence
        ended = self._ended()
        out = []
        for (eid,) in ev.db.execute("SELECT DISTINCT episode_id FROM signals ORDER BY episode_id"):
            for s in ev.signals_for(eid):
                timing = getattr(s, "timing", None)
                arrived = getattr(s, "arrived_at", None) or s.ts
                if timing is None:
                    a, b = parse_time(arrived), parse_time(ended.get(eid))
                    timing = "after_episode" if (a is not None and b is not None and a > b) else "in_episode"
                if timing == "after_episode":
                    out.append(LateSignal(s.signal_id, eid, arrived, getattr(s, "about", None)))
        return out

    def signal_rows(self, ids: Iterable[str]) -> list[dict]:
        want = set(ids)
        ev = self.stores().evidence
        out = []
        for (eid,) in ev.db.execute("SELECT DISTINCT episode_id FROM signals"):
            for s in ev.signals_for(eid):
                if s.signal_id in want:
                    out.append({
                        "signal": s.signal_id, "episode": eid, "source": s.source, "label": s.label,
                        "ts": s.ts, "refers_to": s.refers_to,
                        "arrived_at": getattr(s, "arrived_at", None), "timing": getattr(s, "timing", None),
                        "about": getattr(s, "about", None),
                    })  # fmt: skip
        return out

    def episode_calls(self) -> list[EpisodeCalls]:
        ended = self._ended()
        out = []
        for f in self._episode_facts():
            called = []
            for item, u in f.items.items():
                # observed calls where the record has them (ok, refused or error), else static call sites
                if (sum(u.calls) > 0) if u.calls is not None else (u.uses > 0 or u.used or u.errored or u.refused):
                    called.append(item)
            if called:
                out.append(EpisodeCalls(f.episode_id, ended.get(f.episode_id), tuple(sorted(called))))
        return out

    def episode_items(self, episode: str) -> tuple[str, ...]:
        """What an episode used (the items its uses name) or produced (items citing it, captures of it)."""
        used = set()
        for f in self._episode_facts():
            if f.episode_id == episode:
                used.update(f.items)
        try:
            used.update(self.stores().evidence.items_citing(episode))
        except Exception:  # noqa: BLE001
            pass
        for name, row in self.capture_rows().items():
            if split_staging_name(name)[0] == episode and isinstance(row.get("capture_id"), str) and row["capture_id"]:
                used.add(CAPTURE_PREFIX + row["capture_id"])
        return tuple(sorted(used))

    def gate_refusals(self) -> list[GateRefusal]:
        """Each draft the gate refused: a pass refused whole (it wrote a patch, kept as a draft), and each item a
        landed pass refused, with the episodes the pass covered (from its end event)."""
        ev = self.stores().evidence
        covered = _pass_episodes(self.paths.events)
        out = []
        for pass_id, passed, items_refused, patch in ev.db.execute(
            "SELECT pass_id, passed, items_refused, patch_blob FROM passes ORDER BY rowid",
        ):
            try:
                refused = json.loads(items_refused) if items_refused else {}
            except ValueError:
                refused = {}
            items = tuple(sorted(str(i) for i in refused)) if isinstance(refused, (dict, list)) else ()
            whole = not passed and bool(patch)  # a pass that wrote nothing left no draft to refuse
            if not items and not whole:
                continue
            out.append(GateRefusal(str(pass_id), tuple(covered.get(str(pass_id), ())), items, whole))
        return out

    def record_deliveries(self) -> dict[int, str]:
        return writer_deliveries(self.paths.home)

    def record_entries(self) -> list[dict]:
        try:
            from unify.agents import run_record

            return [r for r in run_record.read_entries(self.paths.home) if isinstance(r, dict)]
        except Exception:  # noqa: BLE001 - not built, or unreadable: the rows directly
            return _read_rows(self.state_dir / "agents" / "run-record.jsonl")


def _actor_to_writer(r: Mapping[str, Any]) -> bool:
    """A run-record entry by an actor, addressed to the writer."""
    if not isinstance(r, Mapping) or not isinstance(r.get("seq"), int):
        return False
    role = r.get("role")
    author = r.get("author") if isinstance(r.get("author"), str) else ""
    is_actor = role == "actor" or (role is None and author.startswith("actor-"))
    mentions = [m for m in r.get("mentions") or [] if isinstance(m, str)]
    return is_actor and "writer" in mentions


def _library_names(item: str) -> tuple[str, ...]:
    """The names *item* has in the library: its id, and for a function the dotted name a cell calls it by
    (``memory.<package>.<module>:<name>`` is also ``memory.<package>.<module>.<name>``)."""
    from .layout import FUNCTION_ID

    m = FUNCTION_ID.match(item)
    return (item, f"memory.{m.group(1)}.{m.group(2)}.{m.group(3)}") if m else (item,)


def named_items(text: str, items: Iterable[str]) -> list[str]:
    """Which of *items* (the items an episode used, from the harness's records) *text* names by a library name of its
    own (:func:`_library_names`), whole: not inside a longer name and not as the prefix of an attribute. An exact
    identifier, never a word: a description of what an item does names nothing."""
    out: list[str] = []
    for item in items:
        for name in _library_names(item):
            if re.search(r"(?<![\w.:/-])" + re.escape(name) + r"(?![\w]|[.:]\w)", text or ""):
                out.append(item)
                break
    return out


def _pass_episodes(events: Path) -> dict[str, list[str]]:
    """Each pass's covered episodes, from the consolidation end events (``events.jsonl``)."""
    out: dict[str, list[str]] = {}
    for r in _read_rows(Path(events)):
        if r.get("type") == "consolidation" and r.get("phase") == "end" and isinstance(r.get("pass_id"), str):
            eps = r.get("episodes")
            if isinstance(eps, list):
                out[r["pass_id"]] = [e for e in eps if isinstance(e, str)]
    return out


def split_staging_name(name: str) -> tuple[str, int | None]:
    """``(episode, segment)`` of a staging dir name (``<episode>`` or ``<episode>.s<n>``; :mod:`.segments`)."""
    from .segments import split_staging_name as split

    return split(name)
