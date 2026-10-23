"""Capture groups: which captures the writer reads together, and when a group is due.

The writer's job is transfer: it reads related captures together with the library items they touch, and makes or
reshapes items that hold beyond the episodes they came from. Transfer needs at least two instances of the same kind of
work, so a pass is handed a new capture only together with its **history**: earlier captures related to it. A capture
with no related capture is never written up alone; it stays served for reuse (the library exports every capture as
soon as it is stored) and waits until a related capture arrives. An entry whose episode left no capture is closed by
the harness (:data:`NO_CAPTURE`) and never runs a pass.

**Membership** comes from the harness's records, never from an id, a word, a request's text or a stream position, and
nothing is run. Two captures are related when their records show at least one of:

* ``sequence``: their procedures make the same ordered sequence of at least :data:`SEQUENCE_MIN` tool calls
  (``env.<channel>.<method>`` in the order written, a call repeated at once kept once:
  :func:`.capture_shape.call_sequence`), so they do the same steps through the world;
* ``subsequence``: their procedures share a run of :data:`SEQUENCE_MIN` consecutive calls;
* ``items``: their episodes used a shared library item: called it, or it refused or errored when called (the
  evidence store's use rows with an observed call); an episode that used the other capture itself counts the same. An item that was only at the episode's pin, shown or served
  relates nothing: the same few general items are served to nearly every episode;
* ``input``: they have the same entry form and the same non-empty input shape below the shared envelope: the key
  paths, named or not, and leaf types of the recorded data the capture's signature describes, less the key paths
  every capture of the home has (:data:`ENVELOPE_MIN` or more captures with a shape; fewer show no envelope);
* ``judged``: a judge that read the two captures' behaviour (their code, input shape, calls and outcome) named one as
  doing the same kind of work as the other (the ledger's ``kin`` rows).

A call feature (a whole sequence or a run of calls) that more than half of the home's acting captures make, and at
least :data:`ENVELOPE_MIN` of them, is the world's own envelope, like a request envelope every capture reads: it
distinguishes none of them, so it relates nothing. One shared call alone never relates two captures either: in a world
with one action channel every acting capture makes it.

The outcome each member's signals gave is carried to the writer as a fact of the member, not used to join.

**When a group is due** (counters only, the same in every world): a group of related captures with at least one open
(new, not yet written) capture is due when it holds at least two captures, the new ones and their history; at most one
pass per group at a time (a group with a member handed to a running pass is not due). There is no due-alone rule: no
count of later captures and no idle time hands a capture with no related capture.

Every other kind of entry (bad news, late evidence, a message, a deferral, a repeated use) is handed at once, as
before: it is about an item that exists, which is its history.

**What a group pass is given** (:func:`group_doc`, written to ``/inputs/attention/group.json``): its new members, its
history (the related closed captures, the most related first, then the most recent, at most :data:`HISTORY_MAX`), why
each pair is related, and the library items the group touches: those the members' episodes used (called, or a call
refused or errored), and those earlier passes landed from or decided on related captures. An item that was only at the
pin, shown or served is not listed.

**Why a group formed** is logged with its pass (the ledger's ``group`` row, :func:`reasons`): how many related pairs
it has and how many of them each relation kind joined, so an audit can count them.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

#: The most related closed captures (the history) a group's pass is handed, the most related first.
HISTORY_MAX = 8
#: The fewest tool calls an ordered call sequence (or a shared run of calls) must hold to relate two captures.
SEQUENCE_MIN = 2
#: The reason the harness closes an entry whose episode left no capture: nothing to write up, no pass.
NO_CAPTURE = "no_capture"
#: The reason the harness closes a capture entry of a group no pass finished (:data:`.attention.GIVEN_UP`).
GIVEN_UP = "group_given_up"
#: The entry kinds that wait for their group; every other kind is handed at once.
GROUPED_KINDS = frozenset({"new_experience"})
GROUP_FILE = "group.json"
#: The fewest captures with an input shape from which the key paths all of them share are a shared envelope.
ENVELOPE_MIN = 3
#: The item_use counts that mean an item was used (an observed call: made, refused or errored), not only served.
USED_COUNTS = ("called", "refused", "errored", "refused_modified", "errored_modified",
               "runtime_ok", "runtime_refused", "runtime_errored")
#: The relation kinds, as :func:`related` names them.
RELATION_KINDS = ("sequence", "subsequence", "items", "input", "judged")
@dataclass(frozen=True)
class Profile:
    """What the harness's records say of one entry's capture (nothing run): its entry form, the ``env`` chains its
    procedure names, its input shape, the library items its episodes used and its outcome."""

    entry: str
    captures: tuple[str, ...] = ()
    form: str | None = None
    effects: frozenset = frozenset()
    input: frozenset = frozenset()
    items: frozenset = frozenset()
    outcome: str | None = None
    sequence: tuple[str, ...] = ()
    #: the entries a judge of behaviour named as the same kind of work (the ledger's ``kin`` rows)
    kin: frozenset = frozenset()


@dataclass
class Group:
    """One connected set of related captures: its *open* members (owed a pass), its *context* (related entries an
    earlier pass decided), whether one of them is in a running pass (*pending*), why each related pair is related,
    and the reason it is due (None: not due)."""

    id: str
    open: list = field(default_factory=list)
    context: list = field(default_factory=list)
    pending: bool = False
    relations: list = field(default_factory=list)
    due: str | None = None


def call_features(p: Profile) -> frozenset:
    """The call features of a capture: its whole ordered call sequence (when it has at least :data:`SEQUENCE_MIN`
    calls) and every run of :data:`SEQUENCE_MIN` consecutive calls in it."""
    seq = p.sequence
    runs = {tuple(seq[i : i + SEQUENCE_MIN]) for i in range(len(seq) - SEQUENCE_MIN + 1)}
    return frozenset(runs | ({tuple(seq)} if len(seq) >= SEQUENCE_MIN else set()))


def common_features(profiles: Iterable[Profile]) -> frozenset:
    """The call features more than half of the acting captures make, and at least :data:`ENVELOPE_MIN` of them: the
    world's own envelope of calls, which distinguishes none of them."""
    acting = [p for p in profiles if p.sequence]
    counts: dict[tuple, int] = {}
    for p in acting:
        for f in call_features(p):
            counts[f] = counts.get(f, 0) + 1
    return frozenset(f for f, n in counts.items() if n >= ENVELOPE_MIN and 2 * n > len(acting))


def related(a: Profile, b: Profile, common: frozenset = frozenset()) -> list[str]:
    """Why two captures are related (module docstring), as ``"sequence: env.x.y > env.z.w"``, ``"subsequence: ..."``,
    ``"items: <id>"``, ``"input"`` or ``"judged"``; [] when they are not. Sharing only a tool call, only an outcome, or
    only a call feature in *common* (:func:`common_features`) relates nothing."""
    out: list[str] = []
    whole = len(a.sequence) >= SEQUENCE_MIN and a.sequence == b.sequence and tuple(a.sequence) not in common
    if whole:
        out.append("sequence: " + " > ".join(a.sequence[:8]) + (" > ..." if len(a.sequence) > 8 else ""))
    else:
        runs = sorted((call_features(a) & call_features(b)) - common)
        runs = [r for r in runs if len(r) == SEQUENCE_MIN]
        if runs:
            out.append("subsequence: " + " > ".join(runs[0]))
    # an episode that called a capture is related to it by the same relation as a shared used item
    called = ({"capture:" + c for c in a.captures} & b.items) | ({"capture:" + c for c in b.captures} & a.items)
    items = sorted((a.items & b.items) | called)
    if items:
        out.append("items: " + ", ".join(items[:5]))
    if a.form is not None and a.form == b.form and a.input and a.input == b.input:
        out.append("input")
    if b.entry in a.kin or a.entry in b.kin:
        out.append("judged")
    return out


def reasons(relations: Iterable[Mapping[str, Any]]) -> dict:
    """Why a group formed, for its ledger row: the number of related pairs and, per relation kind
    (:data:`RELATION_KINDS`), how many of them that kind joined."""
    rows = list(relations)
    out = {"pairs": len(rows)}
    for kind in RELATION_KINDS:
        out[kind] = sum(1 for r in rows if any(str(w).split(":", 1)[0] == kind for w in (r.get("why") or [])))
    return out


def _parse(at: str | None) -> dt.datetime | None:
    from .attention import parse_time

    return parse_time(at)


def groups(
    entries: Mapping[str, Any],
    profile_of: Callable[[Any], Profile],
    *,
    now: dt.datetime | None = None,
) -> list[Group]:
    """The groups of the ledger's grouped entries that have a capture (*entries*: the fold, by id), each with whether
    and why it is due (module docstring). An entry with no capture is in no group: the ledger closes it
    (:data:`NO_CAPTURE`). *now* is accepted for the callers' clock and not read: no time makes a group due."""
    live = ("open", "handed", "read", "closed")
    grouped = [e for e in entries.values() if e.kind in GROUPED_KINDS and e.state in live and e.refs.captures]
    if not any(e.state == "open" for e in grouped):
        return []
    profiles = {e.entry: profile_of(e) for e in grouped}
    common = common_features(profiles.values())
    parent = {e.entry: e.entry for e in grouped}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs: list[tuple[str, str, list[str]]] = []
    ids = [e.entry for e in grouped]
    # every relation between an open entry and any other grouped entry (closed ones join only through an open one)
    open_ids = {e.entry for e in grouped if e.state != "closed"}
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if a not in open_ids and b not in open_ids:
                continue
            why = related(profiles[a], profiles[b], common)
            if why:
                pairs.append((a, b, why))
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
    members: dict[str, list] = {}
    for e in grouped:
        members.setdefault(find(e.entry), []).append(e)
    out: list[Group] = []
    for root, es in members.items():
        opened = sorted((e for e in es if e.state == "open"), key=lambda e: (e.created_at, e.entry))
        if not opened:
            continue
        g = Group(
            id=min(es, key=lambda e: (e.created_at, e.entry)).entry,
            open=opened,
            context=sorted((e for e in es if e.state == "closed"), key=lambda e: (e.created_at, e.entry)),
            pending=any(e.state in ("handed", "read") for e in es),
            relations=[{"a": a, "b": b, "why": why} for a, b, why in pairs if find(a) == root],
        )
        # the history a pass reads: the related closed captures, the most related first, at most HISTORY_MAX
        g.context = by_strength(g)[:HISTORY_MAX]
        if g.pending:
            g.due = None
        elif len(g.open) >= 2:
            g.due = "formed"
        elif g.context:
            g.due = "history"
        out.append(g)
    return sorted(out, key=lambda g: (g.open[0].created_at, g.open[0].entry))


# --- the profiles, read from a memory home's records --------------------------------------------------------------


class HomeProfiles:
    """:class:`Profile` of each entry from a memory home's capture store and evidence store; a capture's facts are
    read once (a capture's files never change once written) and an episode's item uses each time (they grow)."""

    def __init__(self, state_dir: Path | str, evidence: Any = None) -> None:
        from .captures import store_dir

        self.store = store_dir(state_dir)
        self.evidence = evidence
        self._capture: dict[str, tuple] = {}

    def _capture_facts(self, cid: str) -> tuple:
        if cid not in self._capture:
            self._capture[cid] = _capture_facts(self.store / cid)
        return self._capture[cid]

    def _use_rows(self, episodes: Iterable[str]) -> list[tuple[str, bool]]:
        """``(item, used)`` of each use row of the episodes: *used* when the row has an observed call
        (:data:`USED_COUNTS`); a store without those columns has no used rows."""
        if self.evidence is None:
            return []
        try:
            have = {str(r[1]) for r in self.evidence.db.execute("PRAGMA table_info(item_use)").fetchall()}
        except Exception:  # noqa: BLE001 - an unreadable store relates nothing
            return []
        cols = [c for c in USED_COUNTS if c in have]
        used_sql = ("(" + " + ".join(f"COALESCE({c}, 0)" for c in cols) + ") > 0") if cols else "0"
        out: list[tuple[str, bool]] = []
        for eid in episodes:
            try:
                rows = self.evidence.db.execute(f"SELECT item, {used_sql} FROM item_use WHERE episode_id=?",
                                                (eid,)).fetchall()  # fmt: skip
            except Exception:  # noqa: BLE001 - an unreadable store relates nothing
                continue
            out.extend((str(r[0]), bool(r[1])) for r in rows if isinstance(r[0], str))
        return out

    def items_of(self, episodes: Iterable[str]) -> frozenset:
        """The library items the episodes used (an observed call: made, refused or errored)."""
        return frozenset(item for item, used in self._use_rows(episodes) if used)

    def envelope(self) -> frozenset:
        """The input key paths every capture of the home with an input shape has (:data:`ENVELOPE_MIN` or more such
        captures; fewer: none): a shared envelope, which distinguishes none of them."""
        try:
            cids = sorted(d.name for d in self.store.iterdir() if d.is_dir() and not d.is_symlink())
        except OSError:
            cids = []
        key = tuple(cids)
        if getattr(self, "_envelope_key", None) != key:
            shapes = [s for s in (self._capture_facts(c)[2] for c in cids) if s]
            self._envelope = frozenset.intersection(*shapes) if len(shapes) >= ENVELOPE_MIN else frozenset()
            self._envelope_key = key
        return self._envelope

    def __call__(self, entry: Any) -> Profile:
        cids = tuple(entry.refs.captures)
        form, effects, shape, outcome, sequence = None, frozenset(), frozenset(), None, ()
        for cid in cids[:1]:
            form, effects, shape, outcome, sequence = self._capture_facts(cid)
        if shape:
            shape = shape - self.envelope()   # only what is below the shared envelope can relate
        own = {"capture:" + c for c in cids}
        items = frozenset(i for i in self.items_of(entry.refs.episodes) if i not in own)
        kin = frozenset(k for k in getattr(entry, "kin", ()) if isinstance(k, str))
        return Profile(entry.entry, cids, form, effects, shape, items, outcome, sequence, kin)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _capture_facts(d: Path) -> tuple:
    """``(form, effects, input, outcome, sequence)`` of the capture in *d*, from its files (nothing run)."""
    from .capture_shape import call_sequence, shape_of
    from .memory_helper import signature as key_paths

    try:
        source = (d / "procedure.py").read_text(encoding="utf-8", errors="replace")
    except OSError:
        source = None
    address = shape_of(source) if source is not None else None
    form = address.get("form") if isinstance(address, dict) else None
    effects = frozenset(c for c in (address.get("env_calls") or []) if isinstance(c, str)) if address else frozenset()
    sig = _read_json(d / "signature.json")
    shape: frozenset = frozenset()
    if isinstance(sig, dict) and isinstance(sig.get("input"), dict) and sig["input"].get("kind") == "value":
        try:
            paths = key_paths(sig["input"])
        except Exception:  # noqa: BLE001 - an unreadable shape relates nothing
            paths = {}
        shape = frozenset(f"{p}:{t}" for p, t in (paths or {}).items())
    rec = _read_json(d / "record.json")
    outcome = rec.get("outcome") if isinstance(rec, dict) and isinstance(rec.get("outcome"), str) else None
    return form, effects, shape, outcome, tuple(call_sequence(source))


def by_strength(group: Group) -> list:
    """The group's related closed entries, the most related first: by how many relations (kinds of relation, over every
    open member) each shares with the group's open members, then the most recent first. The strength of a relation
    decides which earlier work the writer reads, never its age alone."""
    members = {e.entry for e in group.open}
    strength: dict[str, int] = {}
    for r in group.relations:
        a, b = r.get("a"), r.get("b")
        for mine, other in ((a, b), (b, a)):
            if other in members and mine not in members:
                strength[mine] = strength.get(mine, 0) + len(r.get("why") or [])
    return sorted(group.context, key=lambda e: (-strength.get(e.entry, 0), _neg_time(e.created_at), e.entry))


def _neg_time(at: str | None) -> float:
    t = _parse(at)
    return -t.timestamp() if t is not None else 0.0


def group_doc(group: Group, profile_of: Callable[[Any], Profile], *, landed: Mapping[str, Any] | None = None) -> dict:
    """``/inputs/attention/group.json`` for a pass handed *group*: its members and related closed entries (ids, kinds,
    episodes, captures, form, outcome, and for a closed one what was decided and on which items), why each related
    pair is related, the reason it is due, and the library items it used or is linked to (*landed*: capture id -> the
    item landed from it, when one was)."""
    landed = landed or {}

    def member(e: Any) -> dict:
        p = profile_of(e)
        row = {
            "entry": e.entry,
            "kind": e.kind,
            "episodes": list(e.refs.episodes),
            "captures": list(p.captures),
            "form": p.form,
            "outcome": p.outcome,
            "effects": sorted(p.effects),
            "sequence": list(p.sequence),
            "used_items": sorted(p.items),
        }
        if e.state == "closed":
            row.update(decision=e.decision, decided_items=list(e.decided_items))
            if getattr(e, "reason_ref", None) == GIVEN_UP:
                row["closed_by"] = "the harness: no pass finished its group"
        return row

    # used (an observed call) or linked (landed from a member's capture, decided by an earlier pass); an item that was
    # only at the pin, shown or served is not the group's
    touched: set[str] = set()
    for e in [*group.open, *group.context]:
        p = profile_of(e)
        touched.update(p.items)
        touched.update(str(landed[c]) for c in p.captures if landed.get(c))
        if e.state == "closed":
            touched.update(i for i in e.decided_items if isinstance(i, str))
    return {
        "v": 1,
        "group": group.id,
        "due": group.due,
        "members": [member(e) for e in group.open],
        "related_closed": [member(e) for e in by_strength(group)],
        "relations": list(group.relations),
        "library_items": sorted(touched),
    }
