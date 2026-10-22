"""Capture groups: which new captures the writer reads together, and when a group is due (rev 2.1).

The writer's job is transfer: it reads several related captures together with the library items they touch, and
makes or reshapes items that hold beyond the episodes they came from. A pass over one capture cannot see what varies
and what stays, so each ``new_experience`` entry waits until its group is due, and a pass is handed one group.

**Membership** comes from the harness's own records, never from an id, a word, a request's text or a stream
position, and nothing is run. Two captures are related when their records show at least one of:

* ``sequence``: their procedures make the same ordered sequence of at least :data:`SEQUENCE_MIN` tool calls
  (``env.<channel>.<method>`` in the order written, a call repeated at once kept once:
  :func:`.capture_shape.call_sequence`), so they do the same steps through the world. One shared call is not enough:
  in a world with one action channel every acting capture makes it, so a single call would chain all work into one
  group;
* ``items``: their episodes used, were served or failed with a shared library item (the evidence store's use rows:
  imported, called, refused or errored);
* ``input``: they have the same entry form and the same non-empty input shape: the key paths, named or not, and leaf
  types of the recorded data the capture's signature describes. For a ``value`` capture that is the value it reads;
  for a ``state`` or ``state_env`` capture, the structured data its state holds (the JSON its messages carry). A state
  with no structured data (text only) has no input shape, by the same rule ``memory.find`` uses: text has no
  structure to compare.

A group is a connected set of related captures. The outcome each member's signals gave is carried to the writer as
a fact of the member, not used to join: an edge on the outcome alone would join unrelated work in any world with one
kind of verdict. The writer decides membership by reading: it may decide members differently (which splits the
group), and it may take any closed entry into its pass read-only (:func:`.attention_tools.take`).

**When a group is due** (counters only, the same in every world; the capture-group design's readiness rule):

* *formed*: two or more of its members are open;
* *joined*: an open member is related to a capture an earlier pass already decided (a closed entry), so that
  earlier work is read again together with the new;
* *waited*: a member still alone (no open or closed relation) is due after :data:`W` later new captures, or once no
  new capture has arrived for :data:`IDLE_S` seconds, so nothing waits for ever;
* at most one pass per group at a time: a group with a member handed to a running pass is not due; what joins it
  meanwhile waits for that pass to end.

Every other kind of entry (bad news, late evidence, a message, a deferral, a repeated use) is handed at once, as
before: it is about one thing that happened.

**What a group pass is given** (:func:`group_doc`, written to ``/inputs/attention/group.json``): its members (the
handed entries and their captures), the related closed entries (their captures and what earlier passes decided), why
each pair is related, and the library items the group touches: those the members' episodes used, were served or
failed with, and those earlier passes landed from related captures.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

#: Later new captures after which a capture still alone is due.
W = 8
#: The fewest tool calls an ordered call sequence must hold to relate two captures.
SEQUENCE_MIN = 2
#: Seconds with no new capture after which every waiting capture is due.
IDLE_S = 600
#: The entry kinds that wait for their group; every other kind is handed at once.
GROUPED_KINDS = frozenset({"new_experience"})
GROUP_FILE = "group.json"
@dataclass(frozen=True)
class Profile:
    """What the harness's records say of one entry's capture (nothing run): its entry form, the ``env`` chains its
    procedure names, its input shape, the library items its episodes touched and its outcome."""

    entry: str
    captures: tuple[str, ...] = ()
    form: str | None = None
    effects: frozenset = frozenset()
    input: frozenset = frozenset()
    items: frozenset = frozenset()
    outcome: str | None = None
    sequence: tuple[str, ...] = ()


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


def related(a: Profile, b: Profile) -> list[str]:
    """Why two captures are related (module docstring), as ``"sequence: env.x.y > env.z.w"``, ``"items: <id>"`` or
    ``"input"``; [] when they are not. Sharing only a tool call, or only an outcome, relates nothing."""
    out: list[str] = []
    if len(a.sequence) >= SEQUENCE_MIN and a.sequence == b.sequence:
        out.append("sequence: " + " > ".join(a.sequence[:8]) + (" > ..." if len(a.sequence) > 8 else ""))
    items = sorted(a.items & b.items)
    if items:
        out.append("items: " + ", ".join(items[:5]))
    if a.form is not None and a.form == b.form and a.input and a.input == b.input:
        out.append("input")
    return out


def _parse(at: str | None) -> dt.datetime | None:
    from .attention import parse_time

    return parse_time(at)


def groups(
    entries: Mapping[str, Any],
    profile_of: Callable[[Any], Profile],
    *,
    now: dt.datetime,
) -> list[Group]:
    """The groups of the ledger's grouped entries (*entries*: the fold, by id), each with whether and why it is due
    (module docstring). An entry with no readable capture is a profile with nothing to relate: it waits alone."""
    live = ("open", "handed", "read", "closed")
    grouped = [e for e in entries.values() if e.kind in GROUPED_KINDS and e.state in live]
    if not any(e.state == "open" for e in grouped):
        return []
    profiles = {e.entry: profile_of(e) for e in grouped}
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
            why = related(profiles[a], profiles[b])
            if why:
                pairs.append((a, b, why))
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
    members: dict[str, list] = {}
    for e in grouped:
        members.setdefault(find(e.entry), []).append(e)
    created = sorted(
        (e for e in entries.values() if e.kind in GROUPED_KINDS),
        key=lambda e: (e.created_at, e.entry),
    )
    last_new = _parse(created[-1].created_at) if created else None
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
        if g.pending:
            g.due = None
        elif len(g.open) >= 2:
            g.due = "formed"
        elif g.context:
            g.due = "joined"
        else:
            first = g.open[0]
            later = sum(1 for e in created if (e.created_at, e.entry) > (first.created_at, first.entry))
            idle = (now - last_new).total_seconds() if last_new is not None else 0.0
            if later >= W:
                g.due = "waited"
            elif idle >= IDLE_S:
                g.due = "idle"
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

    def items_of(self, episodes: Iterable[str]) -> frozenset:
        """The library items the episodes used, were served or failed with (any use row)."""
        if self.evidence is None:
            return frozenset()
        out: set[str] = set()
        for eid in episodes:
            try:
                rows = self.evidence.db.execute("SELECT item FROM item_use WHERE episode_id=?", (eid,)).fetchall()
            except Exception:  # noqa: BLE001 - an unreadable store relates nothing
                continue
            out.update(str(r[0]) for r in rows if isinstance(r[0], str))
        return frozenset(out)

    def __call__(self, entry: Any) -> Profile:
        cids = tuple(entry.refs.captures)
        form, effects, shape, outcome, sequence = None, frozenset(), frozenset(), None, ()
        for cid in cids[:1]:
            form, effects, shape, outcome, sequence = self._capture_facts(cid)
        own = {"capture:" + c for c in cids}
        items = frozenset(i for i in self.items_of(entry.refs.episodes) if i not in own)
        return Profile(entry.entry, cids, form, effects, shape, items, outcome, sequence)


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
    pair is related, the reason it is due, and the library items it touches (*landed*: capture id -> the item landed
    from it, when one was)."""
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
            "items_touched": sorted(p.items),
        }
        if e.state == "closed":
            row.update(decision=e.decision, decided_items=list(e.decided_items))
        return row

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
