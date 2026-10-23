"""Serving on the situation plus the failure record.

With the switch on, what the actor and the librarian are shown about each stored item is evidence, never a claim
that it fits:

* each item's **failure record** (:func:`item_evidence`): one row per failed use, ``{"episode", "pin", "why",
  "ref"}``, ``why`` one of :data:`WHYS`: the call raised (``error``), its stated result condition was false
  (``holds_false``), a contrary signal its episode's regime observes followed it (``contrary_signal``; when every such
  signal arrived after the episode ended, ``late_contrary``), or the actor reported a problem with it on the run
  record (``actor_reported``); ``pin`` is the library commit that was served when it was used;
* what its status **rests on** (:func:`.trust.status_basis`: verdicts, outcomes at episode end, implicit signals, or
  no outcome signal at all) and how its uses were judged, stated as text (:func:`.trust.basis_text`);
* its **later uses**: the calls made of it after it was stored, by how they ended (:func:`later_uses`);
* its **stated conditions**: the first line of its ``applies`` (what it says it applies to).

A captured procedure's uses are read from the same place as a library item's: the evidence store's ``item_use`` rows
under its ``capture:<id>``, which each request's use record gives from the calls its cells made. Its record in the
capture store holds what the capture step saw (its episodes, its cases and their outcomes) and the same uses, written
from those rows when a request that called it ends (:func:`record_capture_uses`); what it rests on is computed here
(:func:`capture_support`), and its status and evidence level by the one derivation its record keeps too
(:func:`.captures.use_fields`), so the two never disagree.

The export writes these as :data:`EVIDENCE_FILE` beside the library copy; the package reads them for its cards and
its structural checks, and the librarian's catalogue and search rows carry them (:mod:`.librarian` reads the same
cards in its box). Library items the writer retired or superseded
are withdrawn as captures are (:func:`withdrawn_items`, :data:`WITHDRAWN_ITEMS_FILE`): they leave every listing,
and asking for one by its id gives what replaced it.

Ids, labels, counts and the items' own stated text only: nothing here reads a world, a request's text or a stream
position. No test result (the writer's stated tests, the gate's findings) and no judgement of an item's code (pure or
not, acting or not, a fixed answer or not) is read: an item is served on its uses' outcomes and its source episodes
only.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: Beside the library copy: each item's evidence (failure record, status basis, later uses, conditions).
EVIDENCE_FILE = ".memory/item_evidence.json"
#: Beside the library copy: the library items the writer retired or superseded, each with what replaces it.
WITHDRAWN_ITEMS_FILE = ".memory/withdrawn_items.json"
#: Why a use failed, as a failure record names it.
WHYS = ("error", "holds_false", "contrary_signal", "late_contrary", "actor_reported")
#: How :func:`.trust.judge` names a failing signal, as a failure record's ``why`` (before late is told apart).
_FROM_JUDGE = {
    "error": "error",
    "holds_false": "holds_false",
    "rejected": "contrary_signal",
    "contrary_end": "contrary_signal",
    "corrected": "contrary_signal",
    "disagreed": "contrary_signal",
    "negative": "contrary_signal",
}


def _contrary_signals(evidence: Any, eid: str, regime: str) -> list[Any]:
    """The contrary signals of episode *eid* that its regime observes (as :func:`.lifecycle.counted_signals` counts
    them)."""
    from .signals import _CONTRARY, REGIME_SOURCES

    if regime == "none":
        return []
    allowed = REGIME_SOURCES.get(regime, frozenset())
    out = []
    for s in evidence.signals_for(eid):
        if s.source not in allowed or (s.source, s.label) not in _CONTRARY:
            continue
        if s.source == "checker" and not ((s.visible_to_actor is True)):
            continue
        out.append(s)
    return out


def _failure_row(evidence: Any, eid: str, pin: str, regime: str, why: str) -> dict:
    """One failure-record row of a failed use in episode *eid* (pin *pin*), failed on *why* (:func:`.trust.judge`'s
    name)."""
    kind = _FROM_JUDGE.get(why, "contrary_signal")
    ref = f"episode:{eid}"
    if kind == "contrary_signal":
        contrary = _contrary_signals(evidence, eid, regime)
        if contrary:
            last = max(contrary, key=lambda s: (s.ts, s.signal_id))
            ref = f"signal:{last.signal_id}"
            if all(getattr(s, "timing", "") == "after_episode" for s in contrary):
                kind = "late_contrary"
    return {"episode": eid, "pin": pin, "why": kind, "ref": ref}


def _pins(evidence: Any) -> dict[str, str]:
    """Each indexed episode's served library commit (``memory_main``)."""
    return {
        str(eid): str(main or "")
        for eid, main in evidence.db.execute("SELECT episode_id, memory_main FROM episodes")
    }


def _use_pins(evidence: Any) -> dict[tuple[str, str], list[tuple[str, int]]]:
    """(episode, item) -> each pin its calls were served under with its error count, in the order recorded
    (``item_use_pins``, ``UNIFY_MEMORY=in_episode``: memory can change during an episode); {} when none is."""
    out: dict[tuple[str, str], list[tuple[str, int]]] = {}
    db = evidence.db
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='item_use_pins'").fetchone():
        return out
    for item, eid, pin, error in db.execute("SELECT item, episode_id, pin, error FROM item_use_pins ORDER BY rowid"):
        out.setdefault((str(eid), str(item)), []).append((str(pin), int(error or 0)))
    return out


def served_pin(rows: list[tuple[str, int]] | None, why: str, default: str) -> str:
    """The version a failed use was served under: of the pins its episode's calls were served under (*rows*,
    :func:`_use_pins`), the last whose calls raised for an error, else the last; *default* (the episode's
    ``memory_main``) when none is recorded."""
    if not rows:
        return default
    if why == "error":
        raised = [p for p, n in rows if n > 0]
        if raised:
            return raised[-1]
    return rows[-1][0]


def actor_reports(home: Path | str | None) -> list[dict]:
    """Every run-record entry in which an actor reported on a stored item (its ``refs`` name ``item:<id>`` or
    ``capture:<id>``): ``{"item", "episode", "seq"}``, oldest first; [] without a run record. Read through the run
    record's own reader (:mod:`unify.agents.run_record`)."""
    if home is None:
        return []
    try:
        from unify.agents import run_record
    except ImportError:
        return []
    out: list[dict] = []
    for e in run_record.read_entries(Path(home)):
        if not isinstance(e, dict) or e.get("role") != "actor":
            continue
        for ref in e.get("refs") or []:
            if not isinstance(ref, str):
                continue
            if ref.startswith("item:"):
                item = ref[len("item:") :]
            elif ref.startswith("capture:"):
                item = ref
            else:
                continue
            out.append({"item": item, "episode": e.get("episode") or "", "seq": e.get("seq")})
    return out


def later_uses(item: str, facts: Any) -> dict:
    """*item*'s later uses over *facts* (:func:`.lifecycle.facts_from_evidence`): ``{"episodes", "returned",
    "refused", "raised"}``, the episodes in which it was called and its calls by how they ended, from the calls
    observed at run time (``item_use``'s runtime columns); an episode recorded without them counts one call of each
    kind its static counts show. An episode that only imported it or named it is no use."""
    out = {"episodes": 0, "returned": 0, "refused": 0, "raised": 0}
    for f in facts:
        u = f.items.get(item)
        if u is None:
            continue
        ok, refused, raised = u.calls if u.calls is not None else (int(u.used), int(u.refused), int(u.errored))
        if ok + refused + raised == 0:
            continue
        out["episodes"] += 1
        out["returned"] += ok
        out["refused"] += refused
        out["raised"] += raised
    return out


def later_uses_text(uses: Mapping[str, Any] | None) -> str:
    """A card's later-use line: ``none yet``, or how many calls in how many episodes and how they ended."""
    rows = uses if isinstance(uses, Mapping) else {}

    def n(key: str) -> int:
        v = rows.get(key)
        return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else 0

    calls = n("returned") + n("refused") + n("raised")
    if not calls:
        return "none yet"
    eps = n("episodes")
    parts = [f"{n(k)} {w}" for k, w in (("returned", "returned"), ("refused", "refused its input"),
                                          ("raised", "raised")) if n(k)]  # fmt: skip
    return (f"{calls} call{'s' if calls != 1 else ''} in {eps} episode{'s' if eps != 1 else ''}: "
            + ", ".join(parts))  # fmt: skip


def capture_support(record: Mapping[str, Any] | None, uses: list) -> dict:
    """What a recorded procedure rests on: its episodes and those whose work was accepted (its record's cases, as
    the capture step wrote them) and its confirmed uses (its later uses whose outcome confirmed it: an accepted
    verdict or a positive end, :func:`.trust.track`)."""
    from . import trust
    from .captures import record_support

    t = trust.track(uses)
    confirmed = int(t["verdicts"]["accepted"]) + int(t["end_outcomes"]["positive"])
    return {**record_support(record), "confirmed_uses": confirmed}


def _uses_in(f: Any, item: str) -> list:
    """*item*'s uses in episode facts *f* as :class:`.trust.Use` values (:func:`.lifecycle.trust_uses`); [] when it
    was not used there."""
    from . import lifecycle

    u = f.items.get(item)
    return [] if u is None else lifecycle.trust_uses(f, u)


def capture_uses(facts: Any, item: str) -> list:
    """Every use of *item* on record over *facts* (:func:`.lifecycle.facts_from_evidence`), in recording order: what
    :func:`item_evidence` judges it on and what its capture record keeps (:func:`record_capture_uses`)."""
    return [x for f in facts for x in _uses_in(f, item)]


def record_capture_uses(stores: Any, store: Path | str, items: Any) -> list[str]:
    """Write into the capture record of each captured procedure of *items* (``capture:<id>``; other ids are skipped)
    every use of it the evidence store holds (:func:`capture_uses`, :func:`.captures.record_uses`), so its record and
    the evidence store agree. Called when a request that called it ends, and when a late signal judges such a request
    again. Returns the ids whose record changed."""
    from . import lifecycle
    from .captures import record_uses

    want = sorted({i for i in items or () if isinstance(i, str) and i.startswith("capture:")})
    if not want:
        return []
    facts = lifecycle.facts_from_evidence(stores.evidence)
    return [i for i in want if record_uses(store, i[len("capture:") :], capture_uses(facts, i))]


def item_evidence(
    stores: Any,
    *,
    captures: Mapping[str, dict] | None = None,
    home: Path | str | None = None,
) -> dict[str, dict]:
    """Each stored item's evidence, by id, from one source for every kind of item: the evidence store's use index
    (:func:`.lifecycle.facts_from_evidence`: the ``item_use`` rows each request's use record gave, the calls observed
    at run time included; a captured procedure's under its ``capture:<id>``, as a library item's under its id) and the
    actors' reports on the run record. Each is ``{"failure_record", "status_basis", "counts", "later_uses"}``. Every
    captured procedure of *captures* (:func:`.captures.load`) has a row, used or not, which also holds what it rests
    on (``support``, :func:`capture_support`) and the ``status`` and ``level`` its uses give it
    (:func:`.trust.apply_outcome`; a capture whose own episode failed is never adopted). A library item with no use
    and no report is left out. Uses and reports only: no test result is read."""
    from . import lifecycle, trust
    from .captures import use_fields

    ev = stores.evidence
    pins, use_pins = _pins(ev), _use_pins(ev)
    out: dict[str, dict] = {}
    facts = lifecycle.facts_from_evidence(ev)
    own = {"capture:" + cid: doc for cid, doc in sorted((captures or {}).items())}
    for item in sorted({i for f in facts for i in f.items} | set(own)):
        rows: list[dict] = []
        seen: set[tuple] = set()
        uses: list = []
        for f in facts:
            mine = _uses_in(f, item)
            uses += mine
            for x in mine:
                kind, why, _ = trust.judge(x)
                if kind != trust.FAILED or (f.episode_id, why) in seen:
                    continue
                seen.add((f.episode_id, why))
                served = served_pin(use_pins.get((f.episode_id, item)), why or "", f.memory_main)
                rows.append(_failure_row(ev, f.episode_id, served, f.regime, why or ""))
        basis, counts = lifecycle.evidence_basis(item, facts)
        out[item] = {
            "failure_record": rows,
            "status_basis": basis,
            "counts": counts,
            "later_uses": later_uses(item, facts),
        }
        if item in own:
            doc = own[item]
            rec = doc.get("record") if isinstance(doc, Mapping) else None
            # the one derivation the capture's record also keeps (captures.use_fields): the two cannot disagree
            fields = use_fields(rec, uses)
            out[item].update(support=capture_support(rec, uses), status=fields["status"], level=fields["level"])
    for r in actor_reports(home):
        item = r["item"]
        pin = served_pin(use_pins.get((r["episode"], item)), "", pins.get(r["episode"], ""))
        row = {"episode": r["episode"], "pin": pin, "why": "actor_reported", "ref": f"entry:{r['seq']}"}
        entry = out.setdefault(
            item,
            {"failure_record": [], "status_basis": trust.BASIS_NO_OUTCOME_SIGNALS,
             "counts": {trust.CLEAN: 0, trust.FAILED: 0, trust.NEUTRAL: 0},
             "later_uses": later_uses(item, ())},
        )
        if row not in entry["failure_record"]:
            entry["failure_record"].append(row)
    return out


def _first_doc_line(fn: ast.AST) -> str:
    doc = ast.get_docstring(fn) if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) else None
    for line in (doc or "").splitlines():
        if line.strip():
            return " ".join(line.split())
    return ""


def applies_line(source: str | None, name: str = "applies") -> str:
    """The first docstring line of the function *name* in *source* (a captured procedure's ``applies``), or ''."""
    if not source:
        return ""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return ""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return _first_doc_line(node)
    return ""


def conditions(tree: Path | str, records: Mapping[str, dict] | None, captures: Mapping[str, dict] | None) -> dict:
    """Each procedure's stated conditions: the first docstring line of the function its item record names as its
    ``applies`` (read from the library copy at *tree*), or of a captured procedure's own ``applies``; items that
    state none are left out."""
    from . import layout

    out: dict[str, str] = {}
    root = Path(tree)
    for item, rec in sorted((records or {}).items()):
        named = rec.get("applies") if isinstance(rec, Mapping) else None
        if not isinstance(named, str) or ":" not in named:
            continue
        try:
            source = (root / layout.item_path(named)).read_text(encoding="utf-8")
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        line = applies_line(source, named.rsplit(":", 1)[1])
        if line:
            out[item] = line
    for cid, doc in sorted((captures or {}).items()):
        line = applies_line(doc.get("procedure") if isinstance(doc, Mapping) else None)
        if line:
            out["capture:" + cid] = line
    return out


def evidence_doc(evidence: Mapping[str, dict], stated: Mapping[str, str], statuses: Mapping[str, str]) -> dict:
    """:data:`EVIDENCE_FILE`'s content: ``{"v": 1, "items": {id: {"failure_record", "status_basis", "counts",
    "status_text", "later_uses", "later_uses_text", "conditions"}}}`` for every item with evidence or stated
    conditions; *statuses* gives each library item's status for its status text (on trial when it has none). A
    captured procedure's row also carries its ``support``, ``status`` and ``level`` (:func:`item_evidence`)."""
    from . import trust

    items: dict[str, dict] = {}
    for item in sorted(set(evidence) | set(stated)):
        e = evidence.get(item) or {}
        basis = e.get("status_basis") or trust.BASIS_NO_OUTCOME_SIGNALS
        counts = e.get("counts") or {trust.CLEAN: 0, trust.FAILED: 0, trust.NEUTRAL: 0}
        uses = dict(e.get("later_uses") or {})
        items[item] = {
            "failure_record": list(e.get("failure_record") or []),
            "status_basis": basis,
            "counts": dict(counts),
            # a recorded procedure is named by what it is, never by a trial status that reads as second-class
            "status_text": trust.basis_text(
                "recorded procedure" if item.startswith("capture:") else statuses.get(item) or trust.ON_TRIAL,
                basis,
                counts,
            ),
            "later_uses": uses,
            "later_uses_text": later_uses_text(uses),
            "conditions": stated.get(item, ""),
        }
        for key in ("support", "status", "level"):
            if key in e:
                items[item][key] = e[key]
    return {"v": 1, "items": items}


def withdrawn_items(records: Mapping[str, dict] | None) -> dict[str, dict]:
    """Library item id -> ``{"status", "by", "reason"}`` for each item its record shows the writer retired or
    superseded: ``by`` is what replaces it, its record's ``superseded_by`` or, for an alias, the item it forwards to
    (then ``superseded``); None when it was retired with nothing in its place. The records' decisions are the only
    input."""
    from .item_states import status_from_stored

    out: dict[str, dict] = {}
    for item, rec in sorted((records or {}).items()):
        if not isinstance(rec, Mapping):
            continue
        by = rec.get("superseded_by") or rec.get("alias_of")
        by = by if isinstance(by, str) and by and by != item else None
        status = rec.get("status")
        if by is None and status_from_stored(status) != "retired":
            continue
        reason = rec.get("status_reason") if isinstance(rec.get("status_reason"), str) else ""
        out[item] = {
            "status": "superseded" if by is not None else "retired",
            "by": by,
            "reason": (reason or ("superseded by " + by if by else "retired"))[:200],
        }
    # a replacement that is itself withdrawn is followed to what replaces it (a loop or a dead end gives None)
    for item, row in out.items():
        seen, by = {item}, row["by"]
        while by is not None and by in out and by not in seen:
            seen.add(by)
            by = out[by]["by"]
        row["by"] = None if by in seen else by
    return out


def file_bytes(doc: Mapping[str, Any]) -> bytes:
    """A generated JSON file's bytes (sorted keys, one trailing newline)."""
    return (json.dumps(doc, sort_keys=True, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


__all__ = [
    "EVIDENCE_FILE",
    "WHYS",
    "WITHDRAWN_ITEMS_FILE",
    "actor_reports",
    "applies_line",
    "capture_support",
    "capture_uses",
    "conditions",
    "evidence_doc",
    "file_bytes",
    "item_evidence",
    "later_uses",
    "later_uses_text",
    "record_capture_uses",
    "withdrawn_items",
]
