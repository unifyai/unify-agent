"""Serving on the situation plus the failure record (``UNIFY_MEMORY_SERVING_FAILURE_RECORD``).

With the switch on, what the actor and the librarian are shown about each stored item is evidence, never a claim
that it fits:

* each item's **failure record** (:func:`item_evidence`): one row per failed use, ``{"episode", "pin", "why",
  "ref"}``, ``why`` one of :data:`WHYS`: the call raised (``error``), its stated result condition was false
  (``holds_false``), a contrary signal its episode's regime observes followed it (``contrary_signal``; when every such
  signal arrived after the episode ended, ``late_contrary``), or the actor reported a problem with it on the run
  record (``actor_reported``); ``pin`` is the library commit that was served when it was used;
* its **contradictions**: what the writer's stated tests found not as stated, and an effect its tests contradicted,
  from the item's evidence label when the label exists (:func:`contradictions`);
* what its status **rests on** (:func:`.trust.status_basis`: verdicts, outcomes at episode end, implicit signals, or
  no outcome signal at all) and how its uses were judged, stated as text (:func:`.trust.basis_text`);
* its **stated conditions**: the first line of its ``applies`` (what it says it applies to).

The export writes these as :data:`EVIDENCE_FILE` beside the library copy; the package reads them for its cards and
its structural checks, the librarian's catalogue and search rows carry them, and each question to the librarian
carries them for the procedures its facts name (:mod:`.librarian`). Library items the writer retired or superseded
are withdrawn as captures are (:func:`withdrawn_items`, :data:`WITHDRAWN_ITEMS_FILE`): they leave every listing,
and asking for one by its id gives what replaced it.

Ids, labels, counts and the items' own stated text only: nothing here reads a world, a request's text or a stream
position.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

#: Beside the library copy: each item's evidence (failure record, contradictions, status basis, conditions).
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
#: The item-record statuses after which a library item is withdrawn.
WITHDRAWN_STATUSES = ("retired", "superseded")


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
    (``item_use_pins``, ``UNIFY_MEMORY_IN_EPISODE``: memory can change during an episode); {} when none is."""
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


def _regimes(evidence: Any) -> dict[str, str]:
    return {str(eid): str(r or "") for eid, r in evidence.db.execute("SELECT episode_id, regime FROM episodes")}


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


def contradictions(label: Mapping[str, Any] | None) -> list[str]:
    """What an item's evidence label (the writer's stated tests and effects) contradicts, as short lines: tests
    not as stated, and an effect the item declared that its replay tests contradicted. [] without a label."""
    if not isinstance(label, Mapping):
        return []
    out: list[str] = []
    tests = label.get("tests") if isinstance(label.get("tests"), Mapping) else {}
    n = tests.get("not_as_stated")
    if isinstance(n, int) and not isinstance(n, bool) and n > 0:
        out.append(f"{n} of its stated tests did not go as stated")
    effects = label.get("effects") if isinstance(label.get("effects"), Mapping) else {}
    declared = effects.get("declares")
    seen = effects.get("seen_in_tests") if isinstance(effects.get("seen_in_tests"), Mapping) else {}
    writes = seen.get("write") if isinstance(seen.get("write"), int) else 0
    reads = seen.get("read") if isinstance(seen.get("read"), int) else 0
    if declared == "none" and (writes or reads):
        out.append(f"it declares no effects, but its tests saw {reads} read(s) and {writes} write(s)")
    elif declared == "reads" and writes:
        out.append(f"it declares reads only, but its tests saw {writes} write(s)")
    return out


def _labels() -> Any:
    """The writer's evidence labels by item (``writer_tests.label_of``), or None while that module has none."""
    try:
        from . import writer_tests
    except ImportError:
        return None
    return getattr(writer_tests, "label_of", None)


def item_evidence(
    stores: Any,
    *,
    captures: Mapping[str, dict] | None = None,
    home: Path | str | None = None,
    labels: Any = None,
    recorded_only: bool = False,
) -> dict[str, dict]:
    """Each stored item's evidence, by id: library items from the evidence store's use index
    (:func:`.lifecycle.facts_from_evidence`), captured procedures (``capture:<id>``) from their records' uses
    (*captures*: :func:`.captures.load`). Each is ``{"failure_record": [...], "status_basis", "counts",
    "contradictions": [...]}``; an item with no use and no report is left out. *labels* (an item id -> its evidence
    label, or None) gives the contradictions; by default the writer's labels where they exist. With *recorded_only*
    (frozen memory) an actor's report counts only when its episode is recorded in the evidence store, so a frozen
    request's own reports never change what the next one is served."""
    from . import lifecycle, trust

    ev = stores.evidence
    pins, regimes, use_pins = _pins(ev), _regimes(ev), _use_pins(ev)
    out: dict[str, dict] = {}
    facts = lifecycle.facts_from_evidence(ev)
    for item in sorted({i for f in facts for i in f.items}):
        rows: list[dict] = []
        seen: set[tuple] = set()
        for f in facts:
            u = f.items.get(item)
            if u is None:
                continue
            for x in lifecycle.trust_uses(f, u):
                kind, why, _ = trust.judge(x)
                if kind != trust.FAILED or (f.episode_id, why) in seen:
                    continue
                seen.add((f.episode_id, why))
                served = served_pin(use_pins.get((f.episode_id, item)), why, f.memory_main)
                rows.append(_failure_row(ev, f.episode_id, served, f.regime, why))
        basis, counts = lifecycle.evidence_basis(item, facts)
        out[item] = {"failure_record": rows, "status_basis": basis, "counts": counts}
    for cid, doc in sorted((captures or {}).items()):
        rec = doc.get("record") if isinstance(doc, Mapping) else None
        if not isinstance(rec, Mapping):
            continue
        uses = [u for u in (trust.use_from_row(r) for r in rec.get("uses") or []) if u is not None]
        rows, seen = [], set()
        for x in uses:
            kind, why, _ = trust.judge(x)
            if kind != trust.FAILED or (x.episode, why) in seen:
                continue
            seen.add((x.episode, why))
            rows.append(
                _failure_row(ev, x.episode, pins.get(x.episode, ""), regimes.get(x.episode, x.regime), why),
            )
        out["capture:" + cid] = {
            "failure_record": rows,
            "status_basis": trust.status_basis(uses),
            "counts": trust.use_counts(uses),
        }
    for r in actor_reports(home):
        if recorded_only and r["episode"] not in pins:
            continue
        item = r["item"]
        pin = served_pin(use_pins.get((r["episode"], item)), "", pins.get(r["episode"], ""))
        row = {"episode": r["episode"], "pin": pin, "why": "actor_reported", "ref": f"entry:{r['seq']}"}
        entry = out.setdefault(
            item,
            {"failure_record": [], "status_basis": trust.BASIS_NO_OUTCOME_SIGNALS,
             "counts": {trust.CLEAN: 0, trust.FAILED: 0, trust.NEUTRAL: 0}},
        )
        if row not in entry["failure_record"]:
            entry["failure_record"].append(row)
    label_of = labels if labels is not None else _labels()
    for item, entry in out.items():
        label = label_of(item) if label_of is not None else None
        entry["contradictions"] = contradictions(label)
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
    """:data:`EVIDENCE_FILE`'s content: ``{"v": 1, "items": {id: {"failure_record", "contradictions", "status_basis",
    "counts", "status_text", "conditions"}}}`` for every item with evidence or stated conditions; *statuses* gives each
    item's status for its status text (on trial when it has none)."""
    from . import trust

    items: dict[str, dict] = {}
    for item in sorted(set(evidence) | set(stated)):
        e = evidence.get(item) or {}
        basis = e.get("status_basis") or trust.BASIS_NO_OUTCOME_SIGNALS
        counts = e.get("counts") or {trust.CLEAN: 0, trust.FAILED: 0, trust.NEUTRAL: 0}
        items[item] = {
            "failure_record": list(e.get("failure_record") or []),
            "contradictions": list(e.get("contradictions") or []),
            "status_basis": basis,
            "counts": dict(counts),
            "status_text": trust.basis_text(statuses.get(item) or trust.ON_TRIAL, basis, counts),
            "conditions": stated.get(item, ""),
        }
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
    "conditions",
    "contradictions",
    "evidence_doc",
    "file_bytes",
    "item_evidence",
    "withdrawn_items",
]


def iter_failures(evidence: Mapping[str, Any] | None, item: str) -> Iterable[dict]:
    """The failure-record rows of *item* in an evidence file's content (none when it has none)."""
    rows = ((evidence or {}).get("items") or {}).get(item, {}).get("failure_record") if evidence else None
    return [r for r in rows or [] if isinstance(r, dict)]
