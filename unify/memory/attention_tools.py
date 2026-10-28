"""The writer's tools for the attention ledger and its stated tests, as host handlers of in-code calls.

``tools.open_entries()``, ``tools.take(entry_id)``, ``tools.decide(entry_ids, decision, items=[], reason=...)``,
``tools.find(query, scope="items", k=10)`` and ``tools.test(item, on, expect)``. Each runs on the host, in the pass's
worker pool, against the pass's own state (:class:`AttentionPass`, set by the writer pass): the ledger, the entries
handed to the pass, its read evidence as it stands now and its record directory. Every reply is JSON text; a refusal
says what was given, what is expected and one correct call.

A read counts once the harness has seen it: a ``tools.read`` call when it returns, a cell's opens when the cell ends.
So a decision made in the same cell as the read of its material is refused, saying so; decide in the next cell.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import attention as _att
from . import writer_tools as _wt

#: Hits ``find`` returns at most when *k* is not given.
FIND_K = 10
FIND_SCOPES = ("items", "summaries", "open_entries")


@dataclass
class AttentionPass:
    """One pass's attention state: its ledger and id, the entries handed to it, where its record lives, the episodes
    and the inputs directory the box sees at ``/inputs`` (a taken entry's material is exported there)."""

    ledger: _att.Ledger
    pass_id: str
    record_dir: Path
    inputs: Path
    home: Path
    handed: list = field(default_factory=list)
    sources: Any = None
    tests: Any = None  # writer_tests.TestContext
    exported: set = field(default_factory=set)
    #: the capture group the pass was handed (``/inputs/attention/group.json``), None for a pass with no group
    group: dict | None = None

    @property
    def decisions_path(self) -> Path:
        return self.record_dir / _att.DECISIONS_FILE

    @property
    def read_log_path(self) -> Path:
        return self.record_dir / _att.READ_LOG_FILE


def reads_of(writer: Any) -> _att.PassReads:
    """What the pass's agents have read so far, from the harness's own evidence: the read logs, the tool rows and the
    boxes' open rows (the pass's ``box-opens.jsonl`` when it keeps one: it has the helpers' too)."""
    opens: list = list(getattr(writer, "box_opens", None) or [])
    att = getattr(writer, "_attention", None)
    if att is not None:
        path = att.record_dir / "box-opens.jsonl"
        try:
            rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
            opens = [r for r in rows if isinstance(r, dict)] or opens
        except (OSError, ValueError):
            pass
    return _att.PassReads.collect(
        read_logs=dict(getattr(writer, "_read_logs", None) or {}),
        tool_events=list(getattr(writer, "tool_events", None) or []),
        box_opens=opens,
    )


def _state(ctx: Any) -> tuple[Any, AttentionPass | None]:
    st = ctx.state
    writer = getattr(st, "writer", None)
    return writer, getattr(writer, "_attention", None)


def _json(doc: Any) -> str:
    return json.dumps(doc, ensure_ascii=False, sort_keys=True, default=str)


def _refused(tool: str, reason: str) -> "_wt.Reply":
    return _wt.refused(tool, "refused", reason=str(reason)[:1500])


def _no_ledger(tool: str) -> "_wt.Reply":
    return _refused(tool, "this pass has no attention ledger: it is not an attention pass")


# --- open entries, take, decide ---------------------------------------------------------------------------------------


def open_entries(ctx: Any, args: dict) -> "_wt.Reply":
    """``tools.open_entries()``: every open entry of the ledger (not yet handed to a pass): its id, kind, age in
    seconds and a one-line goal, as its episode's summary states it."""
    _writer, att = _state(ctx)
    if att is None:
        return _no_ledger("open_entries")
    rows = [asdict(o) for o in att.ledger.open_entries()]
    return _wt.Reply(_json({"open": rows, "count": len(rows)}), paths=({"path": "/inputs/attention", "mode": "list"},))


def take(ctx: Any, args: dict) -> "_wt.Reply":
    """``tools.take(entry_id)``: pull an open entry into this pass; its material is exported under /inputs and it is
    owed a read and a decision like the handed ones. A closed entry (an earlier pass decided it) is taken read-only:
    its material is exported and what was decided is returned, so earlier captures of the same kind can be read with
    the new ones; nothing about it is owed or changed."""
    writer, att = _state(ctx)
    if att is None:
        return _no_ledger("take")
    eid = args.get("entry_id")
    if not isinstance(eid, str) or not eid.strip():
        return _refused("take", "entry_id is one entry id, a string; for example tools.take('a1b2c3d4e5f60718')")
    closed = att.ledger.get(eid.strip())
    if closed is not None and closed.state == "closed":
        paths = export_context(att, [closed])
        return _wt.Reply(_json({"entry": closed.entry, "kind": closed.kind, "taken": "read-only",
                                "decision": closed.decision, "decided_items": list(closed.decided_items),
                                "material": paths}))  # fmt: skip
    got = att.ledger.take(att.pass_id, eid.strip())
    if not got.taken:
        return _refused("take", got.detail + "; tools.open_entries() lists the open entries")
    entry = att.ledger.get(got.entry)
    if entry is not None:
        export_entries(writer, att, [entry])
    _attribute(got.entry)
    return _wt.Reply(_json({"entry": got.entry, "kind": got.kind, "taken": True, "detail": got.detail,
                            "material": list(got.material)}))  # fmt: skip


def _attribute(entry_id: str) -> None:
    """The pass's later model calls name a taken entry too (metering: the request metadata headers,
    ids only); nothing where the attribution is not built. Never raises."""
    try:
        from unify.common.llm_client import add_attributed_entry
    except ImportError:
        return
    try:
        add_attributed_entry(entry_id)
    except Exception:  # noqa: BLE001 - metering never stops a take
        pass


def decide(ctx: Any, args: dict) -> "_wt.Reply":
    """``tools.decide(entry_ids, decision, items=[], reason=...)``: close entries of this pass with one decision,
    each verified read from the pass's own read evidence."""
    writer, att = _state(ctx)
    if att is None:
        return _no_ledger("decide")
    absent = _not_held(writer, att, str(args.get("decision") or ""), args.get("items") or [])
    if absent:
        return _refused(
            "decide",
            f"retire names {', '.join(absent)}, which the library does not hold (no function, note or capture has "
            "that id), so the decision could not take effect; nothing was decided. tools.find(query) lists the "
            "library's items with their ids, and an entry's refs name the items it is about",
        )
    reads = reads_of(writer)
    delivered = _att.delivered_seqs(att.home, att.pass_id)
    try:
        res = att.ledger.decide(
            att.pass_id,
            args.get("entry_ids"),
            str(args.get("decision") or ""),
            args.get("items") or [],
            str(args.get("reason") or ""),
            reads=reads,
            delivered=delivered,
            decisions_path=att.decisions_path,
        )
    except _att.LedgerError as exc:
        text = str(exc)
        if "not read yet" in text:
            text += " (a cell's reads count when that cell ends: decide in a later cell, or read with tools.read)"
        return _refused("decide", text)
    return _wt.Reply(_json({"closed": list(res.entries), "decision": res.decision, "items": list(res.items),
                            "reason_ref": res.reason_ref, "deferred": res.deferred}))  # fmt: skip


def _not_held(writer: Any, att: AttentionPass, decision: str, items: Any) -> list[str]:
    """The items a ``retire`` names that it could not take effect on: a ``capture:<id>`` the capture store does not
    hold, or another id that is no function or note of the pass's library copy. [] for any other decision, and when
    the library copy cannot be read (nothing is refused then that could have been held)."""
    if decision != "retire" or not isinstance(items, (list, tuple)):
        return []
    names = [i.strip() for i in items if isinstance(i, str) and i.strip()]
    if not names:
        return []
    from .captures import load, store_dir
    from .layout import discover

    held: set[str] = set()
    if any(n.startswith(_att.CAPTURE_PREFIX) for n in names):
        try:
            held.update(_att.CAPTURE_PREFIX + cid for cid in load(store_dir(att.ledger.state_dir)))
        except OSError:
            return []
    if any(not n.startswith(_att.CAPTURE_PREFIX) for n in names):
        box = _box_of(writer)
        if box is None:
            return []
        try:
            lib = discover(box)
        except Exception:  # noqa: BLE001 - an unreadable copy refuses nothing
            return []
        held.update(f.item_id for f in lib.functions)
        held.update(n.item_id for n in lib.notes)
    return [n for n in names if n not in held]


# --- find ----------------------------------------------------------------------------------------------------------------


def _summary_rows(att: AttentionPass) -> list[dict]:
    from .captures import asked
    from .library_export import _section

    root = att.ledger.state_dir / "memory-staging"
    out = []
    try:
        names = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.is_symlink())
    except OSError:
        return out
    for name in names:
        p = root / name / "summary.md"
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        goal = asked(text)
        body = " ".join(_section(text, s) for s in ("Asked", "Approach", "Worked", "Failed"))
        readable = (att.inputs / "staging" / name / "summary.md").is_file()
        out.append({
            "id": name, "kind": "summary",
            "path": f"/inputs/staging/{name}/summary.md" if readable else None,
            "card": goal, "text": f"{goal} {body}".strip(),
        })  # fmt: skip
    return out


def _entry_rows(att: AttentionPass) -> list[dict]:
    out = []
    for o in att.ledger.open_entries():
        out.append({"id": o.entry, "kind": o.kind, "path": None, "card": o.goal, "text": f"{o.goal} {o.kind}"})
    return out


def _item_rows(writer: Any) -> list[dict]:
    from .library_search import rows

    box = getattr(getattr(writer, "_attention", None), "box", None) or _box_of(writer)
    if box is None:
        return []
    try:
        return rows(box)
    except Exception:  # noqa: BLE001 - an unreadable copy has no items to rank
        return []


def _box_of(writer: Any) -> Path | None:
    live = getattr(writer, "_live", None)
    return Path(live[0]) if isinstance(live, tuple) and live and live[0] else None


def find(ctx: Any, args: dict) -> "_wt.Reply":
    """``tools.find(query, scope="items", k=10)``: the library's items by their "When to use" text, the episodes'
    summaries, or the open entries, against *query*; the first *k*. Ranked by meaning with the run's embedding backend
    (:func:`.library_search.backend`, the librarian's), else not ranked at all: listed in their own order, the result
    saying why (no word matching ranks anything). The writer drives it: nothing here proposes a group."""
    from .library_search import NO_BACKEND, backend, index_data, ranking

    writer, att = _state(ctx)
    if att is None:
        return _no_ledger("find")
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return _refused("find", "query is the situation in words, a non-empty string; for example "
                                "tools.find('total the amounts of a table', scope='summaries')")  # fmt: skip
    scope = args.get("scope", "items") or "items"
    if scope not in FIND_SCOPES:
        return _refused("find", f"scope is one of {', '.join(FIND_SCOPES)}, given {scope!r}")
    k = args.get("k", FIND_K)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        return _refused("find", f"k is a positive whole number of hits, given {k!r}")
    found = _item_rows(writer) if scope == "items" else _summary_rows(att) if scope == "summaries" else _entry_rows(att)
    # the same ranking as the librarian's search: by meaning when the run has an embedding route, else none
    try:
        use = backend()
    except Exception:  # noqa: BLE001 - no embedder: nothing is ranked, and the result says why
        use = NO_BACKEND
    ranked, how = ranking(index_data(found, use), query, use)
    hits = [
        {"id": r.get("id"), "kind": r.get("kind"), "path": r.get("path"),
         "score": round(float(s), 4) if s is not None else None, "card": r.get("card")}
        for s, r in ranked[:k]
    ]  # fmt: skip
    return _wt.Reply(_json({"scope": scope, "ranked": how, "hits": hits, "of": len(ranked)}))


# --- the stated test ------------------------------------------------------------------------------------------------------


def test(ctx: Any, args: dict) -> "_wt.Reply":
    """``tools.test(item, on, expect)``: a test stated before it runs (:mod:`.writer_tests`)."""
    from . import writer_tests as wt

    writer, att = _state(ctx)
    tests = writer._tests
    lock = getattr(ctx.state, "memory_lock", None)
    if lock is not None:
        from .writer_pass import _mirror

        def snapshot(src: Path, dst: Path) -> None:
            # the copy of /memory a test runs on is taken under the pass's /memory lock, as check's is
            with lock:
                _mirror(Path(src), Path(dst))

        tests.snapshot = snapshot
    try:
        res = wt.run_test(tests, args.get("item"), args.get("on"), args.get("expect"))
    except wt.WriterTestRefused as exc:
        return _refused("test", f"{exc}; nothing was recorded or run")
    return _wt.Reply(_json(res.row()))


# --- what an entry's material needs in /inputs -------------------------------------------------------------------------


def export_entries(writer: Any, att: AttentionPass, entries: list) -> None:
    """Export what *entries* owe a read into the pass's ``/inputs``: their staging (summary, procedure, drafts), their
    captures, their attention files (``use.json``, ``signal.json``, ``note.json``) and the drafts a deferral points
    at. Host-side writes into the inputs directory; nothing in the box changes but what it can read."""
    from . import captures as _captures
    from . import staging as _staging

    state = att.ledger.state_dir
    inputs = att.inputs
    fresh = [e for e in entries if e.entry not in att.exported]
    if not fresh:
        return
    # the entries exported earlier in the pass stay in the index (export_material writes the fresh ones' index)
    index = inputs / "attention" / "index.json"
    try:
        doc = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        doc = {"v": _att.SCHEMA, "entries": []}
    if not isinstance(doc, dict) or not isinstance(doc.get("entries"), list):
        doc = {"v": _att.SCHEMA, "entries": []}
    _att.export_material(fresh, inputs / "attention", ledger=att.ledger, sources=att.sources)
    root = state / "memory-staging"
    cids: list[str] = []
    for e in fresh:
        att.exported.add(e.entry)
        for name in _att.pass_episodes([e]):
            dest = inputs / "staging" / name
            if dest.exists():
                continue
            view = _staging.read(root, name)
            if view is not None:
                _staging.export(view, dest)
        cids += list(e.refs.captures)
        cids += [i[len(_att.CAPTURE_PREFIX) :] for i in e.refs.items if i.startswith(_att.CAPTURE_PREFIX)]
    if cids:
        try:
            _captures.export_for_writer(_captures.store_dir(state), cids, inputs / "captures")
        except OSError:
            pass
    have = {r.get("entry") for r in doc["entries"] if isinstance(r, dict)}
    for e in fresh:
        if e.entry not in have:
            doc["entries"].append(json.loads((inputs / "attention" / e.entry / "entry.json").read_text(encoding="utf-8")))
    index.write_text(json.dumps(doc, sort_keys=True, indent=1) + "\n", encoding="utf-8")


def export_context(att: AttentionPass, entries: list) -> list[str]:
    """Export closed *entries*' material read-only under the pass's ``/inputs`` (their staging and captures), for
    reading beside the pass's own entries; nothing is owed for them. Returns the box paths exported."""
    from . import captures as _captures
    from . import staging as _staging

    state = att.ledger.state_dir
    root = state / "memory-staging"
    out: list[str] = []
    cids: list[str] = []
    for e in entries:
        for name in _att.pass_episodes([e]):
            dest = att.inputs / "staging" / name
            if not dest.exists():
                view = _staging.read(root, name)
                if view is not None:
                    _staging.export(view, dest)
            if dest.exists():
                out.append(f"{_att.STAGING_INPUTS}/{name}/")
        cids += list(e.refs.captures)
    if cids:
        try:
            done = _captures.export_for_writer(_captures.store_dir(state), cids, att.inputs / "captures")
        except OSError:
            done = []
        out += [f"{_att.CAPTURES_INPUTS}/{c}/" for c in done]
    return out


def export_group(att: AttentionPass, handed: list) -> dict | None:
    """Write ``/inputs/attention/group.json`` (:func:`.capture_groups.group_doc`): for a pass handed a capture group
    (:meth:`.attention.Ledger.group_of`) its members, its related closed entries, why each pair is related and the
    library items it touches, with the material of its most related closed entries (at most
    :data:`.attention.K`; the rest are listed, and ``tools.take`` exports any of them) exported read-only; for any
    other pass its entries and the library items they touch (``group`` null). Returns the document (None only when
    the pass has no entries); :attr:`AttentionPass.group` keeps it when it is a group."""
    from . import capture_groups as cg
    from . import captures as _captures

    if not handed:
        return None
    row = att.ledger.group_of(att.pass_id) or {"group": None, "due": None, "members": [e.entry for e in handed]}
    entries = att.ledger.entries()
    members = [e for e in handed if e.entry in set(row.get("members") or [])] or list(handed)
    context = [entries[c] for c in row.get("context") or [] if c in entries]
    try:
        evidence = att.sources.stores().evidence if att.sources is not None else None
    except Exception:  # noqa: BLE001 - without the evidence store the captures' own records still describe them
        evidence = None
    profiles = cg.HomeProfiles(att.ledger.state_dir, evidence)
    # the home's shared call features relate nothing, as when the group was formed
    common = cg.common_features(
        profiles(e) for e in entries.values() if e.kind in cg.GROUPED_KINDS and e.refs.captures and e.state != "not_reached"
    )
    pairs = []
    everyone = [*members, *context]
    for i, a in enumerate(everyone):
        for b in everyone[i + 1 :]:
            why = cg.related(profiles(a), profiles(b), common)
            if why:
                pairs.append({"a": a.entry, "b": b.entry, "why": why})
    group = cg.Group(id=str(row.get("group") or ""), open=members, context=context, relations=pairs,
                     due=row.get("due") if isinstance(row.get("due"), str) else None)  # fmt: skip
    landed = {cid: doc.get("item") for cid, doc in _captures.landings(_captures.store_dir(att.ledger.state_dir)).items()}
    doc = cg.group_doc(group, profiles, landed=landed)
    if row.get("group") is None:
        doc["group"] = None
    # the most related closed entries first (how many relations each shares with the members), then the most recent
    shown = cg.by_strength(group)[: _att.K]
    doc["exported_closed"] = [e.entry for e in shown]
    export_context(att, shown)
    path = att.inputs / "attention" / cg.GROUP_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    if doc["group"] is not None:
        att.group = doc
    return doc


def write_read_log(writer: Any) -> None:
    """At the pass's end: keep its read evidence (:func:`reads_of`) beside its records (``read-log.json``), for the
    service's :meth:`.attention.Ledger.verify_reads`. Never raises."""
    att = getattr(writer, "_attention", None)
    if att is None:
        return
    try:
        _att.write_pass_reads(att.read_log_path, reads_of(writer))
    except Exception:  # noqa: BLE001 - the service then finds nothing read, and the entries come back
        pass


def first_message(att: AttentionPass, req: Any, wall: str) -> str:
    """An attention pass's first user message: its id, its entries and its wall clock (byte-stable apart from the
    ids and the number)."""
    handed = list(getattr(req, "attention", None) or [])
    ids = ", ".join(f"{e.entry} ({e.kind})" for e in handed)
    group = getattr(att, "group", None)
    if group is not None:
        closed = [m for m in group.get("related_closed") or [] if isinstance(m, dict)]
        # who closed each: an earlier pass's decision, or the harness when no pass finished its group
        by_harness = sum(1 for m in closed if m.get("closed_by"))
        by_pass = len(closed) - by_harness
        history = [f"{n} related {'capture' if n == 1 else 'captures'} {who}" for n, who in (
            (by_pass, "an earlier pass decided"),
            (by_harness, "the harness closed when no pass finished its group"),
        ) if n]  # fmt: skip
        lib = len(group.get("library_items") or [])
        # captures, unless the group holds an episode that came for its lessons (no capture) or a message handed along
        other = [e for e in handed if e.kind != "new_experience"
                 or (getattr(e, "refs", None) is not None and not e.refs.captures)]  # fmt: skip
        noun = ("entry" if len(handed) == 1 else "entries") if other else ("capture" if len(handed) == 1 else "captures")
        return (
            f"Pass {att.pass_id}: a group of {len(handed)} related "
            f"{noun} needs your attention: {ids}"
            + (", with " + " and ".join(history) if history else "")
            + ".\n"
            "/inputs/attention/group.json says why the harness put them together (each related pair with its "
            "relations), what was decided on the earlier ones, "
            f"and the {lib} library {'item' if lib == 1 else 'items'} the group touches. Read them "
            "together with the library, and decide what to keep. What each entry "
            "owes a read is in /inputs/attention/index.json; decide each with tools.decide.\n"
            f"Wall clock for this pass: {wall} s.\n"
        )
    return (
        f"Pass {att.pass_id}: {len(handed)} {'entry needs' if len(handed) == 1 else 'entries need'} your attention: "
        f"{ids}.\n"
        "What each owes a read is in /inputs/attention/index.json; decide each with tools.decide.\n"
        f"Wall clock for this pass: {wall} s.\n"
    )
