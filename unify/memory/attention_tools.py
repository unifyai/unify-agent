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
    owed a read and a decision like the handed ones."""
    writer, att = _state(ctx)
    if att is None:
        return _no_ledger("take")
    eid = args.get("entry_id")
    if not isinstance(eid, str) or not eid.strip():
        return _refused("take", "entry_id is one entry id, a string; for example tools.take('a1b2c3d4e5f60718')")
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
    """``tools.find(query, scope="items", k=10)``: rank the library's items by their "When to use" text (the
    librarian's ranking), the episodes' summaries, or the open entries, against *query*; the best *k*. The writer
    drives it: nothing here proposes a group."""
    from .library_search import LEXICAL, index_data, ranking

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
    # the same ranking as the librarian's search; the writer's search makes no embedding call (no route for one here)
    ranked, how = ranking(index_data(found, LEXICAL), query, LEXICAL)
    hits = [
        {"id": r.get("id"), "kind": r.get("kind"), "path": r.get("path"), "score": round(float(s), 4),
         "card": r.get("card")}
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
    return (
        f"Pass {att.pass_id}: {len(handed)} {'entry needs' if len(handed) == 1 else 'entries need'} your attention: "
        f"{ids}.\n"
        "What each owes a read is in /inputs/attention/index.json; decide each with tools.decide.\n"
        f"Wall clock for this pass: {wall} s.\n"
    )

