"""The records the harness keeps, each field with the code that writes it and the code that reads it, and the flow a
captured procedure's uses take from a cell's call to every place the capture is shown.

A field a reader reads must be written by its writer (or the reader removed). The registry below names, for the
capture store's ``record.json``, a request's use record (``memory_use.json``), the evidence store's ``item_use`` rows,
the signals, the attention ledger's rows and a writer pass's ``group.json``, each field with its writer and its
readers. The tests drive the real writers, then
run the real readers over what they wrote and record every field each reader asked for: a reader that asks for a
field nothing writes fails here.

The flow: a captured procedure is stored by the capture step; a cell calls it twice through the generated package,
once successfully and once with an input it fails on; the cell's audit events become the request's use record, which
the evidence store indexes; the next export then shows those two uses on the capture's card, in the librarian's
catalogue and search rows and in the evidence the librarian reads, and gives the capture its status from them.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from tests.memory.test_capture_examples import _cell
from tests.memory.test_use_library_mode import library_root  # noqa: F401 - a fixture
from tests.memory.test_capture_trace import ENV_PROC, VALUE_PROC, _staging
from tests.memory.test_capture_trace import _ep as _capture_ep
from tests.memory.test_capture_trace import _write_action
from tests.memory.test_episodes import _ep
from tests.memory.test_layout import _tree
from unify.memory import attention as at
from unify.memory import capture_groups as cg
from unify.memory import captures as cp
from unify.memory import evidence as ev_mod
from unify.memory import librarian, library_search, lifecycle, pool, serving, trust
from unify.memory import library_export as lx
from unify.memory import library_helper as lh
from unify.memory import land_capture, writer_pass
from unify.memory.catalogue import write_files
from unify.memory.episodes import EpisodeWriter
from unify.memory.integration import hooks
from unify.memory.integration.adapters.audit import CellAudit
from unify.memory.integration.consolidate import open_stores
from unify.memory.integration.paths import Paths
from unify.memory.integration.request import (
    add_runtime_calls,
    runtime_calls_empty,
    with_answer_attributions,
    with_answered_from,
    with_pins,
    with_reasks,
    with_runtime_calls,
)
from unify.memory.layout import discover
from unify.memory.redact import Redactor
from unify.memory.signals import Signal, post_signal

# --- the registry ----------------------------------------------------------------------------------------------------

#: Each record, its writer, each field it writes and the readers of each field (``module:function``). ``conditional``
#: fields are written only in the case their note names.
REGISTRY: dict[str, dict] = {
    "capture record.json": {
        "writer": "unify.memory.captures:capture_trace_step (stored), unify.memory.captures:record_uses (its uses, "
        "at the end of a request that called it), unify.memory.captures:listed_record (exported)",
        "fields": {
            "v": ["unify.memory.captures:capture_trace_step"],
            "capture_id": [],
            "episodes": [
                "unify.memory.captures:record_support",
                "unify.memory.library_helper:_Catalog.generalised_by",
                "unify.memory.library_export:dispatch_data",
                "unify.memory.pool:from_capture",
            ],
            "cases": ["unify.memory.captures:record_support", "unify.memory.captures:refresh",
                      "unify.memory.library_export:dispatch_data"],
            "outcome": [
                "unify.memory.captures:negative",
                "unify.memory.library_helper:_capture_lines",
                "unify.memory.library_helper:_Catalog._capture_failed",
                "unify.memory.librarian:library_items",
                "unify.memory.integration.hooks:library_counts",
                "unify.memory.capture_groups:_capture_facts",
                "unify.memory.library_export:catalogue_text",
            ],
            "deterministic": [],
            "applies": [],
            "holds": [],
            "entry_point": [
                "unify.memory.library_helper:_capture_form",
                "unify.memory.librarian:acting_captures",
                "unify.memory.library_export:_capture_reader",
                "unify.memory.land_capture:read_capture",
                "unify.memory.writer_pass:_capture_lands",
            ],
            "trace": [],
            "value": [],
            "own_check": [],
            "effects_seen": [],
            "shape": [],
            "occurrences": ["unify.memory.library_export:dispatch_data"],
            # rev 2.3: what it reads (the request, or the declared input for one built from an accepted answer's
            # function) and that function's example check when it was made
            "reads": ["unify.memory.library_helper:_capture_lines", "unify.memory.library_helper:_reads_declared"],
            "checked": ["unify.memory.library_helper:_capture_lines"],
            # exported only (captures.listed_record): what it rests on as far as its record shows it
            "support": ["unify.memory.library_helper:_capture_support", "unify.memory.library_export:capture_support"],
            # its uses, from the evidence store's item_use rows (serving.record_capture_uses), and what they give it
            # (captures.use_fields, the derivation the served evidence uses too)
            "uses": ["unify.memory.captures:refresh"],
            "track": [],
            "record_line": [],
            "status": [],
            "status_basis": [],
            "status_rule": [],
            "status_reason": [],
            "status_text": [],
            "level": [],
        },
        "conditional": {},
    },
    "item_use": {
        "writer": "unify.memory.evidence:EvidenceStore.index_episode (_use_rows)",
        "readers": ["unify.memory.lifecycle:facts_from_evidence"],
    },
    "memory_use.json": {
        "writer": "unify.memory.analysis.use:request_use, unify.memory.integration.request:with_runtime_calls, "
        "with_pins, with_answer_attributions, with_reasks and with_answered_from",
        "readers": [
            "unify.memory.evidence:EvidenceStore.index_episode",
            "unify.memory.integration.outcomes:implicit_evidence",
            "unify.memory.usage:writer_usage_record",
        ],
    },
    "signal": {
        "writer": "unify.memory.signals:post_signal (EvidenceStore.add_signal), read back by EvidenceStore.signals_for",
        "readers": [
            "unify.memory.serving:_contrary_signals",
            "unify.memory.serving:_failure_row",
            "unify.memory.trust:signal_outcome",
            "unify.memory.lifecycle:counted_signals",
            "unify.memory.captures:judged_outcome",
            "unify.memory.captures:final_value_signals",
        ],
    },
    "ledger row": {
        "writer": "unify.memory.attention:Ledger.create, Ledger.record_kin",
        "readers": ["unify.memory.attention:fold"],
    },
    "group.json": {
        "writer": "unify.memory.capture_groups:group_doc (attention_tools.export_group writes it)",
        "fields": ["v", "group", "due", "members", "related_closed", "relations", "library_items"],
        "member_fields": ["entry", "kind", "episodes", "captures", "form", "outcome", "effects", "sequence",
                          "used_items"],
        "closed_member_fields": ["decision", "decided_items"],
        "readers": ["the writer, through /inputs/attention/group.json"],
    },
}


def _resolve(name: str):
    """The function a registry name names (``module:function`` or ``module:Class.method``)."""
    import importlib

    module, _, attr = name.partition(":")
    obj = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def test_every_reader_the_registry_names_exists():
    for record, row in REGISTRY.items():
        readers = list(row.get("readers") or [])
        for names in (row.get("fields") or {}).values() if isinstance(row.get("fields"), dict) else ():
            readers += names
        for name in readers:
            if ":" in name:
                assert callable(_resolve(name)), (record, name)


# --- recording what a reader asks for ------------------------------------------------------------------------------


class _Asked(dict):
    """A capture record that records each field asked of it."""

    asked: set = set()

    def get(self, key, default=None):
        _Asked.asked.add(key)
        return super().get(key, default)

    def __getitem__(self, key):
        _Asked.asked.add(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        _Asked.asked.add(key)
        return super().__contains__(key)


def _recording_loads(real):
    def loads(*args, **kwargs):
        value = real(*args, **kwargs)
        if isinstance(value, dict) and "capture_id" in value and "cases" in value:
            return _Asked(value)
        return value

    return loads


# --- the flow ----------------------------------------------------------------------------------------------------------

ASKED = 'Total these: {"rows": {"a": [1, 2, 3], "b": [4, 5]}}'
CALLS = '''
seen = []
sys.addaudithook(lambda e, a: seen.append([e, list(a)]) if e == "memory.call" else None)
c = memory.get(CID)
ok = c(ASKED)
bad = err(c, "nothing to total here")
acting = err(memory.get(ACTING))
acting_text = memory.get(ACTING).text
print(json.dumps({"ok": ok, "bad": bad, "acting": acting, "acting_text": acting_text, "events": seen}))
'''


def _store_two_captures(home: Path) -> tuple[object, Path, str, str]:
    """The run's stores and capture store with a value capture (from episode e1) and an acting one (from e0), as the
    capture step stores them."""
    stores = open_stores(Paths.under(home))
    state = Path(stores.paths.state_dir)
    store = cp.store_dir(state)
    row = cp.capture_trace_step(_staging(state, "e1", {"procedure.py": VALUE_PROC}), _capture_ep("e1"), store)
    acting = cp.capture_trace_step(
        _staging(state, "e0", {"procedure.py": ENV_PROC}), _capture_ep("e0", actions=[_write_action()]), store
    )
    assert row["safety"] == "ok" and acting["safety"] == "ok"
    return stores, store, row["capture_id"], acting["capture_id"]


def _export(tmp: Path, store: Path, served: dict | None) -> tuple[Path, dict]:
    """A request's library copy as the export writes it, with the capture snapshot and the evidence *served*."""
    tree = _tree(tmp)
    snapshot = cp.export_files(store)
    files = lx.generated_library_mode(tree, librarian=True, captures=snapshot,
                                      **({"serving": served} if served is not None else {}))  # fmt: skip
    write_files(tree, files)
    for rel, data in snapshot.items():
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_bytes(data)
    return tree, files


def _served(stores, store: Path, tree: Path) -> dict:
    caps = cp.load(store)
    found = serving.item_evidence(stores, captures=caps, home=None)
    return {"evidence": serving.evidence_doc(found, serving.conditions(tree, {}, caps), {}), "withdrawn_items": {}}


def _index_use(stores, eid: str, events: list) -> None:
    """The request's use record from a cell's audit events, as the worker's hook drains them, indexed by the evidence
    store as the episode's end indexes it."""
    hook = CellAudit(())
    hook.on = True
    for event, args in events:
        hook(event, tuple(args))
    acc = runtime_calls_empty()
    add_runtime_calls(acc, hook.drain())
    use = with_runtime_calls(
        {"version": 6, "items": {}, "items_outcome_unknown": [], "outcomes_known": True, "items_at_pin": [],
         "layout": "library_mode"},
        acc,
    )  # fmt: skip
    ep = _ep(episode_id=eid, regime="none", memory_main="c0ffee" * 6 + "abcd")
    ep.memory_use = use
    sha = EpisodeWriter(stores.episodes, stores.blobs, Redactor()).write(ep)
    stores.evidence.index_episode(ep, sha)


def test_a_captures_uses_reach_its_card_the_catalogue_the_search_rows_the_librarian_and_its_status(tmp_path):
    stores, store, cid, acting = _store_two_captures(tmp_path / "home")
    first, _ = _export(tmp_path / "first", store, _served(stores, store, tmp_path / "first"))
    got = _cell(
        first,
        f"CID, ACTING, ASKED = {'capture:' + cid!r}, {'capture:' + acting!r}, {ASKED!r}\n" + CALLS,
    )
    # the actor's two calls: one returned, one raised; the acting capture is not run by a call, and raises no event
    assert got["ok"] == [6, 9]
    assert got["bad"][0] == "ArgumentError"
    assert got["acting"][0] == "WrongCall" and f"print(memory.get('capture:{acting}').text)" in got["acting"][1]
    assert "exec(" not in got["acting"][1] and got["acting_text"] == ENV_PROC
    assert sorted(e[1][2] for e in got["events"]) == ["error", "ok"]
    assert {e[1][0] for e in got["events"]} == {f"capture:{cid}"}

    _index_use(stores, "e2", got["events"])
    found = serving.item_evidence(stores, captures=cp.load(store), home=None)
    row = found[f"capture:{cid}"]
    assert row["later_uses"] == {
        "episodes": 1, "returned": 1, "refused": 0, "raised": 1, "accepted": 0, "rejected": 0, "unknown": 1
    }
    assert [(r["episode"], r["why"]) for r in row["failure_record"]] == [("e2", "error")]
    # trust and status come from those uses (with the capture's own episodes), never from its record
    assert row["counts"] == {trust.CLEAN: 1, trust.FAILED: 1, trust.NEUTRAL: 0}
    assert row["status"] == trust.ON_TRIAL and row["support"] == {"episodes": 1, "accepted": 0, "confirmed_uses": 0}
    assert found[f"capture:{acting}"]["later_uses"]["episodes"] == 0  # every capture has a row, used or not
    # the request's end writes the same uses into the capture's record: the record and the evidence agree
    assert serving.record_capture_uses(stores, store, [f"capture:{cid}", f"capture:{acting}"]) == [f"capture:{cid}"]
    rec = json.loads((store / cid / cp.RECORD_FILE).read_text())
    facts = lifecycle.facts_from_evidence(stores.evidence)
    assert rec["uses"] == [u.row() for u in serving.capture_uses(facts, f"capture:{cid}")]
    assert [u["episode"] for u in rec["uses"]] == ["e2", "e2"]
    assert (rec["status"], rec["level"]) == (row["status"], row["level"])

    root, files = _export(tmp_path / "second", store, _served(stores, store, tmp_path / "second"))
    # rev 2.3: then the uses whose call returned, by outcome (no verdict was attributed to this one)
    later = "2 calls in 1 episode: 1 returned, 1 raised; uses: 0 accepted, 0 rejected, 1 unknown"
    card = _cell(root, f"print(json.dumps(str(memory.get({'capture:' + cid!r}))))\n")
    assert f"Later uses: {later}" in card and "Recorded failures: 1: error in episode e2" in card
    assert card.startswith(f"capture:{cid}  [recorded procedure from 1 episode]")
    catalogue = files[".memory/catalogue.md"].decode()
    section = catalogue.split(f"### capture:{cid}\n", 1)[1].split("###", 1)[0]
    assert f"Later uses: {later}" in section and "Recorded failures: 1: error in episode e2" in section
    rows = {r["id"]: r for r in json.loads(files[".memory/search.json"])["items"]}
    assert f"later uses: {later}" in rows[f"capture:{cid}"]["card"]
    assert "recorded failures: 1: error in episode e2" in rows[f"capture:{cid}"]["card"]
    # the librarian reads the same: the evidence beside its copy, its search results and the cards in its box
    seen = librarian.read_evidence(root)[f"capture:{cid}"]
    assert seen["later_uses_text"] == later and seen["failure_record"][0]["why"] == "error"
    searched = library_search.search_text(library_search.read_index(root), "total the rows")
    assert f"later uses: {later}" in searched


#: A captured procedure that refuses an input it was not built for (its own ``MemoryInputError``, as a library function
#: raises one).
REFUSING_PROC = '''"""Total each named row of the request's table. Accepted: not shown."""


class MemoryInputError(ValueError):
    """The request carries no table of named rows."""


def read_input(request):
    """The request's rows, by name."""
    if not request.data or not isinstance(request.data[0], dict) or "rows" not in request.data[0]:
        raise MemoryInputError("the request carries no table of named rows")
    return request.data[0]["rows"]


def run(value):
    """Each row's total, in name order."""
    return [sum(value[k]) for k in sorted(value)]
'''
ONE_CALL = '''
seen = []
sys.addaudithook(lambda e, a: seen.append([e, list(a)]) if e == "memory.call" else None)
got = err(memory.get(CID), ASKED) if REFUSE else memory.get(CID)(ASKED)
print(json.dumps({"got": got, "events": seen}))
'''


def test_capture_use_is_metered(tmp_path):
    """A capture called once cleanly (one request) and once refused (another) shows two later uses, one clean and one
    refused: on its card, in the served evidence and in its record; the writer's usage table counts both calls, and the
    metering view of the items a decision named counts both requests, one of them clean."""
    from unify.memory import metering, usage

    home = tmp_path / "home"
    stores = open_stores(Paths.under(home))
    state = Path(stores.paths.state_dir)
    store = cp.store_dir(state)
    row = cp.capture_trace_step(_staging(state, "e1", {"procedure.py": REFUSING_PROC}), _capture_ep("e1"), store)
    assert row["safety"] == "ok"
    item = "capture:" + row["capture_id"]
    first, _ = _export(tmp_path / "first", store, _served(stores, store, tmp_path / "first"))
    clean = _cell(first, f"CID, ASKED, REFUSE = {item!r}, {ASKED!r}, False\n" + ONE_CALL)
    refused = _cell(first, f"CID, ASKED, REFUSE = {item!r}, 'nothing to total here', True\n" + ONE_CALL)
    assert clean["got"] == [6, 9]
    assert refused["got"][0] == "ArgumentError" and "refused this input" in refused["got"][1], refused["got"]
    assert [e[1][2] for e in clean["events"]] == ["ok"] and [e[1][2] for e in refused["events"]] == ["refused"]
    _index_use(stores, "e2", clean["events"])
    _index_use(stores, "e3", refused["events"])

    found = serving.item_evidence(stores, captures=cp.load(store), home=None)[item]
    assert found["later_uses"] == {
        "episodes": 2, "returned": 1, "refused": 1, "raised": 0, "accepted": 0, "rejected": 0, "unknown": 1
    }
    assert serving.record_capture_uses(stores, store, [item]) == [item]
    rec = json.loads((store / row["capture_id"] / cp.RECORD_FILE).read_text())
    assert sorted((u["episode"], u["outcome"]) for u in rec["uses"]) == [("e2", "returned"), ("e3", "refused")]

    root, _ = _export(tmp_path / "second", store, _served(stores, store, tmp_path / "second"))
    card = _cell(root, f"print(json.dumps(str(memory.get({item!r}))))\n")
    assert "Later uses: 2 calls in 2 episodes: 1 returned, 1 refused" in card, card

    # the writer's usage table: two requests called it, two calls
    table = usage.usage_table(stores.evidence, ["e2", "e3"], [item])
    line = next(x for x in table.splitlines() if x.startswith(item + " |"))
    assert [c.strip() for c in line.split("|")][4:6] == ["2", "2"], line  # requests calling, calls

    # metering: an earlier decision named the capture; its later uses are the two requests, one clean
    seq = dict(stores.evidence.db.execute("SELECT episode_id, seq FROM episodes").fetchall())
    ledger = state / "memory-attention" / "ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps({"op": "close", "entry": "E1", "pass_id": "p1", "decision": "note",
                                  "items": [item]}) + "\n")  # fmt: skip
    passes = [{"pass_id": "p1", "entries": {"closed": ["E1"]}, "episodes_seq_at_end": min(seq["e2"], seq["e3"]) - 1}]
    view = metering.items_view(home, passes, [{"entry": "E1", "basis": "read_equal_split", "share_usd": "0.01"}])
    assert [(r["item"], r["later_calls"], r["later_clean"]) for r in view] == [(item, 2, 1)]


def test_every_field_a_capture_reader_asks_for_is_written(tmp_path, monkeypatch):
    stores, store, cid, acting = _store_two_captures(tmp_path / "home")
    root, _ = _export(tmp_path / "copy", store, _served(stores, store, tmp_path / "copy"))
    stored = json.loads((store / cid / cp.RECORD_FILE).read_text())
    exported = json.loads((root / cp.EXPORT_DIR / cid / cp.RECORD_FILE).read_text())
    fields = REGISTRY["capture record.json"]["fields"]
    # what the writers write is what the registry names: nothing more, nothing less
    assert set(stored) | set(exported) == set(fields), sorted(set(stored) ^ set(fields))
    assert set(exported) - set(stored) == {"support"}

    monkeypatch.setattr(json, "loads", _recording_loads(json.loads))
    monkeypatch.setattr(lh, "_ROOT", str(root))
    _Asked.asked = set()
    snapshot = cp.export_files(store)
    caps = cp.load(store)
    # the package's readers, in this process on the request's copy
    cat = lh._Catalog("memory.get(...)")
    for c in (cid, acting):
        str(cat.item("capture:" + c))
        lh._capture_form(c)
        lh._run_capture(acting, (), {})
    cat.listed()
    cat.generalises(cat.functions[0])
    lh._capture_candidates()
    # the librarian's, the export's and the search index's readers
    librarian.library_items(root)
    librarian.acting_captures(root)
    hooks.library_counts(root)
    lx.catalogue_text(root, captures=snapshot)
    lx.dispatch_data(discover(root), captures=snapshot)
    library_search.rows(root, captures=snapshot)
    # serving, the writer's and the pool's readers
    serving.item_evidence(stores, captures=caps, home=None)
    for c in (cid, acting):
        pool.from_capture(c, caps[c])
        writer_pass._capture_lands(store, c)
        cg._capture_facts(store / c)
        try:
            land_capture.read_capture(store, c)
        except Exception:  # noqa: BLE001 - an acting capture is refused after its record is read
            pass
        cp.negative(caps[c]["record"])
    cp.export_for_writer(store, [cid, acting], tmp_path / "inputs")
    asked = set(_Asked.asked)
    assert asked, "no reader read a capture record"
    unwritten = asked - (set(stored) | set(exported))
    assert not unwritten, f"fields read that no writer writes: {sorted(unwritten)}"
    assert asked <= set(fields)


# --- item_use: the columns its readers select are the ones its writer fills -----------------------------------------


def _selected_columns(fn, table: str) -> set[str]:
    """The columns *fn*'s SQL selects from *table* (a parse of its string literals, never a run)."""
    out: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(fn).lstrip())):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and f"FROM {table}" in node.value:
            head = node.value.split("SELECT", 1)[1].split(f"FROM {table}", 1)[0]
            out |= {c.strip() for c in head.split(",") if c.strip()}
    return out


def test_the_item_use_columns_a_reader_selects_are_written(tmp_path):
    stores = open_stores(Paths.under(tmp_path / "home"))
    have = {r[1] for r in stores.evidence.db.execute("PRAGMA table_info(item_use)")}
    written = {"item", "episode_id", *ev_mod._USE_COUNTS}
    assert written <= have
    selected = set()
    for reader in REGISTRY["item_use"]["readers"]:
        selected |= _selected_columns(_resolve(reader), "item_use")
    assert selected and selected <= written, sorted(selected - written)
    # and the reader takes a captured procedure's rows as it takes a library item's
    ep = _ep(episode_id="e5", regime="none")
    ep.memory_use = {"items": {"capture:" + "a" * 16: {"runtime": {"ok": 1, "refused": 0, "error": 1}}},
                     "outcomes_known": True, "runtime_calls": {"cells": 1}}  # fmt: skip
    sha = EpisodeWriter(stores.episodes, stores.blobs, Redactor()).write(ep)
    stores.evidence.index_episode(ep, sha)
    (f,) = [f for f in lifecycle.facts_from_evidence(stores.evidence) if f.episode_id == "e5"]
    assert f.items["capture:" + "a" * 16].calls == (1, 0, 1)


# --- memory_use.json: what its readers ask of the record and of each item's row is what its writers wrote -------------


class _Asking(dict):
    """A use record (or an item's row of it) that records each field asked of it under its *level*."""

    def __init__(self, data: dict, level: str, asked: dict) -> None:
        super().__init__(data)
        self.level, self.asked = level, asked

    def _note(self, key) -> None:
        self.asked.setdefault(self.level, set()).add(key)

    def get(self, key, default=None):
        self._note(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self._note(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        self._note(key)
        return super().__contains__(key)


def test_every_field_a_use_record_reader_asks_for_is_written(tmp_path, library_root):
    from tests.memory.test_use_library_mode import ITEMS_LIBRARY, _use_library
    from unify.memory.integration import outcomes
    from unify.memory.usage import writer_usage_record

    item = ITEMS_LIBRARY[0]
    record = _use_library(library_root, ["from memory.text.dates import parse_date\nparse_date('2026-10-09')\n"])
    acc = runtime_calls_empty()
    add_runtime_calls(acc, {"calls": {"counts": {item: {"ok": 1, "refused": 0, "error": 1}}}})
    record = with_runtime_calls(record, acc)
    # UNIFY_MEMORY=in_episode: the pin each call was served under
    record = with_pins(record, {"c0ffee" * 6 + "abcd": {item: {"ok": 1, "refused": 0, "error": 1}}}, ["c0ffee" * 6 + "abcd"])
    # rev 2.3: the verdicts taken within the request, with the items whose returned value the judged answer was
    record = with_answer_attributions(record, [{"label": "fail", "obs": 1, "items": [item], "short": False}])
    # rev 2.3: each time the librarian was asked again on new evidence
    record = with_reasks(record, [{"obs": 2, "events": ["outcome"], "fit": item}])
    # rev 2.3: each answer made with tools.answer_from
    record = with_answered_from(record, [{"item": item, "source": "declared", "answered": True, "obs": 1}])
    rows = record["items"]
    assert item in rows and "runtime" in rows[item]
    written = {"record": set(record), "item": {k for row in rows.values() for k in row}}
    asked: dict = {}
    wrapped = _Asking({**record, "items": {k: _Asking(v, "item", asked) for k, v in rows.items()}}, "record", asked)

    stores = open_stores(Paths.under(tmp_path / "home"))
    ep = _ep(episode_id="e3", regime="none", memory_main="c0ffee" * 6 + "abcd")
    ep.memory_use = dict(record)
    sha = EpisodeWriter(stores.episodes, stores.blobs, Redactor()).write(ep)
    ep.memory_use = wrapped
    stores.evidence.index_episode(ep, sha)
    outcomes.implicit_evidence(wrapped)
    writer_usage_record(wrapped)
    assert asked.get("record") and asked.get("item"), asked
    for level, keys in asked.items():
        unwritten = keys - written[level]
        assert not unwritten, f"{level} fields read that no writer writes: {sorted(unwritten)}"


# --- signals: the fields their readers use round-trip through the store ------------------------------------------------


def _attributes_read(fn) -> set[str]:
    """The attributes *fn* reads of a signal: ``s.<name>`` and ``getattr(s, "<name>")`` on its signal variables."""
    out: set[str] = set()
    names = {"s", "sig", "last"}
    for node in ast.walk(ast.parse(inspect.getsource(fn).lstrip())):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in names:
            out.add(node.attr)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in names
            and isinstance(node.args[1], ast.Constant)
        ):
            out.add(node.args[1].value)
    return out


def test_every_signal_field_a_reader_uses_is_written_and_read_back(tmp_path):
    import dataclasses

    stores = open_stores(Paths.under(tmp_path / "home"))
    ep = _ep(episode_id="e7", regime="sparse")
    sha = EpisodeWriter(stores.episodes, stores.blobs, Redactor()).write(ep)
    stores.evidence.index_episode(ep, sha)
    sent = Signal("e7.checker.1", "e7", "checker", "fail", "2026-10-08T01:05:00Z", refers_to="e7/actions/0",
                  regime="sparse", visible_to_actor=True, arrived_at="2026-10-08T01:05:00Z",
                  episode_ended_at="2026-10-08T01:01:00Z", timing="after_episode")  # fmt: skip
    post_signal(sent, stores.episodes, sha, stores.evidence)
    (back,) = stores.evidence.signals_for("e7")
    fields = {f.name for f in dataclasses.fields(Signal)}
    used = set()
    for reader in REGISTRY["signal"]["readers"]:
        used |= _attributes_read(_resolve(reader))
    assert used and used <= fields, sorted(used - fields)
    for name in used:
        assert getattr(back, name) == getattr(sent, name), name


# --- the attention ledger: what fold reads of a row is what the ledger wrote ------------------------------------------


def test_the_ledger_rows_fold_reads_are_the_rows_the_ledger_writes(tmp_path, monkeypatch):
    led = at.Ledger(tmp_path / "home")
    a = led.create("new_experience", at.Refs(episodes=("e1",), captures=("1" * 16,)), "e1")
    b = led.create("new_experience", at.Refs(episodes=("e2",), captures=("2" * 16,)), "e2")
    assert led.record_kin(b, [a], "both total a table")
    rows = led.rows()
    written: dict[str, set] = {}
    for r in rows:
        written.setdefault(r["op"], set()).update(r)

    class Row(dict):
        def get(self, key, default=None):
            asked.setdefault(dict.get(self, "op"), set()).add(key)
            return super().get(key, default)

        def __getitem__(self, key):
            asked.setdefault(dict.get(self, "op"), set()).add(key)
            return super().__getitem__(key)

    asked: dict[str, set] = {}
    folded = at.fold([Row(r) for r in rows])
    assert folded[b].kin == (a,)
    for op, keys in asked.items():
        assert keys <= written.get(op, set()), (op, sorted(keys - written.get(op, set())))


# --- group.json: the fields the writer is handed are the ones the registry names ------------------------------------


def test_the_group_document_holds_the_fields_the_registry_names():
    members = [
        at.AttentionEntry("a" * 16, "new_experience", "open", "2026-10-20T00:00:00.000Z",
                          at.Refs(episodes=("a" * 16,), captures=("a" * 16,))),
    ]  # fmt: skip
    closed = at.AttentionEntry("c" * 16, "new_experience", "closed", "2026-10-19T00:00:00.000Z",
                               at.Refs(episodes=("c" * 16,), captures=("c" * 16,)), decision="new_item",
                               decided_items=("memory.p.m:f",))  # fmt: skip

    def profile(e):
        return cg.Profile(e.entry, (e.entry[:16],), "state_env", frozenset(), frozenset(), frozenset(), "none", ())

    doc = cg.group_doc(cg.Group(id="a" * 16, open=members, context=[closed], due="formed"), profile)
    spec = REGISTRY["group.json"]
    assert sorted(doc) == sorted(spec["fields"])
    assert sorted(doc["members"][0]) == sorted(spec["member_fields"])
    assert sorted(doc["related_closed"][0]) == sorted(spec["member_fields"] + spec["closed_member_fields"])


@pytest.mark.parametrize("name", cp.USE_FIELDS)
def test_a_capture_record_keeps_its_uses_and_what_they_give_it(tmp_path, name):
    _, store, cid, _ = _store_two_captures(tmp_path / "home")
    rec = json.loads((store / cid / cp.RECORD_FILE).read_text())
    # stored with no use yet, and the fields that none gives it
    assert name in rec and rec["uses"] == []
    # a record rewritten keeps them as its uses give them (one derivation), never a value written over them
    old = {**rec, name: "x"}
    assert cp.refresh(old)[name] == cp.use_fields(rec, [])[name]
