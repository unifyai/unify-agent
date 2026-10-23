"""The capture store: every completed attempt captured at once, by the harness, outside the library; its signature
holds no request text and keeps every key as written (rev 2.1: no identifier is stripped or masked); a failure is negative knowledge; outcomes join one record; the writer reads the store and stays
the only agent that edits the library. Rev 2 (the trace contract): a capture is never run; its outcome comes from the
episode's signals only."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from unify.memory import captures as cp
from unify.memory import pool, serving
from unify.memory import trust as tr
from unify.memory.episodes import Action, Episode
from unify.memory.integration import fork as fork_mod
from unify.memory.integration import fork_worker as fw
from unify.memory.integration.consolidate import open_stores
from unify.memory.integration.paths import Paths
from unify.memory.signals import Signal

needs_bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap required")

REQUEST = json.dumps(
    {"ticket": "lunar-otter", "case": "0f3a9c7e1b", "rows": {"ab12cd34ef": [1, 2, 3], "run_000123": [4, 5]}},
)
PROC = '''"""Total each named row of the request's table. Accepted: "checker: pass"."""

import json


def read_input(text):
    """The request's rows, by name."""
    return json.loads(text)["rows"]


def run(value):
    """Each row's total, in name order."""
    return [sum(value[k]) for k in sorted(value)]


def holds(value, result):
    """One total per row."""
    return len(result) == len(value)
'''
BROKE = "- totals: summing the header row failed; \"ValueError: invalid literal\"\n"


def _ep(eid, request=REQUEST, value=(6, 9), regime="dense"):
    return Episode(
        episode_id=eid,
        started_at="2026-10-13T00:00:00Z",
        ended_at="2026-10-13T00:01:00Z",
        build="b",
        model="m",
        effort="low",
        regime=regime,
        memory_main="0" * 40,
        worktree_before=None,
        worktree_after=None,
        request=[request],
        transcript=[],
        cells=[],
        actions=[Action(0, "env", "act", [list(value)], {}, "ok, thanks", "ok", "unknown", kind="dialogue")],
    )


def _staging(root: Path, eid: str, files: dict[str, str]) -> Path:
    d = root / "memory-staging" / eid
    d.mkdir(parents=True)
    for rel, text in files.items():
        (d / rel).write_text(text)
    (d / "fork.json").write_text(json.dumps({"episode": eid, "status": "ok"}))
    return d


def _checker(eid, label, regime="dense"):
    return Signal(f"s-{eid}", eid, "checker", label, "t", regime=regime, visible_to_actor=True)


# --- capture at once, outside the library ------------------------------------------------------------------------


def test_a_completed_attempt_is_captured_with_its_signature_keys_as_written(tmp_path):
    staging = _staging(tmp_path, "e1", {"procedure.py": PROC, "broke.md": "none\n"})
    store = cp.store_dir(tmp_path)
    row = cp.capture_trace_step(
        staging,
        _ep("e1"),
        store,
        signals=[_checker("e1", "pass")],
    )
    cid = cp.capture_id(PROC)
    assert row["capture_id"] == cid and row["safety"] == "ok"
    # rev 2: never run (no reproduction, no conditions judged); the outcome is the signals'
    assert (row["run"], row["own_check"]["ran"], row["outcome"]) == ("not run", False, "positive")
    assert "reproduces" not in row and "holds" not in row
    assert row["published_at"] is not None and row["seconds"] is not None
    # the store sits beside staging, under the harness's state dir, never inside the fork's writable staging dir
    d = store / cid
    assert sorted(os.listdir(d)) == [cp.OCCURRENCES_FILE, "procedure.py", "record.json", "saved.json", "signature.json"]
    assert staging not in d.parents and store.parent == tmp_path
    saved_text = (d / cp.SAVED_FILE).read_text()  # what the capture was saved against: never the request text
    assert REQUEST not in saved_text
    sig_text = (d / "signature.json").read_text()
    # rev 2.1: the recorded data's keys are its own, kept as written (the rows' names here), never a placeholder
    tree = json.dumps(json.loads(sig_text)["input"])
    for key in ("ab12cd34ef", "run_000123"):
        assert key in tree, key
    assert "<id>" not in sig_text
    assert REQUEST not in sig_text and "rows" not in json.loads(sig_text)["input"].get("lengths", {})
    sig = json.loads(sig_text)
    # rev 2: nothing read the request (read_input never ran), so the input is described by the recording alone
    assert sig["input"]["kind"] == "value"
    assert sig["constraints"]["holds"]["stated"] and not sig["constraints"]["applies"]["stated"]
    rec = json.loads((d / "record.json").read_text())
    # the record keeps what the capture step saw, never uses or a status from them (those are the evidence store's)
    assert rec["episodes"] == ["e1"] and not set(cp.USE_FIELDS) & set(rec)
    # a capture starts on trial and unchecked, and nothing credits a use at capture time (the writer reads and groups)
    stores = open_stores(Paths.under(tmp_path / "home"))
    served = serving.item_evidence(stores, captures=cp.load(store), home=None)[f"capture:{cid}"]
    assert (served["status"], served["level"]) == (tr.ON_TRIAL, tr.UNCHECKED)
    assert served["later_uses"] == {"episodes": 0, "returned": 0, "refused": 0, "raised": 0}
    assert served["support"] == {"episodes": 1, "accepted": 1, "confirmed_uses": 0}
    events = [json.loads(x) for x in (store / cp.EVENTS_FILE).read_text().splitlines()]
    assert [e["capture_id"] for e in events] == [cid]
    assert cp.episode_row(store, "e1")["capture_id"] == cid
    # captured once: a second call for the same settled episode does nothing
    assert cp.capture_settled(tmp_path, ["e1"], episode=lambda e: _ep(e), signals=lambda e: []) == []


def test_a_key_shaped_procedure_is_refused_before_anything_runs(tmp_path):
    planted = PROC + '\nTOKEN = "sk-or-v1-' + "a" * 64 + '"\n'
    staging = _staging(tmp_path, "e1", {"procedure.py": planted})
    row = cp.capture_trace_step(staging, _ep("e1"), cp.store_dir(tmp_path))
    assert row["safety"] == "key-shaped string" and row["capture_id"] is None
    assert not any(cp._ID.fullmatch(n) for n in os.listdir(cp.store_dir(tmp_path)))


def test_an_attempt_that_handed_nothing_back_is_captured_without_a_value(tmp_path):
    """Rev 2 (the trace contract): every settled attempt is captured, whether or not it handed a value back; its
    record says no value was handed back."""
    ep = _ep("e1")
    ep.actions = []
    staging = _staging(tmp_path, "e1", {"procedure.py": PROC})
    row = cp.capture_trace_step(staging, ep, cp.store_dir(tmp_path))
    assert row["safety"] == "ok" and row["capture_id"] == cp.capture_id(PROC)
    assert row["value"]["present"] is False and row["outcome"] == "none"


# --- negative knowledge -------------------------------------------------------------------------------------------


def test_a_failure_is_captured_as_negative_knowledge(tmp_path):
    """Rev 2: an attempt whose episode carried a rejected verdict is captured as contrary, with what broke."""
    staging = _staging(tmp_path, "e1", {"procedure.py": PROC, "broke.md": BROKE})
    store = cp.store_dir(tmp_path)
    row = cp.capture_trace_step(staging, _ep("e1"), store, signals=[_checker("e1", "fail")])
    assert (row["run"], row["outcome"]) == ("not run", cp.CONTRARY)
    doc = cp.load(store)[row["capture_id"]]
    assert doc["record"]["outcome"] == cp.CONTRARY and not set(cp.USE_FIELDS) & set(doc["record"])
    assert cp.negative(doc["record"])
    assert "invalid literal" in doc["broke"]
    entry = pool.from_capture(row["capture_id"], doc)
    assert entry.kind == "failure" and "broke.md" in entry.files
    # a staged broke.md with no procedure enters the pool as a failure entry of its own
    view = type("V", (), {"files": {"broke.md": BROKE.encode()}, "source": "fork"})()
    [failure] = [e for e in pool.from_staging("e3", view) if e.kind == "failure"]
    assert failure.files == {"broke.md": BROKE}
    nothing = type("V", (), {"files": {"broke.md": b"none\n"}, "source": "fork"})()
    assert pool.from_staging("e4", nothing) == []


# --- what reads the store ---------------------------------------------------------------------------------------


def test_the_writer_reads_the_capture_store_and_no_other_agent_edits_the_library(tmp_path, monkeypatch):
    from tests.memory.test_pool import EMPTY, _Stores, _finish_with
    from tests.memory.test_writer_pass import _call

    state = tmp_path / "state"
    store = cp.store_dir(state)
    staging = _staging(state, "e2", {"procedure.py": PROC, "broke.md": BROKE})
    cid = cp.capture_trace_step(staging, _ep("e2"), store)["capture_id"]
    store_before = {p: p.read_bytes() for p in store.rglob("*") if p.is_file()}
    st = _Stores(tmp_path)
    _finish_with(monkeypatch, EMPTY)
    out, model = st.run(
        [
            _call("i", "read", {"path": "/inputs/pool/INDEX.md"}),
            _call("b", "read", {"path": "/inputs/batch_map.json"}),
            _call("c", "read", {"path": f"/inputs/captures/{cid}/procedure.py"}),
        ],
        "p1",
        staging_root=str(state / "memory-staging"),
    )
    assert pool.entry_id("capture", cid) in model.outputs["i"]
    assert f'"id": "{cid}"' in model.outputs["b"]
    assert "Total each named row" in model.outputs["c"]
    assert "/inputs/captures/<id>/" in model.seen[0]["content"]
    assert out.pool["at_start"]["captures"]["entries"] == 1
    # the writer read the store as input; nothing it or the pass did changed the store
    assert {p: p.read_bytes() for p in store.rglob("*") if p.is_file()} == store_before


def test_the_fork_box_can_write_only_its_staging_dir_never_the_store_or_the_library(tmp_path, monkeypatch):
    monkeypatch.setattr(fork_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    staging = tmp_path / "memory-staging" / "e1"
    staging.mkdir(parents=True)
    argv = fork_mod.box_argv(staging, die_with_parent=False)
    writable = [argv[i + 1] for i, a in enumerate(argv) if a == "--bind"]
    assert writable == [str(staging)]
    assert str(cp.store_dir(tmp_path)) not in argv


# --- the fork writes what is captured -----------------------------------------------------------------------------


def test_the_fork_may_stage_a_procedure_and_what_broke_and_is_told_how(monkeypatch):
    assert fw.allowed("procedure.py") is None and fw.allowed("broke.md") is None
    assert (fw.kind("procedure.py"), fw.kind("broke.md")) == ("procedure", "broke")
    assert fw.allowed("procedures/x.py") is not None
    text = fork_mod.fork_instruction()
    assert fork_mod.CAPTURE_BULLETS in text
    assert "`read_input(request)`" in text and "`broke.md`" in text
    # the procedure reads the values the request carries from its data (rev 2's evaluated wording)
    assert "`request` is its text, `request.data` the list of the JSON objects and arrays it contains" in text


def test_a_fork_reply_with_a_procedure_records_it_in_fork_json(tmp_path):
    reply = "```path: procedure.py\n" + PROC + "```\n\n```path: broke.md\n" + BROKE + "```\n"
    doc = fw.run(
        {"episode": "e1", "url": "http://x", "headers": {}, "body": {"model": "m", "messages": []}},
        tmp_path,
        post=lambda u, h, b: {"choices": [{"message": {"role": "assistant", "content": reply}}], "usage": {}},
    )
    assert doc["procedure"] is True and doc["broke"] is True
    assert (tmp_path / "procedure.py").read_text() == PROC


# --- the real confined box ----------------------------------------------------------------------------------------


