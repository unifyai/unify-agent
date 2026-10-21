"""The capture store: every completed attempt captured at once, by the harness, outside the library; its signature
holds no identifier; a failure is negative knowledge; outcomes join one record; the writer reads the store and stays
the only agent that edits the library."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import pytest

from unify.memory import captures as cp
from unify.memory import pool
from unify.memory import trust as tr
from unify.memory.episodes import Action, Episode
from unify.memory.integration import fork as fork_mod
from unify.memory.integration import fork_worker as fw
from unify.memory.sandbox_run import SandboxResult
from unify.memory.signals import Signal

needs_bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap required")

IDS = ("0f3a9c7e1b", "ab12cd34ef", "run_000123", "lunar-otter")
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


def _box(value, result, *, result2=None, outcome="returned", **extra):
    """A stand-in for the confined box: it writes the result a box would, and records how it was called. No model
    code runs."""
    calls: list = []

    def runner(argv, *, ro=None, rw=None, cwd="/tmp", timeout_s=120.0, env=None):
        calls.append({"argv": argv, "ro": ro, "rw": rw, "env": env})
        [out] = [h for h, b in (rw or {}).items() if b == "/out"]
        doc = {"outcome": outcome, "value": json.dumps(value), **extra}
        if outcome == "returned":
            doc["result"] = json.dumps(result)
            doc["result2"] = json.dumps(result if result2 is None else result2)
        (Path(out) / "result.json").write_text(json.dumps(doc))
        return SandboxResult(0, "", "", False)

    return runner, calls


def _checker(eid, label, regime="dense"):
    return Signal(f"s-{eid}", eid, "checker", label, "t", regime=regime, visible_to_actor=True)


VALUE = {"ab12cd34ef": [1, 2, 3], "run_000123": [4, 5]}


# --- capture at once, outside the library ------------------------------------------------------------------------


def test_a_completed_attempt_is_captured_with_its_signature_and_holds_no_identifiers(tmp_path):
    staging = _staging(tmp_path, "e1", {"procedure.py": PROC, "broke.md": "none\n"})
    store = cp.store_dir(tmp_path)
    runner, calls = _box(VALUE, [6, 9], holds=True)
    row = cp.capture(
        staging,
        _ep("e1"),
        store,
        signals=[_checker("e1", "pass")],
        checker_visible=True,
        runner=runner,
    )
    cid = cp.capture_id(PROC)
    assert row["capture_id"] == cid and row["safety"] == "ok"
    assert (row["reproduces"], row["deterministic"], row["holds"], row["outcome"]) == (
        "reproduces",
        True,
        True,
        "positive",
    )
    assert row["published_at"] is not None and row["seconds"] is not None
    # confined: the procedure and the request read-only, only /out writable, an empty environment
    [call] = calls
    assert call["env"] == {} and sorted(call["ro"].values()) == ["/case"] and sorted(call["rw"].values()) == ["/out"]
    # the store sits beside staging, under the harness's state dir, never inside the fork's writable staging dir
    d = store / cid
    assert sorted(os.listdir(d)) == ["procedure.py", "record.json", "saved.json", "signature.json"]
    assert staging not in d.parents and store.parent == tmp_path
    saved_text = (d / cp.SAVED_FILE).read_text()  # what the capture was saved against, stripped like the signature
    for ident in IDS:
        assert ident not in saved_text, ident
    assert REQUEST not in saved_text
    sig_text = (d / "signature.json").read_text()
    for ident in IDS:
        assert ident not in sig_text, ident
    assert REQUEST not in sig_text and "rows" not in json.loads(sig_text)["input"].get("lengths", {})
    sig = json.loads(sig_text)
    assert sig["input"]["kind"] == "value" and cp.ID_MARK in json.dumps(sig["input"]["tree"])
    assert sig["constraints"]["holds"]["stated"] and not sig["constraints"]["applies"]["stated"]
    rec = json.loads((d / "record.json").read_text())
    assert rec["episodes"] == ["e1"] and rec["level"] == tr.UNCHECKED  # a capture starts unchecked (I.2.3)
    assert rec["status"] == tr.ON_TRIAL and rec["track"]["verdicts"]["accepted"] == 1
    events = [json.loads(x) for x in (store / cp.EVENTS_FILE).read_text().splitlines()]
    assert [e["capture_id"] for e in events] == [cid]
    assert cp.episode_row(store, "e1")["capture_id"] == cid
    # captured once: a second call for the same settled episode does nothing
    assert cp.capture_settled(tmp_path, ["e1"], episode=lambda e: _ep(e), signals=lambda e: []) == []


def test_a_key_shaped_procedure_is_refused_before_anything_runs(tmp_path):
    planted = PROC + '\nTOKEN = "sk-or-v1-' + "a" * 64 + '"\n'
    staging = _staging(tmp_path, "e1", {"procedure.py": planted})
    runner, calls = _box(VALUE, [6, 9])
    row = cp.capture(staging, _ep("e1"), cp.store_dir(tmp_path), runner=runner)
    assert row["safety"] == "key-shaped string" and row["capture_id"] is None and calls == []
    assert not any(cp._ID.fullmatch(n) for n in os.listdir(cp.store_dir(tmp_path)))


def test_an_attempt_that_handed_nothing_back_is_not_captured(tmp_path):
    ep = _ep("e1")
    ep.actions = []
    staging = _staging(tmp_path, "e1", {"procedure.py": PROC})
    runner, calls = _box(VALUE, [6, 9])
    row = cp.capture(staging, ep, cp.store_dir(tmp_path), runner=runner)
    assert row["safety"].startswith("not a completed attempt") and calls == []


# --- negative knowledge -------------------------------------------------------------------------------------------


def test_a_failure_is_captured_as_negative_knowledge(tmp_path):
    staging = _staging(tmp_path, "e1", {"procedure.py": PROC, "broke.md": BROKE})
    store = cp.store_dir(tmp_path)
    runner, _ = _box(VALUE, [6, 10])  # it gives another result than the one handed back
    row = cp.capture(staging, _ep("e1"), store, runner=runner)
    assert (row["reproduces"], row["outcome"]) == ("differs", "failed")
    doc = cp.load(store)[row["capture_id"]]
    assert doc["record"]["outcome"] == "failed" and doc["record"]["level"] == tr.UNCHECKED
    assert "invalid literal" in doc["broke"]
    entry = pool.from_capture(row["capture_id"], doc)
    assert entry.kind == "failure" and "broke.md" in entry.files
    # an attempt that reproduced but whose episode carried a rejected verdict is a failure too
    staging2 = _staging(tmp_path, "e2", {"procedure.py": PROC.replace("Total", "Sum")})
    runner, _ = _box(VALUE, [6, 9])
    row2 = cp.capture(staging2, _ep("e2"), store, signals=[_checker("e2", "fail")], checker_visible=True, runner=runner)
    assert (row2["reproduces"], row2["outcome"]) == ("reproduces", "failed")
    # a staged broke.md with no procedure enters the pool as a failure entry of its own
    view = type("V", (), {"files": {"broke.md": BROKE.encode()}, "source": "fork"})()
    [failure] = [e for e in pool.from_staging("e3", view) if e.kind == "failure"]
    assert failure.files == {"broke.md": BROKE}
    nothing = type("V", (), {"files": {"broke.md": b"none\n"}, "source": "fork"})()
    assert pool.from_staging("e4", nothing) == []


# --- one record per code, outcomes per signal regime ---------------------------------------------------------------


def test_attempts_with_identical_code_share_one_capture_and_their_outcomes_join_one_record(tmp_path):
    store = cp.store_dir(tmp_path)
    for i in range(3):
        eid = f"e{i}"
        staging = _staging(tmp_path, eid, {"procedure.py": PROC})
        runner, _ = _box(VALUE, [6, 9])
        row = cp.capture(
            staging, _ep(eid), store, signals=[_checker(eid, "pass")], checker_visible=True, runner=runner
        )
        assert row["capture_id"] == cp.capture_id(PROC)
    [(cid, doc)] = [(k, v) for k, v in cp.load(store).items()]
    rec = doc["record"]
    assert rec["episodes"] == ["e0", "e1", "e2"] and len(rec["uses"]) == 3
    # three accepted answers in the dense regime: adopted, and confirmed by outcomes
    assert (rec["status"], rec["status_basis"], rec["level"]) == (
        tr.ADOPTED,
        tr.BASIS_VERDICTS,
        tr.CONFIRMED_BY_OUTCOMES,
    )
    # two rejected later uses retire it and take the confirmation away
    for eid in ("u1", "u2"):
        cp.record_use(store, cid, tr.Use(eid, "dense", answered=True, verdict="rejected"))
    rec = cp.load(store)[cid]["record"]
    assert (rec["status"], rec["status_rule"], rec["level"]) == (tr.RETIRED, "rejected", tr.UNCHECKED)


def test_without_verdicts_a_capture_is_adopted_on_implicit_signals_only(tmp_path):
    store = cp.store_dir(tmp_path)
    for i in range(3):
        staging = _staging(tmp_path, f"n{i}", {"procedure.py": PROC})
        runner, _ = _box(VALUE, [6, 9])
        cp.capture(staging, _ep(f"n{i}", regime="none"), store, runner=runner)
    rec = cp.load(store)[cp.capture_id(PROC)]["record"]
    assert (rec["status_text"], rec["level"]) == ("adopted (implicit)", tr.UNCHECKED)
    for text in (rec["record_line"], rec["status_reason"] or "", rec["status_text"]):
        assert not re.search(r"\b(correct|right|verified)\b", text, re.I), text
    assert rec["record_line"] == "its record has no verdicts; it ran without errors in 3 episodes"


def test_a_repeat_is_recognised_by_running_stored_captures_on_the_new_request(tmp_path):
    store = cp.store_dir(tmp_path)
    first = _staging(tmp_path, "e1", {"procedure.py": PROC})
    runner, _ = _box(VALUE, [6, 9])
    a = cp.capture(first, _ep("e1"), store, runner=runner)
    # the same work with other identifiers and values: same structure, so the stored capture is run on it
    other = REQUEST.replace("ab12cd34ef", "cd56ef78ab").replace("lunar-otter", "quiet-heron")
    variant = PROC.replace("Each row's total", "Every row's total")
    second = _staging(tmp_path, "e2", {"procedure.py": variant})
    value2 = {"cd56ef78ab": [1, 2, 3], "run_000123": [4, 5]}
    runner, calls = _box(value2, [6, 9])
    b = cp.capture(second, _ep("e2", request=other), store, runner=runner)
    assert len(calls) == 2  # its own run, then the stored capture's run on this request
    assert b["same_as"] == [a["capture_id"]]
    rec = cp.load(store)[a["capture_id"]]["record"]
    assert rec["recurrence"] == [{"episode": "e2", "gives": "same"}] and len(rec["uses"]) == 2
    assert cp.batch_row(store, "e2")["same_as"] == [a["capture_id"]]


# --- what reads the store ---------------------------------------------------------------------------------------


def test_the_actor_snapshot_and_the_writer_inputs_carry_the_capture_files(tmp_path):
    store = cp.store_dir(tmp_path)
    staging = _staging(tmp_path, "e1", {"procedure.py": PROC, "broke.md": BROKE})
    runner, _ = _box(VALUE, [6, 9])
    cid = cp.capture(staging, _ep("e1"), store, runner=runner)["capture_id"]
    snap = cp.export_files(store)
    assert sorted(snap) == sorted(f"{cp.EXPORT_DIR}/{cid}/{n}" for n in (*cp.CAPTURE_FILES, cp.SAVED_FILE))
    # what memory.match hands its matcher besides the signature: the docstring, the summary sections, what broke
    saved = json.loads(snap[f"{cp.EXPORT_DIR}/{cid}/{cp.SAVED_FILE}"])
    assert set(saved) == {"docstring", "asked", "approach", "broke"} and isinstance(saved["broke"], list)
    assert not any(p.endswith("INDEX.md") for p in snap)
    dest = tmp_path / "inputs" / "captures"
    assert cp.export_for_writer(store, [cid], dest) == [cid]
    assert (dest / cid / "procedure.py").read_text() == PROC


def test_the_pool_takes_the_store_and_refreshes_its_record(tmp_path):
    from unify.memory.gitio import Repo

    store = cp.store_dir(tmp_path)
    staging = _staging(tmp_path, "e1", {"procedure.py": PROC})
    runner, _ = _box(VALUE, [6, 9])
    cid = cp.capture(staging, _ep("e1"), store, runner=runner)["capture_id"]
    before = {p: p.read_bytes() for p in store.rglob("*") if p.is_file()}
    repo = Repo.init_bare(tmp_path / "mem.git")
    p = pool.Pool(repo).load()
    assert p.enter_captures(store, "p1") == {"entries": 1, "added": 1}
    p.save("p1")
    cp.record_use(store, cid, tr.Use("u1", "none"))
    p = pool.Pool(repo).load()
    assert p.enter_captures(store, "p2")["added"] == 0
    [entry] = p.open_entries()
    assert entry.kind == "capture" and entry.data["capture_id"] == cid
    assert json.loads(entry.files["record.json"])["track"]["implicit"]["ran_ok"] == 2
    # the pool only reads the store
    after = {q: q.read_bytes() for q in store.rglob("*") if q.is_file()}
    assert set(before) == set(after)


def test_the_writer_reads_the_capture_store_and_no_other_agent_edits_the_library(tmp_path, monkeypatch):
    from tests.memory.test_pool import EMPTY, _Stores, _finish_with
    from tests.memory.test_writer_pass import _call

    state = tmp_path / "state"
    store = cp.store_dir(state)
    staging = _staging(state, "e2", {"procedure.py": PROC, "broke.md": BROKE})
    runner, _ = _box(VALUE, [6, 9])
    cid = cp.capture(staging, _ep("e2"), store, runner=runner)["capture_id"]
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
        fork=True,
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
    from unify.common import in_code_switch

    for on in (False, True):
        monkeypatch.setattr(in_code_switch, "tools_in_code", lambda on=on: on)
        text = fork_mod.fork_instruction()
        assert fork_mod.CAPTURE_BULLETS in text
        assert "`read_input(request)`" in text and "`broke.md`" in text
        # the procedure reads the values the request carries from its data, not from its layout
        assert "`request.data`" in text and "not from line positions or the prose around them" in text


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


@needs_bwrap
def test_a_real_confined_capture_reproduces_and_marks_non_determinism(tmp_path):
    staging = _staging(tmp_path, "e1", {"procedure.py": PROC})
    store = cp.store_dir(tmp_path)
    row = cp.capture(staging, _ep("e1"), store, work=tmp_path / "w")
    assert (row["reproduces"], row["deterministic"], row["holds"]) == ("reproduces", True, True), row
    # a procedure that ignores its input reproduces its own case and differs on another request of the same shape
    constant = PROC.replace('return [sum(value[k]) for k in sorted(value)]', "return [6, 9]")
    staging2 = _staging(tmp_path, "e2", {"procedure.py": constant})
    assert cp.capture(staging2, _ep("e2"), store, work=tmp_path / "w2")["reproduces"] == "reproduces"
    other = REQUEST.replace("[4, 5]", "[4, 6]")
    run = cp.run_box(constant, other, work=tmp_path / "w3")
    assert cp.compare(run, True, [6, 10]) == "differs"
    noisy = PROC.replace(
        'return [sum(value[k]) for k in sorted(value)]',
        "import random\n    return [random.random()]",
    )
    staging3 = _staging(tmp_path, "e3", {"procedure.py": noisy})
    assert cp.capture(staging3, _ep("e3"), store, work=tmp_path / "w4")["deterministic"] is False
