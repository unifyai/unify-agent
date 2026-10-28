"""One route per episode: an episode whose captured attempt can be landed reaches the writer as that capture alone.

Its staging is exported without the fork's copy of the procedure (the host's staging keeps it, and the pool reads it
there, so a staged ``broke.md`` stays with the capture), its batch-map row's ``staging.capture`` names the capture and
the one call that lands it, and the capture's pool entry closes by the route the writer took: merged when
``land_capture`` landed it, superseded when a candidate staged from its episode was adopted instead. A failed capture's
episode keeps its staged procedure. Landing through a whole writer pass, in real boxes, is
``tests/memory/test_capture_end_to_end.py::test_a_writer_pass_lands_the_capture_as_a_library_item``.
"""

from __future__ import annotations

import json

from tests.memory.test_captures import BROKE, PROC, _ep
from tests.memory.test_pool import EMPTY, ITEM, MAN, _finish_with, _open, _Stores
from tests.memory.test_pool_start import CID, LID, _fake_merge, _stage
from tests.memory.test_writer_pass import _call
from unify.memory import captures as cp
from unify.memory import pool, staging
from unify.memory.gate import GateResult
from unify.memory.gitio import Repo

#: e1's procedure: another shape than e2's (a capture of the same shape would fold into e2's), whose episode a signal
#: contradicts
FAILED_PROC = PROC.replace("Total", "Sum").replace(
    "    return [sum(value[k]) for k in sorted(value)]\n",
    "    names = sorted(value)\n    return [sum(value[k]) for k in names]\n",
)


def _world(tmp_path):
    """Two settled forks and their captures: e2 staged a candidate with its case, notes, a procedure and what broke
    (its capture can be landed); e1 staged a procedure whose episode a visible checker verdict rejected, so its
    capture is negative knowledge (rev 2: a capture is never run; its outcome comes from the signals only). Returns
    the staging root, the capture store and the two capture ids."""
    from unify.memory.signals import Signal

    state = tmp_path / "state"
    root = state / "memory-staging"
    store = cp.store_dir(state)
    d2 = _stage(root, "e2")
    (d2 / "procedure.py").write_text(PROC)
    (d2 / "broke.md").write_text(BROKE)
    landable = cp.capture_trace_step(d2, _ep("e2"), store)["capture_id"]
    d1 = root / "e1"
    d1.mkdir(parents=True)
    (d1 / "procedure.py").write_text(FAILED_PROC)
    (d1 / "fork.json").write_text(json.dumps({"episode": "e1", "status": "ok"}))
    rejected = Signal("s-e1", "e1", "checker", "fail", "2026-10-13T00:01:30Z", refers_to="e1/actions/0",
                      regime="dense", visible_to_actor=True)  # fmt: skip
    failed = cp.capture_trace_step(d1, _ep("e1"), store, signals=[rejected])["capture_id"]
    assert failed != landable and cp.load(store)[failed]["record"]["outcome"] == cp.CONTRARY
    assert not cp.negative(cp.load(store)[landable]["record"])
    return root, store, landable, failed


def _rows(text):
    return {r["episode_id"]: r for r in json.loads(text.split("\n[…")[0])["episodes"]}


def _adopting(st, monkeypatch, ref):
    """A writer whose one item adopts the staged draft *ref*, and a gate that lands it."""
    manifest = {**MAN, "items": [{**ITEM, "adopts": [{"ref": f"/inputs/staging/{ref}", "how": "edited"}]}]}
    _finish_with(monkeypatch, {**manifest, "summary": "s"})
    monkeypatch.setattr(st.gate, "merge", _fake_merge(st, ITEM["item"], [f"staging:{ref}"]))
    monkeypatch.setattr(st.gate, "check", lambda *a, **k: GateResult(True, {}, [], []))


# --- one route into the writer's inputs ---------------------------------------------------------------------------


def test_a_landable_capture_reaches_the_writer_once_and_a_failed_one_keeps_its_staged_procedure(
    tmp_path,
    monkeypatch,
):
    root, _, landable, failed = _world(tmp_path)
    host = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    st = _Stores(tmp_path)
    _finish_with(monkeypatch, EMPTY)
    out, model = st.run(
        [
            _call("m", "read", {"path": "/inputs/batch_map.json"}),
            _call("p2", "read", {"path": "/inputs/staging/e2/procedure.py"}),
            _call("c2", "read", {"path": "/inputs/staging/e2/candidates/fill.py"}),
            _call("k2", "read", {"path": f"/inputs/captures/{landable}/procedure.py"}),
            _call("p1", "read", {"path": "/inputs/staging/e1/procedure.py"}),
            _open("e1"),
            _open("e2"),
        ],
        "p1",
        eids=("e1", "e2"),
        staging_root=str(root),
    )
    # e2's procedure reaches the writer as its capture only; its drafts stay staged
    assert "does not exist" in model.outputs["p2"] and "def read_input" not in model.outputs["p2"]
    assert "def fill(" in model.outputs["c2"]
    assert "Total each named row" in model.outputs["k2"]
    # e1's capture failed, so its staged procedure is handed on as before
    assert "Sum each named row" in model.outputs["p1"]
    rows = _rows(model.outputs["m"])
    e2, e1 = rows["e2"]["staging"], rows["e1"]["staging"]
    assert "procedure.py" not in e2["files"]
    assert {"candidates/fill.py", "cases/fill.json", "notes.md", "broke.md"} <= set(e2["files"])
    assert [c["file"] for c in e2["candidates"]] == ["/inputs/staging/e2/candidates/fill.py"]
    assert rows["e2"]["capture"]["id"] == landable and rows["e1"]["capture"]["id"] == failed
    assert "procedure.py" in e1["files"] and e1["capture"] is None
    # e2's row names its capture and the one call that lands it, as the writer makes it (in code, through ``tools.``)
    assert e2["capture"] == {"id": landable, "land": f"tools.land_capture('{landable}', package, name)"}
    # the host's staging is untouched: the capture step and the pool read it there
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == host
    # what broke stays with the capture's entry: it is not a second, failure entry of its own
    entries = st.pool().entries
    assert "broke.md" in entries[pool.entry_id("capture", landable)].files
    assert pool.entry_id("failure", BROKE) not in entries
    assert [e.id for e in entries.values() if e.kind == "failure"] == [pool.entry_id("capture", failed)]
    # nothing was landed or adopted: the capture's entry stays open
    assert entries[pool.entry_id("capture", landable)].status == "open"
    assert out.pool["captures"]["e2"] == {"capture_id": landable, "landed": None, "superseded_by": None}


def test_the_brief_says_where_the_procedure_is_and_which_call_lands_it():
    from unify.memory import library_prompts as lp

    doc = dict(lp.row_keys())["staging"]
    assert "`capture` (`{id, land}` when the episode's capture can be landed" in doc
    assert "its procedure then under /inputs/captures/<id>/ only; else null" in doc
    text = lp.captures_input()
    assert "When an episode's capture can be landed, its procedure is here only, not in /inputs/staging/." in text
    assert "its row's `staging.capture.land` is that call." in text
    assert "`tools.land_capture(capture_id, package, name)`" in text
    # a capture is a library entry from the moment it is stored; nothing runs it
    assert "A capture is a library entry from the moment it is stored" in text and "The harness runs no capture" in text


# --- the capture's pool entry closes by the route taken -----------------------------------------------------------


def test_adopting_a_candidate_from_the_episode_supersedes_its_capture_once(tmp_path, monkeypatch):
    root, _, landable, _ = _world(tmp_path)
    cap = pool.entry_id("capture", landable)
    st = _Stores(tmp_path)
    _adopting(st, monkeypatch, "e2/candidates/fill.py")
    turns = [_open("e2"), _call("f", "finish", {"summary": "adopted fill"})]
    out, _ = st.run(turns, "p1", staging_root=str(root))
    assert out.finished and out.passed and out.items_merged == [ITEM["item"]], out.reasons
    e = st.pool().entries[cap]
    assert (e.status, e.status_detail) == ("superseded", {"item": ITEM["item"], "via": "candidate", "pass": "p1"})
    assert out.pool["changed"].count(cap) == 1
    assert out.pool["captures"]["e2"] == {"capture_id": landable, "landed": None, "superseded_by": ITEM["item"]}
    # the adopted candidate merged as before; the notes were not adopted
    assert st.pool().entries[CID].status == "merged" and st.pool().entries[LID].status == "open"
    # a later pass adopting the same candidate closes nothing again
    out2, _ = st.run(turns, "p2", staging_root=str(root))
    assert out2.passed, out2.reasons
    assert cap not in out2.pool["changed"] and st.pool().entries[cap].status_detail["pass"] == "p1"


def test_adopting_only_the_episodes_notes_leaves_its_capture_open(tmp_path, monkeypatch):
    root, _, landable, _ = _world(tmp_path)
    st = _Stores(tmp_path)
    _adopting(st, monkeypatch, "e2/notes.md")
    out, _ = st.run([_open("e2"), _call("f", "finish", {"summary": "s"})], "p1", staging_root=str(root))
    assert out.finished and out.passed, out.reasons
    assert st.pool().entries[LID].status == "merged"
    assert st.pool().entries[pool.entry_id("capture", landable)].status == "open"
    assert out.pool["captures"]["e2"] == {"capture_id": landable, "landed": None, "superseded_by": None}


def test_a_landing_merges_the_capture_entry_once_and_only_when_its_item_landed(tmp_path):
    root, store, landable, _ = _world(tmp_path)
    cap = pool.entry_id("capture", landable)
    p = pool.Pool(Repo.init_bare(tmp_path / "mem.git")).load()
    p.enter_staging({e: staging.read(root, e) for e in ("e1", "e2")}, "p1")
    p.enter_captures(store, "p1")
    p.episode_captures = {"e2": cap}  # the writer pass maps each batch episode whose capture can be landed
    item = "memory.readings.procedures:row_sums"
    landing = {"item": item, "reader": "memory.readings.readers:read_row_sums", "episode": "e2"}
    landings = {landable: {**landing, "pass": "p1"}}
    adopts = [{"item": ITEM["item"], "adopts": ["staging:e2/candidates/fill.py"]}]
    # the landed procedure did not pass the gate: the entry stays open
    assert pool.close_captures(p, "p1", landed={ITEM["item"]}, landings=landings) == []
    assert p.entries[cap].status == "open"
    # a landing an earlier pass made (its item refused then) never closes it, even when an item of that id lands now
    assert pool.close_captures(p, "p1", landed={item}, landings={landable: {**landing, "pass": "p0"}}) == []
    assert p.entries[cap].status == "open"
    # it landed: merged through land_capture, and the candidate adopted in the same pass does not close it again
    got = pool.close_captures(p, "p1", landed={item, ITEM["item"]}, landings=landings, provenance=adopts)
    assert got == [cap]
    assert (p.entries[cap].status, p.entries[cap].status_detail) == (
        "merged",
        {"item": item, "via": "land_capture", "pass": "p1"},
    )
    assert pool.close_captures(p, "p2", landed={item, ITEM["item"]}, landings=landings, provenance=adopts) == []
    assert p.entries[cap].status_detail["pass"] == "p1"
