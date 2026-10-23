"""Captures the writer closed, withdrawn from what the actor is offered (``UNIFY_MEMORY_LIBRARY_WITHDRAW_CLOSED_CAPTURES``).

On, a captured procedure whose pool entry the writer closed as superseded or retired leaves the actor's snapshot, so
nothing offers it (``memory.items``, ``memory.get``, the structural match, the dispatch file, the catalogue, the search
index, and so the librarian's answers); the item that superseded it is offered in its place: ``memory.get`` of the old
id gives it, the structural match finds it where the withdrawn capture would have matched, and its catalogue card and
search text carry what the withdrawn capture was asked. A capture merged into an item stays offered. The rule reads
only the writer's decision in the pool: no task, position or word. No model is called.
"""

from __future__ import annotations

import inspect
import json

from tests.memory.test_capture_examples import GRID, ROWS, _cell, _copy, _snapshot, _store
from unify.memory import captures, library_search, memory_helper, pool
from unify.memory import library_export as lx
from unify.memory.gitio import Repo
from unify.memory.integration.request import RequestRun
from unify.memory.layout import discover


# --- the pool's closed captures ---------------------------------------------------------------------------------------


def _doc(source: str, outcome: str = "none") -> dict:
    return {"record": {"outcome": outcome, "episodes": ["ep-1"]}, "signature": {}, "procedure": source, "broke": None}


def test_the_pool_names_each_superseded_or_retired_capture_and_what_replaces_it(tmp_path):
    repo = Repo.init_bare(tmp_path / "mem.git")
    assert pool.withdrawn_captures(repo) == {}  # no pool yet
    p = pool.Pool(repo).load()
    src = {k: f"def run(v):\n    return v + {i}\n" for i, k in enumerate("ABCDEFGHIJK")}
    cid = {k: captures.capture_id(s) for k, s in src.items()}
    eid = {}
    for k, s in src.items():
        e = pool.from_capture(cid[k], _doc(s, "failed" if k == "J" else "none"))
        eid[k] = p.upsert(e, "p1").id
    cand = pool.Entry(id=pool.entry_id("candidate", "c"), kind="candidate", title="c", files={"candidate.py": "x"})
    open_cand = pool.Entry(id=pool.entry_id("candidate", "o"), kind="candidate", title="o", files={"candidate.py": "y"})
    p.upsert(cand, "p1")
    p.upsert(open_cand, "p1")
    p.set_status(eid["A"], "superseded", {"by": "memory.grid.cells:fill"}, "p2")
    p.set_status(eid["B"], "superseded", {"by": eid["C"]}, "p2")  # C stays open: the capture itself replaces B
    p.set_status(eid["D"], "retired", {"reason": "a hard-coded answer"}, "p2")
    p.set_status(eid["E"], "merged", {"item": "memory.grid.cells:rows"}, "p2")  # merged: stays offered
    p.set_status(cand.id, "merged", {"item": "memory.text.dates:parse_date"}, "p2")
    p.set_status(eid["F"], "superseded", {"by": cand.id}, "p2")  # through an entry merged into an item
    p.set_status(eid["G"], "superseded", {"item": "memory.grid.cells:rows", "via": "candidate"}, "p2")
    p.set_status(eid["H"], "superseded", {"by": eid["I"]}, "p2")
    p.set_status(eid["I"], "superseded", {"by": eid["H"]}, "p2")  # a loop: nothing is offered in their place
    p.set_status(eid["J"], "retired", {"reason": "broke"}, "p2")  # a failed attempt's entry
    p.set_status(eid["K"], "superseded", {"by": open_cand.id}, "p2")  # an open candidate is never offered
    p.save("p2")
    got = pool.withdrawn_captures(repo)
    assert {k: got[cid[k]]["by"] for k in "ABDFGHIJK"} == {
        "A": "memory.grid.cells:fill",
        "B": f"capture:{cid['C']}",
        "D": None,
        "F": "memory.text.dates:parse_date",
        "G": "memory.grid.cells:rows",
        "H": None,
        "I": None,
        "J": None,
        "K": None,
    }
    assert cid["C"] not in got and cid["E"] not in got  # open and merged captures stay offered
    assert got[cid["D"]] == {"status": "retired", "by": None, "reason": "a hard-coded answer"}
    assert got[cid["A"]]["status"] == "superseded" and got[cid["A"]]["reason"] == "superseded by memory.grid.cells:fill"


# --- the actor's snapshot ---------------------------------------------------------------------------------------------


def _withdrawn(ids: dict[str, str]) -> dict[str, dict]:
    return {
        ids["grid"]: {"status": "superseded", "by": f"capture:{ids['table']}", "reason": "superseded by an entry"},
        ids["retired"]: {"status": "retired", "by": None, "reason": "one hand's action"},
    }


def test_off_nothing_is_withdrawn(tmp_path):
    store, ids = _store(tmp_path)
    snap = captures.export_files(store)
    assert snap == captures.export_files(store, withdrawn=None)
    assert not any(p.endswith(captures.WITHDRAWN_FILE) for p in snap)
    assert {p.split("/")[2] for p in snap} == set(ids.values())


def test_withdrawn_captures_leave_the_snapshot_and_name_their_replacement(tmp_path):
    store, ids = _store(tmp_path)
    snap = captures.export_files(store, withdrawn=_withdrawn(ids))
    assert {p.split("/")[2] for p in snap if p.count("/") == 3} == {ids["table"], ids["fixed"]}
    doc = json.loads(snap[f"{captures.EXPORT_DIR}/{captures.WITHDRAWN_FILE}"])
    grid = doc["captures"][ids["grid"]]
    assert grid["by"] == f"capture:{ids['table']}" and grid["status"] == "superseded"
    assert grid["asked"] == "Turn grid rows into columns."
    assert grid["signature"]["input"] == json.loads(json.dumps(memory_helper.value_shape(GRID)))
    assert doc["captures"][ids["retired"]]["by"] is None
    # the store itself is only read
    assert sorted(p.name for p in store.iterdir() if p.is_dir()) == sorted(ids.values())


# --- the librarian's files ---------------------------------------------------------------------------------------------


def test_the_dispatch_file_offers_no_withdrawn_capture(tmp_path):
    ids = _store(tmp_path)[1]  # capture ids are digests of their code: the same in every store
    tree, snap, _ = _snapshot(tmp_path, withdrawn=_withdrawn(ids))
    rows = lx.dispatch_data(discover(tree), captures=snap)["procedures"]
    got = sorted(r["id"] for r in rows if r["id"].startswith("capture:"))
    assert got == sorted(f"capture:{ids[k]}" for k in ("table", "fixed"))


def test_the_catalogue_and_the_search_index_show_the_replacement_in_the_withdrawn_captures_place(tmp_path):
    ids = _store(tmp_path)[1]
    tree, snap, _ = _snapshot(tmp_path, withdrawn=_withdrawn(ids))
    text = lx.catalogue_text(tree, captures=snap)
    assert f"capture:{ids['grid']}" not in text.replace(f"In place of capture:{ids['grid']}", "")
    assert f"- In place of capture:{ids['grid']} (superseded): Asked: Turn grid rows into columns.\n" in text
    assert f"capture:{ids['retired']}" not in text
    rows = {r["id"]: r for r in library_search.rows(tree, captures=snap)}
    assert "Turn grid rows into columns." in rows[f"capture:{ids['table']}"]["text"]
    assert f"capture:{ids['grid']}" not in rows and f"capture:{ids['retired']}" not in rows


# --- the package, as a cell runs it -----------------------------------------------------------------------------------


def _withdrawn_copy(tmp_path):
    ids = _store(tmp_path)[1]
    root, _ = _copy(tmp_path, withdrawn=_withdrawn(ids))
    return root, ids


def test_the_package_lists_no_withdrawn_capture_and_gives_its_replacement_by_its_old_id(tmp_path):
    root, ids = _withdrawn_copy(tmp_path)
    grid, retired, table = (f"capture:{ids[k]}" for k in ("grid", "retired", "table"))
    got = _cell(
        root,
        f"listed = [i.id for i in memory.items()]\n"
        f"moved = memory.get({grid!r})\n"
        f"print(json.dumps({{'listed': listed, 'moved': moved.id, 'card': str(moved), "
        f"'retired': err(memory.get, {retired!r}), 'run': memory.run({grid!r}, 'x').detail}}))\n",
    )
    assert grid not in got["listed"] and retired not in got["listed"] and table in got["listed"]
    assert got["moved"] == table
    assert got["card"].startswith(f"{grid} is no longer offered: the library's writer superseded it")
    assert f"{table} is offered in its place." in got["card"]
    assert got["retired"][0] == "NotFound" and "is no longer offered: the library's writer retired it" in got["retired"][1]
    assert "no longer offered" in got["run"] and f"memory.get({table!r})" in got["run"]


def test_the_replacement_matches_where_the_withdrawn_capture_would_have(tmp_path):
    root, ids = _withdrawn_copy(tmp_path)
    table = f"capture:{ids['table']}"
    code = f"print(json.dumps([bool(memory.get({table!r}).fits({GRID!r})), bool(memory.get({table!r}).fits({ROWS!r}))]))\n"
    assert _cell(root, code) == [True, True]  # the grid through the withdrawn capture's signature, the table its own
    (root / captures.EXPORT_DIR / captures.WITHDRAWN_FILE).unlink()
    assert _cell(root, code) == [False, True]  # without the withdrawal, its own signature only


# --- the request ------------------------------------------------------------------------------------------------------


def test_the_request_always_withdraws_closed_captures_and_records_what_it_did():
    # the copy is written by _export_copy (at the request's open, and at each re-pin under UNIFY_MEMORY=in_episode);
    # rev 2 has no withdrawal switch: closed captures are always withdrawn
    opening = inspect.getsource(RequestRun._export_copy)
    assert "library_withdraw_closed_captures" not in opening
    assert "withdrawn = withdrawn_captures(Repo(paths.memory))" in opening
    assert "withdrawn=withdrawn)" in opening and '"offered_in_place": sorted(' in opening
    assert opening.index("withdrawn_captures(Repo(paths.memory))") < opening.index("export_actor_library_mode(")
