"""Closed pool entries stay in the pool and come back when they are relevant: a closed entry is linked to an entry,
a staged draft or a batch episode whose code shares a normalised code unit with its code, or that cites one of its
episodes; the batch map, ``pool show`` and /inputs/pool/related_closed.json show the links, and ``pool`` lists and
filters closed entries. Nothing reopens an entry on its own."""

import json

from unify.memory import batch_map as bm
from unify.memory import library_prompts as lp
from unify.memory import pool
from unify.memory import writer_tools as wt
from unify.memory.episodes import Cell
from tests.memory.test_pool import EMPTY, _finish_with, _open, _Stores
from tests.memory.test_writer_library_tools import _episode
from tests.memory.test_writer_pass import _call

FLOOD = (
    "def flood(grid, start, colour=3):\n"
    "    seen = set()\n"
    "    todo = [start]\n"
    "    while todo:\n"
    "        r, c = todo.pop()\n"
    "        if (r, c) in seen or grid[r][c] != 0:\n"
    "            continue\n"
    "        seen.add((r, c))\n"
    "        grid[r][c] = colour\n"
    "        todo.extend([(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)])\n"
    "    return grid\n"
)
# the same code with other names and constants
FILL = (
    "def fill_region(board, origin, value=7):\n"
    "    visited = set()\n"
    "    stack = [origin]\n"
    "    while stack:\n"
    "        y, x = stack.pop()\n"
    "        if (y, x) in visited or board[y][x] != 5:\n"
    "            continue\n"
    "        visited.add((y, x))\n"
    "        board[y][x] = value\n"
    "        stack.extend([(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)])\n"
    "    return board\n"
)
PARSE = (
    "def parse_header(text):\n"
    "    lines = text.splitlines()\n"
    "    pairs = [ln.split(':', 1) for ln in lines if ':' in ln]\n"
    "    return {k.strip(): v.strip() for k, v in pairs}\n"
)
REASON = "too specific to one grid"
LOAD = {e: _episode(e) for e in ("e2", "e3", "e4")}


def _cid(code: str) -> str:
    return pool.entry_id(
        "candidate", json.dumps({"candidate.py": code}, sort_keys=True)
    )


def _candidate(eid: str, code: str) -> pool.Entry:
    e = pool.Entry(
        id=_cid(code),
        kind="candidate",
        title="",
        data={"name": "f"},
        files={"candidate.py": code},
    )
    e.add_provenance([eid], [], ["fork"])
    return e


def _stage_candidate(root, eid, name, code):
    d = root / eid
    (d / "candidates").mkdir(parents=True)
    (d / "candidates" / f"{name}.py").write_text(code)
    (d / "fork.json").write_text(json.dumps({"status": "ok"}))


# --- resemblance --------------------------------------------------------------------------------------------------


def test_code_units_match_across_names_and_constants_and_not_otherwise():
    assert pool.code_units(FLOOD) & pool.code_units(FILL)
    assert not pool.code_units(FLOOD) & pool.code_units(PARSE)
    assert pool.code_units("def (:") == frozenset()


def test_a_closed_candidate_is_linked_by_code_with_its_reason_pass_and_evidence():
    p = pool.Pool(None)
    old = p.upsert(_candidate("e1", FLOOD), "p1")
    p.set_status(old.id, "retired", {"reason": REASON}, "p1")
    new = p.upsert(_candidate("e5", FILL), "p2")
    other = p.upsert(_candidate("e6", PARSE), "p2")
    links = p.related_to_entry(new)
    assert [x["id"] for x in links] == [old.id]
    link = links[0]
    assert link["status"] == "retired" and link["reason"] == REASON
    assert link["closed_in"] == "p1" and link["evidence_episodes"] == ["e1"]
    assert link["shared_code"]["candidate.py"] > 0 and link["shared_episodes"] == []
    assert p.related_to_entry(other) == []
    # nothing changed status: the link is for the writer to act on
    assert p.entries[old.id].status == "retired" and p.entries[new.id].status == "open"


def test_a_lesson_without_code_is_linked_only_through_shared_episodes():
    p = pool.Pool(None)
    lesson = pool.Entry(
        id=pool.entry_id("lesson", "n"),
        kind="lesson",
        title="",
        files={"notes.md": "# flood fill from the start cell\n"},
    )
    lesson.add_provenance(["e1"], [], ["fork"])
    p.upsert(lesson, "p1")
    p.set_status(lesson.id, "merged", {"item": "memory.grid:flood"}, "p1")
    same_episode = p.upsert(_candidate("e1", PARSE), "p2")
    other_episode = p.upsert(_candidate("e9", FLOOD), "p2")
    links = p.related_to_entry(same_episode)
    assert [x["id"] for x in links] == [lesson.id]
    assert links[0]["reason"] == "merged into memory.grid:flood"
    assert links[0]["shared_episodes"] == ["e1"] and links[0]["shared_code"] == {}
    assert p.related_to_entry(other_episode) == []


def test_a_closed_draft_is_linked_through_the_code_its_patch_adds():
    patch = (
        "diff --git a/memory/grid.py b/memory/grid.py\n"
        "--- a/memory/grid.py\n"
        "+++ b/memory/grid.py\n"
        "@@ -0,0 +1,11 @@\n" + "".join("+" + ln + "\n" for ln in FLOOD.splitlines())
    )
    p = pool.Pool(None)
    d = p.upsert(pool.draft("p1", patch, "refused", ["memory.grid:flood"]), "p1")
    p.set_status(d.id, "superseded", {"reason": "rewritten in p2"}, "p2")
    links = p.related_closed({"cells": pool.code_units(FILL)})
    assert [x["id"] for x in links] == [d.id]
    assert links[0]["reason"] == "rewritten in p2" and links[0]["closed_in"] == "p2"
    assert links[0]["shared_code"]["cells"] > 0


def test_batch_rows_list_related_closed_entries_a_page_at_a_time(monkeypatch):
    p = pool.Pool(None)
    old = p.upsert(_candidate("e1", FLOOD), "p1")
    p.set_status(old.id, "retired", {"reason": REASON}, "p1")
    eps = {
        "e3": _episode("e3", [Cell(0, FILL, "")]),
        "e4": _episode("e4", [Cell(0, PARSE, "")]),
        "e1": _episode("e1"),
    }
    rows = [{"episode_id": e} for e in ("e3", "e4", "e1")]
    got = bm.add_related_closed(rows, p, eps.__getitem__)
    by = {r["episode_id"]: r["related_closed"] for r in rows}
    assert by["e3"]["count"] == 1 and by["e3"]["entries"][0]["id"] == old.id
    assert by["e3"]["entries"][0]["shared_code"]["cells"] > 0
    assert by["e4"] == {"count": 0, "entries": []}
    # an episode the closed entry cites is linked through it
    assert by["e1"]["entries"][0]["shared_episodes"] == ["e1"]
    assert set(got) == {"e3", "e1"} and p.episode_links == got
    # bounded by the stated page only: the count names every link
    monkeypatch.setattr(pool, "RELATED_PAGE", 0)
    rows = [{"episode_id": "e3"}]
    bm.add_related_closed(rows, p, eps.__getitem__)
    assert rows[0]["related_closed"] == {"count": 1, "entries": []}


# --- the pool tool ------------------------------------------------------------------------------------------------


def _staged_pool(tmp_path):
    p = pool.Pool(None)
    old = p.upsert(_candidate("e1", FLOOD), "p1")
    p.set_status(old.id, "retired", {"reason": REASON}, "p1")
    new = p.upsert(_candidate("e5", FILL), "p2")
    other = p.upsert(_candidate("e6", PARSE), "p2")
    p.episode_links = {"e5": p.related_closed({"cells": pool.code_units(FILL)}, ["e5"])}
    assert p.stage(tmp_path / "pool") == 2
    (tmp_path / "box").mkdir()
    return old.id, new.id, other.id, tmp_path / "pool", tmp_path / "box"


def test_closed_entries_are_staged_listed_filtered_and_shown(tmp_path):
    old, new, other, pdir, box = _staged_pool(tmp_path)
    assert (pdir / old / "candidate.py").read_text() == FLOOD
    index = (pdir / "INDEX.md").read_text()
    assert "## Other closed entries" in index and f"`{old}`" not in index
    related = json.loads((pdir / pool.RELATED_FILE).read_text())
    assert [x["id"] for x in related["entries"][new]] == [old]
    assert other not in related["entries"]
    assert [x["id"] for x in related["episodes"]["e5"]] == [old]

    got = wt.pool({"op": "list"}, pdir, box)
    assert got.text.startswith("2 open entries") and old not in got.text
    got = wt.pool({"op": "list", "status": "closed"}, pdir, box)
    assert got.text.startswith("1 closed entry")
    assert f"`{old}` candidate (retired: {REASON}; closed in p1)" in got.text
    got = wt.pool({"op": "list", "status": "any"}, pdir, box)
    assert got.text.startswith("3 entries")
    assert wt.pool({"op": "list", "status": "gone"}, pdir, box).form == "bad_status"
    got = wt.pool({"op": "filter", "related": new}, pdir, box)
    assert got.text.startswith("1 entry") and old in got.text
    got = wt.pool({"op": "filter", "related": "e5"}, pdir, box)
    assert old in got.text
    got = wt.pool({"op": "filter", "related": other}, pdir, box)
    assert got.text.startswith("0 entries")

    shown = wt.pool({"op": "show", "id": new}, pdir, box).text
    assert "Related closed entries: 1" in shown
    assert f"`{old}` candidate (retired: {REASON}; closed in p1)" in shown
    assert "evidence: e1" in shown and "code unit(s) with candidate.py" in shown
    assert (
        "Related closed entries: 0"
        in wt.pool({"op": "show", "id": other}, pdir, box).text
    )
    assert '"status": "retired"' in wt.pool({"op": "show", "id": old}, pdir, box).text


def test_the_writer_decides_reopening_and_nothing_else_applies_to_a_closed_entry(
    tmp_path,
):
    old, new, _, pdir, box = _staged_pool(tmp_path)
    wt.write_manifest(box, {"items": [], "summary": "s"})
    bad = wt.pool(
        {"op": "decide", "id": old, "decision": "retired", "reason": "r"}, pdir, box
    )
    assert bad.form == "closed_entry" and "retired" in bad.text
    bad = wt.pool(
        {"op": "decide", "id": new, "decision": "reopened", "reason": "r"}, pdir, box
    )
    assert bad.form == "open_entry"
    bad = wt.pool({"op": "decide", "id": old, "decision": "reopened"}, pdir, box)
    assert bad.form == "bad_decision" and "needs a reason" in bad.text
    got = wt.pool(
        {"op": "decide", "id": old, "decision": "reopened", "reason": "e5 needs it"},
        pdir,
        box,
    )
    assert got.status == "ok", got.text
    man = json.loads((box / ".pass/manifest.json").read_text())
    assert man["pool"] == {"reopened": {old: "e5 needs it"}}
    p = pool.Pool(None)
    e = p.upsert(_candidate("e1", FLOOD), "p1")
    p.set_status(e.id, "retired", {"reason": REASON}, "p1")
    assert pool.block_problems(man, p) == []
    wt.pool({"op": "decide", "id": old, "decision": "open"}, pdir, box)
    assert "pool" not in json.loads((box / ".pass/manifest.json").read_text())


# --- across passes ------------------------------------------------------------------------------------------------


def test_a_draft_retired_in_one_pass_is_named_when_similar_code_is_staged_later(
    tmp_path,
    monkeypatch,
):
    st = _Stores(tmp_path)
    one = tmp_path / "staging-1"
    _stage_candidate(one, "e2", "flood", FLOOD)
    old = _cid(FLOOD)
    _finish_with(monkeypatch, {**EMPTY, "pool": {"retired": {old: REASON}}})
    out, _ = st.run([_open("e2")], "p1", load=LOAD.__getitem__, staging_root=str(one))
    assert out.finished and old in out.pool["changed"], out.reasons
    assert st.pool().entries[old].status == "retired"

    two = tmp_path / "staging-2"
    _stage_candidate(two, "e3", "fill_region", FILL)
    _stage_candidate(two, "e4", "parse_header", PARSE)
    _finish_with(monkeypatch, EMPTY)
    near, far = _cid(FILL), _cid(PARSE)
    _, model = st.run(
        [
            _call("m", "read", {"path": "/inputs/batch_map.json"}),
            _call("s", "pool", {"op": "show", "id": near}),
            _call("t", "pool", {"op": "show", "id": far}),
            _call("c", "pool", {"op": "list", "status": "closed"}),
            _open("e3"),
            _open("e4"),
        ],
        "p2",
        eids=("e3", "e4"),
        load=LOAD.__getitem__,
        staging_root=str(two),
    )
    bmap = json.loads(model.outputs["m"].split("\n[…")[0])
    rows = {r["episode_id"]: r for r in bmap["episodes"]}
    link = rows["e3"]["related_closed"]["entries"][0]
    assert rows["e3"]["related_closed"]["count"] == 1
    assert (
        link["id"] == old and link["status"] == "retired" and link["reason"] == REASON
    )
    assert link["closed_in"] == "p1" and link["evidence_episodes"] == ["e2"]
    assert link["shared_code"]["staging:e3/candidates/fill_region.py"] > 0
    assert rows["e4"]["related_closed"] == {"count": 0, "entries": []}
    shown = model.outputs["s"]
    assert (
        "Related closed entries: 1" in shown
        and f"`{old}` candidate (retired: {REASON}" in shown
    )
    assert (
        "Related closed entries: 0" in model.outputs["t"]
        and old not in model.outputs["t"]
    )
    assert old in model.outputs["c"]
    # the link reopened nothing
    assert st.pool().entries[old].status == "retired"
    # every row key is in the brief's table
    assert set(rows["e3"]) == {k for k, _ in lp.row_keys(staging=True, pool=True)}


def test_the_brief_states_closed_entries_and_the_row_key():
    text = lp.write_brief_now(staging=True, pool=True)
    assert (
        "`related_closed` (" in text
        and "Closed entries (merged, superseded or retired) are kept" in text
    )
    assert '`"reopened": {"<id>": "<reason>"}` opens a closed entry again' in text
    assert "`related_closed`" not in lp.write_brief_now(staging=True, pool=False)
    assert f"the first {pool.RELATED_PAGE} as" in text


def test_an_unreadable_links_file_is_said_never_shown_as_no_links(tmp_path):
    _, new, _, pdir, box = _staged_pool(tmp_path)
    (pdir / pool.RELATED_FILE).write_text("{not json")
    shown = wt.pool({"op": "show", "id": new}, pdir, box).text
    assert "related_closed.json could not be read" in shown
    got = wt.pool({"op": "filter", "related": new}, pdir, box)
    assert "could not be read" in got.text
    (pdir / pool.RELATED_FILE).unlink()  # a pool staged without links: none to show
    assert (
        "Related closed entries: 0"
        in wt.pool({"op": "show", "id": new}, pdir, box).text
    )
