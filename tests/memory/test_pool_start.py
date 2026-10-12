"""The pool at a pass's start, adoption and its provenance, the pool's titles and order, closed entries with new
evidence, the miner's old entries, a pass with staging off, and the counts at each hand-off from the fork's reply to
the writer's adoption."""

import json

import pytest

from unify.memory import chain, pool
from unify.memory.gate import GateResult
from unify.memory.gitio import Repo
from unify.memory.manifest import (
    ADOPT_FORMS,
    ManifestError,
    adopt_ref,
    parse_manifest,
)
from unify.memory.staging import StagingView
from tests.memory.test_gate import ITEM, MAN
from tests.memory.test_pool import EMPTY, INDEX, _finish_with, _open, _Stores
from tests.memory.test_writer_pass import _call, needs_bwrap
from tests.memory.library_example_world import (
    library_example_world,
)  # noqa: F401 (a fixture)

FILL = (
    "def fill(grid, colour=0):\n"
    '    """Fill every enclosed region with colour.\n\n    More text.\n    """\n'
    "    return grid\n"
)
NOTES = "# Regions touching the border stay open\n\nDetails.\n"
STAGED = "/inputs/staging/e2/candidates/fill.py"
CANDIDATE_TITLE = "fill(grid, colour=0): Fill every enclosed region with colour."


def _canon(obj):
    return json.dumps(obj, sort_keys=True, default=str)


CID = pool.entry_id(
    "candidate",
    _canon({"candidate.py": FILL, "cases.json": "[[1, 1]]"}),
)
LID = pool.entry_id("lesson", NOTES)


def _stage(root, eid="e2", blocks=True):
    d = root / eid
    (d / "candidates").mkdir(parents=True)
    (d / "cases").mkdir()
    (d / "candidates" / "fill.py").write_text(FILL)
    (d / "cases" / "fill.json").write_text("[[1, 1]]")
    (d / "notes.md").write_text(NOTES)
    (d / "fork.json").write_text(json.dumps({"status": "ok"}))
    if blocks:
        rows = [
            {
                "index": 0,
                "path": "candidates/fill.py",
                "kind": "candidate",
                "status": "parsed",
            },
            {"index": 1, "path": "cases/fill.json", "kind": "case", "status": "parsed"},
            {"index": 2, "path": "notes.md", "kind": "notes", "status": "parsed"},
        ]
        (d / "fork-blocks.json").write_text(
            json.dumps({"episode": eid, "blocks": rows}),
        )
    return d


def _view(eid="e2"):
    return StagingView(
        "ok",
        "fork",
        {
            "candidates/fill.py": FILL.encode(),
            "cases/fill.json": b"[[1, 1]]",
            "notes.md": NOTES.encode(),
        },
    )


# --- 1. staging enters the pool at the pass's start -------------------------------------------------------------


def test_a_staged_candidate_is_an_open_entry_during_its_own_pass(tmp_path, monkeypatch):
    root = tmp_path / "staging-root"
    _stage(root)
    st = _Stores(tmp_path)
    _finish_with(monkeypatch, EMPTY)
    out, model = st.run(
        [
            _call("r", "read", {"path": INDEX}),
            _call("m", "read", {"path": "/inputs/batch_map.json"}),
            _open("e2"),  # read, so the pass keeps no unread entry for it
        ],
        "p1",
        staging_root=str(root),
    )
    index = model.outputs["r"]
    cand = f"- `{CID}` candidate: {CANDIDATE_TITLE} (1 episode; opened in p1)"
    lesson = f"- `{LID}` lesson: Regions touching the border stay open (1 episode; opened in p1)"
    assert cand in index and lesson in index and index.index(cand) < index.index(lesson)
    row = json.loads(model.outputs["m"].split("\n[…")[0])["episodes"][0]
    assert {"id": CID, "kind": "candidate", "title": CANDIDATE_TITLE} in row[
        "pool_entries"
    ]
    # the start commit holds the entries; the end of the pass changed nothing more
    assert out.pool["at_start"]["entries"] == 2 and out.pool["at_start"]["added"] == 2
    assert out.pool["at_start"]["commit"] and out.pool["added"] == 0
    e = st.pool().entries[CID]
    assert (
        e.status == "open"
        and e.opened_in == "p1"
        and e.provenance["episodes"] == ["e2"]
    )
    assert out.chain == {
        "parsed": 1,
        "staged": 1,
        "pooled_at_start": 1,
        "batch_map_rows": 1,
        "read_by_writer": 0,
        "adopted": 0,
    }


def _fake_merge(st, item, adopts):
    """The gate's merge, reduced to its record: *item* lands with the gate's provenance row."""

    def merge(parent, candidate, manifest, pass_id, kind, channel, usd, **_):
        res = GateResult(
            True,
            {},
            [],
            [],
            merged=candidate,
            items_merged=[item],
            provenance=[{"item": item, "adopts": adopts}],
        )
        st.gate._record(parent, candidate, pass_id, kind, channel, usd, res)
        return res

    return merge


def test_an_adopted_staged_candidate_merges_in_the_same_pass(tmp_path, monkeypatch):
    root = tmp_path / "staging-root"
    _stage(root)
    st = _Stores(tmp_path)
    manifest = {**MAN, "items": [{**ITEM, "adopts": [STAGED]}], "summary": "s"}
    _finish_with(monkeypatch, manifest)
    monkeypatch.setattr(
        st.gate,
        "merge",
        _fake_merge(st, ITEM["item"], ["staging:e2/candidates/fill.py"]),
    )
    monkeypatch.setattr(st.gate, "check", lambda *a, **k: GateResult(True, {}, [], []))
    out, model = st.run(
        [
            _call("r", "read", {"path": STAGED}),
            _open("e2"),
            _call("f", "finish", {"summary": "adopted fill"}),
        ],
        "p1",
        staging_root=str(root),
    )
    assert (
        out.finished and out.passed and out.items_merged == [ITEM["item"]]
    ), out.reasons
    assert out.provenance == [
        {"item": ITEM["item"], "adopts": ["staging:e2/candidates/fill.py"]},
    ]
    e = st.pool().entries[CID]
    assert e.status == "merged" and e.status_detail == {
        "item": ITEM["item"],
        "via": "adopts",
        "pass": "p1",
    }
    assert CID in out.pool["changed"] and st.pool().entries[LID].status == "open"
    # every hop carried the one candidate
    assert out.chain == dict.fromkeys(chain.KEYS, 1)


def test_finish_refuses_an_adopt_ref_that_names_nothing_staged(tmp_path, monkeypatch):
    root = tmp_path / "staging-root"
    _stage(root)
    st = _Stores(tmp_path)
    bad = {**MAN, "items": [{**ITEM, "adopts": ["staging:e2/candidates/other.py"]}]}
    _finish_with(monkeypatch, bad)
    _, model = st.run(
        [
            _open("e2"),
            _call("f", "finish", {"summary": "s"}),
        ],
        "p1",
        staging_root=str(root),
        max_calls=4,
    )
    assert model.outputs["f"] == (
        f"not finished: G1: {ITEM['item']}: adopts 'staging:e2/candidates/other.py', "
        "which is not a file this pass staged"
    )


# --- 2. a pass with staging off sees no staging anywhere ---------------------------------------------------------


def test_with_staging_off_the_writer_sees_no_staging_anywhere(tmp_path, monkeypatch):
    _stage(
        tmp_path / "unused-root",
    )  # present on disk, but no reader is configured for this pass
    st = _Stores(tmp_path)
    _finish_with(monkeypatch, EMPTY)
    out, model = st.run(
        [
            _call("l", "read", {"path": "/inputs"}),
            _call("m", "read", {"path": "/inputs/batch_map.json"}),
            _call("r", "read", {"path": INDEX}),
            _open("e2"),
        ],
        "p1",
    )
    assert "staging" not in model.outputs["l"]
    rows = json.loads(model.outputs["m"].split("\n[…")[0])["episodes"]
    assert rows and all("staging" not in r for r in rows)
    assert (
        "candidate:" not in model.outputs["r"] and "staging" not in model.outputs["r"]
    )
    assert "/inputs/staging" not in model.seen[1]["content"]
    assert out.chain == dict.fromkeys(chain.KEYS, 0)
    assert not any(e.kind == "candidate" for e in st.pool().entries.values())


# --- 3. adopts in the manifest, provenance and closing ------------------------------------------------------------


def test_adopt_refs_parse_to_one_form_and_bad_ones_name_the_forms():
    assert (
        adopt_ref("/inputs/staging/e2/candidates/fill.py")
        == "staging:e2/candidates/fill.py"
    )
    assert adopt_ref("staging:e2/cases/fill.json") == "staging:e2/cases/fill.json"
    assert adopt_ref("/inputs/pool/" + CID + "/candidate.py") == "pool:" + CID
    assert adopt_ref("/inputs/helpers/c1/notes.md") == "helper:c1/notes.md"
    for bad in (
        "staging:../candidates/x.py",
        "staging:e2/evil.sh",
        "pool:xyz",
        "e2/fill.py",
        3,
    ):
        with pytest.raises(ManifestError) as exc:
            adopt_ref(bad)
        assert ADOPT_FORMS in str(exc.value)
    item = {
        "item": "memory.acct.users:user_id",
        "kind": "function",
        "tests": ["memory/acct/tests/test_users.py"],
    }
    man = parse_manifest(
        {"items": [{**item, "adopts": [STAGED, "pool:" + CID, STAGED]}]},
        library_mode=True,
    )
    assert man.items[0].adopts == ["staging:e2/candidates/fill.py", "pool:" + CID]
    assert parse_manifest({"items": [item]}, library_mode=True).items[0].adopts == []
    with pytest.raises(ManifestError):
        parse_manifest({"items": [{**item, "adopts": STAGED}]}, library_mode=True)
    # a manifest-wide list would credit no item: it is refused, naming where adopts goes
    with pytest.raises(ManifestError, match="belongs on each item"):
        parse_manifest({"items": [item], "adopts": [STAGED]}, library_mode=True)


def test_the_brief_puts_adopts_on_each_item():
    from unify.memory import library_prompts as lp

    brief = lp.write_brief_now(staging=True, pool=True)
    template = next(
        line for line in brief.splitlines() if line.lstrip().startswith('{"items": [')
    )
    assert '"covers": [...], "adopts": [...]}]' in template
    assert '"renames": {}, "adopts"' not in template
    assert (
        "that item's `adopts`" in brief
        and "The manifest itself has no `adopts`" in brief
    )
    plain = lp.write_brief_now()
    assert "adopts" not in plain


def _pool_with_staging(tmp_path):
    p = pool.Pool(Repo.init_bare(tmp_path / "mem.git")).load()
    got = p.enter_staging({"e2": _view()}, "p1")
    assert got == {"entries": 2, "added": 2}
    return p


def test_staged_refs_map_to_their_entries_and_adoption_closes_on_landing(tmp_path):
    p = _pool_with_staging(tmp_path)
    assert p.staged == {
        "staging:e2/candidates/fill.py": CID,
        "staging:e2/cases/fill.json": CID,
        "staging:e2/notes.md": LID,
    }
    man = {
        **MAN,
        "items": [{**ITEM, "adopts": ["staging:e2/cases/fill.json", "pool:" + LID]}],
    }
    assert pool.block_problems(man, p) == []
    retire = {**man, "pool": {"retired": {CID: "not needed"}}}
    assert pool.block_problems(retire, p) == [
        f"pool.retired: {CID} is adopted by {ITEM['item']}, which merges it when it lands",
    ]
    nothing = {**MAN, "items": [{**ITEM, "adopts": ["pool:" + "0" * 16]}]}
    assert pool.block_problems(nothing, p) == [
        f"{ITEM['item']}: adopts 'pool:{'0' * 16}', which is not a pool entry",
    ]
    prov = [
        {"item": ITEM["item"], "adopts": ["staging:e2/cases/fill.json", "pool:" + LID]},
    ]
    # the item was refused: nothing closes
    assert (
        pool.apply_block(p, man, "p1", landed=set(), finished=True, provenance=prov)
        == []
    )
    assert p.entries[CID].status == "open"
    changed = pool.apply_block(
        p,
        man,
        "p1",
        landed={ITEM["item"]},
        finished=True,
        provenance=prov,
    )
    assert changed == [CID, LID]
    assert p.entries[LID].status_detail == {
        "item": ITEM["item"],
        "via": "adopts",
        "pass": "p1",
    }


# --- 4. titles, order and counts -------------------------------------------------------------------------------


def test_titles_carry_content_and_the_index_lists_proposals_first_with_counts(tmp_path):
    p = pool.Pool(None)
    p.upsert(pool.draft("p0", "diff", "gate", ["memory.a.b:c"]), "p0")
    p.upsert(pool.deferred("e9", "the same region rule " * 20), "p0")
    for e in pool.from_staging("e1", _view()):
        p.upsert(e, "p0")
    for e in pool.from_staging("e3", _view()):
        p.upsert(e, "p0")
    assert p.stage(tmp_path / "pool") == 4
    lines = [
        ln
        for ln in (tmp_path / "pool" / "INDEX.md").read_text().splitlines()
        if ln.startswith("- `")
    ]
    assert [ln.split()[2] for ln in lines] == [
        "candidate:",
        "lesson:",
        "deferred:",
        "draft:",
    ]
    assert (
        lines[0] == f"- `{CID}` candidate: {CANDIDATE_TITLE} (2 episodes; opened in p0)"
    )
    assert not any(e in ln.split("`")[2] for ln in lines for e in ("e1", "e3", "e9"))
    assert all(len(e.title) <= pool.TITLE_CHARS for e in p.entries.values())
    long = "def g(x):\n    " + '"""' + "word " * 80 + '"""\n'
    assert len(pool.candidate_title("g", long)) == pool.TITLE_CHARS
    assert pool.candidate_title("h", "not python (") == "h: not python ("


def test_old_titles_are_recomputed_when_the_pool_loads(tmp_path):
    repo = Repo.init_bare(tmp_path / "mem.git")
    p = pool.Pool(repo).load()
    e = pool.from_staging("e1", _view())[0]
    p.upsert(e, "p0")
    e.title = (
        "candidate fill from episode e1"  # an entry saved before titles carried content
    )
    p.save("p0")
    assert pool.Pool(repo).load().entries[CID].title == CANDIDATE_TITLE


# --- 5. closed entries with new evidence ------------------------------------------------------------------------


def test_a_closed_entry_with_new_evidence_is_surfaced_and_can_be_reopened(tmp_path):
    p = _pool_with_staging(tmp_path)
    p.set_status(CID, "merged", {"item": ITEM["item"]}, "p1")
    for e in pool.from_staging("e3", _view()):
        p.upsert(e, "p2")
    e = p.entries[CID]
    assert e.status == "merged" and e.data["new_evidence"] == [
        {"pass": "p2", "episodes": ["e3"]},
    ]
    assert p.stage(tmp_path / "a") == 1  # the lesson is still open
    text = (tmp_path / "a" / "INDEX.md").read_text()
    head, _, section = text.partition("## Closed entries with new evidence")
    assert CID not in head
    assert (
        f"- `{CID}` candidate (merged): {CANDIDATE_TITLE} (1 episode since it closed)"
        in section
    )
    assert (tmp_path / "a" / CID / "candidate.py").read_text() == FILL

    assert pool.block_problems({**EMPTY, "pool": {"reopened": {LID: "r"}}}, p) == [
        f"pool.reopened: {LID} is not a closed entry",
    ]
    assert pool.block_problems({**EMPTY, "pool": {"reopened": {CID: " "}}}, p) == [
        f"pool.reopened: {CID} needs a reason",
    ]
    man = {**EMPTY, "pool": {"reopened": {CID: "e3 needs a second colour"}}}
    assert pool.apply_block(p, man, "p3", landed=set(), finished=False) == []
    assert pool.apply_block(p, man, "p3", landed=set(), finished=True) == [CID]
    assert e.status == "open" and "new_evidence" not in e.data
    assert e.status_detail == {"reopened": "e3 needs a second colour", "pass": "p3"}
    assert p.closed_with_new_evidence() == []


# --- 6. the retired miner's entries -----------------------------------------------------------------------------


def _old_cluster_entry() -> pool.Entry:
    """An entry the retired recurrence miner left in a pool (no code makes these now)."""
    e = pool.Entry(
        id=pool.entry_id("cluster", "a:" + "k" * 64),
        kind="cluster",
        title="code shape kkkkkkkk (function, 9 nodes)",
        data={"key": "k" * 64, "template": "", "instances": []},
    )
    e.add_provenance([], [], ["s0"])
    return e


def test_cluster_entries_close_once_and_a_pool_without_them_is_unchanged(tmp_path):
    repo = Repo.init_bare(tmp_path / "mem.git")
    p = pool.Pool(repo).load()
    c = _old_cluster_entry()
    p.upsert(c, "p0")
    p.upsert(pool.deferred("e1", "later"), "p0")
    p.save("p0")
    p = pool.Pool(repo).load()
    assert p.close_cluster_entries("p1") == [c.id]
    assert p.entries[c.id].status == "superseded"
    assert p.entries[c.id].status_detail == {"reason": pool.MINER_RETIRED, "pass": "p1"}
    p.save("p1")
    p = pool.Pool(repo).load()
    assert p.close_cluster_entries("p2") == [] and p.save("p2") is None
    assert [e.kind for e in p.open_entries()] == ["deferred"]


def test_a_pass_closes_the_old_cluster_entries_when_it_loads_the_pool(
    tmp_path,
    monkeypatch,
):
    st = _Stores(tmp_path)
    p = st.pool()
    c = _old_cluster_entry()
    p.upsert(c, "p0")
    p.save("p0")
    _finish_with(monkeypatch, EMPTY)
    _, model = st.run(
        [
            _call("r", "read", {"path": INDEX}),
            _open("e2"),
        ],
        "p1",
    )
    assert c.id not in model.outputs["r"]
    assert st.pool().entries[c.id].status == "superseded"


# --- 7. the hop counts ------------------------------------------------------------------------------------------


def test_chain_counts_read_events_and_unknown_parse_records(tmp_path):
    _stage(tmp_path, "e1", blocks=False)
    views = {"e1": _view("e1")}
    assert chain.parsed_candidates(tmp_path, ["e1"]) is None  # no parse record: unknown
    assert chain.parsed_candidates(tmp_path, []) == 0
    events = [
        {
            "event": "tool_call",
            "status": "ok",
            "paths": [
                {
                    "path": "/inputs/staging/e1/candidates/../candidates/fill.py",
                    "mode": "read",
                },
            ],
        },
        {
            "event": "tool_call",
            "status": "refused",
            "paths": [
                {"path": "/inputs/staging/e1/candidates/fill.py", "mode": "read"},
            ],
        },
        {"event": "open", "path": "/inputs/staging/e1/cases/fill.json", "mode": "r"},
    ]
    read = chain.read_paths([], events)
    assert read == {
        "/inputs/staging/e1/candidates/fill.py",
        "/inputs/staging/e1/cases/fill.json",
    }
    got = chain.counts(
        views=views,
        root=tmp_path,
        staged_entry={},
        pool_ids=set(),
        batch_rows=[
            {"staging": {"candidates": [{"name": "fill"}]}},
            {"episode_id": "e2"},
        ],
        read=read,
        provenance=[],
    )
    assert got == {
        "parsed": None,
        "staged": 1,
        "pooled_at_start": 0,
        "batch_map_rows": 1,
        "read_by_writer": 1,
        "adopted": 0,
    }


# --- 8. the gate records provenance for each landed item --------------------------------------------------------


@needs_bwrap
def test_the_gate_records_each_landed_items_adopts(library_example_world):  # noqa: F811
    from tests.memory import library_example_world as w
    from tests.memory.test_gate import _candidate

    ITEM_ID = w.ITEM_ID
    mem, ev, blobs, gate = library_example_world
    files, fixtures = w.files_and_fixtures(blobs.get)
    res = gate().merge(
        mem.head(),
        _candidate(mem, files),
        w.manifest(fixtures, w.item(adopts=[STAGED, "pool:" + CID])),
        "p1",
        "write",
        None,
        "0.01",
    )
    assert res.passed, res.reasons
    assert res.provenance == [
        {"item": ITEM_ID, "adopts": ["staging:e2/candidates/fill.py", "pool:" + CID]},
    ]
