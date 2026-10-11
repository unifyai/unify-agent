"""Memory v2.1 m4k (MAIN 11 Oct, the m4j dress-rehearsal diagnosis a1-m4j-rehearsal-diagnosis-20261011.md): the batch
map points at what else the pass holds about each episode, read_episode pages past its parts bound, finish's refusal
names what is missing, the brief states the batch map's schema and no longer speaks of the pass's time, and G3's
exact-assertion refusal shows a form that passes."""

import json

from unify.memory_v2 import pool, prompts_v21 as pv, sol_pass
from unify.memory_v2.batch_map import build_batch_map
from unify.memory_v2.views import Coverage
from tests.memory_v2.test_gate import _candidate
from tests.memory_v2.test_gate_v21 import (
    WEAK_TEST,
    _files_and_fixtures,
    _man,
    world21,
)  # noqa: F401 (a fixture)
from tests.memory_v2.test_pool import EMPTY, _finish_with, _Stores
from tests.memory_v2.test_sol_pass import _call
from tests.memory_v2.test_sol_v21_tools import EPS, _pass
from tests.memory_v2.test_staging import _staged

# --- 1. the batch map's cross-references -------------------------------------------------------------------------


def test_each_row_points_at_its_staged_candidates_and_the_pool_entries_citing_it(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "staging-root"
    d = _staged(root, "e2")
    (d / "cases").mkdir()
    (d / "cases" / "f.json").write_text("[]")
    st = _Stores(tmp_path)
    p = st.pool()
    did = p.upsert(pool.deferred("e2", "seen before"), "p0").id
    p.save("p0")
    _finish_with(monkeypatch, EMPTY)
    _, model = st.run(
        [
            _call("m", "read", {"path": "/inputs/batch_map.json"}),
            _call("x", "dismiss", {"episode": "e2", "reason": "r"}),
        ],
        "p1",
        staging_root=str(root),
    )
    row = json.loads(model.outputs["m"].split("\n[…")[0])["episodes"][0]
    assert row["staging"]["candidates"] == [
        {
            "name": "f",
            "file": "/inputs/staging/e2/candidates/f.py",
            "cases": "/inputs/staging/e2/cases/f.json",
        },
    ]
    assert row["pool_entries"] == [
        {"id": did, "kind": "deferred", "title": "episode e2 deferred: seen before"},
    ]
    assert row["s0_clusters"] == []


def test_s0_clusters_are_listed_under_each_episode_they_include():
    tier_a = [
        {
            "key": "k" * 64,
            "at": [
                {"episode": "e1", "cell": 2, "lines": [1, 3]},
                {"episode": "e2", "cell": 0, "lines": [4, 6]},
            ],
        },
    ]
    tier_b = {
        "units": {
            "u1": {"episode": "e1", "cell": 3},
            "u2": {"episode": "e3", "cell": 1},
        },
        "clusters": [{"units": ["u1", "u2"], "akeys": ["x", "y"]}],
    }
    refs = sol_pass._s0_by_episode(tier_a, tier_b)
    assert [r["tier"] for r in refs["e1"]] == ["a", "b"] and refs["e1"][0]["cells"] == [
        2,
    ]
    assert refs["e2"][0]["pool_entry"] == pool.entry_id("cluster", "a:" + "k" * 64)
    assert (
        refs["e3"][0]["pool_entry"] == pool.entry_id("cluster", "b:x,y")
        and refs["e3"][0]["episodes"] == 2
    )
    assert (
        sol_pass._s0_by_episode([], {}) == {}
        and sol_pass._s0_by_episode("bad", "bad") == {}
    )


# --- 2. read_episode pages past its parts bound --------------------------------------------------------------------


def test_more_parts_than_a_call_holds_are_paged_not_refused(tmp_path):
    turns = [
        _call("many", "read_episode", {"episode": "e2", "parts": ["request"] * 10}),
    ]
    _, model = _pass(tmp_path, turns, max_calls=5)
    out = model.outputs["many"]
    assert out.startswith("== request ==") and out.endswith(
        'next: parts=["request", "request"]]',
    )
    assert sol_pass._parts_from_text({"part": 'parts:["request", "cell:0"]'}) == {
        "parts": ["request", "cell:0"],
    }


# --- 3. finish's refusal names what is missing ----------------------------------------------------------------------


def test_the_finish_refusal_names_each_missing_part_and_what_is_dismissed():
    cov = Coverage(
        {"e1": ["request", "cell:0"], "e2": ["request"], "e3": ["request"]},
        {("e1", "request"): 3, ("e1", "cell:0"): 5, ("e2", "request"): 3, ("e3", "request"): 4},
    )  # fmt: skip
    cov.credit("e1", "request", 0, 3)
    assert cov.dismiss("e2", "nothing here") == "ok"
    text = sol_pass._finish_refusal(cov, cov.missing())
    assert text.splitlines() == [
        "not finished: 2 episode(s) neither covered nor dismissed:",
        "- e1: not yet read in full: cell:0",
        "- e3: not yet read in full: request",
        "Read those parts with read_episode, or dismiss the episode with a reason. Dismissed so far: e2.",
    ]


def test_a_pass_shows_the_new_refusal(tmp_path):
    _, model = _pass(
        tmp_path,
        [_call("f", "finish", {"summary": "s"})],
        eids=("e2",),
        max_calls=4,
    )
    assert model.outputs["f"].startswith(
        "not finished: 1 episode(s) neither covered nor dismissed:\n- e2: not yet read",
    )


# --- 4. the brief states the batch map's schema; 6. no sentence about the pass's time ---------------------------


def test_the_brief_names_every_batch_map_key_and_its_shape():
    text = pv.write_brief_now()
    bmap = build_batch_map(EPS.__getitem__, ["e1"])
    keys = (
        set(bmap)
        | set(bmap["episodes"][0])
        | {"staging", "s0_clusters", "pool_entries", "candidates"}
    )
    for k in sorted(keys):
        assert f"`{k}`" in text or f'\\"{k}\\"' in text or f'"{k}"' in text, k
    assert "`request` (str" in text and '{"version": 1, "episodes": [row, ...]' in text


def test_neither_brief_speaks_of_the_pass_time():
    for text in (pv.write_brief_now(), pv.curate_brief_now()):
        assert "time runs out" not in text and "kept as a draft" in text


# --- 5b. G3's exact-assertion refusal shows a passing form ----------------------------------------------------------


def test_the_exact_assertion_refusal_shows_a_form_that_passes(world21):
    mem, ev, blobs, gate = world21
    files, fixtures = _files_and_fixtures(blobs, test=WEAK_TEST)
    res = gate().check(mem.head(), _candidate(mem, files), _man(fixtures))
    reason = next(r for r in res.reasons if "[v21:exact]" in r)
    assert "A form that passes" in reason and "assert f(load(fixture)) ==" in reason
