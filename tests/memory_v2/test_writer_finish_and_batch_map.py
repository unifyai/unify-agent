"""Memory v2.1 m4k (MAIN 11 Oct, the m4j dress-rehearsal diagnosis a1-m4j-rehearsal-diagnosis-20261011.md): the batch
map points at what else the pass holds about each episode, read_episode pages past its parts bound, finish's refusal
names what is missing, the brief states the batch map's schema and no longer speaks of the pass's time, and G3's
exact-assertion refusal shows a form that passes."""

import json
from pathlib import Path

from unify.memory_v2 import pool, library_prompts as pv, writer_pass
from unify.memory_v2.batch_map import build_batch_map
from unify.memory_v2.views import Coverage
from tests.memory_v2.test_gate import _candidate
from unify.memory_v2 import fixtures as fx
from tests.memory_v2.test_gate_library_mode import (
    E1,
    ITEM_ID,
    TEST,
    WEAK_TEST,
    _files_and_fixtures,
    _man,
    world21,
)  # noqa: F401 (a fixture)
from tests.memory_v2.test_pool import EMPTY, _finish_with, _Stores
from tests.memory_v2.test_writer_pass import _call
from tests.memory_v2.test_writer_library_tools import EPS, _pass
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
    # the pass's own staged candidate entered the pool at the pass's start, so the row names its entry too
    cid = pool.entry_id(
        "candidate",
        json.dumps(
            {"candidate.py": "def f(x):\n    return x\n", "cases.json": "[]"},
            sort_keys=True,
        ),
    )
    assert row["pool_entries"] == [
        {"id": cid, "kind": "candidate", "title": "f(x)"},
        {"id": did, "kind": "deferred", "title": "seen before"},
    ]
    assert "s0_clusters" not in row


def test_a_write_pass_stages_no_miner_input_and_no_cluster_fields(
    tmp_path,
    monkeypatch,
):
    """The miner is not on the writer's path: no /inputs/s0, no cluster field in any batch-map row, no cluster entry
    fed to the pool, and no miner summary on the outcome."""
    seen: dict = {}
    staged = writer_pass.WriterPass._staging_library_mode

    async def spy(self, req, inputs, *args, **kwargs):
        await staged(self, req, inputs, *args, **kwargs)
        seen["inputs"] = sorted(p.name for p in Path(inputs).iterdir())
        seen["rows"] = json.loads((Path(inputs) / "batch_map.json").read_text())[
            "episodes"
        ]

    monkeypatch.setattr(writer_pass.WriterPass, "_staging_library_mode", spy)
    st = _Stores(tmp_path)
    _finish_with(monkeypatch, EMPTY)
    out, _ = st.run(
        [_call("x", "dismiss", {"episode": "e2", "reason": "r"})],
        "p1",
        staging_root=str(tmp_path / "staging-root"),
    )
    assert seen["inputs"] and "s0" not in seen["inputs"]
    assert seen["rows"] and all(
        not any(k.startswith("s0") or "cluster" in k for k in row)
        for row in seen["rows"]
    )
    assert not hasattr(out, "s0")
    assert all(e.kind != "cluster" for e in st.pool().entries.values())
    assert not hasattr(writer_pass, "_s0_by_episode")


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
    assert writer_pass._parts_from_text({"part": 'parts:["request", "cell:0"]'}) == {
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
    text = writer_pass._finish_refusal(cov, cov.missing())
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
        set(bmap) | set(bmap["episodes"][0]) | {"staging", "pool_entries", "candidates"}
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
    reason = next(r for r in res.reasons if "[library_mode:exact]" in r)
    # RUNTIME 18:17Z: the passing form loads the expected value from a fixture (no literal answer in code)
    assert (
        "A form that passes" in reason
        and "assert f(load(fixture_in)) == load(fixture_out)" in reason
    )
    assert "literal" not in reason.split("A form that passes")[1]


FIXTURE_FORM_TEST = TEST.replace(
    '''    observation = json.loads((DATA / "me.json").read_text())
    assert user_id(observation) == "u-1"''',
    """    observation = json.loads((DATA / "me.json").read_text())
    expected = json.loads((DATA / "expected.json").read_text())
    assert user_id(observation) == expected""",
)


def test_the_fixture_form_the_refusal_shows_passes_g3(world21):
    """MAIN 11 Oct: the form G3's refusal shows must pass the gate itself. Both sides are harness-recorded fixtures (the
    expected value a byte slice of the recorded response), with no literal in the test.
    """
    assert (
        FIXTURE_FORM_TEST != TEST
        and '"u-1"'
        not in FIXTURE_FORM_TEST.split("def test_reads")[1].split("@pytest")[0]
    )
    mem, ev, blobs, gate = world21
    files, fixtures = _files_and_fixtures(blobs, test=FIXTURE_FORM_TEST)
    whole = files["memory/acct/tests/data/me.json"].encode()
    at = whole.index(b'"u-1"')
    out, entry = fx.make(
        E1,
        0,
        "memory/acct/tests/data/expected.json",
        blob=blobs.get,
        cut=[at, at + 5],
    )
    assert json.loads(out) == "u-1"
    files["memory/acct/tests/data/expected.json"] = out.decode()
    fixtures["memory/acct/tests/data/expected.json"] = entry
    res = gate().merge(
        mem.head(),
        _candidate(mem, files),
        _man(fixtures),
        "p1",
        "write",
        None,
        "0.01",
    )
    assert res.passed, res.reasons
    assert (
        res.verification[ITEM_ID]["exact_assertions"] == 2
    )  # the fixture-form assertion counts, as the literal did
