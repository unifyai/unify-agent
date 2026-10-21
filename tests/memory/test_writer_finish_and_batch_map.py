"""The batch map points at what else the pass holds about each episode, read_episode pages past its parts bound,
finish's refusal names what is missing, the brief states the batch map's schema and no longer speaks of the pass's
time, and G3's exact-assertion refusal shows a form that passes."""

import json
from pathlib import Path


from unify.memory import pool, library_prompts as pv, writer_pass
from unify.memory.batch_map import build_batch_map
from tests.memory.library_example_world import library_example_world  # noqa: F401 - a fixture
from tests.memory.test_gate import _candidate
from unify.memory import fixtures as fx
from tests.memory import library_example_world as w
from tests.memory.test_pool import EMPTY, _finish_with, _Stores
from tests.memory.test_writer_pass import _call
from tests.memory.test_writer_library_tools import EPS, _pass
from tests.memory.test_staging import _staged

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
    """The miner is not on the writer's path: no similarity-miner input directory, no cluster field in any batch-map row, no cluster entry
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
        [_call("x", "read_episode", {"episode": "e2", "parts": ["request"]})],
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
    from unify.memory.writer_tools import _wanted_parts

    assert _wanted_parts({"part": 'parts:["request", "cell:0"]'}) == (
        ["request", "cell:0"],
        None,
    )


# --- 3. finish never waits on reading -------------------------------------------------------------------------------


def test_finish_is_refused_only_for_its_manifest(tmp_path):
    _, model = _pass(
        tmp_path,
        [_call("f", "finish", {"summary": "s"})],
        eids=("e2",),
        max_calls=4,
    )
    assert model.outputs["f"].startswith("not finished: G1: no manifest")


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


# --- 5b. G3's exact-assertion refusal shows a passing form ----------------------------------------------------------


def test_the_exact_assertion_refusal_shows_a_form_that_passes(library_example_world):
    mem, ev, blobs, gate = library_example_world
    files, fixtures = w.files_and_fixtures(blobs.get, test=w.WEAK_TEST_SOURCE)
    res = gate().check(mem.head(), _candidate(mem, files), w.manifest(fixtures))
    # an evidence finding: it sets the function's evidence level instead of refusing it
    assert res.passed and res.levels[w.ITEM_ID] == "unchecked"
    reason = next(r for r in res.level_reasons[w.ITEM_ID] if "[library_mode:exact]" in r)
    # the passing form loads the expected value from a fixture (no literal answer in code)
    assert (
        "A form that passes" in reason
        and "assert f(load(fixture_in)) == load(fixture_out)" in reason
    )
    assert "literal" not in reason.split("A form that passes")[1]


FIXTURE_FORM_TEST = w.TEST_SOURCE.replace(
    '    assert user_id(load("me_1.json")) == "u-1"',
    '    assert user_id(load("me_1.json")) == load("expected.json")',
)


def test_the_fixture_form_the_refusal_shows_passes_g3(library_example_world):
    """The form G3's refusal shows must pass the gate itself. Both sides are harness-recorded fixtures (the
    expected value a byte slice of the recorded response), with no literal in the test.
    """
    assert (
        FIXTURE_FORM_TEST != w.TEST_SOURCE
        and '"u-1"'
        not in FIXTURE_FORM_TEST.split("def test_reads")[1].split("@pytest")[0]
    )
    mem, ev, blobs, gate = library_example_world
    files, fixtures = w.files_and_fixtures(blobs.get, test=FIXTURE_FORM_TEST)
    whole = files[w.FIXTURE_PATHS[0]].encode()
    at = whole.index(b'"u-1"')
    out, entry = fx.make(
        w.E1,
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
        w.manifest(fixtures),
        "p1",
        "write",
        None,
        "0.01",
    )
    assert res.passed, res.reasons
    # the fixture-form assertion counts, as the literal did: with the other fixture's literal assertion and the
    # recorded-inputs test's (its expected value comes from its fixture), three, as on the literal form
    assert res.verification[w.ITEM_ID]["exact_assertions"] == 3
