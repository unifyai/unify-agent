"""CURATE's trigger, inputs, trailers and brief, and its evidence tables."""

from __future__ import annotations

import inspect
import json
import re

from unify.memory import curate as cu
from unify.memory.curate import CurateState
from unify.memory.evidence import EvidenceStore

F = "memory.a.b:f"
CAND = {
    "rule": "antiunify",
    "items": [F, "memory.a.c:g"],
    "reasons": ["bodies anti-unify: kept share 1.00, 0 hole(s)"],
    "fingerprint": "overlap:1",
}
OVERLAP = {"version": 1, "candidates": [CAND], "truncated": []}
BODY = "def f(x):\n    return x"


def _state(**kw):
    base = dict(
        commit="c" * 40,
        overlap={"version": 1, "candidates": [], "truncated": []},
        suspects={},
        index_tokens=100,
    )
    return CurateState(**{**base, **kw})


def test_nothing_due_on_a_clean_library():
    assert cu.due(_state(), set()) == {}


def test_each_condition_fires_with_its_reason():
    st = _state(
        overlap=OVERLAP,
        suspects={F: {"reasons": ["raised in 2 episodes"], "episodes": ["e1", "e2"]}},
        bodies={F: BODY},
        index_tokens=4001,
    )
    got = cu.due(st, set())
    assert sorted(got.values()) == [
        "index: 4001 tokens, over its 4000-token view",
        "overlap (antiunify): memory.a.b:f, memory.a.c:g",
        "suspect: memory.a.b:f (raised in 2 episodes)",
    ]
    assert "overlap:1" in got and f"index:{'c' * 40}" in got
    assert cu.due(_state(index_tokens=4000), set()) == {}
    assert cu.due(_state(index_tokens=None), set()) == {}


def test_seen_fingerprints_do_not_fire_again_until_content_changes():
    st = _state(
        overlap=OVERLAP,
        suspects={F: {"reasons": ["raised in 2 episodes"]}},
        bodies={F: BODY},
    )
    first = cu.due(st, set())
    assert len(first) == 2 and cu.due(st, set(first)) == {}
    # the suspect item is edited: it fires again; the overlap candidate, unchanged, does not
    st.bodies[F] = "def f(x):\n    return x + 0"
    assert list(cu.due(st, set(first)).values()) == [
        "suspect: memory.a.b:f (raised in 2 episodes)",
    ]


def test_the_index_reason_fires_once_per_library_commit():
    (fp,) = cu.due(_state(index_tokens=5000), set())
    assert cu.due(_state(index_tokens=5000), {fp}) == {}
    assert len(cu.due(_state(commit="d" * 40, index_tokens=5000), {fp})) == 1


def test_the_trigger_has_no_stream_input():
    assert list(inspect.signature(cu.due).parameters) == ["state", "seen", "budget"]
    assert set(CurateState.__dataclass_fields__) == {
        "commit",
        "overlap",
        "suspects",
        "index_tokens",
        "bodies",
        "use",
        "history",
        "aliases",
        "rollback_files",
        "fired",
    }


def test_stage_inputs_writes_every_input_and_the_rollback_files(tmp_path):
    module, test = b"def f(x):\n    return x\n", b"def test_f():\n    pass\n"
    st = _state(
        overlap=OVERLAP,
        suspects={
            F: {
                "reasons": ["raised in 2 episodes"],
                "episodes": ["e1"],
                "bisect": {"first_changed": "b" * 40},
                "rollback": "a" * 40,
            },
        },
        use={F: {"c" * 40: {"uses": 3, "errors": 2}}},
        history={F: ["bbbbbbbbbbbb change f"]},
        aliases={
            "memory.a.b:old": {"target": F, "pass_id": "c0", "commit": "c" * 40},
            "memory.a.b:older": {"target": F, "pass_id": "c9", "commit": "9" * 40},
        },
        rollback_files={F: {"memory/a/b.py": module, "memory/a/tests/test_b.py": test}},
        fired={"overlap:1": "overlap (antiunify): memory.a.b:f, memory.a.c:g"},
    )
    written = cu.stage_inputs(tmp_path, st)
    root = tmp_path / "curate"
    assert written == [
        "/inputs/curate/aliases.json",
        "/inputs/curate/history.json",
        "/inputs/curate/overlap.json",
        "/inputs/curate/rollback/0/memory/a/b.py",
        "/inputs/curate/rollback/0/memory/a/tests/test_b.py",
        "/inputs/curate/rollback/index.json",
        "/inputs/curate/same_shape.json",
        "/inputs/curate/suspects.json",
        "/inputs/curate/trigger.json",
        "/inputs/curate/use.json",
    ]
    assert json.loads((root / "trigger.json").read_text()) == {
        "commit": "c" * 40,
        "reasons": ["overlap (antiunify): memory.a.b:f, memory.a.c:g"],
    }
    aliases = json.loads((root / "aliases.json").read_text())
    assert (
        aliases["memory.a.b:old"]["may_remove"] is False
        and aliases["memory.a.b:older"]["may_remove"] is True
    )
    assert (root / "rollback/0/memory/a/b.py").read_bytes() == module
    assert json.loads((root / "rollback/index.json").read_text()) == [
        {
            "item": F,
            "dir": "/inputs/curate/rollback/0",
            "commit": "a" * 40,
            "files": ["memory/a/b.py", "memory/a/tests/test_b.py"],
        },
    ]
    assert json.loads((root / "overlap.json").read_text()) == OVERLAP


def test_the_message_names_why_and_where():
    msg = cu.message(
        _state(fired={"overlap:1": "overlap (antiunify): memory.a.b:f, memory.a.c:g"}),
    )
    assert msg.startswith(
        "This CURATE pass runs because:\n- overlap (antiunify): memory.a.b:f, memory.a.c:g\n",
    )
    assert "/inputs/curate/" in msg


def test_trailers_name_why_and_every_item():
    man = {
        "why": "two readers\nof one file",
        "items": [{"item": "memory.a.b:new"}],
        "deleted": ["memory.a.b:old"],
        "aliases": {"memory.a.b:old": "memory.a.b:new"},
        "retired": {"memory.a.c:gone": "wrong on every use"},
    }
    assert cu.trailers(man) == {
        "Why": ["two readers of one file"],
        "Items": ["memory.a.b:new", "memory.a.b:old", "memory.a.c:gone"],
    }
    assert cu.trailers(None) == {} and cu.trailers({"why": " "}) == {}


def test_the_brief_states_each_responsibility_and_input_and_no_benchmark_words():
    text = cu.curate_system()
    for phrase in (
        "keeping each old name working as an alias of the merged function",
        "Generalise variants into one parametric function",
        "Repair or roll back suspect items",
        "Retire items that are wrong",
        "Fix broken links",
        "Regroup packages",
        "overlap.json",
        "suspects.json",
        "use.json",
        "history.json",
        "bisect result",
        "retires a test with a stated reason",
        '"aliases"',
        '"retired"',
        '"why"',
    ):
        assert phrase in text, phrase
    assert "{entries}" not in text and "{checks}" not in text
    lowered = text.lower()
    for word in ("apis", "venmo", "crafter", "benchmark", "task", "example", "solved"):
        assert word not in lowered, word
    assert not re.search(r"\bARC\b", text)


def test_curate_tables_are_lazy_and_round_trip(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")

    def tables():
        return {
            r[0]
            for r in ev.db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }

    assert (
        ev.curate_seen() == set()
        and ev.curations() == []
        and ev.aliases() == {}
        and ev.typed_covers() == []
    )
    assert (
        not {"curate_seen", "curations"} & tables()
    )  # a store with no CURATE pass keeps v2's schema
    assert ev.record_curate_seen("c1", ["overlap:a", "index:x"]) == 2
    assert ev.record_curate_seen("c2", ["overlap:a"]) == 0
    assert ev.curate_seen() == {"overlap:a", "index:x"}
    rows = [
        {
            "item": "memory.a.b:old",
            "action": "alias",
            "target": "memory.a.b:new",
            "reason": "same job",
        },
        {
            "item": "memory.a.b:gone",
            "action": "retire",
            "target": None,
            "reason": "wrong on every use",
        },
    ]
    assert ev.record_curations("c1", "1" * 40, rows) == 2
    assert ev.aliases() == {
        "memory.a.b:old": {
            "target": "memory.a.b:new",
            "pass_id": "c1",
            "commit": "1" * 40,
        },
    }
    drop = {
        "item": "memory.a.b:old",
        "action": "drop_alias",
        "target": "memory.a.b:new",
        "reason": "kept one pass",
    }
    assert ev.record_curations("c3", "3" * 40, [drop]) == 1
    assert ev.aliases() == {}
    assert [r["action"] for r in ev.curations()] == ["alias", "retire", "drop_alias"]
    ev.db.execute(
        "CREATE TABLE typed_covers(item TEXT, episode_id TEXT, cover_json TEXT)",
    )
    ev.db.execute(
        "INSERT INTO typed_covers VALUES('memory.a.b:f', 'e1', '{\"type\": \"diff\"}')",
    )
    assert ev.typed_covers() == [("memory.a.b:f", "e1", '{"type": "diff"}')]


def _fired(**kw):
    st = _state(
        overlap=OVERLAP,
        suspects={F: {"reasons": ["raised in 2 episodes"]}},
        bodies={F: BODY},
        index_tokens=5000,
        **kw,
    )
    st.fired = cu.due(st, set())
    return st


def _fp(st, prefix):
    (fp,) = [f for f in st.fired if f.startswith(prefix)]
    return fp


def test_each_reason_names_its_items():
    st = _fired()
    names = cu.named(st)
    assert names[_fp(st, "overlap:")] == [F, "memory.a.c:g"]
    assert names[_fp(st, "suspect:")] == [F]
    assert names[_fp(st, "index:")] == [cu.INDEX_NAME]


def test_a_pass_that_changed_nothing_and_kept_nothing_settles_nothing():
    st = _fired()
    assert cu.settled(st, set(), {}) == []
    # every reason fires again next time
    assert cu.due(st, set(cu.settled(st, set(), {}))) == st.fired


def test_a_reason_is_settled_when_the_pass_changed_an_item_it_names():
    st = _fired()
    # an alias of the overlap's second item: the overlap and the index are dealt with, the suspect is not
    got = cu.settled(st, {"memory.a.c:g"}, {})
    assert got == sorted([_fp(st, "overlap:"), _fp(st, "index:")])
    assert list(cu.due(st, set(got)).values()) == [
        "suspect: memory.a.b:f (raised in 2 episodes)",
    ]


def test_a_reason_is_settled_by_an_explicit_keep_with_a_reason():
    st = _fired()
    keeps = cu.kept(
        {"kept": {"memory.a.c:g": "different jobs: one parses, one prints"}},
    )
    assert cu.settled(st, set(), keeps) == [_fp(st, "overlap:")]
    assert cu.settled(
        st,
        set(),
        cu.kept({"kept": {cu.INDEX_NAME: "every section is needed"}}),
    ) == [
        _fp(st, "index:"),
    ]
    # a keep without a reason is no keep
    assert cu.kept({"kept": {F: "  ", "memory.a.c:g": None, 3: "x"}}) == {}
    assert cu.kept({"kept": ["memory.a.c:g"]}) == {} and cu.kept(None) == {}


def test_a_manifest_that_lands_only_new_notes_leaves_the_reasons_it_was_shown_unaddressed():
    st = _fired()
    notes_only = {"items": [{"item": "notes/a/region_filling.md", "kind": "note"}]}
    left = cu.unaddressed(st, notes_only)
    # the overlap and the suspect name functions the manifest leaves alone; any change meets the index reason
    assert left == [
        line for fp, line in sorted(st.fired.items()) if not fp.startswith("index:")
    ]
    assert set(cu.unaddressed(st, {})) == set(st.fired.values())


def test_a_manifest_addresses_a_reason_by_changing_or_keeping_an_item_it_names():
    st = _fired()
    whole = {
        "aliases": {"memory.a.c:g": F},
        "kept": {F: "the suspect reports come from one caller's bad input"},
    }
    assert cu.unaddressed(st, whole) == []
    assert (
        cu.unaddressed(st, {"deleted": ["memory.a.c:g"], "retired": {F: "wrong"}}) == []
    )
    assert cu.unaddressed(st, {"renames": {"memory.a.x:h": F}}) == []
    # what it addresses is what settled() settles once the changes land
    changed = {"memory.a.c:g", F}
    assert cu.settled(st, changed, cu.kept(whole)) == sorted(st.fired)


def test_changed_items_come_only_from_a_landed_pass():
    curations = [
        {"pass_id": "c1", "item": "memory.a.b:old", "action": "alias", "target": F},
        {
            "pass_id": "c1",
            "item": "memory.a.b:gone",
            "action": "retire",
            "target": None,
        },
        {"pass_id": "c0", "item": "memory.z.z:z", "action": "retire", "target": None},
    ]
    manifest = {"deleted": ["memory.a.b:old", "memory.a.b:gone", "memory.a.b:dropped"]}
    assert cu.changed_items("c1", False, [F], curations, manifest) == set()
    assert cu.changed_items("c1", True, [F], curations, manifest) == {
        F,
        "memory.a.b:old",
        "memory.a.b:gone",
        "memory.a.b:dropped",
    }
