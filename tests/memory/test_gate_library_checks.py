"""LibraryChecks on stub runs: no jail, no git."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from unify.memory import library_checks
from unify.memory.episodes import Cell
from unify.memory.library_checks import LibraryChecks, LibraryConfig
from unify.memory.evidence import EvidenceStore
from unify.memory.gate_policy import NOT_RUN_GATE
from unify.memory.manifest import ManifestError, parse_manifest
from unify.memory.procedures import Cover, cover_raw, parse_cover
from tests.memory.test_episodes import _ep

EP = _ep(
    episode_id="e1",
    cells=[
        Cell(0, "def total(rows):\n    return sum(rows)\n", ""),
        Cell(1, "print(1)", "1"),
    ],
)


class _Run(SimpleNamespace):
    def fail(self, check, reason, item=None):
        self.fails.append((check, reason, item))

    def note(self, reason):
        self.notes.append(reason)

    def not_run(self, item, why):
        self.labels.append((item, why))


def _item(**kw):
    base = dict(
        item="memory.calc.sums:total",
        kind="function",
        tests=[],
        source_episodes=["e1"],
        covers=[],
        typed_covers=[],
        input=None,
    )
    return SimpleNamespace(**{**base, **kw})


def _checks(tmp_path, items, cfg=None, **run):
    gate = SimpleNamespace(
        library_mode=cfg or LibraryConfig(episodes={"e1": EP}.get),
        _blob=lambda s: b"",
        ev=SimpleNamespace(signals_for=lambda e: []),
        lookup=lambda e, i: None,
    )
    base = dict(
        man=SimpleNamespace(items=items, deleted=[], unlisted=[], skeleton=[]),
        fails=[],
        notes=[],
        labels=[],
        verification={},
        curate_due=False,
        item_fail={},
        tmp=tmp_path,
        changed=[],
        p_files={},
        c_files={},
        p_tree=tmp_path / "p",
        c_tree=tmp_path / "c",
        p_bodies={},
        c_bodies={},
        manifest_raw={},
    )
    return LibraryChecks(gate, _Run(**{**base, **run}))


def test_cell_and_diff_covers_need_a_defined_function(tmp_path):
    ok = _item(typed_covers=[Cover("e1", "cell", index=0)])
    bad = _item(
        item="memory.calc.sums:other",
        typed_covers=[
            Cover("e1", "cell", index=1),
            Cover("e1", "diff"),
            Cover("e9", "cell", index=0),
        ],
    )
    v = _checks(tmp_path, [ok, bad])
    for it in (ok, bad):
        for c in it.typed_covers:
            v._typed_cover(it, c)
    reasons = [r for _, r, _ in v.run.fails]
    assert reasons == [
        "memory.calc.sums:other covers cell 1 of e1, which defines no Python function",
        "memory.calc.sums:other covers the work-tree diff of e1, which adds no Python function",
        "memory.calc.sums:other covers episode e9, which cannot be read",
    ]


def test_an_episode_cover_is_labelled_not_run_and_recorded(tmp_path):
    """Rev 2.1: the gate runs no procedure on its recorded episode; the cover is recorded as not run and labelled."""
    it = _item(typed_covers=[Cover("e1", "episode", runner="dialogue", action=0)])
    v = _checks(tmp_path, [it])
    v._typed_cover(it, it.typed_covers[0])
    assert v.run.fails == []
    assert v.run.labels == [("memory.calc.sums:total", f"G2 episode cover e1: {NOT_RUN_GATE}")]
    assert v.run.verification["memory.calc.sums:total"]["procedures"] == [
        {"episode": "e1", "runner": "dialogue", "ok": None, "ran": False},
    ]


def test_g4_notes_and_never_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(
        library_checks,
        "library_index_tokens",
        lambda tree, examples=None: 5000,
    )
    v = _checks(tmp_path, [])
    v.g4()
    # rev 2: no CURATE pass is made due; the size is a note only
    assert v.run.fails == [] and not v.run.curate_due
    assert v.run.notes == [
        "G4: the index is 5000 tokens, over its 4000-token view; growth is not refused",
    ]


def _write(root, files):
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(text)
    return {rel: ("100644", "x") for rel in files}


T = "memory/calc/tests/test_sums.py"


def test_test_loss_needs_a_reason_and_write_may_not_remove(tmp_path):
    p_files = _write(
        tmp_path / "p",
        {
            T: b"def test_a():\n    assert f(1) == 1\n\ndef test_b():\n    assert f(2) == 2\n",
            "memory/calc/sums.py": b"def f(x):\n    return x\n\ndef g(x):\n    return x\n",
        },
    )
    c_files = _write(
        tmp_path / "c",
        {
            T: b"def test_a():\n    assert f(1)\n",
            "memory/calc/sums.py": b"def f(x):\n    return x\n",
        },
    )
    v = _checks(
        tmp_path,
        [_item(tests=[T])],
        p_files=p_files,
        c_files=c_files,
        changed=sorted(p_files),
    )
    v._test_changes()
    got = [(c, r.split(" without")[0], i) for c, r, i in v.run.fails]
    assert (
        "G3",
        f"test {T}::test_b is deleted",
        ["memory.calc.sums:total"],
    ) in got and (
        "G3",
        f"test {T}::test_a is weakened (fewer or looser assertions)",
        ["memory.calc.sums:total"],
    ) in got
    assert any(
        c == "G3" and "fewer tests than its parent (2 to 1)" in r
        for c, r, _ in v.run.fails
    )
    assert any(
        c == "G5" and "may not reduce the number of items (2 to 1)" in r
        for c, r, _ in v.run.fails
    )
    # (CURATE, which could merge tests with a stated reason, is gone in rev 2)


def test_plain_pytest_refuses_the_gate_kit(tmp_path):
    c_files = _write(
        tmp_path / "c",
        {T: b"from memlab.inputs import inputs\n\ndef test_a():\n    assert 1\n"},
    )
    v = _checks(tmp_path, [_item(tests=[T])], c_files=c_files, changed=[T])
    v._plain()
    assert v.run.fails == [
        (
            "G3",
            f"[library_mode:plain] {T} imports the gate's test kit (memlab or the pin plugin); a "
            "library's tests run with plain pytest",
            ["memory.calc.sums:total"],
        ),
    ]


# --- the library-mode manifest, persisted typed covers ----------------------------------------------------------


def _function(**over):
    return {
        "item": "memory.calc.sums:total",
        "kind": "function",
        "source_episodes": ["e1"],
        "tests": ["memory/calc/tests/test_sums.py"],
        **over,
    }


def test_library_manifest_parses_function_ids_and_routes_typed_covers():
    raw = {
        "items": [_function(covers=[{"episode": "e1", "type": "cell", "cell": 0}])],
        "support": ["memory/calc/tests/helpers.py"],
        "deleted": ["memory.calc.old:gone"],
    }
    man = parse_manifest(raw)
    it = man.items[0]
    assert (it.path, it.covers, it.typed_covers) == (
        "memory/calc/sums.py",
        [],
        [Cover("e1", "cell", index=0)],
    )
    assert man.support == ["memory/calc/tests/helpers.py"] and man.deleted == [
        "memory.calc.old:gone",
    ]


def test_library_manifest_refuses_an_action_cover():
    """Replaces the action-cover routing of test_library_manifest_parses_function_ids_and_routes_covers. The rule
    that replaces it: an action cover is refused in library mode; a function's examples are the calls its tests make
    on fixtures.
    """
    raw = {
        "items": [
            _function(covers=[["e1", 0], {"episode": "e1", "type": "cell", "cell": 0}]),
        ],
    }
    with pytest.raises(ManifestError) as exc:
        parse_manifest(raw)
    texts = {
        f"memory.calc.sums:total: the action cover {c!r} is no longer read: list the tests that call it on "
        "fixtures made with fixture (covers takes only cell, diff and episode covers)"
        for c in (["e1", 0], ("e1", 0))
    }
    assert str(exc.value) in texts, str(exc.value)


def test_library_manifest_refuses_bad_ids_tests_and_covers():
    for bad in (
        {"items": [{"item": "env/x:f", "kind": "env_function"}]},
        {
            "items": [
                {
                    "item": "memory.calc.sums:total",
                    "kind": "function",
                    "tests": ["env/x/tests/test_a.py"],
                },
            ],
        },
        {
            "items": [
                {
                    "item": "memory.calc.sums:total",
                    "kind": "function",
                    "covers": [{"episode": "e1"}],
                },
            ],
        },
        {"items": [], "skeleton": ["env/x"]},
    ):
        with pytest.raises(ManifestError):
            parse_manifest(bad)


def test_cover_raw_round_trips_and_the_typed_covers_table_is_lazy(tmp_path):
    covers = [
        Cover("e1", "cell", index=2),
        Cover("e1", "diff"),
        Cover("e1", "episode", runner="dialogue", action=3, params={"n": 1}, paths=()),
        Cover("e2", "episode", runner="worktree", params={}, paths=("out.txt",)),
    ]
    assert [parse_cover(json.loads(json.dumps(cover_raw(c)))) for c in covers] == covers
    ev = EvidenceStore(tmp_path / "e.sqlite")
    assert ev.typed_covers() == [] and not ev._has_table("typed_covers")
    ev.add_typed_cover(
        "memory.a.b:f",
        "e2",
        json.dumps(cover_raw(covers[3]), sort_keys=True),
    )
    assert ev.typed_covers() == [
        ("memory.a.b:f", "e2", json.dumps(cover_raw(covers[3]), sort_keys=True)),
    ]




def _exact_labels(tmp_path, source: str, test: str) -> list[str]:
    """The G3 exact-assertion findings for ``memory.calc.sums:total`` defined by *source*, with *test* as its test."""
    (tmp_path / "c" / "memory/calc/tests").mkdir(parents=True)
    (tmp_path / "c" / "memory/calc/sums.py").write_text(source)
    (tmp_path / "c" / T).write_text(test)
    it = _item(tests=[T])
    v = _checks(tmp_path, [it], c_files={"memory/calc/sums.py": ("100644", "a"), T: ("100644", "b")})
    v._exact(it)
    return [r for c, r, _ in v.run.fails if c == "G3" and "[library_mode:exact]" in r]


def test_a_function_that_hands_back_a_value_needs_an_exact_assertion_of_its_result(tmp_path):
    src = "def total(rows):\n    return sum(rows)\n"
    test = "from memory.calc.sums import total\n\n\ndef test_total():\n    assert total([1, 2])\n"
    assert len(_exact_labels(tmp_path, src, test)) == 1


def test_a_function_whose_outcome_is_what_it_did_is_not_asked_for_an_exact_result(tmp_path):
    """Rev 2.1: in tool, action and conversation worlds the work is the calls; a function that hands back no
    value has no result to assert, so its tests are judged on what it did, never on an exact returned value."""
    src = "def total(env, rows):\n    env.ledger.post(sum(rows))\n    return None\n"
    test = "from memory.calc.sums import total\n\n\ndef test_total():\n    total(None, [1, 2])\n"
    assert _exact_labels(tmp_path, src, test) == []


def test_returns_a_value_reads_the_named_functions_returns_only():
    from unify.memory.fixed_answer import returns_a_value

    mod = "def a():\n    return 1\n\n\ndef b(env):\n    env.x()\n\n\ndef c():\n    return None\n"
    assert returns_a_value(mod, "a") is True
    assert returns_a_value(mod, "b") is False and returns_a_value(mod, "c") is False
    assert returns_a_value(mod, "missing") is True and returns_a_value("def (:", "a") is True
