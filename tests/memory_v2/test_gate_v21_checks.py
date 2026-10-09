"""V21Checks on stub runs: no jail, no git."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from unify.memory_v2 import gate_v21
from unify.memory_v2.episodes import Cell
from unify.memory_v2.gate_v21 import V21Checks, V21Config
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.manifest import ManifestError, parse_manifest
from unify.memory_v2.procedures import Cover, Outcome, cover_raw, parse_cover
from unify.memory_v2.sandbox_run import PYTHON
from tests.memory_v2.test_episodes import _ep

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
        v21=cfg or V21Config(episodes={"e1": EP}.get),
        python=Path("/usr/bin/python3"),
        _blob=lambda s: b"",
        ev=SimpleNamespace(signals_for=lambda e: []),
        lookup=lambda e, i: None,
    )
    base = dict(
        man=SimpleNamespace(items=items, deleted=[], unlisted=[], skeleton=[]),
        fails=[],
        notes=[],
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
    return V21Checks(gate, _Run(**{**base, **run}))


def test_config_refuses_an_unknown_role():
    with pytest.raises(ValueError, match="role"):
        V21Config(role="judge")


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
            v._typed_cover(it, c, 0)
    reasons = [r for _, r, _ in v.run.fails]
    assert reasons == [
        "memory.calc.sums:other covers cell 1 of e1, which defines no Python function",
        "memory.calc.sums:other covers the work-tree diff of e1, which adds no Python function",
        "memory.calc.sums:other covers episode e9, which cannot be read",
    ]


def test_episode_cover_runs_its_procedure_and_records_it(tmp_path, monkeypatch):
    calls = []

    def fake(item, cover, **kw):
        calls.append((item, cover.runner, kw["tree"]))
        return Outcome(False, "its result differs from the recorded action")

    monkeypatch.setattr(gate_v21, "run_procedure", fake)
    it = _item(typed_covers=[Cover("e1", "episode", runner="dialogue", action=0)])
    v = _checks(tmp_path, [it])
    v._typed_cover(it, it.typed_covers[0], 0)
    assert calls == [("memory.calc.sums:total", "dialogue", tmp_path / "c")]
    assert v.run.verification["memory.calc.sums:total"]["procedures"] == [
        {"episode": "e1", "runner": "dialogue", "ok": False},
    ]
    assert (
        v.run.fails[0][0] == "G2" and "procedure on e1 (dialogue)" in v.run.fails[0][1]
    )


def test_g4_notes_curate_and_never_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(gate_v21, "_index_tokens", lambda tree: 5000)
    v = _checks(tmp_path, [])
    v.g4()
    assert v.run.fails == [] and v.run.curate_due
    assert v.run.notes == [
        "G4 CURATE due: the index is 5000 tokens, over its 4000-token view; growth is not refused",
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
        c == "G3" and "may not reduce the number of tests (2 to 1)" in r
        for c, r, _ in v.run.fails
    )
    assert any(
        c == "G5" and "may not reduce the number of items (2 to 1)" in r
        for c, r, _ in v.run.fails
    )
    stated = {"tests_changed": {T: "merged into test_a by CURATE"}}
    v = _checks(
        tmp_path,
        [_item(tests=[T])],
        cfg=V21Config(role="curate"),
        p_files=p_files,
        c_files=c_files,
        changed=sorted(p_files),
        manifest_raw=stated,
    )
    v._test_changes()
    assert v.run.fails == []


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
            f"[v21:plain] {T} imports the gate's test kit (memlab or the pin plugin); a "
            "library's tests run with plain pytest",
            ["memory.calc.sums:total"],
        ),
    ]


# --- the v2.1 manifest, persisted typed covers (Amendment C) ----------------------------------------------------


def test_v21_manifest_parses_function_ids_and_routes_covers():
    raw = {
        "items": [
            {
                "item": "memory.calc.sums:total",
                "kind": "function",
                "source_episodes": ["e1"],
                "tests": ["memory/calc/tests/test_sums.py"],
                "input": "observation",
                "covers": [["e1", 0], {"episode": "e1", "type": "cell", "cell": 0}],
            },
        ],
        "support": ["memory/calc/tests/helpers.py"],
        "deleted": ["memory.calc.old:gone"],
    }
    man = parse_manifest(raw, v21=True)
    it = man.items[0]
    assert (it.path, it.covers, it.typed_covers) == (
        "memory/calc/sums.py",
        [("e1", 0)],
        [Cover("e1", "cell", index=0)],
    )
    assert man.support == ["memory/calc/tests/helpers.py"] and man.deleted == [
        "memory.calc.old:gone",
    ]
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
            parse_manifest(bad, v21=True)
    with pytest.raises(ManifestError):
        parse_manifest(raw)  # v2 is unchanged: no function kind


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


def _stored(*rows):
    return SimpleNamespace(
        typed_covers=lambda: [
            (i, c.episode, json.dumps(cover_raw(c))) for i, c in rows
        ],
        signals_for=lambda e: [],
    )


def test_a_deleted_procedure_must_stay_covered(tmp_path):
    proc = Cover("e1", "episode", runner="worktree", params={})
    gone = "memory.calc.sums:total"
    v = _checks(tmp_path, [], c_bodies={})
    v.gate.ev, v.run.man.deleted = _stored((gone, proc)), [gone]
    v.g5()
    assert v.run.fails == [
        (
            "G5",
            f"{gone} is deleted, but its procedures on e1 are covered by no remaining item; "
            "a kept function must take over each recorded job",
            gone,
        ),
    ]
    taker = _item(
        item="memory.calc.sums:other",
        typed_covers=[
            Cover(
                "e1",
                "episode",
                runner="worktree",
                params={"x": 1},
            ),
        ],
    )
    v = _checks(tmp_path, [taker], c_bodies={})
    v.gate.ev, v.run.man.deleted = _stored((gone, proc)), [gone]
    v.g5()
    assert v.run.fails == []


JOBS = """from pathlib import Path

from memory.proc.util import amount


def total_amounts(root, src, dst):
    rows = [line.split(",") for line in (Path(root) / src).read_text().splitlines()]
    (Path(root) / dst).write_text(str(sum(amount(r) for r in rows)) + "\\n")
"""
UTIL = "def amount(row):\n    return int(row[1])\n"


def _lib(root, util):
    for rel, text in {
        "memory/__init__.py": "",
        "memory/proc/__init__.py": '"""Procedures."""\n',
        "memory/proc/jobs.py": JOBS,
        "memory/proc/util.py": util,
    }.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap required")
def test_a_helper_edit_that_breaks_a_stored_procedure_is_refused(tmp_path):
    from tests.memory_v2.test_procedures import _wt

    ep, cover, files = _wt(tmp_path)
    job, helper = "memory.proc.jobs:total_amounts", "memory.proc.util:amount"
    for util, refused in (
        (UTIL, False),
        (UTIL.replace("int(row[1])", "int(row[1]) + 1"), True),
    ):
        case = tmp_path / ("bad" if refused else "ok")
        v = _checks(
            case,
            [_item(item=helper)],
            cfg=V21Config(episodes={"e1": ep}.get, worktree_files=files),
            p_tree=_lib(case / "p", UTIL),
            c_tree=_lib(case / "c", util),
            changed=["memory/proc/util.py"],
            p_bodies={job: ("function", "j", True), helper: ("function", "a", True)},
            c_bodies={job: ("function", "j", True), helper: ("function", "b", True)},
        )
        v.gate.ev, v.gate.python = _stored((job, cover)), PYTHON
        v._procedure_behaviour()
        if not refused:
            assert v.run.fails == []
        else:
            [(check, reason, owner)] = v.run.fails
            assert check == "G3" and owner == helper
            assert reason.startswith(
                f"{job} changes behaviour without a test: its procedure on e1 (worktree) passes",
            )


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap required")
def test_an_init_rebinding_that_breaks_a_stored_procedure_is_refused(tmp_path):
    """RUNTIME's review B1: the scope comes from the import graph, where a module depends on its package init."""
    from tests.memory_v2.test_procedures import _wt
    from unify.memory_v2 import layout

    ep, cover, files = _wt(tmp_path)
    job = "memory.proc.jobs:total_amounts"
    init = '"""Procedures."""\nfrom . import util\n\nutil.amount = lambda row: int(row[1]) + 1\n'
    p_tree, c_tree = _lib(tmp_path / "p", UTIL), _lib(tmp_path / "c", UTIL)
    (c_tree / "memory/proc/__init__.py").write_text(init)
    v = _checks(
        tmp_path,
        [],
        cfg=V21Config(episodes={"e1": ep}.get, worktree_files=files),
        p_tree=p_tree,
        c_tree=c_tree,
        changed=["memory/proc/__init__.py"],
        p_bodies={job: ("function", "j", True)},
        c_bodies={job: ("function", "j", True)},
        graph=layout.merge_graphs(
            layout.import_graph(p_tree),
            layout.import_graph(c_tree),
        ),
    )
    v.gate.ev, v.gate.python = _stored((job, cover)), PYTHON
    v._procedure_behaviour()
    assert [(c, r.split(":", 1)[0]) for c, r, _ in v.run.fails] == [
        ("G3", f"{job} changes behaviour without a test"),
    ]
