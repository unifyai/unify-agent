"""Compiled code, start-up hooks and stray root entries are refused before anything is extracted or run.

Review I4 (v2.1): on 9deefbfd1 ``layout_allowed`` admitted every non-``.py`` path (``manifest.py:297``, a
non-``.py`` path under a tests directory, and ``manifest.py:300``, any other non-``.py`` path) and
``forbidden`` (``manifest.py:255``) names only ``sitecustomize.py``/``usercustomize.py`` and ``*.pth``. So
``Gate._early`` (``gate.py:624``) passed such a candidate, ``materialise`` extracted it and G3 ran pytest with
``/memory`` holding, say, a sourceless ``env/__init__.pyc`` (run on every ``import env.*``), a root
``sitecustomize.pyc`` (run at interpreter start) or a root ``pytest.pyc`` (run instead of pytest). A file
under ``env/<channel>/tests/`` could even be declared in ``support`` (``support_allowed``,
``manifest.py:281``, admits any name with a non-``.py`` suffix, as no module name holds a dot) and merged. Each test below states which 9deefbfd1 check admitted its path, by asserting that check (unchanged
here), and shows that the gate now refuses it with no test run at all.

Module-level imports are only names 9deefbfd1 already had, so this file also runs against that build (the
fail-before run); the tests of the new policy function import it where they use it.
"""

import ast
import json
import shutil
from pathlib import Path

import pytest

from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.gate import Gate
from unify.memory_v2.gitio import Repo
from unify.memory_v2.manifest import (
    TESTKIT,
    forbidden,
    layout_allowed,
    support_allowed,
)
from tests.memory_v2.test_episodes import _ep
from tests.memory_v2.test_gate import (
    BASE2,
    FILES,
    MAN,
    PROBE_BASE,
    PROBE_ITEM,
    _lookup,
    _probe_runner,
)
from tests.memory_v2.test_kinds_gate import WT_FILES

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap required",
)

# Planted in every refused file: a reason must name the path and the class of problem, never contents.
MARKER = "I4-CONTENT-MARKER"
PAYLOAD = b"\x00" + MARKER.encode() + b"\x00print('ran')\n"


@pytest.fixture
def world(tmp_path):
    """test_gate's store, without its gate: each test builds a gate around a recording runner."""
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ev.index_episode(_ep(episode_id="e1"), "1" * 40)
    return mem, ev


class _Recorder:
    """The probe runner, recording each pytest run and whether the planted path was in its tree."""

    def __init__(self, planted: str) -> None:
        self.planted = planted
        self.calls: list[tuple[str, bool]] = []

    def __call__(self, target, *, python, ro, rw, cwd, timeout_s=300.0, env=None):
        (tree,) = ro
        self.calls.append((target, (Path(tree) / self.planted).exists()))
        return _probe_runner(
            target,
            python=python,
            ro=ro,
            rw=rw,
            cwd=cwd,
            timeout_s=timeout_s,
            env=env,
        )


def _gate(tmp_path, mem, ev, runner):
    return Gate(
        mem,
        ev,
        BlobStore(tmp_path / "lb"),
        action_lookup=_lookup,
        pytest_runner=runner,
    )


def _commit(mem, files):
    with mem.temp_checkout("main") as wt:
        for rel, data in files.items():
            (wt / rel).parent.mkdir(parents=True, exist_ok=True)
            (wt / rel).write_bytes(data if isinstance(data, bytes) else data.encode())
        return mem.commit_all(wt, "pass", {"Pass": "p1"})


def _refused_early(res, check, reason):
    assert not res.passed
    assert res.reasons[0] == f"{check}: {reason}", res.reasons
    assert MARKER not in "\n".join(res.reasons)
    # stopped before extraction: every other check is reported as not evaluated
    others = [c for c in ("G1", "G2", "G3", "G4", "G5", "G6") if c != check]
    assert all(f"{c}: not evaluated" in res.reasons for c in others), res.reasons


# Not declarable (no manifest entry can name them), so 9deefbfd1 refused each only under G1 ("undeclared
# change"), after G3 had run pytest with the file in /memory. Now: refused before extraction, no run.
UNDECLARED = [
    ("json.pyc", "G6", "bytecode or native code file json.pyc"),
    ("pytest.pyc", "G6", "bytecode or native code file pytest.pyc"),
    ("sitecustomize.pyc", "G6", "bytecode or native code file sitecustomize.pyc"),
    ("env/__init__.pyc", "G6", "bytecode or native code file env/__init__.pyc"),
    ("env/venmo/extra.pyc", "G6", "bytecode or native code file env/venmo/extra.pyc"),
    (
        "env/venmo/_fast.cpython-312-x86_64-linux-gnu.so",
        "G6",
        "bytecode or native code file env/venmo/_fast.cpython-312-x86_64-linux-gnu.so",
    ),
    (
        "__pycache__/json.cpython-312.pyc",
        "G6",
        "bytecode cache path __pycache__/json.cpython-312.pyc",
    ),
    (
        "env/venmo/__pycache__/__init__.cpython-312.pyc",
        "G6",
        "bytecode cache path env/venmo/__pycache__/__init__.cpython-312.pyc",
    ),
    (
        "sitecustomize/notes.txt",
        "G6",
        "interpreter start-up hook sitecustomize/notes.txt",
    ),
    (
        "json/notes.txt",
        "G6",
        "root package json/ (json/notes.txt) shadows the json module",
    ),
    (
        "pytest/notes.txt",
        "G6",
        "root package pytest/ (pytest/notes.txt) shadows the pytest module",
    ),
    (
        "notes.txt",
        "G1",
        "root entry notes.txt is outside the layout (the root holds only env/, workflows/ and "
        "unify_memory_testkit.py)",
    ),
    (
        "proposals/a.md",
        "G1",
        "root entry proposals/a.md is outside the layout (the root holds only env/, workflows/ and "
        "unify_memory_testkit.py)",
    ),
]


@pytest.mark.parametrize("path, check, reason", UNDECLARED)
def test_undeclarable_code_and_root_entries_are_refused_before_any_run(
    tmp_path,
    world,
    path,
    check,
    reason,
):
    mem, ev = world
    # what 9deefbfd1's early refusals checked (unchanged): neither refused this path
    assert layout_allowed(path) and not forbidden(path)
    runner = _Recorder(path)
    gate = _gate(tmp_path, mem, ev, runner)
    parent = mem.head()
    cand = _commit(mem, {**PROBE_BASE, path: PAYLOAD})
    res = gate.check(parent, cand, {"items": [PROBE_ITEM]})
    _refused_early(res, check, reason)
    # 9deefbfd1 ran G3's pytest here with the planted file in the tree: runner.calls held (…, True)
    assert runner.calls == []


# Declarable as ``support`` (support_allowed admits a non-``.py`` name under env/<channel>/tests/), so on
# 9deefbfd1 the probe gate merged each of these exactly as test_gate_support_helper_under_tests_is_admitted
# merges a ``.py`` helper. Now: refused before extraction, never merged.
DECLARED = [
    ("env/venmo/tests/helper.pyc", "bytecode or native code file"),
    ("env/venmo/tests/helper.pyo", "bytecode or native code file"),
    ("env/venmo/tests/rows.abi3.so", "bytecode or native code file"),
    ("env/venmo/tests/rows.pyd", "bytecode or native code file"),
    ("env/venmo/tests/rows.dll", "bytecode or native code file"),
    ("env/venmo/tests/rows.dylib", "bytecode or native code file"),
    ("env/venmo/tests/usercustomize.json", "interpreter start-up hook"),
]


@pytest.mark.parametrize("path, what", DECLARED)
def test_declared_compiled_support_is_refused_and_never_merged(
    tmp_path,
    world,
    path,
    what,
):
    mem, ev = world
    assert support_allowed(path) and layout_allowed(path) and not forbidden(path)
    runner = _Recorder(path)
    gate = _gate(tmp_path, mem, ev, runner)
    parent = mem.head()
    cand = _commit(mem, {**PROBE_BASE, path: PAYLOAD})
    man = {"items": [PROBE_ITEM], "support": [path]}
    res = gate.merge(parent, cand, man, "pi4", "incremental", "venmo", "0")
    _refused_early(res, "G6", f"{what} {path}")
    assert runner.calls == [] and mem.head() == parent
    passed, stored = ev.db.execute(
        "SELECT passed, reasons FROM passes WHERE pass_id='pi4'",
    ).fetchone()
    assert passed == 0 and MARKER not in stored
    assert json.loads(stored)[0] == f"G6: {what} {path}"


def test_preview_refuses_sourceless_bytecode_without_running_anything(
    tmp_path,
    world,
    monkeypatch,
):
    mem, ev = world

    def boom(*a, **kw):
        raise AssertionError("nothing may run")

    monkeypatch.setattr("unify.memory_v2.gate.run_plan", boom)
    gate = _gate(tmp_path, mem, ev, boom)
    tree = tmp_path / "tree"
    for rel, data in {**FILES, "env/__init__.pyc": PAYLOAD}.items():
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_bytes(data if isinstance(data, bytes) else data.encode())
    reasons = gate.preview(mem.head(), tree, MAN)
    assert reasons == ["G6: bytecode or native code file env/__init__.pyc"], reasons


def test_many_refused_paths_are_listed_up_to_ten(tmp_path, world):
    mem, ev = world
    runner = _Recorder("env/venmo/tests/h00.pyc")
    gate = _gate(tmp_path, mem, ev, runner)
    parent = mem.head()
    bad = {f"env/venmo/tests/h{i:02d}.pyc": PAYLOAD for i in range(12)}
    cand = _commit(mem, {**PROBE_BASE, **bad})
    res = gate.check(parent, cand, {"items": [PROBE_ITEM], "support": sorted(bad)})
    g6 = [r for r in res.reasons if r.startswith("G6:")]
    assert g6[:10] == [
        f"G6: bytecode or native code file {p}" for p in sorted(bad)[:10]
    ]
    assert g6[10:] == ["G6: and 2 more such paths"]
    assert runner.calls == [] and MARKER not in "\n".join(res.reasons)


@needs_bwrap
def test_a_library_with_data_fixtures_and_a_helper_still_merges(tmp_path, world):
    """No false refusal: data files and a helper under tests/, the test kit and a workflow still merge."""
    mem, ev = world
    runner = _Recorder("env/venmo/tests/rec/rows.json")
    gate = _gate(tmp_path, mem, ev, runner)
    parent = mem.head()
    support = [
        "env/venmo/tests/rec/rows.json",
        "env/venmo/tests/rec/out.tsv",
        "env/venmo/tests/venmo_fixtures.py",
    ]
    files = {
        **PROBE_BASE,
        support[0]: '[{"user_id": "u-1"}]\n',
        support[1]: "a\tb\n1\t2\n",
        support[2]: "ROWS = []\n",
    }
    cand = _commit(mem, files)
    man = {"items": [PROBE_ITEM], "support": support}
    res = gate.merge(parent, cand, man, "ok4", "incremental", "venmo", "0")
    assert res.passed, res.reasons
    assert mem.head() == cand and runner.calls


# --- the policy function itself -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path, check, fragment",
    [
        ("x.pth", "G6", "path configuration file"),
        ("env/venmo/tests/X.PYC", "G6", "bytecode or native code file"),
        ("env/venmo/tests/a.cpython-312.pyc", "G6", "bytecode or native code file"),
        (
            "env/venmo/tests/a.pypy310-pp73-x86_64-linux-gnu.so",
            "G6",
            "bytecode or native",
        ),
        ("env/venmo/tests/a.cpython-312.py", "G6", "bytecode or native code file"),
        ("workflows/__pycache__/x.md", "G6", "bytecode cache path"),
        ("env/venmo/tests/sitecustomize/d.txt", "G6", "interpreter start-up hook"),
        ("_pytest/x.txt", "G6", "shadows the _pytest module"),
        ("pluggy/x.txt", "G6", "shadows the pluggy module"),
        ("env", "G1", "outside the layout"),
        ("workflows", "G1", "outside the layout"),
        ("README.md", "G1", "outside the layout"),
        ("conftest/x.txt", "G1", "outside the layout"),
    ],
)
def test_unsafe_path_classes(path, check, fragment):
    from unify.memory_v2.manifest import unsafe_path

    got = unsafe_path(path)
    assert got is not None and got[0] == check and fragment in got[1] and path in got[1]


def test_root_allow_list_shadows_nothing_and_root_conftest_stays_forbidden():
    from unify.memory_v2.manifest import ROOT_DIRS, root_shadowed

    assert not {*ROOT_DIRS, TESTKIT.removesuffix(".py")} & root_shadowed()
    assert {"json", "pytest", "_pytest", "pluggy", "site"} <= root_shadowed()
    assert forbidden("conftest.py") and forbidden("sitecustomize.py")


def _fixture_library_paths() -> tuple[set[str], int]:
    """Every string key of every library-file dict literal in tests/memory_v2 (this file excepted)."""
    here = Path(__file__)
    keys: set[str] = set()
    dicts = 0
    for f in sorted(here.parent.rglob("*.py")):
        if f == here:
            continue
        for node in ast.walk(ast.parse(f.read_text())):
            if not isinstance(node, ast.Dict):
                continue
            ks = [
                k.value
                for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            ]
            if any(k.startswith(("env/", "workflows/")) or k == TESTKIT for k in ks):
                dicts += 1
                keys.update(ks)
    return keys, dicts


def test_every_library_fixture_is_still_admitted():
    """The new refusals reject nothing a library fixture of tests/memory_v2 holds that 9deefbfd1 admitted."""
    from unify.memory_v2.manifest import unsafe_path

    named = {**FILES, **PROBE_BASE, **BASE2, **WT_FILES}
    for p in named:
        assert layout_allowed(p) and not forbidden(p) and unsafe_path(p) is None, p
    keys, dicts = _fixture_library_paths()
    admitted = {k for k in keys if layout_allowed(k) and not forbidden(k)}
    assert dicts >= 40 and len(admitted) >= 10 and set(named) <= admitted
    # the fixtures that 9deefbfd1 already refused (bad on purpose) stay refused by the older checks
    assert [p for p in sorted(admitted) if unsafe_path(p) is not None] == []
