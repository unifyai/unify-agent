"""Stage 5's kit check (``QAChecks.kit``): the test kit stays test-side, read from the files only.

Library code under ``memory/`` must not use the kit (the actor imports the library without it). A test must import
only what the kit provides, name every module it imports with a constant, and read recorded blobs through
``memlab.inputs.blob`` rather than the ``/inputs`` path. Only changed files are read; nothing runs. The test side
is the library layout's (``memory/<package>/tests/``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from unify.memory import sandbox_run
from unify.memory.gate import CHECKS, GateResult, _Run
from unify.memory.qa import QAChecks

ITEM = "memory.grid.fill:fill"
MODULE = "memory/grid/fill.py"
TEST = "memory/grid/tests/test_fill.py"
HELPER = "memory/grid/tests/helpers.py"


@pytest.fixture
def no_box(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a confined run was started")

    monkeypatch.setattr(sandbox_run, "_spawn_box", refuse)


def _run(tmp_path, files: dict[str, str], changed: list[str]) -> _Run:
    c = tmp_path / "c"
    for rel, text in files.items():
        (c / rel).parent.mkdir(parents=True, exist_ok=True)
        (c / rel).write_text(text)
    item = SimpleNamespace(item=ITEM, kind="function", tests=[TEST])
    run = _Run(GateResult(True, {k: True for k in CHECKS}), SimpleNamespace(items=[item]), {}, "p", "c", tmp_path)
    run.c_tree = c
    run.c_files = {rel: ("100644", rel) for rel in files}
    run.changed = list(changed)
    return run


def _kit(run: _Run) -> _Run:
    QAChecks(SimpleNamespace(), run).kit()
    return run


CLEAN_TEST = (
    "from memlab.inputs import blob\nfrom memory.grid.fill import fill\n\n\n"
    "def test_fill():\n    assert fill([[0]]) == [[0]]\n"
)


def test_a_clean_test_and_clean_library_code_pass(tmp_path, no_box):
    run = _kit(_run(tmp_path, {MODULE: "def fill(g):\n    return g\n", TEST: CLEAN_TEST}, [MODULE, TEST]))
    assert run.res.passed and run.res.reasons == [] and run.label_findings == {} and run.notes == []


def test_library_code_that_uses_the_kit_is_a_finding_of_the_pass(tmp_path, no_box):
    run = _kit(_run(tmp_path, {MODULE: "import memlab\n\n\ndef fill(g):\n    return g\n"}, [MODULE]))
    assert run.res.passed  # labels only: a finding refuses nothing
    assert any(
        n.startswith(f"note: label: the pass: G3: [qa:kit] {MODULE} line 1 uses the test kit") for n in run.notes
    ), run.notes


def test_a_tests_bad_imports_and_inputs_path_label_the_item_that_lists_it(tmp_path, no_box):
    test = (
        "import importlib\nfrom memlab.nowhere import thing\n\n\n"
        "def test_fill(name):\n    importlib.import_module(name)\n    open('/inputs/blobs/' + 'x')\n"
    )
    run = _kit(_run(tmp_path, {TEST: test}, [TEST]))
    found = run.label_findings[ITEM]
    assert any("[qa:kit]" in f and "line 2 imports memlab.nowhere" in f for f in found), found
    assert any("[qa:kit]" in f and "line 6 imports a module whose name is not a constant" in f for f in found), found
    assert any("[qa:kit]" in f and "line 7 names a path under /inputs" in f for f in found), found


def test_a_test_helper_is_test_side_too(tmp_path, no_box):
    run = _kit(_run(tmp_path, {HELPER: "from memlab.replay import RecordedEnv\n"}, [HELPER]))
    assert run.notes == [] and run.label_findings == {}  # a helper under tests/ may use the kit


def test_an_unchanged_file_is_not_read(tmp_path, no_box):
    run = _kit(_run(tmp_path, {MODULE: "import memlab\n"}, []))
    assert run.notes == [] and run.label_findings == {}
