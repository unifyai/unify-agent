"""Stage 5's check (``QAChecks.inputs_paths``): no stored file names the ``/inputs`` path, read from the files only.

``/inputs`` holds one pass's input and the actor's box has none, so library code or a test naming it is labelled
(library code on the pass, a test on the items that list it). Only changed files are read; nothing runs. The test
side is the library layout's (``memory/<package>/tests/``).
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


def _check(run: _Run) -> _Run:
    QAChecks(SimpleNamespace(), run).inputs_paths()
    return run


CLEAN_TEST = "from memory.grid.fill import fill\n\n\ndef test_fill():\n    assert fill([[0]]) == [[0]]\n"


def test_a_clean_test_and_clean_library_code_pass(tmp_path, no_box):
    run = _check(_run(tmp_path, {MODULE: "def fill(g):\n    return g\n", TEST: CLEAN_TEST}, [MODULE, TEST]))
    assert run.res.passed and run.res.reasons == [] and run.label_findings == {} and run.notes == []


def test_library_code_that_names_inputs_is_a_finding_of_the_pass(tmp_path, no_box):
    run = _check(_run(tmp_path, {MODULE: "DATA = '/inputs/blobs'\n\n\ndef fill(g):\n    return g\n"}, [MODULE]))
    assert run.res.passed  # labels only: a finding refuses nothing
    assert any(
        n.startswith(f"note: label: the pass: G3: [qa:inputs] {MODULE} line 1 names a path under /inputs outside a test")
        for n in run.notes
    ), run.notes


def test_a_test_that_names_inputs_labels_the_item_that_lists_it(tmp_path, no_box):
    test = "def test_fill():\n    open('/inputs/blobs/' + 'x')\n"
    run = _check(_run(tmp_path, {TEST: test}, [TEST]))
    assert run.res.passed
    found = run.label_findings[ITEM]
    assert any("[qa:inputs]" in f and "line 2 names a path under /inputs" in f for f in found), found


def test_a_test_helper_is_test_side_too(tmp_path, no_box):
    run = _check(_run(tmp_path, {HELPER: "ROOT = '/inputs'\n"}, [HELPER]))
    assert any(
        f"[qa:inputs] {HELPER} line 1 names a path under /inputs, which only this pass's box has" in n
        for n in run.notes
    ), run.notes


def test_an_unchanged_file_is_not_read(tmp_path, no_box):
    run = _check(_run(tmp_path, {MODULE: "DATA = '/inputs'\n"}, []))
    assert run.notes == [] and run.label_findings == {}
