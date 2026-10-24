"""The function an accepted answer was made with, stored as the request's procedure (rev 2.3).

When a request's accepted answer was made by ``tools.answer_from(fn)``, the harness builds ``fn``'s closure from the
recorded cells by AST nodes only (its ``def``, the ``def``/``class``/``import`` nodes its names need, plain-data
constants read by ``ast.literal_eval``), never copying a whole cell and never running one. The closure fails, and
nothing is stored, when it reaches a harness name or imports a harness module (``fn``'s own body included), or reaches
a name bound to anything else. The stored procedure reads the declared input as data; the capture step takes it in
place of the fork's, which is told to write none; the card says how it runs and how it was checked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.memory.integration.test_request import _abort, _begin, mv2  # noqa: F401
from tests.memory.test_captures import _ep, _staging
from unify.memory import captures as cp
from unify.memory import closure
from unify.memory.integration import fork
from unify.memory.integration import request as RQ

CELLS = [
    "import json\nfrom collections import Counter as Cnt\nSCALE = 3\nLIMITS = {'low': [0, 1], 'high': None}\n",
    "def helper(row):\n    return [SCALE * v for v in row]\n",
    "class Grid:\n    def __init__(self, rows):\n        self.rows = rows\n",
    "def solve(value):\n    g = Grid(value['rows'])\n    Cnt(json.dumps(g.rows))\n    return [helper(r) for r in g.rows]\n",
    "print(solve({'rows': [[1]]}))\nnotes = open('/workspace/x')\n",
]


def _run(source: str, value):
    ns: dict = {}
    exec(compile(source, "procedure.py", "exec"), ns)  # the test's own fixture code, never an agent's
    return ns["run"](ns["read_input"](value))


# --- the closure ----------------------------------------------------------------------------------------------------


def test_the_closure_holds_what_fn_needs_and_nothing_else_and_runs_as_fn():
    source = closure.build(CELLS, "solve")
    assert "def solve(value)" in source and "def helper(row)" in source and "class Grid" in source
    assert "import json" in source and "Counter as Cnt" in source and "SCALE = 3" in source
    assert "print(" not in source and "notes" not in source and "LIMITS" not in source  # no whole cell copied
    assert _run(source, {"rows": [[1, 2]]}) == [[3, 6]]
    from unify.memory import capture_trace

    assert capture_trace.safety(source) is None  # the capture step's checks pass


def test_the_last_binding_of_each_name_is_the_one_kept():
    cells = ["def f(v):\n    return 1\n", "def f(v):\n    return K\n", "K = [1, 'a', None]\n"]
    assert _run(closure.build(cells, "f"), 0) == [1, "a", None]


@pytest.mark.parametrize(
    "cells, why",
    [
        (["def f(v):\n    return tools.read(v)\n"], "reaches tools"),
        (["def f(v):\n    return memory.get('x')(v)\n"], "reaches memory"),
        (["def f(v):\n    return query_llm(v)\n"], "reaches query_llm"),
        (["def f(v):\n    return list_llms()\n"], "reaches list_llms"),
        (["def f(v):\n    import unillm\n    return v\n"], "imports unillm"),
        (["import unify.memory.librarian as L\ndef f(v):\n    return L.x(v)\n"], "imports unify.memory.librarian"),
        (["from unify import settings\ndef f(v):\n    return settings\n"], "imports unify"),
        (["import tools as t\ndef f(v):\n    return t.read(v)\n"], "imports tools"),
        (["def g(v):\n    from memory import get\n    return get(v)\ndef f(v):\n    return g(v)\n"], "imports memory"),
        (["h = open('a')\ndef f(v):\n    return h.read()\n"], "something other than"),
        (["S = {1, 2}\ndef f(v):\n    return S\n"], "something other than"),  # a set is not plain data
        (["K = 1\nK += 1\ndef f(v):\n    return K\n"], "something other than"),
        (["import json\nm = json\ndef f(v):\n    return m.dumps(v)\n"], "something other than"),  # a module object
        (["def f(v):\n    return missing(v)\n"], "do not bind"),
        (["f = lambda v: v\n"], "not bound by a def"),
        (["def run(v):\n    return v\n"], "can be stored"),
    ],
)
def test_a_closure_that_reaches_the_harness_or_anything_but_plain_definitions_fails(cells, why):
    with pytest.raises(closure.ClosureError) as got:
        closure.build(cells, "run" if "def run" in cells[0] else "f")
    assert why in str(got.value), str(got.value)


# --- the store at the request's end ---------------------------------------------------------------------------------


def _answered(run, fn="solve", item=None, answered=True):
    run.note_answered_from({"item": item, "fn": fn, "source": "declared", "input_digest": "a", "output_digest": "b",
                            "input": {"rows": [[1, 2]]}, "output": [[3, 6]], "kept": True, "answered": answered,
                            "checked": {"passed": 2, "total": 2}, "withheld": False})  # fmt: skip


def _ep_with(cells: list[str]):
    from types import SimpleNamespace

    from unify.memory.episodes import Cell

    return SimpleNamespace(episode_id="20261024T000000Z-ab12cd34", cells=[Cell(i, c, "", None) for i, c in
                                                                          enumerate(cells)])  # fmt: skip


@pytest.mark.parametrize(
    "verdict, item, stored",
    [("pass", None, True), ("fail", None, False), (None, None, False), ("pass", "capture:00112233445566aa", False)],
)
def test_only_an_accepted_answer_from_a_session_function_is_stored(mv2, monkeypatch, verdict, item, stored):  # noqa: F811
    run = _begin(mv2, "the request")
    try:
        monkeypatch.setattr(run, "_observations_now", lambda: 1)
        _answered(run, item=item)
        if verdict:
            run.take_checker({"label": verdict})
        ep = _ep_with(CELLS)
        run._store_answered(ep)
        d = Path(run.paths.state_dir) / RQ.ANSWERED_DIR / ep.episode_id
        assert run.stored_procedure is stored and (d / "procedure.py").exists() is stored
        if stored:
            doc = json.loads((d / "answered.json").read_text())
            assert doc == {"fn": "solve", "checked": {"passed": 2, "total": 2}, "input_digest": "a",
                           "input": {"rows": [[1, 2]]}}  # fmt: skip
            assert run.answered_from[-1]["stored"] is True
    finally:
        _abort(mv2, run)


def test_a_closure_that_fails_stores_nothing_and_says_why(mv2, monkeypatch):  # noqa: F811
    run = _begin(mv2, "the request")
    try:
        monkeypatch.setattr(run, "_observations_now", lambda: 1)
        _answered(run, fn="f")
        run.take_checker({"label": "pass"})
        run._store_answered(_ep_with(["def f(v):\n    return tools.read(v)\n"]))
        assert run.stored_procedure is False
        assert run.answered_from[-1]["stored"] is False and "reaches tools" in run.answered_from[-1]["why"]
    finally:
        _abort(mv2, run)


def test_the_fork_is_told_to_write_no_procedure_only_when_one_was_stored():
    assert fork.stored_instruction().startswith(fork.INSTRUCTION)
    assert fork.STORED_PROCEDURE_LINE in fork.stored_instruction()
    assert fork.STORED_PROCEDURE_LINE not in fork.fork_instruction()


# --- the capture step and the card -----------------------------------------------------------------------------------


def test_the_capture_step_takes_the_stored_procedure_in_place_of_the_forks(tmp_path):
    eid = "20261024T000000Z-ab12cd34"
    source = closure.build(CELLS, "solve")
    d = tmp_path / "memory-answered" / eid
    d.mkdir(parents=True)
    (d / "procedure.py").write_text(source)
    (d / "answered.json").write_text(json.dumps({"fn": "solve", "checked": {"passed": 2, "total": 2},
                                                 "input": {"rows": [[1, 2]]}}))  # fmt: skip
    staging = _staging(tmp_path, eid, {"procedure.py": "def run(state):\n    return 'the fork wrote this'\n"})
    store = cp.store_dir(tmp_path)
    row = cp.capture_trace_step(staging, _ep(eid), store)
    assert row["answered"] is True and row["safety"] == "ok"
    cid = row["capture_id"]
    assert (store / cid / "procedure.py").read_text() == source
    rec = json.loads((store / cid / "record.json").read_text())
    assert rec["reads"] == "declared_input" and rec["checked"] == {"passed": 2, "total": 2}
    sig = json.loads((store / cid / "signature.json").read_text())
    assert sig["input"] == cp.shape({"rows": [[1, 2]]})


def test_without_a_stored_procedure_the_capture_step_is_unchanged(tmp_path):
    eid = "20261024T000000Z-ab12cd35"
    staging = _staging(tmp_path, eid, {"procedure.py": "def run(state):\n    return 1\n"})
    row = cp.capture_trace_step(staging, _ep(eid), cp.store_dir(tmp_path))
    assert row["answered"] is False and row["safety"] == "ok"


def test_the_card_says_it_runs_on_the_declared_input_and_how_it_was_checked():
    from unify.memory import library_helper as LH

    lines = LH._capture_lines("capture:00112233445566aa", {"reads": "declared_input",
                                                           "checked": {"passed": 3, "total": 3}})  # fmt: skip
    assert any("runs it on the input the current request declares" in ln for ln in lines)
    assert "Checked on 3 of 3 examples when it was made." in lines
    lines = LH._capture_lines("capture:00112233445566aa", {"reads": "declared_input",
                                                           "checked": {"passed": 0, "total": 0}})  # fmt: skip
    assert not any(ln.startswith("Checked on") for ln in lines)


def test_a_no_argument_call_passes_the_declared_input_or_says_to_pass_one(monkeypatch):
    from unify.memory import library_helper as LH

    called = []

    class Target:
        id = "capture:00112233445566aa"

        def __call__(self, value):
            called.append(value)
            return value

    item = LH.Item("capture:00112233445566aa", "function", "s", "", "unchecked", "on-trial", "", "", (), ())
    monkeypatch.setattr(LH.Item, "_on_request", lambda self: Target())
    monkeypatch.setattr(LH, "_capture_form", lambda cid: "value")
    monkeypatch.setattr(LH, "_reads_declared", lambda iid: True)
    monkeypatch.setattr(LH, "_declared_value", lambda: (True, {"rows": [[1]]}))
    assert item() == {"rows": [[1]]} and called == [{"rows": [[1]]}]
    monkeypatch.setattr(LH, "_declared_value", lambda: (False, None))
    with pytest.raises(LH.WrongCall) as got:
        item()
    assert "declares none" in str(got.value) and "(<input>)" in str(got.value)
