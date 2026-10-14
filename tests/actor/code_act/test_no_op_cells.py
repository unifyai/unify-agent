"""Symbolic: a cell that only shows text written in it is not run; its result is an error that teaches.

``pass``, a bare constant, ``print`` of constant text only, or a comment-only cell computes nothing (the loop stop's
definition). The actor's ``execute_code`` does not run it (unify/actor/no_op_cells.py); the cell's result is the
cell as written, what is wrong, what a cell is for with an example, and where the thought or the answer goes. It is a
result, not a raised refusal: nothing counts such cells or stops on them.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from unify.actor import in_code_tools, no_op_cells
from unify.settings import SETTINGS

NO_OP = [
    "pass",
    "print('thinking about the grid')",
    "print(\"\")",
    "print()",
    "print('a', 'b', sep='-', end='')",
    "print(f'no placeholder here')",
    "'a bare string'",
    "# only a comment\n# and another",
    "print('one')\nprint('two')\npass",
]

COMPUTES = [
    "print(x)",
    "print(f'{x}')",
    "x = 1",
    "print(len(rows))",
    "tools.answer('42')",
    "import os",
    "print('rows', rows)",
    "print('text', file=sys.stderr)",
    "print('start')\nresult = solve(data)",
]


@pytest.mark.parametrize("code", NO_OP)
def test_a_cell_that_only_shows_written_text_is_a_no_op(code):
    assert no_op_cells.no_op(code)


@pytest.mark.parametrize("code", COMPUTES)
def test_a_cell_that_reads_or_computes_anything_is_not(code):
    assert not no_op_cells.no_op(code)


@pytest.mark.parametrize("code", ["", "   \n", "print('unclosed", None, 3])
def test_empty_code_and_code_that_does_not_parse_are_not_judged(code):
    assert not no_op_cells.no_op(code)
    assert no_op_cells.refusal_text(code, can_answer_in_code=True) is None


def test_the_error_has_the_four_parts():
    text = no_op_cells.refusal_text("print('still thinking')", can_answer_in_code=True)
    first, expected, hint = text.split("\n")
    # the cell as written, then what is wrong
    assert first.startswith("execute_code(code=\"print('still thinking')\"): this cell computes nothing")
    assert "It was not run." in first
    # what was expected, with an example
    assert expected.startswith("Expected a cell that computes something") and "e.g. execute_code(code=" in expected
    # where the thought and the answer go
    assert hint == (
        'Write your thinking in your reply, not in a cell. To answer, call tools.answer("...") in a cell, or '
        "reply with the answer as text."
    )
    without = no_op_cells.refusal_text("pass", can_answer_in_code=False)
    assert without.endswith("To answer, reply with the answer as text.") and "tools.answer" not in without


def _session(tmp_path):
    return in_code_tools.InCodeTools(
        agent="actor",
        name="root",
        record=SimpleNamespace(),
        workspace_root=str(tmp_path),
        shadow={},
        cell_limit_s=None,
        events_folder=tmp_path / "events",
    )


@pytest.mark.asyncio
async def test_the_in_code_surface_answers_the_cell_without_running_it(tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "off")
    started = []

    async def start(**kw):  # an interpreter must never be started for such a cell
        started.append(kw)
        raise AssertionError("the interpreter was started")

    session = _session(tmp_path)
    session._start = start
    cells = ("print('step 1: look at the input')", "pass", "# thinking")
    try:
        for code in cells:
            out = await session.run_cell(code)
            assert "this cell computes nothing" in out.error
            assert 'tools.answer("...")' in out.error  # the actor answers in its cells
            assert out.stdout == []
        assert started == [] and session.interp is None
    finally:
        await session.close()
    path = tmp_path / "events" / "tool-events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    cell_rows = [r for r in rows if r.get("tool") == "execute_code"]
    # each such cell keeps its number and its row, marked as not run
    assert [r["call"] for r in cell_rows] == [0, 1, 2]
    assert all(r["refusal"] == no_op_cells.REFUSAL and r["error"] is True for r in cell_rows), cell_rows


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_full_surface_answers_the_cell_without_running_it():
    from unify.actor.code_act_actor import CodeActActor

    actor = CodeActActor()
    try:
        execute_code = actor.get_tools("act")["execute_code"]
        out = await execute_code("think", "print('the next step is to compare the rows')")
        assert isinstance(out, dict) and out["stdout"] == "" and out["result"] is None
        assert "this cell computes nothing" in out["error"]
        assert out["error"].endswith("To answer, reply with the answer as text.")
        ran = await execute_code("count", "rows = [1, 2, 3]\nprint(len(rows))")
        err = ran.get("error") if isinstance(ran, dict) else getattr(ran, "error", None)
        assert err is None  # a cell that computes runs as before
    finally:
        await actor.close()
