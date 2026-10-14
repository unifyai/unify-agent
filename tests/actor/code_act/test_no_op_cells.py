"""Symbolic: a cell that only shows text written in it is refused before it runs, with an error that teaches.

``pass``, a bare literal, ``print`` of literals only, or a comment-only cell computes nothing; the actor's
``execute_code`` refuses it (unify/actor/no_op_cells.py) with the cell as written, what is wrong, what a cell is for
with an example, and where the thought or the answer goes. Every such cell is refused alike: nothing is counted.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from unify.actor import in_code_tools, no_op_cells
from unify.common.tool_errors import ToolInputError
from unify.settings import SETTINGS

NO_OP = [
    "pass",
    "print('thinking about the grid')",
    "print(\"\")",
    "print()",
    "print('a', 'b', sep='-', end='')",
    "print(f'no placeholder here')",
    "print(-3, (1, 2), [3, 'x'])",
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
    assert no_op_cells.refusal(code, can_answer_in_code=True) is None


def test_the_refusal_has_the_four_parts():
    err = no_op_cells.refusal("print('still thinking')", can_answer_in_code=True)
    assert isinstance(err, ToolInputError)
    text = err.as_tool_result()
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
    without = no_op_cells.refusal("pass", can_answer_in_code=False).as_tool_result()
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
async def test_the_in_code_surface_refuses_the_cell_without_running_it(tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "off")
    started = []

    async def start(**kw):  # an interpreter must never be started for a refused cell
        started.append(kw)
        raise AssertionError("the interpreter was started")

    session = _session(tmp_path)
    session._start = start
    try:
        for code in ("print('step 1: look at the input')", "pass", "# thinking"):
            with pytest.raises(ToolInputError) as info:
                await session.run_cell(code)
            assert "this cell computes nothing" in info.value.as_tool_result()
            assert 'tools.answer("...")' in info.value.as_tool_result()  # the actor answers in its cells
        assert started == [] and session.interp is None
        assert session.calls == 0  # no cell number was spent
    finally:
        await session.close()
