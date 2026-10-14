"""A cell that only shows text written in it is not run: its result is an error that teaches what to do instead.

Such a cell (``pass``, a bare constant, ``print`` of constant text only, or comments only: the loop stop's own
definition, :func:`unify.common._async_tool.loop_stop.is_noop_cell`) computes nothing, so its output can tell the
model nothing it did not write itself. Recorded runs show long chains of them carrying the model's reasoning from
call to call. The tool does not run such a cell; the cell's result is the error with the four parts of an error that
teaches (the cell as written, what is wrong, what a cell is for with an example, and where the thought or the answer
goes instead). It is the cell's result, not a raised refusal, so nothing new counts or stops anything: each such
cell gets the same answer, and the loop's own rules (the loop stop included) see the cell as before.

Code that does not parse is not judged here (it runs, and its own error teaches). A cell with no code at all is not
judged either (a surface may give it a meaning of its own).
"""

from __future__ import annotations

from typing import Optional

from unify.common._async_tool.loop_stop import is_noop_cell
from unify.common.contract_runtime import call_text, render

__all__ = ["REFUSAL", "no_op", "refusal_text"]

#: The ``refusal`` an event row of such a cell carries.
REFUSAL = "no_op_cell"


def no_op(code: object) -> bool:
    """Whether *code* only shows text written in it (:func:`~unify.common._async_tool.loop_stop.is_noop_cell`).
    Empty code and code that does not parse are not judged."""
    if not isinstance(code, str) or not code.strip():
        return False
    try:
        return is_noop_cell(code)
    except ValueError:  # a null byte: the interpreter's own error says so
        return False


def refusal_text(code: object, *, can_answer_in_code: bool) -> Optional[str]:
    """The error a no-op cell (:func:`no_op`) gets in place of running, else None. *can_answer_in_code*: the agent
    has ``tools.answer`` in its cells."""
    if not no_op(code):
        return None
    answer = (
        'To answer, call tools.answer("...") in a cell, or reply with the answer as text.'
        if can_answer_in_code
        else "To answer, reply with the answer as text."
    )
    return render(
        call_text("execute_code", kwargs={"code": code}),
        "this cell computes nothing: it only shows text written in it, so its output cannot tell you anything you "
        "did not write. It was not run.",
        expected="a cell that computes something: it reads your data or variables, calls a tool or writes a file",
        example="execute_code(code=\"rows = [r for r in data if r]\\nprint(len(rows))\")",
        hint=f"Think in the text of your message beside a cell: it stays in the conversation and does not end your "
        f"work. {answer}",
    )
