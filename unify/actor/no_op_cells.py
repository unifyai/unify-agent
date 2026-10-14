"""A cell that only shows text written in it: refused before it runs, with an error that teaches what to do instead.

Such a cell (``pass``, a bare literal, ``print`` of literals only, or comments only) computes nothing, so its output
can tell the model nothing it did not write itself. Recorded runs show long chains of them carrying the model's
reasoning from call to call. The tool refuses each one alike, with the four parts of an error that teaches (the cell
as written, what is wrong, what a cell is for with an example, and where the thought or the answer goes instead).
Nothing is counted or capped. Code that does not parse is not judged here: it runs, and its own error teaches. A cell
with no code at all is not judged either (a surface may give it a meaning of its own).
"""

from __future__ import annotations

import ast
from typing import Optional

from unify.common.contract_runtime import call_text, render
from unify.common.tool_errors import ToolInputError

__all__ = ["no_op", "refusal"]


def _literal(node: ast.AST) -> bool:
    """A value written in the cell: a constant, an f-string with no placeholder, a signed number, or a tuple, list
    or set of those."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.JoinedStr):
        return all(isinstance(v, ast.Constant) for v in node.values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _literal(node.operand)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_literal(e) for e in node.elts)
    return False


def _shows_only_written_text(stmt: ast.stmt) -> bool:
    if isinstance(stmt, ast.Pass):
        return True
    if isinstance(stmt, ast.Expr):
        value = stmt.value
        if _literal(value):
            return True
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "print":
            return all(_literal(a) for a in value.args) and all(
                k.arg is not None and _literal(k.value) for k in value.keywords
            )
    return False


def no_op(code: object) -> bool:
    """Whether *code* only shows text written in it: every statement is ``pass``, a literal or a ``print`` of
    literals, or it holds no statement (comments only). Empty code and code that does not parse are not judged."""
    if not isinstance(code, str) or not code.strip():
        return False
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    return all(_shows_only_written_text(s) for s in tree.body)


def refusal(code: object, *, can_answer_in_code: bool) -> Optional[ToolInputError]:
    """The refusal for *code* when it is a no-op cell (:func:`no_op`), else None. *can_answer_in_code*: the agent
    has ``tools.answer`` in its cells."""
    if not no_op(code):
        return None
    answer = (
        'To answer, call tools.answer("...") in a cell, or reply with the answer as text.'
        if can_answer_in_code
        else "To answer, reply with the answer as text."
    )
    return ToolInputError(
        render(
            call_text("execute_code", kwargs={"code": code}),
            "this cell computes nothing: it only shows text written in it, so its output cannot tell you anything "
            "you did not write. It was not run.",
            expected="a cell that computes something: it reads your data or variables, calls a tool or writes a file",
            example="execute_code(code=\"rows = [r for r in data if r]\\nprint(len(rows))\")",
            hint=f"Write your thinking in your reply, not in a cell. {answer}",
        ),
    )
