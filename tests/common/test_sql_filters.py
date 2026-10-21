"""Symbolic: a clause joined with a scope cannot step outside it."""

from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unify.common.sql_filters import (
    UnsafeClauseError,
    and_clauses,
    or_clauses,
    require_self_contained,
)

ESCAPES = [
    "1=1) OR (1=1",
    "1=1 --",
    "1=1 /* */",
    "1=1; DELETE FROM functions",
    "name = 'x",
    'name = "x',
    "(1=1",
]


@pytest.mark.timeout(30)
@pytest.mark.parametrize("clause", ESCAPES)
def test_a_clause_that_escapes_its_parentheses_is_refused(clause):
    with pytest.raises(UnsafeClauseError, match="self-contained"):
        and_clauses("is_primitive = 0", clause)
    with pytest.raises(UnsafeClauseError, match="self-contained"):
        or_clauses(clause, "is_primitive = 1")


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "clause",
    [
        "name LIKE 'report_%'",
        "docstring LIKE '%(see below)%'",
        "docstring LIKE '%--%'",
        "docstring LIKE '%; then%'",
        "title = 'Owner''s runbook'",
        "\"name\" = 'a'",
        "[name] = 'a'",
        "function_id IN (1, 2) AND (name = 'a' OR name = 'b')",
    ],
)
def test_a_self_contained_clause_is_joined_unchanged(clause):
    assert require_self_contained(clause) == clause
    assert and_clauses("is_primitive = 0", clause) == (
        f"(is_primitive = 0) AND ({clause})"
    )


@_handle_project
@pytest.mark.timeout(60)
def test_a_model_filter_cannot_widen_a_scoped_manager():
    from unify.function_manager.function_manager import FunctionManager

    FunctionManager(include_primitives=False).add_functions(
        implementations=[
            'def report_total(xs):\n    """Sum."""\n    return sum(xs)\n',
            'def wipe_store():\n    """Delete."""\n    return None\n',
        ],
    )
    scoped = FunctionManager(
        include_primitives=False,
        filter_scope="name LIKE 'report_%'",
    )
    assert [row["name"] for row in scoped.filter_functions(filter="1=1")] == [
        "report_total",
    ]
    escaped = scoped.filter_functions(filter="1=1) OR (1=1")
    assert isinstance(escaped, dict)
    assert escaped["error_kind"] == "invalid_filter"
    assert "self-contained" in escaped["message"]
