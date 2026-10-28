"""Symbolic: the contract checks (tests/contract_lint.py) pass a call that meets the contract and name each gap of
one that does not, and pass only errors that teach."""

from __future__ import annotations

from typing import Literal

from tests.contract_lint import contract_gaps, error_gaps
from unify.common import contract_runtime as cr


@cr.checked(name="memory.items")
def good(topic: str | None = None, kind: Literal["function", "note"] | None = None) -> list[str]:
    """Browse the library.

    Args:
        topic: a topic or package; None for all.
        kind: only functions or only notes.

    Returns:
        The items' ids.

    Raises:
        WrongCall: topic is an item's id.

    Example:
        memory.items('notes/grid')
    """
    return []


def bad(topic, kind=None):
    """Browse."""
    return []


def test_a_call_that_meets_the_contract_has_no_gaps():
    assert contract_gaps("memory.items", good) == []


def test_each_gap_of_a_call_is_named():
    gaps = contract_gaps("memory.bad", bad)
    for part in (
        "not checked at the call",
        "parameter 'topic' has no type hint",
        "parameter 'kind' has no type hint",
        "no return type",
        "no Args section",
        "no Returns section",
        "no Raises section",
        "no Example",
    ):
        assert any(part in g for g in gaps), (part, gaps)


def test_only_errors_that_teach_pass():
    assert error_gaps(cr.NotFound("memory.get('x')", "no item named 'x'.", expected="an item id", example="memory.get('notes/a/b.md')")) == []
    assert error_gaps(cr.WrongCall("memory.items('notes/a/b.md')", "that is an item, not a topic.", hint="Open it with memory.get('notes/a/b.md').")) == []
    assert "neither what was expected" in error_gaps(cr.NotFound("memory.get('x')", "no item."))[0]
    assert any("unfilled" in g for g in error_gaps(cr.NotFound("memory.get('x')", "no {path}.", hint="Use memory.items().")))
    assert any("host path" in g for g in error_gaps(cr.NotFound("memory.get('x')", "no /home/u/x.", hint="Use memory.items().")))
    assert "not a ContractError" in error_gaps(KeyError("x"))[0]
