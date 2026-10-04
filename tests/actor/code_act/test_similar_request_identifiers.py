"""Symbolic: ``UNIFY_SIMILAR_REQUEST_IDENTIFIERS`` keeps whole identifiers as tokens.

The retrieval-matching audit (research artifact retrieval-matching-audit-v1,
5 Oct) scored request-to-request ``similar_request`` on 2,277 recorded task
starts. The tokenizer splits an id such as ``task-ddc8a32b`` into ``ddc``,
``8``, ``a``, ``32``, ``b``, pieces other ids share; keeping the whole
identifier as a token of its own cut ARC no-correct-entry lists from 25% to
10% of queries at a threshold of 0.15.
"""

from __future__ import annotations

import pytest

from unify.function_manager import task_origin
from unify.settings import ProductionSettings, SETTINGS

PREAMBLE = (
    "You are working through a stream of table puzzles. Each instance gives a "
    "puzzle id and an input table; puzzles recur with fresh tables. Reply with "
    "the output table as rows of integers on the last line of your reply."
)


def _visit(puzzle: str, table: list[list[int]]) -> str:
    rows = "\n".join(" ".join(str(v) for v in row) for row in table)
    size = f"{len(table)}x{len(table[0])}"
    return f"{PREAMBLE}\n\nNew instance. Puzzle id: {puzzle}\nInput ({size}):\n{rows}\n"


def _score(a: str, b: str, *others: str) -> float:
    a, b = task_origin.bounded_text(a), task_origin.bounded_text(b)
    weights = task_origin.token_weights([a, b, *map(task_origin.bounded_text, others)])
    return task_origin.similarity(a, b, weights)


@pytest.fixture
def identifiers(monkeypatch):
    def set_(on: bool) -> None:
        monkeypatch.setattr(SETTINGS, "UNIFY_SIMILAR_REQUEST_IDENTIFIERS", on)

    return set_


# ── whole identifiers ────────────────────────────────────────────────────


def test_an_identifier_is_also_a_token_of_its_own(identifiers):
    text = "Puzzle task-ddc8a32b, grid 3x4, ref inv_2024q3; abc12 abcdef 12345678"
    identifiers(False)
    runs = task_origin.tokens(text)
    assert "#task-ddc8a32b" not in runs and {"ddc", "8", "a", "32", "b"} <= runs
    identifiers(True)
    assert task_origin.tokens(text) == runs | {"#task-ddc8a32b", "#inv_2024q3"}


OTHER_PUZZLES = [
    _visit(puzzle, table)
    for puzzle, table in (
        ("task-55e10f3a", [[3, 3, 9], [3, 0, 4]]),
        ("task-71c2f0b9", [[2, 0, 6], [0, 2, 8], [2, 2, 1]]),
        ("task-a04d9e21", [[7, 1, 7, 9, 4]]),
        ("task-6b3e7c10", [[6, 6], [1, 2]]),
        ("task-c9d84f2e", [[8, 4, 0], [9, 5, 1]]),
        ("task-0e5a6b97", [[3, 7], [2, 6], [5, 8]]),
    )
]


def test_the_same_id_with_a_new_grid_outscores_another_id_with_a_similar_grid(
    identifiers,
):
    stored = _visit("task-ddc8a32b", [[1, 0, 2], [0, 1, 0]])
    same_id = _visit("task-ddc8a32b", [[5, 5], [0, 5], [5, 0], [7, 7]])
    # Another id made of the same letter and digit runs, and a close grid.
    other_id = _visit("task-b32a8ddc", [[1, 0, 2], [0, 2, 0]])

    def scores():
        return (
            _score(same_id, stored, other_id, *OTHER_PUZZLES),
            _score(other_id, stored, same_id, *OTHER_PUZZLES),
        )

    identifiers(False)
    same, other = scores()
    assert other > same  # the shredded id cannot tell them apart
    identifiers(True)
    same, other = scores()
    assert same > other


ASSISTANT_REQUESTS = [
    "Send the Q3 revenue report to finance.",
    "Send the Q4 revenue report to finance.",
    "Reply to the email from Jordan about the conference sponsorship and say "
    "we can do the silver tier.",
    "Text each of my tenants at 14 Elm St a reminder that October rent of "
    "$1,850 is due on the 1st.",
    "Write release notes for version 2.8 from the merged pull requests since 2.7.",
]


def test_text_without_identifiers_scores_exactly_as_shipped(identifiers):
    pairs = [(a, b) for a in ASSISTANT_REQUESTS for b in ASSISTANT_REQUESTS]
    identifiers(False)
    off = [_score(a, b, *ASSISTANT_REQUESTS) for a, b in pairs]
    identifiers(True)
    on = [_score(a, b, *ASSISTANT_REQUESTS) for a, b in pairs]
    assert on == off


def test_the_setting_defaults_off_and_parses_booleans():
    assert ProductionSettings().UNIFY_SIMILAR_REQUEST_IDENTIFIERS is False
    assert (
        ProductionSettings(
            UNIFY_SIMILAR_REQUEST_IDENTIFIERS="1",
        ).UNIFY_SIMILAR_REQUEST_IDENTIFIERS
        is True
    )
