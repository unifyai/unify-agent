"""Symbolic: ``UNIFY_FUNCTION_PATCH`` fixes a stored function by one exact excerpt.

As shipped a stored function changes only by resending its whole source with
``add_functions(overwrite=True)``, which also resets its precondition and
dependencies to whatever that call passes. ``patch_function(name, old, new,
why)`` replaces ``old`` -- which must occur exactly once in the current
source -- keeps the precondition and dependencies, and stores the result
through ``add_functions(overwrite=True)``, so the store check and the verify
gate apply unchanged. The replaced version goes to ``function_history`` with
``why``. With the switch off the method refuses and changes nothing. No model
is called.
"""

from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unify import db
from unify.common.exact_patch import PatchRefused, apply_once, occurrences
from unify.function_manager.function_manager import (
    DEFAULT_OVERWRITE_REASON,
    FunctionManager,
)
from unify.settings import SETTINGS

SRC = (
    "def total_minor(rows: list) -> int:\n"
    '    """Return the sum of `amount` over the rows, in minor units."""\n'
    "    total = 0\n"
    "    for row in rows:\n"
    "        total += row['amount']\n"
    "    return total\n"
)


def _FM() -> FunctionManager:
    return FunctionManager(include_primitives=False)


def _stored(name: str) -> dict:
    row = db.query_one("SELECT * FROM functions WHERE name = ?", (name,))
    return db.decode(dict(row), db.FUNCTION_JSON_COLUMNS)


def _history_mark() -> int:
    return int(
        db.query_one("SELECT MAX(history_id) AS m FROM function_history")["m"] or 0,
    )


def _history_since(mark: int) -> list[dict]:
    rows = db.query(
        "SELECT * FROM function_history WHERE history_id > ? ORDER BY history_id",
        (mark,),
    )
    for row in rows:
        row["previous"] = db.loads(row["previous"])
    return rows


@pytest.fixture
def patch_on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", True)


# --------------------------------------------------------------------------- #
#  The exact-excerpt rule                                                      #
# --------------------------------------------------------------------------- #


def test_one_occurrence_is_replaced():
    assert apply_once("a = 1\nb = 2\n", "b = 2", "b = 3", what="x") == "a = 1\nb = 3\n"
    assert apply_once("keep\ndrop\n", "drop\n", "", what="x") == "keep\n"


def test_zero_occurrences_are_refused_with_the_closest_line():
    with pytest.raises(PatchRefused) as exc:
        apply_once(SRC, "total += row['amt']", "x", what="the source of 'f'")
    message = str(exc.value)
    assert message.startswith("`old` occurs 0 times in the source of 'f'")
    assert "nothing was changed" in message
    assert "The closest line is 5:" in message
    assert "   5 |         total += row['amount']" in message
    assert "whitespace is ignored" not in message


def test_zero_occurrences_say_when_only_whitespace_differs():
    with pytest.raises(PatchRefused, match="occurs 0 times") as exc:
        apply_once(SRC, "total  +=  row['amount']", "x", what="w")
    assert "It does match when whitespace is ignored" in str(exc.value)


def test_several_occurrences_are_refused_with_where_they_are():
    text = "x = 1\ny = 2\nx = 1\nz = 3\nx = 1\nx = 1\n"
    with pytest.raises(PatchRefused) as exc:
        apply_once(text, "x = 1", "x = 9", what="t")
    message = str(exc.value)
    assert message.startswith("`old` occurs 4 times in t (at lines 1, 3, 5, …)")
    assert "   3 | x = 1" in message
    assert message.count("----") == 2  # three excerpts at most


def test_overlapping_occurrences_count():
    assert occurrences("aaa", "aa") == [0, 1]
    with pytest.raises(PatchRefused, match="occurs 2 times"):
        apply_once("aaa", "aa", "b", what="t")


@pytest.mark.parametrize(
    "old, new, match",
    [("", "x", "non-empty"), ("same", "same", "nothing to change")],
)
def test_empty_or_no_op_patches_are_refused(old, new, match):
    with pytest.raises(PatchRefused, match=match):
        apply_once("same text", old, new, what="t")


# --------------------------------------------------------------------------- #
#  patch_function                                                              #
# --------------------------------------------------------------------------- #


@_handle_project
def test_a_unique_excerpt_is_patched_in_place_with_history(patch_on):
    fm = _FM()
    fm.add_functions(implementations=SRC)
    before = _stored("total_minor")
    mark = _history_mark()

    out = fm.patch_function(
        name="total_minor",
        old="        total += row['amount']\n",
        new="        total += int(row.get('amount', 0))\n",
        why="rows without an amount raised KeyError",
    )
    assert out == {
        "name": "total_minor",
        "status": "patched",
        "function_id": before["function_id"],
    }
    after = _stored("total_minor")
    assert after["function_id"] == before["function_id"]
    assert after["implementation"] == SRC.replace(
        "row['amount']",
        "int(row.get('amount', 0))",
    )
    assert after["created_at"] == before["created_at"]
    rows = _history_since(mark)
    assert len(rows) == 1
    assert rows[0]["reason"] == "rows without an amount raised KeyError"
    assert rows[0]["previous"]["implementation"] == SRC
    # The patched function runs.
    namespace: dict = {}
    exec(after["implementation"], namespace)
    assert namespace["total_minor"]([{"amount": 2}, {}]) == 2


@_handle_project
@pytest.mark.parametrize(
    "old, count",
    [("total += row['amt']", "0 times"), ("total", "4 times")],
)
def test_zero_or_several_matches_change_nothing(patch_on, old, count):
    fm = _FM()
    fm.add_functions(implementations=SRC)
    mark = _history_mark()
    out = fm.patch_function(name="total_minor", old=old, new="x", why="w")
    assert set(out) == {"name", "error"}
    assert f"`old` occurs {count} in the source of 'total_minor'" in out["error"]
    assert " | " in out["error"]  # an excerpt is shown
    assert _stored("total_minor")["implementation"] == SRC
    assert _history_since(mark) == []


@_handle_project
def test_precondition_and_dependencies_are_carried_over(patch_on):
    fm = _FM()
    fm.add_functions(
        implementations=SRC,
        preconditions={"total_minor": {"requires": "rows fetched"}},
        dependencies=["tabulate>=0.9"],
    )
    fm.patch_function(
        name="total_minor",
        old="    total = 0\n",
        new="    total = 0  # minor units\n",
        why="note the unit",
    )
    after = _stored("total_minor")
    assert after["precondition"] == {"requires": "rows fetched"}
    assert after["dependencies"] == ["tabulate>=0.9"]
    # Without the carry-over, the shipped overwrite resets both.
    fm.add_functions(implementations=SRC, overwrite=True)
    reset = _stored("total_minor")
    assert reset["precondition"] is None
    assert reset["dependencies"] == []


@_handle_project
def test_guidance_links_survive_a_patch(patch_on):
    from unify.guidance_manager.guidance_manager import GuidanceManager

    fm = _FM()
    fm.add_functions(implementations=SRC)
    fid = _stored("total_minor")["function_id"]
    gid = GuidanceManager().add_guidance(
        title="Totals",
        content="Use total_minor.",
        function_ids=[fid],
    )["details"]["guidance_id"]
    fm.patch_function(name="total_minor", old="total = 0", new="total = 0 ", why="w")
    assert fm.list_functions()["total_minor"]["guidance_ids"] == [gid]


@_handle_project
def test_the_store_check_still_refuses_a_bad_patch(patch_on, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_CHECK", "resolve")
    fm = _FM()
    fm.add_functions(implementations=SRC)
    mark = _history_mark()
    out = fm.patch_function(
        name="total_minor",
        old="row['amount']",
        new="to_minor(row['amount'])",
        why="convert units",
    )
    assert "error" in out
    assert "'total_minor' was not stored" in out["error"]
    assert "`to_minor`" in out["error"]
    assert _stored("total_minor")["implementation"] == SRC
    assert _history_since(mark) == []


@_handle_project
def test_the_verify_gate_sees_the_patched_source(patch_on, monkeypatch):
    fm = _FM()
    fm.add_functions(implementations=SRC)
    seen: list[str] = []

    def gate(*, name, node, source):
        seen.append(source)
        raise ValueError(f"'{name}' was not stored, because it has not passed")

    monkeypatch.setattr(
        FunctionManager,
        "_store_verify_enabled",
        staticmethod(lambda: True),
    )
    monkeypatch.setattr(fm, "_store_verify_gate", gate)
    out = fm.patch_function(
        name="total_minor",
        old="total = 0",
        new="total = 00",
        why="w",
    )
    assert out["error"].startswith("'total_minor' was not stored")
    assert seen == [SRC.replace("total = 0", "total = 00")]
    assert _stored("total_minor")["implementation"] == SRC


@_handle_project
@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"name": "missing"}, "no stored function is named 'missing'"),
        ({"why": "  "}, "say `why`"),
        (
            {"old": "def total_minor(", "new": "def grand_total("},
            "renames the function to 'grand_total'",
        ),
        ({"old": "    return total\n", "new": "  return total\n"}, "not one function"),
    ],
)
def test_other_refusals_change_nothing(patch_on, kwargs, match):
    fm = _FM()
    fm.add_functions(implementations=SRC)
    mark = _history_mark()
    call = {"name": "total_minor", "old": "total = 0", "new": "total = 1", "why": "w"}
    out = fm.patch_function(**{**call, **kwargs})
    assert match in out["error"]
    assert _stored("total_minor")["implementation"] == SRC
    assert set(fm.list_functions()) == {"total_minor"}
    assert _history_since(mark) == []


@_handle_project
def test_off_the_method_refuses_and_changes_nothing(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", False)
    fm = _FM()
    fm.add_functions(implementations=SRC)
    mark = _history_mark()
    out = fm.patch_function(
        name="total_minor",
        old="total = 0",
        new="total = 1",
        why="w",
    )
    assert "UNIFY_FUNCTION_PATCH is off" in out["error"]
    assert _stored("total_minor")["implementation"] == SRC
    assert _history_since(mark) == []


@_handle_project
def test_a_patch_reason_differs_from_a_plain_overwrite(patch_on):
    fm = _FM()
    fm.add_functions(implementations=SRC)
    mark = _history_mark()
    fm.patch_function(name="total_minor", old="total = 0", new="total = 1", why="w1")
    fm.add_functions(implementations=SRC, overwrite=True)
    reasons = [row["reason"] for row in _history_since(mark)]
    assert reasons == ["w1", DEFAULT_OVERWRITE_REASON]
