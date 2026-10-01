"""Symbolic: ``UNIFY_FUNCTION_PATCH`` fixes a stored function by replacing excerpts.

As shipped a stored function changes only by resending its whole source with
``add_functions(overwrite=True)``, which also resets its precondition and
dependencies to whatever that call passes. ``patch_function(name, old, new,
why)`` -- or ``edits=[{old, new}, ...]`` for several changes, applied in
order and all or none -- replaces each ``old``, which must match once (the
matching ladder is tested in ``tests/common/test_patch_ladder.py``), keeps
the precondition and dependencies, checks the result parses, and stores it
through ``add_functions(overwrite=True)``, so the store check and the verify
gate apply unchanged. The replaced version goes to ``function_history`` with
``why``, one row per call. With the switch off the method refuses and
changes nothing. No model is called.
"""

from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unify import db
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
        "edits": [{"match": "exact", "replaced": 1}],
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
        (
            {"old": "    return total\n", "new": "  return total\n"},
            "not one function that parses, so nothing was changed: line 6: "
            "unindent does not match",
        ),
        (
            {"old": "def total_minor(", "new": "x = 1\ndef total_minor("},
            "not one function, so nothing was changed",
        ),
        (
            {"old": None, "new": None, "edits": [{"old": "total = 0"}]},
            "`new` is missing",
        ),
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


# --------------------------------------------------------------------------- #
#  Batches, the matching ladder and the argument aliases                       #
# --------------------------------------------------------------------------- #


@_handle_project
def test_a_batch_is_stored_once_with_one_history_row(patch_on):
    fm = _FM()
    fm.add_functions(implementations=SRC)
    before = _stored("total_minor")
    mark = _history_mark()
    out = fm.patch_function(
        name="total_minor",
        why="skip rows without an amount and report in whole units",
        edits=[
            {
                "old": "        total += row['amount']\n",
                "new": "        total += row.get('amount', 0)\n",
            },
            {"old": "row.get('amount', 0)", "new": "int(row.get('amount', 0))"},
            {"old": "    return total\n", "new": "    return total // 100\n"},
        ],
    )
    assert out == {
        "name": "total_minor",
        "status": "patched",
        "function_id": before["function_id"],
        "edits": [{"match": "exact", "replaced": 1}] * 3,
    }
    after = _stored("total_minor")["implementation"]
    assert after == SRC.replace(
        "row['amount']",
        "int(row.get('amount', 0))",
    ).replace("return total", "return total // 100")
    rows = _history_since(mark)
    assert len(rows) == 1
    assert rows[0]["previous"]["implementation"] == SRC
    assert rows[0]["reason"] == "skip rows without an amount and report in whole units"
    namespace: dict = {}
    exec(after, namespace)
    assert namespace["total_minor"]([{"amount": 250}, {}]) == 2


@_handle_project
def test_when_edit_two_of_three_fails_nothing_is_stored(patch_on):
    fm = _FM()
    fm.add_functions(implementations=SRC)
    mark = _history_mark()
    out = fm.patch_function(
        name="total_minor",
        why="w",
        edits=[
            {"old": "total = 0", "new": "total = 1"},
            {"old": "row['amt']", "new": "row['amount']"},
            {"old": "return total", "new": "return -total"},
        ],
    )
    assert out["error"].startswith(
        "Edit 2 of 3 (matched in the text as edit 1 left it; no edit was kept): "
        "`old` occurs 0 times in the source of 'total_minor'",
    )
    assert "The closest text is line 5" in out["error"]
    assert _stored("total_minor")["implementation"] == SRC
    assert _history_since(mark) == []


@_handle_project
def test_a_batch_whose_result_does_not_parse_is_refused(patch_on):
    fm = _FM()
    fm.add_functions(implementations=SRC)
    mark = _history_mark()
    out = fm.patch_function(
        name="total_minor",
        why="w",
        edits=[
            {"old": "    for row in rows:\n", "new": "    for row in rows\n"},
            {"old": "total = 0", "new": "total = 1"},
        ],
    )
    assert out["error"].startswith(
        "the patched source is not one function that parses, so nothing was "
        "changed: line 4:",
    )
    assert "   4 |     for row in rows" in out["error"]
    assert _stored("total_minor")["implementation"] == SRC
    assert _history_since(mark) == []


@_handle_project
def test_the_stage_one_whitespace_miss_now_patches(patch_on):
    """The one failed Stage 1 patch: `old` copied with the wrong indentation."""
    fm = _FM()
    fm.add_functions(implementations=SRC)
    out = fm.patch_function(
        name="total_minor",
        # Copied without its indentation, as a model quoting the lines does.
        old="for row in rows:\n    total += row['amount']\n",
        new="for row in rows:\n    if row.get('amount') is None:\n        continue\n    total += row['amount']\n",
        why="rows without an amount raised KeyError",
    )
    assert out["status"] == "patched"
    assert out["edits"] == [{"match": "indentation", "replaced": 1}]
    after = _stored("total_minor")["implementation"]
    assert after == SRC.replace(
        "    for row in rows:\n",
        "    for row in rows:\n"
        "        if row.get('amount') is None:\n"
        "            continue\n",
    )
    namespace: dict = {}
    exec(after, namespace)
    assert namespace["total_minor"]([{"amount": 2}, {}]) == 2


PICK = (
    "def pick(rows: list) -> list:\n"
    '    """Keep the truthy rows, twice over when there are any."""\n'
    "    out = []\n"
    "    for row in rows:\n"
    "        if row:\n"
    "            out.append(row)\n"
    "    for row in rows:\n"
    "    \tif row:\n"
    "            out.append(row)\n"
    "    return out\n"
)


@_handle_project
@pytest.mark.parametrize(
    "old, when, lines",
    [
        # Two whole-line blocks once indentation is taken relative.
        (
            "if row:\n    out.append(row)",
            "when indentation is compared relative to the block",
            "5-6, 8-9",
        ),
        # Two blocks once runs of spaces and tabs count as one.
        (
            "if row:\n out.append(row)",
            "when runs of spaces and tabs count as one",
            "5-6, 8-9",
        ),
    ],
)
def test_an_ambiguous_fuzzy_match_is_never_applied(patch_on, old, when, lines):
    fm = _FM()
    fm.add_functions(implementations=PICK)
    mark = _history_mark()
    out = fm.patch_function(name="pick", old=old, new="pass", why="w")
    assert (
        f"`old` occurs 2 times in the source of 'pick' {when} (at lines {lines})"
        in out["error"]
    )
    assert "set `replace_all`" in out["error"]
    assert _stored("pick")["implementation"] == PICK
    assert _history_since(mark) == []


@_handle_project
def test_replace_all_and_the_edit_tool_argument_names(patch_on):
    fm = _FM()
    fm.add_functions(implementations=SRC)
    mark = _history_mark()
    out = fm.patch_function(
        name="total_minor",
        old_string="row",
        new_string="item",
        replace_all=True,
        why="name the loop variable after what it holds",
    )
    assert out["edits"] == [{"match": "exact", "replaced": 5}]
    after = _stored("total_minor")["implementation"]
    assert after == SRC.replace("row", "item")
    assert len(_history_since(mark)) == 1
    out = fm.patch_function(
        name="total_minor",
        why="w",
        edits=[{"old_string": "total = 0", "new_string": "total = 1"}],
    )
    assert out["edits"] == [{"match": "exact", "replaced": 1}]
    # Renaming every "total" would rename the function: refused as a rename.
    out = fm.patch_function(
        name="total_minor",
        old="total",
        new="acc",
        replace_all=True,
        why="w",
    )
    assert "renames the function to 'acc_minor'" in out["error"]


@_handle_project
def test_with_the_switch_off_a_batch_is_refused_too(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", False)
    fm = _FM()
    fm.add_functions(implementations=SRC)
    mark = _history_mark()
    out = fm.patch_function(
        name="total_minor",
        why="w",
        edits=[{"old": "total = 0", "new": "total = 1"}],
    )
    assert out == {
        "name": "total_minor",
        "error": "patching is not enabled here (UNIFY_FUNCTION_PATCH is off)",
    }
    assert _stored("total_minor")["implementation"] == SRC
    assert _history_since(mark) == []


def test_the_tool_schema_offers_old_new_or_a_batch_of_edits():
    from unify.common.llm_helpers import method_to_schema

    schema = method_to_schema(_FM().patch_function)["function"]
    params = schema["parameters"]
    assert list(params["properties"]) == [
        "name",
        "old",
        "new",
        "why",
        "edits",
        "replace_all",
        "old_string",
        "new_string",
    ]
    assert params["required"] == ["name", "why"]
    item = params["properties"]["edits"]["items"]
    assert item["type"] == "object"
    assert set(item["properties"]) == {"old", "new", "replace_all"}
    assert item["required"] == ["old", "new"]
    assert params["properties"]["replace_all"] == {"type": "boolean"}
    assert "several as ``edits``" in schema["description"]
