"""A librarian fit's call must match the real API of the items it uses (rev 2.3).

Rev 2.2's ``check_answer`` checked only that a call goes through ``memory``; nine fits offered
``memory.get('capture:…').run()``, a member an item does not have, and the actor's call raised AttributeError. Here the
call is parsed against the items' recorded forms in the library copy: the members of the Item ``memory.get`` returns, a
recorded procedure's one-input call, a function's parameters from its source, a note that is read and never called.
Nothing is run and no word of the call is matched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from unify.memory import librarian as L

FUNC = "memory.tables.count:count_rows"
FUNC_KW = "memory.tables.count:scale"
RECIPE_FUNC = "memory.tables.count:apply_total"
NOTE = "notes/tables/sums.md"
CAPTURE = "capture:0123456789abcdef"
IDS = [FUNC, FUNC_KW, RECIPE_FUNC, NOTE, CAPTURE]

SOURCE = '''
def count_rows(rows, limit=None):
    """Count rows."""
    return len(rows[:limit])


def scale(rows, *, factor=2):
    return [r * factor for r in rows]


def apply_total(text):
    return text
'''


@pytest.fixture
def copy(tmp_path: Path) -> Path:
    root = tmp_path / "copy"
    (root / "memory" / "tables").mkdir(parents=True)
    (root / "memory" / "__init__.py").write_text("")
    (root / "memory" / "tables" / "count.py").write_text(SOURCE)
    rows = {
        FUNC: {"kind": "function", "path": "memory/tables/count.py"},
        FUNC_KW: {"kind": "function", "path": "memory/tables/count.py"},
        RECIPE_FUNC: {"kind": "function", "path": "memory/tables/count.py", "recipe": RECIPE_FUNC},
        NOTE: {"kind": "note", "path": NOTE},
    }
    (root / ".memory").mkdir()
    (root / ".memory" / "items.json").write_text(json.dumps({"items": rows}))
    d = root / ".memory" / "captures" / CAPTURE.split(":", 1)[1]
    d.mkdir(parents=True)
    (d / "procedure.py").write_text("def read_input(text):\n    return text\n\n\ndef run(value):\n    return value\n")
    (d / "record.json").write_text(json.dumps({"capture_id": CAPTURE[8:], "entry_point": {"form": "value"}}))
    return root


def _check(call: str, copy: Path, iid: str = CAPTURE, recipes=None):
    return L.check_answer({"fits": [{"id": iid, "call": call, "why": "w"}]}, IDS, recipes=recipes, copy=copy)


@pytest.mark.parametrize(
    "call, iid, why",
    [
        # rev 2.2's nine fits: .run() on an Item
        (f"memory.get({CAPTURE!r}).run()", CAPTURE, "reads 'run' on the item"),
        # through a bound name (calling ``p.run(...)`` is already refused by rev 2.2's through-memory check)
        (f"p = memory.get({CAPTURE!r})\nprint(p.run)", CAPTURE, "reads 'run' on the item"),
        (f"memory.get({CAPTURE!r})(1, 2)", CAPTURE, "takes none (the current request) or one input"),
        (f"memory.get({CAPTURE!r})(text=memory.request())", CAPTURE, "takes none (the current request) or one input"),
        (f"memory.get({NOTE!r})()", NOTE, "is read, not called"),
        (f"memory.get({FUNC!r})()", FUNC, "has no recipe that runs it on the current request"),
        (f"memory.get({FUNC!r})(1, 2, 3)", FUNC, "takes at most 2"),
        (f"memory.get({FUNC_KW!r})([1], scale=3)", FUNC_KW, "has no such parameter"),
    ],
)
def test_a_call_that_does_not_match_the_items_api_is_refused_with_the_reason(copy, call, iid, why):
    got, reason = _check(call, copy, iid)
    assert got is None and why in reason, reason


@pytest.mark.parametrize(
    "call, iid",
    [
        (f"memory.get({CAPTURE!r})()", CAPTURE),
        (f"memory.get({CAPTURE!r})(memory.request())", CAPTURE),
        (f"print(memory.get({CAPTURE!r}).text)", CAPTURE),
        (f"memory.get({CAPTURE!r}).fits({{'a': 1}})", CAPTURE),
        (f"memory.get({FUNC!r})(rows)", FUNC),
        (f"memory.get({FUNC!r})(rows, limit=3)", FUNC),
        (f"memory.get({FUNC_KW!r})(rows, factor=3)", FUNC_KW),
        (f"memory.get({RECIPE_FUNC!r})()", RECIPE_FUNC),
        (f"print(memory.get({NOTE!r}))", NOTE),
        (f"f = memory.get({FUNC!r})\nprint(f(rows))", FUNC),
    ],
)
def test_a_call_that_matches_the_items_api_passes_unchanged(copy, call, iid):
    got, reason = _check(call, copy, iid, recipes={RECIPE_FUNC: RECIPE_FUNC})
    assert reason == "" and got == {"fits": [{"id": iid, "call": call, "why": "w"}]}


def test_without_the_copy_the_api_is_not_read_and_rev_2_2s_checks_alone_apply():
    got, reason = L.check_answer(
        {"fits": [{"id": CAPTURE, "call": f"memory.get({CAPTURE!r}).run()", "why": "w"}]}, IDS
    )
    assert reason == "" and got is not None


def test_the_members_are_those_of_the_item_memory_get_returns():
    from unify.memory.library_helper import Item

    members = L._item_members()
    assert "run" not in members
    assert {"text", "fits", "id", "summary"} <= members
    import dataclasses

    assert "fits" in members and not any(m.startswith("_") for m in members)
    assert {f.name for f in dataclasses.fields(Item)} <= members


def test_an_unreadable_source_never_refuses_a_call(copy):
    (copy / "memory" / "tables" / "count.py").write_text("def broken(:\n")
    got, reason = _check(f"memory.get({FUNC!r})(1, 2, 3)", copy, FUNC)
    assert reason == "" and got is not None


def test_the_finish_tool_checks_against_the_copy_the_librarian_reads(copy):
    from types import SimpleNamespace

    import asyncio

    state = L.LibrarianCalls(ids=IDS, copy=copy)
    got = asyncio.run(
        L.finish(SimpleNamespace(state=state), {"answer": {"fits": [{"id": CAPTURE, "call": f"memory.get({CAPTURE!r}).run()", "why": "w"}]}})
    )
    assert not state.finished and "reads 'run' on the item" in got.text


# --- the dotted form: memory.<package>.<module>.<name>(...) is the same function, checked the same way -------------


@pytest.mark.parametrize(
    "call, why",
    [
        ("memory.tables.count.count_rows(1, 2, 3)", "takes at most 2"),
        ("memory.tables.count.scale([1], scale=3)", "has no such parameter"),
        ("memory.tables.count.count_rows()", "has no recipe that runs it on the current request"),
    ],
)
def test_a_dotted_call_is_checked_as_the_item_it_names(copy, call, why):
    got, reason = _check(call, copy, FUNC)
    assert got is None and why in reason, reason


@pytest.mark.parametrize(
    "call",
    [
        "memory.tables.count.count_rows(rows)",
        "memory.tables.count.scale(rows, factor=3)",
        "print(memory.tables.count.count_rows.__doc__)",  # the function itself, not an Item: its own attributes
        "memory.tables.unknown_module.f(1, 2, 3, 4)",  # not an item the copy lists: left to the other checks
    ],
)
def test_a_dotted_call_that_matches_passes(copy, call):
    got, reason = _check(call, copy, FUNC)
    assert reason == "" and got is not None
