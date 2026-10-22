"""One example call quoted from a function's tests: the quote, the record that carries it (kept from a gate before
rev 2.1, which picked it) and ``memory.show``."""

from __future__ import annotations

import json

from tests.memory.test_layout import LIB, _tree
from tests.memory.test_library_helper import _py
from unify.memory import example_view as ev
from unify.memory import library_export as lx
from unify.memory import lifecycle as lc
from unify.memory.catalogue import write_files
from unify.memory.item_records import empty_record

PD = "memory.text.dates:parse_date"
TESTS = "memory/text/tests/test_dates.py"
TEST = TESTS + "::test_parse_date"
QUOTED = "parse_date('2026-10-09')"
MULTI = """from memory.grid import fill


def test_fills():
    grid = [[0, 1], [1, 1]]
    got = fill.flood_fill(
        grid,   # the recorded grid
        (0,  0),
        3,
    )
    assert got
    assert flood_fill(flood_fill(grid, (0, 0), 3), (1, 1), 4)
"""


# --- the quote --------------------------------------------------------------------------------------------


def test_a_one_line_call_is_quoted_as_written(tmp_path):
    root = _tree(tmp_path / "lib")
    assert ev.call_text(root, TESTS, 5, "parse_date") == QUOTED
    assert ev.call_text(root, TESTS, 5, "week") is None  # no call of that name there
    assert ev.call_text(root, TESTS, 4, "parse_date") is None  # not at that line
    assert ev.call_text(root, "memory/text/tests/test_nope.py", 5, "parse_date") is None


def test_a_multi_line_call_is_collapsed_without_its_comments_from_any_of_its_lines():
    want = "fill.flood_fill( grid, (0, 0), 3, )"
    assert ev.quote_call(MULTI, 6, "flood_fill") == want
    assert (
        ev.quote_call(MULTI, 8, "flood_fill") == want
    )  # a line inside the call finds it too
    # two calls on one line: the outer one, first in the source
    assert (
        ev.quote_call(MULTI, 12, "flood_fill")
        == "flood_fill(flood_fill(grid, (0, 0), 3), (1, 1), 4)"
    )


def test_a_long_call_is_cut_and_the_quote_is_byte_stable():
    source = (
        "def test_x():\n    f(" + ", ".join(f"value_{i}" for i in range(60)) + ")\n"
    )
    text = ev.quote_call(source, 2, "f")
    assert len(text) == ev.EXAMPLE_MAX_CHARS and text.endswith(" …")
    assert text.startswith("f(value_0, value_1, ")
    assert ev.quote_call(source, 2, "f") == text
    assert ev.quote_call("def broken(:\n", 1, "f") is None


def test_only_files_under_memory_are_read(tmp_path):
    root = _tree(tmp_path / "lib")
    (tmp_path / "outside.py").write_text("parse_date('x')\n")
    (root / "memory/text/tests/test_link.py").symlink_to(tmp_path / "outside.py")
    for rel in (
        "../outside.py",
        str(tmp_path / "outside.py"),
        "memory/../../outside.py",
        "memory/text/tests/test_link.py",
        "notes/text/dates.md",
    ):
        assert ev.call_text(root, rel, 1, "parse_date") is None, rel


# --- the pick ---------------------------------------------------------------------------------------------


# --- the record, the export and show ----------------------------------------------------------------------


def test_a_recorded_example_reaches_the_record_and_input_is_gone():
    base = dict(
        items={PD: "function"},
        prev={},
        facts=[],
        since={PD: {"a" * 40}},
        changed_at={PD: "a" * 40},
        functions={PD: "memory.text.dates"},
        graph={"memory.text.dates": set()},
        note_uses={},
    )
    example = {"test": TEST, "line": 5, "call": QUOTED}
    recs, _ = lc.next_records(
        **base,
        verification={PD: {"examples": {"calls": 1, "example": example}}},
    )
    assert recs[PD]["example"] == example and "input" not in recs[PD]
    assert lc.next_records(**base)[0][PD]["example"] is None
    assert empty_record(PD, "function")["example"] is None
    assert "input" not in empty_record(PD, "function")


def _records(example: dict | None) -> dict:
    return {PD: {**empty_record(PD, "function"), "example": example}}


def test_show_prints_the_example_under_the_signature(tmp_path):
    root = _tree(tmp_path / "lib")
    write_files(
        root,
        lx.generated_library_mode(
            root,
            records=_records({"test": TEST, "line": 5, "call": QUOTED}),
        ),
    )
    out = json.loads(
        _py(root, f"import json, memory\nprint(json.dumps(memory.show({PD!r})))"),
    )
    assert out.splitlines()[:3] == [
        "memory.text.dates:parse_date(s: str) -> str  [on-trial, curated item from 0 episodes]",
        f"Example (from {TEST}): {QUOTED}",
        "",
    ]
    week = json.loads(
        _py(
            root,
            "import json, memory\nprint(json.dumps(memory.show('memory.text.dates:week')))",
        ),
    )
    assert "Example (from" not in week


def test_the_export_quotes_an_example_only_while_its_test_still_holds_it(tmp_path):
    root = _tree(tmp_path / "lib")
    records = _records({"test": TEST, "line": 5, "call": QUOTED})
    files = lx.generated_library_mode(root, records=records)
    items = json.loads(files[".memory/items.json"])["items"]
    assert items[PD]["example"] == {"test": TEST, "line": 5, "call": QUOTED}
    assert items["memory.text.dates:week"]["example"] is None
    # a copy without its tests (the actor's export) reads them through *tests*
    bare = _tree(
        tmp_path / "bare",
        {k: v for k, v in LIB.items() if "/tests/" not in k},
    )
    assert (
        json.loads(
            lx.generated_library_mode(bare, records=records)[".memory/items.json"],
        )["items"][PD]["example"]
        is None
    )
    via = lx.generated_library_mode(
        bare,
        records=records,
        tests=ev.tree_reader(root),
    )
    assert via[".memory/items.json"] == files[".memory/items.json"]
    assert via["INDEX.md"] == files["INDEX.md"]
    # the test edited since: the record's line quotes something else, so nothing is shown
    (root / TESTS).write_text("\n" + LIB[TESTS])
    stale = lx.generated_library_mode(root, records=records)
    assert json.loads(stale[".memory/items.json"])["items"][PD]["example"] is None
    assert b"e.g." not in stale["INDEX.md"].split(b"\n## ", 1)[1]
    # a recorded call text that differs from the file's is never shown
    (root / TESTS).write_text(LIB[TESTS])
    wrong = _records({"test": TEST, "line": 5, "call": "parse_date('1999-01-01')"})
    assert (
        json.loads(
            lx.generated_library_mode(root, records=wrong)[".memory/items.json"],
        )["items"][PD]["example"]
        is None
    )
