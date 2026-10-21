"""links.json, INDEX.md and the index view: complete, sorted, deterministic."""

from __future__ import annotations

import json

import pytest

from tests.memory.test_layout import FUNCTIONS, LIB, _tree
from unify.memory import layout
from unify.memory.library_helper import find_section
from unify.memory.library_index import INDEX_HEAD, build_links, index_view, listed, render_index, render_links

PARSE_DATE = "memory.text.dates:parse_date"
NOTE_ID = "notes/text/dates.md"
EXPECTED_INDEX = (
    INDEX_HEAD + "\n## memory.text — Text helpers.\n\n"
    "- `memory.text.dates:parse_date(s: str) -> str` — Parse a date written as YYYY-MM-DD. [on-trial]; "
    "notes: notes/text/dates.md\n"
    "- `memory.text.dates:week(s)` — The ISO week of a date. [on-trial]\n"
    "- `memory.text.parse:tokens(s)` — Split text on whitespace. [on-trial]\n"
    "- `memory.text.parse:words(s)` — Count the words of a text. [on-trial]\n"
    "- `memory.text.report:summary(s)` — One line about a date. [on-trial]\n"
    "\n## memory.web\n\n"
    "- `memory.web.fetch:get(u)` — Fetch nothing; split the address. [on-trial]\n"
    "\n## notes/text\n\n"
    "- `notes/text/dates.md` — How dates are written. [on-trial]; functions: memory.text.dates:parse_date\n"
)


def _lib(tmp_path):
    return layout.discover(_tree(tmp_path))


def test_links_are_derived_both_ways_and_dangling_ones_are_named(tmp_path):
    links = build_links(_lib(tmp_path))
    assert sorted(links["items"]) == sorted(FUNCTIONS + [NOTE_ID])
    assert links["items"][PARSE_DATE] == {
        "kind": "function",
        "path": "memory/text/dates.py",
        "links_to": [NOTE_ID],
        "linked_from": [NOTE_ID],
    }
    assert links["items"][NOTE_ID]["links_to"] == ["memory.text.dates:gone", PARSE_DATE]
    assert links["items"][NOTE_ID]["linked_from"] == [PARSE_DATE]
    assert links["dangling"] == [
        {"from": NOTE_ID, "to": "memory.text.dates:gone", "why": "missing"},
    ]
    text = render_links(links)
    assert json.loads(text) == links and text.endswith("\n")


def test_malformed_and_wrong_kind_links_are_dangling(tmp_path):
    files = dict(LIB)
    files["memory/text/dates.py"] = LIB["memory/text/dates.py"].replace(
        "Notes: notes/text/dates.md",
        "Notes: notes/text/dates.md, see the ledger, memory.text.parse:tokens",
    )
    links = build_links(layout.discover(_tree(tmp_path, files)))
    assert {"from": PARSE_DATE, "to": "see the ledger", "why": "malformed"} in links[
        "dangling"
    ]
    assert {
        "from": PARSE_DATE,
        "to": "memory.text.parse:tokens",
        "why": "not a note",
    } in links["dangling"]


def test_retired_items_leave_the_index_but_not_the_links(tmp_path):
    lib = _lib(tmp_path)
    links = build_links(lib)
    status = {
        PARSE_DATE: "retired",
        "memory.web.fetch:get": "retired",
        "memory.text.dates:week": "adopted",
    }
    index = render_index(lib, links, lambda i: status.get(i, "on-trial"))
    assert PARSE_DATE + "(" not in index and "## memory.web" not in index
    assert "- `memory.text.dates:week(s)` — The ISO week of a date. [adopted]\n" in index
    assert (
        "- `notes/text/dates.md` — How dates are written. [on-trial]\n" in index
    )  # no hidden link shown
    assert PARSE_DATE in links["items"]
    with pytest.raises(ValueError, match="unknown status 'gold'"):
        render_index(lib, links, lambda i: "gold")
    # an old status name is not one the index renders: item_records.status_of maps it before it gets here
    with pytest.raises(ValueError, match="unknown status 'experimental'"):
        render_index(lib, links, lambda i: "experimental")


def test_an_empty_library_has_an_index_with_no_items(tmp_path):
    lib = layout.discover(tmp_path)
    index = render_index(lib, build_links(lib))
    assert index == INDEX_HEAD + "\nThe library is empty.\n" and listed(index) == 0


def test_the_view_is_the_index_within_budget_else_headings_with_counts(tmp_path):
    lib = _lib(tmp_path)
    index = render_index(lib, build_links(lib))
    assert index_view(index) == index
    view = index_view(index, budget_tokens=10)
    assert view.startswith(INDEX_HEAD)
    assert "## memory.text — Text helpers. (5 functions)\n" in view
    assert "## memory.web (1 function)\n" in view and "## notes/text (1 note)\n" in view
    assert (
        '`memory.index("memory.text")`' in view and "INDEX.md holds all of it" in view
    )
    assert "- `" not in view


def test_sections_are_found_by_package_or_notes_topic(tmp_path):
    lib = _lib(tmp_path)
    index = render_index(lib, build_links(lib))
    text = find_section(index, "text")
    assert (
        text.startswith("## memory.text — Text helpers.\n") and "memory.web" not in text
    )
    assert (
        find_section(index, "memory.text") == text == find_section(index, "memory/text")
    )
    assert find_section(index, "notes/text").startswith("## notes/text\n")
    with pytest.raises(
        LookupError,
        match="no index section 'memory.nope' or 'notes/nope'; the sections are: memory.text",
    ):
        find_section(index, "nope")


def test_a_functions_derived_effect_shows_beside_its_status(tmp_path):
    lib = _lib(tmp_path)
    links = build_links(lib)
    effects = {"memory.text.dates:week": "write", "memory.text.parse:tokens": "read"}
    index = render_index(lib, links, None, effects.get)
    assert (
        "- `memory.text.dates:week(s)` — The ISO week of a date. [on-trial, effect: write]\n"
        in index
    )
    assert "Split text on whitespace. [on-trial, effect: read]\n" in index
    assert (
        "Count the words of a text. [on-trial]\n" in index
    )  # none derived: status only
    assert "effect: write" in INDEX_HEAD  # the head says what the bracket means
    # no effect function: the index is byte for byte the one without effects
    assert render_index(lib, links) == render_index(lib, links, None, lambda i: None)
    # a value outside the known effects is never rendered
    assert (
        "effect"
        not in render_index(lib, links, None, lambda i: "bogus").split("\n## ", 1)[1]
    )


EXAMPLE_TEST = "memory/text/tests/test_dates.py::test_parse_date"
EXAMPLE_LINE = f"  e.g. `parse_date('2026-10-09')` ({EXAMPLE_TEST})\n"


def _example_of(tmp_path):
    from unify.memory.example_view import example_of, tree_reader
    from unify.memory.item_records import empty_record

    example = {"test": EXAMPLE_TEST, "line": 5, "call": "parse_date('2026-10-09')"}
    records = {PARSE_DATE: {**empty_record(PARSE_DATE, "function"), "example": example}}
    return example_of(records, tree_reader(tmp_path))


def test_the_index_grows_by_the_example_lines_only(tmp_path):
    from unify.memory.catalogue import estimate_tokens

    lib = _lib(tmp_path)
    links = build_links(lib)
    plain = render_index(lib, links)
    shown = render_index(lib, links, None, None, _example_of(tmp_path))
    assert shown.replace(EXAMPLE_LINE, "") == plain
    assert len(shown) - len(plain) == len(EXAMPLE_LINE)
    grown = estimate_tokens(shown) - estimate_tokens(plain)
    assert 0 < grown <= estimate_tokens(EXAMPLE_LINE) + 1


def test_a_stale_example_is_never_quoted_in_the_index(tmp_path):
    lib = _lib(tmp_path)
    links = build_links(lib)
    test_file = tmp_path / "memory/text/tests/test_dates.py"
    test_file.write_text("# edited\n" + test_file.read_text())
    assert render_index(lib, links, None, None, _example_of(tmp_path)) == EXPECTED_INDEX


def test_a_functions_evidence_level_shows_beside_its_status(tmp_path):
    from unify.memory.item_records import empty_record, level_of

    lib = _lib(tmp_path)
    links = build_links(lib)
    week, tokens = "memory.text.dates:week", "memory.text.parse:tokens"
    records = {
        week: {**empty_record(week, "function"), "level": "unchecked"},
        tokens: {**empty_record(tokens, "function"), "level": "reproduces-its-case"},
        NOTE_ID: {**empty_record(NOTE_ID, "note")},
    }
    index = render_index(
        lib, links, None, {week: "write"}.get, level_of=level_of(records)
    )
    assert (
        "- `memory.text.dates:week(s)` — The ISO week of a date. [on-trial, evidence: unchecked, effect: write]\n"
        in index
    )
    assert "Split text on whitespace. [on-trial, evidence: reproduces-its-case]\n" in index
    # a function with no record shows no level, nor does a note
    assert "Count the words of a text. [on-trial]\n" in index
    assert "How dates are written. [on-trial]; " in index
    assert "evidence: <level>" in INDEX_HEAD  # the head says what the bracket means
    # no level function, or none known: the index is byte for byte the one without levels
    assert render_index(lib, links) == render_index(lib, links, level_of=lambda i: None)
    assert render_index(lib, links, level_of=level_of(records)) == render_index(
        lib, links, level_of=level_of(records)
    )
    # a value outside the known levels is never rendered
    assert "evidence" not in render_index(lib, links, level_of=lambda i: "bogus").split("\n## ", 1)[1]
