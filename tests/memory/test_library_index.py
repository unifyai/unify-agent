"""links.json, INDEX.md and the index view: complete, sorted, deterministic."""

from __future__ import annotations

import json

import pytest

from tests.memory.test_layout import FUNCTIONS, LIB, _tree
from unify.memory import layout
from unify.memory.library_helper import find_section
from unify.memory.library_index import INDEX_HEAD, build_links, listed, render_index, render_links

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


def test_the_index_shows_no_effect_and_no_test_breadth(tmp_path):
    """Nothing a test or a judgement of an item's code gave is served: the index's brackets hold its status and its
    support only."""
    lib = _lib(tmp_path)
    index = render_index(lib, build_links(lib))
    assert "effect" not in index and "examples from one episode" not in index
    assert "effect" not in INDEX_HEAD


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
    shown = render_index(lib, links, None, _example_of(tmp_path))
    assert shown.replace(EXAMPLE_LINE, "") == plain
    assert len(shown) - len(plain) == len(EXAMPLE_LINE)
    grown = estimate_tokens(shown) - estimate_tokens(plain)
    assert 0 < grown <= estimate_tokens(EXAMPLE_LINE) + 1


def test_a_stale_example_is_never_quoted_in_the_index(tmp_path):
    lib = _lib(tmp_path)
    links = build_links(lib)
    test_file = tmp_path / "memory/text/tests/test_dates.py"
    test_file.write_text("# edited\n" + test_file.read_text())
    assert render_index(lib, links, None, _example_of(tmp_path)) == EXPECTED_INDEX


def test_an_items_support_shows_beside_its_status(tmp_path):
    """Rev 2.1, fix 9: every index line says what its item rests on (its kind, the episodes it comes from and its
    confirmed uses) in place of the evidence level the gate used to set by running it."""
    from unify.memory.item_records import empty_record, support

    lib = _lib(tmp_path)
    links = build_links(lib)
    week, tokens = "memory.text.dates:week", "memory.text.parse:tokens"
    confirmed = {"verdicts": {"accepted": 2, "rejected": 0}, "end_outcomes": {"positive": 1, "contrary": 0}}
    records = {
        week: {**empty_record(week, "function"), "provenance": {"episodes": ["e1"]}},
        tokens: {**empty_record(tokens, "function"), "provenance": {"episodes": ["e1", "e2"]}, "track": confirmed},
        NOTE_ID: {**empty_record(NOTE_ID, "note"), "provenance": {"episodes": ["e3"]}},
    }

    def support_of(item):
        return support(records.get(item)) if item in records else None

    index = render_index(lib, links, None, support_of=support_of)
    assert "- `memory.text.dates:week(s)` — The ISO week of a date. [on-trial, curated item from 1 episode]\n" in index
    assert "Split text on whitespace. [on-trial, curated item from 2 episodes, 3 confirmed uses]\n" in index
    # an item with no record shows nothing it does not know; a note with one shows its support too
    assert "Count the words of a text. [on-trial]\n" in index
    assert "How dates are written. [on-trial, curated item from 1 episode]; " in index
    assert "curated item from <n> episodes" in INDEX_HEAD and "evidence:" not in INDEX_HEAD
    # no support function: the index is byte for byte the one without support
    assert render_index(lib, links) == render_index(lib, links, support_of=lambda i: None)
    assert "evidence:" not in render_index(lib, links, support_of=support_of)
