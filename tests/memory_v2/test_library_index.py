"""links.json, INDEX.md and the index view (spec §4.3, §4.5): complete, sorted, deterministic."""

from __future__ import annotations

import json

import pytest

from tests.memory_v2.test_layout import FUNCTIONS, LIB, _tree
from unify.memory_v2 import layout
from unify.memory_v2.library_helper import find_section
from unify.memory_v2.library_index import (
    INDEX_HEAD,
    build_links,
    index_view,
    listed,
    packages_of,
    render_index,
    render_links,
)

PARSE_DATE = "memory.text.dates:parse_date"
NOTE_ID = "notes/text/dates.md"
EXPECTED_INDEX = (
    INDEX_HEAD + "\n## memory.text — Text helpers.\n\n"
    "- `memory.text.dates:parse_date(s: str) -> str` — Parse a date written as YYYY-MM-DD. [experimental]; "
    "notes: notes/text/dates.md\n"
    "- `memory.text.dates:week(s)` — The ISO week of a date. [experimental]\n"
    "- `memory.text.parse:tokens(s)` — Split text on whitespace. [experimental]\n"
    "- `memory.text.parse:words(s)` — Count the words of a text. [experimental]\n"
    "- `memory.text.report:summary(s)` — One line about a date. [experimental]\n"
    "\n## memory.web\n\n"
    "- `memory.web.fetch:get(u)` — Fetch nothing; split the address. [experimental]\n"
    "\n## notes/text\n\n"
    "- `notes/text/dates.md` — How dates are written. [experimental]; functions: memory.text.dates:parse_date\n"
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


def test_index_is_complete_sorted_and_byte_stable(tmp_path):
    lib = _lib(tmp_path)
    index = render_index(lib, build_links(lib))
    assert index == EXPECTED_INDEX
    again = layout.discover(_tree(tmp_path / "again"))
    assert render_index(again, build_links(again)) == index
    assert listed(index) == 7 and packages_of(index) == ["text", "web"]


def test_suspect_and_deprecated_items_leave_the_index_but_not_the_links(tmp_path):
    lib = _lib(tmp_path)
    links = build_links(lib)
    status = {
        PARSE_DATE: "suspect",
        "memory.web.fetch:get": "deprecated",
        "memory.text.dates:week": "stable",
    }
    index = render_index(lib, links, lambda i: status.get(i, "experimental"))
    assert PARSE_DATE + "(" not in index and "## memory.web" not in index
    assert "- `memory.text.dates:week(s)` — The ISO week of a date. [stable]\n" in index
    assert (
        "- `notes/text/dates.md` — How dates are written. [experimental]\n" in index
    )  # no hidden link shown
    assert PARSE_DATE in links["items"]
    with pytest.raises(ValueError, match="unknown status 'gold'"):
        render_index(lib, links, lambda i: "gold")


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
        match="no index section 'memory.nope'; the sections are: memory.text",
    ):
        find_section(index, "nope")
