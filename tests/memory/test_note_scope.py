"""A note's scope (rev 2.4, P1): its front matter takes ``applies_to`` (what it holds for) and ``stated_by`` (who stated
it, by name or role), and every place that serves the note shows them: the library's index, its search rows, the items
data ``memory.get`` and ``memory.items`` read, and an item's one line. Real readers on a real library tree.
"""

from __future__ import annotations

import pytest

from unify.memory import layout

# --- a note's scope, in its front matter, is shown wherever the note is served -------------------------------------------


NOTE_FILE = (
    "---\ntitle: Reports as tab-separated files\ndescription: send reports as .tsv files\nuses: []\n"
    "applies_to: monthly reports for the planning team\nstated_by: the planning lead\nkind: preference\n---\n"
    "## Rule\n- Send each report as a .tsv file.\n\n"
    "## When to use\n- **Saved for:** sending a report.\n- **Recognise:** a report is asked for.\n"
    "- **Not for:** other files.\n- **Example request:** Send the monthly report.\n"
)
SCOPE = "applies to: monthly reports for the planning team; stated by: the planning lead"


def test_a_note_carries_its_scope_and_the_index_search_items_and_cards_show_it(tmp_path):
    from unify.memory import library_export, library_helper, library_index, library_search

    meta = layout.parse_front_matter(NOTE_FILE)
    assert (meta.applies_to, meta.stated_by) == ("monthly reports for the planning team", "the planning lead")
    assert meta.kind == "preference"
    # kind is optional (a note with none is shown as a note), and when given it is one of the four
    assert layout.parse_front_matter(NOTE_FILE.replace("kind: preference\n", "")).kind == ""
    assert layout.parse_front_matter(NOTE_FILE.replace("kind: preference", "kind: Rule")).kind == "rule"
    with pytest.raises(layout.FrontMatterError, match="kind is one of preference, rule, caution, lesson"):
        layout.parse_front_matter(NOTE_FILE.replace("kind: preference", "kind: hunch"))
    # a note without them reads as before, with no scope
    plain = layout.parse_front_matter(NOTE_FILE.replace("applies_to: monthly reports for the planning team\n", "")
                                      .replace("stated_by: the planning lead\n", "")
                                      .replace("kind: preference\n", ""))  # fmt: skip
    assert (plain.applies_to, plain.stated_by, plain.kind) == ("", "", "")
    tree = tmp_path / "lib"
    (tree / "notes" / "reports").mkdir(parents=True)
    (tree / "notes" / "reports" / "tsv.md").write_text(NOTE_FILE)
    (tree / "notes" / "reports" / "README.md").write_text("Notes on reports.\n")
    lib = layout.discover(tree)
    assert not lib.errors, lib.errors
    (note,) = lib.notes
    assert note.scope == SCOPE and note.uses == [] and note.kind == "preference"  # a note with no function stands on its own
    assert f"send reports as .tsv files ({SCOPE})" in library_index.render_index(lib, library_index.build_links(lib))
    (row,) = [r for r in library_search.rows(tree, lib) if r["kind"] == "note"]
    assert SCOPE in row["card"]
    data = library_export.items_data(lib, None, {}, True)["items"]["notes/reports/tsv.md"]
    assert (data["applies_to"], data["stated_by"]) == ("monthly reports for the planning team", "the planning lead")
    assert library_helper._note_summary(data) == f"send reports as .tsv files ({SCOPE})"
