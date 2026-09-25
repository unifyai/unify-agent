"""Tests for the builtin guidance library.

The ``builtin_guidance`` table is seeded from the committed snapshot and read
alongside the assistant's own ``guidance`` rows through the ``all_guidance``
view. ``db.clear()`` leaves the seeded catalogue alone, so a test that seeds
its own entries restores the snapshot before it returns.
"""

from __future__ import annotations

import pytest
from unify import db
from unify.guidance_manager.builtins import (
    load_snapshot,
    seed_builtin_guidance,
    stable_guidance_id,
)
from unify.guidance_manager.guidance_manager import (
    GUIDANCE_PREVIEW_CHARS,
    GuidanceManager,
)

_ENTRIES = {
    "test/ffmpeg-frames": {
        "title": "[test] ffmpeg-frames",
        "content": (
            "Extract frames from a video file. Use ffmpeg with an output "
            "pattern to grab still images from any video."
        ),
    },
    "test/arxiv-search": {
        "title": "[test] arxiv-search",
        "content": (
            "Search arXiv for academic papers. Query the arXiv API with "
            "keywords and parse the Atom feed of results."
        ),
    },
}


@pytest.fixture
def test_entries():
    """Seed the two test entries; restore the snapshot afterwards."""
    seed_builtin_guidance(entries=_ENTRIES)
    yield _ENTRIES
    seed_builtin_guidance()


def _builtin_rows() -> dict[str, dict]:
    rows = db.query(
        "SELECT guidance_id, title, content, is_builtin FROM all_guidance"
        " WHERE is_builtin = 1",
    )
    return {row["title"]: row for row in rows}


# --------------------------------------------------------------------------- #
# Seeding                                                                      #
# --------------------------------------------------------------------------- #


def test_seed_builtin_guidance_delta_and_idempotent():
    try:
        assert seed_builtin_guidance(entries=_ENTRIES) is True
        rows = _builtin_rows()
        assert set(rows) == {entry["title"] for entry in _ENTRIES.values()}
        for entry in _ENTRIES.values():
            row = rows[entry["title"]]
            assert row["is_builtin"] == 1
            assert row["guidance_id"] == stable_guidance_id(entry["title"])
            assert row["content"] == entry["content"]

        # Converged: re-seeding writes nothing.
        assert seed_builtin_guidance(entries=_ENTRIES) is False

        # Changed content is rewritten; the other row keeps its content.
        changed = {key: dict(entry) for key, entry in _ENTRIES.items()}
        changed["test/arxiv-search"]["content"] = "Updated arXiv instructions."
        assert seed_builtin_guidance(entries=changed) is True
        rows = _builtin_rows()
        assert rows["[test] arxiv-search"]["content"] == "Updated arXiv instructions."
        assert rows["[test] ffmpeg-frames"]["content"] == (
            _ENTRIES["test/ffmpeg-frames"]["content"]
        )

        # Removal: entries dropped from the snapshot disappear from the table.
        only_ffmpeg = {"test/ffmpeg-frames": _ENTRIES["test/ffmpeg-frames"]}
        assert seed_builtin_guidance(entries=only_ffmpeg) is True
        assert set(_builtin_rows()) == {"[test] ffmpeg-frames"}

        # An empty snapshot empties the table, then converges.
        assert seed_builtin_guidance(entries={}) is True
        assert _builtin_rows() == {}
        assert seed_builtin_guidance(entries={}) is False
    finally:
        seed_builtin_guidance()


def test_default_catalogue_is_converged():
    """Constructing a GuidanceManager seeds the snapshot; re-seeding is a no-op."""
    GuidanceManager()
    assert seed_builtin_guidance() is False
    snapshot = load_snapshot()
    assert len(snapshot) == 14
    assert set(_builtin_rows()) == {entry["title"] for entry in snapshot.values()}


# --------------------------------------------------------------------------- #
# Default library (imported-skills snapshot)                                   #
# --------------------------------------------------------------------------- #


def test_default_library_surfaces_through_guidance_manager():
    entries = load_snapshot()
    gm = GuidanceManager()

    builtin_rows = gm.filter(filter="is_builtin = 1", limit=100)
    assert {row.title for row in builtin_rows} == {
        entry["title"] for entry in entries.values()
    }
    assert gm._num_items() == len(entries)

    # List-style reads return bounded previews (large skills would otherwise
    # flood the caller's context window), each pointing at get_guidance.
    preview_slack = 200  # truncation marker text
    for row in builtin_rows:
        assert len(row.content) <= GUIDANCE_PREVIEW_CHARS + preview_slack
    truncated = [
        row for row in builtin_rows if "get_guidance(guidance_id=" in row.content
    ]
    assert truncated, "expected at least one truncated preview"

    # get_guidance returns the complete content verbatim, including entries
    # far beyond the preview cap.
    by_title = {entry["title"]: entry for entry in entries.values()}
    for row in truncated:
        full = gm.get_guidance(guidance_id=row.guidance_id)
        assert full.content == by_title[full.title]["content"]
        assert len(full.content) > GUIDANCE_PREVIEW_CHARS


def test_default_library_semantic_search():
    gm = GuidanceManager()

    results = gm.search(
        references={"content": "create a powerpoint presentation slide deck"},
        k=3,
    )
    assert results and results[0].title == "[anthropic] pptx"

    multi = gm.search(
        references={"content": "fill out pdf form fields", "title": "pdf"},
        k=3,
    )
    assert multi and multi[0].title == "[anthropic] pdf"
    assert all(row.is_builtin for row in multi)


def test_get_guidance_resolves_own_and_builtin_entries(test_entries):
    gm = GuidanceManager()
    outcome = gm.add_guidance(title="mine", content="my own entry")
    own_id = outcome["details"]["guidance_id"]

    own = gm.get_guidance(guidance_id=own_id)
    assert (own.title, own.content, own.is_builtin) == ("mine", "my own entry", False)

    builtin_id = stable_guidance_id("[test] arxiv-search")
    builtin = gm.get_guidance(guidance_id=builtin_id)
    assert builtin.is_builtin is True
    assert builtin.content == test_entries["test/arxiv-search"]["content"]

    with pytest.raises(ValueError, match="No guidance found"):
        gm.get_guidance(guidance_id=999999999)


# --------------------------------------------------------------------------- #
# Reads over both populations                                                  #
# --------------------------------------------------------------------------- #


def test_guidance_reads_blend_builtins_and_own_entries(test_entries):
    gm = GuidanceManager()
    gm.add_guidance(
        title="My deploy checklist",
        content="Run the deploy script and watch the logs.",
    )

    rows = gm.filter(limit=100)
    by_title = {row.title: row for row in rows}
    assert "My deploy checklist" in by_title
    assert by_title["My deploy checklist"].is_builtin is False
    for entry in test_entries.values():
        assert entry["title"] in by_title
        assert by_title[entry["title"]].is_builtin is True
        assert by_title[entry["title"]].guidance_id == stable_guidance_id(
            entry["title"],
        )

    assert gm._num_items() == 3

    # Filtering on the provenance flag targets each population explicitly.
    builtin_only = gm.filter(filter="is_builtin = 1", limit=100)
    assert {row.title for row in builtin_only} == {
        entry["title"] for entry in test_entries.values()
    }
    own_only = gm.filter(filter="is_builtin = 0", limit=100)
    assert [row.title for row in own_only] == ["My deploy checklist"]


def test_single_term_search_returns_builtins(test_entries):
    gm = GuidanceManager()

    results = gm.search(
        references={"content": "extract still images from a video with ffmpeg"},
        k=2,
    )
    assert results
    assert results[0].title == "[test] ffmpeg-frames"
    assert results[0].is_builtin is True


def test_multi_term_search_combines_builtins_scores(test_entries):
    gm = GuidanceManager()

    results = gm.search(
        references={
            "content": "find academic papers about machine learning",
            "title": "arxiv search",
        },
        k=2,
    )
    assert results
    assert results[0].title == "[test] arxiv-search"
    assert results[0].is_builtin is True


def test_exclude_ids_apply_to_builtins(test_entries):
    gm = GuidanceManager()
    excluded = stable_guidance_id("[test] ffmpeg-frames")
    gm.exclude_ids = frozenset({excluded})

    titles = {row.title for row in gm.filter(limit=100)}
    assert "[test] ffmpeg-frames" not in titles
    assert "[test] arxiv-search" in titles


# --------------------------------------------------------------------------- #
# Immutability                                                                 #
# --------------------------------------------------------------------------- #


def test_update_and_delete_builtin_guidance_refused(test_entries):
    gm = GuidanceManager()
    builtin_id = stable_guidance_id("[test] arxiv-search")

    with pytest.raises(ValueError, match="built-in"):
        gm.update_guidance(guidance_id=builtin_id, content="tampered")
    with pytest.raises(ValueError, match="built-in"):
        gm.delete_guidance(guidance_id=builtin_id)

    # The catalogue row is untouched and the assistant's own CRUD still works normally.
    rows = _builtin_rows()
    assert rows["[test] arxiv-search"]["content"] == (
        test_entries["test/arxiv-search"]["content"]
    )
    outcome = gm.add_guidance(title="mine", content="my own entry")
    own_id = outcome["details"]["guidance_id"]
    gm.update_guidance(guidance_id=own_id, content="my updated entry")
    gm.delete_guidance(guidance_id=own_id)
    assert db.query("SELECT 1 FROM guidance") == []
