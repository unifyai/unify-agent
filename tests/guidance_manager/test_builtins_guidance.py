"""Tests for seeding the builtin guidance table.

The code freeze baked builtin guidance off: a GuidanceManager neither seeds
nor reads the catalogue. The snapshot and its seeding functions stay for
scripts/skill_migration, and are tested here.
"""

from __future__ import annotations

from unify import db
from unify.guidance_manager.builtins import (
    seed_builtin_guidance,
    stable_guidance_id,
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


# --------------------------------------------------------------------------- #
# Default library (imported-skills snapshot)                                   #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Reads over both populations                                                  #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Immutability                                                                 #
# --------------------------------------------------------------------------- #
