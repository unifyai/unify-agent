"""Determinism unit tests for the cache-stability quick wins: content-hash
spill filenames and byte-identical truncation in formatting._spill_full_tool_text().
"""

from __future__ import annotations

from unify.common._async_tool.formatting import (
    _spill_full_tool_text,
    _truncate_tool_text,
)

# ---------------------------------------------------------------------------
# Content-hash spill filenames (formatting.py)
# ---------------------------------------------------------------------------


def test_spill_filename_is_content_hash_and_idempotent():
    text = "x" * 100

    path_1 = _spill_full_tool_text(text)
    path_2 = _spill_full_tool_text(text)

    assert path_1 is not None
    # Identical content spills to the identical path (idempotent overwrite),
    # not a fresh randomly-named file each call.
    assert path_1 == path_2

    with open(path_1, "r", encoding="utf-8") as f:
        assert f.read() == text

    # Different content spills to a different path.
    other_path = _spill_full_tool_text("y" * 100)
    assert other_path != path_1


def test_truncated_tool_text_is_byte_identical_across_renders():
    long_text = "abcdefgh" * 10_000  # well over TOOL_RESULT_TEXT_CHAR_LIMIT

    rendered_1 = _truncate_tool_text(long_text)
    rendered_2 = _truncate_tool_text(long_text)

    assert rendered_1 == rendered_2
