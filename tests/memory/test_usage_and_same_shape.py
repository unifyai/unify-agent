"""Measured use for the writer and the when-to-use section."""

from unify.memory import library_prompts as pv
from unify.memory.evidence import EvidenceStore
from unify.memory.usage import MAX_TABLE_ROWS, usage_table


def test_the_library_usage_table_can_list_every_item(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ids = [f"memory.p.m:f{i:03d}" for i in range(MAX_TABLE_ROWS + 5)]
    head = usage_table(ev, [], ids)
    full = usage_table(ev, [], ids, max_rows=None)
    assert "more items not shown" in head and "more items not shown" not in full
    assert all(i in full for i in ids)


def test_write_asks_for_each_items_when_to_use_section_and_a_linked_note_of_preconditions():
    """Each item says when to use it itself (its section, unify/memory/when_to_use.py); a linked note keeps the
    function's preconditions and known failures."""
    text = pv.write_brief_now()
    assert "linked note says its preconditions and its known failures" in text
    # rev 2 (labels only): the section and its four lines as written, and a missing one is labelled, not refused
    shared = pv.WHEN_TO_USE_TEXT.split("  The gate refuses")[0]
    assert shared.endswith("Example request: Sum these payments per currency.\n") and shared in text
    assert "A new or changed function or note without the section or one of its four lines is labelled so." in text
    assert "The items of a folder the pass changes are labelled when the folder has no README.md" in text
    assert "The gate refuses" not in text.split("- Every function and note carries")[1].split("- Tests go")[0]
    assert pv.benchmark_words(text) == [] and pv.example_checks(text) == []
