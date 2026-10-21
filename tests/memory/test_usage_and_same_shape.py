"""Measured use for the writer, same-shape functions for CURATE, the when-to-use section."""

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
    assert pv.WHEN_TO_USE_TEXT in text
    assert pv.benchmark_words(text) == [] and pv.example_checks(text) == []


def _with_replay(monkeypatch):
    """The replay sentence appears only when the replay tool is registered for the writer."""
    from unify.memory import tool_spec

    spec = tool_spec.ToolSpec(
        name="replay",
        description="Verify an item by replay.",
        parameters={"type": "object", "properties": {}},
        roles=frozenset({"write"}),
    )
    monkeypatch.setattr(tool_spec, "registry", lambda: (spec,))


def test_the_notes_and_replay_sentences_inform_and_never_set_a_bar(monkeypatch):
    """A live writer read the old generality sentence as "only cross-task, verified
    knowledge" and dismissed recurring solved tasks. The wording now encourages generality and values a rule for one
    kind of request that comes back; the replay sentence offers verification, it does not require it.
    """
    _with_replay(monkeypatch)
    text = pv.write_brief_now()
    notes = text.split("A note states its rule")[1].split("\n")[0]
    assert "worth keeping when that kind of request comes back" in notes
    assert (
        "beyond the episodes it came from" not in text
        and "only" not in notes
        and "must" not in notes
    )
    replay = text.split("A function lifted from the working model's cells")[1].split(
        "\n",
    )[0]
    assert (
        replay.startswith(" can be verified")
        and "only" not in replay
        and "must" not in replay
    )
    assert (
        "Store anything that repeats or is likely to repeat" in text
    )  # rule 1 unchanged
    assert pv.benchmark_words(text) == [] and pv.example_checks(text) == []
