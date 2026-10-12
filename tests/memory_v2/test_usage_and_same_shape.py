"""Measured use for the writer, same-shape functions for CURATE, the when-to-use note."""

from unify.memory_v2 import curate, library_prompts as pv
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.usage import MAX_TABLE_ROWS, usage_table


def test_curate_lists_library_functions_with_the_same_shape():
    bodies = {
        "memory.a.b:first": "def first(xs):\n    return [x * 2 for x in xs if x > 3]\n",
        "memory.a.c:second": "def second(items):\n    return [i * 5 for i in items if i > 9]\n",
        "memory.a.d:third": "def third(xs):\n    return sorted(xs)\n",
    }
    assert curate.same_shape(bodies) == [["memory.a.b:first", "memory.a.c:second"]]


def test_the_library_usage_table_can_list_every_item(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ids = [f"memory.p.m:f{i:03d}" for i in range(MAX_TABLE_ROWS + 5)]
    head = usage_table(ev, [], ids)
    full = usage_table(ev, [], ids, max_rows=None)
    assert "more items not shown" in head and "more items not shown" not in full
    assert all(i in full for i in ids)


def test_write_asks_for_a_linked_when_to_use_note_without_rules_about_ids():
    text = pv.write_brief_now()
    assert (
        "linked note says when to use it, its preconditions and its known failures"
        in text
    )
    assert pv.benchmark_words(text) == [] and pv.example_checks(text) == []


def _with_replay(monkeypatch):
    """The replay sentence appears only when the replay tool is registered for the writer."""
    from unify.memory_v2 import tool_spec

    spec = tool_spec.ToolSpec(
        name="replay",
        description="Verify an item by replay.",
        parameters={"type": "object", "properties": {}},
        roles=frozenset({"write"}),
    )
    monkeypatch.setattr(tool_spec, "registry", lambda: (spec,))


def test_write_points_lifted_functions_to_replay_without_requiring_it(monkeypatch):
    """WRITE says how to verify a lifted function when the replay tool exists."""
    _with_replay(monkeypatch)
    text = pv.write_brief_now()
    assert (
        "listing in its manifest entry the code it `replaces` and calling `replay`"
        in text
    )
    assert (
        "must" not in text.split("can be verified by listing")[1].split("\n")[0]
    )  # informs, never forces
    assert pv.benchmark_words(text) == [] and pv.example_checks(text) == []
    assert "calling `replay`" not in pv.curate_brief_now()


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
