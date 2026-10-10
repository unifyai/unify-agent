"""Memory v2.1 r5 S3 and S4: measured use for the writer, same-shape functions for CURATE, the when-to-use note."""

from unify.memory_v2 import curate, prompts_v21 as pv
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.usage import MAX_TABLE_ROWS, usage_table


def test_curate_lists_library_functions_with_the_same_shape():
    bodies = {
        "memory.a.b:first": "def first(xs):\n    return [x * 2 for x in xs if x > 3]\n",
        "memory.a.c:second": "def second(items):\n    return [i * 5 for i in items if i > 9]\n",
        "memory.a.d:third": "def third(xs):\n    return sorted(xs)\n",
    }
    assert curate.same_shape(bodies) == [["memory.a.b:first", "memory.a.c:second"]]


def test_the_v21_usage_table_can_list_every_item(tmp_path):
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
