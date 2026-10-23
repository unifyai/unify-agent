"""The example rule as the writer and the index read it: one episode is enough. An earlier rule asked for two
episodes, and writers read it as the same technique evidenced twice and stored no function at all. (The gate no longer
runs a function's examples, rev 2.1; the index still shows the breadth a record kept from an earlier gate.)"""

from unify.memory import library_prompts as lp


def test_the_pool_text_gives_no_second_episode_reason_to_defer():
    from unify.common import tool_registry

    brief = lp.write_brief_now()
    assert "lacks only another episode" not in brief
    # rev 2: the decisions are stated by the pool tool's own help, which the brief points to
    assert 'tools.help("pool")' in brief
    pool_help = tool_registry.registry()["pool"].description
    assert "'retired' (reason; closes the entry for good)" in pool_help
    assert "lacks only another episode" not in pool_help


def test_a_function_with_examples_from_one_episode_says_so_in_the_index():
    from unify.memory.item_records import episodes_of
    from unify.memory.library_index import _breadth

    records = {
        "memory.a.b:one": {"verification": {"examples": {"episodes": ["e1"]}}},
        "memory.a.b:two": {"verification": {"examples": {"episodes": ["e1", "e2"]}}},
        "memory.a.b:old": {"verification": {}},
    }
    got = episodes_of(records)
    assert [
        got(i) for i in ("memory.a.b:one", "memory.a.b:two", "memory.a.b:old", "x")
    ] == [1, 2, None, None]
    assert _breadth(got, "memory.a.b:one") == ", examples from one episode"
    assert (
        _breadth(got, "memory.a.b:two")
        == _breadth(got, "memory.a.b:old")
        == _breadth(None, "x")
        == ""
    )


