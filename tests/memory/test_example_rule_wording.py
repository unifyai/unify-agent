"""The example rule as the writer and the index read it: one episode is enough. An earlier rule asked for two
episodes, and writers read it as the same technique evidenced twice and stored no function at all. (The gate runs no
function's examples, and the index shows no breadth of them.)"""

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
