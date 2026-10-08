"""Symbolic: the agent record's tunables are the documented defaults."""

from unify.agents.options import Options


def test_defaults_are_the_documented_ones():
    assert Options() == Options(
        delivery="mentions",
        max_live=4,
        max_total=8,
        deliver_max_tokens=4000,
        others_line=True,
        max_entries=5000,
        min_post_interval_s=1.0,
    )
