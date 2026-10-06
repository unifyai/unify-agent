"""Symbolic: UNIFY_AGENTS and UNIFY_AGENTS_OPTIONS parse strictly and default to as shipped."""

import pytest

from unify.agents.options import Options, parse_options
from unify.settings import ProductionSettings


def test_defaults_are_the_documented_ones():
    assert parse_options("") == Options()
    assert Options() == Options(
        delivery="mentions",
        max_live=4,
        max_total=8,
        deliver_max_tokens=4000,
        others_line=True,
        max_entries=5000,
        min_post_interval_s=1.0,
    )


def test_every_key_parses():
    got = parse_options(
        "delivery=all, max_live=2,max_total=0,deliver_max_tokens=900,"
        "others_line=0,max_entries=10,min_post_interval_s=0.5",
    )
    assert got == Options("all", 2, 0, 900, False, 10, 0.5)


@pytest.mark.parametrize(
    "text, words",
    [
        ("delivery=everyone", "delivery"),
        ("max_live=-1", "max_live"),
        ("nonsense=1", "unknown"),
        ("max_live", "key=value"),
        ("interrupt=1", "reserved"),
        ("cancel=boundary", "reserved"),
    ],
)
def test_bad_options_are_refused_with_the_reason(text, words):
    with pytest.raises(ValueError, match=words):
        parse_options(text)


def test_the_switch_accepts_only_empty_or_record():
    assert ProductionSettings(UNIFY_AGENTS="").UNIFY_AGENTS == ""
    assert ProductionSettings(UNIFY_AGENTS=" Record ").UNIFY_AGENTS == "record"
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_AGENTS="pool")
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_AGENTS_OPTIONS="delivery=everyone")


def test_enabled_follows_the_switch(monkeypatch):
    from unify import agents
    from unify.settings import SETTINGS

    monkeypatch.setattr(SETTINGS, "UNIFY_AGENTS", "")
    assert agents.enabled() is False
    monkeypatch.setattr(SETTINGS, "UNIFY_AGENTS", "record")
    assert agents.enabled() is True
