"""``UNIFY_MEMORY_PROFILE=redesign``: every switch of the redesign on together, and refused when one is set otherwise."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from unify.memory.integration import switch

#: The library settings every memory-on redesign run carries (what the switches act on).
_LIBRARY = dict(
    UNIFY_MEMORY="on",
    UNIFY_MEMORY_LIBRARY="on",
    UNIFY_MEMORY_LIBRARY_FORK="on",
    UNIFY_MEMORY_LIBRARY_POOL="on",
)


def _s(**kw):
    return SimpleNamespace(**{**_LIBRARY, **kw})


def test_the_profile_parses_its_values_and_is_off_by_default():
    assert switch.parse_memory_profile("") == ""
    assert switch.parse_memory_profile(" Redesign ") == "redesign"
    with pytest.raises(
        ValueError, match="UNIFY_MEMORY_PROFILE must be empty or 'redesign'"
    ):
        switch.parse_memory_profile("paper")
    assert switch.PARSERS[switch.MEMORY_PROFILE] is switch.parse_memory_profile
    assert not switch.redesign_profile(SimpleNamespace())
    assert switch.redesign_profile(_s(UNIFY_MEMORY_PROFILE="redesign"))


def test_the_profile_lists_exactly_the_planned_switches_and_values():
    assert switch.REDESIGN_SWITCHES == {
        "UNIFY_AGENTS_RUN_RECORD": "on",
        "UNIFY_MEMORY_WRITER_SERVICE": "on",
        "UNIFY_MEMORY_METERING": "on",
        "UNIFY_MEMORY_CAPTURE_TRACE": "on",
        "UNIFY_MEMORY_ATTENTION_LEDGER": "on",
        "UNIFY_MEMORY_LEDGER_K": "4",
        "UNIFY_MEMORY_IMPURE_ITEMS": "on",
        "UNIFY_MEMORY_WRITER_TESTS": "value,replay",
        "UNIFY_MEMORY_IN_EPISODE": "on",
        "UNIFY_MEMORY_SERVING_FAILURE_RECORD": "on",
        "UNIFY_MEMORY_LIBRARY_WITHDRAW_CLOSED_CAPTURES": "on",
        "UNIFY_MEMORY_REGIME": "inferred",
    }


def test_every_profile_switch_is_a_parsed_setting():
    """Each switch the profile gives is a real setting with its own parser (holds once every element is merged)."""
    from unify.settings import ProductionSettings

    for name in switch.REDESIGN_SWITCHES:
        assert name in ProductionSettings.model_fields, name
        assert name in switch.PARSERS, name


def test_an_empty_switch_takes_the_profiles_value_and_a_set_one_keeps_its_own():
    on = _s(UNIFY_MEMORY_PROFILE="redesign")
    for name, value in switch.REDESIGN_SWITCHES.items():
        assert switch.profile_setting(on, name) == switch._setting_text(value), name
        assert switch.profile_setting(SimpleNamespace(**_LIBRARY), name) == "", name
    assert switch.profile_setting(on, "UNIFY_MEMORY_WRITER_TESTS") == "replay,value"
    assert switch.profile_setting(on, "UNIFY_MEMORY_NOT_A_PROFILE_SWITCH") == ""
    # an explicit value wins, with or without the profile
    assert (
        switch.profile_setting(
            _s(UNIFY_MEMORY_SERVING_FAILURE_RECORD="off"),
            "UNIFY_MEMORY_SERVING_FAILURE_RECORD",
        )
        == "off"
    )
    assert switch.library_withdraw_closed_captures(on)
    assert not switch.library_withdraw_closed_captures(SimpleNamespace(**_LIBRARY))
    assert switch.library_withdraw_closed_captures(
        _s(UNIFY_MEMORY_LIBRARY_WITHDRAW_CLOSED_CAPTURES="on")
    )


def test_with_memory_off_the_profile_gives_the_memory_switches_nothing():
    """The no-memory arm under the profile: the run record stays, no memory switch is on, nothing is refused."""
    off = SimpleNamespace(UNIFY_MEMORY="", UNIFY_MEMORY_PROFILE="redesign")
    assert switch.profile_setting(off, "UNIFY_AGENTS_RUN_RECORD") == "on"
    for name in switch.REDESIGN_SWITCHES:
        if name.startswith("UNIFY_MEMORY_"):
            assert switch.profile_setting(off, name) == "", name
    assert not switch.library_withdraw_closed_captures(off)
    assert switch.profile_refusal(off) is None


def test_a_switch_set_otherwise_under_the_profile_is_refused_by_name():
    refused = switch.profile_refusal(
        _s(
            UNIFY_MEMORY_PROFILE="redesign",
            UNIFY_MEMORY_SERVING_FAILURE_RECORD="off",
            UNIFY_MEMORY_LEDGER_K="6",
        ),
    )
    assert refused == (
        "UNIFY_MEMORY_PROFILE=redesign turns on every switch of the redesign together; "
        "UNIFY_MEMORY_LEDGER_K, UNIFY_MEMORY_SERVING_FAILURE_RECORD set to another value (leave them empty or set "
        "the profile's value), so nothing runs"
    )
    # the profile's own value, set explicitly (in any order or case), is no conflict
    same = _s(
        UNIFY_MEMORY_PROFILE="redesign",
        UNIFY_MEMORY_WRITER_TESTS="Replay, value",
        UNIFY_MEMORY_LEDGER_K=4,
        UNIFY_MEMORY_REGIME="inferred",
        UNIFY_MEMORY_LIBRARY_WITHDRAW_CLOSED_CAPTURES="on",
    )
    assert switch.redesign_refusal(same) is None


@pytest.mark.parametrize(
    "name, value",
    [
        ("UNIFY_MEMORY_LIBRARY_REVIEW_EACH_CAPTURE", "on"),
        ("UNIFY_MEMORY_LIBRARY_FIXED_ANSWERS_AS_EXAMPLES", "on"),
        ("UNIFY_MEMORY_NOTEBOOK", "on"),
        ("UNIFY_MEMORY_PASSES", "off"),
        ("UNIFY_MEMORY_WRITER_PASSES", "off"),
    ],
)
def test_a_conflicting_setting_under_the_profile_is_refused(name, value):
    refused = switch.redesign_refusal(
        _s(UNIFY_MEMORY_PROFILE="redesign", **{name: value})
    )
    assert (
        refused
        == f"UNIFY_MEMORY_PROFILE=redesign cannot run with {name}={value}; unset them, so nothing runs"
    )
    assert (
        switch.redesign_refusal(_s(**{name: value})) is None
    )  # no profile: nothing to conflict with


def test_without_the_profile_nothing_changes():
    assert switch.redesign_refusal(_s()) is None
    assert switch.redesign_refusal(SimpleNamespace()) is None
    assert switch.profile_refusal(_s()) is None


def test_the_profiles_withdrawal_still_needs_the_pool():
    refused = switch.profile_refusal(
        _s(UNIFY_MEMORY_PROFILE="redesign", UNIFY_MEMORY_LIBRARY_POOL="off")
    )
    assert (
        refused
        == "UNIFY_MEMORY_LIBRARY_WITHDRAW_CLOSED_CAPTURES=on needs UNIFY_MEMORY_LIBRARY_POOL=on; nothing runs"
    )


def test_each_elements_refusal_runs_after_the_profiles_own(monkeypatch):
    seen = []

    def part(settings):
        seen.append(settings)
        return "an element's own reason"

    monkeypatch.setattr(switch, "REDESIGN_PART_REFUSALS", [part])
    on = _s(UNIFY_MEMORY_PROFILE="redesign")
    assert switch.redesign_refusal(on) == "an element's own reason" and seen == [on]
    assert switch.redesign_refusal(_s()) is None and seen == [
        on
    ]  # no profile: no element refusal is asked
    bad = _s(UNIFY_MEMORY_PROFILE="redesign", UNIFY_MEMORY_NOTEBOOK="on")
    assert "cannot run with UNIFY_MEMORY_NOTEBOOK=on" in switch.redesign_refusal(
        bad
    ) and seen == [on]


def test_the_settings_parse_the_profile_and_keep_an_empty_withdrawal_empty(monkeypatch):
    from unify.settings import ProductionSettings

    monkeypatch.setenv("UNIFY_MEMORY_PROFILE", "redesign")
    monkeypatch.delenv("UNIFY_MEMORY_LIBRARY_WITHDRAW_CLOSED_CAPTURES", raising=False)
    s = ProductionSettings()
    assert s.UNIFY_MEMORY_PROFILE == "redesign"
    assert s.UNIFY_MEMORY_LIBRARY_WITHDRAW_CLOSED_CAPTURES == ""
    monkeypatch.setenv("UNIFY_MEMORY_PROFILE", "other")
    with pytest.raises(
        Exception, match="UNIFY_MEMORY_PROFILE must be empty or 'redesign'"
    ):
        ProductionSettings()
