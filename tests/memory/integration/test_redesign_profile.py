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
    UNIFY_TOOLS_IN_CODE="on",  # the run record, the writer's tools and tools.checkpoint are called from code
    UNIFY_MEMORY_WRITER_SERVICE_CAP_USD="10",  # the writer service's own cap, which the kit sets
    UNIFY_HOST_LEASE_END="2026-10-21T06:00:00Z",  # and the host's lease end, also the kit's
    UNIFY_REQUEST_METADATA_HEADERS="on",  # metering's ids travel as request metadata headers
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
        "UNIFY_MEMORY_WRITER_TESTS": "value",
        "UNIFY_MEMORY_IN_EPISODE": "off",  # rev 2 (MAIN, 21 Oct): the main arm runs without in-episode memory
        "UNIFY_MEMORY_SERVING_FAILURE_RECORD": "on",
        "UNIFY_MEMORY_GATE_LABELS_ONLY": "on",  # rev 2: labels only, only pure items run
        "UNIFY_MEMORY_GATE_RECORDED_REPLAYS": "off",
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
    assert switch.profile_setting(on, "UNIFY_MEMORY_WRITER_TESTS") == "value"
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
    off = SimpleNamespace(UNIFY_MEMORY="", UNIFY_MEMORY_PROFILE="redesign", UNIFY_TOOLS_IN_CODE="on")
    assert switch.profile_setting(off, "UNIFY_AGENTS_RUN_RECORD") == "on"
    for name in switch.REDESIGN_SWITCHES:
        if name.startswith("UNIFY_MEMORY_"):
            assert switch.profile_setting(off, name) == "", name
    assert not switch.library_withdraw_closed_captures(off)
    assert switch.profile_refusal(off) is None
    # the run record is on in this arm too, and every agent reaches it from code
    assert "UNIFY_TOOLS_IN_CODE=on" in switch.profile_refusal(
        SimpleNamespace(UNIFY_MEMORY="", UNIFY_MEMORY_PROFILE="redesign")
    )


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
        UNIFY_MEMORY_WRITER_TESTS=" Value",
        UNIFY_MEMORY_LEDGER_K=4,
        UNIFY_MEMORY_REGIME="inferred",
        UNIFY_MEMORY_LIBRARY_WITHDRAW_CLOSED_CAPTURES="on",
    )
    assert switch.redesign_refusal(same) is None


@pytest.mark.parametrize(
    "name, value",
    [
        ("UNIFY_MEMORY_LIBRARY_REVIEW_EACH_CAPTURE", "on"),
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


def test_every_arm_is_a_settings_set_the_build_runs():
    """The evaluation's arms (switch.REDESIGN_ARMS) as kits pin them: each with its companions, the library settings,
    the service's cap and the host's lease is refused by nothing (holds once every element is merged); and each
    differs from the profile's switches only where its change and that change's own dependencies say."""
    kit = {**_LIBRARY, **switch.REDESIGN_COMPANIONS, "UNIFY_HOST_LEASE_END": "2026-10-21T06:00:00Z"}
    for arm, values in switch.REDESIGN_ARMS.items():
        settings = SimpleNamespace(**({**kit, **values} if arm != "no_memory" else {**switch.REDESIGN_COMPANIONS,
                                                                                     **values}))
        assert switch.profile_refusal(settings) is None, (arm, switch.profile_refusal(settings))
    assert switch.REDESIGN_ARMS["redesign"] == {"UNIFY_MEMORY_PROFILE": "redesign"}
    changed = {
        arm: sorted(k for k, v in values.items() if switch.REDESIGN_SWITCHES.get(k) != v)
        for arm, values in switch.REDESIGN_ARMS.items()
        if arm not in ("redesign", "no_memory")
    }
    assert changed == {
        "captures_only": ["UNIFY_MEMORY_WRITER_PASSES", "UNIFY_MEMORY_WRITER_SERVICE"],
        "ledger_off": [
            "UNIFY_MEMORY_ATTENTION_LEDGER",
            "UNIFY_MEMORY_LEDGER_K",
            "UNIFY_MEMORY_LIBRARY_CAPTURES_PER_PASS",
            "UNIFY_MEMORY_LIBRARY_REVIEW_EACH_CAPTURE",
            "UNIFY_MEMORY_WRITER_SERVICE",
        ],
        "in_episode_on": ["UNIFY_MEMORY_IN_EPISODE"],
        "in_episode_repin": ["UNIFY_MEMORY_IN_EPISODE"],
        "failure_record_off": ["UNIFY_MEMORY_SERVING_FAILURE_RECORD"],
        "mesh_off": ["UNIFY_AGENTS_RUN_RECORD"],
    }
    # the part the ledger_off arm turns off with its change is itself refused when left on
    for arm, part in (("ledger_off", "UNIFY_MEMORY_WRITER_SERVICE"),):
        kept = SimpleNamespace(**{**kit, **switch.REDESIGN_ARMS[arm], part: "on"})
        assert switch.profile_refusal(kept) is not None, arm


def test_the_writer_tests_replay_mode_is_refused_until_its_review_is_recorded(monkeypatch):
    """The profile gives ``value`` only, and ``value,replay`` is refused at start-up (with or without the
    profile) while the replay stand-in's confinement review (G2) is not recorded; once it is, replay is allowed."""
    from unify.memory import writer_tests

    kit = {**_LIBRARY, **switch.REDESIGN_COMPANIONS, "UNIFY_HOST_LEASE_END": "2026-10-21T06:00:00Z"}
    assert switch.REDESIGN_SWITCHES["UNIFY_MEMORY_WRITER_TESTS"] == "value"
    monkeypatch.setattr(writer_tests, "REPLAY_REVIEWED", False)
    for extra in ({"UNIFY_MEMORY_PROFILE": "redesign"}, {**switch.REDESIGN_ARMS["in_episode_on"]}):
        replay = SimpleNamespace(**{**kit, **extra, "UNIFY_MEMORY_WRITER_TESTS": "value,replay"})
        why = switch.profile_refusal(replay)
        assert why is not None and "UNIFY_MEMORY_WRITER_TESTS" in why, why
        assert switch.profile_refusal(SimpleNamespace(**{**kit, **extra, "UNIFY_MEMORY_WRITER_TESTS": "value"})) is None
    alone = SimpleNamespace(**{**kit, **switch.REDESIGN_ARMS["in_episode_on"], "UNIFY_MEMORY_WRITER_TESTS": "replay,value"})
    assert "replay stand-in's confinement review" in (switch.profile_refusal(alone) or "")
    monkeypatch.setattr(writer_tests, "REPLAY_REVIEWED", True)
    off_profile = SimpleNamespace(**{**kit, **switch.REDESIGN_ARMS["in_episode_on"], "UNIFY_MEMORY_WRITER_TESTS": "value,replay"})
    assert switch.profile_refusal(off_profile) is None


def test_fixed_answers_as_examples_is_part_of_the_capture_contract_under_rev2():
    """Rev 2: constants are served as worked examples under the trace contract, so the profile no longer refuses the
    switch, and the accessor the request reads is on with the profile whatever the switch says."""
    assert "UNIFY_MEMORY_LIBRARY_FIXED_ANSWERS_AS_EXAMPLES" not in switch.REDESIGN_CONFLICTS
    on = _s(UNIFY_MEMORY_PROFILE="redesign", UNIFY_MEMORY_LIBRARY_FIXED_ANSWERS_AS_EXAMPLES="on")
    assert "FIXED_ANSWERS" not in (switch.redesign_refusal(on) or "")
    assert switch.constant_captures_as_examples(_s(UNIFY_MEMORY="on", UNIFY_MEMORY_PROFILE="redesign"))
    assert not switch.constant_captures_as_examples(_s())


def test_in_episode_takes_repin_for_re_pins_without_segment_forks():
    """Rev 2, item 10(b): ``repin`` re-pins the episode's copy at cell boundaries and starts no segment fork; it needs
    the library and the ledger's service, not the episode-end fork or the checkpoint tool."""
    assert switch.parse_in_episode(" Repin ") == "repin"
    with pytest.raises(ValueError, match="'repin'"):
        switch.parse_in_episode("sometimes")
    repin = _s(UNIFY_MEMORY_IN_EPISODE="repin", UNIFY_MEMORY_ATTENTION_LEDGER="on")
    assert switch.in_episode_repins(repin) and not switch.in_episode(repin)
    on = _s(UNIFY_MEMORY_IN_EPISODE="on")
    assert switch.in_episode_repins(on) and switch.in_episode(on)
    main = _s(UNIFY_MEMORY_PROFILE="redesign")
    assert not switch.in_episode_repins(main) and not switch.in_episode(main)
    no_ledger = _s(UNIFY_MEMORY_IN_EPISODE="repin", UNIFY_MEMORY_ATTENTION_LEDGER="off")
    assert "UNIFY_MEMORY_ATTENTION_LEDGER=on" in (switch.redesign_b_refusal(no_ledger) or "")
