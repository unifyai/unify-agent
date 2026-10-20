"""The run record's, the writer service's and metering's switches: empty is off (on under the redesign profile), each
is refused without what it needs, naming the settings, and the cap and the lease are parsed strictly."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from unify.memory.integration import switch as S


def _s(**kw):
    return SimpleNamespace(**kw)


def test_empty_is_off_and_the_profile_turns_them_on():
    assert not S.run_record_on(_s()) and not S.writer_service_on(_s()) and not S.metering_on(_s())
    on = _s(UNIFY_MEMORY_PROFILE="redesign", UNIFY_MEMORY="on")
    assert S.run_record_on(on) and S.writer_service_on(on) and S.metering_on(on)
    # an explicit off stays off (for the profile's own check to refuse)
    assert not S.run_record_on(_s(UNIFY_MEMORY_PROFILE="redesign", UNIFY_AGENTS_RUN_RECORD="off"))
    # with memory off the profile gives the memory switches nothing, and still the run record
    nomem = _s(UNIFY_MEMORY_PROFILE="redesign")
    assert S.run_record_on(nomem) and not S.writer_service_on(nomem) and not S.metering_on(nomem)


def test_the_run_record_needs_the_tools_in_code():
    assert "UNIFY_TOOLS_IN_CODE=on" in S.redesign_a_refusal(_s(UNIFY_AGENTS_RUN_RECORD="on"))
    assert S.redesign_a_refusal(_s(UNIFY_AGENTS_RUN_RECORD="on", UNIFY_TOOLS_IN_CODE="on")) is None
    assert "UNIFY_TOOLS_IN_CODE=on" in S.profile_refusal(_s(UNIFY_AGENTS_RUN_RECORD="on"))


def test_the_service_needs_the_library_the_ledger_its_cap_and_the_lease_and_not_the_record(monkeypatch):
    monkeypatch.delenv(S.HOST_LEASE_END, raising=False)
    got = S.redesign_a_refusal(_s(UNIFY_MEMORY_WRITER_SERVICE="on"))
    for name in (
        "UNIFY_MEMORY_LIBRARY=on",
        "UNIFY_MEMORY_ATTENTION_LEDGER=on",
        "UNIFY_MEMORY_WRITER_SERVICE_CAP_USD",
        "UNIFY_HOST_LEASE_END",
    ):
        assert name in got, name
    full = _s(
        UNIFY_MEMORY_WRITER_SERVICE="on",
        UNIFY_MEMORY_LIBRARY="on",
        UNIFY_MEMORY_ATTENTION_LEDGER="on",
        UNIFY_AGENTS_RUN_RECORD="on",
        UNIFY_TOOLS_IN_CODE="on",
        UNIFY_MEMORY_WRITER_SERVICE_CAP_USD="12.5",
        UNIFY_HOST_LEASE_END="2026-10-21T06:00:00Z",
    )
    assert S.redesign_a_refusal(full) is None
    # fail closed: no lease (or one that does not parse) refuses the service by name
    for lease in ("", "tomorrow"):
        got = S.redesign_a_refusal(_s(**{**vars(full), "UNIFY_HOST_LEASE_END": lease}))
        assert got == "UNIFY_MEMORY_WRITER_SERVICE=on needs UNIFY_HOST_LEASE_END; nothing runs", got
    assert S.writer_service_cap_usd(full) == Decimal("12.5")
    # the mesh-off arm: the service runs without the run record (its passes keep their own records)
    mesh_off = _s(**{**vars(full), "UNIFY_AGENTS_RUN_RECORD": "off"})
    assert "UNIFY_AGENTS_RUN_RECORD" not in got and S.redesign_a_refusal(mesh_off) is None


def test_metering_needs_the_request_metadata_headers():
    assert "UNIFY_REQUEST_METADATA_HEADERS=on" in S.redesign_a_refusal(_s(UNIFY_MEMORY_METERING="on"))
    assert S.redesign_a_refusal(_s(UNIFY_MEMORY_METERING="on", UNIFY_REQUEST_METADATA_HEADERS="on")) is None


def test_the_cap_and_the_lease_are_parsed_strictly(monkeypatch):
    for bad in ("0", "-1", "1e3", "abc", "1.2.3"):
        with pytest.raises(ValueError):
            S.parse_writer_service_cap(bad)
    assert S.parse_writer_service_cap("") == "" and S.parse_writer_service_cap(" 3.25 ") == "3.25"
    for bad in ("2026-10-21 06:00:00", "2026-10-21T06:00:00", "2026-13-01T00:00:00Z", "tomorrow"):
        with pytest.raises(ValueError):
            S.parse_host_lease_end(bad)
    assert S.host_lease_end(_s(UNIFY_HOST_LEASE_END="2026-10-21T06:00:00Z")) == datetime(
        2026, 10, 21, 6, 0, tzinfo=timezone.utc
    )
    monkeypatch.delenv(S.HOST_LEASE_END, raising=False)
    assert S.host_lease_end(_s()) is None
    monkeypatch.setenv(S.HOST_LEASE_END, "2026-10-21T06:00:00.500Z")
    assert S.host_lease_end(_s()).microsecond == 500000


def test_every_new_setting_has_its_parser():
    from unify.settings import ProductionSettings

    for name in (S.RUN_RECORD, S.WRITER_SERVICE, S.WRITER_SERVICE_CAP, S.METERING, S.HOST_LEASE_END):
        assert name in ProductionSettings.model_fields and name in S.PARSERS
    assert S.redesign_a_refusal in S.REDESIGN_PART_REFUSALS
