"""Implicit use signals and the library-use table (rev 2's attention pass's first message is test_attention_tools's)."""

from __future__ import annotations

from tests.memory.test_use_telemetry import ITEMS, _index, _record
from unify.memory import usage
from unify.memory.evidence import EvidenceStore
from tests.memory import store_reads


def test_item_signals_flag_used_never_used_and_refusing_accepted_inputs(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    items = ITEMS + ["env/x:extra"]
    _index(
        ev,
        "e1",
        _record(
            {"env/x:parse": {"called": 1, "refused": 1, "refused_then_accepted": 1}},
            pin=items,
            unknown={"*": 1},
            shown=ITEMS,
            channels=["x"],
        ),
        "2026-10-08T01:00:00Z",
    )
    _index(
        ev,
        "e2",
        _record({"env/x:strict": {"referenced": 1}}, pin=items, channels=["x"]),
        "2026-10-08T02:00:00Z",
    )
    # a request that edited channel x: its refusals are not the stored item's
    _index(
        ev,
        "e3",
        _record(
            {"env/x:lookup": {"refused_modified": 2, "errored_modified": 1}},
            pin=items,
            modified=["x"],
        ),
        "2026-10-08T03:00:00Z",
    )
    parse = usage.item_signals("env/x:parse", ev)
    assert parse["used"] and parse["refusing_accepted_inputs"]
    assert not parse["never_used"] and parse["uncertain"]
    assert parse["last_call_seq"] == store_reads.seq_of(ev, "e1")
    lookup = usage.item_signals("env/x:lookup", ev)
    assert lookup["never_used"] and lookup["never_used_basis"] == "item"
    assert (lookup["refusals"], lookup["errors"]) == (0, 0)
    assert lookup["refusals_report"] == "refusals 0 of 3 known (+0 unknown)"
    assert parse["refusals_report"] == "refusals 1 of 3 known (+0 unknown)"
    assert parse["refused_accepted"] == parse["refused_accepted_at_least"] == 1
    assert (
        lookup["requests_at_pin"],
        lookup["requests_shown"],
        lookup["requests_channel_shown"],
    ) == (3, 1, 2)
    strict = usage.item_signals("env/x:strict", ev)
    assert not strict["used"] and not strict["never_used"]  # referenced, never called
    # only its channel was ever shown: never used, on that basis
    extra = usage.item_signals("env/x:extra", ev)
    assert extra["never_used"] and extra["never_used_basis"] == "channel"
    assert extra["requests_shown"] == 0 and extra["requests_at_pin"] == 3
    # held at the pin but never shown, not even by channel: not "never used"
    _index(ev, "e4", _record({}, pin=["env/q:hidden"]), "2026-10-08T04:00:00Z")
    hidden = usage.item_signals("env/q:hidden", ev)
    assert not hidden["never_used"] and hidden["never_used_basis"] is None
    assert hidden["requests_at_pin"] == 1
    unseen = usage.item_signals("env/y:nothing", ev)
    assert not unseen["never_used"] and unseen["requests_at_pin"] == 0


def test_signals_and_the_table_say_where_the_shown_counts_came_from(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    for n, source in enumerate(("record", "legacy_text", "unknown")):
        _index(
            ev,
            f"e{n}",
            _record({}, shown=ITEMS, channels=["x"], source=source),
            f"2026-10-08T0{n}:00:00Z",
        )
    sig = usage.item_signals("env/x:parse", ev)
    assert sig["exposure_sources"] == {"record": 1, "legacy_text": 1, "unknown": 1}
    table = usage.usage_table(ev, ["e0", "e1", "e2"], ITEMS)
    assert table.splitlines()[-1] == (
        "(shown counts: 1 requests from the harness's record of what it showed, 1 read from the "
        "prompt's text (legacy), 1 unknown and counted as showing nothing)"
    )
    # every request from the record: no note, the table is as before
    only = usage.usage_table(ev, ["e0"], ITEMS)
    assert "shown counts" not in only
    assert only.splitlines()[-1].startswith("env/x:strict | ")
