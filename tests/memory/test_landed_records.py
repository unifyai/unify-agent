"""A procedure landed from a capture carries the capture's reader, signature and ``applies`` into its item record.

* ``land_capture`` records each landing in the harness's own capture store (``landed.json``): the procedure item, its
  reader, the item that is its stated ``applies`` and the episode. No box ever sees it.
* The item records take ``landed_from``, ``signature`` and ``applies`` from the landings (``captures.landed_items``);
  a function no landing names keeps its previous record's.
* The exports read a function's one reader: the landing's, else the gate's when it records exactly one; never one
  picked from several.

Keyless; the landing runs code confined, so it needs bubblewrap and prlimit.
"""

from __future__ import annotations

import json

import pytest

from tests.memory.test_capture_end_to_end import recorded_episode
from tests.memory.test_land_capture import ROWS, _land, _store
from tests.memory.test_tool_channel_box import needs_box
from unify.memory import captures
from unify.memory import library_export as lx
from unify.memory import lifecycle as lc
from unify.memory.item_records import empty_record

PROC = "memory.readings.procedures:row_sums"
READER = "memory.readings.readers:read_row_sums"
APPLIES = "memory.readings.procedures:_row_sums_applies"
OTHER = "memory.readings.totals:column_totals"
#: The capture's applicability signature as the capture step writes it (identifiers already removed).
SIGNATURE = {
    "input": {"kind": "list", "items": {"kind": "list", "items": {"kind": "int"}}},
    "constraints": {"applies": {"stated": True}},
}


def _records(landed: dict, prev: dict | None = None, verification: dict | None = None) -> dict:
    sha = "a" * 40
    items = {PROC: "function", OTHER: "function"}
    recs, _ = lc.next_records(
        items=items,
        prev=prev or {},
        facts=[],
        since={i: {sha} for i in items},
        changed_at={i: sha for i in items},
        functions={PROC: "memory.readings.procedures", OTHER: "memory.readings.totals"},
        graph={"memory.readings.procedures": set(), "memory.readings.totals": set()},
        note_uses={},
        verification=verification,
        landed=landed,
    )
    return recs


@needs_box
@pytest.mark.timeout(600)
def test_a_landing_is_recorded_in_the_store_and_carried_into_its_item_record(tmp_path):
    ep = recorded_episode("e7", rows=ROWS)
    store = tmp_path / "store"
    cid = _store(store)
    (store / cid / captures.SIGNATURE_FILE).write_text(json.dumps(SIGNATURE), encoding="utf-8")
    reply = _land(tmp_path, {"e7": ep}, capture_id=cid)
    assert reply.status == "ok", reply.text
    # the test's landing names no pass; a writer pass's names its own (test_capture_one_route)
    landing = {"item": PROC, "reader": READER, "applies": APPLIES, "episode": "e7", "pass": None,
               "verified": {"ran": False, "why": "rev2: a capture is never replayed to verify it; its tests are written "
                                                 "from its code"}}  # fmt: skip
    got = captures.landings(store)
    assert set(got) == {cid} and isinstance(got[cid].pop("at"), float)
    # rev 2: nothing landed runs to decide its listing; the landing says so explicitly
    recognition = got[cid].pop("recognition")
    assert got[cid] == landing
    assert recognition == {"ran": False, "why": "rev2: a landed procedure is never run to decide its listing"}
    landed = captures.landed_items(store)
    assert landed == {
        PROC: {
            "capture": cid,
            "reader": READER,
            "signature": SIGNATURE,
            "applies": APPLIES,
            "recognition": recognition,
        },
    }
    # the harness's own record: never in the box, the writer's inputs or the actor's snapshot
    assert not [p for p in (tmp_path / "box").rglob("*") if p.name == captures.LANDED_FILE]
    assert not [p for p in captures.export_files(store) if p.endswith(captures.LANDED_FILE)]
    recs = _records(landed, verification={PROC: {"readers": [READER]}, OTHER: {"readers": []}})
    assert recs[PROC]["landed_from"] == {"capture": cid, "reader": READER}
    assert recs[PROC]["signature"] == SIGNATURE and recs[PROC]["applies"] == APPLIES
    assert recs[PROC]["recognition"] == recognition
    assert (recs[OTHER]["landed_from"], recs[OTHER]["signature"], recs[OTHER]["applies"]) == (None, None, None)
    assert recs[OTHER]["recognition"] is None


def test_a_function_no_landing_names_keeps_its_previous_records_fields():
    prev = {PROC: {**empty_record(PROC, "function"), "landed_from": {"capture": "c" * 16, "reader": READER},
                   "signature": SIGNATURE, "applies": APPLIES}}
    recs = _records({}, prev=prev)
    assert recs[PROC]["landed_from"] == {"capture": "c" * 16, "reader": READER}
    assert recs[PROC]["signature"] == SIGNATURE and recs[PROC]["applies"] == APPLIES


def test_landed_items_skips_a_landing_without_an_item_and_reads_no_store_as_none(tmp_path):
    store = tmp_path / "store"
    cid = "0123456789abcdef"
    captures.record_landing(store, cid, {"item": "", "reader": READER})
    assert captures.landings(store) == {} and captures.landed_items(store) == {}
    captures.record_landing(store, cid, {"item": PROC, "reader": READER, "pass": "p9", "extra": "dropped"})
    want = {
        "item": PROC, "reader": READER, "applies": None, "episode": None, "pass": "p9", "recognition": None,
        "verified": None, "at": None,
    }
    assert captures.landings(store)[cid] == want
    assert captures.landed_items(None) == {} and captures.landings(tmp_path / "missing") == {}
    with pytest.raises(ValueError):
        captures.record_landing(store, "../escape", {"item": PROC})


def test_the_exports_read_one_reader_never_one_picked_from_several():
    gate_one = _records({}, verification={PROC: {"readers": [READER]}, OTHER: {"readers": []}})
    assert lx._reader_of(gate_one[PROC]) == READER and lx._reader_of(gate_one[OTHER]) is None
    landed = {PROC: {"capture": "c" * 16, "reader": READER, "signature": SIGNATURE, "applies": APPLIES}}
    two = _records(landed, verification={PROC: {"readers": [READER, "memory.readings.readers:read_other"]}})
    assert lx._reader_of(two[PROC]) == READER  # the landing's reader, not one of the gate's two
    unlanded_two = _records({}, verification={PROC: {"readers": [READER, "memory.readings.readers:read_other"]}})
    assert lx._reader_of(unlanded_two[PROC]) is None
    assert lx._no_reader(unlanded_two[PROC]).startswith("none of its 2 (memory.readings.readers:read_other, ")
    # a record that still holds one id is read as it is
    assert lx._reader_of({"reader": READER}) == READER and lx._reader_of({"reader": ""}) is None
