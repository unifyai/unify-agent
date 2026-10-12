"""Persistent item identity: a uid given at an item's first record, carried across renames and aliases, with
use, status episodes and source episodes joined through it."""

from tests.memory.test_gate_curate_checks import BEFORE, WHY, _checks, _reasons
from tests.memory.test_overlap import SPLIT, SPLIT_ID, SPLIT_PATH, TOKENS
from tests.memory.test_gate_library_checks import _item
from unify.memory import lifecycle as lc
from unify.memory.evidence import EvidenceStore
from unify.memory.item_records import empty_record, new_uid

WEEK = "memory.text.dates:week"
WEEKDAY = "memory.text.calendar:weekday"
DAY = "memory.text.days:weekday"
A, B, C = "a" * 40, "b" * 40, "c" * 40
OK = lc.ItemUse(True, 1, False, False, False)
RENAMED_SPLIT = "memory.text.words:split_words"
RENAMED = {
    **{k: v for k, v in BEFORE.items() if k != SPLIT_PATH},
    "memory/text/words.py": SPLIT,
}


def _records(
    items,
    prev=None,
    facts=(),
    curations=(),
    aliases=None,
    changed_at=None,
    provenance=None,
):
    changed_at = changed_at or dict.fromkeys(items, B)
    return lc.next_records(
        items=items,
        prev=prev or {},
        facts=list(facts),
        since={i: {A, B, C} for i in items},
        changed_at=changed_at,
        functions={i: i.split(":")[0] for i in items},
        graph={i.split(":")[0]: set() for i in items},
        note_uses={},
        provenance=provenance,
        aliases=aliases,
        curations=list(curations),
    )[0]


def _prev(item, uid, *, at=A, lineage=()):
    return {
        **empty_record(item, "function"),
        "uid": uid,
        "changed_at": at,
        "renamed_from": list(lineage),
    }


def _rename(old, new):
    return {"item": old, "action": "rename", "target": new, "reason": ""}


def test_the_first_record_assigns_a_uid_and_later_maps_keep_it():
    first = _records({WEEK: "function"}, changed_at={WEEK: A})[WEEK]
    assert first["uid"] == new_uid(WEEK, A) and first["renamed_from"] == []
    # a new version of the item changes changed_at but never its identity
    later = _records({WEEK: "function"}, prev={WEEK: first}, changed_at={WEEK: B})[WEEK]
    assert later["uid"] == first["uid"]


def test_a_record_written_before_uids_gets_the_uid_its_first_record_would_have():
    old = {**empty_record(WEEK, "function"), "changed_at": A}
    old.pop("uid")
    assert _records({WEEK: "function"}, prev={WEEK: old})[WEEK]["uid"] == new_uid(
        WEEK,
        A,
    )


def test_a_rename_keeps_the_uid_and_records_the_lineage():
    recs = _records(
        {WEEKDAY: "function"},
        prev={WEEK: _prev(WEEK, "u1")},
        curations=[_rename(WEEK, WEEKDAY)],
    )
    assert recs[WEEKDAY]["uid"] == "u1" and recs[WEEKDAY]["renamed_from"] == [WEEK]
    # two renames between two maps, and a later map, keep the whole lineage
    twice = _records(
        {DAY: "function"},
        prev={WEEK: _prev(WEEK, "u1")},
        curations=[_rename(WEEK, WEEKDAY), _rename(WEEKDAY, DAY)],
    )[DAY]
    assert twice["uid"] == "u1" and twice["renamed_from"] == [WEEK, WEEKDAY]
    again = _records({DAY: "function"}, prev={DAY: twice})[DAY]
    assert again["uid"] == "u1" and again["renamed_from"] == [WEEK, WEEKDAY]


def test_use_and_source_episodes_before_a_rename_count_after_it():
    facts = [
        lc.EpisodeUse("e1", A, "dense", False, False, {WEEK: OK}),
        lc.EpisodeUse("e2", A, "dense", False, False, {WEEK: OK}),
        lc.EpisodeUse("e3", C, "dense", False, False, {WEEKDAY: OK}),
    ]
    rec = _records(
        {WEEKDAY: "function"},
        prev={WEEK: _prev(WEEK, "u1")},
        facts=facts,
        curations=[_rename(WEEK, WEEKDAY)],
        provenance={
            WEEK: {"episodes": ["e0"], "pass": "p1"},
            WEEKDAY: {"episodes": ["e3"], "pass": "p2"},
        },
    )[WEEKDAY]
    assert rec["use"][A]["episodes"] == 2 and rec["use"][C]["episodes"] == 1
    assert sorted(rec["provenance"]["episodes"]) == ["e0", "e3"]
    # the status rules count the old id's episodes as the item's: three clean uses make it stable
    assert rec["status"] == "stable"


def test_an_alias_target_carries_the_old_items_uid():
    recs = _records(
        {SPLIT_ID: "function", TOKENS: "function"},
        prev={SPLIT_ID: _prev(SPLIT_ID, "u1")},
        aliases={SPLIT_ID: {"target": TOKENS, "pass_id": "p", "commit": B}},
        facts=[lc.EpisodeUse("e1", A, "dense", False, False, {SPLIT_ID: OK})],
    )
    assert recs[TOKENS]["uid"] == "u1" and recs[TOKENS]["renamed_from"] == [SPLIT_ID]
    assert recs[SPLIT_ID]["uid"] == "u1" and recs[SPLIT_ID]["alias_of"] == TOKENS
    assert recs[TOKENS]["use"][A]["episodes"] == 1


def test_an_existing_alias_target_keeps_its_own_uid():
    recs = _records(
        {SPLIT_ID: "function", TOKENS: "function"},
        prev={SPLIT_ID: _prev(SPLIT_ID, "u1"), TOKENS: _prev(TOKENS, "u2")},
        aliases={SPLIT_ID: {"target": TOKENS}},
    )
    assert recs[TOKENS]["uid"] == "u2" and recs[SPLIT_ID]["uid"] == "u1"


def test_a_write_pass_rename_is_recorded_for_landing(tmp_path):
    v = _checks(
        tmp_path,
        BEFORE,
        RENAMED,
        {"renames": {SPLIT_ID: RENAMED_SPLIT}},
        role="write",
    )
    v.curate()
    assert v.run.fails == []
    assert v.run.curations == [_rename(SPLIT_ID, RENAMED_SPLIT)]


def test_a_curate_pass_rename_needs_no_alias_or_retirement(tmp_path):
    v = _checks(
        tmp_path,
        BEFORE,
        RENAMED,
        {"why": WHY, "renames": {SPLIT_ID: RENAMED_SPLIT}},
        items=[_item(item=RENAMED_SPLIT)],
        deleted=[SPLIT_ID],
    )
    v.curate()
    assert v.run.fails == []
    assert v.run.curations == [_rename(SPLIT_ID, RENAMED_SPLIT)]


def test_a_rename_of_an_unknown_id_or_to_a_missing_id_is_refused(tmp_path):
    v = _checks(
        tmp_path,
        BEFORE,
        RENAMED,
        {
            "renames": {
                "memory.text.nowhere:gone": RENAMED_SPLIT,
                SPLIT_ID: "memory.text.words:missing",
            },
        },
        role="write",
    )
    v.curate()
    assert _reasons(v) == [
        "rename of memory.text.nowhere:gone: it is not an item of the parent",
        f"rename of {SPLIT_ID} to memory.text.words:missing: memory.text.words:missing is not an item of the "
        "candidate",
    ]
    assert v.run.curations == []


def test_a_rename_onto_an_existing_item_or_of_a_malformed_map_is_refused(tmp_path):
    v = _checks(tmp_path, BEFORE, BEFORE, {"renames": {SPLIT_ID: TOKENS}}, role="write")
    v.curate()
    assert _reasons(v) == [
        f"rename of {SPLIT_ID} to {TOKENS}: {TOKENS} is already an item of the parent; to merge two items, "
        "keep the old name as an alias (CURATE)",
    ]
    bad = _checks(
        tmp_path / "bad",
        BEFORE,
        BEFORE,
        {"renames": [SPLIT_ID]},
        role="write",
    )
    bad.curate()
    assert _reasons(bad) == ["renames maps old item ids to new item ids (strings)"]


def test_rename_rows_are_stored_and_are_not_aliases(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ev.record_curations("p1", B, [_rename(WEEK, WEEKDAY)])
    assert [r["action"] for r in ev.curations()] == ["rename"]
    assert ev.aliases() == {}
