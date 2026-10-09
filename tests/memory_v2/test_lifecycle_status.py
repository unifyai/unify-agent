"""Statuses (spec §10.1), taint (§10.1(c), §11 rule 3) and failure-only provenance (§8.2 rule 3, §11 rule 1)."""

from __future__ import annotations

from tests.memory_v2.test_episodes import _ep
from unify.memory_v2 import lifecycle as lc
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.item_records import empty_record

PD, WEEK, TOK = (
    "memory.text.dates:parse_date",
    "memory.text.dates:week",
    "memory.text.parse:tokens",
)
WORDS, SUM, GET = (
    "memory.text.parse:words",
    "memory.text.report:summary",
    "memory.web.fetch:get",
)
NOTE = "notes/text/dates.md"
A, B = "a" * 40, "b" * 40
OK = lc.ItemUse(True, 1, False, False, False)
RAISED = lc.ItemUse(True, 1, False, True, False)
GRAPH = {
    "memory.text.parse": set(),
    "memory.text.dates": {"memory.text.parse"},
    "memory.text.report": {"memory.text.dates"},
    "memory.web.fetch": set(),
}
FUNCS = {
    PD: "memory.text.dates",
    WEEK: "memory.text.dates",
    TOK: "memory.text.parse",
    WORDS: "memory.text.parse",
    SUM: "memory.text.report",
    GET: "memory.web.fetch",
}


def _f(eid, item, u, *, main=A, regime="dense", pos=False, neg=False):
    return lc.EpisodeUse(eid, main, regime, pos, neg, {item: u})


def test_one_episode_never_decides_a_status():
    assert lc.decide_status(WEEK, None, [_f("e1", WEEK, RAISED)])[0] == "experimental"
    two = [_f("e1", WEEK, RAISED), _f("e2", WEEK, RAISED)]
    assert lc.decide_status(WEEK, None, two)[:2] == ("suspect", "errors")
    assert (
        lc.decide_status(WEEK, None, [_f("e1", WEEK, OK, neg=True)])[0]
        == "experimental"
    )
    negative = [_f("e1", WEEK, OK, neg=True), _f("e2", WEEK, OK, neg=True)]
    assert lc.decide_status(WEEK, None, negative)[:2] == ("suspect", "negative_signals")
    assert (
        lc.decide_status(WEEK, None, [_f(f"e{i}", WEEK, OK) for i in range(2)])[0]
        == "experimental"
    )
    three = [_f(f"e{i}", WEEK, OK) for i in range(3)]
    assert lc.decide_status(WEEK, None, three) == (
        "stable",
        None,
        None,
        ["e0", "e1", "e2"],
    )
    assert (
        lc.decide_status(WEEK, None, three + [_f("e3", WEEK, RAISED)])[0]
        == "experimental"
    )  # an error blocks it
    assert (
        lc.decide_status(WEEK, "stable", three + [_f("e3", WEEK, RAISED)])[0]
        == "stable"
    )  # one error: kept
    demoted = three + [_f("e3", WEEK, RAISED), _f("e4", WEEK, RAISED)]
    assert lc.decide_status(WEEK, "stable", demoted)[0] == "suspect"
    assert (
        lc.decide_status(WEEK, "deprecated", demoted)[0] == "deprecated"
    )  # CURATE's, never the harness's


def test_in_a_no_signal_regime_only_errors_count():
    quiet = [_f(f"e{i}", WEEK, OK, regime="none", neg=True) for i in range(3)]
    assert lc.decide_status(WEEK, None, quiet)[0] == "stable"
    raised = [_f(f"r{i}", WEEK, RAISED, regime="none") for i in range(2)]
    assert lc.decide_status(WEEK, None, raised)[:2] == ("suspect", "errors")


def test_taint_reaches_importers_and_notes_but_not_siblings():
    status = {i: "experimental" for i in FUNCS} | {NOTE: "experimental", TOK: "suspect"}
    out = lc.taint(status, FUNCS, GRAPH, {NOTE: [PD]})
    why = f"depends on suspect {TOK}"
    assert out == {
        PD: (why, [TOK]),
        WEEK: (why, [TOK]),
        SUM: (why, [TOK]),
        NOTE: (f"uses suspect {PD}", [PD]),
    }  # WORDS (the same module) and GET (no import) are untouched


def test_failure_only_provenance_is_found_through_the_citing_items(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    for eid in ("e1", "e2", "e3"):
        ev.index_episode(_ep(episode_id=eid, regime="implicit"), "1" * 40)
    for item, eid in ((WEEK, "e1"), (WEEK, "e2"), (PD, "e1"), (TOK, "e1"), (TOK, "e3")):
        ev.add_item_evidence(item, eid, "source")
    neg = {e: lc.EpisodeUse(e, A, "implicit", False, True, {}) for e in ("e1", "e2")}
    facts = {**neg, "e3": lc.EpisodeUse("e3", A, "implicit", True, False, {})}
    assert lc.failure_only(ev, [WEEK, PD, TOK], facts) == {
        WEEK: ["e1", "e2"],
        PD: ["e1"],
    }
    ev.add_cover(
        WEEK,
        "e3",
        0,
    )  # a recorded input from a successful episode: its tests check a good value
    assert lc.failure_only(ev, [WEEK, PD, TOK], facts) == {PD: ["e1"]}


def test_next_records_restarts_a_new_version_and_reports_changes():
    items = {WEEK: "function", PD: "function", NOTE: "note"}
    prev = {
        WEEK: {**empty_record(WEEK, "function"), "status": "suspect", "changed_at": A},
        PD: {
            **empty_record(PD, "function"),
            "status": "deprecated",
            "alias_of": WEEK,
            "changed_at": A,
        },
    }
    base = dict(
        items=items,
        prev=prev,
        facts=[_f("e1", WEEK, RAISED), _f("e2", WEEK, RAISED)],
        functions={WEEK: "memory.text.dates", PD: "memory.text.dates"},
        graph={"memory.text.dates": set()},
        note_uses={NOTE: []},
    )
    every = {WEEK: {A, B}, PD: {A, B}, NOTE: {A, B}}
    same, changes = lc.next_records(
        **base,
        since=every,
        changed_at={WEEK: A, PD: A, NOTE: A},
    )
    assert (
        same[WEEK]["status"],
        same[WEEK]["status_rule"],
        same[WEEK]["status_evidence"],
    ) == (
        "suspect",
        "errors",
        ["e1", "e2"],
    )
    assert same[PD]["status"] == "deprecated" and same[PD]["alias_of"] == WEEK
    assert same[NOTE]["status"] == "experimental" and changes == {}
    fixed = {**every, WEEK: {B}}
    new, changes = lc.next_records(
        **base,
        since=fixed,
        changed_at={WEEK: B, PD: A, NOTE: A},
    )
    assert new[WEEK]["status"] == "experimental" and new[WEEK]["use"][A]["errors"] == 2
    assert changes == {WEEK: "experimental"}
    flagged, _ = lc.next_records(
        **{**base, "facts": []},
        since=fixed,
        changed_at={WEEK: B, PD: A, NOTE: A},
        failure={WEEK: ["e1"]},
    )
    assert (
        flagged[WEEK]["status"] == "experimental"
        and flagged[WEEK]["provenance"]["failure_only"] is True
    )
    hidden, _ = lc.next_records(
        **{**base, "facts": []},
        since=fixed,
        changed_at={WEEK: B, PD: A, NOTE: A},
        failure={WEEK: ["e1", "e2"]},
    )
    assert (hidden[WEEK]["status"], hidden[WEEK]["status_rule"]) == (
        "suspect",
        "failure_only",
    )


def test_curate_aliases_and_retirements_deprecate_items():
    """P6's decisions reach the records (MAIN, 9 Oct): a live alias is deprecated with alias_of its target; a
    retired item still in the tree is deprecated with its reason; a dropped alias no longer forwards.
    """
    items = {WEEK: "function", PD: "function", NOTE: "note"}
    prev = {
        PD: {
            **empty_record(PD, "function"),
            "status": "deprecated",
            "alias_of": WEEK,
            "changed_at": A,
        },
    }
    base = dict(
        items=items,
        prev=prev,
        facts=[],
        since={WEEK: {A}, PD: {A}, NOTE: {A}},
        changed_at={WEEK: A, PD: A, NOTE: A},
        functions={WEEK: "memory.text.dates", PD: "memory.text.dates"},
        graph={"memory.text.dates": set()},
        note_uses={NOTE: []},
    )
    live = {PD: {"target": WEEK, "pass_id": "c1", "commit": A}}
    retired = [
        {
            "pass_id": "c1",
            "commit": A,
            "item": NOTE,
            "action": "retire",
            "target": None,
            "reason": "stale",
        },
    ]
    recs, changes = lc.next_records(**base, aliases=live, curations=retired)
    assert (recs[PD]["status"], recs[PD]["alias_of"], recs[PD]["status_rule"]) == (
        "deprecated",
        WEEK,
        "curate",
    )
    assert (recs[NOTE]["status"], recs[NOTE]["status_reason"]) == (
        "deprecated",
        "retired: stale",
    )
    assert recs[WEEK]["status"] == "experimental" and recs[WEEK]["alias_of"] is None
    assert changes == {NOTE: "deprecated"}
    dropped, _ = lc.next_records(**base, aliases={}, curations=[])
    assert dropped[PD]["alias_of"] is None
