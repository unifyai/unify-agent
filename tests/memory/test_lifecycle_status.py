"""Statuses, taint and failure-only provenance."""

from __future__ import annotations

from tests.memory.test_episodes import _ep
from unify.memory import lifecycle as lc
from unify.memory.evidence import EvidenceStore
from unify.memory.item_records import empty_record
from unify.memory.item_states import ADOPTED, CURATE_RULE, ON_TRIAL, RETIRED
from tests.memory import store_reads

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
    assert lc.decide_status(WEEK, None, [_f("e1", WEEK, RAISED)])[0] == ON_TRIAL
    two = [_f("e1", WEEK, RAISED), _f("e2", WEEK, RAISED)]
    assert lc.decide_status(WEEK, None, two)[:2] == (RETIRED, "errors")
    assert (
        lc.decide_status(WEEK, None, [_f("e1", WEEK, OK, neg=True)])[0]
        == ON_TRIAL
    )
    negative = [_f("e1", WEEK, OK, neg=True), _f("e2", WEEK, OK, neg=True)]
    assert lc.decide_status(WEEK, None, negative)[:2] == (RETIRED, "negative_signals")
    assert (
        lc.decide_status(WEEK, None, [_f(f"e{i}", WEEK, OK) for i in range(2)])[0]
        == ON_TRIAL
    )
    three = [_f(f"e{i}", WEEK, OK) for i in range(3)]
    assert lc.decide_status(WEEK, None, three) == (
        ADOPTED,
        None,
        None,
        ["e0", "e1", "e2"],
    )
    assert (
        lc.decide_status(WEEK, None, three + [_f("e3", WEEK, RAISED)])[0]
        == ON_TRIAL
    )  # an error blocks it
    assert (
        lc.decide_status(WEEK, ADOPTED, three + [_f("e3", WEEK, RAISED)])[0]
        == ADOPTED
    )  # one error: kept
    demoted = three + [_f("e3", WEEK, RAISED), _f("e4", WEEK, RAISED)]
    assert lc.decide_status(WEEK, ADOPTED, demoted)[0] == RETIRED
    # a curator's retirement is CURATE's, never the harness's: kept, with its rule, so it stays sticky
    assert lc.decide_status(WEEK, RETIRED, demoted, CURATE_RULE) == (RETIRED, CURATE_RULE, None, [])
    assert lc.decide_status(WEEK, RETIRED, three, CURATE_RULE)[:2] == (RETIRED, CURATE_RULE)
    # the old stored name of a curator's retirement is one too
    assert lc.decide_status(WEEK, "deprecated", three)[:2] == (RETIRED, CURATE_RULE)
    # a retirement by outcomes is decided again from the facts
    assert lc.decide_status(WEEK, RETIRED, three, "errors")[:2] == (ADOPTED, None)
    assert lc.decide_status(WEEK, "suspect", three, "errors")[:2] == (ADOPTED, None)
    # the old stored names of the other statuses are read in today's names
    assert lc.decide_status(WEEK, "stable", three + [_f("e3", WEEK, RAISED)])[0] == ADOPTED


def test_in_a_no_signal_regime_only_errors_count():
    quiet = [_f(f"e{i}", WEEK, OK, regime="none", neg=True) for i in range(3)]
    assert lc.decide_status(WEEK, None, quiet)[0] == ADOPTED
    raised = [_f(f"r{i}", WEEK, RAISED, regime="none") for i in range(2)]
    assert lc.decide_status(WEEK, None, raised)[:2] == (RETIRED, "errors")


def test_taint_reaches_importers_and_notes_but_not_siblings():
    status = {i: ON_TRIAL for i in FUNCS} | {NOTE: ON_TRIAL, TOK: RETIRED}
    out = lc.taint(status, FUNCS, GRAPH, {NOTE: [PD]}, {TOK: "errors"})
    why = f"depends on retired {TOK}"
    assert out == {
        PD: (why, [TOK]),
        WEEK: (why, [TOK]),
        SUM: (why, [TOK]),
        NOTE: (f"uses retired {PD}", [PD]),
    }  # WORDS (the same module) and GET (no import) are untouched
    # without rules a retirement counts as one by outcomes, and the old stored name is read as retired
    assert lc.taint(status, FUNCS, GRAPH, {NOTE: [PD]}) == out
    assert lc.taint({**status, TOK: "suspect"}, FUNCS, GRAPH, {NOTE: [PD]}) == out


def test_a_curators_retirement_taints_nothing():
    status = {i: ON_TRIAL for i in FUNCS} | {NOTE: ON_TRIAL, TOK: RETIRED}
    assert lc.taint(status, FUNCS, GRAPH, {NOTE: [TOK]}, {TOK: CURATE_RULE}) == {}


def test_a_retired_reader_taints_the_functions_that_load_its_fixtures():
    """A reader (GET) retired by its outcomes taints the function whose tests load what it read (WORDS, in a
    module that does not import it), and through the import graph what depends on that function."""
    status = {i: ON_TRIAL for i in FUNCS} | {NOTE: ON_TRIAL, GET: RETIRED}
    out = lc.taint(status, FUNCS, GRAPH, {NOTE: [WEEK]}, {GET: "errors"}, {WORDS: [GET]})
    read = f"its tests load fixtures read by retired {GET}"
    dep = f"depends on retired {WORDS}"
    assert out == {
        WORDS: (read, [GET]),
        PD: (dep, [WORDS]),
        WEEK: (dep, [WORDS]),
        SUM: (dep, [WORDS]),
        NOTE: (f"uses retired {WEEK}", [WEEK]),
    }  # TOK shares WORDS's module and is not tainted
    # a reader the curator retired taints nothing
    assert lc.taint(status, FUNCS, GRAPH, {}, {GET: CURATE_RULE}, {WORDS: [GET]}) == {}


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
    store_reads.add_cover(ev,
        WEEK,
        "e3",
        0,
    )  # a recorded input from a successful episode: its tests check a good value
    assert lc.failure_only(ev, [WEEK, PD, TOK], facts) == {PD: ["e1"]}


def test_next_records_restarts_a_new_version_and_reports_changes():
    items = {WEEK: "function", PD: "function", NOTE: "note"}
    prev = {
        WEEK: {**empty_record(WEEK, "function"), "status": RETIRED, "status_rule": "errors", "changed_at": A},
        PD: {
            **empty_record(PD, "function"),
            "status": RETIRED,
            "status_rule": CURATE_RULE,
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
        RETIRED,
        "errors",
        ["e1", "e2"],
    )
    assert same[PD]["status"] == RETIRED and same[PD]["alias_of"] == WEEK
    assert same[PD]["status_rule"] == CURATE_RULE
    assert same[NOTE]["status"] == ON_TRIAL and changes == {}
    fixed = {**every, WEEK: {B}}
    new, changes = lc.next_records(
        **base,
        since=fixed,
        changed_at={WEEK: B, PD: A, NOTE: A},
    )
    assert new[WEEK]["status"] == ON_TRIAL and new[WEEK]["use"][A]["errors"] == 2
    assert changes == {WEEK: ON_TRIAL}
    flagged, _ = lc.next_records(
        **{**base, "facts": []},
        since=fixed,
        changed_at={WEEK: B, PD: A, NOTE: A},
        failure={WEEK: ["e1"]},
    )
    assert (
        flagged[WEEK]["status"] == ON_TRIAL
        and flagged[WEEK]["provenance"]["failure_only"] is True
    )
    hidden, _ = lc.next_records(
        **{**base, "facts": []},
        since=fixed,
        changed_at={WEEK: B, PD: A, NOTE: A},
        failure={WEEK: ["e1", "e2"]},
    )
    assert (hidden[WEEK]["status"], hidden[WEEK]["status_rule"]) == (
        RETIRED,
        "failure_only",
    )


def test_curate_aliases_and_retirements_retire_items():
    """CURATE's decisions reach the records: a live alias is retired (rule curate) with alias_of its target; a
    retired item still in the tree is retired with its reason; a dropped alias no longer forwards.
    """
    items = {WEEK: "function", PD: "function", NOTE: "note"}
    prev = {
        PD: {
            **empty_record(PD, "function"),
            "status": RETIRED,
            "status_rule": CURATE_RULE,
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
        RETIRED,
        WEEK,
        CURATE_RULE,
    )
    assert (recs[NOTE]["status"], recs[NOTE]["status_rule"], recs[NOTE]["status_reason"]) == (
        RETIRED,
        CURATE_RULE,
        "retired: stale",
    )
    assert recs[WEEK]["status"] == ON_TRIAL and recs[WEEK]["alias_of"] is None
    assert changes == {NOTE: RETIRED}
    dropped, _ = lc.next_records(**base, aliases={}, curations=[])
    assert dropped[PD]["alias_of"] is None
    # the retirement stays the curator's (sticky), with its rule, so the map after this keeps it too
    assert (dropped[PD]["status"], dropped[PD]["status_rule"]) == (RETIRED, CURATE_RULE)


IMPORTED = lc.ItemUse(False, 0, False, False, False, seen=True)
REFUSED = lc.ItemUse(False, 1, True, False, False, seen=True)
REFUSED_THEN_RETURNED = lc.ItemUse(True, 2, True, False, False)


def test_an_import_alone_never_makes_an_item_adopted():
    imported = [_f(f"e{i}", WEEK, IMPORTED) for i in range(5)]
    assert lc.decide_status(WEEK, None, imported) == (ON_TRIAL, None, None, [])
    # a negative signal in an episode that only imported it is not counted against it
    quiet = [_f(f"e{i}", WEEK, IMPORTED, neg=True) for i in range(3)]
    assert lc.decide_status(WEEK, None, quiet)[0] == ON_TRIAL


def test_a_refusal_is_never_a_failure():
    refused = [_f(f"e{i}", WEEK, REFUSED) for i in range(4)]
    status, rule, reason, evidence = lc.decide_status(WEEK, None, refused)
    assert (status, rule, evidence) == (ON_TRIAL, None, [])
    assert "refused inputs it was not built for in 4 episodes" in reason
    assert "not a failure" in reason
    # refusals beside calls that returned: the item can still be adopted
    mixed = [_f(f"e{i}", WEEK, REFUSED_THEN_RETURNED) for i in range(3)]
    assert lc.decide_status(WEEK, None, mixed)[:2] == (ADOPTED, None)
    # nor do they demote an adopted item
    assert lc.decide_status(WEEK, ADOPTED, refused)[0] == ADOPTED


def test_retirement_fires_on_errors_and_on_negative_signals_after_a_use():
    assert lc.decide_status(
        WEEK,
        None,
        [_f("e1", WEEK, RAISED), _f("e2", WEEK, RAISED)],
    ) == (RETIRED, "errors", "it raised in 2 episodes", ["e1", "e2"])
    used = [
        _f("e1", WEEK, OK, neg=True),
        _f("e2", WEEK, REFUSED_THEN_RETURNED, neg=True),
    ]
    assert lc.decide_status(WEEK, None, used)[:2] == (RETIRED, "negative_signals")


def test_the_effect_the_gate_derived_reaches_the_record():
    base = dict(
        items={WEEK: "function", PD: "function"},
        prev={},
        facts=[],
        since={WEEK: {A}, PD: {A}},
        changed_at={WEEK: A, PD: A},
        functions={WEEK: "memory.text.dates", PD: "memory.text.dates"},
        graph={"memory.text.dates": set()},
        note_uses={},
    )
    recs, _ = lc.next_records(
        **base,
        verification={WEEK: {"effect": "write"}, PD: {"effect": "bogus"}},
    )
    assert recs[WEEK]["effect"] == "write" and recs[PD]["effect"] is None
    assert lc.next_records(**base)[0][WEEK]["effect"] is None
