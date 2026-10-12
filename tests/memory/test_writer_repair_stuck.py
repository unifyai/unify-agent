"""Stuck refusals in an uncapped library-mode pass: a refusal the gate repeats for the same item and reason in
repair.STUCK_REPEATS consecutive rounds gets no further repair round; once every refusal of a round is stuck, the
pass goes to its final merge. A round whose candidate did not change repeats every refusal. Capped passes keep their
round limit and never mark a refusal stuck.
"""

import time
from decimal import Decimal

from unify.memory import repair
from unify.memory.drafts import draft_states
from unify.memory.gate import NOT_RED_HINT, GateResult
from unify.memory.repair import (
    PASS_WIDE,
    STUCK_REPEATS,
    StuckRefusals,
    refusal_reason,
    refusals,
    repair_message,
)
from unify.memory.writer_pass import PassOutcome
from tests.memory.test_gate import MAN, TEST
from tests.memory.test_writer_library_repair import FILES_CODE
from tests.memory.test_writer_pass import Turns, _call, _run, _write, _writer, needs_bwrap

ME, OTHER = "env/venmo:me", "env/venmo:other"
TEST_PATH = "/memory/env/venmo/tests/test_me.py"
NOT_RED = (
    "G3: the tests in env/venmo/tests/test_me.py already pass on the parent (not red). " + NOT_RED_HINT
)


def _refused(lines: dict[str, str]) -> GateResult:
    """A refused check result: one attributed reason line per item."""
    return GateResult(
        False,
        {},
        list(lines.values()),
        sorted({ln.split(":", 1)[0] for ln in lines.values()}),
        items_refused={i: [ln.split(":", 1)[0]] for i, ln in lines.items()},
        item_reasons={i: [ln] for i, ln in lines.items()},
    )


def _edits(rounds: int) -> list[dict]:
    """A writer that changes a declared file in every round, then finishes."""
    out = [_call("w0", "execute_code", {"code": FILES_CODE}), _call("f0", "finish", {"summary": "s"})]
    for k in range(1, rounds):
        out += [
            _call(f"w{k}", "execute_code", {"code": _write(TEST_PATH, TEST + f"# round {k}\n")}),
            _call(f"f{k}", "finish", {"summary": f"round {k}"}),
        ]
    return out


# --- the comparable reason and the tracker -------------------------------------------------------------


def test_the_comparable_reason_ignores_output_timings_ids_and_paths():
    a = (
        "env/venmo/tests/test_me.py is not green on the candidate (rc=1 valid=True timed_out=False "
        "failed=['test_me'] passed=0) E   KeyError: 'user'\n  at line 3 [full output: blob " + "a1" * 32 + "]"
    )
    b = (
        "env/venmo/tests/test_me.py is not green on the candidate (rc=2 valid=False timed_out=True "
        "failed=[] passed=4) Timeout"
    )
    assert refusal_reason(a) == refusal_reason(b) == "env/venmo/tests/test_me.py is not green on the candidate"
    assert refusal_reason(
        "behaviour check not completed after 12.5 s in /tmp/gate-x1/run at 0123456789abcdef",
    ) == refusal_reason("behaviour check not completed after 13 s in /tmp/gate-y2/run at fedcba9876543210")
    # a different item or wording is a different reason
    assert refusal_reason("the tests in a.py already pass") != refusal_reason("the tests in b.py already pass")


def test_refusals_are_keyed_by_check_and_item():
    res = _refused({ME: NOT_RED, OTHER: "G2: env/venmo:other covers no recorded action"})
    res.reasons += ["G1: undeclared change memory/x.py", "note: pre-existing: y"]
    got = refusals(res)
    assert set(got) == {("G3", ME), ("G2", OTHER), ("G1", PASS_WIDE)}  # notes are not refusals
    assert got[("G3", ME)].startswith("the tests in env/venmo/tests/test_me.py already pass on the parent")
    # without attributed lines, the lines of each check an item is refused under stand for it
    bare = GateResult(False, {}, ["G2: covers no recorded action"], ["G2"], items_refused={ME: ["G2"]})
    assert refusals(bare) == {("G2", ME): "covers no recorded action"}


def test_the_same_refusal_becomes_stuck_after_its_repeats_and_a_change_resets_it():
    t = StuckRefusals()
    for k in range(STUCK_REPEATS - 1):
        assert t.update(_refused({ME: NOT_RED}), ("tree", k)) == []
    (held,) = t.update(_refused({ME: NOT_RED}), ("tree", "last"))
    assert held[:2] == ("G3", ME) and t.all_stuck() and t.report()["refusals"][0]["rounds"] == STUCK_REPEATS
    # another refusal of the same item and check is a new one: its count starts again
    assert t.update(_refused({ME: "G3: env/venmo/tests/test_me.py is not green on the candidate"}), 1) == []
    assert not t.all_stuck()
    # one stuck refusal beside a fresh one: not every refusal is stuck
    t2 = StuckRefusals()
    for k in range(STUCK_REPEATS):
        held = t2.update(_refused({ME: NOT_RED, OTHER: f"G2: variant {k}"}), k)
    assert [h[1] for h in held] == [ME] and not t2.all_stuck()
    assert t2.report() == {
        "repeats": STUCK_REPEATS,
        "stopped": False,
        "refusals": [{"check": "G3", "item": ME, "reason": held[0][2], "rounds": STUCK_REPEATS}],
    }
    assert StuckRefusals().report() is None


def test_an_unchanged_candidate_repeats_every_refusal():
    t = StuckRefusals()
    words = ["alpha", "beta", "gamma"]
    for k, w in enumerate(words):
        held = t.update(_refused({ME: f"G2: refusal {w}"}), ("same tree", "same manifest"))
    assert [h[:2] for h in held] == [("G2", ME)] and t.all_stuck()
    # the same texts on changing candidates are not repeats
    t = StuckRefusals()
    for k, w in enumerate(words):
        assert t.update(_refused({ME: f"G2: refusal {w}"}), ("tree", k)) == []
    # an unknown candidate is never a repeat by construction
    t = StuckRefusals()
    for w in words:
        assert t.update(_refused({ME: f"G2: refusal {w}"}), None) == []


def test_the_repair_message_names_the_stuck_items_and_why():
    res = _refused({ME: NOT_RED, OTHER: "G2: env/venmo:other covers no recorded action"})
    plain = repair_message(4, "/inputs/gate/result-3.md", res, None)
    assert plain.endswith("the items that pass land and the rest are kept as a draft for the next pass.")
    held = [("G3", ME, refusals(res)[("G3", ME)])]
    msg = repair_message(4, "/inputs/gate/result-3.md", res, None, stuck=held)
    assert msg.startswith(plain)
    assert f"No further repair round is offered for {ME} (G3)" in msg
    assert f"in {STUCK_REPEATS} rounds in a row" in msg and "already pass on the parent (not red)" in msg
    assert "kept as a draft" in msg and "this round is for the other refusals" in msg


def test_the_not_red_hint_says_what_satisfies_the_rule():
    assert "fails on the parent because this pass changes the behaviour it checks" in NOT_RED_HINT
    assert "only pin what the parent already does" in NOT_RED_HINT
    assert "manifest" in NOT_RED_HINT and "draft" in NOT_RED_HINT


def test_the_end_event_carries_the_stuck_refusals(tmp_path):
    from tests.memory.integration.test_consolidate import _stores
    from unify.memory.integration import consolidate

    stuck = {
        "repeats": STUCK_REPEATS,
        "stopped": True,
        "refusals": [{"check": "G3", "item": ME, "reason": "r", "rounds": 3}],
    }
    out = PassOutcome("p1", True, None, "0", 1, ["ok"], "", 0, ["ok"], coverage={}, repair_stuck=stuck)
    end = consolidate._end_event(_stores(tmp_path), "p1", out, [], 1.0, None)
    assert end["repair_stuck"] == stuck
    out.repair_stuck = None
    assert consolidate._end_event(_stores(tmp_path / "b"), "p1", out, [], 1.0, None)["repair_stuck"] is None


# --- the writer pass ---------------------------------------------------------------------------------


@needs_bwrap
def test_a_repeated_refusal_stops_repair_and_the_rest_of_the_manifest_lands(tmp_path):
    model = Turns(_edits(12))
    mem, ev, writer = _writer(tmp_path, model, library_mode=True, max_usd=None, deadline_s=300.0)
    checked: list[str] = []

    def check(parent, candidate, manifest, **kwargs):
        k = len(checked)
        checked.append(candidate)
        # the other item is refused differently in rounds 0..3, then passes
        lines = {ME: NOT_RED} | ({OTHER: f"G2: env/venmo:other covers no recorded action of kind {k}"} if k < 4 else {})
        return _refused(lines)

    merged: list[tuple[str, dict]] = []

    def merge(parent, candidate, manifest, pass_id, kind, channel, usd, **kwargs):
        merged.append((candidate, manifest))
        # per-item admission: the other item lands, the stuck one is refused; the pass row keeps what it refused
        res = GateResult(
            True,
            {},
            [f"item refused: {NOT_RED}"],
            [],
            merged=parent,
            items_merged=[OTHER],
            items_refused={ME: ["G3"]},
        )
        writer.gate._record(parent, candidate, pass_id, kind, channel, usd, res)
        return res

    writer.gate.check, writer.gate.merge = check, merge
    out = _run(writer)
    # rounds 0..2 make the refusal stuck; rounds 3 and 4 run for the other item; round 4 refuses only the stuck one
    assert len(checked) == 5 and out.rounds == 5, out.reasons
    assert len(merged) == 1 and merged[0][0] == checked[-1]  # the last checked candidate goes to the one merge
    assert out.passed and out.items_merged == [OTHER] and out.items_refused == {ME: ["G3"]}
    stuck_notes = [r for r in out.reasons if r.startswith("repair: stuck: ")]
    assert len(stuck_notes) == 1 and f"refused {ME} (G3) for the same reason in {STUCK_REPEATS} rounds" in stuck_notes[0]
    assert any(r.startswith("repair: no round 5: every refusal is stuck") for r in out.reasons), out.reasons
    assert not any("below the round time reserve" in r for r in out.reasons)
    assert out.repair_stuck["stopped"] is True and out.repair_stuck["repeats"] == STUCK_REPEATS
    (row,) = out.repair_stuck["refusals"]
    assert (row["check"], row["item"], row["rounds"]) == ("G3", ME, 5)
    assert row["reason"].startswith("the tests in env/venmo/tests/test_me.py already pass on the parent (not red)")
    # the writer was told, in the rounds after it became stuck, that the item gets no further round
    msgs = [m["content"] for m in model.seen if m.get("role") == "user" and "This is repair round" in m["content"]]
    assert len(msgs) == 4
    told = [m for m in msgs if f"No further repair round is offered for {ME} (G3)" in m]
    assert told == msgs[2:]
    # kept as a draft: the pass row's patch (what the merge refused) and its final result, staged for the next pass
    assert draft_states(ev.pass_rounds()) == {"px": "open"}
    (pr,) = ev.pass_rounds()
    final = writer.gate.blobs.get(pr["gate_blob"]).decode()
    assert "repair: stuck: " in final and "item refused: G3" in final


@needs_bwrap
def test_refusals_that_differ_each_round_keep_getting_rounds_until_time(tmp_path):
    model = Turns(_edits(80))
    mem, ev, writer = _writer(tmp_path, model, library_mode=True, max_usd=None, deadline_s=20.0)
    n = [0]

    def check(parent, candidate, manifest, **kwargs):
        n[0] += 1
        time.sleep(0.2)
        return _refused({ME: f"G2: env/venmo:me covers no recorded action of kind {n[0]}"})

    writer.gate.check = check
    out = _run(writer)
    # a round the time reserve refuses before its check is judged by the final merge alone
    assert out.rounds > STUCK_REPEATS + 1 and n[0] in (out.rounds, out.rounds - 1), out.reasons
    assert out.repair_stuck is None
    assert not any(r.startswith("repair: stuck") or "every refusal is stuck" in r for r in out.reasons)
    assert any("below the round time reserve" in r or "pass deadline" in r for r in out.reasons), out.reasons


@needs_bwrap
def test_an_unchanged_candidate_counts_as_a_repeat_in_the_pass(tmp_path):
    # after round 0 the writer only calls finish: every later candidate is the same tree and manifest
    model = Turns([_call("w0", "execute_code", {"code": FILES_CODE}), _call("f0", "finish", {"summary": "s"})])
    mem, ev, writer = _writer(tmp_path, model, library_mode=True, max_usd=None, deadline_s=300.0)
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    seen: list[str] = []

    def check(parent, candidate, manifest, **kwargs):
        seen.append(candidate)
        # wording that differs every round, which no normalisation would equate
        return _refused({ME: f"G2: env/venmo:me refusal {words[(len(seen) - 1) % len(words)]}"})

    writer.gate.check = check
    out = _run(writer)
    assert len(seen) == STUCK_REPEATS and out.rounds == STUCK_REPEATS, out.reasons
    assert len({mem.run("rev-parse", f"{c}^{{tree}}").strip() for c in seen}) == 1
    assert out.repair_stuck["stopped"] is True
    assert [(r["check"], r["item"], r["reason"]) for r in out.repair_stuck["refusals"]] == [
        ("G2", ME, "env/venmo:me refusal gamma"),
    ]
    assert any(r.startswith(f"repair: no round {STUCK_REPEATS}: every refusal is stuck") for r in out.reasons)
    # the real merge refused the pass: its whole patch is the draft
    assert not out.passed and draft_states(ev.pass_rounds()) == {"px": "open"}


@needs_bwrap
def test_a_capped_pass_keeps_its_round_limit_and_marks_nothing_stuck(tmp_path):
    model = Turns(_edits(6))
    mem, ev, writer = _writer(tmp_path, model, library_mode=True, max_usd=Decimal("10"), max_calls=40)
    n = [0]

    def check(parent, candidate, manifest, **kwargs):
        n[0] += 1
        return _refused({ME: NOT_RED})

    writer.gate.check = check
    out = _run(writer)
    # rounds 0 and 1 are checked; the last round goes to the final merge unchecked, as before
    assert n[0] == repair.REPAIR_ROUNDS and out.rounds == repair.REPAIR_ROUNDS + 1, out.reasons
    assert f"repair: no round {repair.REPAIR_ROUNDS + 1}: at most {repair.REPAIR_ROUNDS} repair rounds per pass" in (
        out.reasons
    )
    assert out.repair_stuck is None and not any("stuck" in r for r in out.reasons)
    msgs = [m["content"] for m in model.seen if m.get("role") == "user" and "This is repair round" in m["content"]]
    assert len(msgs) == repair.REPAIR_ROUNDS and not any("No further repair round" in m for m in msgs)


def test_the_uncapped_brief_states_the_stuck_rule_as_information():
    from unify.memory import library_prompts

    text = library_prompts.write_brief_now()
    assert f"in {STUCK_REPEATS} rounds in a row gets no further round of its own" in text
    assert "There is no set number of repair rounds" in text
    assert library_prompts.limits_now(finish_capped=True).stuck_repeats is None
    assert library_prompts.example_checks(text) == [] and library_prompts.benchmark_words(text) == []
