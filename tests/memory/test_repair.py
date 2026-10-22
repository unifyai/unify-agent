
from unify.memory.gate import GateResult
from unify.memory.repair import (
    may_repair,
    render_result,
    repair_message,
)

KEY = "sk-or-v1-" + "ab" * 32  # pragma: allowlist secret


def test_may_repair_time_and_step_rule():
    # no round count: time and the step guard alone
    assert may_repair(5, 600.0, 10, [1.0]) is None
    # time: the median seconds of the completed rounds, gate checks included
    assert may_repair(0, 9.9, 5, [10.0]) == (
        "repair: no round 1: 9.9 s left, below the round time reserve of 10.0 s (a round and the final merge's "
        "gate run)"
    )
    assert may_repair(1, 7.0, 5, [10.0, 4.0]) is None
    assert may_repair(0, 1e9, 0, [1.0]) == (
        "repair: no round 1: the pass's call cap is reached"
    )


def test_render_keeps_every_line_whole_and_redacts():
    long = "G1: fixture provenance: memory/acct/tests/data/me_1.json does not re-derive " + "y" * 5000
    res = GateResult(
        False,
        {"G3": False},
        [long, "G2: secret " + KEY, "note: pre-existing: x"],
        ["G3", "G2"],
        items_refused={"env/venmo:me": ["G3"]},
    )
    text = render_result("p1", 0, res, final=False, candidate="c" * 40)
    assert text.startswith("# Gate result: pass p1, round 0, check (not merged)\n")
    assert "\n" + long + "\n" in text  # whole: no 200- or 300-character cut
    assert (
        "## Reasons (2)" in text
        and "## Notes (1)" in text
        and "\nnote: pre-existing: x\n" in text
    )
    assert "- refused checks: G3, G2" in text and "  - env/venmo:me: G3" in text
    assert "- candidate: " + "c" * 40 in text
    assert KEY not in text
    assert "## Test output" not in text  # the gate runs no test, so there is no output to keep
    final = render_result(
        "p1",
        1,
        GateResult(True, {}, [], [], items_merged=["env/venmo:me"]),
        final=True,
        pass_notes=["note: 1 unpriced calls"],
    )
    assert final.startswith("# Gate result: pass p1, round 1, final (merge)\n")
    assert "- items merged: env/venmo:me" in final and "## Pass notes (1)" in final
    assert "- items refused: none" in final


def test_repair_message_points_at_the_full_result():
    res = GateResult(
        False,
        {},
        ["G3: x"],
        ["G3"],
        items_refused={"env/venmo:me": ["G3"]},
    )
    msg = repair_message(1, "/inputs/gate/result-0.md", res)
    assert "/inputs/gate/result-0.md" in msg and "env/venmo:me (G3)" in msg
    # rev 2 names the calls as the writer makes them, from code
    assert "repair round 1." in msg and "call tools.finish(summary) again" in msg
    assert "read it with tools.read(path)" in msg


def test_the_time_reserve_covers_the_final_merge():
    """Amendment A: the final merge runs the whole gate again, so a round needs its median time plus the last
    check's seconds; a round is refused when only the median round fits."""
    assert may_repair(0, 15.0, 5, [10.0], last_check_s=4.0) is None
    assert may_repair(0, 12.0, 5, [10.0], last_check_s=4.0) == (
        "repair: no round 1: 12.0 s left, below the round time reserve of 14.0 s (a round and the final merge's "
        "gate run)"
    )
    assert (
        may_repair(0, 12.0, 5, [10.0]) is None
    )  # without a check time, the median alone
