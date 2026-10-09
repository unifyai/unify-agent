from decimal import Decimal

import pytest

from unify.memory_v2.gate import GateResult
from unify.memory_v2.repair import (
    REPAIR_ROUNDS,
    may_repair,
    render_result,
    repair_message,
    round_reserve,
)

KEY = "sk-or-v1-" + "ab" * 32  # pragma: allowlist secret


def test_round_reserve_is_the_median_of_the_pass_s_own_rounds():
    # the first repair round: what the pass spent from its start to the first finish
    assert round_reserve([Decimal("0.30")]) == Decimal("0.30")
    assert round_reserve([Decimal("0.30"), Decimal("0.10")]) == Decimal("0.20")
    assert round_reserve(
        [Decimal("0.30"), Decimal("0.10"), Decimal("0.12")],
    ) == Decimal("0.12")
    # once offline replay has calibrated ROUND_RESERVE (spec §15) the constant is used
    assert round_reserve([Decimal("0.30")], Decimal("0.05")) == Decimal("0.05")
    with pytest.raises(ValueError):
        round_reserve([])


def test_may_repair_reserve_rule():
    one = [Decimal("0.30")]
    assert REPAIR_ROUNDS == 2
    # remaining exactly the reserve: the round starts; a cent less: it does not
    assert may_repair(0, Decimal("0.30"), 100.0, 5, one, [10.0]) is None
    assert may_repair(0, Decimal("0.29"), 100.0, 5, one, [10.0]) == (
        "repair: no round 1: 0.29 USD left, below the round reserve of 0.30 USD"
    )
    # the second repair round's reserve is the median of rounds 0 and 1
    assert may_repair(
        1,
        Decimal("0.19"),
        100.0,
        5,
        [Decimal("0.30"), Decimal("0.10")],
        [10.0, 4.0],
    ) == ("repair: no round 2: 0.19 USD left, below the round reserve of 0.20 USD")
    # time: the median seconds of the completed rounds, gate checks included
    assert may_repair(0, Decimal("1"), 9.9, 5, one, [10.0]) == (
        "repair: no round 1: 9.9 s left, below the round time reserve of 10.0 s (a round and the final merge's "
        "gate run)"
    )
    assert may_repair(1, Decimal("1"), 7.0, 5, one * 2, [10.0, 4.0]) is None
    assert may_repair(2, Decimal("9"), 1e9, 5, one * 3, [1.0] * 3) == (
        "repair: no round 3: at most 2 repair rounds per pass"
    )
    assert may_repair(0, Decimal("9"), 1e9, 0, one, [1.0]) == (
        "repair: no round 1: the pass's call cap is reached"
    )


def test_render_keeps_every_line_whole_and_redacts():
    long = "G3: env/venmo/tests/test_me.py is not green on the candidate " + "y" * 5000
    out = "E   KeyError: 'user'\n" + ("`" * 3 + "\n") * 2 + "z" * 6000 + KEY
    res = GateResult(
        False,
        {"G3": False},
        [long, "G2: secret " + KEY, "note: pre-existing: x"],
        ["G3", "G2"],
        items_refused={"env/venmo:me": ["G3"]},
        outputs={"env/venmo/tests/test_me.py (candidate)": out},
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
    # fenced with four backticks, past the output's own runs of three
    assert "z" * 6000 in text and "`" * 4 + "text\nE   KeyError" in text
    assert (
        "earlier bytes dropped" in text
    )  # the runner's own bound is stated, never silent
    assert KEY not in text
    final = render_result(
        "p1",
        1,
        GateResult(True, {}, [], [], items_merged=["env/venmo:me"]),
        final=True,
        pass_notes=["note: 1 unpriced calls"],
    )
    assert final.startswith("# Gate result: pass p1, round 1, final (merge)\n")
    assert "- items merged: env/venmo:me" in final and "## Pass notes (1)" in final
    assert "- items refused: none" in final and "## Test output (0)" in final


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
    assert "repair round 1 of 2" in msg and "call finish again" in msg


def test_the_time_reserve_covers_the_final_merge():
    """Amendment A: the final merge runs the whole gate again, so a round needs its median time plus the last
    check's seconds; a round is refused when only the median round fits."""
    one = [Decimal("0.30")]
    assert may_repair(0, Decimal("1"), 15.0, 5, one, [10.0], last_check_s=4.0) is None
    assert may_repair(0, Decimal("1"), 12.0, 5, one, [10.0], last_check_s=4.0) == (
        "repair: no round 1: 12.0 s left, below the round time reserve of 14.0 s (a round and the final merge's "
        "gate run)"
    )
    assert (
        may_repair(0, Decimal("1"), 12.0, 5, one, [10.0]) is None
    )  # without a check time, the median alone
