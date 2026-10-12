"""The example rules as the writer reads them are the rules the gate checks, and each example refusal ends with what
satisfies it.

The cross-episode rule is stated once (``library_prompts.EPISODES_RULE``): the returning calls of a function take
fixture values from at least two recorded episodes between them, and the call on the second episode's value needs
no exact expected value. A brief that said "examples must use recorded values from at least 2 episodes" was read by
writers as the same technique evidenced twice, and they stored no function at all."""

import ast
import json
import re
from pathlib import Path

from unify.memory import example_evidence as xe
from unify.memory import example_perturb
from unify.memory import library_prompts as lp
from unify.memory.example_calls import MAX_CALL_LINES
from tests.memory.test_example_evidence import ITEM as EV_ITEM, arg, call, check
from tests.memory.test_gate_examples import _arg, _call, _checks, _reasons

GOLDEN = json.loads(
    (Path(__file__).parent / "golden" / "example_refusals_recorded.json").read_text(),
)


def _fix(cause: str) -> str:
    return lp.UNJUDGED_FIX[cause].format(
        budget_s=example_perturb.PERTURB_BUDGET_S,
        max_calls=MAX_CALL_LINES,
    )


def test_the_brief_states_the_cross_episode_rule_as_the_gate_checks_it():
    assert lp.EXAMPLE_EPISODES == xe.MIN_EXAMPLE_EPISODES
    for brief in (
        lp.write_brief_now(),
        lp.write_brief_now(staging=True, pool=True, helper=True, similar=True),
    ):
        assert lp.EPISODES_RULE in brief
        assert "Examples must use recorded values from at least" not in brief
        assert "fixture values from at least 2 episodes" not in brief
    # the rule's own words: a returning call on another episode's value is enough, with no exact expected value;
    # refused and raised calls count for nothing; the episode may come from outside the batch
    for words in (
        "that call needs no exact expected value",
        "A call that refuses or raises counts for nothing",
        "in this batch or not",
        f"from at least {xe.MIN_EXAMPLE_EPISODES} recorded episodes between them",
    ):
        assert words in lp.EPISODES_RULE, words


def test_an_entry_short_only_of_another_episode_can_wait_in_the_pool():
    brief = lp.write_brief_now(pool=True)
    assert (
        "an entry that lacks only another episode's evidence can wait there for later episodes to supply it"
        in brief
    )
    assert "`retired` closes an entry for good: it is wrong or not reusable" in brief


def test_the_episodes_refusal_ends_with_the_call_that_satisfies_it():
    (reason,), _ = check(
        [call(0, arg("colour", 3))],
    )  # no call on fixture values: 0 episodes
    assert reason.startswith(
        f"[examples:episodes] {EV_ITEM}'s calls that return use recorded values from ",
    )
    assert reason.endswith(lp.EPISODES_FIX)


def test_the_grounded_refusal_ends_with_what_to_load_from_a_fixture(tmp_path):
    ex, run, _, _ = _checks(
        tmp_path,
        [_call(_arg("grid", [[0, 1]]), _arg("colour", 3), line=14)],
    )
    ex.judge()
    (reason,) = [r for r in _reasons(run) if r.startswith("[examples:grounded]")]
    assert reason.endswith(lp.GROUNDED_FIX)


def test_every_unjudged_cause_has_its_fix_and_every_fix_is_used():
    src = Path(example_perturb.__file__).read_text()
    used = set()
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "fail"
            and len(node.args) == 2
        ):
            used |= {
                n.value
                for n in ast.walk(node.args[1])
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
            }
    assert used == set(lp.UNJUDGED_FIX)
    for cause in lp.UNJUDGED_FIX:
        text = example_perturb._unjudged("memory.a.b:f", "why", cause)
        assert text.endswith(": " + _fix(cause)) and "{" not in text


#: the recorded cause of an unjudged refusal -> the fix its refusal now names (the causes example_perturb passes)
RECORDED_CAUSE = {
    "the budget ran out before the control run": "budget",
    "the budget ran out before the run with every field changed": "budget",
    "the control run failed": "failed",
    "the run with every field changed did not finish": "failed",
    "the call record of the control run is incomplete": "record",
    "the call record of the run with every field changed is incomplete": "record",
    "no call of it could be matched across runs (<n> unmatched)": "unmatched",
}


def test_each_recorded_example_refusal_now_names_its_fix(tmp_path):
    """The example refusals of 12 recorded writer passes, rendered by today's code: each names what satisfies it."""
    rows = GOLDEN["rows"]
    assert len(rows) == 72
    by_code = {}
    for _, _, code, why in rows:
        by_code[code] = by_code.get(code, 0) + 1
        if code == "unjudged":
            cause = RECORDED_CAUSE[
                why
            ]  # every recorded cause is one the code names a fix for
            assert _fix(cause) in example_perturb._unjudged("memory.a.b:f", why, cause)
    assert by_code == {"grounded": 54, "unjudged": 15, "episodes": 3}
    test_the_grounded_refusal_ends_with_what_to_load_from_a_fixture(tmp_path)
    test_the_episodes_refusal_ends_with_the_call_that_satisfies_it()
    # the causes as the code words them: a renamed cause would leave the recorded ones unmapped
    src = Path(example_perturb.__file__).read_text()
    for why in RECORDED_CAUSE:
        assert re.sub(r" \(<n> unmatched\)$", "", why) in src, why
