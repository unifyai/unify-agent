"""The example rules as the writer reads them are the rules the gate checks, and each example refusal ends with what
satisfies it.

The episode rule is stated once (``library_prompts.EPISODES_RULE``): a function's returning calls take fixture values
from a recorded episode, and one episode is enough; values from more episodes are recorded and shown beside the
function, never required. An earlier rule asked for two episodes, and writers read it as the same technique evidenced
twice and stored no function at all."""

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


def test_the_brief_states_the_episode_rule_as_the_gate_checks_it():
    assert lp.EXAMPLE_EPISODES == xe.MIN_EXAMPLE_EPISODES == 1
    for brief in (
        lp.write_brief_now(),
        lp.write_brief_now(staging=True, pool=True, helper=True, similar=True),
    ):
        assert lp.EPISODES_RULE in brief
        assert "Examples must use recorded values from at least" not in brief
        assert (
            "at least 2" not in brief and "recorded episodes between them" not in brief
        )
    for words in (
        "one episode is enough",
        "never required",
        "in this batch or not",
        "A call that refuses or raises counts for nothing",
    ):
        assert words in lp.EPISODES_RULE, words


def test_the_pool_text_gives_no_second_episode_reason_to_defer():
    brief = lp.write_brief_now(pool=True)
    assert "lacks only another episode" not in brief
    assert "`retired` closes an entry for good: it is wrong or not reusable" in brief


def test_a_function_with_examples_from_one_episode_says_so_in_the_index():
    from unify.memory.item_records import episodes_of
    from unify.memory.library_index import _breadth

    records = {
        "memory.a.b:one": {"verification": {"examples": {"episodes": ["e1"]}}},
        "memory.a.b:two": {"verification": {"examples": {"episodes": ["e1", "e2"]}}},
        "memory.a.b:old": {"verification": {}},
    }
    got = episodes_of(records)
    assert [
        got(i) for i in ("memory.a.b:one", "memory.a.b:two", "memory.a.b:old", "x")
    ] == [1, 2, None, None]
    assert _breadth(got, "memory.a.b:one") == ", examples from one episode"
    assert (
        _breadth(got, "memory.a.b:two")
        == _breadth(got, "memory.a.b:old")
        == _breadth(None, "x")
        == ""
    )


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
