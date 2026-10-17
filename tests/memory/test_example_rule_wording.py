"""The example rules as the writer reads them are the rules the gate checks, and each example refusal ends with what
satisfies it.

The episode rule is stated once (``library_prompts.EPISODES_RULE``): a function's returning calls take fixture values
from a recorded episode, and one episode is enough; values from more episodes are recorded and shown beside the
function, never required. An earlier rule asked for two episodes, and writers read it as the same technique evidenced
twice and stored no function at all."""

import json
from pathlib import Path

from unify.memory import example_evidence as xe
from unify.memory import library_prompts as lp
from tests.memory.test_example_evidence import ITEM as EV_ITEM, arg, call, check
from tests.memory.test_gate_examples import _arg, _call, _checks, _reasons

GOLDEN = json.loads(
    (Path(__file__).parent / "golden" / "example_refusals_recorded.json").read_text(),
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


def test_the_brief_for_tools_called_from_code_states_the_same_episode_rule(monkeypatch):
    from unify.common import in_code_switch

    monkeypatch.setattr(in_code_switch, "tools_in_code", lambda: True)
    for brief in (
        lp.write_brief_now(),
        lp.write_brief_now(staging=True, pool=True, helper=True, similar=True),
    ):
        assert lp.EPISODES_RULE_IN_CODE in brief and lp.EPISODES_RULE not in brief
        assert "at least 2" not in brief and "lacks only another episode" not in brief
    for words in (
        "one episode is enough",
        "never required",
        "in this batch or not",
        "A call that refuses or raises counts for nothing",
    ):
        assert words in lp.EPISODES_RULE_IN_CODE, words


def test_the_pool_text_gives_no_second_episode_reason_to_defer():
    brief = lp.write_brief_now(pool=True)
    assert "lacks only another episode" not in brief
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
    # it says concretely what satisfies it: a call on values loaded from a recorded episode's fixture, in the
    # loading idiom of the library's tests, on one line like every reason
    assert "make one call that returns take each structured argument" in lp.GROUNDED_FIX
    assert "recorded episode" in lp.GROUNDED_FIX and "`tests/data/`" in lp.GROUNDED_FIX
    assert 'json.loads((Path(__file__).parent / "data" / "rows.json").read_text())' in lp.GROUNDED_FIX
    assert "assert function(rows, 2) == <trusted value>" in lp.GROUNDED_FIX
    assert "\n" not in lp.GROUNDED_FIX
    assert lp.example_checks(lp.GROUNDED_FIX) == [] and lp.benchmark_words(lp.GROUNDED_FIX) == []


def test_each_recorded_example_refusal_now_names_its_fix(tmp_path):
    """The example refusals of 12 recorded writer passes, rendered by today's code: each grounded or episodes refusal
    names what satisfies it. The recorded unjudged ones are no longer refusals: the check with unseen fixture values
    records why it did not complete and refuses nothing."""
    rows = GOLDEN["rows"]
    assert len(rows) == 72
    by_code = {}
    for _, _, code, _ in rows:
        by_code[code] = by_code.get(code, 0) + 1
    assert by_code == {"grounded": 54, "unjudged": 15, "episodes": 3}
    assert not hasattr(lp, "UNJUDGED_FIX")
    test_the_grounded_refusal_ends_with_what_to_load_from_a_fixture(tmp_path)
    test_the_episodes_refusal_ends_with_the_call_that_satisfies_it()
