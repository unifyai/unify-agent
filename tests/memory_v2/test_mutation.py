"""The gate's mutation operators (memory v2.1 stage 5): sites, application, seeded choice.

Pure: every mutant is compiled and executed here in the test process (no model code, no sandbox).
"""

import ast
import hashlib

import pytest

from unify.memory_v2.mutation import OPERATORS, Site, apply, choose, reprint, sites

# Abridged from the offline sweep's merged ARC feedback reader (docs/design/memory-v2-end-to-end-example.md).
MODULE = '''"""ARC submission feedback."""

__all__ = ["feedback_state"]


class MemoryInputError(ValueError):
    pass


def helper(x):
    return x + 1


def feedback_state(observation):
    """Read a submission feedback message: (attempts used, correct, failed).

    Effect: read
    Input: observation
    """
    if not isinstance(observation, dict) or observation.get("type") != "SubmitFeedback":
        raise MemoryInputError("expected a SubmitFeedback object")
    attempts = observation.get("attempts_used")
    if not isinstance(attempts, int) or attempts < 0:
        raise MemoryInputError("attempts_used must be a non-negative integer")
    return attempts, observation.get("correct") is True, observation.get("failed") is True
'''


def _obs(n, correct=False, failed=False, kind="SubmitFeedback"):
    return {
        "type": kind,
        "valid": True,
        "correct": correct,
        "failed": failed,
        "attempts_used": n,
    }


def _load(source):
    ns: dict = {}
    exec(compile(source, "<mutant>", "exec"), ns)  # noqa: S102 - our own fixture source
    return ns


def _outcome(ns, value):
    try:
        return ("ok", ns["feedback_state"](value))
    except ns["MemoryInputError"]:
        return ("refused", None)
    except Exception as exc:  # noqa: BLE001
        return ("error", type(exc).__name__)


def test_sites_cover_every_operator_kind_in_walk_order():
    found = sites(MODULE, "feedback_state")
    kinds = [s.op for s in found]
    assert sorted(set(kinds)) == sorted(OPERATORS)
    assert kinds.count("negate") == 2  # the two guards
    assert kinds.count("boolop") == 2
    assert kinds.count("drop_raise") == 2
    assert kinds.count("cmp") == 4  # != type, < 0, is True, is True
    assert kinds.count("const") == 1  # the 0 (True is a bool, never a number here)
    assert kinds.count("return_none") == 1
    assert len(found) == 12
    assert all(
        s.line >= 15 for s in found
    )  # inside the function, never the helper above
    assert found == sites(MODULE, "feedback_state")  # deterministic


def test_sites_of_a_missing_or_unparsable_function_are_empty():
    assert sites(MODULE, "absent") == []
    assert sites("def f(:\n", "f") == []


@pytest.mark.parametrize("op", OPERATORS)
def test_each_operator_changes_behaviour_as_named(op):
    original = _load(MODULE)
    for site in [s for s in sites(MODULE, "feedback_state") if s.op == op]:
        text = apply(MODULE, "feedback_state", site)
        assert text is not None
        ast.parse(text)  # always valid Python
        mutant = _load(text)
        inputs = [
            _obs(0),
            _obs(1),
            _obs(2, correct=True),
            _obs(3, failed=True),
            _obs(-1),
            {"type": "DemoReply", "attempts_used": 2},
            "SubmitFeedback",
            {**_obs(1), "attempts_used": "1"},
        ]
        before = [_outcome(original, v) for v in inputs]
        after = [_outcome(mutant, v) for v in inputs]
        assert before != after, (
            site,
            text,
        )  # every planted mutant here is observable on some input
        assert mutant["helper"](1) == 2  # other functions are untouched


def test_operator_semantics():
    by = {}
    for s in sites(MODULE, "feedback_state"):
        by.setdefault(s.op, []).append(s)
    ret = _load(apply(MODULE, "feedback_state", by["return_none"][0]))
    assert ret["feedback_state"](_obs(1)) is None
    const = _load(apply(MODULE, "feedback_state", by["const"][0]))
    with pytest.raises(const["MemoryInputError"]):
        const["feedback_state"](_obs(0))  # attempts < 1 now
    dropped = [_load(apply(MODULE, "feedback_state", s)) for s in by["drop_raise"]]
    assert ("ok", (-1, False, False)) in [_outcome(m, _obs(-1)) for m in dropped]
    swapped = [apply(MODULE, "feedback_state", s) for s in by["boolop"]]
    assert all(" and " in t.split("def feedback_state", 1)[1] for t in swapped)


def test_apply_refuses_a_site_that_does_not_fit():
    assert apply(MODULE, "feedback_state", Site("cmp", 1, 10_000)) is None
    assert apply(MODULE, "feedback_state", Site("unknown", 1, 0)) is None
    negate_on_wrong_node = Site("negate", 1, 0)  # node 0 is the docstring expression
    assert apply(MODULE, "feedback_state", negate_on_wrong_node) is None
    assert apply(MODULE, "absent", Site("cmp", 1, 0)) is None


def test_reprint_is_the_unmutated_control():
    text = reprint(MODULE)
    assert text is not None and text != MODULE  # formatting changes
    a, b = _load(MODULE), _load(text)
    for v in (_obs(1), _obs(3, failed=True), {"type": "X"}):
        assert _outcome(a, v) == _outcome(b, v)
    assert reprint("def f(:\n") is None


def test_choose_is_bounded_seeded_and_spread_over_kinds():
    all_sites = sites(MODULE, "feedback_state")
    seed = hashlib.sha256(b"candidate-1").digest()
    a = choose(all_sites, seed, "env/arc:feedback_state", 6)
    assert len(a) == 6 and len(set(a)) == 6
    assert {s.op for s in a} == set(
        OPERATORS,
    )  # one of each kind before a second of any
    assert a == choose(all_sites, seed, "env/arc:feedback_state", 6)  # reproducible
    other = [
        choose(
            all_sites,
            hashlib.sha256(f"c{i}".encode()).digest(),
            "env/arc:feedback_state",
            6,
        )
        for i in range(20)
    ]
    assert any(o != a for o in other)  # the seed changes the choice
    assert choose(all_sites, seed, "env/arc:feedback_state", 100) == choose(
        all_sites,
        seed,
        "env/arc:feedback_state",
        100,
    )
    assert len(choose(all_sites, seed, "x", 100)) == len(all_sites)
    assert choose([], seed, "x", 8) == []
