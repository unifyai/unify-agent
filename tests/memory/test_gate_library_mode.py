"""The whole gate with the library mode's checks, in the jail (bubblewrap).

The tests run on the shared world of :mod:`tests.memory.library_example_world`: a function that takes a response,
whose examples are the calls its tests make on fixtures copied from two episodes, and whose manifest item has no
``input`` and no action cover.
"""

import json
import re
import shutil

import pytest

from tests.memory import library_example_world as w
from tests.memory.library_example_world import (
    library_example_world,
)  # noqa: F401 (fixture)

pytestmark = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap required",
)

#: the gate takes a function item without ``input`` and runs its examples through its tests
#: the manifest takes a function item without ``input``, and the gate reads no Input: line
#: G2 no longer refuses a function that covers no recorded action (its examples are its tests' calls)

# --- the gate on the shared world ------------------------------------------------------------------------------


def _merge(world, files, man, pass_id="p1"):
    mem, _, _, gate = world
    return gate().merge(
        mem.head(),
        w.candidate(mem, files),
        man,
        pass_id,
        "write",
        None,
        "0.01",
    )


def test_library_gate_passes_and_records_verification(library_example_world):
    mem, ev, blobs, gate = library_example_world
    files, fixtures = w.files_and_fixtures(blobs.get)
    res = _merge(library_example_world, files, w.manifest(fixtures))
    assert res.passed, res.reasons
    rec = res.verification[w.ITEM_ID]
    # every evidence check holds: the function lands at the level the gate gives for that
    assert res.levels == {w.ITEM_ID: "reproduces-its-case"} and res.level_reasons == {w.ITEM_ID: []}
    assert rec["level"] == "reproduces-its-case" and rec["level_reasons"] == []
    # the library gate draws no inputs and runs no mutants
    assert "mutation" not in rec and "guard" not in rec and "drawn_inputs" not in rec
    assert rec["exact_assertions"] >= 2 and rec["tests"] == [w.TEST_FILE]
    examples = rec["examples"]
    assert (
        examples["calls"],
        examples["accepted"],
        examples["refused"],
        examples["raised"],
    ) == (5, 4, 1, 0)
    assert examples["grounded_calls"] == 4 and examples["episodes"] == ["e1", "e2"]
    assert examples["fixtures"] == sorted([*w.FIXTURE_PATHS, w.INPUTS_PATH])
    assert (
        examples["perturbation"]["judged"] > 0
        and examples["perturbation"]["unstable"] == 0
    )
    first = w.recorded_calls(blobs.get)[0]
    assert examples["example"] == {
        "test": w.EXAMPLE_TEST,
        "line": first.line,
        "call": 'user_id(load("me_1.json"))',
    }
    assert "cross_episode" not in rec and "input" not in rec
    reasons = json.loads(
        ev.db.execute("SELECT reasons FROM passes WHERE pass_id='p1'").fetchone()[0],
    )
    assert any(r.startswith("library_mode-verification ") for r in reasons)
    # the landed commit's shapes are frozen under the item's id, for the actor's find
    assert w.ITEM_ID in (ev.commit_shapes(res.merged) or {})


def test_a_fixture_not_made_by_fixture_tool_is_refused(library_example_world):
    mem, ev, blobs, gate = library_example_world
    files, fixtures = w.files_and_fixtures(blobs.get)
    fixtures.pop(w.FIXTURE_PATHS[0])
    res = gate().check(mem.head(), w.candidate(mem, files), w.manifest(fixtures))
    assert not res.passed
    assert any("me_1.json did not come through fixture()" in r for r in res.reasons)


def test_no_trusted_exact_assertion_sets_the_lowest_level(library_example_world):
    """An evidence finding no longer refuses: the function lands at ``unchecked`` with the finding as its reason."""
    mem, ev, blobs, gate = library_example_world
    assert w.WEAK_TEST_SOURCE != w.TEST_SOURCE and " == " not in w.WEAK_TEST_SOURCE
    files, fixtures = w.files_and_fixtures(blobs.get, test=w.WEAK_TEST_SOURCE)
    res = gate().check(mem.head(), w.candidate(mem, files), w.manifest(fixtures))
    assert res.passed, res.reasons
    assert not any("[library_mode:exact]" in r for r in res.reasons if not r.startswith("note:"))
    assert any(r.startswith(f"note: evidence level: {w.ITEM_ID} is unchecked: ") for r in res.reasons)
    assert res.levels[w.ITEM_ID] == "unchecked"
    assert any("[library_mode:exact]" in r for r in res.level_reasons[w.ITEM_ID])
    assert res.verification[w.ITEM_ID]["level"] == "unchecked"


def test_a_function_whose_parameters_are_all_required_passes_the_gate(
    library_example_world,
):
    """The gate calls a function only as its tests call it: a second required parameter with no default is never
    refused as an unbound argument, so no writer needs to invent a default to pass."""
    import ast

    mem, ev, blobs, gate = library_example_world
    module = (
        w.MODULE_SOURCE.replace(
            "def user_id(response: dict) -> str:",
            "def user_id(response: dict, field: str) -> str:",
        )
        .replace('"user_id" not in response', "field not in response")
        .replace('return response["user_id"]', "return response[field]")
    )
    test = re.sub(
        r'user_id\((load\("[^"]+"\))\)',
        r'user_id(\1, "user_id")',
        w.TEST_SOURCE,
    )
    test = test.replace(
        'user_id(row["input"])',
        'user_id(row["input"], "user_id")',
    ).replace(
        'user_id({"name": "Ada"})',
        'user_id({"name": "Ada"}, "user_id")',
    )
    (fn,) = [
        n
        for n in ast.walk(ast.parse(module))
        if isinstance(n, ast.FunctionDef) and n.name == "user_id"
    ]
    assert [a.arg for a in fn.args.args] == [
        "response",
        "field",
    ] and fn.args.defaults == []
    assert 'user_id(load("me_1.json"), "user_id")' in test
    files, fixtures = w.files_and_fixtures(blobs.get, module=module, test=test)
    res = _merge(library_example_world, files, w.manifest(fixtures))
    assert res.passed, res.reasons
    assert not any("unbound" in r or "default" in r for r in res.reasons)


def test_curate_due_is_a_note_not_a_refusal(library_example_world):
    mem, ev, blobs, gate = library_example_world
    files, fixtures = w.files_and_fixtures(blobs.get)
    res = gate(index_tokens=1).check(
        mem.head(),
        w.candidate(mem, files),
        w.manifest(fixtures),
    )
    assert res.passed, res.reasons
    assert res.curate_due and any("G4 CURATE due" in r for r in res.reasons)


def test_write_may_not_remove_an_item(library_example_world):
    mem, ev, blobs, gate = library_example_world
    files, fixtures = w.files_and_fixtures(blobs.get)
    assert _merge(library_example_world, files, w.manifest(fixtures)).passed
    gone = {w.MODULE: None, w.TEST_FILE: None}
    man = {
        "items": [],
        "deleted": [w.ITEM_ID],
        "deleted_tests": [w.TEST_FILE],
        "unlisted": [],
        "support": [],
        "tests_changed": {w.TEST_FILE: "the function is retired"},
    }
    res = gate().check(mem.head(), w.candidate(mem, gone), man)
    assert not res.passed and any(
        "removals and merges belong to CURATE" in r for r in res.reasons
    )


SPECIAL_CASED = w.MODULE_SOURCE.replace(
    '    return response["user_id"]\n',
    '    return {"u-1": "u-1", "u-2": "u-2"}[response["user_id"]]\n',
)


def _flagged(res, item: str) -> list[str]:
    """The level reasons and gate reasons that judge how specific *item* is (the check with unseen values)."""
    return [
        r
        for r in list(res.level_reasons.get(item, [])) + list(res.reasons)
        if "[examples:special-cased]" in r or "[examples:perturbed-refused]" in r or "[examples:unjudged]" in r
    ]


def test_a_function_grounded_on_one_episode_passes_and_a_memorising_one_lands_without_a_flag(
    library_example_world,
):
    """One recorded episode is enough evidence for a function: the same request shape gets a reusable function even
    if no other request is served by it. A function that only handles its recorded values on that same single
    episode lands too: the check with unseen values still runs for its derived links, but what it records never
    flags the function or sets its evidence level.
    """
    mem, ev, blobs, gate = library_example_world
    files, fixtures = w.files_and_fixtures(blobs.get, sources=(w.E1, w.E1))
    man = w.manifest(fixtures, w.item(source_episodes=["e1"]))
    res = gate().check(mem.head(), w.candidate(mem, files), man)
    assert res.passed, res.reasons
    assert not any("[examples:episodes]" in r for r in res.reasons)
    assert res.verification[w.ITEM_ID]["examples"]["episodes"] == ["e1"]
    assert res.levels[w.ITEM_ID] == "reproduces-its-case", res.level_reasons
    files, fixtures = w.files_and_fixtures(
        blobs.get,
        sources=(w.E1, w.E1),
        module=SPECIAL_CASED,
    )
    res = gate().check(
        mem.head(),
        w.candidate(mem, files),
        w.manifest(fixtures, w.item(source_episodes=["e1"])),
    )
    assert res.passed, res.reasons
    assert not _flagged(res, w.ITEM_ID), res.level_reasons


#: user_id with a decision rule: a value computed from the input, replaced by a constant under a condition
RULE_MODULE = w.MODULE_SOURCE.replace(
    '    return response["user_id"]\n',
    '    uid = response["user_id"]\n    if uid == "":\n        uid = "unknown"\n    return uid\n',
)
#: the same rule, memorising its recorded outputs instead of reading them
RULE_MEMORISED = RULE_MODULE.replace(
    '    uid = response["user_id"]\n',
    '    uid = {"u-1": "u-1", "u-2": "u-2"}[response["user_id"]]\n',
)


def test_a_decision_rule_grounded_on_one_episode_passes_and_one_that_memorises_lands_without_a_flag(
    library_example_world,
):
    """A function that encodes a decision rule needs examples from one recorded episode, like every function; one
    that only reproduces its recorded outputs lands too, never flagged by the check with unseen values.
    """
    from unify.memory.analysis.overrides import find_overrides
    import ast

    (fn,) = [
        n
        for n in ast.walk(ast.parse(RULE_MODULE))
        if isinstance(n, ast.FunctionDef) and n.name == "user_id"
    ]
    assert find_overrides(
        fn,
    ).line, "the module must encode a rule the gate's analysis flags"
    mem, ev, blobs, gate = library_example_world
    files, fixtures = w.files_and_fixtures(
        blobs.get,
        sources=(w.E1, w.E1),
        module=RULE_MODULE,
    )
    res = gate().check(
        mem.head(),
        w.candidate(mem, files),
        w.manifest(fixtures, w.item(source_episodes=["e1"])),
    )
    assert res.passed, res.reasons
    assert not any("encodes such a rule" in r for r in res.reasons)
    files, fixtures = w.files_and_fixtures(
        blobs.get,
        sources=(w.E1, w.E1),
        module=RULE_MEMORISED,
    )
    res = gate().check(
        mem.head(),
        w.candidate(mem, files),
        w.manifest(fixtures, w.item(source_episodes=["e1"])),
    )
    assert res.passed, res.reasons
    assert not _flagged(res, w.ITEM_ID), res.level_reasons


def test_a_function_that_handles_only_its_recorded_values_is_never_flagged_for_it(
    library_example_world,
):
    """The gate still reruns a function's tests with the fixture values they load replaced by unseen values of the
    same type (for the derived links of its examples), and records what its calls do; a call that returned before
    and raises now neither refuses the function nor sets its evidence level, and no reason or note names it.
    """
    mem, ev, blobs, gate = library_example_world
    assert SPECIAL_CASED != w.MODULE_SOURCE
    files, fixtures = w.files_and_fixtures(blobs.get, module=SPECIAL_CASED)
    res = gate().check(mem.head(), w.candidate(mem, files), w.manifest(fixtures))
    assert res.passed, res.reasons
    assert not _flagged(res, w.ITEM_ID), res.level_reasons


def test_a_deleted_package_test_is_refused_unless_a_reason_is_stated(
    library_example_world,
):
    mem, ev, blobs, gate = library_example_world
    files, fixtures = w.files_and_fixtures(blobs.get)
    assert _merge(library_example_world, files, w.manifest(fixtures)).passed
    man = {
        "items": [],
        "deleted": [],
        "deleted_tests": [w.TEST_FILE],
        "unlisted": [],
        "support": [],
    }
    lost = "no longer passes the parent's tests"
    res = gate(role="curate").check(
        mem.head(),
        w.candidate(mem, {w.TEST_FILE: None}),
        man,
    )
    assert not res.passed and any(
        f"suite memory/acct/tests {lost}" in r for r in res.reasons
    )
    stated = {
        **man,
        "tests_changed": {w.TEST_FILE: "the function is covered by test_accounts.py"},
    }
    res = gate(role="curate").check(
        mem.head(),
        w.candidate(mem, {w.TEST_FILE: None}),
        stated,
    )
    assert not any(lost in r for r in res.reasons), res.reasons


def test_a_tests_init_file_is_refused_with_the_rule_it_breaks(library_example_world):
    mem, ev, blobs, gate = library_example_world
    files, fixtures = w.files_and_fixtures(blobs.get)
    files["memory/acct/tests/__init__.py"] = ""
    res = _merge(library_example_world, files, w.manifest(fixtures))
    assert not res.passed
    assert (
        "G1: file memory/acct/tests/__init__.py: tests/ takes no __init__.py: the gate's pytest imports each test "
        "file by its path"
    ) in res.reasons, res.reasons
