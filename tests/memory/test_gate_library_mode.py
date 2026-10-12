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
    assert (
        rec["mutation"]["total"] > 0
        and rec["mutation"]["killed"] == rec["mutation"]["total"]
    )
    assert rec["guard"] == {"killed": 1, "total": 1}
    assert rec["exact_assertions"] >= 2 and rec["tests"] == [w.TEST_FILE]
    # e3's acct.me response is of the fixtures' family and copied by none: stage 5 draws it
    assert rec["drawn_inputs"] >= 1 and rec["drawn_inputs_read"] == rec["drawn_inputs"]
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


def test_no_trusted_exact_assertion_is_refused(library_example_world):
    mem, ev, blobs, gate = library_example_world
    assert w.WEAK_TEST_SOURCE != w.TEST_SOURCE and " == " not in w.WEAK_TEST_SOURCE
    files, fixtures = w.files_and_fixtures(blobs.get, test=w.WEAK_TEST_SOURCE)
    res = gate().check(mem.head(), w.candidate(mem, files), w.manifest(fixtures))
    assert not res.passed and any("[library_mode:exact]" in r for r in res.reasons)


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


def test_examples_on_values_from_one_episode_are_refused(library_example_world):
    """Replaces test_cross_episode_text_input_that_crashes_is_refused (with the special-casing test below). The
    rule that replaces the cross-episode run: a function's examples run on recorded values from at least 2
    episodes. Two fixtures copied from one episode count once.
    """
    mem, ev, blobs, gate = library_example_world
    files, fixtures = w.files_and_fixtures(blobs.get, sources=(w.E1, w.E1))
    man = w.manifest(fixtures, w.item(source_episodes=["e1"]))
    res = gate().check(mem.head(), w.candidate(mem, files), man)
    assert not res.passed
    assert (
        f"G2: [examples:episodes] {w.ITEM_ID}'s examples run on recorded values from 1 episode(s) (e1); "
        "cross-episode evidence needs examples on values from at least 2 episodes"
    ) in res.reasons, res.reasons
    assert res.verification[w.ITEM_ID]["examples"]["episodes"] == ["e1"]


SPECIAL_CASED = w.MODULE_SOURCE.replace(
    '    return response["user_id"]\n',
    '    return {"u-1": "u-1", "u-2": "u-2"}[response["user_id"]]\n',
)
SPECIAL = re.compile(
    rf"G2: \[examples:special-cased\] {re.escape(w.ITEM_ID)} handles (?P<fixture>\S+) only for its recorded "
    r"values: with (?P<field>\S+) replaced by an unseen value of its type, its call at "
    r"(?P<file>[^:]+):(?P<line>\d+) raises KeyError where it returned before\Z",
)


def test_a_function_that_handles_only_its_recorded_values_is_refused(
    library_example_world,
):
    """Replaces test_cross_episode_text_input_that_crashes_is_refused (with the one-episode test above). The rule
    that replaces running a function on another episode's input: the gate replaces the fixture values its tests
    load with unseen values of the same type, and a call that returned before must not raise now (special-cased).
    """
    mem, ev, blobs, gate = library_example_world
    assert SPECIAL_CASED != w.MODULE_SOURCE
    files, fixtures = w.files_and_fixtures(blobs.get, module=SPECIAL_CASED)
    res = gate().check(mem.head(), w.candidate(mem, files), w.manifest(fixtures))
    assert not res.passed
    found = [m for m in map(SPECIAL.match, res.reasons) if m]
    assert found, res.reasons
    lines = {c.line for c in w.recorded_calls(blobs.get) if c.outcome == "returned"}
    for m in found:
        assert m["fixture"] in {*w.FIXTURE_PATHS, w.INPUTS_PATH}
        assert m["field"].endswith("user_id")
        assert m["file"] == w.TEST_FILE and int(m["line"]) in lines


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
