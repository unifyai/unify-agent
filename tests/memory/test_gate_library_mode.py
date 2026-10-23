"""The whole gate with the library mode's checks, in the jail (bubblewrap).

The tests run on the shared world of :mod:`tests.memory.library_example_world`: a function that takes a response,
whose examples are the calls its tests make on fixtures copied from two episodes, and whose manifest item has no
``input`` and no action cover.
"""

import json
import re
import shutil

import pytest

from tests.memory.library_example_world import library_example_world  # noqa: F401 - a fixture
from tests.memory import library_example_world as w
from unify.memory.gate_policy import NOT_RUN_GATE

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
    # rev 2.1: the gate runs no code, so the function lands unchecked; its labels say what was not run
    assert res.levels == {w.ITEM_ID: "unchecked"}
    assert rec["level"] == "unchecked" and rec["level_reasons"] == [f"not run: {NOT_RUN_GATE}"]
    assert rec["exact_assertions"] >= 2 and rec["tests"] == [w.TEST_FILE]
    assert f"not run: G3 {w.TEST_FILE}: {NOT_RUN_GATE}" in rec["label_findings"], rec["label_findings"]
    assert f"not run: G2 examples: {NOT_RUN_GATE}" in rec["label_findings"], rec["label_findings"]
    # nothing a run would have measured is in the record
    assert not {"examples", "mutation", "guard", "drawn_inputs", "drawn_inputs_read"} & set(rec), sorted(rec)
    assert "cross_episode" not in rec and "input" not in rec
    reasons = json.loads(
        ev.db.execute("SELECT reasons FROM passes WHERE pass_id='p1'").fetchone()[0],
    )
    assert any(r.startswith("library_mode-verification ") for r in reasons)
    # the argument shapes the actor's find reads are read statically from the function's recorded-inputs fixture,
    # each naming its one parameter (rev 2.1: nothing is run), and frozen for the landed commit
    row = (ev.commit_shapes(res.merged) or {}).get(w.ITEM_ID)
    assert row and row["shapes"] and all(d.get("param") == "response" for d in row["shapes"]), row


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


#: user_id with a decision rule: a value computed from the input, replaced by a constant under a condition
RULE_MODULE = w.MODULE_SOURCE.replace(
    '    return response["user_id"]\n',
    '    uid = response["user_id"]\n    if uid == "":\n        uid = "unknown"\n    return uid\n',
)


def test_a_decision_rule_grounded_on_one_episode_passes(
    library_example_world,
):
    """A function that encodes a decision rule needs examples from one recorded episode, like every function."""
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


def test_a_deleted_package_test_is_labelled_unless_a_reason_is_stated(
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
    unstated = "is deleted without a stated reason"
    res = gate().check(
        mem.head(),
        w.candidate(mem, {w.TEST_FILE: None}),
        man,
    )
    # rev 2: a weakened suite is a label on the pass, never a refusal (rev 2.1: the suite is not run, so a lost test
    # is found only by its file's deletion)
    assert res.passed and not res.refused, res.reasons
    assert any(r.startswith("note: label: the pass: G3: ") and unstated in r for r in res.reasons), res.reasons
    stated = {
        **man,
        "tests_changed": {w.TEST_FILE: "the function is covered by test_accounts.py"},
    }
    res = gate().check(
        mem.head(),
        w.candidate(mem, {w.TEST_FILE: None}),
        stated,
    )
    assert not any(unstated in r for r in res.reasons), res.reasons


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
