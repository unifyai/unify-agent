"""G3 judges functions, not test files: any edit to memory can land (library mode, writer role, in the jail).

A pass may change only tests, only the expected values in the tests of a function it did not change, or only
notes. A new or changed test that already passes on the parent is accepted as a *keeping* test and named in a
note; it records existing behaviour and never stands in for the red test an edited function still needs. A
fixture new in the pass is copied into the parent's red run, so a missing file never makes a fake red.

The tests run on the shared world of :mod:`tests.memory.library_example_world`, landed first through the gate.
"""

from __future__ import annotations

import hashlib
import shutil

import pytest

from tests.memory import library_example_world as w
from tests.memory.library_example_world import (  # noqa: F401 (fixture)
    library_example_world,
)
from unify.memory import fixtures as fx
from unify.memory import gate as gate_module

pytestmark = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap required",
)

# --- assumed names (adapt here) ---------------------------------------------------------------------------------
#: the hint appended to the two remaining per-function G3 refusals, replacing NOT_RED_HINT
RED_TEST_HINT_NAME = "RED_TEST_HINT"
RED_TEST_HINT_START = (
    "A function you edit needs at least one listed test that fails on the parent's version and passes on yours."
)
REMOVED_HINT_NAME = "NOT_RED_HINT"
#: the keeping note's three forms
KEEPS_EXISTING = "passes on the parent: it records existing behaviour"
REPAIRS_RED = "repairs a test the parent had red"
#: the per-function refusals that stay
EDITED_NOT_RED = "is edited, but none of its tests is red on the parent's library"
#: the per-test refusal that goes
OLD_NOT_RED = "already pass on the parent (not red)"
# -----------------------------------------------------------------------------------------------------------------

NAME_ID = "memory.acct.users:user_name"
NAMES_TEST = "memory/acct/tests/test_names.py"
NAMES_EXAMPLE = f"{NAMES_TEST}::test_reads_the_recorded_names"
NAMES_INPUTS = fx.inputs_file(NAME_ID)
MORE_USERS_TEST = "memory/acct/tests/test_users_recorded.py"
MORE_NAMES_TEST = "memory/acct/tests/test_names_recorded.py"
NEW_FIXTURE = "memory/acct/tests/data/me_3.json"
STRIP_TEST = "memory/acct/tests/test_users_spacing.py"
NOTE_PATH = "notes/acct/users.md"

NAME_FUNCTION = f'''

def user_name(response: dict) -> str:
    """The account holder's name, read from an acct.me response.

    Effect: read
    Example: {NAMES_EXAMPLE}
    """
    if not isinstance(response, dict) or "name" not in response:
        raise MemoryInputError("an acct.me response with a name field is needed")
    return response["name"]
'''
TWO_FUNCTIONS = w.MODULE_SOURCE + NAME_FUNCTION

NAMES_TEST_SOURCE = """import json
from pathlib import Path

import pytest

from memory.acct.users import MemoryInputError, user_name

DATA = Path(__file__).parent / "data"
INPUTS = [json.loads(x) for x in (DATA / "users.user_name.inputs.jsonl").read_text().splitlines() if x]


def load(name):
    return json.loads((DATA / name).read_text())


def test_reads_the_recorded_names():
    assert user_name(load("me_1.json")) == "Ada"
    assert user_name(load("me_2.json")) == "Bo"


@pytest.mark.parametrize("row", INPUTS)
def test_every_recorded_input_has_a_name(row):
    assert user_name(row["input"]) == row["input"]["name"]


def test_refuses_a_response_without_a_name():
    with pytest.raises(MemoryInputError):
        user_name({"user_id": "u-1"})
"""


def _keeping_source(function: str, fixture: str, expected: str) -> str:
    """A new test file that only pins what the parent already does: green on both trees."""
    return f"""import json
from pathlib import Path

from memory.acct.users import {function}

DATA = Path(__file__).parent / "data"


def test_the_recorded_value_is_read_again():
    assert {function}(json.loads((DATA / "{fixture}").read_text())) == "{expected}"
"""


MORE_USERS_SOURCE = _keeping_source("user_id", "me_2.json", "u-2")
MORE_NAMES_SOURCE = _keeping_source("user_name", "me_1.json", "Ada")

#: user_id, edited so that it keeps every recorded result (the parent's tests stay green on it)
STRIPPED = w.MODULE_SOURCE.replace(
    '    return response["user_id"]\n',
    '    return response["user_id"].strip()\n',
)
#: a test of the stripping, on a literal: red on the parent (which keeps the spaces), green on STRIPPED
STRIP_TEST_SOURCE = """from memory.acct.users import user_id


def test_spaces_around_the_id_are_dropped():
    assert user_id({"user_id": " u-1 "}) == "u-1"
"""
#: a test that reads a fixture new in this pass: green on the parent once the fixture is copied in
NEW_FIXTURE_TEST = "memory/acct/tests/test_users_third.py"
NEW_FIXTURE_TEST_SOURCE = _keeping_source("user_id", "me_3.json", "u-3")

SPECIAL_CASED = w.MODULE_SOURCE.replace(
    '    return response["user_id"]\n',
    '    return {"u-1": "u-1", "u-2": "u-2"}[response["user_id"]]\n',
)

NOTE_TEXT = (
    "---\ntitle: Account users\ndescription: Where an acct.me response keeps the user id.\n"
    "uses: [memory.acct.users:user_id]\n---\nThe user id is the response's user_id field.\n"
)


def _name_item(**over) -> dict:
    return {
        "item": NAME_ID,
        "kind": "function",
        "source_episodes": ["e1", "e2"],
        "tests": [NAMES_TEST],
        "covers": [],
        **over,
    }


def _name_inputs(blob) -> tuple[bytes, dict]:
    lines, entry = b"", None
    for ep in (w.E1, w.E2):
        line, entry = fx.make(ep, 0, NAMES_INPUTS, blob=blob, append=True, existing=entry)
        lines += line
    return lines, entry


def _merge(world, files, man, pass_id):
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


def _check(world, files, man):
    mem, _, _, gate = world
    return gate().check(mem.head(), w.candidate(mem, files), man)


def _land_world(world, *, two: bool = False, test: str | None = None) -> None:
    """Land the shared world (with *two*, user_id and user_name in one module) as pass ``p0``."""
    mem, _, blobs, _ = world
    files, fixtures = w.files_and_fixtures(
        blobs.get,
        module=TWO_FUNCTIONS if two else w.MODULE_SOURCE,
        test=test,
    )
    items = [w.item()]
    if two:
        data, entry = _name_inputs(blobs.get)
        files[NAMES_TEST] = NAMES_TEST_SOURCE
        files[NAMES_INPUTS] = data.decode()
        fixtures[NAMES_INPUTS] = entry
        items.append(_name_item())
    res = _merge(world, files, w.manifest(fixtures, *items), "p0")
    assert res.passed and res.merged == mem.head(), res.reasons


def _bypass(world, files: dict[str, str]) -> None:
    """Move main to a commit the gate never saw (a library stored before a newer rule)."""
    mem = world.mem
    old = mem.head()
    mem.fast_forward("main", w.candidate(mem, files), expected_old=old)


def _refusals(res) -> list[str]:
    return [r for r in res.reasons if not r.startswith("note:")]


def _g3_refusals(res) -> list[str]:
    return [r for r in res.reasons if r.startswith("G3:")]


def _keeping_notes(res, test: str) -> list[str]:
    return [r for r in res.reasons if r.startswith("note:") and test in r and "G3" in r]


def _landed(world, res) -> None:
    assert res.passed, res.reasons
    assert res.merged is not None and world.mem.head() == res.merged


# --- the hint ---------------------------------------------------------------------------------------------------


def test_the_not_red_hint_is_replaced_by_the_red_test_hint():
    hint = getattr(gate_module, RED_TEST_HINT_NAME, None)
    assert isinstance(hint, str) and hint.startswith(RED_TEST_HINT_START), hint
    assert "records existing behaviour and cannot show this change" in hint
    assert not hasattr(gate_module, REMOVED_HINT_NAME)


# --- changes that keep behaviour land ---------------------------------------------------------------------------


def test_a_test_only_change_for_an_unchanged_function_lands_with_a_keeping_note(
    library_example_world,
):
    world = library_example_world
    _land_world(world)
    man = w.manifest({}, w.item(tests=[w.TEST_FILE, MORE_USERS_TEST]))
    res = _merge(world, {MORE_USERS_TEST: MORE_USERS_SOURCE}, man, "p1")
    _landed(world, res)
    assert not _g3_refusals(res), res.reasons
    assert not any(OLD_NOT_RED in r for r in res.reasons), res.reasons
    notes = _keeping_notes(res, MORE_USERS_TEST)
    assert len(notes) == 1 and KEEPS_EXISTING in notes[0], res.reasons
    assert w.ITEM_ID in res.items_merged


def test_a_pin_only_change_for_an_unchanged_function_lands(library_example_world):
    """The parent's test checked the second fixture's call loosely; the pass pins its expected value. The
    function does not change, so the test needs only pass."""
    world = library_example_world
    loose = w.TEST_SOURCE.replace(
        'assert user_id(load("me_2.json")) == "u-2"',
        'assert isinstance(user_id(load("me_2.json")), str)',
    )
    assert loose != w.TEST_SOURCE
    _land_world(world, test=loose)
    res = _merge(world, {w.TEST_FILE: w.TEST_SOURCE}, w.manifest({}), "p1")
    _landed(world, res)
    assert not _g3_refusals(res), res.reasons
    assert not any(OLD_NOT_RED in r for r in res.reasons), res.reasons
    assert _keeping_notes(res, w.TEST_FILE), res.reasons


def test_a_weakened_pin_draws_only_the_weakening_refusal_and_lands_with_its_reason(
    library_example_world,
):
    world = library_example_world
    _land_world(world)
    weaker = w.TEST_SOURCE.replace(
        'assert user_id(load("me_2.json")) == "u-2"',
        'assert isinstance(user_id(load("me_2.json")), str)',
    )
    res = _check(world, {w.TEST_FILE: weaker}, w.manifest({}))
    assert not res.passed
    g3 = _g3_refusals(res)
    assert g3 and all("without a stated reason" in r for r in g3), res.reasons
    assert any("weakened" in r for r in g3), res.reasons
    stated = w.manifest(
        {},
        tests_changed={w.TEST_FILE: "the second fixture's id is checked by its shape only"},
    )
    res = _merge(world, {w.TEST_FILE: weaker}, stated, "p1")
    _landed(world, res)
    assert not _g3_refusals(res), res.reasons


def test_a_notes_only_change_lands(library_example_world):
    world = library_example_world
    _land_world(world)
    man = {
        "items": [{"item": NOTE_PATH, "kind": "note", "source_episodes": ["e1"]}],
        "deleted": [],
        "unlisted": [],
        "support": [],
        "fixtures": {},
    }
    res = _merge(world, {NOTE_PATH: NOTE_TEXT}, man, "p1")
    _landed(world, res)
    assert not _g3_refusals(res), res.reasons
    assert NOTE_PATH in res.items_merged


def test_tests_only_for_two_unchanged_functions_land(library_example_world):
    world = library_example_world
    _land_world(world, two=True)
    man = w.manifest(
        {},
        w.item(tests=[w.TEST_FILE, MORE_USERS_TEST]),
        _name_item(tests=[NAMES_TEST, MORE_NAMES_TEST]),
    )
    files = {MORE_USERS_TEST: MORE_USERS_SOURCE, MORE_NAMES_TEST: MORE_NAMES_SOURCE}
    res = _merge(world, files, man, "p1")
    _landed(world, res)
    assert not _g3_refusals(res), res.reasons
    assert set(res.items_merged) >= {w.ITEM_ID, NAME_ID}
    for test in (MORE_USERS_TEST, MORE_NAMES_TEST):
        notes = _keeping_notes(res, test)
        assert len(notes) == 1 and KEEPS_EXISTING in notes[0], res.reasons


def test_an_edited_function_with_a_red_test_plus_a_keeping_test_lands(
    library_example_world,
):
    """The edited function's red test is its own; the other function's new test is a keeping test and never
    counts as anyone's red test."""
    world = library_example_world
    _land_world(world, two=True)
    files = {
        w.MODULE: STRIPPED + NAME_FUNCTION,
        STRIP_TEST: STRIP_TEST_SOURCE,
        MORE_NAMES_TEST: MORE_NAMES_SOURCE,
    }
    man = w.manifest(
        {},
        w.item(tests=[w.TEST_FILE, STRIP_TEST]),
        _name_item(tests=[NAMES_TEST, MORE_NAMES_TEST]),
    )
    res = _merge(world, files, man, "p1")
    _landed(world, res)
    assert not _g3_refusals(res), res.reasons
    notes = _keeping_notes(res, MORE_NAMES_TEST)
    assert len(notes) == 1 and KEEPS_EXISTING in notes[0], res.reasons
    assert not _keeping_notes(res, STRIP_TEST), res.reasons


def test_a_repaired_red_test_lands_and_its_note_says_so(library_example_world):
    """The parent's own test file is red on the parent (stored before a newer rule); the pass fixes the test only."""
    world = library_example_world
    _land_world(world)
    broken = w.TEST_SOURCE.replace(
        'assert user_id(load("me_2.json")) == "u-2"',
        'assert user_id(load("me_2.json")) == "u-9"',
    )
    _bypass(world, {w.TEST_FILE: broken})
    res = _merge(world, {w.TEST_FILE: w.TEST_SOURCE}, w.manifest({}), "p1")
    _landed(world, res)
    assert not _g3_refusals(res), res.reasons
    notes = _keeping_notes(res, w.TEST_FILE)
    assert len(notes) == 1 and REPAIRS_RED in notes[0], res.reasons


# --- an unchanged function whose tests alone change ---------------------------------------------------------------


def test_an_unchanged_function_whose_tests_alone_change_skips_its_own_code_checks(
    library_example_world,
):
    """A function stored before a newer rule (here: one that only handles its recorded values, which the check
    with unseen values flags today) does not block a pass that only adds a test for it: the checks that judge a
    function's own code skip a listed function whose body did not change."""
    world = library_example_world
    _land_world(world)
    _bypass(world, {w.MODULE: SPECIAL_CASED})
    man = w.manifest({}, w.item(tests=[w.TEST_FILE, MORE_USERS_TEST]))
    res = _merge(world, {MORE_USERS_TEST: MORE_USERS_SOURCE}, man, "p1")
    _landed(world, res)
    assert w.ITEM_ID in res.items_merged
    assert not res.item_reasons.get(w.ITEM_ID), res.item_reasons
    assert not any(
        r.startswith("G2: [examples:special-cased]") for r in res.reasons
    ), res.reasons
    assert not _g3_refusals(res), res.reasons


# --- what G3 still refuses ----------------------------------------------------------------------------------------


def test_a_behaviour_change_without_a_red_test_is_refused_with_the_red_test_hint(
    library_example_world,
):
    world = library_example_world
    _land_world(world)
    files = {w.MODULE: STRIPPED, MORE_USERS_TEST: MORE_USERS_SOURCE}
    man = w.manifest({}, w.item(tests=[w.TEST_FILE, MORE_USERS_TEST]))
    res = _check(world, files, man)
    assert not res.passed
    edited = [r for r in _g3_refusals(res) if f"{w.ITEM_ID} {EDITED_NOT_RED}" in r]
    assert len(edited) == 1, res.reasons
    assert RED_TEST_HINT_START in edited[0], edited
    # the per-test refusal is gone: the keeping test is a note, not a reason of its own
    assert not any(OLD_NOT_RED in r for r in res.reasons), res.reasons
    assert all(EDITED_NOT_RED in r for r in _g3_refusals(res)), res.reasons
    assert w.ITEM_ID in res.items_refused


def test_an_edited_function_with_no_changed_test_is_refused_with_the_red_test_hint(
    library_example_world,
):
    world = library_example_world
    _land_world(world)
    res = _check(world, {w.MODULE: STRIPPED}, w.manifest({}))
    assert not res.passed
    (line,) = [r for r in _g3_refusals(res) if "has no new or changed test" in r]
    assert line.startswith(f"G3: {w.ITEM_ID} has no new or changed test")
    assert RED_TEST_HINT_START in line


def test_a_fixture_new_in_the_pass_no_longer_makes_a_fake_red(library_example_world):
    """An edited function whose only "red" test reads a fixture made in this pass: the fixture is copied into the
    parent's red run, so the test is green there and the edit has no red test."""
    world = library_example_world
    mem, _, blobs, _ = world
    _land_world(world)
    data, entry = fx.make(w.E3, 0, NEW_FIXTURE, blob=blobs.get)
    files = {
        w.MODULE: STRIPPED,
        NEW_FIXTURE: data.decode(),
        NEW_FIXTURE_TEST: NEW_FIXTURE_TEST_SOURCE,
    }
    man = w.manifest(
        {NEW_FIXTURE: entry},
        w.item(
            source_episodes=["e1", "e2", "e3"],
            tests=[w.TEST_FILE, NEW_FIXTURE_TEST],
        ),
    )
    res = _check(world, files, man)
    assert not res.passed, res.reasons
    assert any(f"{w.ITEM_ID} {EDITED_NOT_RED}" in r for r in _g3_refusals(res)), res.reasons


def test_the_fixture_copy_keeps_a_new_function_red_through_its_import(
    library_example_world,
):
    """A function new in the pass has no parent version: its test stays red on the parent through the failed
    import even though its new fixture is copied in, and the pass lands."""
    world = library_example_world
    _, _, blobs, _ = world
    _land_world(world)
    data, entry = fx.make(w.E3, 0, NEW_FIXTURE, blob=blobs.get)
    inputs, inputs_entry = _name_inputs(blobs.get)
    files = {
        w.MODULE: TWO_FUNCTIONS,
        NAMES_TEST: NAMES_TEST_SOURCE,
        NAMES_INPUTS: inputs.decode(),
        NEW_FIXTURE: data.decode(),
        MORE_NAMES_TEST: _keeping_source("user_name", "me_3.json", "Cy"),
    }
    man = w.manifest(
        {NEW_FIXTURE: entry, NAMES_INPUTS: inputs_entry},
        _name_item(
            source_episodes=["e1", "e2", "e3"],
            tests=[NAMES_TEST, MORE_NAMES_TEST],
        ),
    )
    res = _merge(world, files, man, "p1")
    _landed(world, res)
    assert NAME_ID in res.items_merged
    assert not _keeping_notes(res, MORE_NAMES_TEST), res.reasons


def test_the_red_run_holds_the_candidates_fixture_bytes(library_example_world):
    """The copied fixture is the candidate's, byte for byte: a test that compares the new fixture's digest is
    green on the parent's red run as on the candidate."""
    world = library_example_world
    _, _, blobs, _ = world
    _land_world(world)
    data, entry = fx.make(w.E3, 0, NEW_FIXTURE, blob=blobs.get)
    digest = hashlib.sha256(data).hexdigest()
    test = f"""from pathlib import Path

from memory.acct.users import user_id
import json

DATA = Path(__file__).parent / "data"


def test_the_third_fixture_is_the_recorded_one():
    raw = (DATA / "me_3.json").read_bytes()
    import hashlib
    assert hashlib.sha256(raw).hexdigest() == "{digest}"
    assert user_id(json.loads(raw)) == "u-3"
"""
    files = {NEW_FIXTURE: data.decode(), NEW_FIXTURE_TEST: test}
    man = w.manifest(
        {NEW_FIXTURE: entry},
        w.item(source_episodes=["e1", "e2", "e3"], tests=[w.TEST_FILE, NEW_FIXTURE_TEST]),
    )
    res = _merge(world, files, man, "p1")
    _landed(world, res)
    notes = _keeping_notes(res, NEW_FIXTURE_TEST)
    assert len(notes) == 1 and KEEPS_EXISTING in notes[0], res.reasons
