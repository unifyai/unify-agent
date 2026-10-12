"""CURATE through the whole gate, including the import-graph scope of its checks.

The library tests run on the shared world of :mod:`tests.memory.library_example_world`, plus a second function
``account_id`` whose test loads a fixture copied from episode e3. Behaviour is compared on the calls the parent's
tests make (an alias that answers differently on one of them is refused), and a deleted function's fixtures must stay
used by a remaining function's examples.
"""

import shutil
from types import SimpleNamespace

import pytest

from unify.memory import fixtures as fx
from unify.memory.gate import Gate, GateResult
from unify.memory.library_checks import LibraryConfig
from tests.memory import library_example_world as w
from tests.memory.library_example_world import (
    library_example_world,
)  # noqa: F401 (fixture)
from tests.memory.test_gate import (
    FILES,
    MAN,
    _candidate,
    _merged,
    world,
)  # noqa: F401 (fixture)

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap required",
)
#: the gate runs a function's examples through its tests and compares the parent's calls on the candidate

ITEM_ID, T, USERS = w.ITEM_ID, w.TEST_FILE, w.MODULE
ACCOUNT = "memory.acct.ids:account_id"
NAME_ID = "memory.acct.users:account_name"
IDS = "memory/acct/ids.py"
T_IDS = "memory/acct/tests/test_ids.py"
ME_3 = "memory/acct/tests/data/me_3.json"
WHY = "account_id repeated user_id"
#: the near-duplicate a second WRITE pass left: user_id's body under another name, with its own example on e3
IDS_MOD = f'''"""Account ids."""

from memory.acct.users import MemoryInputError


def account_id(response: dict) -> str:
    """The account's id.

    Effect: read
    Example: {T_IDS}::test_account_id_of_the_third_response
    """
    if not isinstance(response, dict) or "user_id" not in response:
        raise MemoryInputError("an acct.me response with a user_id field is needed")
    return response["user_id"]
'''
IDS_TEST = """import json
from pathlib import Path

from memory.acct.ids import account_id

DATA = Path(__file__).parent / "data"


def test_account_id_of_the_third_response():
    assert account_id(json.loads((DATA / "me_3.json").read_text())) == "u-3"
"""
NAME_FN = '''

def account_name(response: dict) -> str:
    """The account's display name.

    Effect: read
    """
    if not isinstance(response, dict) or "name" not in response:
        raise MemoryInputError("an acct.me response with a name field is needed")
    return response["name"]
'''
ALIAS_IDS = '"""Account ids: account_id is kept as an alias."""\n\nfrom memory.acct.users import user_id as account_id\n'
RETURN = '    return response["user_id"]\n'
REFACTOR = w.MODULE_SOURCE.replace(
    RETURN,
    '    found = response["user_id"]\n    return found\n',
)
TWISTED = w.MODULE_SOURCE.replace(
    RETURN,
    '    return response.get("name", response["user_id"])\n',
)
#: user_id's tests also load e3's fixture, so they take over the fixture of a retired account_id
TAKEN_TEST = w.TEST_SOURCE.replace(
    '    assert user_id(load("me_2.json")) == "u-2"\n',
    '    assert user_id(load("me_2.json")) == "u-2"\n    assert user_id(load("me_3.json")) == "u-3"\n',
)


def _parent_files(blobs, ids_mod=IDS_MOD):
    files, _ = w.files_and_fixtures(blobs.get)
    me_3, _ = fx.make(w.E3, 0, ME_3, blob=blobs.get)
    return {
        **files,
        USERS: w.MODULE_SOURCE + NAME_FN,
        IDS: ids_mod,
        T_IDS: IDS_TEST,
        ME_3: me_3.decode(),
    }


def _entry(item, tests, episodes):
    """A function item as the library-mode manifest takes it: no input and no action cover."""
    return {
        "item": item,
        "kind": "function",
        "source_episodes": episodes,
        "tests": tests,
        "covers": [],
    }


def _curate_man(**over):
    man = {
        "items": [_entry(ITEM_ID, [T, T_IDS], ["e1", "e2", "e3"])],
        "deleted": [ACCOUNT],
        "deleted_tests": [],
        "unlisted": [],
        "support": [],
        "skeleton": ["memory.acct.ids"],
        "aliases": {ACCOUNT: ITEM_ID},
        "why": WHY,
        "summary": "account_id is user_id",
    }
    return {**man, **over}


def _parent(mem, blobs, **kw):
    return _merged(
        mem,
        _parent_files(blobs, **kw),
    )  # a stand-in for the earlier gated merges


def test_the_clean_up_pass_is_curate_under_library_mode():
    gate = Gate.__new__(Gate)
    run = SimpleNamespace(man=SimpleNamespace(items=[]), p_bodies={})
    gate.library_mode = LibraryConfig(role="curate")
    assert gate._cleanup(run) is True
    gate.library_mode = LibraryConfig(role="write")
    assert gate._cleanup(run) is False


def test_curations_default_empty():
    assert GateResult(True).curations == []


@needs_bwrap
def test_v2s_gate_records_no_curation(world):
    mem, ev, gate = world
    res = gate.merge(
        mem.head(),
        _candidate(mem, FILES),
        MAN,
        "p1",
        "incremental",
        "venmo",
        "0.01",
    )
    assert res.passed and res.curations == []
    assert (
        ev.db.execute("SELECT 1 FROM sqlite_master WHERE name='curations'").fetchone()
        is None
    )


@needs_bwrap
def test_an_alias_merge_lands_and_records_its_curation(library_example_world):
    mem, ev, blobs, gate = library_example_world
    parent = _parent(mem, blobs)
    res = gate(role="curate").merge(
        parent,
        _candidate(mem, {IDS: ALIAS_IDS}),
        _curate_man(),
        "c1",
        "curate",
        None,
        "0",
    )
    assert res.passed, res.reasons
    assert res.curations == [
        {"item": ACCOUNT, "action": "alias", "target": ITEM_ID, "reason": WHY},
    ]
    assert ev.aliases() == {
        ACCOUNT: {"target": ITEM_ID, "pass_id": "c1", "commit": res.merged},
    }


#: the behaviour refusal of an alias whose old name answers differently on the parent's one call of it
CHANGED = (
    f"G3: {ACCOUNT} changes behaviour without a test: 1 of 1 calls of it in the parent's tests return, refuse or "
    "raise differently on the candidate's library ("
)
CHANGED_TAIL = (
    "); any change in what a stored function does needs a test, listed under it, that fails on the parent's "
    "library and passes on the candidate's"
)


@needs_bwrap
def test_an_alias_that_answers_differently_is_refused(library_example_world):
    """Behaviour is compared on the calls the parent's tests make: account_id's test calls it on e3's fixture, and
    the alias to account_name returns another value there."""
    mem, ev, blobs, gate = library_example_world
    parent = _parent(mem, blobs)
    wrong = ALIAS_IDS.replace(
        "import user_id as account_id",
        "import account_name as account_id",
    )
    man = _curate_man(
        items=[_entry(NAME_ID, [T_IDS], ["e3"])],
        aliases={ACCOUNT: NAME_ID},
    )
    res = gate(role="curate").check(parent, _candidate(mem, {IDS: wrong}), man)
    assert not res.passed
    found = [
        r for r in res.reasons if r.startswith(CHANGED) and r.endswith(CHANGED_TAIL)
    ]
    assert found, res.reasons
    # the reason names the call by its test, never a value
    assert f"{T_IDS}::test_account_id_of_the_third_response" in found[0]
    assert "u-3" not in found[0] and "Cy" not in found[0]
    assert ev.aliases() == {}  # a check records nothing


@needs_bwrap
def test_a_retirement_needs_a_reason_and_keeps_its_fixtures_used(library_example_world):
    """Replaces test_a_retirement_needs_a_reason_and_keeps_its_covers_covered. The rule that replaces "a deleted
    item's recorded inputs stay covered": the fixtures a deleted function's examples ran on stay used by a remaining
    function's examples (a kept function's tests take each over)."""
    mem, ev, blobs, gate = library_example_world
    parent = _parent(mem, blobs)
    gone = _candidate(mem, {IDS: None, T_IDS: None})
    base = dict(
        items=[],
        aliases={},
        skeleton=[],
        deleted_tests=[T_IDS],
        tests_changed={T_IDS: "account_id is retired"},
    )
    res = gate(role="curate").check(parent, gone, _curate_man(**base))
    assert not res.passed
    assert (
        f"G5: deleted item {ACCOUNT} needs a reason: keep its name as an alias (aliases) or retire it with a "
        "one-line reason (retired)"
    ) in res.reasons
    retired = {**base, "retired": {ACCOUNT: "user_id does the same job"}}
    res = gate(role="curate").check(parent, gone, _curate_man(**retired))
    assert not res.passed
    assert (
        f"G5: {ACCOUNT} is deleted, but the fixtures its examples ran on ({ME_3}) are used by no remaining "
        "function's examples; a kept function's tests must take each over"
    ) in res.reasons, res.reasons
    taken = {
        **retired,
        "items": [_entry(ITEM_ID, [T], ["e1", "e2", "e3"])],
        "tests_changed": {
            T_IDS: "account_id is retired",
            T: "user_id's tests take over account_id's fixture",
        },
    }
    res = gate(role="curate").merge(
        parent,
        _candidate(mem, {IDS: None, T_IDS: None, T: TAKEN_TEST}),
        _curate_man(**taken),
        "c2",
        "curate",
        None,
        "0",
    )
    assert res.passed, res.reasons
    assert ev.curations()[-1] == {
        "pass_id": "c2",
        "commit": res.merged,
        "item": ACCOUNT,
        "action": "retire",
        "target": None,
        "reason": "user_id does the same job",
    }


@needs_bwrap
def test_an_edit_without_a_red_test_lands_only_in_curate_and_only_when_behaviour_is_kept(
    library_example_world,
):
    mem, ev, blobs, gate = library_example_world
    parent = _parent(mem, blobs)
    man = _curate_man(
        items=[_entry(ITEM_ID, [T], ["e1", "e2"])],
        deleted=[],
        aliases={},
        skeleton=[],
    )
    kept = _candidate(mem, {USERS: REFACTOR + NAME_FN})
    res = gate(role="curate").check(parent, kept, man)
    assert res.passed, res.reasons
    res = gate(role="write").check(parent, kept, man)
    assert (
        not res.passed and f"G3: {ITEM_ID} has no new or changed test" in res.reasons
    ), res.reasons
    res = gate(role="curate").check(
        parent,
        _candidate(mem, {USERS: TWISTED + NAME_FN}),
        man,
    )
    assert not res.passed
    assert any(
        r.startswith(f"G3: {ITEM_ID} is edited in a clean-up pass without a red test")
        for r in res.reasons
    )


# --- an old name no parent test calls -------------------------------------------------------------------------

#: the earlier form of account_id: its docstring declares its input, and its covers are stored
STORED_IDS_MOD = IDS_MOD.replace(
    f"    Example: {T_IDS}::test_account_id_of_the_third_response\n",
    "    Input: observation\n",
)


def _parent_without_alias_tests(mem, blobs, ids_mod=IDS_MOD):
    files = {
        k: v for k, v in _parent_files(blobs, ids_mod).items() if k != T_IDS
    }  # the old name has no test
    return _merged(mem, files)


@needs_bwrap
def test_an_alias_that_answers_differently_is_refused_without_a_test_of_the_old_name(
    library_example_world,
):
    """A stored function no parent test calls, but with stored covers and an Input: line, is still compared on its
    stored covers (the transition path), with a note saying so."""
    mem, ev, blobs, gate = library_example_world
    parent = _parent_without_alias_tests(mem, blobs, STORED_IDS_MOD)
    ev.add_cover(ITEM_ID, "e1", 0)
    ev.add_cover(ACCOUNT, "e2", 0)
    wrong = ALIAS_IDS.replace(
        "import user_id as account_id",
        "import account_name as account_id",
    )
    man = _curate_man(
        items=[_entry(NAME_ID, [T], ["e2"])],
        aliases={ACCOUNT: NAME_ID},
    )
    res = gate(role="curate").check(parent, _candidate(mem, {IDS: wrong}), man)
    assert not res.passed
    assert any(
        r.startswith(f"G3: {ACCOUNT} changes behaviour without a test")
        for r in res.reasons
    ), res.reasons
    note = f"G3 behaviour check: {ACCOUNT} has no example call in the parent's tests; compared on its stored covers"
    assert any(r.endswith(note) for r in res.reasons), res.reasons


@needs_bwrap
def test_an_alias_whose_old_name_no_parent_test_calls_and_no_cover_records_is_refused(
    library_example_world,
):
    """Replaces test_an_alias_with_fewer_than_two_recorded_inputs_is_refused. The rule that replaces the alias's
    count of recorded inputs: an alias is compared on the calls the parent's tests make of the old name; with no such
    call and no stored cover its behaviour cannot be compared, so it is refused (fail closed).
    """
    mem, ev, blobs, gate = library_example_world
    parent = _parent_without_alias_tests(mem, blobs)
    res = gate(role="curate").check(
        parent,
        _candidate(mem, {IDS: ALIAS_IDS}),
        _curate_man(),
    )
    assert not res.passed
    assert (
        f"G3: {ACCOUNT} is in a changed module but no test of the parent calls it and it has no stored cover; its "
        "behaviour cannot be compared, so change it only with a test listed under it that fails on the parent's "
        "library"
    ) in res.reasons, res.reasons
