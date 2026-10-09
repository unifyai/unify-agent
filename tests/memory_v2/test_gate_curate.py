"""CURATE through the whole gate (P6; spec v2.1 §10.4, §9.1; D28 on the import-graph scope, D36)."""

import shutil
from types import SimpleNamespace

import pytest

from unify.memory_v2.gate import Gate, GateResult
from unify.memory_v2.gate_v21 import V21Config
from tests.memory_v2.test_gate import (
    FILES,
    MAN,
    _candidate,
    _merged,
    world,
)  # noqa: F401 (fixture)
from tests.memory_v2.test_gate_v21 import (
    ITEM_ID,
    MOD,
    T,
    _files_and_fixtures,
    world21,
)  # noqa: F401 (fixture)

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap required",
)

ACCOUNT = "memory.acct.ids:account_id"
NAME_ID = "memory.acct.users:account_name"
IDS = "memory/acct/ids.py"
T_IDS = "memory/acct/tests/test_ids.py"
USERS = "memory/acct/users.py"
WHY = "account_id repeated user_id"
#: the near-duplicate a second WRITE pass left: user_id's body under another name
IDS_MOD = '''"""Account ids."""

from memory.acct.users import MemoryInputError


def account_id(observation):
    """The account's id.

    Effect: read
    Input: observation
    """
    if not isinstance(observation, dict) or "user_id" not in observation:
        raise MemoryInputError("an acct.me response with a user_id field is needed")
    return observation["user_id"]
'''
IDS_TEST = """from memory.acct.ids import account_id


def test_account_id_of_the_second_response():
    assert account_id({"user_id": "u-2", "name": "Bo"}) == "u-2"
"""
NAME_FN = '''

def account_name(observation):
    """The account's display name.

    Effect: read
    Input: observation
    """
    if not isinstance(observation, dict) or "name" not in observation:
        raise MemoryInputError("an acct.me response with a name field is needed")
    return observation["name"]
'''
ALIAS_IDS = '"""Account ids: account_id is kept as an alias."""\n\nfrom memory.acct.users import user_id as account_id\n'
REFACTOR = MOD.replace(
    '    return observation["user_id"]\n',
    '    found = observation["user_id"]\n    return found\n',
)
TWISTED = MOD.replace(
    '    return observation["user_id"]\n',
    '    return observation.get("name", observation["user_id"])\n',
)


def _parent_files(blobs):
    files, _ = _files_and_fixtures(blobs)
    return {**files, USERS: MOD + NAME_FN, IDS: IDS_MOD, T_IDS: IDS_TEST}


def _entry(item, tests, covers, episodes):
    return {
        "item": item,
        "kind": "function",
        "source_episodes": episodes,
        "tests": tests,
        "covers": [[e, i] for e, i in covers],
        "input": "observation",
    }


def _curate_man(**over):
    man = {
        "items": [_entry(ITEM_ID, [T, T_IDS], [("e1", 0), ("e2", 0)], ["e1", "e2"])],
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


def _parent(mem, ev, blobs):
    parent = _merged(
        mem,
        _parent_files(blobs),
    )  # a stand-in for the earlier gated merges
    ev.add_cover(ITEM_ID, "e1", 0)
    ev.add_cover(ACCOUNT, "e2", 0)
    return parent


def test_the_clean_up_pass_is_curate_under_v21():
    gate = Gate.__new__(Gate)
    run = SimpleNamespace(man=SimpleNamespace(items=[]), p_bodies={})
    gate.v21 = V21Config(role="curate")
    assert gate._cleanup(run) is True
    gate.v21 = V21Config(role="write")
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
def test_an_alias_merge_lands_and_records_its_curation(world21):
    mem, ev, blobs, gate = world21
    parent = _parent(mem, ev, blobs)
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
    assert (
        ITEM_ID,
        "e2",
        0,
    ) in ev.covers()  # the merged function took over the old name's recorded input


@needs_bwrap
def test_an_alias_that_answers_differently_is_refused(world21):
    mem, ev, blobs, gate = world21
    parent = _parent(mem, ev, blobs)
    wrong = ALIAS_IDS.replace(
        "import user_id as account_id",
        "import account_name as account_id",
    )
    man = _curate_man(
        items=[_entry(NAME_ID, [T_IDS], [("e2", 0)], ["e2"])],
        aliases={ACCOUNT: NAME_ID},
    )
    res = gate(role="curate").check(parent, _candidate(mem, {IDS: wrong}), man)
    assert not res.passed
    assert any(
        r.startswith(f"G3: {ACCOUNT} changes behaviour without a test")
        for r in res.reasons
    ), res.reasons
    assert ev.aliases() == {}  # a check records nothing


@needs_bwrap
def test_a_retirement_needs_a_reason_and_keeps_its_covers_covered(world21):
    mem, ev, blobs, gate = world21
    parent = _parent(mem, ev, blobs)
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
    assert any(
        r.startswith(
            f"G5: deleted item {ACCOUNT} covered 1 recorded inputs that no remaining item covers",
        )
        for r in res.reasons
    ), res.reasons
    taken = {
        **retired,
        "items": [_entry(ITEM_ID, [T], [("e1", 0), ("e2", 0)], ["e1", "e2"])],
    }
    res = gate(role="curate").merge(
        parent,
        gone,
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
    world21,
):
    mem, ev, blobs, gate = world21
    parent = _parent(mem, ev, blobs)
    man = _curate_man(
        items=[_entry(ITEM_ID, [T], [("e1", 0)], ["e1"])],
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
