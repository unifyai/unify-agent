"""The whole gate with the library mode's checks, in the jail (bubblewrap).

The tests run on the shared world of :mod:`tests.memory.library_example_world`: a function that takes a response,
whose examples are the calls its tests make on fixtures copied from two episodes, and whose manifest item has no
``input`` and no action cover.
"""

import json
import re
import shutil

import pytest

from unify.memory import fixtures as fx
from unify.memory.blobs import BlobStore
from unify.memory.episodes import Action
from unify.memory.evidence import EvidenceStore
from unify.memory.gate import Gate
from unify.memory.library_checks import LibraryConfig
from unify.memory.gitio import Repo
from tests.memory import library_example_world as w
from tests.memory.library_example_world import library_example_world  # noqa: F401 (fixture)
from tests.memory.test_episodes import _ep

pytestmark = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap required",
)

#: the gate takes a function item without ``input`` and runs its examples through its tests
JUDGED = "needs example_checks.ExampleChecks.judge"
#: the manifest takes a function item without ``input``, and the gate reads no Input: line
PARSED = "needs manifest._parse_library_mode"

# --- the former world -----------------------------------------------------------------------------------------
# Kept only for the test files that still import it (test_test_output, test_memory_writer, test_pool_start,
# test_writer_library_curate, test_writer_finish_and_batch_map). Its item declares ``input`` and an action cover,
# which the library-mode manifest refuses; those files move to library_example_world as they are migrated.

ITEM_ID = "memory.acct.users:user_id"
E1 = _ep(
    episode_id="e1",
    actions=[
        Action(
            0,
            "acct",
            "me",
            [],
            {},
            {"user_id": "u-1", "name": "Ada"},
            "ok",
            "read",
        ),
    ],
)
E2 = _ep(
    episode_id="e2",
    actions=[
        Action(0, "acct", "me", [], {}, {"user_id": "u-2", "name": "Bo"}, "ok", "read"),
    ],
)
E3 = _ep(
    episode_id="e3",
    actions=[
        Action(
            -1,
            "dialogue:user",
            "say",
            [],
            {},
            "hello world",
            "ok",
            "unknown",
            kind="dialogue",
        ),
    ],
)
E4 = _ep(
    episode_id="e4",
    actions=[
        Action(
            -1,
            "dialogue:user",
            "say",
            [],
            {},
            "",
            "ok",
            "unknown",
            kind="dialogue",
        ),
    ],
)
EPS = {e.episode_id: e for e in (E1, E2, E3, E4)}


def _lookup(eid, i):
    ep = EPS.get(eid)
    return ep.actions[i] if ep is not None and 0 <= i < len(ep.actions) else None


MOD = '''"""Accounts."""


class MemoryInputError(ValueError):
    pass


def user_id(observation):
    """The account's user id.

    Effect: read
    Input: observation
    """
    if not isinstance(observation, dict) or "user_id" not in observation:
        raise MemoryInputError("an acct.me response with a user_id field is needed")
    return observation["user_id"]
'''
TEST = """import json
from pathlib import Path

import pytest

from memory.acct.users import MemoryInputError, user_id

DATA = Path(__file__).parent / "data"
INPUTS = [json.loads(x) for x in (DATA / "users.user_id.inputs.jsonl").read_text().splitlines() if x]


def test_reads_the_recorded_user_id():
    observation = json.loads((DATA / "me.json").read_text())
    assert user_id(observation) == "u-1"


@pytest.mark.parametrize("row", INPUTS)
def test_every_recorded_input_has_a_user_id(row):
    assert user_id(row["input"]) == row["input"]["user_id"]


def test_refuses_a_response_without_a_user_id():
    with pytest.raises(MemoryInputError):
        user_id({"name": "Ada"})
"""
WEAK_TEST = """import json
from pathlib import Path

import pytest

from memory.acct.users import MemoryInputError, user_id

DATA = Path(__file__).parent / "data"


def test_returns_a_string():
    assert isinstance(user_id(json.loads((DATA / "me.json").read_text())), str)


def test_refuses_a_response_without_a_user_id():
    with pytest.raises(MemoryInputError):
        user_id({"name": "Ada"})
"""
T = "memory/acct/tests/test_users.py"
INPUTS = fx.inputs_file(ITEM_ID)


@pytest.fixture
def library_gate_world(tmp_path):
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    for n, ep in enumerate(EPS.values()):
        ev.index_episode(ep, str(n + 1) * 40)
    blobs = BlobStore(tmp_path / "b")

    def gate(**cfg):
        return Gate(
            mem,
            ev,
            blobs,
            action_lookup=_lookup,
            library_mode=LibraryConfig(checks=True, episodes=EPS.get, **cfg),
        )

    return mem, ev, blobs, gate


def _files_and_fixtures(blobs, test=TEST):
    me, e_me = fx.make(E1, 0, "memory/acct/tests/data/me.json", blob=blobs.get)
    line, e_in = fx.make(E1, 0, INPUTS, blob=blobs.get, append=True)
    files = {
        "memory/acct/__init__.py": '"""Accounts."""\n',
        "memory/acct/users.py": MOD,
        T: test,
        "memory/acct/tests/data/me.json": me.decode(),
        INPUTS: line.decode(),
    }
    return files, {"memory/acct/tests/data/me.json": e_me, INPUTS: e_in}


def _man(fixtures, **item):
    it = {
        "item": ITEM_ID,
        "kind": "function",
        "source_episodes": ["e1"],
        "tests": [T],
        "covers": [["e1", 0]],
        "input": "observation",
        **item,
    }
    return {
        "items": [it],
        "deleted": [],
        "unlisted": [],
        "support": [],
        "fixtures": fixtures,
    }


# --- the gate on the shared world ------------------------------------------------------------------------------


def _merge(world, files, man, pass_id="p1"):
    mem, _, _, gate = world
    return gate().merge(mem.head(), w.candidate(mem, files), man, pass_id, "write", None, "0.01")


@pytest.mark.xfail(reason=JUDGED, strict=False)
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
    assert (examples["calls"], examples["accepted"], examples["refused"], examples["raised"]) == (5, 4, 1, 0)
    assert examples["grounded_calls"] == 4 and examples["episodes"] == ["e1", "e2"]
    assert examples["fixtures"] == sorted([*w.FIXTURE_PATHS, w.INPUTS_PATH])
    assert examples["perturbation"]["judged"] > 0 and examples["perturbation"]["unstable"] == 0
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


@pytest.mark.xfail(reason=PARSED, strict=False)
def test_a_fixture_not_made_by_fixture_tool_is_refused(library_example_world):
    mem, ev, blobs, gate = library_example_world
    files, fixtures = w.files_and_fixtures(blobs.get)
    fixtures.pop(w.FIXTURE_PATHS[0])
    res = gate().check(mem.head(), w.candidate(mem, files), w.manifest(fixtures))
    assert not res.passed
    assert any("me_1.json did not come through fixture()" in r for r in res.reasons)


WEAK_EXAMPLE_TEST = w.TEST_SOURCE.replace(
    'assert user_id(load("me_1.json")) == "u-1"',
    'assert isinstance(user_id(load("me_1.json")), str)',
).replace(
    'assert user_id(load("me_2.json")) == "u-2"',
    'assert isinstance(user_id(load("me_2.json")), str)',
)


@pytest.mark.xfail(reason=PARSED, strict=False)
def test_no_trusted_exact_assertion_is_refused(library_example_world):
    mem, ev, blobs, gate = library_example_world
    assert WEAK_EXAMPLE_TEST != w.TEST_SOURCE
    files, fixtures = w.files_and_fixtures(blobs.get, test=WEAK_EXAMPLE_TEST)
    res = gate().check(mem.head(), w.candidate(mem, files), w.manifest(fixtures))
    assert not res.passed and any("[library_mode:exact]" in r for r in res.reasons)


@pytest.mark.xfail(reason=JUDGED, strict=False)
def test_curate_due_is_a_note_not_a_refusal(library_example_world):
    mem, ev, blobs, gate = library_example_world
    files, fixtures = w.files_and_fixtures(blobs.get)
    res = gate(index_tokens=1).check(mem.head(), w.candidate(mem, files), w.manifest(fixtures))
    assert res.passed, res.reasons
    assert res.curate_due and any("G4 CURATE due" in r for r in res.reasons)


@pytest.mark.xfail(reason=JUDGED, strict=False)
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


@pytest.mark.xfail(reason="needs example_evidence.evidence_check", strict=False)
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


@pytest.mark.xfail(reason="needs example_perturb.check", strict=False)
def test_a_function_that_handles_only_its_recorded_values_is_refused(library_example_world):
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


@pytest.mark.xfail(reason=JUDGED, strict=False)
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
    res = gate(role="curate").check(mem.head(), w.candidate(mem, {w.TEST_FILE: None}), man)
    assert not res.passed and any(
        f"suite memory/acct/tests {lost}" in r for r in res.reasons
    )
    stated = {
        **man,
        "tests_changed": {w.TEST_FILE: "the function is covered by test_accounts.py"},
    }
    res = gate(role="curate").check(mem.head(), w.candidate(mem, {w.TEST_FILE: None}), stated)
    assert not any(lost in r for r in res.reasons), res.reasons


@pytest.mark.xfail(reason=PARSED, strict=False)
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
