"""The whole gate with v2.1 checks, in the jail (bubblewrap) on keyless w139."""

import json
import shutil

import pytest

from unify.memory_v2 import fixtures as fx
from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.episodes import Action
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.gate import Gate
from unify.memory_v2.gate_v21 import V21Config
from unify.memory_v2.gitio import Repo
from tests.memory_v2.test_episodes import _ep
from tests.memory_v2.test_gate import _candidate

pytestmark = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap required",
)

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
def world21(tmp_path):
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
            v21=V21Config(checks=True, episodes=EPS.get, **cfg),
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


def test_v21_gate_passes_and_records_verification(world21):
    mem, ev, blobs, gate = world21
    files, fixtures = _files_and_fixtures(blobs)
    res = gate().merge(
        mem.head(),
        _candidate(mem, files),
        _man(fixtures),
        "p1",
        "write",
        None,
        "0.01",
    )
    assert res.passed, res.reasons
    rec = res.verification[ITEM_ID]
    assert (
        rec["mutation"]["total"] > 0
        and rec["mutation"]["killed"] == rec["mutation"]["total"]
    )
    assert rec["guard"] == {"killed": 1, "total": 1}
    assert (
        rec["drawn_inputs"] == 1 and rec["drawn_inputs_read"] == 1
    )  # e2's acct.me response was drawn
    assert rec["exact_assertions"] == 2 and rec["tests"] == [T]
    reasons = json.loads(
        ev.db.execute("SELECT reasons FROM passes WHERE pass_id='p1'").fetchone()[0],
    )
    assert any(r.startswith("v21-verification ") for r in reasons)


def test_a_fixture_not_made_by_fixture_tool_is_refused(world21):
    mem, ev, blobs, gate = world21
    files, fixtures = _files_and_fixtures(blobs)
    fixtures.pop("memory/acct/tests/data/me.json")
    res = gate().check(mem.head(), _candidate(mem, files), _man(fixtures))
    assert not res.passed
    assert any("me.json did not come through fixture()" in r for r in res.reasons)


def test_no_trusted_exact_assertion_is_refused(world21):
    mem, ev, blobs, gate = world21
    files, fixtures = _files_and_fixtures(blobs, test=WEAK_TEST)
    res = gate().check(mem.head(), _candidate(mem, files), _man(fixtures))
    assert not res.passed and any("[v21:exact]" in r for r in res.reasons)


def test_curate_due_is_a_note_not_a_refusal(world21):
    mem, ev, blobs, gate = world21
    files, fixtures = _files_and_fixtures(blobs)
    res = gate(index_tokens=1).check(mem.head(), _candidate(mem, files), _man(fixtures))
    assert res.passed, res.reasons
    assert res.curate_due and any("G4 CURATE due" in r for r in res.reasons)


def test_write_may_not_remove_an_item(world21):
    mem, ev, blobs, gate = world21
    files, fixtures = _files_and_fixtures(blobs)
    assert (
        gate()
        .merge(
            mem.head(),
            _candidate(mem, files),
            _man(fixtures),
            "p1",
            "write",
            None,
            "0",
        )
        .passed
    )
    gone = {"memory/acct/users.py": None, T: None}
    man = {
        "items": [],
        "deleted": [ITEM_ID],
        "deleted_tests": [T],
        "unlisted": [],
        "support": [],
        "tests_changed": {T: "the function is retired"},
    }
    res = gate().check(mem.head(), _candidate(mem, gone), man)
    assert not res.passed and any(
        "removals and merges belong to CURATE" in r for r in res.reasons
    )


FIRST_WORD = '''"""Words."""


def first_word(text):
    """The first word of a recorded line.

    Effect: read
    Input: text
    """
    return text.split()[0]
'''
WORD_TEST = """import json
from pathlib import Path

from memory.talk.words import first_word

DATA = Path(__file__).parent / "data"
INPUTS = [json.loads(x) for x in (DATA / "words.first_word.inputs.jsonl").read_text().splitlines() if x]


def test_first_word_of_the_recorded_line():
    assert first_word(INPUTS[0]["input"]) == "hello"
"""


def test_cross_episode_text_input_that_crashes_is_refused(world21):
    mem, ev, blobs, gate = world21
    rel = fx.inputs_file("memory.talk.words:first_word")
    line, entry = fx.make(E3, 0, rel, blob=blobs.get, append=True)
    files = {
        "memory/talk/__init__.py": '"""Talk."""\n',
        "memory/talk/words.py": FIRST_WORD,
        "memory/talk/tests/test_words.py": WORD_TEST,
        rel: line.decode(),
    }
    man = {
        "items": [
            {
                "item": "memory.talk.words:first_word",
                "kind": "function",
                "source_episodes": ["e3"],
                "tests": ["memory/talk/tests/test_words.py"],
                "covers": [["e3", 0]],
                "input": "text",
            },
        ],
        "deleted": [],
        "unlisted": [],
        "support": [],
        "fixtures": {rel: entry},
    }
    res = gate().check(mem.head(), _candidate(mem, files), man)
    assert not res.passed
    assert any(
        "from episodes outside its provenance" in r and "IndexError" in r
        for r in res.reasons
    )
    assert res.verification["memory.talk.words:first_word"]["cross_episode"] == {
        "ran": 1,
        "ok": 0,
    }
