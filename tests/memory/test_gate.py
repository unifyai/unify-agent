import hashlib
import os
import shutil

import pytest

from unify.memory.blobs import BlobStore
from unify.memory.episodes import Action
from unify.memory.evidence import EvidenceStore
from unify.memory.gate import Gate
from unify.memory.library_checks import LibraryConfig
from unify.memory.gitio import Repo
from unify.memory.snapshot import blob_id, tree_listing
from unify.memory.gate_policy import NOT_RUN_GATE
from tests.memory.test_episodes import _ep
from tests.memory.test_overrides import RULE_BODY

pytestmark = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap required",
)

MOD = '''__all__ = ["me"]

def me(apis):
    """Return the logged-in user's id.

    Effect: read
    Input: env
    """
    return apis.venmo.me()["user_id"]
'''
TEST = """from unify_memory_testkit import env_from
from env.venmo import me

def test_me():
    assert me(env_from([("venmo", "me", {}, {"user_id": "u-1"})])) == "u-1"
"""
KIT = """class _C:
    def __init__(s, t, n): s.t, s.n = t, n
    def __getattr__(s, m): return lambda **kw: s.t[(s.n, m, tuple(sorted(kw.items())))]
class _E:
    def __init__(s, t): s.t = t
    def __getattr__(s, n): return _C(s.t, n)
def env_from(rows): return _E({(c, m, tuple(sorted(k.items()))): r for c, m, k, r in rows})
"""
KEY = "sk-or-v1-" + "ab" * 32  # pragma: allowlist secret


def _lookup(eid, i):
    if (eid, i) == ("e1", 0):
        return Action(0, "venmo", "me", [], {}, {"user_id": "u-1"}, "ok")
    if (eid, i) == ("e1", 1):
        return Action(1, "venmo", "pay", [], {}, None, "error")
    if (eid, i) == ("e1", 2):
        return Action(2, "slack", "post", [], {}, {"ok": True}, "ok")
    if (eid, i) == ("e1", 3):
        return Action(3, "venmo", "balance", [], {}, {"balance": 3}, "ok")
    return None


@pytest.fixture
def world(tmp_path):
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ev.index_episode(_ep(episode_id="e1"), "1" * 40)
    gate = Gate(mem, ev, BlobStore(tmp_path / "b"), action_lookup=_lookup, library_mode=LibraryConfig())
    return mem, ev, gate


def _candidate(mem, files, base="main"):
    with mem.temp_checkout(base) as wt:
        for rel, text in files.items():
            if text is None:
                (wt / rel).unlink()
                continue
            (wt / rel).parent.mkdir(parents=True, exist_ok=True)
            (wt / rel).write_text(text)
        return mem.commit_all(wt, "pass", {"Pass": "p1"})


ITEM = {
    "item": "env/venmo:me",
    "kind": "env_function",
    "source_episodes": ["e1"],
    "tests": ["env/venmo/tests/test_me.py"],
    "covers": [["e1", 0]],
    "input": "env",
}
MAN = {
    "items": [ITEM],
    "unlisted": [],
    "deleted": [],
    "support": ["unify_memory_testkit.py"],
}
FILES = {
    "env/venmo/__init__.py": MOD,
    "env/venmo/tests/test_me.py": TEST,
    "unify_memory_testkit.py": KIT,
}


def _man(**item):
    return {**MAN, "items": [{**ITEM, **item}]}


def _merged(mem, files):
    """Land *files* on main directly (a stand-in for an earlier gated merge)."""
    old = mem.head()
    sha = _candidate(mem, files)
    mem.fast_forward("main", sha, expected_old=old)
    return sha


# --- the brief's cases ---------------------------------------------------------------------------------


# --- support files are copied onto the parent for the red run ----------------------------


# --- G1 --------------------------------------------------------------------------------------------------


# --- G2 --------------------------------------------------------------------------------------------------


# --- G3 decision logic, with an injected runner -----------------------------------------------------------


SLACK = '__all__ = ["x"]\n\ndef x():\n    """One.\n\n    Effect: read\n    Input: env\n    """\n    return 1\n'
SLACK_TEST = "from env.slack import x\n\ndef test_x():\n    assert x() == 1\n"


# --- G4, G5, G6 ------------------------------------------------------------------------------------------


# --- fix round 1: the reviewer's probes and the scope rules -----------------------------------


PROBE_TEST = "from env.venmo import me\n\ndef test_me():\n    assert me\n"
PROBE_BASE = {"env/venmo/__init__.py": MOD, "env/venmo/tests/test_me.py": PROBE_TEST}
PROBE_ITEM = {**ITEM}


# --- fix round 2: the reviewer's second probes ---------------------------


def _fn(name, body):
    return f'\n\ndef {name}(apis):\n    """Do it.\n\n    Effect: read\n    Input: env\n    """\n    {body}\n'


def _venmo(
    names=("me", "login", "old", "rare"),
    tok='return "tok-" + apis.venmo.me()["user_id"]',
    login=None,
    doc="Venmo helpers.",
    head="",
    tail="",
):
    src = f'"""{doc}"""\n{head}__all__ = {list(names)!r}\n'.replace("'", '"')
    src += f"\n\ndef _tok(apis):\n    {tok}\n"
    src += _fn("me", 'return apis.venmo.me()["user_id"]')
    src += login or _fn("login", "return _tok(apis)")
    src += _fn("old", "return 1") + _fn("rare", "return 2")
    return src + tail


ROWS = '[("venmo", "me", {}, {"user_id": "u-1"})]'
T_ME2 = (
    "from unify_memory_testkit import env_from\nfrom env.venmo import me\n\n"
    f"def test_me():\n    assert me(env_from({ROWS})) == 'u-1'\n"
)
T_LOGIN = (
    "from unify_memory_testkit import env_from\nfrom env.venmo import login\n\n"
    f"def test_login():\n    assert login(env_from({ROWS})) == 'tok-u-1'\n"
)
T_OLD = "from env.venmo import old\n\ndef test_old():\n    assert old(None) == 1\n"
LOGIN_FAST = (
    '\n\ndef login(apis, fast=False):\n    """Log in.\n\n    Effect: read\n    Input: env\n    """\n'
    '    return "FAST" if fast else "broken"\n'
)
T_FAST = (
    "from unify_memory_testkit import env_from\nfrom env.venmo import login\n\n"
    f"def test_fast():\n    assert login(env_from({ROWS}), fast=True) == 'FAST'\n"
)
BASE2 = {
    "env/venmo/__init__.py": _venmo(),
    "env/venmo/tests/test_me.py": T_ME2,
    "env/venmo/tests/test_login.py": T_LOGIN,
    "env/venmo/tests/test_old.py": T_OLD,
    "env/venmo/NOTES.md": "Read me first.\n\n## Auth\nLog in first.\n",
    "unify_memory_testkit.py": KIT,
    "workflows/pay.md": "---\ntitle: Pay\n---\nStep 1: check the balance.\n",
}


def _fitem(iid, tests=(), covers=(("e1", 0),)):
    return {
        "item": iid,
        "kind": "env_function",
        "source_episodes": ["e1"],
        "tests": list(tests),
        "covers": [list(c) for c in covers],
        "input": "env",
    }


@pytest.fixture
def r2(world):
    mem, ev, gate = world
    return mem, ev, gate, _merged(mem, BASE2)


# --- fix round 3: plausible consolidator mistakes ---------------------------------------

# --- fix round 4: liveness when a channel was already red -----------------------------------

SLACK_POST = '__all__ = ["post"]\n' + _fn("post", "return 3")
T_POST = "from env.slack import post\n\ndef test_post():\n    assert post(None) == 3\n"
T_POST_RED = (
    "from env.slack import post\n\ndef test_post():\n    assert post(None) == 4\n"
)
T_POST_COLL = (
    "from env.slack import post\nfrom os import no_such_name\n\n"
    "def test_post():\n    assert post(None) == 3\n"
)
T_ME2 = (
    "from unify_memory_testkit import env_from\nfrom env.venmo import me2\n\n"
    "def test_me2():\n"
    '    assert me2(env_from([("venmo", "balance", {}, {"balance": 3})])) == 3\n'
)
VENMO_PASS = (
    {
        "env/venmo/__init__.py": _venmo(
            names=("me", "login", "old", "rare", "me2"),
            tail=_fn("me2", 'return apis.venmo.balance()["balance"]'),
        ),
        "env/venmo/tests/test_me2.py": T_ME2,
    },
    {
        "items": [
            _fitem(
                "env/venmo:me2",
                ["env/venmo/tests/test_me2.py"],
                covers=[("e1", 3)],
            ),
        ],
    },
)
POST_ITEM = _fitem(
    "env/slack:post",
    ["env/slack/tests/test_post.py"],
    covers=[("e1", 2)],
)


def _notes(res):
    return [r for r in res.reasons if r.startswith("note:")]


# --- fix round 5: test repairs cannot carry function changes; a hanging parent is red ---------------------

POST_CHANGED = "return 3 if apis is None else 0"  # same answer for the test's input, a new behaviour


# --- preview: the cheap, read-only checks on an uncommitted tree (the writer's check tool) ------------------------


def _tree(root, files):
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def _objects(mem):
    return sorted(
        p.relative_to(mem.git_dir) for p in (mem.git_dir / "objects").rglob("*")
    )


def test_tree_listing_hashes_in_python_and_refuses_names_git_would_not_take(tmp_path):
    assert blob_id(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    assert blob_id(b"ab", "sha256") == hashlib.sha256(b"blob 2\0ab").hexdigest()
    root = tmp_path / "t"
    (root / "env" / "venmo" / ".git").mkdir(parents=True)
    (root / "env" / "venmo" / ".git" / "config").write_text("x")
    (root / "a").write_text("one")
    (root / "a\r").write_text("two")
    with open(os.path.join(os.fsencode(root), b"bad\xff"), "wb") as f:
        f.write(b"x")
    files, refused = tree_listing(root)
    assert files == {
        "a": ("100644", blob_id(b"one")),
        "a\r": ("100644", blob_id(b"two")),
    }
    assert len(refused) == 1 and refused[0].startswith("unsafe path"), refused


# --- G2: a function that replaces a value it computed from its input under a condition -------

RULE_MOD = MOD.replace('    return apis.venmo.me()["user_id"]\n', RULE_BODY)


# --- the library mode: the whole output of each failing run, and one stage-5 seed per pass -------------------


# --- any edit can land: G3 judges functions, not test files (library mode, writer role) ------------------------

from tests.memory import library_example_world as lw  # noqa: E402
from tests.memory.library_example_world import library_example_world  # noqa: E402,F401 (fixture)

SECOND_TEST = "memory/acct/tests/test_users_second.py"
NAME_TEST = "memory/acct/tests/test_names.py"
NAME_ID = "memory.acct.users:user_name"
NOTE = "notes/acct/ids.md"
#: the module with a second function, user_name, beside user_id
TWO = lw.MODULE_SOURCE + '''

def user_name(response: dict) -> str:
    """The account's display name, read from an acct.me response."""
    if not isinstance(response, dict) or "name" not in response:
        raise MemoryInputError("an acct.me response with a name field is needed")
    return response["name"]
'''
_LOAD = '''import json
from pathlib import Path

from memory.acct.users import {name}

DATA = Path(__file__).parent / "data"


def load(name):
    return json.loads((DATA / name).read_text())
'''
SECOND_SOURCE = _LOAD.format(name="user_id") + '''

def test_reads_the_second_recorded_user():
    assert user_id(load("me_2.json")) == "u-2"
'''
NAME_SOURCE = _LOAD.format(name="user_name") + '''

def test_reads_the_recorded_names():
    assert user_name(load("me_1.json")) == "Ada"
    assert user_name(load("me_2.json")) == "Bo"
'''


def _land_through_gate(world):
    """The shared world's function, landed through the gate (its fixture provenance is kept)."""
    mem, ev, blobs, gate = world
    files, fixtures = lw.files_and_fixtures(blobs.get)
    res = gate().merge(
        mem.head(),
        lw.candidate(mem, files),
        lw.manifest(fixtures),
        "base",
        "write",
        None,
        "0",
    )
    assert res.passed, res.reasons
    return files


def _land_directly(world, files):
    """*files* on main without the gate (a stand-in for an earlier landed library)."""
    mem = world[0]
    old = mem.head()
    sha = lw.candidate(mem, files)
    mem.fast_forward("main", sha, expected_old=old)
    return sha


def _g3_reasons(res):
    return [r for r in res.reasons if r.startswith("G3")]


def _not_run(res, item):
    """The not-run labels of *item* (rev 2.1: the gate runs no test, suite or stored function)."""
    return [r for r in res.verification.get(item, {}).get("label_findings", []) if r.startswith("not run: G3 ")]


def test_any_edit_a_test_only_change_lands_with_its_test_labelled_not_run(library_example_world):
    mem, ev, blobs, gate = library_example_world
    _land_through_gate(library_example_world)
    cand = lw.candidate(mem, {SECOND_TEST: SECOND_SOURCE})
    res = gate().merge(
        mem.head(),
        cand,
        lw.manifest({}, lw.item(tests=[SECOND_TEST])),
        "p2",
        "write",
        None,
        "0",
    )
    assert res.passed and _g3_reasons(res) == [], res.reasons
    assert mem.head() == res.merged
    # rev 2.1: the listed test is not run; its item's label says so
    assert f"not run: G3 {SECOND_TEST}: {NOT_RUN_GATE}" in _not_run(res, lw.ITEM_ID), res.reasons


def test_any_edit_a_pin_only_change_lands_and_a_weakened_one_needs_its_reason(library_example_world):
    mem, ev, blobs, gate = library_example_world
    files = _land_through_gate(library_example_world)
    pinned = files[lw.TEST_FILE].replace(
        'assert user_id(load("me_1.json")) == "u-1"',
        'assert user_id(load("me_1.json")) == load("me_1.json")["user_id"]',
    )
    assert pinned != files[lw.TEST_FILE]
    why = {lw.TEST_FILE: "the expected value is read from the recorded response"}
    res = gate().check(
        mem.head(),
        lw.candidate(mem, {lw.TEST_FILE: pinned}),
        lw.manifest({}, lw.item(), tests_changed=why),
    )
    assert res.passed and _g3_reasons(res) == [], res.reasons
    assert f"not run: G3 {lw.TEST_FILE}: {NOT_RUN_GATE}" in _not_run(res, lw.ITEM_ID), res.reasons
    # a weaker assertion without a stated reason: found by the weakening rule only, never for being not red; rev 2's
    # gate is labels only, so the finding is a label on the item and the change lands
    weaker = files[lw.TEST_FILE].replace(
        'assert user_id(load("me_1.json")) == "u-1"',
        'assert isinstance(user_id(load("me_1.json")), str)',
    )
    res = gate().check(
        mem.head(),
        lw.candidate(mem, {lw.TEST_FILE: weaker}),
        lw.manifest({}, lw.item()),
    )
    findings = [r for r in res.verification[lw.ITEM_ID]["label_findings"] if not r.startswith("not run: ")]
    assert res.passed and findings and all("is weakened" in r for r in findings), res.reasons
    assert all(f"note: label: {lw.ITEM_ID}: {r}" in res.reasons for r in findings), res.reasons


def test_any_edit_a_notes_only_change_lands(library_example_world):
    mem, ev, blobs, gate = library_example_world
    _land_through_gate(library_example_world)
    note = "---\ntitle: Account ids\ndescription: Where a user id is read from.\n---\nThe acct.me response holds it.\n"
    res = gate().merge(
        mem.head(),
        lw.candidate(mem, {NOTE: note}),
        lw.manifest({}, {"item": NOTE, "kind": "note", "source_episodes": ["e1"]}),
        "p2",
        "write",
        None,
        "0",
    )
    assert res.passed and _g3_reasons(res) == [], res.reasons
    assert mem.head() == res.merged


def test_any_edit_tests_only_for_two_unchanged_functions_land(library_example_world):
    mem, ev, blobs, gate = library_example_world
    files, fixtures = lw.files_and_fixtures(blobs.get)
    _land_directly(library_example_world, {**files, lw.MODULE: TWO})
    res = gate().merge(
        mem.head(),
        lw.candidate(mem, {SECOND_TEST: SECOND_SOURCE, NAME_TEST: NAME_SOURCE}),
        lw.manifest(
            {},
            lw.item(tests=[SECOND_TEST]),
            lw.item(item=NAME_ID, tests=[NAME_TEST]),
        ),
        "p2",
        "write",
        None,
        "0",
    )
    assert res.passed and _g3_reasons(res) == [], res.reasons
    assert f"not run: G3 {SECOND_TEST}: {NOT_RUN_GATE}" in _not_run(res, lw.ITEM_ID), res.reasons
    assert f"not run: G3 {NAME_TEST}: {NOT_RUN_GATE}" in _not_run(res, NAME_ID), res.reasons


def test_any_edit_a_behaviour_change_is_labelled_not_run(library_example_world):
    """Rev 2.1: the gate runs no test, so whether a test is red on the parent is not known; the edited function's
    label says its tests and its recorded calls were not run, and the change lands (labels only)."""
    mem, ev, blobs, gate = library_example_world
    files = _land_through_gate(library_example_world)
    edited = lw.MODULE_SOURCE.replace('    return response["user_id"]\n', '    return str(response["user_id"])\n')
    assert edited != lw.MODULE_SOURCE
    res = gate().check(
        mem.head(),
        lw.candidate(mem, {lw.MODULE: edited, lw.TEST_FILE: files[lw.TEST_FILE] + "\n# touched\n"}),
        lw.manifest({}, lw.item()),
    )
    assert res.passed, res.reasons
    labels = _not_run(res, lw.ITEM_ID)
    assert f"not run: G3 {lw.TEST_FILE}: {NOT_RUN_GATE}" in labels, labels
    assert any(r.startswith(f"not run: G3 behaviour on its recorded calls: {NOT_RUN_GATE}") for r in labels), labels


HELPER = lw.MODULE_SOURCE.replace(
    '    return response["user_id"]\n',
    '    return response[_field()]\n\n\ndef _field():\n    return "user_id"\n',
)


def test_any_edit_a_behaviour_change_through_a_helper_is_labelled(library_example_world):
    mem, ev, blobs, gate = library_example_world
    files, fixtures = lw.files_and_fixtures(blobs.get)
    _land_directly(library_example_world, {**files, lw.MODULE: HELPER})
    changed = HELPER.replace('    return "user_id"\n', '    return "name"\n')
    res = gate().check(mem.head(), lw.candidate(mem, {lw.MODULE: changed}), lw.manifest({}, lw.item()))
    # rev 2.1: the gate runs neither the suite nor the function, so a change made through a helper is not found by
    # running anything; the function in the changed module is labelled not run, and the change lands
    assert res.passed, res.reasons
    labels = _not_run(res, lw.ITEM_ID)
    assert any(r.startswith(f"not run: G3 behaviour on its recorded calls: {NOT_RUN_GATE}") for r in labels), labels
    assert f"note: suite memory/acct/tests not run: {NOT_RUN_GATE}" in res.reasons, res.reasons


NAME_TEST_2 = "memory/acct/tests/test_names_second.py"
NAME_SOURCE_2 = _LOAD.format(name="user_name") + '''

def test_reads_the_second_recorded_name():
    assert user_name(load("me_2.json")) == "Bo"
'''
RED_TEST = "memory/acct/tests/test_ids.py"
RED_SOURCE = '''from memory.acct.users import user_id


def test_reads_an_id_field():
    assert user_id({"id": "u-9"}) == "u-9"
'''


def test_any_edit_an_edited_function_and_a_new_test_of_another_land_labelled_not_run(library_example_world):
    mem, ev, blobs, gate = library_example_world
    files, fixtures = lw.files_and_fixtures(blobs.get)
    _land_directly(library_example_world, {**files, lw.MODULE: TWO, NAME_TEST: NAME_SOURCE})
    edited = TWO.replace(
        '    if not isinstance(response, dict) or "user_id" not in response:\n'
        '        raise MemoryInputError("an acct.me response with a user_id field is needed")\n',
        '    if isinstance(response, dict) and "user_id" not in response and "id" in response:\n'
        '        return response["id"]\n'
        '    if not isinstance(response, dict) or "user_id" not in response:\n'
        '        raise MemoryInputError("an acct.me response with a user_id field is needed")\n',
    )
    assert edited != TWO
    res = gate().check(
        mem.head(),
        lw.candidate(mem, {lw.MODULE: edited, RED_TEST: RED_SOURCE, NAME_TEST_2: NAME_SOURCE_2}),
        lw.manifest(
            {},
            lw.item(tests=[lw.TEST_FILE, RED_TEST]),
            lw.item(item=NAME_ID, tests=[NAME_TEST, NAME_TEST_2]),
        ),
    )
    assert res.passed and _g3_reasons(res) == [], res.reasons
    for item, test in ((lw.ITEM_ID, RED_TEST), (NAME_ID, NAME_TEST_2)):
        assert f"not run: G3 {test}: {NOT_RUN_GATE}" in _not_run(res, item), res.reasons


