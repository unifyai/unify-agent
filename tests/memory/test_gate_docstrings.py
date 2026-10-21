"""The gate in the library mode: the lean docstring standard (G1), its examples as doctests (G3), the soft size
budget (G4), reserved generated paths, and the input shapes a merge records for the export's catalogue.

``lab`` is the gate with every library-mode switch on (``UNIFY_MEMORY_DOCSTRINGS=on``,
``UNIFY_MEMORY_SURFACING=catalogue``, ``UNIFY_MEMORY_SOFT_BUDGET=on``); the defaults are covered by
``test_library_switch_defaults.py``.

Gates here run for real (confined pytest, held-out values), so bubblewrap is required.
"""

import json
import shutil

import pytest

from unify.memory.blobs import BlobStore
from unify.memory.episodes import Action
from unify.memory.evidence import EvidenceStore
from unify.memory.gate import Gate
from unify.memory.library_checks import LibraryConfig
from unify.memory.gitio import Repo
from tests.memory.test_episodes import _ep
from tests.memory.test_gate import KIT, TEST, _candidate

pytestmark = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap required",
)

ERROR_CLASS = '''

class MemoryInputError(ValueError):
    """The input differs in shape from what this function was built from."""
'''
ME = '''

def me(apis):
    """Return the logged-in user's id.

    Args:
        apis: the environment object (input: env), with a `venmo` namespace.

    Returns:
        The user id string that `venmo.me` returns.

    Raises:
        MemoryInputError: when the response has no string `user_id`.

    Example:
        >>> import json
        >>> from unify_memory_testkit import env_from
        >>> rec = json.load(open("env/venmo/tests/me.json"))
        >>> me(env_from([("venmo", "me", {}, rec)]))
        'u-1'

    Effect: read
    Input: env
    """
    user = apis.venmo.me().get("user_id")
    if not isinstance(user, str):
        raise MemoryInputError(
            "expected venmo.me to return a string user_id; call apis.venmo.me() directly",
        )
    return user
'''
DOC_MOD = '__all__ = ["me"]\n' + ERROR_CLASS + ME
BALANCE = '''

def balance(apis):
    """Return the account balance.

    Args:
        apis: the environment object (input: env).

    Returns:
        The balance number that `venmo.balance` returns.

    Raises:
        MemoryInputError: when the response has no numeric `balance`.

    Example:
        >>> import json
        >>> from unify_memory_testkit import env_from
        >>> rec = json.load(open("env/venmo/tests/balance.json"))
        >>> balance(env_from([("venmo", "balance", {}, rec)]))
        3

    Effect: read
    Input: env
    """
    value = apis.venmo.balance().get("balance")
    if not isinstance(value, (int, float)):
        raise MemoryInputError(
            "expected venmo.balance to return a numeric balance; call apis.venmo.balance() directly",
        )
    return value
'''
BALANCE_TEST = """from unify_memory_testkit import env_from
from env.venmo import balance

def test_balance():
    assert balance(env_from([("venmo", "balance", {}, {"balance": 3})])) == 3
"""
ME_JSON = "env/venmo/tests/me.json"
DOC_FILES = {
    "env/venmo/__init__.py": DOC_MOD,
    "env/venmo/tests/test_me.py": TEST,
    ME_JSON: '{"user_id": "u-1"}',  # the recorded response venmo.me gave (cover e1,0)
    "unify_memory_testkit.py": KIT,
}
ME_ITEM = {
    "item": "env/venmo:me",
    "kind": "env_function",
    "source_episodes": ["e1"],
    "tests": ["env/venmo/tests/test_me.py"],
    "covers": [["e1", 0]],
    "input": "env",
}
DOC_MAN = {
    "items": [ME_ITEM],
    "support": ["unify_memory_testkit.py", ME_JSON],
    "skeleton": ["env/venmo"],
}

FEEDBACK = {"type": "Feedback", "score": 3, "notes": ["a"]}
DLG_MOD = '__all__ = ["parse_feedback"]\n' + ERROR_CLASS + '''

def parse_feedback(obs):
    """Read the score from a feedback observation.

    Args:
        obs: the feedback observation dict, as recorded (input: observation).

    Returns:
        The integer score.

    Raises:
        MemoryInputError: when the observation is not a dict with an integer `score`.

    Example:
        >>> import json
        >>> parse_feedback(json.load(open("env/dialogue_user/tests/feedback.json")))
        3

    Effect: read
    Input: observation
    """
    if not isinstance(obs, dict) or not isinstance(obs.get("score"), int):
        raise MemoryInputError(
            "expected a feedback dict with an integer score; read the observation directly",
        )
    return obs["score"]
'''
DLG_TEST = """from env.dialogue_user import parse_feedback

def test_parse_feedback():
    assert parse_feedback({"type": "Feedback", "score": 3, "notes": ["a"]}) == 3
"""
FEEDBACK_JSON = "env/dialogue_user/tests/feedback.json"
DLG_ITEM = {
    "item": "env/dialogue_user:parse_feedback",
    "kind": "env_function",
    "source_episodes": ["d1"],
    "tests": ["env/dialogue_user/tests/test_parse_feedback.py"],
    "covers": [["d1", 0]],
    "input": "observation",
}
DLG_MAN = {
    "items": [DLG_ITEM],
    "support": [FEEDBACK_JSON],
    "skeleton": ["env/dialogue_user"],
}


def _dlg_files(module=DLG_MOD, fixture=json.dumps(FEEDBACK)):
    return {
        "env/dialogue_user/__init__.py": module,
        "env/dialogue_user/tests/test_parse_feedback.py": DLG_TEST,
        FEEDBACK_JSON: fixture,
    }


def _lookup(eid, i):
    if (eid, i) == ("e1", 0):
        return Action(0, "venmo", "me", [], {}, {"user_id": "u-1"}, "ok")
    if (eid, i) == ("e1", 3):
        return Action(3, "venmo", "balance", [], {}, {"balance": 3}, "ok")
    if (eid, i) == ("d1", 0):
        return Action(
            0,
            "dialogue:user",
            "say",
            ["hello"],
            {},
            FEEDBACK,
            "ok",
            kind="dialogue",
        )
    return None


@pytest.fixture
def lab(tmp_path):
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ev.index_episode(_ep(episode_id="e1"), "1" * 40)
    ev.index_episode(_ep(episode_id="d1"), "2" * 40)
    gate = Gate(
        mem,
        ev,
        BlobStore(tmp_path / "b"),
        action_lookup=_lookup,
        ry_mode=LibraryConfig()
    )
    return mem, ev, gate


def _gate(mem, gate, files, man):
    parent = mem.head()
    return gate.check(parent, _candidate(mem, files), man)


# --- G1: the standard --------------------------------------------------------------------------------------


# --- G3: examples run as doctests --------------------------------------------------------------------------


CALL = '        >>> me(env_from([("venmo", "me", {}, rec)]))\n        \'u-1\'\n'


# --- G4: a soft budget -------------------------------------------------------------------------------------


# --- generated paths ---------------------------------------------------------------------------------------


# --- input shapes, recorded at a merge and found from the export -------------------------------------------


# --- a fixture of exactly a low-information recorded shape passes -------------------------------------------

PAIRS_MOD = '__all__ = ["read_pairs"]\n' + ERROR_CLASS + '''

def read_pairs(path):
    """Read a headerless table of integer pairs.

    Args:
        path: the pairs CSV file, without a header (a path).

    Returns:
        A list of (int, int) tuples, one per row.

    Raises:
        MemoryInputError: when a row is not two integers.

    Example:
        >>> read_pairs("env/worktree_workspace/tests/pairs.csv")
        [(1, 2), (3, 4)]

    Effect: read
    Input: path
    """
    out = []
    with open(path) as fh:
        for line in fh.read().splitlines():
            cells = line.split(",")
            if len(cells) != 2 or not all(c.strip().lstrip("-").isdigit() for c in cells):
                raise MemoryInputError(
                    "expected rows of two comma-separated integers; read the file directly",
                )
            out.append((int(cells[0]), int(cells[1])))
    return out
'''
PAIRS_TEST = """from env.worktree_workspace import read_pairs

def test_read_pairs():
    assert read_pairs("env/worktree_workspace/tests/pairs.csv") == [(1, 2), (3, 4)]
"""
GRID_MOD = '__all__ = ["grid_size"]\n' + ERROR_CLASS + '''

def grid_size(obs):
    """Return a grid observation's (rows, columns).

    Args:
        obs: the grid, a list of lists of integers, as recorded (input: observation).

    Returns:
        A (rows, columns) tuple.

    Raises:
        MemoryInputError: when the observation is not a non-empty list of integer lists.

    Example:
        >>> import json
        >>> grid_size(json.load(open("env/dialogue_user/tests/grid.json")))
        (2, 2)

    Effect: read
    Input: observation
    """
    if not isinstance(obs, list) or not obs or not all(
        isinstance(r, list) and all(isinstance(c, int) for c in r) for r in obs
    ):
        raise MemoryInputError(
            "expected a non-empty list of integer lists; read the observation directly",
        )
    return len(obs), len(obs[0])
'''
GRID_TEST = """from env.dialogue_user import grid_size

def test_grid_size():
    assert grid_size([[0, 1], [1, 0]]) == (2, 2)
"""
TOTAL_MOD = '__all__ = ["total"]\n' + ERROR_CLASS + '''

def total(obs):
    """Sum a list-of-integers observation.

    Args:
        obs: the integers, a list, as recorded (input: observation).

    Returns:
        Their sum.

    Raises:
        MemoryInputError: when the observation is not a list of integers.

    Example:
        >>> import json
        >>> total(json.load(open("env/dialogue_user/tests/numbers.json")))
        6

    Effect: read
    Input: observation
    """
    if not isinstance(obs, list) or not all(isinstance(n, int) for n in obs):
        raise MemoryInputError(
            "expected a list of integers; read the observation directly",
        )
    return sum(obs)
'''
TOTAL_TEST = """from env.dialogue_user import total

def test_total():
    assert total([4, 5, 6]) == 15
"""


def _shape_lab(tmp_path, action, eid):
    """A gate with every library-mode switch on whose only recorded action is *action* at (*eid*, 0)."""
    mem = Repo.init_bare(tmp_path / "shape-mem.git")
    ev = EvidenceStore(tmp_path / "shape-e.sqlite")
    ev.index_episode(_ep(episode_id=eid, actions=[action]), "4" * 40)
    blobs = BlobStore(tmp_path / "shape-b")
    gate = Gate(
        mem,
        ev,
        blobs,
        action_lookup=lambda e, i: action if (e, i) == (eid, 0) else None,
        ry_mode=LibraryConfig()
    )
    return mem, gate, blobs


def _shape_case(tmp_path, kind, fixture):
    """(gate, files, manifest) for one low-information shape: a headerless table, a raw grid or a list of scalars."""
    if kind == "pairs":
        blobs = BlobStore(tmp_path / "shape-b")
        sha = blobs.put(b"5,6\n7,8\n")
        action = Action(
            0,
            "worktree:workspace",
            "read",
            ["data/pairs.csv"],
            {},
            {
                "blob_before": sha,
                "blob_after": sha,
                "size": 8,
                "shape": {"format": "csv"},
            },
            "ok",
            "read",
            kind="worktree",
        )
        channel, name, mod, test, fx = (
            "worktree_workspace",
            "read_pairs",
            PAIRS_MOD,
            PAIRS_TEST,
            "pairs.csv",
        )
        form = "path"
    else:
        obs = [[0, 1], [1, 0]] if kind == "grid" else [4, 5, 6]
        action = Action(
            0,
            "dialogue:user",
            "reply",
            ["do"],
            {},
            obs,
            "ok",
            kind="dialogue",
        )
        channel, form = "dialogue_user", "observation"
        name, mod, test, fx = (
            ("grid_size", GRID_MOD, GRID_TEST, "grid.json")
            if kind == "grid"
            else ("total", TOTAL_MOD, TOTAL_TEST, "numbers.json")
        )
    mem, gate, _ = _shape_lab(tmp_path, action, "s1")
    fixture_path = f"env/{channel}/tests/{fx}"
    test_path = f"env/{channel}/tests/test_{name}.py"
    files = {f"env/{channel}/__init__.py": mod, test_path: test, fixture_path: fixture}
    man = {
        "items": [
            {
                "item": f"env/{channel}:{name}",
                "kind": "env_function",
                "source_episodes": ["s1"],
                "tests": [test_path],
                "covers": [["s1", 0]],
                "input": form,
            },
        ],
        "support": [fixture_path],
        "skeleton": [f"env/{channel}"],
    }
    return mem, gate, files, man, fixture_path, f"env/{channel}:{name}"


@pytest.mark.parametrize(
    ("kind", "fixture"),
    [("pairs", "1,2\n3,4\n"), ("grid", "[[1, 0], [0, 1]]"), ("list", "[1, 2, 3]")],
    ids=["headerless-csv", "raw-grid", "scalar-list"],
)
def test_a_fixture_of_exactly_a_low_information_shape_passes(tmp_path, kind, fixture):
    """No named key, yet the fixture is the recorded input's shape; the gate admits it."""
    mem, gate, files, man, _, _ = _shape_case(tmp_path, kind, fixture)
    channel = man["skeleton"][0].split("/", 1)[1]
    res = gate.merge(
        mem.head(),
        _candidate(mem, files),
        man,
        "p-n1",
        "incremental",
        channel,
        "0",
    )
    assert res.passed, res.reasons


def test_a_headed_fixture_for_a_headerless_recorded_table_is_refused(tmp_path):
    mem, gate, files, man, fixture_path, item = _shape_case(
        tmp_path,
        "pairs",
        "a,b\n1,2\n",
    )
    res = gate.check(mem.head(), _candidate(mem, files), man)
    assert not res.passed
    assert (
        f"G3: the examples of {item} read no fixture shaped like an input it covers ({fixture_path})"
    ) in res.reasons, res.reasons


# --- a fixture holding a collection of recorded inputs (validation of 394bd9afd, 8 Oct) ----------------------
# The writer's natural fixture is a JSON list of the recorded observations (or records by name), and an example picks
# one; the whole file is not shaped like one input, so the check also shapes the collection's elements.

_PICKS = {
    "list": ("[{0}, {1}]", "[0]"),
    "records": ('{{"first": {0}, "second": {1}}}', '["first"]'),
    "wrapped": ('{{"rows": [{0}, {1}]}}', '["rows"][0]'),
}


def test_fixture_elements_are_bounded_and_value_free():
    from unify.memory.gate import FIXTURE_ELEMENTS, _fixture_elements

    big = json.dumps([{"score": i} for i in range(1000)]).encode()
    assert len(_fixture_elements(big)) == FIXTURE_ELEMENTS
    lines = b"\n".join(json.dumps({"score": i}).encode() for i in range(50))
    assert len(_fixture_elements(lines)) == FIXTURE_ELEMENTS
    assert _fixture_elements(b"\xff\xfe") == [] and _fixture_elements(b"3") == []
    nested = json.dumps({"a": [{"x": 1}] * 40, "b": [{"x": 2}] * 40}).encode()
    assert len(_fixture_elements(nested)) <= 2 * FIXTURE_ELEMENTS
