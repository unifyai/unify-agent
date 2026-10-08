"""The gate under v2.1: the lean docstring standard (G1), its examples as doctests (G3), the soft size
budget (G4), reserved generated paths, and the input shapes a merge records for the export's catalogue.

``lab`` is the gate with every v2.1 switch on (``UNIFY_MEMORY_V2_DOCSTRINGS=on``,
``UNIFY_MEMORY_V2_SURFACING=catalogue``, ``UNIFY_MEMORY_V2_SOFT_BUDGET=on``); the defaults are covered by
``test_v21_switch_defaults.py``.

Gates here run for real (confined pytest, held-out values), so bubblewrap is required.
"""

import importlib.util
import json
import shutil

import pytest

from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.catalogue import write_generated
from unify.memory_v2.episodes import Action
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.gate import Gate
from unify.memory_v2.gitio import Repo
from unify.memory_v2.integration.checkout import export_checkout
from unify.memory_v2.memory_helper import value_shape
from unify.memory_v2.shape_rows import lookup_from, shapes_at
from tests.memory_v2.test_episodes import _ep
from tests.memory_v2.test_gate import FILES, KIT, MOD, TEST, _candidate, _merged

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
        docstring_standard=True,
        surfacing="catalogue",
        soft_budget=True,
    )
    return mem, ev, gate


def _gate(mem, gate, files, man):
    parent = mem.head()
    return gate.check(parent, _candidate(mem, files), man)


# --- G1: the standard --------------------------------------------------------------------------------------


def test_a_function_meeting_the_standard_passes(lab):
    mem, _, gate = lab
    res = _gate(mem, gate, DOC_FILES, DOC_MAN)
    assert res.passed, res.reasons
    assert all(res.checks.values())


def test_a_missing_required_section_is_refused_with_a_precise_reason(lab):
    mem, _, gate = lab
    no_returns = DOC_MOD.replace(
        "    Returns:\n        The user id string that `venmo.me` returns.\n\n",
        "",
    )
    assert no_returns != DOC_MOD
    res = _gate(mem, gate, {**DOC_FILES, "env/venmo/__init__.py": no_returns}, DOC_MAN)
    assert not res.passed and not res.checks["G1"]
    assert (
        "G1: env/venmo:me docstring has no Returns: section" in res.reasons
    ), res.reasons
    assert res.checks["G3"]  # its example still runs green


def test_the_first_argument_must_name_its_input_form(lab):
    mem, _, gate = lab
    vague = DOC_MOD.replace(
        "(input: env), with a `venmo` namespace",
        "with a `venmo` namespace",
    )
    res = _gate(mem, gate, {**DOC_FILES, "env/venmo/__init__.py": vague}, DOC_MAN)
    assert (
        "G1: env/venmo:me docstring Args: the entry of the first parameter `apis` does not name "
        "its input form `env`"
    ) in res.reasons, res.reasons


def test_a_refusal_without_a_useful_message_is_refused(lab):
    mem, _, gate = lab
    terse = DOC_MOD.replace(
        '"expected venmo.me to return a string user_id; call apis.venmo.me() directly"',
        '"bad"',
    )
    res = _gate(mem, gate, {**DOC_FILES, "env/venmo/__init__.py": terse}, DOC_MAN)
    assert not res.checks["G1"]
    assert any(
        r.startswith(
            "G1: env/venmo:me raises MemoryInputError with a 3-character message",
        )
        for r in res.reasons
    ), res.reasons


def test_the_check_tool_reports_a_missing_section_before_any_run(lab, tmp_path):
    mem, _, gate = lab
    tree = tmp_path / "tree"
    no_example = (
        DOC_MOD.split("    Example:")[0]
        + '    Effect: read\n    Input: env\n    """\n    return 1\n'
    )
    for rel, text in {**DOC_FILES, "env/venmo/__init__.py": no_example}.items():
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text(text)
    reasons = gate.preview(mem.head(), tree, DOC_MAN)
    assert (
        "G1: env/venmo:me docstring has no Example: section with a `>>>` example"
        in reasons
    )


def test_unchanged_legacy_functions_are_not_held_to_the_standard(lab):
    mem, _, gate = lab
    _merged(mem, FILES)  # `me` without the sections, from before the standard
    module = (
        MOD.replace('__all__ = ["me"]', '__all__ = ["me", "balance"]')
        + ERROR_CLASS
        + BALANCE
    )
    files = {
        **FILES,
        "env/venmo/__init__.py": module,
        "env/venmo/tests/test_balance.py": BALANCE_TEST,
        "env/venmo/tests/balance.json": '{"balance": 3}',
    }
    man = {
        "items": [
            ME_ITEM,
            {
                **ME_ITEM,
                "item": "env/venmo:balance",
                "tests": ["env/venmo/tests/test_balance.py"],
                "covers": [["e1", 3]],
            },
        ],
        "support": ["unify_memory_testkit.py", "env/venmo/tests/balance.json"],
        "skeleton": ["env/venmo"],
    }
    res = _gate(mem, gate, files, man)
    assert res.passed, res.reasons


def test_without_the_switch_the_standard_is_not_applied(lab, tmp_path):
    mem, ev, _ = lab
    plain = Gate(mem, ev, BlobStore(tmp_path / "b2"), action_lookup=_lookup)
    res = _gate(
        mem,
        plain,
        FILES,
        {**DOC_MAN, "support": ["unify_memory_testkit.py"], "skeleton": []},
    )
    assert res.passed, res.reasons


# --- G3: examples run as doctests --------------------------------------------------------------------------


def test_a_failing_example_refuses_the_function(lab):
    mem, _, gate = lab
    wrong = DOC_MOD.replace("        'u-1'\n", "        'u-2'\n")
    assert wrong != DOC_MOD
    res = _gate(mem, gate, {**DOC_FILES, "env/venmo/__init__.py": wrong}, DOC_MAN)
    assert not res.passed and res.checks["G1"] and not res.checks["G3"]
    assert any(
        r.startswith(
            "G3: an example in the docstring of env/venmo:me fails as a doctest",
        )
        for r in res.reasons
    ), res.reasons


CALL = '        >>> me(env_from([("venmo", "me", {}, rec)]))\n        \'u-1\'\n'


@pytest.mark.parametrize(
    ("example", "reason"),
    [
        (
            '        >>> me(env_from([("venmo", "me", {}, rec)]))  # doctest: +SKIP\n        \'u-1\'\n',
            "Example: an example is skipped (`+SKIP`); every example must run",
        ),
        (
            "        >>> x = 1\n",
            "Example: no example calls `me(...)` with an argument and shows its result",
        ),
        (
            '        >>> me(env_from([("venmo", "me", {}, rec)]))\n        ...\n',
            "Example: no example calls `me(...)` with an argument and shows its result",
        ),
        (
            "        >>> rec\n        {...}\n",
            "Example: no example calls `me(...)` with an argument and shows its result",
        ),
    ],
    ids=["skip", "trivial", "ellipsis-only", "no-call"],
)
def test_an_example_that_is_not_a_real_call_is_refused(lab, example, reason):
    mem, _, gate = lab
    module = DOC_MOD.replace(CALL, example)
    assert module != DOC_MOD
    res = _gate(mem, gate, {**DOC_FILES, "env/venmo/__init__.py": module}, DOC_MAN)
    assert not res.checks["G1"]
    assert any(
        r.startswith(f"G1: env/venmo:me docstring {reason}") for r in res.reasons
    ), res.reasons


def test_an_example_must_read_a_fixture_the_commit_holds(lab):
    mem, _, gate = lab
    inline = DOC_MOD.replace(
        '        >>> rec = json.load(open("env/venmo/tests/me.json"))\n',
        '        >>> rec = {"user_id": "u-1"}\n',
    )
    res = _gate(mem, gate, {**DOC_FILES, "env/venmo/__init__.py": inline}, DOC_MAN)
    assert (
        "G1: env/venmo:me docstring Example: no example reads a recorded input from a fixture under "
        "env/venmo/tests/"
    ) in res.reasons, res.reasons
    missing = DOC_MOD.replace("tests/me.json", "tests/gone.json")
    res = _gate(mem, gate, {**DOC_FILES, "env/venmo/__init__.py": missing}, DOC_MAN)
    assert (
        "G1: env/venmo:me docstring Example: reads env/venmo/tests/gone.json, which the commit does "
        "not hold"
    ) in res.reasons, res.reasons


def test_the_runner_requires_the_function_to_be_called(lab):
    """An example that shows the call but never makes it (its name is rebound first) passes G1's static
    check and fails G3: the counter never fires."""
    mem, _, gate = lab
    dodge = DOC_MOD.replace(
        CALL,
        '        >>> me = lambda apis: "u-1"\n' + CALL,
    )
    assert dodge != DOC_MOD
    res = _gate(mem, gate, {**DOC_FILES, "env/venmo/__init__.py": dodge}, DOC_MAN)
    assert res.checks["G1"] and not res.checks["G3"], res.reasons
    assert any("fails as a doctest" in r for r in res.reasons), res.reasons


@pytest.mark.parametrize(
    "example",
    [
        '        >>> me(env_from([("venmo", "me", {}, rec)])) if False else "u-1"\n        \'u-1\'\n',
        '        >>> me(env_from([("venmo", "me", {}, rec)])) == me(env_from([("venmo", "me", {}, rec)]))\n'
        "        True\n",
        '        >>> (me(env_from([("venmo", "me", {}, rec)])), 3)[1]\n        3\n',
        '        >>> isinstance(me(env_from([("venmo", "me", {}, rec)])), str)\n        True\n',
        '        >>> [me(env_from([("venmo", "me", {}, rec)]))]\n        [...]\n',
    ],
    ids=["conditional", "self-comparison", "discarded", "wrapped", "bracket-ellipsis"],
)
def test_a_vacuous_example_is_refused_before_any_run(lab, example):
    """Minor (re-review I2): the shown output must be the call's own value, made once, with content."""
    mem, _, gate = lab
    module = DOC_MOD.replace(CALL, example)
    assert module != DOC_MOD
    res = _gate(mem, gate, {**DOC_FILES, "env/venmo/__init__.py": module}, DOC_MAN)
    assert not res.checks["G1"]
    assert any(
        r.startswith("G1: env/venmo:me docstring Example: no example calls `me(...)`")
        for r in res.reasons
    ), res.reasons


def test_an_example_must_pass_the_fixture_to_the_call(lab):
    mem, _, gate = lab
    ignored = DOC_MOD.replace(
        CALL,
        '        >>> me(env_from([("venmo", "me", {}, {"user_id": "u-1"})]))\n        \'u-1\'\n',
    )
    res = _gate(mem, gate, {**DOC_FILES, "env/venmo/__init__.py": ignored}, DOC_MAN)
    assert (
        "G1: env/venmo:me docstring Example: the example showing `me(...)`'s result does not pass it "
        "the recorded fixture (name the fixture's path, or a name bound to it, in the call's arguments)"
    ) in res.reasons, res.reasons


def test_examples_run_with_a_fixed_hash_seed(lab):
    mem, _, gate = lab
    seeded = DOC_MOD.replace(
        CALL,
        CALL
        + "        >>> import os\n        >>> os.environ.get(\"PYTHONHASHSEED\")\n        '0'\n",
    )
    res = _gate(mem, gate, {**DOC_FILES, "env/venmo/__init__.py": seeded}, DOC_MAN)
    assert res.passed, res.reasons


def test_an_example_fixture_must_have_a_covered_input_shape(lab):
    """The fixture is a recorded input: it has the shape of one of the function's validated covers."""
    mem, _, gate = lab
    parent = mem.head()
    cand = _candidate(mem, _dlg_files(fixture='{"score": 3}'))
    res = gate.merge(
        parent,
        cand,
        DLG_MAN,
        "p-fixture",
        "incremental",
        "dialogue_user",
        "0.01",
    )
    assert not res.passed
    assert (
        "G3: the examples of env/dialogue_user:parse_feedback read no fixture shaped like an input it "
        "covers (env/dialogue_user/tests/feedback.json)"
    ) in res.reasons, res.reasons


# --- G4: a soft budget -------------------------------------------------------------------------------------


def test_growth_past_the_old_4000_token_index_cap_is_not_refused(lab):
    mem, _, gate = lab
    long = DOC_MOD.replace(
        "Return the logged-in user's id.",
        "Return the logged-in user's id, " + "and say so " * 1500 + "at length.",
    )
    res = _gate(mem, gate, {**DOC_FILES, "env/venmo/__init__.py": long}, DOC_MAN)
    assert res.passed and res.checks["G4"], res.reasons
    assert any(
        r.startswith("note: G4 hygiene due: the catalogue") for r in res.reasons
    ), res.reasons


# --- generated paths ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        ".memory/catalog.json",
        "memory.abi3.so",
        "env.py",
        "memory/notes.md",
    ],
)
def test_a_commit_may_not_write_a_reserved_path(lab, path):
    mem, _, gate = lab
    res = _gate(mem, gate, {**DOC_FILES, path: "mine\n"}, DOC_MAN)
    assert not res.passed
    assert any(
        r.startswith(f"G1: file {path} is reserved") for r in res.reasons
    ), res.reasons


# --- input shapes, recorded at a merge and found from the export -------------------------------------------


def test_a_merge_records_input_shapes_and_the_export_finds_by_them(lab, tmp_path):
    mem, ev, gate = lab
    parent = mem.head()
    cand = _candidate(mem, _dlg_files())
    res = gate.merge(
        parent,
        cand,
        DLG_MAN,
        "p-shapes",
        "incremental",
        "dialogue_user",
        "0.01",
    )
    assert res.passed, res.reasons
    item = DLG_ITEM["item"]
    snapshot = ev.commit_shapes(mem.head())  # frozen for the merged commit
    assert snapshot[item]["shapes"] == [value_shape(FEEDBACK)]
    assert snapshot[item]["backfilled"] is False
    export = tmp_path / "memory-checkout"
    export_checkout(mem.git_dir, mem.head(), export)
    write_generated(
        export,
        shapes=lookup_from(shapes_at(mem, ev, mem.head(), export, freeze=True)),
    )
    spec = importlib.util.spec_from_file_location(
        "memory_after_merge",
        export / "memory.py",
    )
    memory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(memory)
    found = memory.find({"type": "Feedback", "score": 9, "notes": ["b"]})
    assert [(f.name, f.match, f.reason) for f in found] == [
        ("env.dialogue_user.parse_feedback", "exact", "matched notes, score, type"),
    ]
    assert memory.find({"room": "hall"}) == []
    assert "- a value with keys notes, score, type" in memory.describe("parse_feedback")


def test_a_refused_merge_records_no_shapes(lab):
    mem, ev, gate = lab
    parent = mem.head()
    wrong = DLG_MOD.replace("        3\n", "        4\n", 1)
    cand = _candidate(mem, _dlg_files(module=wrong))
    res = gate.merge(
        parent,
        cand,
        DLG_MAN,
        "p-refused",
        "incremental",
        "dialogue_user",
        "0.01",
    )
    assert not res.passed
    tables = {
        r[0] for r in ev.db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    if "shape_commits" in tables:
        assert ev.db.execute("SELECT COUNT(*) FROM shape_commits").fetchone() == (0,)
    assert ev.commit_shapes(cand) is None


# --- N1: a fixture of exactly a low-information recorded shape passes ---------------------------------------

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
    """A gate with every v2.1 switch on whose only recorded action is *action* at (*eid*, 0)."""
    mem = Repo.init_bare(tmp_path / "shape-mem.git")
    ev = EvidenceStore(tmp_path / "shape-e.sqlite")
    ev.index_episode(_ep(episode_id=eid, actions=[action]), "4" * 40)
    blobs = BlobStore(tmp_path / "shape-b")
    gate = Gate(
        mem,
        ev,
        blobs,
        action_lookup=lambda e, i: action if (e, i) == (eid, 0) else None,
        docstring_standard=True,
        surfacing="catalogue",
        soft_budget=True,
    )
    return mem, gate, blobs


def _shape_case(tmp_path, kind, fixture):
    """(gate, files, manifest) for one N1 shape: a headerless table, a raw grid or a list of scalars."""
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
    """N1: no named key, yet the fixture is the recorded input's shape; the gate admits it."""
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
# Sol's natural fixture is a JSON list of the recorded observations (or records by name), and an example picks
# one; the whole file is not shaped like one input, so the check also shapes the collection's elements.

_PICKS = {
    "list": ("[{0}, {1}]", '[0]'),
    "records": ('{{"first": {0}, "second": {1}}}', '["first"]'),
    "wrapped": ('{{"rows": [{0}, {1}]}}', '["rows"][0]'),
}


@pytest.mark.parametrize("layout", sorted(_PICKS))
def test_an_example_fixture_may_be_a_collection_of_recorded_inputs(lab, layout):
    mem, _, gate = lab
    template, pick = _PICKS[layout]
    other = dict(FEEDBACK, score=5)
    fixture = template.format(json.dumps(FEEDBACK), json.dumps(other))
    module = DLG_MOD.replace(
        '>>> parse_feedback(json.load(open("env/dialogue_user/tests/feedback.json")))',
        f'>>> parse_feedback(json.load(open("env/dialogue_user/tests/feedback.json")){pick})',
    )
    assert module != DLG_MOD
    res = gate.merge(
        mem.head(),
        _candidate(mem, _dlg_files(module=module, fixture=fixture)),
        DLG_MAN,
        f"p-collection-{layout}",
        "incremental",
        "dialogue_user",
        "0.01",
    )
    assert not any("read no fixture shaped" in r for r in res.reasons), res.reasons
    assert res.passed, res.reasons


def test_a_collection_of_wrongly_shaped_elements_is_still_refused(lab):
    mem, _, gate = lab
    module = DLG_MOD.replace(
        '>>> parse_feedback(json.load(open("env/dialogue_user/tests/feedback.json")))',
        '>>> parse_feedback(json.load(open("env/dialogue_user/tests/feedback.json"))[0])',
    )
    res = gate.merge(
        mem.head(),
        _candidate(mem, _dlg_files(module=module, fixture='[[1, 2], [3, 4]]')),
        DLG_MAN,
        "p-collection-wrong",
        "incremental",
        "dialogue_user",
        "0.01",
    )
    assert not res.passed
    assert (
        "G3: the examples of env/dialogue_user:parse_feedback read no fixture shaped like an input it "
        "covers (env/dialogue_user/tests/feedback.json)"
    ) in res.reasons, res.reasons


def test_fixture_elements_are_bounded_and_value_free():
    from unify.memory_v2.gate import FIXTURE_ELEMENTS, _fixture_elements

    big = json.dumps([{"score": i} for i in range(1000)]).encode()
    # the first FIXTURE_ELEMENTS records, then one level down their values: at most 2 x FIXTURE_ELEMENTS in all
    elements = _fixture_elements(big)
    assert elements[:FIXTURE_ELEMENTS] == [{"score": i} for i in range(FIXTURE_ELEMENTS)]
    assert len(elements) <= 2 * FIXTURE_ELEMENTS
    lines = b"\n".join(json.dumps({"score": i}).encode() for i in range(50))
    assert len(_fixture_elements(lines)) == FIXTURE_ELEMENTS
    assert _fixture_elements(b"\xff\xfe") == [] and _fixture_elements(b"3") == []
    nested = json.dumps({"a": [{"x": 1}] * 40, "b": [{"x": 2}] * 40}).encode()
    assert len(_fixture_elements(nested)) <= 2 * FIXTURE_ELEMENTS
