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
    """An example that names the call but never makes it passes G1's static check and fails G3."""
    mem, _, gate = lab
    dodge = DOC_MOD.replace(
        CALL,
        '        >>> me(env_from([("venmo", "me", {}, rec)])) if False else "u-1"\n        \'u-1\'\n',
    )
    res = _gate(mem, gate, {**DOC_FILES, "env/venmo/__init__.py": dodge}, DOC_MAN)
    assert res.checks["G1"] and not res.checks["G3"], res.reasons
    assert any("fails as a doctest" in r for r in res.reasons), res.reasons


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
