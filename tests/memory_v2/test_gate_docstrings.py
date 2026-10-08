"""The gate under v2.1: the lean docstring standard (G1), its examples as doctests (G3), the soft size
budget (G4), reserved generated paths, and the input shapes a merge records for the export's catalogue.

Gates here run for real (confined pytest, held-out values), so bubblewrap is required.
"""

import importlib.util
import shutil

import pytest

from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.catalogue import body_digest, write_generated
from unify.memory_v2.episodes import Action
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.gate import Gate
from unify.memory_v2.gitio import Repo
from unify.memory_v2.integration.checkout import export_checkout
from unify.memory_v2.memory_helper import value_shape
from unify.memory_v2.snapshot import item_bodies
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
        >>> from unify_memory_testkit import env_from
        >>> me(env_from([("venmo", "me", {}, {"user_id": "u-1"})]))
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
        >>> from unify_memory_testkit import env_from
        >>> balance(env_from([("venmo", "balance", {}, {"balance": 3})]))
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
DOC_FILES = {
    "env/venmo/__init__.py": DOC_MOD,
    "env/venmo/tests/test_me.py": TEST,
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
    "support": ["unify_memory_testkit.py"],
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
        >>> parse_feedback({"type": "Feedback", "score": 3, "notes": ["a"]})
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
        "support": ["unify_memory_testkit.py"],
        "skeleton": ["env/venmo"],
    }
    res = _gate(mem, gate, files, man)
    assert res.passed, res.reasons


def test_without_the_switch_the_standard_is_not_applied(lab, tmp_path):
    mem, ev, _ = lab
    plain = Gate(mem, ev, BlobStore(tmp_path / "b2"), action_lookup=_lookup)
    res = _gate(mem, plain, FILES, {**DOC_MAN, "skeleton": []})
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


def test_an_example_runs_with_the_library_root_as_working_directory(lab):
    mem, _, gate = lab
    reads_fixture = DOC_MOD.replace(
        '        >>> me(env_from([("venmo", "me", {}, {"user_id": "u-1"})]))\n',
        "        >>> import json\n"
        '        >>> rec = json.load(open("env/venmo/tests/me.json"))\n'
        '        >>> me(env_from([("venmo", "me", {}, rec)]))\n',
    )
    files = {
        **DOC_FILES,
        "env/venmo/__init__.py": reads_fixture,
        "env/venmo/tests/me.json": '{"user_id": "u-1"}',
    }
    man = {**DOC_MAN, "support": ["unify_memory_testkit.py", "env/venmo/tests/me.json"]}
    res = _gate(mem, gate, files, man)
    assert res.passed, res.reasons


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


@pytest.mark.parametrize("path", ["README.md", ".memory/catalog.json"])
def test_a_commit_may_not_write_a_generated_path(lab, path):
    mem, _, gate = lab
    res = _gate(mem, gate, {**DOC_FILES, path: "mine\n"}, DOC_MAN)
    assert not res.passed
    assert any(
        r.startswith(f"G1: file {path} is generated by the harness")
        for r in res.reasons
    ), res.reasons


# --- input shapes, recorded at a merge and found from the export -------------------------------------------


def test_a_merge_records_input_shapes_and_the_export_finds_by_them(lab, tmp_path):
    mem, ev, gate = lab
    parent = mem.head()
    files = {
        "env/dialogue_user/__init__.py": DLG_MOD,
        "env/dialogue_user/tests/test_parse_feedback.py": DLG_TEST,
    }
    cand = _candidate(mem, files)
    item = "env/dialogue_user:parse_feedback"
    man = {
        "items": [
            {
                "item": item,
                "kind": "env_function",
                "source_episodes": ["d1"],
                "tests": ["env/dialogue_user/tests/test_parse_feedback.py"],
                "covers": [["d1", 0]],
                "input": "observation",
            },
        ],
        "skeleton": ["env/dialogue_user"],
    }
    res = gate.merge(
        parent,
        cand,
        man,
        "p-shapes",
        "incremental",
        "dialogue_user",
        "0.01",
    )
    assert res.passed, res.reasons
    export = tmp_path / "memory-checkout"
    export_checkout(mem.git_dir, mem.head(), export)
    digest = body_digest(item_bodies(export)[item][1])
    assert ev.input_shapes(item, digest) == [value_shape(FEEDBACK)]
    write_generated(export, shapes=ev.input_shapes)
    spec = importlib.util.spec_from_file_location(
        "memory_after_merge",
        export / "memory.py",
    )
    memory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(memory)
    found = memory.find({"type": "Feedback", "score": 9, "notes": ["b", "c"]})
    assert [(f.name, f.match) for f in found] == [
        ("env.dialogue_user.parse_feedback", "exact"),
    ]
    assert memory.find({"room": "hall"}) == []
    assert "- a value with keys notes, score, type" in memory.describe("parse_feedback")


def test_a_refused_merge_records_no_shapes(lab):
    mem, ev, gate = lab
    parent = mem.head()
    wrong = DLG_MOD.replace("        3\n", "        4\n", 1)
    cand = _candidate(
        mem,
        {
            "env/dialogue_user/__init__.py": wrong,
            "env/dialogue_user/tests/test_parse_feedback.py": DLG_TEST,
        },
    )
    man = {
        "items": [
            {
                "item": "env/dialogue_user:parse_feedback",
                "kind": "env_function",
                "source_episodes": ["d1"],
                "tests": ["env/dialogue_user/tests/test_parse_feedback.py"],
                "covers": [["d1", 0]],
                "input": "observation",
            },
        ],
        "skeleton": ["env/dialogue_user"],
    }
    res = gate.merge(
        parent,
        cand,
        man,
        "p-refused",
        "incremental",
        "dialogue_user",
        "0.01",
    )
    assert not res.passed
    assert ev.db.execute("SELECT COUNT(*) FROM input_shapes").fetchone() == (0,)
