"""Hygiene inside Sol's normal pass (D26): merges, aliases and deletions through the existing gate.

The library is the ARC dialogue pair ``parse_submit_feedback`` / ``submit_state`` (memory-v2.1-hyg's
``arc_env.pysrc``) with its value checks reduced to shape checks, which G2's held-out run requires: the parent's
``submit_state`` re-implements ``parse_submit_feedback``'s validation inline, and ``read_submit_feedback`` is a
second pass's near-duplicate of ``parse_submit_feedback``.
"""

import asyncio
import json
import shutil

import pytest

from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.episodes import Action
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.gate import Gate
from unify.memory_v2.gitio import Repo
from unify.memory_v2.sol_pass import SOL_SYSTEM, PassConfig, SolPass
from unify.memory_v2.trigger import PassRequest
from tests.memory_v2.test_episodes import _ep
from tests.memory_v2.test_gate import _candidate, _merged
from tests.memory_v2.test_sol_pass import Script, _write

pytestmark = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap required",
)

SOLVED = {
    "type": "SubmitFeedback",
    "attempts_used": 1,
    "correct": True,
    "failed": False,
    "valid": True,
}
RETRY = {
    "type": "SubmitFeedback",
    "attempts_used": 2,
    "correct": False,
    "failed": False,
    "valid": True,
}
EXHAUSTED = {
    "type": "SubmitFeedback",
    "attempts_used": 8,
    "correct": False,
    "failed": True,
    "valid": True,
}
RETRY2 = {
    "type": "SubmitFeedback",
    "attempts_used": 3,
    "correct": False,
    "failed": False,
    "valid": True,
}

CH = "env/dialogue_arc"
MODULE = f"{CH}/__init__.py"
PSF, RSF, SS = (
    f"{CH}:{n}"
    for n in ("parse_submit_feedback", "read_submit_feedback", "submit_state")
)
T_PSF, T_RSF, T_SS = (
    f"{CH}/tests/test_{n}.py"
    for n in ("parse_submit_feedback", "read_submit_feedback", "submit_state")
)

HEAD = '''"""Parsers for ARC dialogue observations."""


class MemoryInputError(ValueError):
    """Observation is outside the recorded dialogue shape."""


def parse_submit_feedback(observation):
    """Validate and return a submit feedback observation.

    Effect: read
    Input: observation
    """
    if not isinstance(observation, dict) or set(observation) != {"type", "attempts_used", "correct", "failed", "valid"}:
        raise MemoryInputError("expected SubmitFeedback with five fields")
    if not isinstance(observation["type"], str) or type(observation["attempts_used"]) is not int or any(
            type(observation[k]) is not bool for k in ("correct", "failed", "valid")):
        raise MemoryInputError("invalid SubmitFeedback field types")
    return observation
'''
READ = '''

def read_submit_feedback(observation):
    """Check a submit feedback observation's fields and return it.

    Effect: read
    Input: observation
    """
    if not isinstance(observation, dict):
        raise MemoryInputError("expected a dict observation")
    if sorted(observation) != ["attempts_used", "correct", "failed", "type", "valid"]:
        raise MemoryInputError("expected SubmitFeedback with five fields")
    if not isinstance(observation["type"], str) or type(observation["attempts_used"]) is not int:
        raise MemoryInputError("invalid SubmitFeedback field types")
    for k in ("correct", "failed", "valid"):
        if type(observation[k]) is not bool:
            raise MemoryInputError("invalid SubmitFeedback field types")
    return observation
'''
ALIAS = '''

def read_submit_feedback(observation):
    """Check a submit feedback observation's fields and return it (alias of parse_submit_feedback).

    Effect: read
    Input: observation
    """
    return parse_submit_feedback(observation)
'''
STATE_DOC = '''

def submit_state(observation):
    """Classify consistent ARC submit feedback as solved, retry or exhausted.

    Effect: read
    Input: observation
    """
'''
STATE_TAIL = """    if not observation["valid"] or (observation["correct"] and observation["failed"]):
        raise MemoryInputError("inconsistent submit feedback flags")
    if observation["correct"]:
        return "solved"
    return "exhausted" if observation["failed"] else "retry"
"""
# the near-duplicate: submit_state repeats parse_submit_feedback's checks inline
STATE_INLINE = (
    STATE_DOC
    + """    if not isinstance(observation, dict) or set(observation) != {"type", "attempts_used", "correct", "failed", "valid"}:
        raise MemoryInputError("expected SubmitFeedback with five fields")
    if not isinstance(observation["type"], str) or type(observation["attempts_used"]) is not int or any(
            type(observation[k]) is not bool for k in ("correct", "failed", "valid")):
        raise MemoryInputError("invalid SubmitFeedback field types")
"""
    + STATE_TAIL
)
# the merge: one validation, called (the real ARC library's form)
STATE_MERGED = STATE_DOC + "    parse_submit_feedback(observation)\n" + STATE_TAIL
# the same behaviour in more lines: not a clean-up
STATE_LONGER = STATE_INLINE.replace(
    '    if observation["correct"]:\n        return "solved"\n',
    '    solved = observation["correct"]\n    if solved:\n        return "solved"\n',
)

PARENT_MODULE = HEAD + READ + STATE_INLINE


def _test(name, ok, bad):
    # imports only the function (MemoryInputError is a ValueError), so retiring it needs only that item deleted
    return (
        f"import pytest\nfrom env.dialogue_arc import {name}\n\n\n"
        f"def test_{name}_on_recorded_feedback():\n"
        + "".join(f"    assert {name}({obs!r}) == {want!r}\n" for obs, want in ok)
        + f"\n\ndef test_{name}_refuses_another_shape():\n"
        f"    with pytest.raises(ValueError):\n        {name}({bad!r})\n"
    )


PARENT_FILES = {
    MODULE: PARENT_MODULE,
    T_PSF: _test(
        "parse_submit_feedback",
        [(SOLVED, SOLVED), (RETRY, RETRY)],
        {"type": "SubmitFeedback"},
    ),
    T_RSF: _test("read_submit_feedback", [(RETRY2, RETRY2)], [RETRY2]),
    T_SS: _test(
        "submit_state",
        [(SOLVED, "solved"), (RETRY, "retry"), (EXHAUSTED, "exhausted")],
        {"type": "SubmitFeedback", "correct": True},
    ),
}
# recorded covers of the parent's items (a stand-in for the gated merges that built it)
RECORDED = {PSF: [0, 1], SS: [0, 1, 2], RSF: [3]}


def _item(iid, tests, covers):
    return {
        "item": iid,
        "kind": "env_function",
        "source_episodes": ["a1"],
        "tests": list(tests),
        "covers": [["a1", i] for i in covers],
        "input": "observation",
    }


def _dl(obs):
    return Action(
        0,
        "dialogue:arc",
        "reply",
        ["submit"],
        {},
        obs,
        "ok",
        kind="dialogue",
    )


@pytest.fixture
def arc(tmp_path):
    acts = [_dl(SOLVED), _dl(RETRY), _dl(EXHAUSTED), _dl(RETRY2)]
    ep = _ep(episode_id="a1", request=["Solve the ARC grid task"], actions=acts)
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ev.index_episode(ep, "1" * 40)

    def lookup(eid, i):
        return acts[i] if eid == "a1" and 0 <= i < len(acts) else None

    gate = Gate(mem, ev, BlobStore(tmp_path / "b"), action_lookup=lookup)
    parent = _merged(mem, PARENT_FILES)
    for item, idx in RECORDED.items():
        for i in idx:
            ev.add_cover(item, "a1", i)
    return mem, ev, gate, ep, parent


# --- a merge inside the normal pass ------------------------------------------------------------------------

SOL_READS_COVERS = "print(open('/inputs/library.json').read())"


def test_a_scripted_pass_merges_the_duplicated_validation_and_the_gate_admits_it(arc):
    mem, ev, gate, ep, parent = arc
    man = {
        "items": [_item(SS, [T_SS], RECORDED[SS])],
        "summary": "submit_state calls parse_submit_feedback",
    }
    script = Script(
        [
            SOL_READS_COVERS,
            _write(f"/memory/{MODULE}", HEAD + READ + STATE_MERGED),
            _write("/memory/.pass/manifest.json", json.dumps(man)),
        ],
    )
    sol = SolPass(
        mem,
        gate,
        ev,
        load=lambda eid: ep,
        model_turn=script,
        config=PassConfig(max_calls=10),
    )
    out = asyncio.run(
        sol.run(PassRequest("incremental", "dialogue:arc", ["a1"], False), "p1"),
    )
    assert "Tend the library too." in SOL_SYSTEM
    # Sol is shown the channel's functions with their recorded covers, and can read them
    user = script.first[1]["content"]
    assert (
        f"- {PSF}: 2 recorded covers\n- {RSF}: 1 recorded cover\n- {SS}: 3 recorded covers"
        in user
    )
    library = json.loads(script.outputs["c1"])["functions"]
    assert {row["item"]: row["cover_ids"] for row in library} == {
        item: [["a1", i] for i in RECORDED[item]] for item in sorted(RECORDED)
    }
    # an edit without a red test lands: the pass adds nothing, shrinks the module, and the old tests hold it
    assert out.passed, out.reasons
    assert mem.head() == out.commit


def test_a_behaviour_preserving_edit_that_does_not_shrink_still_needs_a_red_test(arc):
    mem, ev, gate, ep, parent = arc
    cand = _candidate(mem, {MODULE: HEAD + READ + STATE_LONGER})
    res = gate.check(parent, cand, {"items": [_item(SS, [T_SS], RECORDED[SS])]})
    assert not res.passed and "G3" in res.refused
    assert any(
        r.startswith("G3:") and "has no new or changed test" in r for r in res.reasons
    ), res.reasons


# --- aliases and deletions ---------------------------------------------------------------------------------


def test_an_alias_keeps_old_imports_working(arc):
    mem, ev, gate, ep, parent = arc
    # the near-duplicate becomes a thin alias: the parent's test of the old name still imports and passes
    alias = _candidate(mem, {MODULE: HEAD + ALIAS + STATE_INLINE})
    res = gate.check(parent, alias, {"items": [_item(RSF, [T_RSF], RECORDED[RSF])]})
    assert res.passed, res.reasons
    # dropping the name instead breaks that old import, and the gate refuses it
    dropped = _candidate(mem, {MODULE: HEAD + STATE_INLINE})
    man = {"items": [_item(PSF, [T_PSF], [0, 1, 3])], "deleted": [RSF]}
    res = gate.check(parent, dropped, man)
    assert not res.passed and "G3" in res.refused, res.reasons


def test_a_deletion_that_drops_coverage_is_refused(arc):
    mem, ev, gate, ep, parent = arc
    cand = _candidate(mem, {MODULE: HEAD + STATE_INLINE, T_RSF: None})
    res = gate.check(
        parent,
        cand,
        {"items": [], "deleted": [RSF], "deleted_tests": [T_RSF]},
    )
    assert not res.passed and res.refused == ["G5"], res.reasons
    lost = [r for r in res.reasons if r.startswith("G5:")]
    assert lost == [
        f"G5: deleted item {RSF} covered 1 recorded inputs that no remaining item covers "
        "(list them in a remaining item's covers): [['a1', 3]]",
    ]
    # value-free: no recorded value reaches the reason
    assert "SubmitFeedback" not in lost[0] and "attempts" not in lost[0]


def test_a_deletion_whose_covers_are_taken_over_passes(arc):
    mem, ev, gate, ep, parent = arc
    taken = f"{CH}/tests/test_parse_submit_feedback_more.py"
    cand = _candidate(
        mem,
        {
            MODULE: HEAD + STATE_INLINE,
            T_RSF: None,
            # a new test over the taken-over input: green on both sides, which a clean-up pass may add
            taken: _test("parse_submit_feedback", [(RETRY2, RETRY2)], [RETRY2]).replace(
                "def test_parse_submit_feedback_",
                "def test_more_parse_submit_feedback_",
            ),
        },
    )
    man = {
        "items": [_item(PSF, [T_PSF, taken], [0, 1, 3])],
        "deleted": [RSF],
        "deleted_tests": [T_RSF],
    }
    res = gate.merge(parent, cand, man, "p2", "incremental", "dialogue:arc", "0")
    assert res.passed, res.reasons
    assert (PSF, "a1", 3) in ev.covers()
