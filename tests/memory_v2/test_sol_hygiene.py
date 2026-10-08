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
import unify.memory_v2.gate as gate_module
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


# --- the clean-up exemption holds only for a change that keeps behaviour (review batch B) ------------------

# the merge, but the retry with three attempts (a recorded input no old test uses) now reads as exhausted
STATE_TWISTED = STATE_MERGED.replace(
    'return "exhausted" if observation["failed"] else "retry"',
    'return "exhausted" if observation["failed"] or observation["attempts_used"] == 3 else "retry"',
)
# the same code as the parent's submit_state, one tuple reordered: no less code
STATE_REORDERED = STATE_INLINE.replace(
    'for k in ("correct", "failed", "valid")',
    'for k in ("valid", "correct", "failed")',
)

# parse_submit_feedback with less code, refusing the same shapes
PSF_SHORT = HEAD.replace(
    'raise MemoryInputError("invalid SubmitFeedback field types")',
    "raise MemoryInputError",
)
# parse_submit_feedback changing its result on the retry with three attempts only
PSF_TWISTED = HEAD.replace(
    "    return observation\n",
    '    return dict(observation, valid=observation["attempts_used"] != 3)\n',
)


def _refused_reason(item, why):
    return (
        f"G3: {item} is edited in a clean-up pass without a red test, and {why}; a change of "
        "behaviour needs a test that is red on the parent's library"
    )


def test_a_shrinking_pass_that_changes_a_result_on_a_recorded_cover_is_refused(arc):
    mem, ev, gate, ep, parent = arc
    # an earlier pass recorded submit_state on cover 3; this pass lists only covers 0-2, which its old test uses
    ev.add_cover(SS, "a1", 3)
    man = {"items": [_item(SS, [T_SS], RECORDED[SS])]}
    twisted = _candidate(mem, {MODULE: HEAD + READ + STATE_TWISTED})
    res = gate.check(parent, twisted, man)
    assert not res.passed and res.refused == ["G3"], res.reasons
    refusal = [r for r in res.reasons if r.startswith("G3:")]
    assert refusal == [
        _refused_reason(
            SS,
            "its results differ from the parent's on 1 recorded covers: [['a1', 3]]",
        ),
    ], res.reasons
    assert "retry" not in refusal[0] and "exhausted" not in refusal[0]
    # the pure merge does the same as the parent on all four recorded covers, and lands without a red test
    merged = _candidate(mem, {MODULE: HEAD + READ + STATE_MERGED})
    res = gate.check(parent, merged, man)
    assert res.passed, res.reasons


def test_deleting_comments_does_not_make_a_clean_up_pass(arc):
    mem, ev, gate, ep, parent = arc
    commented = _merged(
        mem,
        {
            MODULE: "# Parsers for ARC dialogue observations.\n# Shape checks only.\n"
            + PARENT_MODULE,
        },
    )
    # fewer lines (the comments are gone) but no less code: an edited function still needs a red test
    cand = _candidate(mem, {MODULE: HEAD + READ + STATE_REORDERED})
    res = gate.check(commented, cand, {"items": [_item(SS, [T_SS], RECORDED[SS])]})
    assert not res.passed and res.refused == ["G3"], res.reasons
    assert f"G3: {SS} has no new or changed test" in res.reasons, res.reasons


def test_a_parent_test_that_only_imports_the_function_does_not_hold_it(arc):
    mem, ev, gate, ep, parent = arc
    # submit_state's old test file imports it but calls only parse_submit_feedback (the review's example)
    imports_only = _test(
        "parse_submit_feedback",
        [(SOLVED, SOLVED)],
        {"type": "SubmitFeedback"},
    ).replace(
        "from env.dialogue_arc import parse_submit_feedback\n",
        "from env.dialogue_arc import parse_submit_feedback, submit_state\n",
    )
    base = _merged(mem, {T_SS: imports_only})
    man = {"items": [_item(SS, [T_SS], RECORDED[SS])]}
    res = gate.check(base, _candidate(mem, {MODULE: HEAD + READ + STATE_MERGED}), man)
    assert not res.passed and res.refused == ["G3"], res.reasons
    assert (
        _refused_reason(SS, "no parent test that passed on the parent calls it")
        in res.reasons
    ), res.reasons
    # a call through the module counts
    calls = imports_only + (
        "\n\ndef test_state_through_the_module():\n"
        "    import env.dialogue_arc as arc\n\n"
        f"    assert arc.submit_state({SOLVED!r}) == 'solved'\n"
    )
    base = _merged(mem, {T_SS: calls})
    res = gate.check(base, _candidate(mem, {MODULE: HEAD + READ + STATE_MERGED}), man)
    assert res.passed, res.reasons


def test_a_kept_function_takes_over_a_deleted_ones_covers_only_if_its_behaviour_is_kept(
    arc,
):
    mem, ev, gate, ep, parent = arc
    # an earlier pass recorded parse_submit_feedback on cover 3, which read_submit_feedback also covers
    ev.add_cover(PSF, "a1", 3)
    man = {
        "items": [_item(PSF, [T_PSF], RECORDED[PSF])],
        "deleted": [RSF],
        "deleted_tests": [T_RSF],
    }
    # parse_submit_feedback is edited in the same pass and now answers cover 3 differently: its recorded
    # cover no longer vouches for the deleted function's input (I1)
    twisted = _candidate(mem, {MODULE: PSF_TWISTED + STATE_INLINE, T_RSF: None})
    res = gate.check(parent, twisted, man)
    assert not res.passed and res.refused == ["G3", "G5"], res.reasons
    assert (
        _refused_reason(
            PSF,
            "its results differ from the parent's on 1 recorded covers: [['a1', 3]]",
        )
        in res.reasons
    ), res.reasons
    assert [r for r in res.reasons if r.startswith("G5:")] == [
        f"G5: deleted item {RSF} covered 1 recorded inputs that no remaining item covers "
        "(list them in a remaining item's covers): [['a1', 3]]",
    ]
    # edited the same way it keeps behaviour: its recorded cover takes the deleted function's over
    short = _candidate(mem, {MODULE: PSF_SHORT + STATE_INLINE, T_RSF: None})
    res = gate.check(parent, short, man)
    assert res.passed, res.reasons


# --- the clean-up exemption fails closed (re-review of the C1 fix) -----------------------------------------


def test_a_result_that_is_not_json_voids_the_clean_up_exemption(arc):
    mem, ev, gate, ep, parent = arc
    as_set = STATE_MERGED.replace('return "solved"', 'return {"solved"}')
    cand = _candidate(mem, {MODULE: HEAD + READ + as_set})
    res = gate.check(parent, cand, {"items": [_item(SS, [T_SS], RECORDED[SS])]})
    assert not res.passed and res.refused == ["G3"], res.reasons
    assert (
        _refused_reason(
            SS,
            "its results on 1 recorded covers cannot be compared (not run in time, timed out or not "
            "JSON): [['a1', 0]]",
        )
        in res.reasons
    ), res.reasons


def test_a_cover_that_times_out_voids_the_clean_up_exemption(arc):
    mem, ev, gate, ep, parent = arc
    slow = STATE_MERGED.replace(
        "    parse_submit_feedback(observation)\n",
        "    parse_submit_feedback(observation)\n"
        '    if observation["attempts_used"] == 2:\n'
        "        import time\n\n"
        "        time.sleep(3)\n",
    )
    cand = _candidate(mem, {MODULE: HEAD + READ + slow})
    res = gate.check(parent, cand, {"items": [_item(SS, [T_SS], RECORDED[SS])]})
    assert not res.passed and "G3" in res.refused, res.reasons
    assert (
        _refused_reason(
            SS,
            "its results on 1 recorded covers cannot be compared (not run in time, timed out or not "
            "JSON): [['a1', 1]]",
        )
        in res.reasons
    ), res.reasons


def test_more_than_max_output_covers_voids_the_clean_up_exemption(arc, monkeypatch):
    mem, ev, gate, ep, parent = arc
    monkeypatch.setattr(gate_module, "MAX_OUTPUT_COVERS", 2)
    cand = _candidate(mem, {MODULE: HEAD + READ + STATE_MERGED})
    res = gate.check(parent, cand, {"items": [_item(SS, [T_SS], RECORDED[SS])]})
    assert not res.passed and res.refused == ["G3"], res.reasons
    assert [r for r in res.reasons if r.startswith("G3:")] == [
        _refused_reason(SS, "its 3 recorded covers exceed the 2 compared"),
    ], res.reasons


# --- D28: any change in what a stored function returns on recorded inputs needs a red test ------------------


def _changed_reason(item, why):
    return (
        f"G3: {item} changes behaviour without a test: {why}; any change in what a stored function returns on "
        "recorded inputs needs a test, listed under it in the manifest, that fails on the parent's library "
        "and passes on the candidate's"
    )


def _all_listed(**tests):
    """A skeleton change of the channel: every public function listed, with its parent covers."""
    return {
        "items": [
            _item(PSF, tests.get("psf", [T_PSF]), RECORDED[PSF]),
            _item(RSF, tests.get("rsf", [T_RSF]), RECORDED[RSF]),
            _item(SS, tests.get("ss", [T_SS]), RECORDED[SS]),
        ],
        "skeleton": [CH],
    }


def test_an_assignment_alias_of_a_deleted_function_is_compared_with_it(arc):
    mem, ev, gate, ep, parent = arc
    man = {
        "items": [
            _item(PSF, [T_PSF], [0, 1, 3]),
            _item(SS, [T_SS], RECORDED[SS]),
        ],
        "skeleton": [CH],
        "deleted": [RSF],
        "deleted_tests": [T_RSF],
    }
    # the old name now answers with another function's behaviour
    wrong = HEAD + STATE_INLINE + "\n\nread_submit_feedback = submit_state\n"
    res = gate.check(parent, _candidate(mem, {MODULE: wrong, T_RSF: None}), man)
    assert not res.passed and "G3" in res.refused, res.reasons
    refusal = [r for r in res.reasons if r.startswith(f"G3: {RSF} ")]
    assert len(refusal) == 1, res.reasons
    assert refusal[0].startswith(
        _changed_reason(
            RSF,
            "its results differ from the parent's on 1 recorded covers: [['a1', 3]]",
        ).split("; any change")[0],
    ), refusal
    assert "SubmitFeedback" not in refusal[0] and "retry" not in refusal[0]
    # the same alias of the function that does what it did lands with no new test
    right = HEAD + STATE_INLINE + "\n\nread_submit_feedback = parse_submit_feedback\n"
    res = gate.check(parent, _candidate(mem, {MODULE: right, T_RSF: None}), man)
    assert res.passed, res.reasons


def test_an_unchanged_caller_of_an_edited_function_is_compared(arc):
    mem, ev, gate, ep, _ = arc
    # an earlier merge left read_submit_feedback as a def alias calling parse_submit_feedback; its old test
    # does not use its recorded cover, so only the behaviour check can see a change there
    base = _merged(
        mem,
        {
            MODULE: HEAD + ALIAS + STATE_INLINE,
            T_RSF: _test("read_submit_feedback", [(SOLVED, SOLVED)], [RETRY2]),
        },
    )
    # a clean-up merges submit_state and twists parse_submit_feedback on the alias's recorded cover only
    cand = _candidate(mem, {MODULE: PSF_TWISTED + ALIAS + STATE_MERGED})
    man = {
        "items": [
            _item(PSF, [T_PSF], RECORDED[PSF]),
            _item(SS, [T_SS], RECORDED[SS]),
        ],
    }
    res = gate.check(base, cand, man)
    assert not res.passed and res.refused == ["G3"], res.reasons
    assert (
        _changed_reason(
            RSF,
            "its results differ from the parent's on 1 recorded covers: [['a1', 3]]",
        )
        in res.reasons
    ), res.reasons
    # the edited function itself is seen on the pass's episode action it now answers differently
    assert (
        _refused_reason(
            PSF,
            "its results differ from the parent's on 1 recorded actions of this pass's episodes: "
            "[['a1', 3]]",
        )
        in res.reasons
    ), res.reasons


# submit_state's ending in a private helper with a module constant
STATE_HELPED = STATE_INLINE.replace(
    '    return "exhausted" if observation["failed"] else "retry"\n',
    "    return _ending(observation)\n",
)
HELPER = """

_LAST_ATTEMPT = 8


def _ending(observation):
    return "exhausted" if observation["failed"] or observation["attempts_used"] >= _LAST_ATTEMPT else "retry"
"""
T_LAST = f"{CH}/tests/test_submit_state_last_attempt.py"


def _helped(mem, ev):
    # an earlier pass recorded submit_state on the retry with three attempts, which no old test uses
    ev.add_cover(SS, "a1", 3)
    return _merged(mem, {MODULE: HEAD + READ + HELPER + STATE_HELPED})


def test_a_skeleton_helper_change_needs_a_red_test(arc):
    mem, ev, gate, ep, _ = arc
    base = _helped(mem, ev)
    helper = HELPER.replace(
        '"exhausted" if observation["failed"] or observation["attempts_used"] >= _LAST_ATTEMPT else "retry"',
        '"exhausted" if observation["failed"] or observation["attempts_used"] == 3 else "retry"',
    )
    cand = _candidate(mem, {MODULE: HEAD + READ + helper + STATE_HELPED})
    res = gate.check(base, cand, _all_listed())
    assert not res.passed and res.refused == ["G3"], res.reasons
    assert [r for r in res.reasons if r.startswith("G3:")] == [
        _changed_reason(
            SS,
            "its results differ from the parent's on 1 recorded covers: [['a1', 3]]",
        ),
    ], res.reasons


def test_a_helper_constant_change_needs_a_red_test_and_lands_with_one(arc):
    mem, ev, gate, ep, _ = arc
    base = _helped(mem, ev)
    module = (
        HEAD
        + READ
        + HELPER.replace("_LAST_ATTEMPT = 8", "_LAST_ATTEMPT = 3")
        + STATE_HELPED
    )
    cand = _candidate(mem, {MODULE: module})
    res = gate.check(base, cand, _all_listed())
    assert not res.passed and res.refused == ["G3"], res.reasons
    assert (
        _changed_reason(
            SS,
            "its results differ from the parent's on 1 recorded covers: [['a1', 3]]",
        )
        in res.reasons
    ), res.reasons
    # the same change with a test that fails on the parent's library and passes on the candidate's
    red = (
        "from env.dialogue_arc import submit_state\n\n\n"
        "def test_the_third_attempt_is_the_last():\n"
        f"    assert submit_state({RETRY2!r}) == 'exhausted'\n"
    )
    cand = _candidate(mem, {MODULE: module, T_LAST: red})
    res = gate.check(base, cand, _all_listed(ss=[T_SS, T_LAST]))
    assert res.passed, res.reasons


def test_a_pure_refactor_of_a_helper_needs_no_new_test(arc):
    mem, ev, gate, ep, _ = arc
    base = _helped(mem, ev)
    # the constant inlined: a skeleton change with the same results everywhere
    refactor = HELPER.replace("\n_LAST_ATTEMPT = 8\n", "").replace(
        ">= _LAST_ATTEMPT",
        ">= 8",
    )
    cand = _candidate(mem, {MODULE: HEAD + READ + refactor + STATE_HELPED})
    res = gate.check(base, cand, _all_listed())
    assert res.passed, res.reasons


# submit_state refusing the retry with three attempts, which the environment accepted
STATE_STRICT = STATE_INLINE.replace(
    '    if observation["correct"]:\n',
    '    if observation["attempts_used"] == 3:\n'
    '        raise MemoryInputError("unexpected attempt count")\n'
    '    if observation["correct"]:\n',
)
T_THIRD = f"{CH}/tests/test_submit_state_third_attempt.py"


def test_a_clean_up_repair_off_the_recorded_covers_needs_a_red_test(arc):
    mem, ev, gate, ep, _ = arc
    base = _merged(mem, {MODULE: HEAD + READ + STATE_STRICT})
    # the repair deletes the refusing check: less code, the same results on every recorded cover
    cand = _candidate(mem, {MODULE: HEAD + READ + STATE_INLINE})
    man = {"items": [_item(SS, [T_SS], RECORDED[SS])]}
    res = gate.check(base, cand, man)
    assert not res.passed and res.refused == ["G3"], res.reasons
    assert [r for r in res.reasons if r.startswith("G3:")] == [
        _refused_reason(
            SS,
            "its results differ from the parent's on 1 recorded actions of this pass's episodes: "
            "[['a1', 3]]",
        ),
    ], res.reasons
    # with a test that fails on the parent's library the repair lands
    red = (
        "from env.dialogue_arc import submit_state\n\n\n"
        "def test_the_third_attempt_is_a_retry():\n"
        f"    assert submit_state({RETRY2!r}) == 'retry'\n"
    )
    cand = _candidate(mem, {MODULE: HEAD + READ + STATE_INLINE, T_THIRD: red})
    res = gate.check(base, cand, {"items": [_item(SS, [T_SS, T_THIRD], RECORDED[SS])]})
    assert res.passed, res.reasons


def test_a_red_tested_kept_function_does_not_vouch_for_deleted_covers_unless_listed(
    arc,
):
    mem, ev, gate, ep, parent = arc
    # an earlier pass recorded parse_submit_feedback on cover 3, which read_submit_feedback also covers
    ev.add_cover(PSF, "a1", 3)
    t_new = f"{CH}/tests/test_parse_submit_feedback_third.py"
    narrowed = HEAD.replace(
        "    return observation\n",
        '    if observation["attempts_used"] == 3:\n'
        '        raise MemoryInputError("unexpected attempt count")\n'
        "    return observation\n",
    )
    refuses = (
        "import pytest\nfrom env.dialogue_arc import parse_submit_feedback\n\n\n"
        "def test_the_third_attempt_is_refused():\n"
        "    with pytest.raises(ValueError):\n"
        f"        parse_submit_feedback({RETRY2!r})\n"
    )
    deleting = {"deleted": [RSF], "deleted_tests": [T_RSF]}
    # a red test vets the narrowing, but the narrowed function's old cover 3 no longer vouches for the
    # deleted function's input
    cand = _candidate(
        mem,
        {MODULE: narrowed + STATE_INLINE, T_RSF: None, t_new: refuses},
    )
    man = {"items": [_item(PSF, [T_PSF, t_new], RECORDED[PSF])], **deleting}
    res = gate.check(parent, cand, man)
    assert not res.passed and res.refused == ["G5"], res.reasons
    assert [r for r in res.reasons if r.startswith("G5:")] == [
        f"G5: deleted item {RSF} covered 1 recorded inputs that no remaining item covers "
        "(list them in a remaining item's covers): [['a1', 3]]",
    ]
    # a red-tested change that lists the cover takes it over
    changes = (
        "from env.dialogue_arc import parse_submit_feedback\n\n\n"
        "def test_the_third_attempt_reads_as_invalid():\n"
        f"    assert parse_submit_feedback({RETRY2!r})['valid'] is False\n"
    )
    cand = _candidate(
        mem,
        {MODULE: PSF_TWISTED + STATE_INLINE, T_RSF: None, t_new: changes},
    )
    man = {"items": [_item(PSF, [T_PSF, t_new], [0, 1, 3])], **deleting}
    res = gate.check(parent, cand, man)
    assert res.passed, res.reasons


def test_an_exhausted_behaviour_budget_refuses_the_pass(arc, monkeypatch):
    mem, ev, gate, ep, parent = arc
    cand = _candidate(mem, {MODULE: HEAD + READ + STATE_MERGED})
    man = {"items": [_item(SS, [T_SS], RECORDED[SS])]}
    monkeypatch.setattr(gate_module, "BEHAVIOUR_MAX_CASES", 3)
    res = gate.check(parent, cand, man)
    assert not res.passed and res.refused == ["G3"], res.reasons
    assert any(
        r.startswith("G3: behaviour check not completed:")
        and "exceed the 3 compared" in r
        for r in res.reasons
    ), res.reasons
    monkeypatch.setattr(gate_module, "BEHAVIOUR_MAX_CASES", 2000)
    monkeypatch.setattr(gate_module, "BEHAVIOUR_BUDGET_S", 0.0)
    res = gate.check(parent, cand, man)
    assert not res.passed and res.refused == ["G3"], res.reasons
    assert any(
        r.startswith("G3: behaviour check not completed:") and "budget ran out" in r
        for r in res.reasons
    ), res.reasons
