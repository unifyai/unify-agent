"""D28's ``Gate._behaviour_owner``: which manifest item a behaviour change found by G3's check belongs to.

A function of a changed channel that does something else on its recorded inputs, with no red test, is
refused under G3. The failure is item-scoped (only that item is refused, and per-item admission can land
the rest) exactly when the pass's one change to that channel's module is a single edited environment
function listed in ``items``; otherwise (a skeleton change, a deleted or unlisted item in the channel, two
or no edited functions there, or an unreadable parent library) it belongs to the whole pass. Keyless and
offline: the run state is built directly, nothing runs confined.
"""

from __future__ import annotations

import pytest

from unify.memory_v2.gate import CHECKS, Gate, GateResult, _Probe, _Run
from unify.memory_v2.manifest import Manifest, ManifestItem

VENMO = "env/venmo"
CHANGED = f"{VENMO}:balance"  # the function whose behaviour changed (unchanged itself)


def _item(item: str, kind: str = "env_function") -> ManifestItem:
    channel = item.split(":", 1)[0]
    return ManifestItem(
        item=item,
        kind=kind,
        path=f"{channel}/__init__.py",
        source_episodes=["e1"],
        tests=[],
        covers=[],
    )


def _run(
    tmp_path,
    edited: list[str] = (),
    *,
    unchanged: list[str] = (),
    skeleton: list[str] = (),
    deleted: list[str] = (),
    unlisted: list[str] = (),
) -> _Run:
    """A run whose manifest lists *edited* (bodies differ from the parent's) and *unchanged* (same bodies)."""
    man = Manifest(
        items=[_item(i) for i in (*edited, *unchanged)],
        deleted=list(deleted),
        unlisted=list(unlisted),
        skeleton=list(skeleton),
    )
    run = _Run(
        GateResult(True, {c: True for c in CHECKS}),
        man,
        {},
        "p" * 40,
        "c" * 40,
        tmp_path,
    )
    for item in edited:
        run.p_bodies[item] = ("parent-digest", "def", True)
        run.c_bodies[item] = ("candidate-digest", "def", True)
    for item in unchanged:
        run.p_bodies[item] = run.c_bodies[item] = ("same-digest", "def", True)
    return run


def _probe(why: str | None = None) -> _Probe:
    return _Probe(item=CHANGED, strict=False, why=why, differ=[("e1", 0)])


def _refuse(run: _Run, pr: _Probe) -> str | None:
    """G3's refusal of CHANGED as :meth:`Gate._behaviour` records it; returns the owner it was given."""
    owner = Gate._behaviour_owner(run, CHANGED, pr)
    run.fail("G3", f"{CHANGED} changes behaviour without a test", owner)
    return owner


def test_one_edited_function_in_the_channel_owns_the_change(tmp_path):
    """An unchanged caller differs because the one function edited in its channel did: item-scoped."""
    run = _run(tmp_path, [f"{VENMO}:me"], unchanged=[f"{VENMO}:friends"])
    owner = _refuse(run, _probe())
    assert owner == f"{VENMO}:me"
    assert run.item_fail == {f"{VENMO}:me": ["G3"]} and not run.pass_wide


def test_an_edit_in_another_channel_is_not_the_owner(tmp_path):
    """Only edits of the changed function's own channel count, so one edit elsewhere still owns nothing
    there, and one edit in each channel gives each its own owner."""
    run = _run(tmp_path, ["env/zelle:send"])
    assert Gate._behaviour_owner(run, CHANGED, _probe()) is None
    run = _run(tmp_path, ["env/zelle:send", f"{VENMO}:me"])
    assert Gate._behaviour_owner(run, CHANGED, _probe()) == f"{VENMO}:me"
    assert Gate._behaviour_owner(run, "env/zelle:limits", _probe()) == "env/zelle:send"


@pytest.mark.parametrize(
    "case",
    [
        "two edited functions",
        "no edited function",
        "listed but unchanged",
        "skeleton change",
        "deleted item",
        "unlisted item",
        "parent unreadable",
    ],
)
def test_otherwise_the_change_refuses_the_whole_pass(tmp_path, case):
    me, other = f"{VENMO}:me", f"{VENMO}:friends"
    why = None
    if case == "two edited functions":  # which of the two changed it is not known
        run = _run(tmp_path, [me, other])
    elif case == "no edited function":  # a private helper or constant changed
        run = _run(tmp_path)
    elif case == "listed but unchanged":  # in items, but its body is the parent's
        run = _run(tmp_path, unchanged=[me])
    elif case == "skeleton change":  # module-level code changed besides the edit
        run = _run(tmp_path, [me], skeleton=[VENMO])
    elif case == "deleted item":
        run = _run(tmp_path, [me], deleted=[f"{VENMO}:old"])
    elif case == "unlisted item":
        run = _run(tmp_path, [me], unlisted=[f"{VENMO}:rare"])
    else:
        run = _run(tmp_path, [me])
        why = "the parent's library cannot be read"
    owner = _refuse(run, _probe(why))
    assert owner is None
    assert run.pass_wide and run.item_fail == {}


def test_another_channels_skeleton_or_deletion_leaves_the_attribution_alone(tmp_path):
    run = _run(
        tmp_path,
        [f"{VENMO}:me"],
        skeleton=["env/zelle"],
        deleted=["env/zelle:old"],
        unlisted=["env/zelle:rare"],
    )
    assert Gate._behaviour_owner(run, CHANGED, _probe()) == f"{VENMO}:me"


def test_a_channel_named_as_a_prefix_of_another_is_not_the_same_channel(tmp_path):
    """``env/venmo2`` is not ``env/venmo``: neither its edits nor its deletions count for venmo."""
    run = _run(tmp_path, [f"{VENMO}:me", "env/venmo2:me"], deleted=["env/venmo2:old"])
    assert Gate._behaviour_owner(run, CHANGED, _probe()) == f"{VENMO}:me"
