import asyncio
import json

import pytest

from tests.scripted_model import ScriptedModel, cell, reply, scripted
from unify.actor.code_act_actor import CodeActActor
from unify.function_manager.function_manager import FunctionManager

# ---------------------------------------------------------------------------
# can_store=True — deferred storage via post-completion review loop
# ---------------------------------------------------------------------------


_PARSE_AND_VALIDATE = (
    "def parse_and_validate_records(records):\n"
    '    """Split contact records into valid and invalid ones, with counts."""\n'
    "    valid, invalid = [], []\n"
    "    for index, entry in enumerate(records or []):\n"
    "        if isinstance(entry, dict) and '@' in str(entry.get('email', '')):\n"
    "            valid.append(entry)\n"
    "        else:\n"
    "            invalid.append(index)\n"
    "    return {'valid': valid, 'invalid': invalid,\n"
    "            'stats': {'total': len(valid) + len(invalid)}}\n"
)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_can_store_true_result_resolves_before_the_review_stores():
    """
    With can_store=True, result() resolves after the task phase while the
    storage review is still to come, and what the review stores lands in the
    store. The model is scripted (tests/scripted_model.py).

    The review's first turn waits until the test has the result, so a
    result() that waited for the review would time out here.
    """
    review_may_start = asyncio.Event()

    async def _store_once_released(call):
        await review_may_start.wait()
        return reply(
            calls=[
                cell(
                    "print(await functions.add("
                    f"implementations=[{_PARSE_AND_VALIDATE!r}]))",
                ),
            ],
        )

    model = ScriptedModel(
        actor=[
            reply(
                calls=[
                    cell(
                        _PARSE_AND_VALIDATE + "print(parse_and_validate_records("
                        "[{'email': 'a@b.c'}, {'email': 'nope'}]))",
                    ),
                ],
            ),
            reply("Done: 1 valid, 1 invalid."),
        ],
        review=[_store_once_released, reply("Stored parse_and_validate_records.")],
        allow={"gate"},
    )
    fm = FunctionManager(include_primitives=False)
    actor = CodeActActor(function_manager=fm, timeout=60)
    try:
        with scripted(model):
            handle = await actor.act(
                "Write parse_and_validate_records and test it on two records.",
                can_store=True,
                persist=False,
                clarification_enabled=False,
            )
            result = await asyncio.wait_for(handle.result(), timeout=60)
            assert result == "Done: 1 valid, 1 invalid."
            assert not handle.done(), "the storage review is still to run"
            assert "parse_and_validate_records" not in {
                f.get("name") for f in fm.filter_functions()
            }

            review_may_start.set()
            deadline = asyncio.get_event_loop().time() + 120
            while not handle.done():
                if asyncio.get_event_loop().time() > deadline:
                    raise TimeoutError("Storage loop did not complete in time")
                await asyncio.sleep(0.1)
    finally:
        try:
            await actor.close()
        except Exception:
            pass

    stored_names = {f.get("name", "") for f in fm.filter_functions()}
    assert "parse_and_validate_records" in stored_names
    assert [k for k in model.kinds() if k != "gate"] == [
        "actor",
        "actor",
        "review",
        "review",
    ]
    model.assert_used_up()


# ---------------------------------------------------------------------------
# can_store=True — reorganization (merge + delete)
# ---------------------------------------------------------------------------


_GREET = (
    "def greet(name, style='formal'):\n"
    '    """Greet *name* formally or casually."""\n'
    "    if style == 'casual':\n"
    "        return f'Hey {name}!'\n"
    "    return f'Good day, {name}.'\n"
)

# The review's merge: store the general function, then delete the narrow
# ones it subsumes, through the review's ``functions`` library.
_MERGE_CELL = (
    f"print(await functions.add(implementations=[{_GREET!r}]))\n"
    "for old in ('greet_formal', 'greet_casual'):\n"
    "    fid = (await functions.get(old))['function_id']\n"
    "    print(await functions.delete(fid))\n"
)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_can_store_true_merges_redundant_functions(monkeypatch):
    """
    The storage review can merge overlapping functions in the store: add a
    unified version and delete the old ones.

    Setup: the FunctionManager already contains two narrow greeting
    functions (greet_formal, greet_casual). The actor composes and executes
    a general-purpose `greet` function that subsumes both, and the scripted
    review stores it and deletes the now-redundant entries. Whether a live
    librarian sees the overlap is the model's judgement; this test is of
    what the review's merge does to the store.

    The review gate is answered "review" here, so the review it opens runs
    as shipped; this test is of the review's merge, not of the gate.

    result() resolves after the task phase; storage runs in the background.
    The test waits for done() to confirm the storage loop has completed.
    """
    from unify.actor import review_gate

    gate_asked: list[bool] = []

    async def _review(**_kwargs):
        gate_asked.append(True)
        return review_gate.GateDecision(
            review=True,
            reason="test: the merge review runs",
            decided=True,
        )

    monkeypatch.setattr(review_gate, "decide", _review)

    fm = FunctionManager(include_primitives=False)

    # Seed the store with two narrow, overlapping greeting functions.
    fm.add_functions(
        implementations=[
            'def greet_formal(name):\n    """Return a formal greeting."""\n    return f"Good day, {name}."',
            'def greet_casual(name):\n    """Return a casual greeting."""\n    return f"Hey {name}!"',
        ],
    )
    seeded = fm.filter_functions()
    seeded_ids = {f["function_id"] for f in seeded if isinstance(f, dict)}
    assert len(seeded_ids) == 2, f"Expected 2 seeded functions, got {len(seeded_ids)}"

    model = ScriptedModel(
        actor=[
            reply(calls=[cell(_GREET + "print(greet('Alice', 'formal'))")]),
            reply("Good day, Alice."),
        ],
        review=[
            reply(calls=[cell(_MERGE_CELL)]),
            reply("Merged greet_formal and greet_casual into greet."),
        ],
    )
    actor = CodeActActor(
        function_manager=fm,
        timeout=60,
    )
    try:
        with scripted(model):
            handle = await actor.act(
                "Write a general-purpose Python function called `greet` that takes "
                "`name` and `style` ('formal' or 'casual') parameters. "
                "For formal: return f'Good day, {name}.'; "
                "for casual: return f'Hey {name}!'. "
                "Execute it with name='Alice' and style='formal' to verify.",
                can_store=True,
                persist=False,
                clarification_enabled=False,
            )
            result = await asyncio.wait_for(handle.result(), timeout=120)
            assert result == "Good day, Alice."

            # Wait for storage to complete before asserting side effects.
            deadline = asyncio.get_event_loop().time() + 120
            while not handle.done():
                if asyncio.get_event_loop().time() > deadline:
                    raise TimeoutError("Storage loop did not complete in time")
                await asyncio.sleep(0.5)
    finally:
        try:
            await actor.close()
        except Exception:
            pass

    # The library held entries, so the gate was asked (and said review).
    assert gate_asked == [True]

    # The merged function was stored and the redundant ones deleted.
    final = fm.filter_functions()
    final_names = {f.get("name", "") for f in final if isinstance(f, dict)}
    assert final_names == {"greet"}, final_names
    final_ids = {f["function_id"] for f in final if isinstance(f, dict)}
    assert not seeded_ids & final_ids

    assert model.kinds() == ["actor", "actor", "review", "review"]
    # The review saw what its merge cell printed.
    (_first, last) = model.of("review")
    outputs = json.dumps([m for m in last.messages if m.get("role") == "tool"])
    assert "greet_formal" in outputs and "greet_casual" in outputs
    model.assert_used_up()


# ---------------------------------------------------------------------------
# Description type acceptance
# ---------------------------------------------------------------------------

# These tests stop the handle as soon as act() returns, so no turn is needed;
# whatever request starts first (the actor's, or the storage review after the
# stop) is answered in process rather than by a provider.
_STOPPED_AT_ONCE = {"actor", "review", "gate"}


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_code_act_accepts_dict_description():
    """
    CodeActActor.act should accept a dict description (passed to async tool loop).
    """
    actor = CodeActActor(
        timeout=30,
    )
    # The handle is stopped at once; a request that starts before the stop
    # is answered in process.
    with scripted(ScriptedModel(allow=_STOPPED_AT_ONCE)):
        try:
            # We just verify the call doesn't raise TypeError and creates a handle
            handle = await actor.act(
                {"role": "user", "content": "What is 2+2?"},
                persist=False,
                clarification_enabled=False,
            )
            # Verify we got a handle back (not testing the full loop completion)
            assert handle is not None
            await handle.stop()
        finally:
            try:
                await actor.close()
            except Exception:
                pass


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_code_act_accepts_list_description():
    """
    CodeActActor.act should accept a list description (passed to async tool loop).
    """
    actor = CodeActActor(
        timeout=30,
    )
    with scripted(ScriptedModel(allow=_STOPPED_AT_ONCE)):
        try:
            # We just verify the call doesn't raise TypeError and creates a handle
            handle = await actor.act(
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "What is 2+2?"},
                ],
                persist=False,
                clarification_enabled=False,
            )
            # Verify we got a handle back
            assert handle is not None
            await handle.stop()
        finally:
            try:
                await actor.close()
            except Exception:
                pass
