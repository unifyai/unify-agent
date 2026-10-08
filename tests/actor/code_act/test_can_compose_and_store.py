import asyncio

import pytest

from unify.actor.code_act_actor import CodeActActor
from unify.function_manager.function_manager import FunctionManager

# ---------------------------------------------------------------------------
# can_store=True — deferred storage via post-completion review loop
# ---------------------------------------------------------------------------


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(300)
async def test_can_store_true_defers_storage_to_review_loop():
    """
    When can_store=True, the CodeActActor should compose and execute code,
    then run a post-completion storage-review loop that examines the
    trajectory and stores reusable functions via FunctionManager_add_functions.

    result() resolves after the task phase; storage runs in the background.
    The test waits for done() to confirm the storage loop has completed
    before asserting storage side effects.

    The function must be complex enough that the librarian LLM consistently
    judges it as worth storing (non-trivial logic, validation, edge cases).
    """
    fm = FunctionManager(include_primitives=False)

    actor = CodeActActor(
        function_manager=fm,
        timeout=60,
    )
    try:
        handle = await actor.act(
            "Write a reusable Python function called `parse_and_validate_records` that:\n"
            "1. Takes a list of dicts, each with optional keys: name, email, phone, company\n"
            "2. Validates each entry: name must be non-empty string, email must contain '@',\n"
            "   phone (if present) must be digits/dashes/spaces only and at least 7 chars\n"
            "3. Returns a dict with keys:\n"
            "   - 'valid': list of cleaned entries (strip whitespace, normalize phone to digits-only)\n"
            "   - 'invalid': list of (index, entry, errors) tuples describing validation failures\n"
            "   - 'stats': dict with counts of total, valid, invalid, and entries_with_company\n"
            "4. Handle edge cases: None inputs, empty lists, entries that are not dicts\n\n"
            "Test it with a mixed list of 5 entries including at least 2 invalid ones "
            "and verify the stats are correct.",
            can_store=True,
            persist=False,
            clarification_enabled=False,
        )
        result = await asyncio.wait_for(handle.result(), timeout=120)
        assert result is not None

        # result() resolves after the task phase.  Wait for the storage
        # review loop to finish before asserting storage side effects.
        deadline = asyncio.get_event_loop().time() + 120
        while not handle.done():
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError("Storage loop did not complete in time")
            await asyncio.sleep(0.5)

        stored = fm.filter_functions()
        assert stored, (
            "Expected FunctionManager to contain at least one stored function "
            "after the storage review loop."
        )
        stored_names = {f.get("name", "") for f in stored if isinstance(f, dict)}
        assert "parse_and_validate_records" in stored_names, (
            f"Expected 'parse_and_validate_records' in stored functions, "
            f"got: {stored_names}"
        )
    finally:
        try:
            await actor.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# can_store=True — reorganization (merge + delete)
# ---------------------------------------------------------------------------


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(300)
async def test_can_store_true_merges_redundant_functions(monkeypatch):
    """
    The storage review loop should recognise overlapping functions in the
    store and merge them: add a unified version and delete the old ones.

    Setup: the FunctionManager already contains two narrow greeting
    functions (greet_formal, greet_casual). The actor composes and
    executes a general-purpose `greet` function that subsumes both.
    The storage review should detect the overlap, store the merged
    version, and delete the now-redundant entries.

    The review gate is answered "review" here, so the review it opens runs
    as shipped. The gate itself would skip it when the session already
    stored ``greet`` (a known limitation of the frozen review; FREEZE-TODO),
    and this test is of the review's merge, not of the gate.

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

    actor = CodeActActor(
        function_manager=fm,
        timeout=60,
    )
    try:
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
        assert result is not None

        # Wait for storage to complete before asserting side effects.
        deadline = asyncio.get_event_loop().time() + 120
        while not handle.done():
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError("Storage loop did not complete in time")
            await asyncio.sleep(0.5)

        # The library held entries, so the gate was asked (and said review).
        assert gate_asked == [True]

        # The merged function should have been stored.
        final = fm.filter_functions()
        final_names = {f.get("name", "") for f in final if isinstance(f, dict)}
        assert (
            "greet" in final_names
        ), f"Expected a unified 'greet' function in the store, got: {final_names}"

        # At least one of the old redundant functions should have been deleted.
        final_ids = {f["function_id"] for f in final if isinstance(f, dict)}
        deleted = seeded_ids - final_ids
        assert deleted, (
            f"Expected at least one of the seeded functions ({seeded_ids}) to be "
            f"deleted after merge, but all remain: {final_ids}"
        )
    finally:
        try:
            await actor.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Description type acceptance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(300)
async def test_code_act_accepts_dict_description():
    """
    CodeActActor.act should accept a dict description (passed to async tool loop).
    """
    actor = CodeActActor(
        timeout=30,
    )
    try:
        # We just verify the call doesn't raise TypeError and creates a handle
        # The handle will run an LLM loop, but we stop it immediately
        handle = await actor.act(
            {"role": "user", "content": "What is 2+2?"},
            persist=False,
            clarification_enabled=False,
        )
        # Verify we got a handle back (not testing the full loop completion)
        assert handle is not None
        # Stop the handle to avoid waiting for LLM
        await handle.stop()
    finally:
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(300)
async def test_code_act_accepts_list_description():
    """
    CodeActActor.act should accept a list description (passed to async tool loop).
    """
    actor = CodeActActor(
        timeout=30,
    )
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
        # Stop the handle to avoid waiting for LLM
        await handle.stop()
    finally:
        try:
            await actor.close()
        except Exception:
            pass
