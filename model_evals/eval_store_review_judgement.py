"""An eval of a live model (moved from tests/actor/code_act/test_can_compose_and_store.py): whether the storage review judges a function worth
storing. It needs a real provider key and is run by hand (model_evals/README.md)."""

import asyncio
import pytest
from tests.scripted_model import ScriptedModel, cell, reply, scripted
from unify.actor.code_act_actor import CodeActActor
from unify.function_manager.function_manager import FunctionManager


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
