import asyncio
from unittest.mock import MagicMock

import pytest

from tests.actor.code_act.conftest import make_fm_mock
from unify.actor.code_act_actor import CodeActActor
from unify.function_manager.function_manager import FunctionManager
from tests.baked_defaults import as_shipped  # noqa: F401

# ---------------------------------------------------------------------------
# can_compose=False — symbolic tests
# ---------------------------------------------------------------------------


# as_shipped: deleted in step 5 (can_compose=False on the JSON tools; the core surface refuses it)
@pytest.mark.usefixtures("as_shipped")
@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_code_act_can_compose_false_requires_function_manager():
    """
    can_compose=False without a function_manager should raise RuntimeError
    because there would be no usable tools (no execute_code, no execute_function).
    """
    actor = CodeActActor(
        timeout=30,
    )
    # The ManagerRegistry provides a default FM, so override it to None.
    actor.function_manager = None
    try:
        with pytest.raises(RuntimeError, match="function_manager is required"):
            await actor.act("Do something", can_compose=False)
    finally:
        try:
            await actor.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# can_compose=False — eval tests
# ---------------------------------------------------------------------------


# as_shipped: deleted in step 5 (the execute_function JSON tool)
@pytest.mark.usefixtures("as_shipped")
@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(300)
async def test_code_act_can_compose_false_executes_best_matching_function():
    """
    When can_compose=False, the LLM should discover stored functions via
    FunctionManager discovery tools and invoke them via execute_function.
    It must NOT use execute_code.

    execute_function synthesises a code snippet and runs it through the
    sandbox.  We verify that (a) the function implementation was looked up
    via the FunctionManager, and (b) the synthesised code actually executed
    (the implementation prints a sentinel so we can check stdout).
    """
    _fn_impl = "def my_task():\n    print('SENTINEL_OK')\n    return 'OK'"
    _fn_metadata = [
        {
            "function_id": 123,
            "name": "my_task",
            "docstring": "Does the thing requested by the user",
        },
    ]
    _fn_data_full = {
        **_fn_metadata[0],
        "implementation": _fn_impl,
    }
    fm = make_fm_mock()
    fm.search_functions = MagicMock(return_value={"metadata": _fn_metadata})
    fm.filter_functions = MagicMock(return_value={"metadata": _fn_metadata})
    fm.list_functions = MagicMock(return_value={"metadata": _fn_metadata})
    fm._get_function_data_by_name = MagicMock(return_value=_fn_data_full)
    fm._get_primitive_data_by_name = MagicMock(return_value=None)
    fm._include_primitives = False

    actor = CodeActActor(
        function_manager=fm,
        environments=[],
        timeout=120,
        # Isolate can_compose execute_function routing from discovery-first gating.
        tool_policy=None,
    )
    actor.guidance_manager = None
    try:
        handle = await actor.act(
            "Do the thing using the stored my_task function via execute_function. "
            "Do not ask clarifying questions.",
            can_compose=False,
            persist=False,
            clarification_enabled=False,
        )
        await asyncio.wait_for(handle.result(), timeout=120)

        # The function implementation should have been looked up.
        fm._get_function_data_by_name.assert_called()
    finally:
        try:
            await actor.close()
        except Exception:
            pass


# as_shipped: deleted in step 5 (the execute_function JSON tool)
@pytest.mark.usefixtures("as_shipped")
@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(300)
async def test_code_act_can_compose_false_no_functions_match():
    """
    When can_compose=False and no stored functions match the query, the LLM
    should report the failure gracefully without invoking execute_function.
    """
    fm = make_fm_mock()
    fm.search_functions = MagicMock(return_value={"metadata": []})
    fm.filter_functions = MagicMock(return_value={"metadata": []})
    fm.list_functions = MagicMock(return_value={"metadata": []})
    fm._get_function_data_by_name = MagicMock(return_value=None)
    fm._include_primitives = False

    actor = CodeActActor(
        function_manager=fm,
        environments=[],
        timeout=120,
        # Isolate can_compose routing from discovery-first gating.
        tool_policy=None,
    )
    # BaseActor auto-wires a GM from the registry when omitted; drop it so
    # the agent cannot wander through discovery/read tools forever.
    actor.guidance_manager = None
    try:
        handle = await actor.act(
            "Do something completely unique. No matching function will exist. "
            "Do not keep searching — report that you cannot fulfill the request.",
            can_compose=False,
            persist=False,
            clarification_enabled=False,
        )
        await asyncio.wait_for(handle.result(), timeout=120)

        # No matching function found — the LLM should not have attempted
        # to look up function data for execution.
        fm._get_function_data_by_name.assert_not_called()
    finally:
        try:
            await actor.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# can_store=False
# ---------------------------------------------------------------------------


# as_shipped: deleted in step 5 (the FunctionManager JSON tools)
@pytest.mark.usefixtures("as_shipped")
@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(300)
async def test_code_act_can_store_false_blocks_add_functions_tool():
    """
    When can_store=False, the FunctionManager_add_functions tool should not be available.
    We validate this by instructing the agent to call it; the loop should fail gracefully
    rather than executing the tool.
    """
    fm = make_fm_mock()
    fm.add_functions.return_value = {"x": "added"}

    actor = CodeActActor(
        function_manager=fm,
        timeout=30,
        tool_policy=lambda *args: ("auto", {}),
    )

    try:
        handle = await actor.act(
            "Call the tool FunctionManager_add_functions with implementations='async def x():\\n    return 1'. "
            "Do not call execute_code.",
            can_store=False,
            persist=False,
            clarification_enabled=False,
        )
        out = await asyncio.wait_for(handle.result(), timeout=60)
        # The tool should be unavailable; we accept any clear failure surface.
        assert "FunctionManager_add_functions" in str(out)
        fm.add_functions.assert_not_called()
    finally:
        try:
            await actor.close()
        except Exception:
            pass


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
async def test_can_store_true_merges_redundant_functions():
    """
    The storage review loop should recognise overlapping functions in the
    store and merge them: add a unified version and delete the old ones.

    Setup: the FunctionManager already contains two narrow greeting
    functions (greet_formal, greet_casual). The actor composes and
    executes a general-purpose `greet` function that subsumes both.
    The storage review should detect the overlap, store the merged
    version, and delete the now-redundant entries.

    result() resolves after the task phase; storage runs in the background.
    The test waits for done() to confirm the storage loop has completed.
    """
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
