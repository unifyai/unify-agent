"""Tests for storage-check-on-stop behavior.

When a ``_StorageCheckHandle``-wrapped session is stopped, the storage
review (Phase 2) should still run — it is no longer skipped.  The stop
reason is forwarded to the skill librarian so it can weigh user intent.

* ``test_storage_check_runs_after_stop`` — symbolic infrastructure test
  verifying that Phase 2 is entered and ``_start_storage_check_loop`` is
  called with the stop reason.

* ``test_persist_stop_with_memoize_intent_stores_function`` — eval test
  verifying the full flow: a persist=True session that is stopped with a
  "remember this" signal successfully stores a function.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unify.actor.code_act_actor import CodeActActor, _StorageCheckHandle

# ---------------------------------------------------------------------------
# Symbolic: _StorageCheckHandle runs Phase 2 after stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_storage_check_runs_after_stop():
    """Phase 2 (storage check) runs even when the handle is stopped.

    Uses a mock inner handle and patches ``_start_storage_check_loop``
    to verify it is invoked with the stop reason.
    """
    result_future: asyncio.Future[str] = asyncio.get_event_loop().create_future()

    inner = MagicMock()

    async def _await_result():
        return await result_future

    inner.result = _await_result
    inner.next_notification = AsyncMock(
        side_effect=lambda: asyncio.Event().wait(),
    )

    async def _stop(**kwargs):
        if not result_future.done():
            result_future.set_result("stopped by user")

    inner.stop = AsyncMock(side_effect=_stop)

    mock_client = MagicMock()
    mock_client.messages = [{"role": "user", "content": "do something"}]
    inner._client = mock_client

    mock_task = MagicMock()
    inner._task = mock_task

    actor = MagicMock()
    actor.function_manager = None
    actor.guidance_manager = None

    with (
        patch(
            "unify.actor.code_act_actor._start_storage_check_loop",
        ) as mock_loop,
        patch(
            "unify.actor.code_act_actor.publish_manager_method_event",
            new_callable=AsyncMock,
        ),
    ):
        mock_loop.return_value = None

        handle = _StorageCheckHandle(inner=inner, actor=actor)

        await handle.stop(reason="User wants this workflow saved")

        deadline = asyncio.get_event_loop().time() + 10
        while not handle.done():
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError("Handle did not complete")
            await asyncio.sleep(0.1)

        mock_loop.assert_called_once()
        call_kwargs = mock_loop.call_args.kwargs
        assert call_kwargs["stop_reason"] == "User wants this workflow saved"
        assert call_kwargs["original_result"] == "stopped by user"
        assert call_kwargs["actor"] is actor


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_storage_check_runs_after_stop_no_reason():
    """Phase 2 runs even with no stop reason (stop_reason=None)."""
    result_future: asyncio.Future[str] = asyncio.get_event_loop().create_future()

    inner = MagicMock()

    async def _await_result():
        return await result_future

    inner.result = _await_result
    inner.next_notification = AsyncMock(
        side_effect=lambda: asyncio.Event().wait(),
    )

    async def _stop(**kwargs):
        if not result_future.done():
            result_future.set_result("stopped")

    inner.stop = AsyncMock(side_effect=_stop)

    mock_client = MagicMock()
    mock_client.messages = []
    inner._client = mock_client

    mock_task = MagicMock()
    inner._task = mock_task

    actor = MagicMock()
    actor.function_manager = None
    actor.guidance_manager = None

    with (
        patch(
            "unify.actor.code_act_actor._start_storage_check_loop",
        ) as mock_loop,
        patch(
            "unify.actor.code_act_actor.publish_manager_method_event",
            new_callable=AsyncMock,
        ),
    ):
        mock_loop.return_value = None

        handle = _StorageCheckHandle(inner=inner, actor=actor)

        await handle.stop()

        deadline = asyncio.get_event_loop().time() + 10
        while not handle.done():
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError("Handle did not complete")
            await asyncio.sleep(0.1)

        mock_loop.assert_called_once()
        assert mock_loop.call_args.kwargs["stop_reason"] is None


# ---------------------------------------------------------------------------
# Symbolic: StorageCheck incoming event carries instructions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_storage_check_incoming_event_has_instructions():
    """The incoming ManagerMethod event for StorageCheck must carry a non-null
    ``instructions`` field so that every incoming event has at least one
    content payload (instructions, request, or question)."""
    result_future: asyncio.Future[str] = asyncio.get_event_loop().create_future()

    inner = MagicMock()

    async def _await_result():
        return await result_future

    inner.result = _await_result
    inner.next_notification = AsyncMock(
        side_effect=lambda: asyncio.Event().wait(),
    )

    async def _stop(**kwargs):
        if not result_future.done():
            result_future.set_result("done")

    inner.stop = AsyncMock(side_effect=_stop)

    mock_client = MagicMock()
    mock_client.messages = [{"role": "user", "content": "do something"}]
    inner._client = mock_client

    mock_task = MagicMock()
    inner._task = mock_task

    actor = MagicMock()
    actor.function_manager = None
    actor.guidance_manager = None

    with (
        patch(
            "unify.actor.code_act_actor._start_storage_check_loop",
        ) as mock_loop,
        patch(
            "unify.actor.code_act_actor.publish_manager_method_event",
            new_callable=AsyncMock,
        ) as mock_publish,
    ):
        mock_loop.return_value = None

        handle = _StorageCheckHandle(inner=inner, actor=actor)

        await handle.stop(reason="save this")

        deadline = asyncio.get_event_loop().time() + 10
        while not handle.done():
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError("Handle did not complete")
            await asyncio.sleep(0.1)

    incoming_calls = [
        c
        for c in mock_publish.call_args_list
        if c.kwargs.get("phase") == "incoming" and c.args[2] == "StorageCheck"
    ]
    assert (
        len(incoming_calls) == 1
    ), f"Expected 1 incoming StorageCheck publish call, got {len(incoming_calls)}"
    instructions = incoming_calls[0].kwargs.get("instructions")
    assert instructions and isinstance(
        instructions,
        str,
    ), f"StorageCheck incoming event must have a non-null instructions kwarg, got {instructions!r}"


# ---------------------------------------------------------------------------
# Eval: persist=True + stop with memoize intent stores a function
# ---------------------------------------------------------------------------

pytestmark_eval = pytest.mark.eval


@pytestmark_eval
@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(300)
async def test_persist_stop_with_memoize_intent_stores_function(monkeypatch):
    """A persist=True session stopped with "remember this" stores a function.

    The actor executes a reusable utility during a guided session, then is
    stopped with a memoization-intent reason.  The storage review loop
    should detect the pattern and store it in the FunctionManager.
    """
    from unify.common._async_tool import loop as tool_loop
    from unify.function_manager.function_manager import FunctionManager

    fm = FunctionManager(include_primitives=False)

    # The memoize-stop premise requires the actor to have actually
    # executed the utility before the stop lands — the librarian only
    # stores code that ran successfully, so stopping a still-empty
    # trajectory tests nothing. The discovery-first searches complete
    # first, so wait for a completed ``execute_code`` round-trip
    # specifically, not for the first tool result of any kind, and one
    # that succeeded: a cell that raised (the 09184dc6d re-record's only
    # cell, before the sandbox put ``python`` on PATH) leaves nothing
    # that ran, so the review rightly stores nothing. And the cell must
    # be one that defines ``format_currency``: in the 7299344a6
    # re-record the only cell before the stop was
    # ``help(functions.add)``, which succeeded, so the stop landed before
    # the utility existed and the review rightly said "The task stopped
    # before `format_currency` was implemented or tested".
    def _succeeded(message: dict) -> bool:
        content = message.get("content")
        if isinstance(content, list):
            content = next(
                (
                    part.get("text")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ),
                "",
            )
        try:
            head = json.loads(content or "")
        except (TypeError, ValueError):
            return False
        return isinstance(head, dict) and not head.get("error")

    def _defines_format_currency(call: dict) -> bool:
        arguments = call.get("function", {}).get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                return "def format_currency" in arguments
        code = arguments.get("code") if isinstance(arguments, dict) else None
        return isinstance(code, str) and "def format_currency" in code

    def _execute_code_round_trip_happened(messages) -> bool:
        msgs = [m for m in (messages or []) if isinstance(m, dict)]
        exec_call_ids = {
            call["id"]
            for m in msgs
            for call in (m.get("tool_calls") or [])
            if call.get("function", {}).get("name") == "execute_code"
            and _defines_format_currency(call)
        }
        return any(
            m.get("role") == "tool"
            and any(
                str(m.get("tool_call_id", "")).startswith(call_id)
                for call_id in exec_call_ids
            )
            and _succeeded(m)
            for m in msgs
        )

    # The actor's first turn after the cell that defined format_currency is
    # held (never sent) until the stop cancels it; every other call runs.
    turn_held = asyncio.Event()
    dispatch = tool_loop.generate_with_preprocess

    async def _hold_the_turn_after_the_cell(client, preprocess_msgs, **kwargs):
        if not turn_held.is_set() and _execute_code_round_trip_happened(
            getattr(client, "messages", None),
        ):
            turn_held.set()
            await asyncio.Event().wait()  # until the stop cancels the turn
        return await dispatch(client, preprocess_msgs, **kwargs)

    monkeypatch.setattr(
        tool_loop,
        "generate_with_preprocess",
        _hold_the_turn_after_the_cell,
    )

    actor = CodeActActor(
        function_manager=fm,
        timeout=60,
    )
    try:
        handle = await actor.act(
            "Write a reusable Python function called `format_currency` that:\n"
            "1. Takes amount (float) and currency_code (str, default 'USD')\n"
            "2. Formats with proper symbol ($, €, £) and 2 decimal places\n"
            "3. Handles negative amounts with parentheses: ($1,234.56)\n"
            "4. Adds thousand separators\n"
            "5. Raises ValueError for unsupported currency codes\n\n"
            "Test it with a few examples including negative amounts.",
            can_store=True,
            persist=True,
            clarification_enabled=False,
        )

        # The stop lands before the actor's next turn goes out: that turn is
        # held until the stop cancels it, as a stop does with a turn in
        # flight. Polling the transcript instead raced the turn: with a live
        # model the stop cancelled it in flight, but a replayed response is
        # instant, so in a keyless run the turn (never answered, so never
        # recorded) went out first and missed the cache.
        exec_deadline = asyncio.get_event_loop().time() + 120
        while not turn_held.is_set():
            if handle.done() or asyncio.get_event_loop().time() > exec_deadline:
                pytest.skip(
                    "Actor did not complete a successful execute_code call "
                    "defining format_currency within 120s — "
                    "this is an eval-sensitive path",
                )
            await asyncio.sleep(0.2)

        assert not handle.done(), "persist=True should keep the loop alive"

        await handle.stop(
            reason="User wants this workflow saved for future autonomous execution",
        )
        result = await asyncio.wait_for(handle.result(), timeout=30)
        assert result is not None

        deadline = asyncio.get_event_loop().time() + 120
        while not handle.done():
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError("Storage loop did not complete in time")
            await asyncio.sleep(0.5)

        stored = fm.filter_functions()
        assert stored, (
            "Expected FunctionManager to contain at least one stored function "
            "after stop-with-memoize on a persist=True session."
        )
    finally:
        try:
            await actor.close()
        except Exception:
            pass
