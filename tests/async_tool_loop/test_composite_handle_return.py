"""
tests/async_tool_loop/test_composite_handle_return.py
=====================================================

Tests for composite tool returns: a tool returns both intermediate data
*and* a SteerableToolHandle nested inside a dict / list / tuple.

The loop extracts the handle, replaces it with a sentinel in the
intermediate data, presents the data to the LLM as in-flight progress,
and continues steering the handle as usual.
"""

import asyncio

import pytest

from unify.common.async_tool_loop import (
    SteerableToolHandle,
    start_async_tool_loop,
)
from unify.common._async_tool.tools_data import (
    _extract_nested_handle,
    _HANDLE_SENTINEL,
    _handle_label_sentinel,
)
from unify.common.tool_spec import ToolSpec
from unify.common.llm_client import new_llm_client
from tests.async_helpers import (
    _wait_for_tool_request,
    _wait_for_tool_result,
    _wait_for_assistant_steer_action,
    real_tool_messages,
    any_tool_message_content_contains,
)
from tests.baked_defaults import as_shipped  # noqa: F401

# as_shipped: deleted in step 5 (steerable handles)
pytestmark = [pytest.mark.llm_call, pytest.mark.usefixtures("as_shipped")]

# ─────────────────────────────────────────────────────────────────────────────
#  Unit tests for _extract_nested_handle
# ─────────────────────────────────────────────────────────────────────────────


class _MinimalHandle(SteerableToolHandle):
    """Lightweight concrete handle for unit tests (no event loop needed)."""

    def __init__(self):
        pass

    async def ask(self, question, **kw):
        return ""

    async def interject(self, message, **kw):
        pass

    def stop(self, reason=None, **kw):
        pass

    def pause(self, **kw):
        pass

    def resume(self, **kw):
        pass

    def done(self):
        return True

    async def result(self):
        return "done"

    async def next_clarification(self):
        return {}

    async def next_notification(self):
        return {}

    async def answer_clarification(self, cid, ans):
        pass


class TestExtractNestedHandle:
    """Pure-function unit tests – no LLM calls needed."""

    def _make_handle(self):
        return _MinimalHandle()

    def test_no_handle_returns_none(self):
        obj = {"a": 1, "b": [2, 3]}
        handle, cleaned = _extract_nested_handle(obj)
        assert handle is None
        assert cleaned is obj

    def test_direct_handle_returned(self):
        h = self._make_handle()
        handle, cleaned = _extract_nested_handle(h)
        assert handle is h
        assert cleaned == _HANDLE_SENTINEL

    def test_handle_in_dict(self):
        h = self._make_handle()
        obj = {"data": 42, "handle": h}
        handles, cleaned = _extract_nested_handle(obj)
        assert len(handles) == 1
        assert handles[0][0] is h
        assert handles[0][1] == "h0"
        assert cleaned == {"data": 42, "handle": _handle_label_sentinel("h0")}

    def test_handle_in_list(self):
        h = self._make_handle()
        obj = ["some_data", h]
        handles, cleaned = _extract_nested_handle(obj)
        assert len(handles) == 1
        assert handles[0][0] is h
        assert handles[0][1] == "h0"
        assert cleaned == ["some_data", _handle_label_sentinel("h0")]

    def test_handle_in_tuple(self):
        h = self._make_handle()
        obj = (False, h)
        handles, cleaned = _extract_nested_handle(obj)
        assert len(handles) == 1
        assert handles[0][0] is h
        assert handles[0][1] == "h0"
        assert cleaned == (False, _handle_label_sentinel("h0"))

    def test_handle_deeply_nested(self):
        h = self._make_handle()
        obj = {"result": {"nested": [{"deep": h}]}}
        handles, cleaned = _extract_nested_handle(obj)
        assert len(handles) == 1
        assert handles[0][0] is h
        assert handles[0][1] == "h0"
        assert cleaned == {
            "result": {"nested": [{"deep": _handle_label_sentinel("h0")}]},
        }

    def test_multiple_handles_extracted(self):
        h1 = self._make_handle()
        h2 = self._make_handle()
        obj = {"a": h1, "b": h2}
        handles, cleaned = _extract_nested_handle(obj)
        assert len(handles) == 2
        assert handles[0][0] is h1
        assert handles[0][1] == "h0"
        assert handles[1][0] is h2
        assert handles[1][1] == "h1"
        assert cleaned == {
            "a": _handle_label_sentinel("h0"),
            "b": _handle_label_sentinel("h1"),
        }

    def test_plain_values_untouched(self):
        obj = {"x": 1, "y": "hello", "z": [True, None, 3.14]}
        handle, cleaned = _extract_nested_handle(obj)
        assert handle is None
        assert cleaned is obj

    def test_handle_inside_pydantic_model(self):
        """A handle nested inside a Pydantic model attribute should be found.

        This mirrors the execute_code path: PythonExecutionSession returns a
        dict that gets wrapped in ExecutionResult (a BaseModel). If the
        executed code returned a SteerableToolHandle, it ends up in the
        ``result`` attribute. _extract_nested_handle must detect it.
        """
        from pydantic import BaseModel, ConfigDict
        from typing import Any

        class FakeExecutionResult(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            stdout: str = ""
            stderr: str = ""
            result: Any = None
            error: str | None = None

        h = self._make_handle()
        obj = FakeExecutionResult(result=h)
        handles, cleaned = _extract_nested_handle(obj)
        assert len(handles) == 1
        assert (
            handles[0][0] is h
        ), "_extract_nested_handle should find a handle inside a Pydantic model"
        assert handles[0][1] == "h0"
        # The cleaned model should have the handle replaced with a labeled sentinel
        assert getattr(cleaned, "result", None) == _handle_label_sentinel("h0")


# ─────────────────────────────────────────────────────────────────────────────
#  Integration test: composite dict return with handle
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_composite_dict_return_with_handle(llm_config):
    """Tool returns ``{"found": True, "handle": <handle>}``.

    The outer loop should:
    1. Extract the handle and adopt it for steering.
    2. Present the intermediate data (with sentinel) as progress to the LLM.
    3. Remain steerable via steer(call_id=..., action=...).
    4. Eventually complete with the handle's final result.
    """

    inner_gate = asyncio.Event()

    class SimpleHandle(SteerableToolHandle):
        """Minimal handle backed by an asyncio gate."""

        def __init__(self):
            self._done = asyncio.Event()
            self._gate = inner_gate
            self._stopped = False

        async def ask(self, question, **kw):
            return "progress: waiting for gate"

        async def interject(self, message, **kw):
            pass

        def stop(self, reason=None, **kw):
            self._stopped = True
            self._done.set()

        def pause(self, **kw):
            pass

        def resume(self, **kw):
            pass

        def done(self):
            return self._done.is_set()

        async def result(self):
            await self._gate.wait()
            self._done.set()
            return "inner-complete"

        async def next_clarification(self):
            return {}

        async def next_notification(self):
            return {}

        async def answer_clarification(self, cid, ans):
            pass

    async def composite_tool() -> dict:
        """Return intermediate data alongside a steerable handle."""
        handle = SimpleHandle()
        return {"found": True, "count": 7, "handle": handle}

    composite_tool.__name__ = "composite_tool"
    composite_tool.__qualname__ = "composite_tool"

    client = new_llm_client(**llm_config)
    client.set_system_message(
        "You are running inside an automated test. Perform the steps exactly:\n"
        "1️⃣  Call `composite_tool` with no arguments.\n"
        "2️⃣  Observe the intermediate result. The tool is still running.\n"
        "3️⃣  Continue waiting for it to complete.\n"
        "4️⃣  Once it finishes, respond with exactly 'all done'.",
    )

    outer_handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"composite_tool": ToolSpec(fn=composite_tool, max_total_calls=1)},
        max_steps=20,
        timeout=240,
    )

    # Wait for the tool to be called and the placeholder to be inserted
    await _wait_for_tool_request(client, "composite_tool")
    await _wait_for_tool_result(client, tool_name="composite_tool", min_results=1)

    # Verify the intermediate content is visible in the placeholder
    tool_msgs = real_tool_messages(client.messages, tool_name="composite_tool")
    assert tool_msgs, "Expected a tool message placeholder for composite_tool"
    placeholder_content = tool_msgs[0].get("content", "")
    # The intermediate data should contain "found" and the sentinel
    assert (
        "found" in placeholder_content
    ), f"Intermediate data should contain 'found'; got: {placeholder_content}"
    assert (
        "_placeholder" in placeholder_content
    ), f"Intermediate data should be formatted as progress; got: {placeholder_content}"

    # Release the inner gate so the handle completes
    inner_gate.set()

    final = await outer_handle.result()
    assert final is not None, "Loop should complete with a response"

    # The handle's final result lands either in the placeholder (if it was
    # still mutable when the handle completed) or in a per-child
    # check_status pair (if the placeholder had already been dispatched in
    # an earlier request while the model kept waiting). Either way it must
    # be in the transcript.
    assert any_tool_message_content_contains(client.messages, "inner-complete"), (
        "Expected the handle's final result ('inner-complete') to appear "
        "somewhere in the transcript (placeholder or check_status pair)"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Integration test: composite return with steering (stop)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_composite_return_stop_steering(llm_config):
    """Tool returns composite data + handle. The outer loop should stay
    steerable via steer(action="stop") propagating to the inner handle.
    """

    stop_called = {"count": 0}

    class StoppableHandle(SteerableToolHandle):
        def __init__(self):
            self._done = asyncio.Event()

        async def ask(self, question, **kw):
            return "ask response"

        async def interject(self, message, **kw):
            pass

        def stop(self, reason=None, **kw):
            stop_called["count"] += 1
            self._done.set()

        def pause(self, **kw):
            pass

        def resume(self, **kw):
            pass

        def done(self):
            return self._done.is_set()

        async def result(self):
            await self._done.wait()
            return "stopped-result"

        async def next_clarification(self):
            return {}

        async def next_notification(self):
            return {}

        async def answer_clarification(self, cid, ans):
            pass

    async def lookup_tool() -> dict:
        """Return partial lookup results and a handle for the ongoing operation."""
        return {"partial_results": ["item_a", "item_b"], "handle": StoppableHandle()}

    lookup_tool.__name__ = "lookup_tool"
    lookup_tool.__qualname__ = "lookup_tool"

    client = new_llm_client(**llm_config)
    client.set_system_message(
        "You are running inside an automated test.\n"
        "1️⃣  Call `lookup_tool` with no arguments.\n"
        "2️⃣  If the user says 'stop', immediately call "
        'steer(call_id=<the id from the "[steerable ...]" announcement for '
        'this call>, action="stop") exactly once.\n'
        "3️⃣  Once stopped, reply with exactly 'stopped'.",
    )

    outer_handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"lookup_tool": ToolSpec(fn=lookup_tool, max_total_calls=1)},
        max_steps=20,
        timeout=240,
    )

    await _wait_for_tool_request(client, "lookup_tool")
    await _wait_for_tool_result(client, tool_name="lookup_tool", min_results=1)

    # Interject to trigger the stop action
    await outer_handle.interject("stop")
    await _wait_for_assistant_steer_action(client, "stop")

    final = await outer_handle.result()
    assert final is not None, "Loop should complete"
    assert stop_called["count"] >= 1, "Inner handle stop() should be invoked"


# ─────────────────────────────────────────────────────────────────────────────
#  Integration test: composite return with nested async tool loop
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_composite_return_with_nested_loop(llm_config):
    """Tool returns a dict with metadata + an AsyncToolLoopHandle from
    ``start_async_tool_loop``. The outer loop should present the metadata
    as progress and steer the inner loop to completion.
    """

    async def inner_echo() -> str:
        """Simple inner tool."""
        return "echo-reply"

    inner_echo.__name__ = "inner_echo"
    inner_echo.__qualname__ = "inner_echo"

    async def composite_with_loop() -> dict:
        """Launch a nested loop and return it alongside metadata."""
        inner_client = new_llm_client(**llm_config)
        inner_client.set_system_message(
            "Call `inner_echo` then reply with exactly the result.",
        )
        handle = start_async_tool_loop(
            client=inner_client,
            message="go",
            tools={"inner_echo": inner_echo},
            max_steps=5,
            timeout=60,
        )
        return {"status": "launched", "task_id": "abc-123", "handle": handle}

    composite_with_loop.__name__ = "composite_with_loop"
    composite_with_loop.__qualname__ = "composite_with_loop"

    client = new_llm_client(**llm_config)
    client.set_system_message(
        "You are running inside an automated test.\n"
        "1️⃣  Call `composite_with_loop` with no arguments.\n"
        "2️⃣  Observe the intermediate progress (it should show status and task_id).\n"
        "3️⃣  Wait for it to complete.\n"
        "4️⃣  Reply with exactly 'finished'.",
    )

    outer_handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={
            "composite_with_loop": ToolSpec(
                fn=composite_with_loop,
                max_total_calls=1,
            ),
        },
        max_steps=20,
        timeout=240,
    )

    final = await outer_handle.result()
    assert final is not None, "Outer loop should complete"

    # The intermediate content should have been visible as progress
    tool_msgs = [
        m
        for m in client.messages
        if m.get("role") == "tool" and m.get("name") == "composite_with_loop"
    ]
    assert tool_msgs, "Expected tool messages for composite_with_loop"


# ─────────────────────────────────────────────────────────────────────────────
#  Integration test: multiple handles in return (adopted independently)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_multi_handle_return_completes(llm_config):
    """Tool returns two handles in a dict. Both are adopted as independent
    steerable tasks and the shared placeholder is progressively updated as
    each handle completes.
    """

    gate_a = asyncio.Event()
    gate_b = asyncio.Event()

    class HandleA(SteerableToolHandle):
        def __init__(self):
            self._done = asyncio.Event()

        async def ask(self, question, **kw):
            return "handle_a progress"

        async def interject(self, message, **kw):
            pass

        def stop(self, reason=None, **kw):
            self._done.set()

        def pause(self, **kw):
            pass

        def resume(self, **kw):
            pass

        def done(self):
            return self._done.is_set()

        async def result(self):
            await gate_a.wait()
            self._done.set()
            return "alpha-result"

        async def next_clarification(self):
            return {}

        async def next_notification(self):
            return {}

        async def answer_clarification(self, cid, ans):
            pass

    class HandleB(SteerableToolHandle):
        def __init__(self):
            self._done = asyncio.Event()

        async def ask(self, question, **kw):
            return "handle_b progress"

        async def interject(self, message, **kw):
            pass

        def stop(self, reason=None, **kw):
            self._done.set()

        def pause(self, **kw):
            pass

        def resume(self, **kw):
            pass

        def done(self):
            return self._done.is_set()

        async def result(self):
            await gate_b.wait()
            self._done.set()
            return "beta-result"

        async def next_clarification(self):
            return {}

        async def next_notification(self):
            return {}

        async def answer_clarification(self, cid, ans):
            pass

    async def dual_tool() -> dict:
        """Return two steerable handles."""
        return {"alpha": HandleA(), "beta": HandleB()}

    dual_tool.__name__ = "dual_tool"
    dual_tool.__qualname__ = "dual_tool"

    client = new_llm_client(**llm_config)
    client.set_system_message(
        "You are running inside an automated test.\n"
        "1️⃣  Call `dual_tool` with no arguments.\n"
        "2️⃣  Wait for the results to complete.\n"
        "3️⃣  Reply with exactly 'all done'.",
    )

    outer_handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"dual_tool": ToolSpec(fn=dual_tool, max_total_calls=1)},
        max_steps=25,
        timeout=240,
    )

    # Wait for the tool to be called and the placeholder to appear
    await _wait_for_tool_request(client, "dual_tool")
    await _wait_for_tool_result(client, tool_name="dual_tool", min_results=1)

    # Verify intermediate content shows labeled sentinels
    tool_msgs = real_tool_messages(client.messages, tool_name="dual_tool")
    assert tool_msgs, "Expected a tool message placeholder for dual_tool"
    ph_content = tool_msgs[0].get("content", "")
    assert "h0" in ph_content, f"Expected h0 sentinel; got: {ph_content}"
    assert "h1" in ph_content, f"Expected h1 sentinel; got: {ph_content}"

    # Release both gates so handles complete
    gate_a.set()
    gate_b.set()

    final = await outer_handle.result()
    assert final is not None, "Loop should complete"

    # Each handle's terminal result lands either in the shared placeholder
    # (while still mutable) or its own per-child check_status pair (once
    # the placeholder has been dispatched).
    assert any_tool_message_content_contains(client.messages, "alpha-result"), (
        "Expected alpha-result somewhere in the transcript (placeholder or "
        "check_status pair)"
    )
    assert any_tool_message_content_contains(client.messages, "beta-result"), (
        "Expected beta-result somewhere in the transcript (placeholder or "
        "check_status pair)"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Integration test: tuple return with handle
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_composite_tuple_return(llm_config):
    """Tool returns ``(metadata_dict, handle)`` as a tuple."""

    gate = asyncio.Event()

    class TupleHandle(SteerableToolHandle):
        def __init__(self):
            self._done = asyncio.Event()

        async def ask(self, question, **kw):
            return ""

        async def interject(self, message, **kw):
            pass

        def stop(self, reason=None, **kw):
            self._done.set()

        def pause(self, **kw):
            pass

        def resume(self, **kw):
            pass

        def done(self):
            return self._done.is_set()

        async def result(self):
            await gate.wait()
            self._done.set()
            return "tuple-handle-done"

        async def next_clarification(self):
            return {}

        async def next_notification(self):
            return {}

        async def answer_clarification(self, cid, ans):
            pass

    async def tuple_tool() -> tuple:
        """Return a tuple of (metadata, handle)."""
        return ({"ready": True, "item_count": 3}, TupleHandle())

    tuple_tool.__name__ = "tuple_tool"
    tuple_tool.__qualname__ = "tuple_tool"

    client = new_llm_client(**llm_config)
    client.set_system_message(
        "You are running inside an automated test.\n"
        "1️⃣  Call `tuple_tool` with no arguments.\n"
        "2️⃣  Wait for it to complete.\n"
        "3️⃣  Reply with exactly 'done'.",
    )

    outer_handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"tuple_tool": ToolSpec(fn=tuple_tool, max_total_calls=1)},
        max_steps=20,
        timeout=240,
    )

    await _wait_for_tool_request(client, "tuple_tool")
    await _wait_for_tool_result(client, tool_name="tuple_tool", min_results=1)

    # Verify intermediate content shows the tuple data with sentinel
    tool_msgs = real_tool_messages(client.messages, tool_name="tuple_tool")
    assert tool_msgs, "Expected a tool message for tuple_tool"
    ph_content = tool_msgs[0].get("content", "")
    assert (
        "ready" in ph_content
    ), f"Intermediate data should contain 'ready'; got: {ph_content}"

    # Release gate
    gate.set()

    final = await outer_handle.result()
    assert final is not None, "Loop should complete"

    # The handle's terminal result lands either in the placeholder (while
    # still mutable) or its own check_status pair (once dispatched).
    assert any_tool_message_content_contains(client.messages, "tuple-handle-done"), (
        "Expected the handle's result ('tuple-handle-done') somewhere in "
        "the transcript (placeholder or check_status pair)"
    )
