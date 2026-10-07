"""
Tests for execute_function with primitive functions (is_primitive=True).

Coverage
========
✓ Execute a primitive by name (sync callable)
✓ Execute a primitive by name (async callable)
✓ Execute a primitive that returns a SteerableToolHandle (raw passthrough)
✓ Primitive not found raises ValueError
✓ Primitive callable resolution failure raises ValueError
✓ Primitive with call_kwargs passes arguments through
✓ Compositional function still works (no regression)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.helpers import _handle_project
from unify.function_manager.function_manager import FunctionManager


@pytest.fixture(autouse=True)
def _python_in_process(monkeypatch):
    """Python in process (``UNIFY_WORKSPACE_PYTHON`` empty), the mode these tests exercise: stored functions run in this process. With the sandboxed worker nothing runs here (tests/actor/code_act/test_bind_load_confinement.py)."""
    from unify.settings import SETTINGS

    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "")


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def function_manager_factory():
    """Factory fixture that creates FunctionManager instances."""
    managers = []

    def _create():
        fm = FunctionManager()
        managers.append(fm)
        return fm

    yield _create

    for fm in managers:
        try:
            fm.clear()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────
# Test: Primitive execution via execute_function
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_execute_function_primitive_async(function_manager_factory):
    """execute_function routes to _execute_primitive for a primitive and
    returns the raw result of the async callable."""
    fm = function_manager_factory()

    mock_primitives = MagicMock()
    mock_primitives.actor = MagicMock()
    mock_primitives.actor.act = AsyncMock(return_value="Report summarised")

    result = await fm.execute_function(
        function_name="primitives.actor.act",
        call_kwargs={"request": "Summarise the report"},
        extra_namespaces={"primitives": mock_primitives},
    )

    # Primitive returns the raw result, not a dict envelope.
    assert result == "Report summarised"
    mock_primitives.actor.act.assert_called_once_with(request="Summarise the report")


@_handle_project
@pytest.mark.asyncio
async def test_execute_function_primitive_sync(function_manager_factory):
    """execute_function handles primitives whose callable is synchronous."""
    fm = function_manager_factory()

    mock_primitives = MagicMock()
    mock_primitives.actor = MagicMock()
    # Sync callable (non-coroutine)
    mock_primitives.actor.act = MagicMock(return_value="The sky is blue")

    result = await fm.execute_function(
        function_name="primitives.actor.act",
        call_kwargs={"request": "What colour is the sky?"},
        extra_namespaces={"primitives": mock_primitives},
    )

    assert result == "The sky is blue"
    mock_primitives.actor.act.assert_called_once_with(
        request="What colour is the sky?",
    )


@_handle_project
@pytest.mark.asyncio
async def test_execute_function_primitive_returns_handle(function_manager_factory):
    """execute_function passes through SteerableToolHandle objects from
    primitives without wrapping them in a dict."""
    fm = function_manager_factory()

    # Simulate a primitive that returns a SteerableToolHandle
    from unify.common.async_tool_loop import SteerableToolHandle

    class FakeHandle(SteerableToolHandle):
        def __init__(self):
            pass

        async def ask(self, question, **kw):
            return self

        async def interject(self, message, **kw):
            pass

        async def stop(self, reason=None):
            pass

        async def pause(self):
            return None

        async def resume(self):
            return None

        def done(self):
            return True

        async def result(self):
            return "handle result"

        async def next_clarification(self):
            return {}

        async def next_notification(self):
            return {}

        async def answer_clarification(self, call_id, answer):
            pass

    fake_handle = FakeHandle()

    mock_primitives = MagicMock()
    mock_primitives.actor = MagicMock()
    mock_primitives.actor.act = AsyncMock(return_value=fake_handle)

    result = await fm.execute_function(
        function_name="primitives.actor.act",
        call_kwargs={"request": "Find Bob"},
        extra_namespaces={"primitives": mock_primitives},
    )

    # The raw handle must flow through, not a serialized dict.
    assert isinstance(result, SteerableToolHandle)
    assert result is fake_handle


@_handle_project
@pytest.mark.asyncio
async def test_execute_primitive_forwards_parent_chat_context(function_manager_factory):
    """execute_function should forward _parent_chat_context to primitive
    callables whose signature accepts it.

    This verifies the full path: execute_function receives
    _parent_chat_context → _execute_primitive inspects the callable's
    signature → injects _parent_chat_context into the call kwargs.
    """
    fm = function_manager_factory()

    received = {}

    async def fake_act(request: str, _parent_chat_context: list[dict] | None = None):
        received["ctx"] = _parent_chat_context
        return f"Answered: {request}"

    mock_primitives = MagicMock()
    mock_primitives.actor = MagicMock()
    mock_primitives.actor.act = fake_act

    parent_ctx = [{"role": "user", "content": "Hello"}]

    result = await fm.execute_function(
        function_name="primitives.actor.act",
        call_kwargs={"request": "Who is Alice?"},
        extra_namespaces={"primitives": mock_primitives},
        _parent_chat_context=parent_ctx,
    )

    assert result == "Answered: Who is Alice?"
    assert (
        received["ctx"] is parent_ctx
    ), "_parent_chat_context was not forwarded to the primitive callable"


@_handle_project
@pytest.mark.asyncio
async def test_execute_function_primitive_no_kwargs(function_manager_factory):
    """execute_function works for primitives that take no arguments."""
    fm = function_manager_factory()

    mock_primitives = MagicMock()
    mock_primitives.actor = MagicMock()
    mock_primitives.actor.act = AsyncMock(return_value="Nothing to do")

    result = await fm.execute_function(
        function_name="primitives.actor.act",
        # No call_kwargs provided
        extra_namespaces={"primitives": mock_primitives},
    )

    assert result == "Nothing to do"
    mock_primitives.actor.act.assert_called_once_with()


# ────────────────────────────────────────────────────────────────────────────
# Test: Error cases
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_execute_function_primitive_not_found(function_manager_factory):
    """execute_function raises ValueError for a primitive name that doesn't exist."""
    fm = function_manager_factory()

    with pytest.raises(ValueError, match="not found"):
        await fm.execute_function(
            function_name="NonExistentManager.do_stuff",
        )


@_handle_project
@pytest.mark.asyncio
async def test_execute_function_primitive_callable_resolution_failure(
    function_manager_factory,
):
    """execute_function raises ValueError when get_primitive_callable returns None."""
    fm = function_manager_factory()

    with patch(
        "unify.function_manager.primitives.runtime.get_primitive_callable",
        return_value=None,
    ):
        with pytest.raises(ValueError, match="Could not resolve primitive callable"):
            await fm.execute_function(
                function_name="primitives.actor.act",
                call_kwargs={"request": "Hello"},
            )


# ────────────────────────────────────────────────────────────────────────────
# Test: _parent_chat_context flows to composed functions via
#       ContextForwardingProxy in _execute_in_default_env
# ────────────────────────────────────────────────────────────────────────────


FUNCTION_THAT_CALLS_PRIMITIVES = """
async def delegate_with_context(question: str):
    return await primitives.actor.act(request=question)
""".strip()


@_handle_project
@pytest.mark.asyncio
async def test_execute_function_composed_forwards_parent_chat_context(
    function_manager_factory,
):
    """_parent_chat_context is forwarded to primitive methods called from
    composed functions via ContextForwardingProxy wrapping in
    _execute_in_default_env."""
    fm = function_manager_factory()

    fm.add_functions(implementations=FUNCTION_THAT_CALLS_PRIMITIVES)

    received_ctx = {}

    async def fake_act(
        request: str,
        _parent_chat_context: list[dict] | None = None,
    ):
        received_ctx["value"] = _parent_chat_context
        return f"Answer: {request}"

    mock_primitives = MagicMock()
    mock_primitives.actor = MagicMock()
    mock_primitives.actor.act = fake_act

    parent_ctx = [{"role": "user", "content": "Hello"}]

    result = await fm.execute_function(
        function_name="delegate_with_context",
        call_kwargs={"question": "Who is Alice?"},
        extra_namespaces={"primitives": mock_primitives},
        _parent_chat_context=parent_ctx,
    )

    assert result["error"] is None, f"Unexpected error: {result['error']}"
    assert result["result"] == "Answer: Who is Alice?"
    assert (
        received_ctx.get("value") is parent_ctx
    ), "_parent_chat_context was not forwarded via ContextForwardingProxy"


# ────────────────────────────────────────────────────────────────────────────
# Test: Regression — composed functions still work
# ────────────────────────────────────────────────────────────────────────────


SIMPLE_SYNC_FUNCTION = """
def add_numbers(a, b):
    \"\"\"Add two numbers together.\"\"\"
    return a + b
""".strip()


@_handle_project
@pytest.mark.asyncio
async def test_execute_function_composed_still_works(function_manager_factory):
    """Composed (non-primitive) functions still execute via the existing path."""
    fm = function_manager_factory()

    fm.add_functions(implementations=SIMPLE_SYNC_FUNCTION)

    result = await fm.execute_function(
        function_name="add_numbers",
        call_kwargs={"a": 7, "b": 3},
    )

    assert isinstance(result, dict)
    assert result["error"] is None
    assert result["result"] == 10
