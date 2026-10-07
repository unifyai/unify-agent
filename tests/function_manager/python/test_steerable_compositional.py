"""
Tests for steerable compositional functions.

Verifies that compositional functions can create and return steerable handles,
and that runtime detection via isinstance(result, SteerableToolHandle) works correctly.
"""

import asyncio

import pytest

from unify.common.async_tool_loop import SteerableToolHandle
from unify.function_manager.execution_env import create_execution_globals
from tests.baked_defaults import as_shipped  # noqa: F401


class _ImmediateHandle(SteerableToolHandle):
    def __init__(self, value: str = "done") -> None:
        self.value = value
        self.stopped = False
        self.interjections: list[str] = []

    async def ask(
        self,
        question: str,
        *,
        _parent_chat_context: list[dict] | None = None,
    ) -> SteerableToolHandle:
        return _ImmediateHandle(f"asked: {question}")

    async def interject(
        self,
        message: str,
        *,
        _parent_chat_context_cont: list[dict] | None = None,
    ) -> None:
        self.interjections.append(message)

    async def stop(self, reason: str | None = None) -> None:
        self.stopped = True

    async def pause(self) -> str | None:
        return None

    async def resume(self) -> str | None:
        return None

    def done(self) -> bool:
        return self.stopped

    def result(self) -> str:
        return self.value

    async def next_clarification(self) -> dict:
        return {}

    async def next_notification(self) -> dict:
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        return None


# ────────────────────────────────────────────────────────────────────────────
# Runtime Detection Tests
# ────────────────────────────────────────────────────────────────────────────


def test_steerable_handle_runtime_detection():
    """Runtime isinstance check should correctly identify SteerableToolHandle subclasses."""
    globals_dict = create_execution_globals()

    assert globals_dict["SteerableToolHandle"] is SteerableToolHandle


def test_steerable_detection_with_non_handle():
    """Non-handle return values should not be detected as steerable."""
    # Plain values should not be steerable
    assert not isinstance("hello", SteerableToolHandle)
    assert not isinstance(42, SteerableToolHandle)
    assert not isinstance({"key": "value"}, SteerableToolHandle)
    assert not isinstance(None, SteerableToolHandle)

    # Async functions returning plain values are not steerable
    async def plain_function():
        return "plain result"

    result = asyncio.run(plain_function())
    assert not isinstance(result, SteerableToolHandle)


# ────────────────────────────────────────────────────────────────────────────
# Compositional Function Tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compositional_function_can_return_steerable_handle():
    """A compositional function should be able to return a SteerableToolHandle."""
    globals_dict = create_execution_globals()
    globals_dict["ImmediateHandle"] = _ImmediateHandle

    code = """
async def my_steerable_procedure(goal: str):
    return ImmediateHandle(f"handled: {goal}")
"""
    exec(code, globals_dict)

    handle = await globals_dict["my_steerable_procedure"]("Say hello")

    assert isinstance(handle, SteerableToolHandle)
    assert handle.result() == "handled: Say hello"


@pytest.mark.asyncio
async def test_steerable_handle_stop_method():
    """A steerable handle's stop() method should work correctly."""
    globals_dict = create_execution_globals()
    globals_dict["ImmediateHandle"] = _ImmediateHandle

    code = """
async def create_handle():
    return ImmediateHandle("stop-ready")
"""
    exec(code, globals_dict)

    handle = await globals_dict["create_handle"]()
    assert isinstance(handle, SteerableToolHandle)

    await handle.stop("stopping early")

    assert handle.done() is True
    assert handle.result() == "stop-ready"


@pytest.mark.asyncio
async def test_steerable_handle_result_method():
    """A steerable handle's result() method should return the final result."""
    globals_dict = create_execution_globals()
    globals_dict["ImmediateHandle"] = _ImmediateHandle

    code = """
async def create_simple_handle():
    return ImmediateHandle("done")
"""
    exec(code, globals_dict)

    handle = await globals_dict["create_simple_handle"]()
    assert isinstance(handle, SteerableToolHandle)

    result = handle.result()
    assert result == "done"


# ────────────────────────────────────────────────────────────────────────────
# Edge Cases
# ────────────────────────────────────────────────────────────────────────────


def test_non_steerable_function_returns_plain_value():
    """A non-steerable function should return plain values, not handles."""
    globals_dict = create_execution_globals()

    code = """
async def plain_procedure(x: int, y: int) -> int:
    return x + y
"""
    exec(code, globals_dict)

    result = asyncio.run(globals_dict["plain_procedure"](2, 3))

    assert not isinstance(result, SteerableToolHandle)
    assert result == 5


def test_steerable_detection_pattern():
    """The pattern for detecting steerable results should be straightforward."""
    globals_dict = create_execution_globals()

    # This is the pattern a calling actor uses
    def is_steerable(result):
        """Check if a result is a steerable handle."""
        return isinstance(result, globals_dict["SteerableToolHandle"])

    # Test with various values
    assert not is_steerable("string")
    assert not is_steerable(123)
    assert not is_steerable({"dict": "value"})
    assert not is_steerable(None)
    assert not is_steerable([1, 2, 3])


@pytest.mark.asyncio
async def test_compositional_function_with_type_annotation():
    """Compositional function with SteerableToolHandle return type annotation should work."""
    globals_dict = create_execution_globals()
    globals_dict["ImmediateHandle"] = _ImmediateHandle

    code = """
async def typed_steerable_procedure(goal: str) -> SteerableToolHandle:
    return ImmediateHandle(goal)
"""
    exec(code, globals_dict)

    handle = await globals_dict["typed_steerable_procedure"]("Hi")
    assert isinstance(handle, SteerableToolHandle)
    assert handle.result() == "Hi"


# ────────────────────────────────────────────────────────────────────────────
# CodeActActor Steerable Function Test
# ────────────────────────────────────────────────────────────────────────────


# as_shipped: deleted in step 5 (steerable stored functions)
@pytest.mark.usefixtures("as_shipped")
@pytest.mark.asyncio
@pytest.mark.llm_call
async def test_compositional_function_returns_codeact_actor_handle():
    """
    A compositional function that wraps CodeActActor should be detected as steerable.

    This is a critical pattern: storing CodeActActor configurations as compositional
    functions allows for reusable, customizable agent procedures. The returned handle
    must be detected as steerable so the execution layer can forward steering operations.
    """
    globals_dict = create_execution_globals()

    # Define a compositional function that creates and returns a CodeActActor handle
    code = """
async def codeact_procedure(goal: str) -> SteerableToolHandle:
    \"\"\"
    A steerable procedure powered by CodeActActor.

    This pattern allows storing pre-configured CodeActActor setups as
    compositional functions that can be searched, retrieved, and executed
    with full steering support.
    \"\"\"
    from unify.actor.code_act_actor import CodeActActor

    # Create a CodeActActor with custom configuration
    actor = CodeActActor()

    # Start the actor - returns a SteerableToolHandle
    handle = await actor.act(
        request=goal,
        clarification_enabled=False,
    )

    return handle
"""
    exec(code, globals_dict)

    # Call the compositional function
    handle = await globals_dict["codeact_procedure"]("Say hello briefly")

    # Verify it's detected as steerable (the key assertion)
    assert isinstance(
        handle,
        SteerableToolHandle,
    ), "CodeActActor handle should be detected as SteerableToolHandle"

    # Verify handle methods are available (these are forwarded by the calling actor)
    assert hasattr(handle, "interject")
    assert hasattr(handle, "stop")
    assert hasattr(handle, "pause")
    assert hasattr(handle, "resume")
    assert hasattr(handle, "result")

    # Clean up - stop the actor
    await handle.stop("test cleanup")
    try:
        await asyncio.wait_for(handle.result(), timeout=10.0)
    except Exception:
        pass  # May error due to early stop, that's expected
