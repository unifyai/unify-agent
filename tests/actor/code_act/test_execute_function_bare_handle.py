"""
Tests for execute_function bare-handle unwrapping.

When execute_function calls a primitive that returns a SteerableToolHandle
with no meaningful side output (no stdout, no stderr, no error), it should
return the handle directly instead of wrapping it in an ExecutionResult dict.

This ensures the core loop adopts the handle via the bare-handle path
(adopt_nested, no intermediate LLM turn) rather than the composite path
(adopt_multi_nested, wasteful LLM turn that always calls wait()).

``primitives.actor`` is a ``StaticActorRunner`` stand-in installed on the
``ActorEnvironment``'s ``Primitives`` instance, so the handle the primitive
returns is a completed ``SteerableToolHandle`` and no inner actor runs.
"""

from __future__ import annotations

import inspect
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from tests.actor.code_act.helpers import StaticActorRunner
from unify.actor.code_act_actor import CodeActActor
from unify.actor.environments.actor import ActorEnvironment
from unify.actor.execution import ExecutionResult
from unify.common.async_tool_loop import SteerableToolHandle
from unify.common.llm_helpers import method_to_schema
from unify.function_manager.function_manager import FunctionManager
from tests.baked_defaults import as_shipped  # noqa: F401

# as_shipped: deleted in step 5 (execute_function and steerable handles)
pytestmark = [pytest.mark.llm_call, pytest.mark.usefixtures("as_shipped")]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _static_actor_env() -> ActorEnvironment:
    env = ActorEnvironment()
    env.get_instance()._managers["actor"] = StaticActorRunner()
    return env


@pytest_asyncio.fixture
async def execute_function_tool() -> AsyncIterator[Any]:
    """Yield the execute_function tool closure from a CodeActActor."""
    fm = FunctionManager()
    actor = CodeActActor(environments=[_static_actor_env()], function_manager=fm)

    tools = actor.get_tools("act")
    assert (
        "execute_function" in tools
    ), f"execute_function not found in tools: {list(tools.keys())}"
    fn = tools["execute_function"]
    if hasattr(fn, "fn"):
        fn = fn.fn

    try:
        yield fn
    finally:
        await actor.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class _FakeFunctionManager:
    def _get_function_data_by_name(self, *, name: str):
        if name != "stored_report":
            return None
        return {
            "function_id": 12,
            "name": "stored_report",
            "implementation": "def stored_report():\n    return 'report'",
            "dependencies": ["tabulate>=0.9"],
            "is_primitive": False,
        }

    def search_functions(self, **kwargs):
        return {"metadata": []}

    def filter_functions(self, **kwargs):
        return {"metadata": []}

    def list_functions(self, **kwargs):
        return {"metadata": []}

    async def add_functions(self, **kwargs):
        return {"metadata": []}

    async def delete_function(self, **kwargs):
        return {"deleted": True}

    def reconcile_dependencies(self, **kwargs):
        return {}


@pytest.mark.asyncio
async def test_execute_function_does_not_expose_dependencies():
    fm = _FakeFunctionManager()
    actor = CodeActActor(
        function_manager=fm,  # type: ignore[arg-type]
        can_store=False,
    )

    try:
        execute_function = actor.get_tools("act")["execute_function"]
        if hasattr(execute_function, "fn"):
            execute_function = execute_function.fn

        signature = inspect.signature(execute_function)
        schema = method_to_schema(
            execute_function,
            tool_name="execute_function",
            include_class_name=False,
        )
    finally:
        await actor.close()

    assert "dependencies" not in signature.parameters
    assert "dependencies" not in schema["function"]["parameters"]["properties"]


@pytest.mark.asyncio
async def test_execute_function_docstring_carries_call_kwargs_typing_contract():
    """The ``call_kwargs`` exact-typing rule is a mechanism fact asserted on
    the ``execute_function`` docstring, which feeds both the prompt Tools
    section and the JSON tool schema."""
    fm = _FakeFunctionManager()
    actor = CodeActActor(
        function_manager=fm,  # type: ignore[arg-type]
        can_store=False,
    )

    try:
        execute_function = actor.get_tools("act")["execute_function"]
        if hasattr(execute_function, "fn"):
            execute_function = execute_function.fn

        doc = " ".join((inspect.getdoc(execute_function) or "").split())
        schema = method_to_schema(
            execute_function,
            tool_name="execute_function",
            include_class_name=False,
        )
    finally:
        await actor.close()

    assert "Values keep the callee's own types" in doc
    assert '``{"max_results": 5}``' in doc
    assert '``{"max_results": "5"}``' in doc
    assert "fails type validation at the callee" in doc

    description = " ".join(schema["function"]["description"].split())
    assert "Values keep the callee's own types" in description


@pytest.mark.asyncio
async def test_execute_function_ensures_stored_dependencies_before_running(
    monkeypatch,
):
    """The function's recorded dependencies are ensured before the cell runs."""
    from unify import environment

    fm = _FakeFunctionManager()
    actor = CodeActActor(
        function_manager=fm,  # type: ignore[arg-type]
        can_store=False,
    )
    order: list[str] = []
    ensured: list[list[str]] = []

    def _fake_ensure(specifiers):
        ensured.append(list(specifiers))
        order.append("ensure")

    async def _fake_execute(**kwargs):
        order.append("execute")
        return {
            "stdout": [],
            "stderr": [],
            "result": "report",
            "error": None,
            "state_mode": kwargs["state_mode"],
            "session_id": kwargs["session_id"],
            "session_created": False,
            "duration_ms": 0,
        }

    monkeypatch.setattr(environment, "ensure", _fake_ensure)
    actor._session_executor.execute = AsyncMock(side_effect=_fake_execute)  # type: ignore[method-assign]

    try:
        execute_function = actor.get_tools("act")["execute_function"]
        if hasattr(execute_function, "fn"):
            execute_function = execute_function.fn

        result = await execute_function(
            thought="Running the stored report to check its dependency wiring.",
            function_name="stored_report",
            call_kwargs={},
        )
    finally:
        await actor.close()

    assert result.result == "report"
    assert ensured == [["tabulate>=0.9"]]
    assert order == ["ensure", "execute"]


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_execute_function_returns_bare_handle_for_primitive(
    execute_function_tool,
):
    """execute_function should return a bare SteerableToolHandle when
    the primitive produces no side output (stdout/stderr/error)."""
    result = await execute_function_tool(
        thought="Delegating a question to a sub-actor.",
        function_name="primitives.actor.act",
        call_kwargs={"request": "What is 2 + 2?"},
    )

    assert isinstance(
        result,
        SteerableToolHandle,
    ), f"Expected bare SteerableToolHandle, got {type(result).__name__}: {result!r}"
    assert await result.result() == "done: What is 2 + 2?"


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_execute_function_returns_composite_when_side_output_present():
    """execute_function should return the full ExecutionResult when
    the execution produces stdout alongside the handle."""
    # A composed function that prints AND returns a handle — the print
    # output is meaningful intermediate content the LLM should observe.
    fm = FunctionManager()
    fm.add_functions(
        implementations="""
async def delegate_with_log(request: str):
    print("About to delegate...")
    handle = await primitives.actor.act(request=request)
    return handle
""".strip(),
    )

    actor = CodeActActor(environments=[_static_actor_env()], function_manager=fm)

    try:
        tools = actor.get_tools("act")
        fn = tools["execute_function"]
        if hasattr(fn, "fn"):
            fn = fn.fn

        result = await fn(
            thought="Running the composed delegation to capture its stdout.",
            function_name="delegate_with_log",
            call_kwargs={"request": "What is 2 + 2?"},
        )

        # Composed function runs through the sandbox → stdout is captured
        # → result should be an ExecutionResult (not a bare handle).
        assert isinstance(result, (dict, ExecutionResult)), (
            f"Expected dict/ExecutionResult when stdout is present, "
            f"got {type(result).__name__}"
        )

        inner_result = (
            result.get("result")
            if isinstance(result, dict)
            else getattr(result, "result", None)
        )
        assert isinstance(inner_result, SteerableToolHandle), (
            f"Expected inner result to be SteerableToolHandle, "
            f"got {type(inner_result).__name__}"
        )
        assert await inner_result.result() == "done: What is 2 + 2?"
    finally:
        await actor.close()
