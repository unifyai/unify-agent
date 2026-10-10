"""
Tests for execute_code output propagation across primitive handles.

These tests verify that when execute_code runs code that calls a primitive
(``primitives.actor.act``) and awaits the handle's result, the stdout and
result fields are properly captured and returned to the caller.

The primitive is a ``StaticActorRunner`` stand-in installed on the
``ActorEnvironment``'s ``Primitives`` instance, so no LLM is involved: the
subject is the sandbox's capture of values that come back through a handle.

The bug this catches: execute_code running `print(await handle.result())` but
returning empty stdout because the nested result wasn't captured.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest
import pytest_asyncio

from tests.actor.code_act.helpers import StaticActorRunner
from unify.actor.code_act_actor import CodeActActor
from unify.actor.environments.actor import ActorEnvironment
from unify.actor.execution import ExecutionResult, parts_to_text

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _answer_for(request: str) -> str:
    return f"delegated answer for: {request}"


@pytest_asyncio.fixture
async def actor_with_primitives() -> (
    AsyncIterator[tuple[CodeActActor, StaticActorRunner]]
):
    """A CodeActActor whose ``primitives.actor`` is a static stand-in runner."""
    env = ActorEnvironment()
    runner = StaticActorRunner(_answer_for)
    env.get_instance()._managers["actor"] = runner
    actor = CodeActActor(environments=[env], function_manager=None)

    # Strip FunctionManager tools to focus on primitives
    act_tools = actor.get_tools("act")
    actor.add_tools("act", {"execute_code": act_tools["execute_code"]})

    try:
        yield actor, runner
    finally:
        await actor.close()


@pytest_asyncio.fixture
async def execute_code_tool(actor_with_primitives) -> tuple[Any, StaticActorRunner]:
    """Get the execute_code tool directly for isolated testing."""
    actor, runner = actor_with_primitives
    tools = actor.get_tools("act")
    return tools["execute_code"], runner


# ---------------------------------------------------------------------------
# Helper functions for handling ExecutionResult vs dict
# ---------------------------------------------------------------------------


def get_output_field(out: Any, field: str, default: Any = None) -> Any:
    """Get a field from execute_code output, handling both dict and ExecutionResult."""
    if isinstance(out, dict):
        return out.get(field, default)
    elif hasattr(out, field):
        return getattr(out, field, default)
    return default


def get_stdout_text(out: Any) -> str:
    """Extract stdout text from execute_code output."""
    stdout = get_output_field(out, "stdout", [])
    if isinstance(stdout, list):
        return parts_to_text(stdout)
    return str(stdout) if stdout else ""


def get_error(out: Any) -> str | None:
    """Get error from execute_code output."""
    return get_output_field(out, "error", None)


def get_result(out: Any) -> Any:
    """Get result from execute_code output."""
    return get_output_field(out, "result", None)


# ---------------------------------------------------------------------------
# Test: query_llm structured output inside control flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_execute_code_can_branch_on_query_llm_structured_output(
    execute_code_tool,
    monkeypatch: pytest.MonkeyPatch,
):
    """execute_code can use typed semantic judgments in symbolic control flow."""
    execute_code, _runner = execute_code_tool

    sent: list = []

    async def fake_query_llm(prompt: str, *, response_format=None, **kwargs):
        assert "Classify" in prompt
        sent.append(response_format)
        answer = {"category": "billing", "needs_reply": True}
        # As query_llm does: a schema dict gets the parsed JSON, a class an
        # instance. With Python in the sandboxed worker (the default) a
        # cell's class crosses as its schema and is validated in the worker.
        if isinstance(response_format, dict):
            return answer
        return response_format(**answer)

    import unify.common.reasoning as reasoning_module

    monkeypatch.setattr(reasoning_module, "query_llm", fake_query_llm)

    code = """
from pydantic import BaseModel

class Decision(BaseModel):
    category: str
    needs_reply: bool

Decision.model_rebuild()

decision = await query_llm(
    "Classify this email: Please approve the renewal quote.",
    response_format=Decision,
)
"queue_reply" if decision.category == "billing" and decision.needs_reply else "archive"
"""

    out = await execute_code(
        "test query_llm structured output",
        code,
        state_mode="stateless",
    )

    assert get_error(out) is None
    assert get_result(out) == "queue_reply"
    (response_format,) = sent
    if isinstance(response_format, dict):
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["name"] == "Decision"
        schema = response_format["json_schema"]["schema"]
        assert set(schema["properties"]) == {"category", "needs_reply"}


# ---------------------------------------------------------------------------
# Test: Basic stdout capture from primitives.actor.act().result()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_execute_code_captures_stdout_from_primitive_result(
    execute_code_tool,
):
    """
    Verifies stdout is captured when code prints a primitive result.

    - Code runs: handle = await primitives.actor.act(...); result = await handle.result(); print(result)
    - Expected: stdout contains the answer the handle carried
    """
    execute_code, runner = execute_code_tool

    code = """
handle = await primitives.actor.act("How many records are in the system?")
result = await handle.result()
print(f"ANSWER: {result}")
"""

    out = await execute_code(
        "test primitive stdout",
        code,
        state_mode="stateless",
    )

    error = get_error(out)
    assert error is None, f"Execution failed: {error}"

    stdout_text = get_stdout_text(out)
    assert stdout_text.strip() == (
        "ANSWER: " + _answer_for("How many records are in the system?")
    ), f"Unexpected stdout: {stdout_text!r}"
    assert [c["request"] for c in runner.act_calls] == [
        "How many records are in the system?",
    ]


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_execute_code_result_field_populated_from_primitive(
    execute_code_tool,
):
    """
    Verifies the 'result' field is populated when code returns a primitive result.

    The execute_code docstring says:
    - result: The evaluated result of the last expression (Any), or None.

    When code ends with `await handle.result()`, that should be captured.
    """
    execute_code, _runner = execute_code_tool

    code = """
handle = await primitives.actor.act("List all records")
await handle.result()
"""

    out = await execute_code(
        "test result field",
        code,
        state_mode="stateless",
    )

    error = get_error(out)
    assert error is None, f"Execution failed: {error}"
    assert get_result(out) == _answer_for("List all records")


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_execute_code_captures_stdout_with_multiple_prints(
    execute_code_tool,
):
    """Verifies multiple print statements are all captured in stdout, in order."""
    execute_code, _runner = execute_code_tool

    code = """
print("BEFORE")
handle = await primitives.actor.act("How many records do we have?")
result = await handle.result()
print(f"RESULT: {result}")
print("AFTER")
"""

    out = await execute_code(
        "test multiple prints",
        code,
        state_mode="stateless",
    )

    error = get_error(out)
    assert error is None, f"Execution failed: {error}"

    lines = get_stdout_text(out).splitlines()
    assert lines == [
        "BEFORE",
        "RESULT: " + _answer_for("How many records do we have?"),
        "AFTER",
    ]


# ---------------------------------------------------------------------------
# Test: Sequential and concurrent primitive calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_execute_code_sequential_primitive_calls(
    execute_code_tool,
):
    """Verifies two sequential primitive calls both have their results captured."""
    execute_code, runner = execute_code_tool

    code = """
h1 = await primitives.actor.act("How many records exist?")
r1 = await h1.result()
print(f"FIRST: {r1}")

h2 = await primitives.actor.act("List all record names")
r2 = await h2.result()
print(f"SECOND: {r2}")
"""

    out = await execute_code(
        "test sequential calls",
        code,
        state_mode="stateless",
    )

    error = get_error(out)
    assert error is None, f"Execution failed: {error}"

    stdout_text = get_stdout_text(out)
    assert "FIRST: " + _answer_for("How many records exist?") in stdout_text
    assert "SECOND: " + _answer_for("List all record names") in stdout_text
    assert [c["request"] for c in runner.act_calls] == [
        "How many records exist?",
        "List all record names",
    ]


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_execute_code_concurrent_primitive_calls(
    execute_code_tool,
):
    """Verifies concurrent primitive calls via asyncio.gather work correctly."""
    execute_code, _runner = execute_code_tool

    code = """
import asyncio

h1 = await primitives.actor.act("Question 1: How many records?")
h2 = await primitives.actor.act("Question 2: List record names")

# Wait for both concurrently
r1, r2 = await asyncio.gather(h1.result(), h2.result())

print(f"RESULT1: {r1}")
print(f"RESULT2: {r2}")
"""

    out = await execute_code(
        "test concurrent calls",
        code,
        state_mode="stateless",
    )

    error = get_error(out)
    assert error is None, f"Execution failed: {error}"

    stdout_text = get_stdout_text(out)
    assert "RESULT1: " + _answer_for("Question 1: How many records?") in stdout_text
    assert "RESULT2: " + _answer_for("Question 2: List record names") in stdout_text


# ---------------------------------------------------------------------------
# Test: Error propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_execute_code_primitive_error_propagates(
    execute_code_tool,
):
    """Verifies that errors from primitive access are properly captured."""
    execute_code, _runner = execute_code_tool

    # This code intentionally accesses a non-existent primitive namespace
    code = """
try:
    handle = await primitives.nonexistent_manager.act("test")
    result = await handle.result()
    print(f"RESULT: {result}")
except AttributeError as e:
    print(f"ERROR: {e}")
"""

    out = await execute_code(
        "test error handling",
        code,
        state_mode="stateless",
    )

    # The code has a try/except, so it shouldn't have an error
    error = get_error(out)
    assert error is None, f"Unexpected execution error: {error}"

    stdout_text = get_stdout_text(out)

    # Should have caught and printed the error
    assert "ERROR:" in stdout_text, f"Missing error output in stdout: {stdout_text!r}"


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_execute_code_unhandled_error_captured(
    execute_code_tool,
):
    """Verifies unhandled exceptions are captured in the error field."""
    execute_code, _runner = execute_code_tool

    code = """
# This will raise an AttributeError
handle = await primitives.nonexistent_manager.act("test")
"""

    out = await execute_code(
        "test unhandled error",
        code,
        state_mode="stateless",
    )

    # Should have an error captured
    error = get_error(out)
    assert (
        error is not None
    ), "Expected error to be captured for invalid primitive access"
    assert "AttributeError" in str(
        error,
    ), f"Expected AttributeError in error field: {error}"


# ---------------------------------------------------------------------------
# Test: Stdout isolation (only the cell's own prints reach stdout)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_execute_code_stdout_isolation_from_primitive_calls(
    execute_code_tool,
):
    """
    Verifies that only explicit print() calls in the cell reach stdout.

    Awaiting a primitive handle must not inject anything of its own into the
    sandbox's captured output.
    """
    execute_code, _runner = execute_code_tool

    code = """
print("START")
handle = await primitives.actor.act("List records")
result = await handle.result()
print("END")
"""

    out = await execute_code(
        "test stdout isolation",
        code,
        state_mode="stateless",
    )

    error = get_error(out)
    assert error is None, f"Execution failed: {error}"

    assert get_stdout_text(out).splitlines() == ["START", "END"]


# ---------------------------------------------------------------------------
# Test: Stateful session preserves primitives across calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_execute_code_stateful_session_with_primitives(
    execute_code_tool,
):
    """Verifies stateful sessions maintain access to primitives across calls."""
    execute_code, _runner = execute_code_tool

    # First call: store result in a variable
    out1 = await execute_code(
        "first call",
        """
handle = await primitives.actor.act("How many records?")
stored_result = await handle.result()
print(f"STORED: {stored_result}")
""",
        state_mode="stateful",
        session_id=0,
    )

    error1 = get_error(out1)
    assert error1 is None, f"First call failed: {error1}"

    # Second call: access the stored variable
    out2 = await execute_code(
        "second call",
        """
print(f"RETRIEVED: {stored_result}")
""",
        state_mode="stateful",
        session_id=0,
    )

    error2 = get_error(out2)
    assert error2 is None, f"Second call failed: {error2}"

    assert get_stdout_text(out2).strip() == (
        "RETRIEVED: " + _answer_for("How many records?")
    )


# ---------------------------------------------------------------------------
# Test: ExecutionResult formatting for LLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_execution_result_includes_stdout_in_llm_content(
    execute_code_tool,
):
    """
    Verifies that ExecutionResult.to_llm_content() includes stdout when present.

    This is critical for the outer LLM to see the primitive results.
    """
    execute_code, _runner = execute_code_tool

    code = """
handle = await primitives.actor.act("List records")
result = await handle.result()
print(f"ANSWER: {result}")
"""

    out = await execute_code(
        "test llm content",
        code,
        state_mode="stateless",
    )

    error = get_error(out)
    assert error is None, f"Execution failed: {error}"

    exec_result = out if isinstance(out, ExecutionResult) else ExecutionResult(**out)
    llm_content = exec_result.to_llm_content()

    # Should have content blocks
    assert isinstance(llm_content, list), f"Expected list, got {type(llm_content)}"
    assert len(llm_content) > 0, "Expected at least one content block"

    # Combine all text content
    all_text = " ".join(
        block.get("text", "") for block in llm_content if block.get("type") == "text"
    )

    assert (
        "ANSWER: " + _answer_for("List records") in all_text
    ), f"Expected the answer in LLM content, got: {all_text!r}"
