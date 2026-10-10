"""
Tests for the state_mode parameter of execute_function.

This tests the full matrix of state modes for stored Python functions:
- stateful: State persists across calls (Jupyter-notebook style)
- read_only: Reads existing state but doesn't persist changes
- stateless: Fresh environment each time (pure function behavior)
"""

from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unify.function_manager.function_manager import FunctionManager


@pytest.fixture(autouse=True)
def _python_in_process(monkeypatch):
    """Python in process (the function manager's in-process mode, for non-actor callers), the mode these tests exercise: stored functions run in this process. With the sandboxed worker nothing runs here (tests/actor/code_act/test_bind_load_confinement.py)."""

    monkeypatch.setattr("unify.actor.execution.worker.enabled", lambda: False)


# ────────────────────────────────────────────────────────────────────────────
# Sample Functions
# ────────────────────────────────────────────────────────────────────────────

SET_VAR_FUNC = """
def set_var(value):
    global my_var
    my_var = value
    return f"Set my_var to {value}"
""".strip()

# The storage check refuses a function that reads a global it cannot
# resolve. A function that reads session state therefore declares it as
# module state it assigns (``global x; x = x``): the check accepts it, and
# the read still fails where the state does not exist.
GET_VAR_FUNC = """
def get_var():
    global my_var
    my_var = my_var
    return my_var
""".strip()

CHECK_VAR_FUNC = """
def check_var():
    global my_var
    try:
        my_var = my_var
    except NameError:
        return "NOT_DEFINED"
    return my_var
""".strip()

SIMPLE_FUNC = """
def simple_func():
    return "hello"
""".strip()

INCREMENT_FUNC = """
def increment():
    global counter
    try:
        counter += 1
    except NameError:
        counter = 1
    return counter
""".strip()

SET_MULTIPLE_VARS_FUNC = """
def set_multiple():
    global int_var, str_var, list_var, dict_var
    int_var = 42
    str_var = "hello"
    list_var = [1, 2, 3]
    dict_var = {"a": 1, "b": 2}
    return "done"
""".strip()

GET_MULTIPLE_VARS_FUNC = """
def get_multiple():
    global int_var, str_var, list_var, dict_var
    int_var, str_var, list_var, dict_var = int_var, str_var, list_var, dict_var
    return {
        "int": int_var,
        "str": str_var,
        "list": list_var,
        "dict": dict_var,
    }
""".strip()

ASYNC_SET_VAR_FUNC = """
async def async_set_var(value):
    global async_var
    async_var = value
    return f"Set async_var to {value}"
""".strip()

ASYNC_GET_VAR_FUNC = """
async def async_get_var():
    global async_var
    async_var = async_var
    return async_var
""".strip()


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
# Stateful Mode Tests (In-Process)
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_in_process_stateful_mode_persists_state(function_manager_factory):
    """Stateful mode should persist variables across function calls (in-process)."""
    fm = function_manager_factory()

    fm.add_functions(implementations=SET_VAR_FUNC)
    fm.add_functions(implementations=GET_VAR_FUNC)

    # Execute set_var in stateful mode
    result1 = await fm.execute_function(
        function_name="set_var",
        call_kwargs={"value": 42},
        state_mode="stateful",
    )
    assert result1["error"] is None, f"Unexpected error: {result1['error']}"
    assert "42" in result1["result"]

    # Execute get_var in stateful mode - should see the persisted value
    result2 = await fm.execute_function(
        function_name="get_var",
        state_mode="stateful",
    )
    assert result2["error"] is None, f"Unexpected error: {result2['error']}"
    assert result2["result"] == 42


@_handle_project
@pytest.mark.asyncio
async def test_in_process_stateful_mode_increments(function_manager_factory):
    """Stateful mode should allow incremental state changes (in-process)."""
    fm = function_manager_factory()

    fm.add_functions(implementations=INCREMENT_FUNC)

    # Call increment multiple times - each should see previous value
    for expected in [1, 2, 3, 4, 5]:
        result = await fm.execute_function(
            function_name="increment",
            state_mode="stateful",
        )
        assert result["error"] is None
        assert result["result"] == expected


@_handle_project
@pytest.mark.asyncio
async def test_in_process_stateful_mode_complex_types(function_manager_factory):
    """Stateful mode should persist complex types (lists, dicts)."""
    fm = function_manager_factory()

    fm.add_functions(implementations=SET_MULTIPLE_VARS_FUNC)
    fm.add_functions(implementations=GET_MULTIPLE_VARS_FUNC)

    # Set multiple variables
    result1 = await fm.execute_function(
        function_name="set_multiple",
        state_mode="stateful",
    )
    assert result1["error"] is None

    # Get them back
    result2 = await fm.execute_function(
        function_name="get_multiple",
        state_mode="stateful",
    )
    assert result2["error"] is None
    data = result2["result"]
    assert data["int"] == 42
    assert data["str"] == "hello"
    assert data["list"] == [1, 2, 3]
    assert data["dict"] == {"a": 1, "b": 2}


@_handle_project
@pytest.mark.asyncio
async def test_in_process_stateful_mode_async_functions(function_manager_factory):
    """Stateful mode should work with async functions (in-process)."""
    fm = function_manager_factory()

    fm.add_functions(implementations=ASYNC_SET_VAR_FUNC)
    fm.add_functions(implementations=ASYNC_GET_VAR_FUNC)

    # Set value using async function
    result1 = await fm.execute_function(
        function_name="async_set_var",
        call_kwargs={"value": "async_value"},
        state_mode="stateful",
    )
    assert result1["error"] is None

    # Get value using async function
    result2 = await fm.execute_function(
        function_name="async_get_var",
        state_mode="stateful",
    )
    assert result2["error"] is None
    assert result2["result"] == "async_value"


@_handle_project
@pytest.mark.asyncio
async def test_in_process_stateful_mode_multiple_sessions(function_manager_factory):
    """Different session_ids should have independent state."""
    fm = function_manager_factory()

    fm.add_functions(implementations=SET_VAR_FUNC)
    fm.add_functions(implementations=GET_VAR_FUNC)

    # Set value 100 in session 0
    await fm.execute_function(
        function_name="set_var",
        call_kwargs={"value": 100},
        state_mode="stateful",
        session_id=0,
    )

    # Set value 200 in session 1
    await fm.execute_function(
        function_name="set_var",
        call_kwargs={"value": 200},
        state_mode="stateful",
        session_id=1,
    )

    # Verify session 0 has 100
    result0 = await fm.execute_function(
        function_name="get_var",
        state_mode="stateful",
        session_id=0,
    )
    assert result0["error"] is None
    assert result0["result"] == 100

    # Verify session 1 has 200
    result1 = await fm.execute_function(
        function_name="get_var",
        state_mode="stateful",
        session_id=1,
    )
    assert result1["error"] is None
    assert result1["result"] == 200


# ────────────────────────────────────────────────────────────────────────────
# Read-Only Mode Tests (In-Process)
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_in_process_read_only_mode_reads_state(function_manager_factory):
    """Read-only mode should be able to read existing state."""
    fm = function_manager_factory()

    fm.add_functions(implementations=SET_VAR_FUNC)
    fm.add_functions(implementations=GET_VAR_FUNC)

    # First, set state using stateful mode
    await fm.execute_function(
        function_name="set_var",
        call_kwargs={"value": 100},
        state_mode="stateful",
    )

    # Now read state using read_only mode - should see the value
    result = await fm.execute_function(
        function_name="get_var",
        state_mode="read_only",
    )
    assert result["error"] is None, f"Unexpected error: {result['error']}"
    assert result["result"] == 100


@_handle_project
@pytest.mark.asyncio
async def test_in_process_read_only_mode_does_not_persist_changes(
    function_manager_factory,
):
    """Read-only mode should not persist changes to state."""
    fm = function_manager_factory()

    fm.add_functions(implementations=SET_VAR_FUNC)
    fm.add_functions(implementations=GET_VAR_FUNC)

    # Set initial state to 100
    await fm.execute_function(
        function_name="set_var",
        call_kwargs={"value": 100},
        state_mode="stateful",
    )

    # Try to change state in read_only mode (should execute but not persist)
    result_modify = await fm.execute_function(
        function_name="set_var",
        call_kwargs={"value": 999},
        state_mode="read_only",
    )
    assert result_modify["error"] is None
    assert "999" in result_modify["result"]

    # Verify original state is unchanged (still 100, not 999)
    result_check = await fm.execute_function(
        function_name="get_var",
        state_mode="stateful",
    )
    assert result_check["error"] is None
    assert result_check["result"] == 100  # Original value preserved


@_handle_project
@pytest.mark.asyncio
async def test_in_process_read_only_mode_empty_session(function_manager_factory):
    """Read-only mode with no existing session should work (fresh globals)."""
    fm = function_manager_factory()

    fm.add_functions(implementations=CHECK_VAR_FUNC)

    # Read-only mode with no pre-existing state should return NOT_DEFINED
    result = await fm.execute_function(
        function_name="check_var",
        state_mode="read_only",
        session_id=999,  # Non-existent session
    )
    assert result["error"] is None
    assert result["result"] == "NOT_DEFINED"


@_handle_project
@pytest.mark.asyncio
async def test_in_process_read_only_mode_uses_correct_session(function_manager_factory):
    """Read-only mode should read from the correct session_id."""
    fm = function_manager_factory()

    fm.add_functions(implementations=SET_VAR_FUNC)
    fm.add_functions(implementations=GET_VAR_FUNC)

    # Set different values in different sessions
    await fm.execute_function(
        function_name="set_var",
        call_kwargs={"value": "session_0_value"},
        state_mode="stateful",
        session_id=0,
    )
    await fm.execute_function(
        function_name="set_var",
        call_kwargs={"value": "session_1_value"},
        state_mode="stateful",
        session_id=1,
    )

    # Read-only from session 0
    result0 = await fm.execute_function(
        function_name="get_var",
        state_mode="read_only",
        session_id=0,
    )
    assert result0["error"] is None
    assert result0["result"] == "session_0_value"

    # Read-only from session 1
    result1 = await fm.execute_function(
        function_name="get_var",
        state_mode="read_only",
        session_id=1,
    )
    assert result1["error"] is None
    assert result1["result"] == "session_1_value"


# ────────────────────────────────────────────────────────────────────────────
# Stateless Mode Tests (In-Process) - Verify Backward Compatibility
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_in_process_stateless_mode_fresh_each_time(function_manager_factory):
    """Stateless mode should not inherit any state from previous calls."""
    fm = function_manager_factory()

    fm.add_functions(implementations=SET_VAR_FUNC)
    fm.add_functions(implementations=CHECK_VAR_FUNC)

    # Set state in stateful mode
    await fm.execute_function(
        function_name="set_var",
        call_kwargs={"value": 42},
        state_mode="stateful",
    )

    # Check in stateless mode - should NOT see the variable
    result = await fm.execute_function(
        function_name="check_var",
        state_mode="stateless",
    )
    assert result["error"] is None
    assert result["result"] == "NOT_DEFINED"


@_handle_project
@pytest.mark.asyncio
async def test_in_process_stateless_mode_does_not_affect_stateful_state(
    function_manager_factory,
):
    """Stateless execution should not affect stateful session state."""
    fm = function_manager_factory()

    fm.add_functions(implementations=SET_VAR_FUNC)
    fm.add_functions(implementations=GET_VAR_FUNC)

    # Set initial state to 42 (stateful)
    await fm.execute_function(
        function_name="set_var",
        call_kwargs={"value": 42},
        state_mode="stateful",
    )

    # Execute something in stateless mode that tries to change state
    await fm.execute_function(
        function_name="set_var",
        call_kwargs={"value": 999},
        state_mode="stateless",
    )

    # Stateful state should be unchanged
    result = await fm.execute_function(
        function_name="get_var",
        state_mode="stateful",
    )
    assert result["error"] is None
    assert result["result"] == 42  # Still 42, not 999


@_handle_project
@pytest.mark.asyncio
async def test_in_process_default_mode_is_stateless(function_manager_factory):
    """Default mode should be stateless for backward compatibility."""
    fm = function_manager_factory()

    fm.add_functions(implementations=SET_VAR_FUNC)
    fm.add_functions(implementations=CHECK_VAR_FUNC)

    # Set value without specifying state_mode (should be stateless)
    await fm.execute_function(
        function_name="set_var",
        call_kwargs={"value": 42},
        # state_mode not specified - defaults to "stateless"
    )

    # Check without specifying state_mode - should be NOT_DEFINED
    result = await fm.execute_function(
        function_name="check_var",
        # state_mode not specified - defaults to "stateless"
    )
    assert result["error"] is None
    assert result["result"] == "NOT_DEFINED"


# ────────────────────────────────────────────────────────────────────────────
# Session Management Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_clear_in_process_sessions_all(function_manager_factory):
    """clear_in_process_sessions() should clear all sessions."""
    fm = function_manager_factory()

    fm.add_functions(implementations=SET_VAR_FUNC)
    fm.add_functions(implementations=CHECK_VAR_FUNC)

    # Create state in multiple sessions
    await fm.execute_function(
        function_name="set_var",
        call_kwargs={"value": 1},
        state_mode="stateful",
        session_id=0,
    )
    await fm.execute_function(
        function_name="set_var",
        call_kwargs={"value": 2},
        state_mode="stateful",
        session_id=1,
    )

    # Clear all sessions
    fm.clear_in_process_sessions()

    # Both sessions should now show NOT_DEFINED
    for sid in [0, 1]:
        result = await fm.execute_function(
            function_name="check_var",
            state_mode="stateful",
            session_id=sid,
        )
        assert result["error"] is None
        assert result["result"] == "NOT_DEFINED"


@_handle_project
@pytest.mark.asyncio
async def test_clear_in_process_sessions_specific(function_manager_factory):
    """clear_in_process_sessions(session_id) should clear only that session."""
    fm = function_manager_factory()

    fm.add_functions(implementations=SET_VAR_FUNC)
    fm.add_functions(implementations=GET_VAR_FUNC)
    fm.add_functions(implementations=CHECK_VAR_FUNC)

    # Create state in multiple sessions
    await fm.execute_function(
        function_name="set_var",
        call_kwargs={"value": 100},
        state_mode="stateful",
        session_id=0,
    )
    await fm.execute_function(
        function_name="set_var",
        call_kwargs={"value": 200},
        state_mode="stateful",
        session_id=1,
    )

    # Clear only session 0
    fm.clear_in_process_sessions(session_id=0)

    # Session 0 should show NOT_DEFINED
    result0 = await fm.execute_function(
        function_name="check_var",
        state_mode="stateful",
        session_id=0,
    )
    assert result0["error"] is None
    assert result0["result"] == "NOT_DEFINED"

    # Session 1 should still have its value
    result1 = await fm.execute_function(
        function_name="get_var",
        state_mode="stateful",
        session_id=1,
    )
    assert result1["error"] is None
    assert result1["result"] == 200
