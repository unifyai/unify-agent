"""
Tests for _InProcessFunctionProxy state mode API (.stateless(), .read_only()).

These tests verify that the proxy-based API for controlling execution state
works correctly when calling stored functions from CodeActActor.

Covers:
1. Default stateful behavior (state persists in shared namespace)
2. .stateless() mode (fresh environment via execute_function)
3. .read_only() mode (reads session state but doesn't persist changes)
4. Integration with return_callable=True
"""

from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unify.function_manager.function_manager import (
    FunctionManager,
    _InProcessFunctionProxy,
)
from unify.settings import SETTINGS


@pytest.fixture(autouse=True)
def _python_in_process(monkeypatch):
    """Python in process (``UNIFY_WORKSPACE_PYTHON`` empty): the in-process proxy exists only where stored functions are loaded into this process. With the
    sandboxed worker a read binds them by source and nothing runs here (tests/actor/code_act/test_bind_load_confinement.py).
    """
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "")


# ────────────────────────────────────────────────────────────────────────────
# Sample Functions
# ────────────────────────────────────────────────────────────────────────────

SET_VAR_FUNC = """
async def set_var(value):
    global my_var
    my_var = value
    return f"Set my_var to {value}"
""".strip()

# The storage check refuses a function that reads a global it cannot
# resolve. A function that reads session state therefore declares it as
# module state it assigns (``global x; x = x``): the check accepts it, and
# the read still fails where the state does not exist.
GET_VAR_FUNC = """
async def get_var():
    global my_var
    my_var = my_var
    return my_var
""".strip()

CHECK_VAR_FUNC = """
async def check_var():
    global my_var
    try:
        my_var = my_var
    except NameError:
        return "NOT_DEFINED"
    return my_var
""".strip()

INCREMENT_FUNC = """
async def increment():
    global counter
    try:
        counter
    except NameError:
        counter = 0
    counter += 1
    return counter
""".strip()

SYNC_ADD_FUNC = """
def add(a, b):
    return a + b
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
# Helper Functions
# ────────────────────────────────────────────────────────────────────────────


def _get_callable_via_return_callable(
    fm: FunctionManager,
    func_name: str,
    namespace: dict,
) -> _InProcessFunctionProxy:
    """Get a function proxy via list_functions with return_callable=True."""
    result = fm.list_functions(
        _return_callable=True,
        _namespace=namespace,
    )
    return result[func_name]


# ────────────────────────────────────────────────────────────────────────────
# 1. Basic Proxy Creation Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_in_process_proxy_has_correct_type(function_manager_factory):
    """return_callable=True should return an _InProcessFunctionProxy."""
    fm = function_manager_factory()
    fm.add_functions(implementations=SET_VAR_FUNC)

    namespace = {}
    set_var = _get_callable_via_return_callable(fm, "set_var", namespace)

    assert isinstance(set_var, _InProcessFunctionProxy)
    assert set_var.__name__ == "set_var"


@_handle_project
@pytest.mark.asyncio
async def test_in_process_proxy_has_state_mode_methods(function_manager_factory):
    """Proxy should have .stateless() and .read_only() methods."""
    fm = function_manager_factory()
    fm.add_functions(implementations=SET_VAR_FUNC)

    namespace = {}
    set_var = _get_callable_via_return_callable(fm, "set_var", namespace)

    assert hasattr(set_var, "stateless")
    assert hasattr(set_var, "read_only")
    assert callable(set_var.stateless)
    assert callable(set_var.read_only)


# ────────────────────────────────────────────────────────────────────────────
# 2. Default Stateful Mode Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_in_process_proxy_stateful_default_persists_state(
    function_manager_factory,
):
    """Default await func() should persist state in shared namespace."""
    fm = function_manager_factory()
    fm.add_functions(implementations=SET_VAR_FUNC)
    fm.add_functions(implementations=GET_VAR_FUNC)

    namespace = {}
    set_var = _get_callable_via_return_callable(fm, "set_var", namespace)
    get_var = _get_callable_via_return_callable(fm, "get_var", namespace)

    # Set value via default (stateful) call
    result1 = await set_var(value=42)
    assert "42" in result1

    # Get value via default (stateful) call - should see persisted state
    result2 = await get_var()
    assert result2 == 42


@_handle_project
@pytest.mark.asyncio
async def test_in_process_proxy_stateful_counter_increments(function_manager_factory):
    """Counter should increment across stateful calls in shared namespace."""
    fm = function_manager_factory()
    fm.add_functions(implementations=INCREMENT_FUNC)

    namespace = {}
    increment = _get_callable_via_return_callable(fm, "increment", namespace)

    # Multiple stateful calls should see incrementing counter
    assert await increment() == 1
    assert await increment() == 2
    assert await increment() == 3


@_handle_project
@pytest.mark.asyncio
async def test_in_process_proxy_sync_function_works(function_manager_factory):
    """Sync functions should work correctly via proxy."""
    fm = function_manager_factory()
    fm.add_functions(implementations=SYNC_ADD_FUNC)

    namespace = {}
    add = _get_callable_via_return_callable(fm, "add", namespace)

    result = await add(a=2, b=3)
    assert result == 5


# ────────────────────────────────────────────────────────────────────────────
# 3. Stateless Mode Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_in_process_proxy_stateless_provides_isolation(function_manager_factory):
    """await func.stateless() should not inherit shared namespace state."""
    fm = function_manager_factory()
    fm.add_functions(implementations=SET_VAR_FUNC)
    fm.add_functions(implementations=CHECK_VAR_FUNC)
    fm.add_functions(implementations=GET_VAR_FUNC)

    namespace = {}
    set_var = _get_callable_via_return_callable(fm, "set_var", namespace)
    check_var = _get_callable_via_return_callable(fm, "check_var", namespace)
    get_var = _get_callable_via_return_callable(fm, "get_var", namespace)

    # Set state via stateful call (persists in shared namespace)
    await set_var(value=100)

    # Stateless call should NOT see the namespace state
    result = await check_var.stateless()
    assert result == "NOT_DEFINED"

    # Stateful call should still see original state
    result = await get_var()
    assert result == 100


@_handle_project
@pytest.mark.asyncio
async def test_in_process_proxy_stateless_does_not_affect_namespace(
    function_manager_factory,
):
    """Stateless execution should not modify shared namespace state."""
    fm = function_manager_factory()
    fm.add_functions(implementations=SET_VAR_FUNC)
    fm.add_functions(implementations=GET_VAR_FUNC)

    namespace = {}
    set_var = _get_callable_via_return_callable(fm, "set_var", namespace)
    get_var = _get_callable_via_return_callable(fm, "get_var", namespace)

    # Set initial state to 42 (stateful)
    await set_var(value=42)

    # Try to change state via stateless call
    await set_var.stateless(value=999)

    # Stateful state should be unchanged
    result = await get_var()
    assert result == 42  # Still 42, not 999


@_handle_project
@pytest.mark.asyncio
async def test_in_process_proxy_stateless_counter_always_fresh(
    function_manager_factory,
):
    """Stateless calls should always start fresh (counter = 1)."""
    fm = function_manager_factory()
    fm.add_functions(implementations=INCREMENT_FUNC)

    namespace = {}
    increment = _get_callable_via_return_callable(fm, "increment", namespace)

    # Stateful calls increment in namespace
    assert await increment() == 1
    assert await increment() == 2

    # Stateless call starts fresh
    assert await increment.stateless() == 1

    # Stateful state unchanged (continues from 2)
    assert await increment() == 3


# ────────────────────────────────────────────────────────────────────────────
# 4. Read-Only Mode Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_in_process_proxy_read_only_reads_session_state(function_manager_factory):
    """await func.read_only() should read from in-process session state."""
    fm = function_manager_factory()
    fm.add_functions(implementations=SET_VAR_FUNC)
    fm.add_functions(implementations=GET_VAR_FUNC)

    namespace = {}
    set_var = _get_callable_via_return_callable(fm, "set_var", namespace)
    get_var = _get_callable_via_return_callable(fm, "get_var", namespace)

    # First, establish state via execute_function with stateful mode
    # (This populates the FunctionManager's _in_process_sessions)
    await fm.execute_function(
        function_name="set_var",
        call_kwargs={"value": 100},
        state_mode="stateful",
        session_id=0,
    )

    # Read-only call should see the session state
    result = await get_var.read_only()
    assert result == 100


@_handle_project
@pytest.mark.asyncio
async def test_in_process_proxy_read_only_does_not_persist_changes(
    function_manager_factory,
):
    """await func.read_only() should not persist changes to session state."""
    fm = function_manager_factory()
    fm.add_functions(implementations=SET_VAR_FUNC)
    fm.add_functions(implementations=GET_VAR_FUNC)

    namespace = {}
    set_var = _get_callable_via_return_callable(fm, "set_var", namespace)
    get_var = _get_callable_via_return_callable(fm, "get_var", namespace)

    # Establish session state via execute_function
    await fm.execute_function(
        function_name="set_var",
        call_kwargs={"value": 100},
        state_mode="stateful",
        session_id=0,
    )

    # Modify via read_only (should execute but not persist)
    result_modify = await set_var.read_only(value=999)
    assert "999" in result_modify

    # Session state should be unchanged
    result_check = await fm.execute_function(
        function_name="get_var",
        state_mode="stateful",
        session_id=0,
    )
    assert result_check["result"] == 100  # Still 100, not 999


@_handle_project
@pytest.mark.asyncio
async def test_in_process_proxy_read_only_empty_session(function_manager_factory):
    """Read-only with no session state should work (empty session)."""
    fm = function_manager_factory()
    fm.add_functions(implementations=CHECK_VAR_FUNC)

    namespace = {}
    check_var = _get_callable_via_return_callable(fm, "check_var", namespace)

    # Read-only with no pre-existing session state
    result = await check_var.read_only()
    assert result == "NOT_DEFINED"


# ────────────────────────────────────────────────────────────────────────────
# 5. Mixed Mode Sequence Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_in_process_proxy_mixed_mode_sequence(function_manager_factory):
    """Complex sequence mixing stateful, stateless, and read_only modes."""
    fm = function_manager_factory()
    fm.add_functions(implementations=SET_VAR_FUNC)
    fm.add_functions(implementations=GET_VAR_FUNC)
    fm.add_functions(implementations=CHECK_VAR_FUNC)

    namespace = {}
    set_var = _get_callable_via_return_callable(fm, "set_var", namespace)
    get_var = _get_callable_via_return_callable(fm, "get_var", namespace)
    check_var = _get_callable_via_return_callable(fm, "check_var", namespace)

    # Step 1: Set state in shared namespace (stateful default)
    await set_var(value=10)

    # Step 2: Read state (stateful) - should see 10
    assert await get_var() == 10

    # Step 3: Stateless check - should NOT see namespace state
    assert await check_var.stateless() == "NOT_DEFINED"

    # Step 4: Also set up session state for read_only to use
    await fm.execute_function(
        function_name="set_var",
        call_kwargs={"value": 50},
        state_mode="stateful",
        session_id=0,
    )

    # Step 5: Read-only should see session state (50), not namespace state (10)
    assert await get_var.read_only() == 50

    # Step 6: Read-only modification should not persist
    await set_var.read_only(value=999)

    # Step 7: Session state unchanged
    result = await fm.execute_function(
        function_name="get_var",
        state_mode="stateful",
        session_id=0,
    )
    assert result["result"] == 50

    # Step 8: Namespace state also unchanged
    assert await get_var() == 10


@_handle_project
@pytest.mark.asyncio
async def test_in_process_proxy_stateless_counter_mixed_with_stateful(
    function_manager_factory,
):
    """Stateless calls interleaved with stateful shouldn't affect each other."""
    fm = function_manager_factory()
    fm.add_functions(implementations=INCREMENT_FUNC)

    namespace = {}
    increment = _get_callable_via_return_callable(fm, "increment", namespace)

    # Stateful builds up counter in namespace
    assert await increment() == 1

    # Stateless is isolated
    assert await increment.stateless() == 1

    # Stateful continues from namespace
    assert await increment() == 2

    # Multiple stateless calls, each fresh
    assert await increment.stateless() == 1
    assert await increment.stateless() == 1

    # Stateful still continues
    assert await increment() == 3


# ────────────────────────────────────────────────────────────────────────────
# 6. Error Handling Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_in_process_proxy_stateless_handles_errors(function_manager_factory):
    """Errors in stateless execution should be propagated correctly."""
    fm = function_manager_factory()

    error_func = """
async def raise_error():
    raise ValueError("Intentional error")
""".strip()
    fm.add_functions(implementations=error_func)

    namespace = {}
    raise_error = _get_callable_via_return_callable(fm, "raise_error", namespace)

    with pytest.raises(RuntimeError, match="Intentional error"):
        await raise_error.stateless()
