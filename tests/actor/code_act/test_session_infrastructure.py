import pytest

from tests.helpers import _handle_project
from unify.actor.code_act_actor import CodeActActor
from unify.actor.execution import (
    PythonExecutionSession,
    SessionExecutor,
    _CURRENT_SANDBOX,
    _validate_execution_params,
    parts_to_text,
)
from unify.common.tool_errors import ToolInputError


@pytest.mark.parametrize(
    "kwargs,expect_error_substr",
    [
        (
            dict(
                state_mode="stateless",
                session_id=0,
                session_name=None,
            ),
            "Cannot use state_mode='stateless' with a session",
        ),
        (
            dict(
                state_mode="read_only",
                session_id=None,
                session_name=None,
            ),
            "Cannot use state_mode='read_only' without specifying a session",
        ),
        (
            dict(
                state_mode="stateful",
                session_id=1,
                session_name="repo_nav",
                resolve_session_name=lambda n: (0 if n == "repo_nav" else None),
            ),
            "refer to different sessions",
        ),
        (
            dict(
                state_mode="read_only",
                session_id=None,
                session_name="does_not_exist",
                resolve_session_name=lambda _n: None,
            ),
            "not found for read_only",
        ),
        (
            dict(
                state_mode="stateful",
                session_id=None,
                session_name="new_session",
                resolve_session_name=lambda _n: None,
                max_sessions_total=2,
                active_session_count=2,
            ),
            "Session limit exceeded",
        ),
        (
            dict(
                state_mode="stateful",
                session_id=99,
                session_name=None,
                max_sessions_total=2,
                active_session_count=2,
                session_exists=lambda _s: False,
            ),
            "Session limit exceeded",
        ),
        (
            dict(
                state_mode="read_only",
                session_id=4,
                session_name=None,
                session_exists=lambda _s: False,
            ),
            "Session 4 does not exist for read_only execution",
        ),
    ],
)
def test_validate_execution_params_matrix(kwargs, expect_error_substr: str):
    with pytest.raises(ToolInputError) as excinfo:
        _validate_execution_params(**kwargs)
    assert expect_error_substr.lower() in excinfo.value.message.lower()


def test_a_refusal_reads_as_itself_not_as_a_traceback():
    """The rendered refusal carries what the caller needs to fix the call.

    The loop surfaces this text in place of a traceback, so everything the
    caller acts on has to survive the raise: what is wrong, what to change, and
    which arguments the complaint is about.
    """
    with pytest.raises(ToolInputError) as excinfo:
        _validate_execution_params(
            state_mode="stateless",
            session_id=5,
            session_name="contact_lookup",
        )
    rendered = excinfo.value.as_tool_result()
    assert "Cannot use state_mode='stateless' with a session" in rendered
    assert "Suggestion: Remove session_id/session_name" in rendered
    assert "session_name='contact_lookup'" in rendered


def test_matching_name_and_id_pass_validation():
    assert (
        _validate_execution_params(
            state_mode="stateful",
            session_id=3,
            session_name="audit",
            resolve_session_name=lambda n: (3 if n == "audit" else None),
        )
        is None
    )


@pytest.mark.asyncio
async def test_session_executor_python_stateful_reuses_session():
    ex = SessionExecutor(
        environments={},  # no primitives injection needed for this unit test
        timeout=5.0,
    )
    try:
        r1 = await ex.execute(
            code="x = 1\nx",
            state_mode="stateful",
            session_id=0,
        )
        assert r1["error"] is None
        assert r1["session_created"] is True

        r2 = await ex.execute(
            code="x = x + 1\nx",
            state_mode="stateful",
            session_id=0,
        )
        assert r2["error"] is None
        assert r2["session_created"] is False
        # Result value can vary depending on sandbox result-capture logic, but stdout should be empty.
        assert isinstance(r2["duration_ms"], int)
    finally:
        await ex.close()


@pytest.mark.asyncio
async def test_session_executor_sessions_are_keyed_by_id_alone():
    """A session's identity is its integer id; listing and closing use it."""
    ex = SessionExecutor(environments={}, timeout=5.0)
    try:
        await ex.execute(code="x = 1", state_mode="stateful", session_id=2)
        assert ex.has_python_session(session_id=2)
        assert not ex.has_python_session(session_id=3)

        listed = ex.list_in_process_python_sessions()
        assert [s["session_id"] for s in listed] == [2]
        assert set(listed[0]) == {
            "session_id",
            "created_at",
            "last_used",
            "state_summary",
        }

        assert await ex.close_in_process_python_session(session_id=2) is True
        assert await ex.close_in_process_python_session(session_id=2) is False
        assert ex.list_in_process_python_sessions() == []
    finally:
        await ex.close()


@pytest.mark.asyncio
async def test_session_executor_python_read_only_does_not_mutate_state():
    ex = SessionExecutor(
        environments={},
        timeout=5.0,
    )
    try:
        r1 = await ex.execute(
            code="x = 1",
            state_mode="stateful",
            session_id=0,
        )
        assert r1["error"] is None

        ro = await ex.execute(
            code="x = 999",
            state_mode="read_only",
            session_id=0,
        )
        assert ro["error"] is None

        r2 = await ex.execute(
            code="print(x)",
            state_mode="stateful",
            session_id=0,
        )
        assert r2["error"] is None
        assert "1" in parts_to_text(r2["stdout"])
    finally:
        await ex.close()


@pytest.mark.asyncio
@_handle_project
async def test_read_only_session_zero_branches_off_the_bound_sandbox():
    """Session 0 is the per-call sandbox for read_only, as it is for stateful.

    A read_only cell on session 0 reads what stateful cells left there, and
    nothing it assigns persists.
    """
    actor = CodeActActor(environments=[])
    execute_code = actor.get_tools("act")["execute_code"]
    sandbox = PythonExecutionSession(environments={})
    token = _CURRENT_SANDBOX.set(sandbox)
    try:
        seeded = await execute_code(
            thought="Seed the session.",
            code="prices = [3, 5]\ntotal = sum(prices)",
            state_mode="stateful",
            session_id=0,
        )
        assert seeded.error is None

        what_if = await execute_code(
            thought="Try a variant without keeping it.",
            code="total = total * 10\ndraft = 'discarded'\nprint(total)",
            state_mode="read_only",
            session_id=0,
        )
        assert what_if.error is None
        assert what_if.session_id == 0
        assert parts_to_text(what_if.stdout).strip() == "80"

        assert sandbox.global_state["total"] == 8
        assert "draft" not in sandbox.global_state
    finally:
        _CURRENT_SANDBOX.reset(token)
        await sandbox.close()
        await actor.close()


@pytest.mark.asyncio
async def test_session_executor_isolation_between_python_sessions():
    ex = SessionExecutor(
        environments={},
        timeout=5.0,
    )
    try:
        a1 = await ex.execute(
            code="x = 1",
            state_mode="stateful",
            session_id=0,
        )
        b1 = await ex.execute(
            code="x = 2",
            state_mode="stateful",
            session_id=1,
        )
        assert a1["error"] is None
        assert b1["error"] is None

        a2 = await ex.execute(
            code="print(x)",
            state_mode="stateful",
            session_id=0,
        )
        b2 = await ex.execute(
            code="print(x)",
            state_mode="stateful",
            session_id=1,
        )
        assert "1" in parts_to_text(a2["stdout"])
        assert "2" in parts_to_text(b2["stdout"])
    finally:
        await ex.close()
