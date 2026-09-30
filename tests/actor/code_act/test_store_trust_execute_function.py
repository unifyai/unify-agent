"""Symbolic: with ``UNIFY_STORE_TRUST=ramp`` the actor's ``execute_function`` tool is evidence too.

The tool does not call the boundary-wrapped function that records reuse: it
prepends the stored implementation to a synthesized call and runs that in the
session. In a Stage 1 AppWorld cell, 4 of the task actor's 12 calls of stored
functions went through this tool, all 4 raised (two 401s from a placeholder
token such as ``"{{access_token}}"``, a ValueError, a TypeError), and none was
recorded, so the functions stayed on probation. With the switch on, a call
through the tool that reports an error (or whose install fails) is a failure
-- failures on two distinct inputs quarantine the function -- and one that
returns is a pass counted with its arguments. A failure the caller caused (a
placeholder token, arguments that do not fit the signature) is not held
against the function, and the tool's reply says so. A call a correction
reached while it ran is not evidence. With the switch off nothing is recorded
and the tool returns exactly what it returns as shipped.
Functions run in-process through the actor's own session executor; no model
is called.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.helpers import _handle_project
from unify import db
from unify.actor.code_act_actor import CodeActActor
from unify.actor.execution import PythonExecutionSession, _CURRENT_SANDBOX
from unify.function_manager import store_trust
from unify.function_manager.function_manager import FunctionManager
from unify.settings import SETTINGS

DIVIDE = "def divide(a: int, b: int) -> float:\n    return a / b\n"
LOGIN = (
    "def fetch_profile(access_token: str) -> dict:\n"
    "    if not access_token.isalnum():\n"
    "        raise PermissionError(\n"
    "            f'401 Unauthorized: {access_token!r} is not a token'\n"
    "        )\n"
    "    return {'user': 'ada'}\n"
)
PLACEHOLDER_ERROR = (
    "PermissionError: 401 Unauthorized: '{{access_token}}' is not a token"
)
BAD_TOKEN_ERROR = "PermissionError: 401 Unauthorized: 'expired-1' is not a token"


@pytest.fixture
def trust_on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_TRUST", "ramp")


def _rows() -> list[dict]:
    return db.query("SELECT * FROM function_trust ORDER BY function_id")


def _trust(fm: FunctionManager, name: str) -> store_trust.Trust:
    return store_trust.trust(int(fm.list_function_name_to_ids()[name]))


def _error(out: Any) -> Any:
    return out.get("error") if isinstance(out, dict) else getattr(out, "error", None)


def _result(out: Any) -> Any:
    return out.get("result") if isinstance(out, dict) else getattr(out, "result", None)


def _last_line(text: Any) -> Any:
    return text.strip().splitlines()[-1] if isinstance(text, str) else text


class _Actor:
    """A CodeActActor over ``fm`` whose ``execute_function`` tool is called directly."""

    def __init__(self, fm: FunctionManager):
        self.actor = CodeActActor(function_manager=fm, can_store=False)
        tool = self.actor.get_tools("act")["execute_function"]
        self.tool = getattr(tool, "fn", tool)

    async def call(self, function_name: str, **call_kwargs: Any) -> Any:
        return await self.tool(
            thought="Reusing a stored function.",
            function_name=function_name,
            call_kwargs=call_kwargs,
        )

    async def close(self) -> None:
        await self.actor.close()


# --------------------------------------------------------------------------- #
#  Failures and passes                                                         #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.asyncio
async def test_failures_through_the_tool_on_two_inputs_quarantine_the_function(
    trust_on,
):
    fm = FunctionManager(include_primitives=False)
    fm.add_functions(implementations=[LOGIN, DIVIDE])
    actor = _Actor(fm)
    try:
        out = await actor.call("fetch_profile", access_token="expired-1")
        # the model still sees the error, reported rather than raised, as shipped
        assert _result(out) is None
        assert _last_line(_error(out)) == BAD_TOKEN_ERROR
        t = _trust(fm, "fetch_profile")
        # one failure can be a bad input: still on probation
        assert (t.state, t.passes, t.failures, t.clean_uses) == ("probation", 0, 1, 0)
        assert t.last_failure == BAD_TOKEN_ERROR
        await actor.call("fetch_profile", access_token="expired-2")
    finally:
        await actor.close()
    t = _trust(fm, "fetch_profile")
    assert (t.state, t.failures) == ("quarantined", 2)
    assert [r["function_id"] for r in _rows()] == [t.function_id]
    # and the quarantine takes effect: the next loaded read leaves it out
    namespace: dict = {}
    fm.list_functions(_return_callable=True, _namespace=namespace)
    assert "fetch_profile" not in namespace and "divide" in namespace


async def _off_and_on(monkeypatch, function_name: str, **call_kwargs) -> tuple:
    """The tool's reply to one call with the switch off, then on (a fresh store each)."""
    replies = []
    for switch in ("", "ramp"):
        db.clear()
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_TRUST", switch)
        fm = FunctionManager(include_primitives=False)
        fm.add_functions(implementations=[LOGIN, DIVIDE])
        actor = _Actor(fm)
        try:
            replies.append(await actor.call(function_name, **call_kwargs))
        finally:
            await actor.close()
    return replies[0], replies[1]


@_handle_project
@pytest.mark.asyncio
async def test_a_placeholder_token_is_not_held_against_the_function(monkeypatch):
    """The Stage 1 case: the model passed "{{access_token}}" and the login returned 401."""
    off, on = await _off_and_on(
        monkeypatch,
        "fetch_profile",
        access_token="{{access_token}}",
    )
    assert _rows() == []
    assert _last_line(_error(off)) == PLACEHOLDER_ERROR
    # the reply is the shipped one plus a note telling the model what to fix
    assert _error(on) == (
        f"{_error(off).rstrip()}\n\n"
        "Not counted against the stored function `fetch_profile`: "
        "`access_token` was given an unfilled placeholder instead of a real "
        "value; pass the actual value from your session.\n"
    )
    assert _result(on) == _result(off) is None


@_handle_project
@pytest.mark.asyncio
async def test_arguments_that_do_not_fit_are_not_held_against_the_function(
    monkeypatch,
):
    off, on = await _off_and_on(monkeypatch, "divide", a=1, denominator=2)
    assert _rows() == []
    assert "TypeError" in _error(off)
    assert _error(on) == (
        f"{_error(off).rstrip()}\n\n"
        "Not counted against the stored function `divide`: the arguments do "
        "not fit the signature of `divide(a, b)`: missing a required "
        "argument: 'b'.\n"
    )


@_handle_project
@pytest.mark.asyncio
async def test_a_call_through_the_tool_that_returns_is_a_pass(trust_on):
    fm = FunctionManager(include_primitives=False)
    fm.add_functions(implementations=[DIVIDE])
    actor = _Actor(fm)
    try:
        out = await actor.call("divide", a=6, b=3)
        assert (_result(out), _error(out)) == (2.0, None)
        t = _trust(fm, "divide")
        assert (t.state, t.passes, t.distinct_inputs, t.failures) == (
            "probation",
            1,
            1,
            0,
        )
        assert t.input_hashes == (store_trust.input_hash({"a": 6, "b": 3}),)
        # read-only: trusted after 3 passes over 2 distinct inputs
        await actor.call("divide", a=6, b=3)
        assert _trust(fm, "divide").state == "probation"
        await actor.call("divide", a=8, b=2)
        t = _trust(fm, "divide")
        assert (t.state, t.passes, t.distinct_inputs) == ("trusted", 3, 2)
        # a failure through the tool demotes a trusted function, a second
        # on another input quarantines it
        out = await actor.call("divide", a=1, b=0)
        assert _last_line(_error(out)) == "ZeroDivisionError: division by zero"
        t = _trust(fm, "divide")
        assert (t.state, t.passes, t.failures, t.last_failure) == (
            "probation",
            3,
            1,
            "ZeroDivisionError: division by zero",
        )
        await actor.call("divide", a=2, b=0)
        assert _trust(fm, "divide").state == "quarantined"
    finally:
        await actor.close()


@_handle_project
@pytest.mark.asyncio
async def test_a_failed_install_through_the_tool_is_a_failed_reuse(
    trust_on,
    monkeypatch,
):
    from unify import environment

    fm = FunctionManager(include_primitives=False)
    fm.add_functions(implementations=[DIVIDE])
    db.execute(
        "UPDATE functions SET dependencies = ? WHERE name = ?",
        (db.dumps(["no-such-package==0"]), "divide"),
    )

    def no_install(requirements):
        raise RuntimeError("uv pip install failed: no matching distribution")

    monkeypatch.setattr(environment, "ensure", no_install)
    actor = _Actor(fm)
    try:
        with pytest.raises(RuntimeError, match="uv pip install failed"):
            await actor.call("divide", a=1, b=1)
    finally:
        await actor.close()
    t = _trust(fm, "divide")
    assert (t.state, t.passes, t.failures) == ("probation", 0, 1)
    assert t.last_failure.startswith("RuntimeError: uv pip install failed")


@_handle_project
@pytest.mark.asyncio
async def test_a_call_a_correction_stopped_is_not_evidence(trust_on, monkeypatch):
    from unify.function_manager import steering_patcher
    from unify.function_manager.steering import InterruptionRequest

    async def stop_author(*, interjections, session):
        return InterruptionRequest(reason="the user stopped it", stop=True)

    monkeypatch.setattr(steering_patcher, "build_patch_author", lambda: stop_author)
    fm = FunctionManager(include_primitives=False)
    fm.add_functions(implementations=[DIVIDE])
    actor = _Actor(fm)
    sandbox = PythonExecutionSession(environments={})
    token = _CURRENT_SANDBOX.set(sandbox)
    try:
        interjections: asyncio.Queue = asyncio.Queue()
        await interjections.put("stop, that is the wrong one")
        out = await actor.tool(
            thought="Reusing a stored function.",
            function_name="divide",
            call_kwargs={"a": 1, "b": 0},
            _interject_queue=interjections,
        )
        # stopped before it ran: no error, and no pass either
        assert _error(out) is None
        assert _result(out) == {"status": "stopped", "reason": "the user stopped it"}
        assert _rows() == []
        # the same binding with nothing interjected is recorded
        out = await actor.tool(
            thought="Reusing a stored function.",
            function_name="divide",
            call_kwargs={"a": 4, "b": 2},
            _interject_queue=asyncio.Queue(),
        )
        assert _result(out) == 2.0
        assert _trust(fm, "divide").passes == 1
    finally:
        _CURRENT_SANDBOX.reset(token)
        await actor.close()


# --------------------------------------------------------------------------- #
#  Failure history across an overwrite                                         #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.asyncio
async def test_a_repair_after_tool_failures_keeps_their_history(trust_on):
    """The Stage 1 case: a function that raised 3 times, then was overwritten by the review."""
    fm = FunctionManager(include_primitives=False)
    fm.add_functions(implementations=[LOGIN])
    actor = _Actor(fm)
    try:
        for token in ("expired-1", "expired-2", "expired-3"):
            await actor.call("fetch_profile", access_token=token)
        t = _trust(fm, "fetch_profile")
        assert (t.state, t.failures) == ("quarantined", 3)
        assert t.last_failure == (
            "PermissionError: 401 Unauthorized: 'expired-3' is not a token"
        )

        repaired = LOGIN.replace("isalnum()", "strip()")
        assert fm.add_functions(implementations=[repaired], overwrite=True) == {
            "fetch_profile": "updated",
        }
        t = _trust(fm, "fetch_profile")
        # back on probation for the new source, the old failures still on record
        assert (t.state, t.passes, t.distinct_inputs, t.clean_uses) == (
            "probation",
            0,
            0,
            0,
        )
        assert (t.failures, t.last_failure) == (
            3,
            "PermissionError: 401 Unauthorized: 'expired-3' is not a token",
        )
        assert t.source_hash == store_trust.sha256(repaired)
        assert store_trust.needs_repair() == []  # no longer waiting for repair

        # the repaired version is callable again and earns its own passes
        out = await actor.call("fetch_profile", access_token="tok123")
        assert _result(out) == {"user": "ada"}
        t = _trust(fm, "fetch_profile")
        assert (t.state, t.passes, t.failures) == ("probation", 1, 3)
        # new failures add to the history; the new version counts its own
        # failing inputs, and the review's note shows the whole history
        await actor.call("fetch_profile", access_token="   ")
        assert (
            _trust(fm, "fetch_profile").state,
            _trust(fm, "fetch_profile").failures,
        ) == (
            "probation",
            4,
        )
        await actor.call("fetch_profile", access_token=" ")
    finally:
        await actor.close()
    t = _trust(fm, "fetch_profile")
    assert (t.state, t.passes, t.failures) == ("quarantined", 1, 5)
    assert "`fetch_profile`: 5 failure(s) after 1 pass(es)" in (
        store_trust.needs_repair_note()
    )


# --------------------------------------------------------------------------- #
#  Switch off                                                                  #
# --------------------------------------------------------------------------- #


async def _scenario(fm: FunctionManager) -> list:
    """Passes and failures through the tool, then a repair; what the model sees."""
    fm.add_functions(implementations=[DIVIDE, LOGIN])
    actor = _Actor(fm)
    seen: list = []
    try:
        for name, kwargs in (
            ("divide", {"a": 6, "b": 3}),
            ("divide", {"a": 1, "b": 0}),
            ("fetch_profile", {"access_token": "expired-1"}),
            ("fetch_profile", {"access_token": "tok123"}),
            ("divide", {"a": 8, "b": 2}),
        ):
            out = await actor.call(name, **kwargs)
            seen.append((type(out).__name__, _result(out), _last_line(_error(out))))
    finally:
        await actor.close()
    fixed = DIVIDE.replace("return a / b", "return a / b if b else 0.0")
    seen.append(fm.add_functions(implementations=[fixed], overwrite=True))
    return seen


@_handle_project
@pytest.mark.asyncio
async def test_switch_off_records_nothing_and_the_tool_returns_the_same(
    monkeypatch,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_TRUST", "")
    off = await _scenario(FunctionManager(include_primitives=False))
    assert _rows() == []
    assert off[1][2] == "ZeroDivisionError: division by zero"
    assert off[2][2] == BAD_TOKEN_ERROR

    db.clear()
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_TRUST", "ramp")
    fm = FunctionManager(include_primitives=False)
    on = await _scenario(fm)
    assert on == off
    divide, profile = _trust(fm, "divide"), _trust(fm, "fetch_profile")
    # divide was repaired: probation with its failure kept; fetch_profile
    # failed once, on one input, and stays on probation
    assert (divide.state, divide.passes, divide.failures) == ("probation", 0, 1)
    assert (profile.state, profile.passes, profile.failures) == ("probation", 1, 1)
