"""Symbolic: the headers each model call carries.

Every model call carries ``X-Unify-Session``, ``X-Unify-Request``,
``X-Unify-Call-Kind`` and ``X-Unify-Msg-Count`` (and a fork
``X-Unify-Parent``) as ``extra_headers``, so a proxy can attribute it. Requests are read at unillm's transport, below the request
building (``tests/scripted_model.py``), so nothing leaves the process. The
acts run their cells in the sandboxed Python worker.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Iterator

import pytest

from tests import cache_discipline_helpers as h
from tests.scripted_model import ScriptedModel, cell, reply, scripted
from unify.common import llm_client
from unify.common._async_tool import context_compression as cc

HEADERS = {
    "X-Unify-Session",
    "X-Unify-Request",
    "X-Unify-Call-Kind",
    "X-Unify-Msg-Count",
}
FORK_HEADERS = HEADERS | {"X-Unify-Parent"}
# Never kept from the transport's arguments: the session and HTTP client are
# objects, and a key is a credential.
_NOT_KEPT = {"shared_session", "client", "api_key"}


@contextlib.contextmanager
def transport(model: ScriptedModel) -> Iterator[list[dict]]:
    """Install *model* and keep every call's transport keyword arguments,
    in call order (``model.calls[i]`` is ``sent[i]``)."""
    import unillm.clients.uni_llm as uni_llm

    sent: list[dict] = []
    with scripted(model):
        answer = uni_llm._acompletion_with_transient_retry

        async def spy(**kw):
            sent.append(
                json.loads(
                    json.dumps(
                        {k: v for k, v in kw.items() if k not in _NOT_KEPT},
                        default=str,
                    ),
                ),
            )
            return await answer(**kw)

        uni_llm._acompletion_with_transient_retry = spy
        try:
            yield sent
        finally:
            uni_llm._acompletion_with_transient_retry = answer


def _of(model: ScriptedModel, sent: list[dict], kind: str) -> list[dict]:
    assert len(sent) == len(model.calls)
    return [kw for call, kw in zip(model.calls, sent) if call.kind == kind]


def _headers(kw: dict, expected: set[str] = HEADERS) -> dict[str, str]:
    headers = kw["extra_headers"]
    assert set(headers) == expected
    for value in headers.values():
        assert llm_client.HEADER_VALUE.match(value), value
    assert headers["X-Unify-Msg-Count"] == str(len(kw["messages"]))
    return headers


@contextlib.asynccontextmanager
async def _actor():
    from unify.actor.code_act_actor import CodeActActor

    actor = CodeActActor()
    yield actor


async def _next_response(handle, bound: float = 30.0) -> dict:
    while True:
        event = await asyncio.wait_for(handle.next_notification(), bound)
        if isinstance(event, dict) and event.get("type") == "response":
            return event


# ── the values ─────────────────────────────────────────────────────────────


def test_an_origin_label_is_put_in_the_header_charset():
    value = llm_client._header_value
    assert value("CodeActActor.act") == "CodeActActor.act"
    assert value("CodeActActor.act#purpose=planning") == "CodeActActor.act_purpose_planning"
    assert value("a b/c") == "a_b_c"
    assert value("x" * 100) == "x" * 64
    assert value(None) == value("") == "none"


def test_a_fork_has_its_own_session_and_names_its_parent():
    parent = h.new_client()
    fork = llm_client.fork_llm_client(parent, origin="compress_context")
    grandchild = llm_client.fork_llm_client(fork, origin="compress_context")
    sessions = {c._unify_request_metadata.session for c in (parent, fork, grandchild)}
    assert len(sessions) == 3
    assert parent._unify_request_metadata.parent is None
    assert fork._unify_request_metadata.parent == (
        parent._unify_request_metadata.session
    )
    assert grandchild._unify_request_metadata.parent == (
        fork._unify_request_metadata.session
    )


# ── each kind of call ─────────────────────────────────────────────────────


async def _scripted_act() -> tuple[ScriptedModel, list[dict]]:
    model = ScriptedModel(
        actor=[reply(calls=[cell("x = 41\nprint(x + 1)")]), reply("42")],
    )
    async with _actor() as actor:
        with transport(model) as sent:
            handle = await actor.act("Compute.")
            assert await asyncio.wait_for(handle.result(), 60) == "42"
    assert model.kinds() == ["actor", "actor"]
    return model, sent


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_actors_calls_carry_one_session_and_its_first_request():
    model, sent = await _scripted_act()
    first, second = (_headers(kw) for kw in _of(model, sent, "actor"))
    assert first["X-Unify-Session"] == second["X-Unify-Session"]
    assert first["X-Unify-Request"] == second["X-Unify-Request"] == "1"
    assert first["X-Unify-Call-Kind"] == "CodeActActor.act"
    assert int(second["X-Unify-Msg-Count"]) > int(first["X-Unify-Msg-Count"])


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_compression_fork_names_its_parent_and_the_restart_counts_nothing():
    from unify.common.async_tool_loop import start_async_tool_loop

    async def execute_code(code: str) -> str:
        """Run code.

        Args:
            code: The code.
        """
        return f"result of {code}"

    # The first reply's prompt reaches the compaction point, so the loop
    # compacts before its next model call.
    model = ScriptedModel(
        actor=[
            reply(calls=[("execute_code", {"code": "one"})], prompt_tokens=cc.COMPACTION_AT_TOKENS),
            reply("done"),
        ],
        compression_fork=[reply("Summary: one ran.")],
    )
    with transport(model) as sent:
        handle = start_async_tool_loop(
            h.new_client(),
            "Run one.",
            {"execute_code": execute_code},
            loop_id="HeaderLoop",
            log_steps=False,
            timeout=60,
            tools_in_code=True,
        )
        assert await asyncio.wait_for(handle.result(), 60) == "done"
    assert model.kinds() == ["actor", "compression_fork", "actor"]
    model.assert_used_up()
    before, after = (_headers(kw) for kw in _of(model, sent, "actor"))
    (fork,) = (
        _headers(kw, FORK_HEADERS) for kw in _of(model, sent, "compression_fork")
    )
    session = before["X-Unify-Session"]
    # The loop keeps its client across the restart, and the restart's
    # loop-authored first message is not a requester message.
    assert after["X-Unify-Session"] == session
    assert before["X-Unify-Request"] == after["X-Unify-Request"] == "1"
    assert before["X-Unify-Call-Kind"] == "HeaderLoop"
    assert fork["X-Unify-Session"] != session
    assert fork["X-Unify-Parent"] == session
    assert fork["X-Unify-Request"] == "1"
    assert fork["X-Unify-Call-Kind"] == "compress_context"


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_cells_query_llm_call_is_its_own_session():
    model = ScriptedModel(
        actor=[
            reply(calls=[cell('answer = await query_llm("Say hi.")\nprint(answer)')]),
            reply("Said."),
        ],
        query_llm=[reply("hi")],
    )
    async with _actor() as actor:
        with transport(model) as sent:
            handle = await actor.act("Greet.")
            assert await asyncio.wait_for(handle.result(), 60) == "Said."
    model.assert_used_up()
    (actor_session,) = {
        _headers(kw)["X-Unify-Session"] for kw in _of(model, sent, "actor")
    }
    (query,) = (_headers(kw) for kw in _of(model, sent, "query_llm"))
    assert query["X-Unify-Session"] != actor_session
    assert query["X-Unify-Request"] == "0"
    # A cell's call carries the cell call kind, whatever origin it names.
    assert query["X-Unify-Call-Kind"] == "cell"


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_request_counter_goes_up_with_each_request_of_a_session():
    model = ScriptedModel(actor=[reply("First answer."), reply("Second answer.")])
    async with _actor() as actor:
        with transport(model) as sent:
            handle = await actor.act("First request.", persist=True)
            first = await _next_response(handle)
            await handle.submit("Second request.")
            second = await _next_response(handle)
            await handle.stop("done")
            await asyncio.wait_for(handle.result(), 30)
    assert (first["content"], second["content"]) == ("First answer.", "Second answer.")
    calls = [_headers(kw) for kw in _of(model, sent, "actor")]
    assert [c["X-Unify-Request"] for c in calls] == ["1", "2"]
    assert len({c["X-Unify-Session"] for c in calls}) == 1
