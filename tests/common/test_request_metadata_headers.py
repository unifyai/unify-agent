"""Symbolic: ``UNIFY_REQUEST_METADATA_HEADERS`` and the headers each model call carries.

With the setting on, every model call carries ``X-Unify-Session``,
``X-Unify-Request``, ``X-Unify-Call-Kind`` and ``X-Unify-Msg-Count`` (and a
fork ``X-Unify-Parent``) as ``extra_headers``, the request body unchanged;
off, the call's arguments are as shipped, without even an empty
``extra_headers``. Requests are read at unillm's transport, below the request
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
from tests.scripted_model import (
    Always,
    ScriptedModel,
    cell,
    reply,
    scripted,
)
from unify.common import llm_client
from unify.settings import ProductionSettings, SETTINGS

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


@pytest.fixture
def headers_on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_REQUEST_METADATA_HEADERS", "on")


@pytest.fixture
def headers_off(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_REQUEST_METADATA_HEADERS", "")


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
    try:
        yield actor
    finally:
        await actor.close()


async def _next_response(handle, bound: float = 30.0) -> dict:
    while True:
        event = await asyncio.wait_for(handle.next_notification(), bound)
        if isinstance(event, dict) and event.get("type") == "response":
            return event


# ── the setting and the values ─────────────────────────────────────────────


def test_the_setting_is_off_by_default_and_takes_on_or_off():
    field = ProductionSettings.model_fields["UNIFY_REQUEST_METADATA_HEADERS"]
    assert field.default == ""
    parse = ProductionSettings.parse_request_metadata_headers
    assert parse(" On ") == "on"
    assert parse("off") == ""
    assert parse(None) == ""
    with pytest.raises(ValueError):
        parse("yes")


def test_an_origin_label_is_put_in_the_header_charset():
    value = llm_client._header_value
    assert value("CodeActActor.act") == "CodeActActor.act"
    assert value("StorageCheck#purpose=planning") == "StorageCheck_purpose_planning"
    assert value("a b/c") == "a_b_c"
    assert value("x" * 100) == "x" * 64
    assert value(None) == value("") == "none"


def test_a_fork_has_its_own_session_and_names_its_parent(headers_on):
    parent = h.new_client()
    fork = llm_client.fork_llm_client(parent, origin="compress_context")
    grandchild = llm_client.fork_llm_client(fork, origin="StorageCheck")
    sessions = {c._unify_request_metadata.session for c in (parent, fork, grandchild)}
    assert len(sessions) == 3
    assert parent._unify_request_metadata.parent is None
    assert fork._unify_request_metadata.parent == (
        parent._unify_request_metadata.session
    )
    assert grandchild._unify_request_metadata.parent == (
        fork._unify_request_metadata.session
    )


def test_off_a_client_gets_no_header_state(headers_off):
    client = h.new_client()
    fork = llm_client.fork_llm_client(client, origin="compress_context")
    for c in (client, fork):
        assert not hasattr(c, "_unify_request_metadata")
        assert "_generate" not in vars(c)
    llm_client.count_requester_message(client)  # nothing to count


# ── off: the call's arguments are as shipped ──────────────────────────────


async def _scripted_act() -> tuple[ScriptedModel, list[dict]]:
    model = ScriptedModel(
        actor=[reply(calls=[cell("x = 41\nprint(x + 1)")]), reply("42")],
    )
    async with _actor() as actor:
        with transport(model) as sent:
            handle = await actor.act("Compute.", can_store=False)
            assert await asyncio.wait_for(handle.result(), 60) == "42"
    assert model.kinds() == ["actor", "actor"]
    return model, sent


def _comparable(kw: dict) -> dict:
    """The call without its headers; the session's home, the clock and the
    scripted call ids masked (the scripted model numbers its calls across a
    process, so two runs of one scenario name them differently)."""
    import os
    import re

    text = json.dumps(kw, sort_keys=True)
    home = os.environ.get("UNIFY_HOME") or ""
    if home:
        text = text.replace(home, "$UNIFY_HOME")
    text = re.sub(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?", "<clock>", text)
    text = re.sub(r"call_\d+", "call_N", text)
    return json.loads(text)


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_off_no_header_is_added_and_on_only_headers_are(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_REQUEST_METADATA_HEADERS", "")
    _, off = await _scripted_act()
    for kw in off:
        assert "extra_headers" not in kw

    monkeypatch.setattr(SETTINGS, "UNIFY_REQUEST_METADATA_HEADERS", "on")
    _, on = await _scripted_act()
    assert [set(kw) for kw in on] == [set(kw) | {"extra_headers"} for kw in off]
    for kw_on, kw_off in zip(on, off):
        _headers(kw_on)
        without = {k: v for k, v in kw_on.items() if k != "extra_headers"}
        assert _comparable(without) == _comparable(kw_off)


# ── on: each kind of call ─────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_actors_calls_carry_one_session_and_its_first_request(
    headers_on,
):
    model, sent = await _scripted_act()
    first, second = (_headers(kw) for kw in _of(model, sent, "actor"))
    assert first["X-Unify-Session"] == second["X-Unify-Session"]
    assert first["X-Unify-Request"] == second["X-Unify-Request"] == "1"
    assert first["X-Unify-Call-Kind"] == "CodeActActor.act"
    assert int(second["X-Unify-Msg-Count"]) > int(first["X-Unify-Msg-Count"])


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_storage_review_fork_has_its_own_session_and_a_parent(
    headers_on,
):
    model = ScriptedModel(
        actor=[reply(calls=[cell("print('done')")]), reply("Finished.")],
        review=Always(reply("Nothing to store.")),
        allow={"gate"},
    )
    async with _actor() as actor:
        with transport(model) as sent:
            handle = await actor.act("Do the task.", can_store=True)
            assert await asyncio.wait_for(handle.result(), 60) == "Finished."
            await asyncio.wait_for(handle._lifecycle_task, 60)
    (actor_session,) = {
        _headers(kw)["X-Unify-Session"] for kw in _of(model, sent, "actor")
    }
    reviews = [_headers(kw, FORK_HEADERS) for kw in _of(model, sent, "review")]
    assert reviews
    (review_session,) = {r["X-Unify-Session"] for r in reviews}
    assert review_session != actor_session
    assert {r["X-Unify-Parent"] for r in reviews} == {actor_session}
    # The review's loop answers no requester: it reports the request it forked at.
    assert {r["X-Unify-Request"] for r in reviews} == {"1"}
    assert all(r["X-Unify-Call-Kind"] != "CodeActActor.act" for r in reviews)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_compression_fork_names_its_parent_and_the_restart_counts_nothing(
    headers_on,
):
    from unify.common.async_tool_loop import start_async_tool_loop

    async def execute_code(code: str) -> str:
        """Run code.

        Args:
            code: The code.
        """
        return f"result of {code}"

    # The model asks for the compression itself (compress_context), which
    # needs no model input limit: a keyless test cannot resolve one for the
    # context threshold.
    model = ScriptedModel(
        actor=[
            reply(calls=[("execute_code", {"code": "one"})]),
            reply(calls=[("compress_context", {})]),
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
            bind_request=True,
        )
        assert await asyncio.wait_for(handle.result(), 60) == "done"
    assert model.kinds() == ["actor", "actor", "compression_fork", "actor"]
    model.assert_used_up()
    before, turn, after = (_headers(kw) for kw in _of(model, sent, "actor"))
    (fork,) = (
        _headers(kw, FORK_HEADERS) for kw in _of(model, sent, "compression_fork")
    )
    session = before["X-Unify-Session"]
    # The loop keeps its client across the restart, and the restart's
    # loop-authored first message is not a requester message.
    assert turn["X-Unify-Session"] == after["X-Unify-Session"] == session
    assert before["X-Unify-Request"] == after["X-Unify-Request"] == "1"
    assert before["X-Unify-Call-Kind"] == "HeaderLoop"
    assert fork["X-Unify-Session"] != session
    assert fork["X-Unify-Parent"] == session
    assert fork["X-Unify-Request"] == "1"
    assert fork["X-Unify-Call-Kind"] == "compress_context"


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_cells_query_llm_call_is_its_own_session(headers_on):
    model = ScriptedModel(
        actor=[
            reply(calls=[cell('answer = await query_llm("Say hi.")\nprint(answer)')]),
            reply("Said."),
        ],
        query_llm=[reply("hi")],
    )
    async with _actor() as actor:
        with transport(model) as sent:
            handle = await actor.act("Greet.", can_store=False)
            assert await asyncio.wait_for(handle.result(), 60) == "Said."
    model.assert_used_up()
    (actor_session,) = {
        _headers(kw)["X-Unify-Session"] for kw in _of(model, sent, "actor")
    }
    (query,) = (_headers(kw) for kw in _of(model, sent, "query_llm"))
    assert query["X-Unify-Session"] != actor_session
    assert query["X-Unify-Request"] == "0"
    assert query["X-Unify-Call-Kind"] == "CodeActActor.query_llm"


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_request_counter_goes_up_with_each_request_of_a_session(
    headers_on,
):
    model = ScriptedModel(actor=[reply("First answer."), reply("Second answer.")])
    async with _actor() as actor:
        with transport(model) as sent:
            handle = await actor.act("First request.", can_store=False, persist=True)
            first = await _next_response(handle)
            await handle.submit("Second request.")
            second = await _next_response(handle)
            await handle.stop("done")
            await asyncio.wait_for(handle.result(), 30)
    assert (first["content"], second["content"]) == ("First answer.", "Second answer.")
    calls = [_headers(kw) for kw in _of(model, sent, "actor")]
    assert [c["X-Unify-Request"] for c in calls] == ["1", "2"]
    assert len({c["X-Unify-Session"] for c in calls}) == 1
