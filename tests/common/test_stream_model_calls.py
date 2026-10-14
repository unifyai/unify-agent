"""Symbolic: ``UNIFY_STREAM_MODEL_CALLS`` and the clients whose calls it streams,
and ``UNIFY_STREAM_IDLE_S``, the silence limit those calls get from unillm.

With the setting on, every client ``new_llm_client`` makes, and every fork of
one, carries unillm's ``stream_transport`` flag: its calls go out as streams
that unillm reads whole and reassembles. Off, the clients are as shipped.
Requests are read at unillm's transport (``tests/cache_discipline_helpers.py``),
so nothing leaves the process; a scripted reply is already whole and is used
as it is, so the actor's answer and requests must not change.
"""

from __future__ import annotations

import os

import pytest

from tests import cache_discipline_helpers as h
from unify.common import llm_client
from unify.settings import ProductionSettings, SETTINGS


def test_the_setting_is_off_by_default_and_takes_on_or_off():
    field = ProductionSettings.model_fields["UNIFY_STREAM_MODEL_CALLS"]
    assert field.default == ""
    parse = ProductionSettings.parse_stream_model_calls
    assert parse(" On ") == "on"
    assert parse("off") == ""
    assert parse(None) == ""
    with pytest.raises(ValueError):
        parse("yes")


@pytest.mark.parametrize("value, streamed", [("on", True), ("", False)])
def test_new_clients_and_their_forks_carry_the_flag(monkeypatch, value, streamed):
    monkeypatch.setattr(SETTINGS, "UNIFY_STREAM_MODEL_CALLS", value)
    client = h.new_client()
    clients = [
        client,
        llm_client.fork_llm_client(client, origin="compress_context"),
        llm_client.new_llm_client(h.MODEL, async_client=False),
        llm_client.new_slow_brain_llm_client(h.MODEL),
    ]
    assert [c.stream_transport for c in clients] == [streamed] * len(clients)


async def _act(monkeypatch, value: str):
    """``CodeActActor.act`` under the setting: its answer, its requests as
    sent, and whether each call asked the transport to stream."""
    asked: list = []
    answer = h.Provider.__call__

    async def call(self, **kw):
        asked.append(kw.get("stream"))
        return await answer(self, **kw)

    with monkeypatch.context() as patch:
        patch.setattr(SETTINGS, "UNIFY_STREAM_MODEL_CALLS", value)
        patch.setattr(h.Provider, "__call__", call)
        result, _, requests = await h.scenario_actor()
    return result, [h.request_bytes(r) for r in requests], asked


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_an_actor_request_is_answered_alike_streamed_or_not(monkeypatch):
    off_result, off_requests, off_asked = await _act(monkeypatch, "")
    on_result, on_requests, on_asked = await _act(monkeypatch, "on")

    assert on_result == off_result
    assert on_requests == off_requests
    assert set(off_asked) == {False}
    assert set(on_asked) == {True}


def test_the_idle_limit_is_empty_by_default_and_takes_seconds_or_off():
    field = ProductionSettings.model_fields["UNIFY_STREAM_IDLE_S"]
    assert field.default == ""
    parse = ProductionSettings.parse_stream_idle_s
    assert parse("") == "" and parse(None) == ""
    assert parse(" 90 ") == "90"
    assert parse("120.5") == "120.5"
    assert parse("off") == "0" and parse("0") == "0"
    for bad in ("-1", "soon", "nan", "inf", True):
        with pytest.raises(ValueError):
            parse(bad)


@pytest.mark.parametrize("streamed", ["on", ""])
def test_a_set_idle_limit_reaches_unillm_and_an_empty_one_leaves_it(monkeypatch, streamed):
    """unillm reads ``UNILLM_STREAM_IDLE_S`` on every streamed call."""
    monkeypatch.setattr(SETTINGS, "UNIFY_STREAM_MODEL_CALLS", streamed)
    monkeypatch.setenv("UNILLM_STREAM_IDLE_S", "33")
    monkeypatch.setattr(SETTINGS, "UNIFY_STREAM_IDLE_S", "")
    h.new_client()
    assert os.environ["UNILLM_STREAM_IDLE_S"] == "33"

    monkeypatch.setattr(SETTINGS, "UNIFY_STREAM_IDLE_S", "90")
    client = h.new_client()
    llm_client.fork_llm_client(client, origin="compress_context")
    assert os.environ["UNILLM_STREAM_IDLE_S"] == "90"


def test_unillm_reads_the_limit_it_is_given(monkeypatch):
    """The installed unillm has the silence limit and reads the variable passed to it."""
    from unillm.clients import uni_llm

    monkeypatch.delenv("UNILLM_STREAM_IDLE_S", raising=False)  # restored after the test
    monkeypatch.setattr(SETTINGS, "UNIFY_STREAM_IDLE_S", "45")
    llm_client.pass_stream_idle_limit()
    assert uni_llm.STREAM_IDLE_ENV == llm_client.UNILLM_STREAM_IDLE_ENV
    assert uni_llm._stream_idle_s() == 45.0
    monkeypatch.setattr(SETTINGS, "UNIFY_STREAM_IDLE_S", "0")
    llm_client.pass_stream_idle_limit()
    assert uni_llm._stream_idle_s() is None
