"""Symbolic: ``UNIFY_STREAM_MODEL_CALLS`` and the clients whose calls it streams.

With the setting on, every client ``new_llm_client`` makes, and every fork of
one, carries unillm's ``stream_transport`` flag: its calls go out as streams
that unillm reads whole and reassembles. Off, the clients are as shipped.
Requests are read at unillm's transport (``tests/cache_discipline_helpers.py``),
so nothing leaves the process; a scripted reply is already whole and is used
as it is, so the actor's answer and requests must not change.
"""

from __future__ import annotations

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
