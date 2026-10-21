"""Symbolic: every model call the harness makes is streamed.

Every client ``new_llm_client`` makes, and every fork of one, carries unillm's
``stream_transport`` flag: its calls go out as streams that unillm reads whole
and reassembles into the completion an unstreamed call returns. Requests are
read at unillm's transport (``tests/cache_discipline_helpers.py``), so nothing
leaves the process.
"""

from __future__ import annotations

import pytest

from tests import cache_discipline_helpers as h
from unify.common import llm_client


def test_new_clients_and_their_forks_stream():
    client = h.new_client()
    clients = [
        client,
        llm_client.fork_llm_client(client, origin="compress_context"),
        llm_client.new_llm_client(h.MODEL, async_client=False),
    ]
    assert [c.stream_transport for c in clients] == [True] * len(clients)


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_every_call_of_an_actor_request_asks_for_a_stream(monkeypatch):
    asked: list = []
    answer = h.Provider.__call__

    async def call(self, **kw):
        asked.append(kw.get("stream"))
        return await answer(self, **kw)

    monkeypatch.setattr(h.Provider, "__call__", call)
    await h.scenario_actor()
    assert asked and set(asked) == {True}
