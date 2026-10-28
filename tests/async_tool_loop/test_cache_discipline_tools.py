"""Symbolic: a session keeps one tool list (the cache discipline).

A provider reuses its prompt cache only as far as a request matches the last
one byte for byte, and the tool list comes first. The list is computed once
and sent unchanged on every call of a session, a compaction's restart
included.

The requests are captured at unillm's transport (see
``tests/cache_discipline_helpers.py``).
"""

from __future__ import annotations

import pytest

from tests import cache_discipline_helpers as h


def _names(request: dict) -> list[str]:
    return [t["function"]["name"] for t in request["tools"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", h.ONE_SESSION)
async def test_the_tool_list_is_identical_on_every_call(scenario):
    _result, _counter, requests = await h.SCENARIOS[scenario]()
    assert len(requests) >= 2
    tool_bytes = {h.request_bytes(r)["tools"] for r in requests}
    assert len(tool_bytes) == 1, [_names(r) for r in requests]


@pytest.mark.asyncio
async def test_the_list_holds_the_callers_tools_and_nothing_else():
    _result, _counter, requests = await h.scenario_interrupt()
    assert _names(requests[0]) == ["execute_code"]
