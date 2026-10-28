"""Symbolic: the loop adds no tool of its own, not even to compress.

The loop adds no ``wait``, ``steer``, ``ask_about_completed_tool`` or
``compress_context``, and announces no call as ``[steerable ...]``: every
request carries the caller's tools alone. The actor's kind of loop
(``tools_in_code``) compacts by itself at the compaction point, with no forced
turn. The model is a scripted transport; nothing leaves the process.
"""

from __future__ import annotations

import pytest

from tests import cache_discipline_helpers as h


def _tool_names(request: dict) -> list[str]:
    return [t["function"]["name"] for t in request["tools"] or []]


@pytest.mark.asyncio
async def test_a_compacting_session_carries_only_the_callers_tools():
    result, _counter, requests = await h.scenario_compress()
    assert result == "done"
    assert all(_tool_names(r) == ["execute_code"] for r in requests)
    assert all(r.get("tool_choice") in (None, "auto") for r in requests)
    for request in requests:
        for message in request["messages"]:
            text = str(message.get("content") or "")
            assert "[steerable" not in text and "[askable" not in text


@pytest.mark.asyncio
async def test_a_plain_loop_adds_no_tool():
    _result, _counter, requests = await h.scenario_interrupt()
    assert _tool_names(requests[0]) == ["execute_code"]
