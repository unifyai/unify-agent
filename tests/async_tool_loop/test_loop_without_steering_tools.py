"""Symbolic: the loop has no steering tools, and offers compression tools only when it must compress.

``UNIFY_TOOL_SURFACE=core`` sends ``execute_code`` as the actor's only JSON tool
(unify/actor/core_surface.py). The loop adds no ``wait``, ``steer`` or
``ask_about_completed_tool`` (it has none) and announces no call as
``[steerable ...]``; by default it adds ``compress_context`` to every request.
With ``compression_tools_on_demand=True`` it offers ``compress_context`` and
the caller's extra compression tools only on the turn it asks for
compression, and leaves them out of every other request, the session's
fixed list included. The model is a scripted
transport; nothing leaves the process.
"""

from __future__ import annotations

import pytest

from tests import cache_discipline_helpers as h


def _tool_names(request: dict) -> list[str]:
    return [t["function"]["name"] for t in request["tools"] or []]


# ── the loop: compression on demand, no steering surface ────────────────────


COMPRESS_THEN_ANSWER = (
    lambda: h.completion(
        calls=[("execute_code", {"code": "big"})],
        prompt_tokens=900_000,
    ),
    lambda: h.completion(calls=[("compress_context", {})]),
    lambda: h.completion(content=h.SUMMARY),
    lambda: h.completion(content="done"),
)


@pytest.mark.asyncio
async def test_compression_tools_are_offered_only_on_the_turn_that_compresses():
    counter: dict = {}
    tools = h.make_tools(counter)

    async def store_skills(request: str) -> str:
        """Store skills from the trajectory.

        Args:
            request: What to store.
        """
        return "stored nothing"

    with h.scripted(COMPRESS_THEN_ANSWER) as provider:
        result = await h._run(
            h.new_client(),
            {"execute_code": tools["execute_code"], "store_skills": store_skills},
            "Do the task.",
            extra_compression_tools=["store_skills"],
            compression_tools_on_demand=True,
        )
    assert result == "done"
    session = h.session_requests(provider.requests)
    # Every ordinary turn: the caller's tool and nothing else.
    assert _tool_names(session[0]) == ["execute_code"]
    # The turn that must compress: as shipped, compress_context and the
    # caller's extra compression tool.
    assert sorted(_tool_names(session[1])) == ["compress_context", "store_skills"]
    assert session[1]["tool_choice"] == "required"
    # After the restart the list is the session's own again (plus, as
    # shipped, the compactor's unpack_messages when it rewrote the history).
    assert set(_tool_names(session[-1])) - {"unpack_messages"} == {"execute_code"}
    # No call was announced as steerable or askable.
    for request in provider.requests:
        for message in request["messages"]:
            text = str(message.get("content") or "")
            assert "[steerable" not in text and "[askable" not in text


@pytest.mark.asyncio
async def test_by_default_the_loop_adds_only_compress_context():
    counter: dict = {}
    tools = h.make_tools(counter)
    with h.scripted(h.INTERRUPT_REPLIES) as provider:
        await h._run(
            h.new_client(),
            {"execute_code": tools["execute_code"]},
            "Do the task.",
        )
    names = _tool_names(provider.requests[0])
    assert names == ["execute_code", "compress_context"]
