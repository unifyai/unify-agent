"""Symbolic: a loop without the steering tools, offering compression tools only when it must compress.

``UNIFY_TOOL_SURFACE=core`` sends ``execute_code`` as the actor's only JSON tool
(unify/actor/core_surface.py). As shipped the loop adds ``compress_context``,
``wait``, ``steer`` and ``ask_about_completed_tool`` to every request and
announces each call as ``[steerable ...]``. With ``steering_tools=False`` (an
actor without sub-actors) it adds none of the steering tools or their
announcements; with ``compression_tools_on_demand=True`` it offers
``compress_context`` and the caller's extra compression tools only on the turn
it asks for compression -- exactly as that turn offers them as shipped -- and
leaves them out of every other request, ``UNIFY_CACHE_DISCIPLINE``'s fixed list
included. With neither parameter the loop is as shipped. The model is a
scripted transport; nothing leaves the process.
"""

from __future__ import annotations

import pytest

from tests import cache_discipline_helpers as h
from unify.settings import SETTINGS


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
@pytest.mark.parametrize("discipline", [False, True])
async def test_compression_tools_are_offered_only_on_the_turn_that_compresses(
    monkeypatch,
    discipline,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", discipline)
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
            interrupt_llm_with_interjections=False,
            extra_compression_tools=["store_skills"],
            compression_tools_on_demand=True,
            steering_tools=False,
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
async def test_without_the_parameters_the_loop_is_as_shipped():
    counter: dict = {}
    tools = h.make_tools(counter)
    with h.scripted(h.INTERRUPT_REPLIES) as provider:
        await h._run(
            h.new_client(),
            {"execute_code": tools["execute_code"]},
            "Do the task.",
        )
    names = _tool_names(provider.requests[0])
    assert names == [
        "execute_code",
        "compress_context",
        "wait",
        "steer",
        "ask_about_completed_tool",
    ]
    assert any(
        "[steerable" in str(m.get("content")) for m in provider.requests[-1]["messages"]
    )


@pytest.mark.asyncio
async def test_without_the_steering_tools_the_model_is_woken_once_per_batch(
    monkeypatch,
):
    """With no `wait` to call, no turn starts while a sibling call runs (as
    under UNIFY_BATCH_WAKE), and none is forced to call a tool."""
    from tests.async_tool_loop import test_lean_loop as lean

    requests, sent, handle = await lean._run(
        monkeypatch,
        [lean._batch("fast_tool", "medium_tool"), *lean._done()],
        switches=lean.SHIPPED,
        steering_tools=False,
    )
    assert len(requests) == 2
    assert lean._results_seen(requests[1]) == {"FAST_RESULT", "MEDIUM_RESULT"}
    assert not lean._pending_placeholder(requests[1])
    assert handle._runtime_state.cancelled_turns == 0
    assert all(r["tool_choice"] != "required" for r in requests)
    assert not any(lean._notices(r) for r in requests)
