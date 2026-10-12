"""The agent loop's compaction never carries a credential into what the model or the records see: with a provider
key, a gateway URL and key, and a header value planted in the environment and in the client's own configuration, a
loop driven past the compaction point by a scripted model compacts through its fork summary, and none of the planted
values appears in any request the model is sent (its messages, tools and tool choice: the summary request and the
restart included), in the conversation the loop continues from, in the session transcript or in the loop's events.
A teammate on the agent record runs the same loop (``actor.act``), so this covers it too.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests import cache_discipline_helpers as h
from tests.scripted_model import Always, ScriptedModel, reply, scripted
from unify.common._async_tool import context_compression as cc

TOKEN = "planted-provider-key-for-the-compaction-test"
GATEWAY_URL = "https://gateway-planted.invalid/v1"
GATEWAY_KEY = "planted-gateway-key-for-the-compaction-test"
HEADER = "planted-header-value-for-the-compaction-test"
PLANTED = (TOKEN, GATEWAY_URL, "gateway-planted.invalid", GATEWAY_KEY, HEADER)
SUMMARY = "Summary: looked once; the answer is 42."


def look() -> str:
    """Look again."""
    return "looked"


def _answer(call):
    if any(SUMMARY in str(m.get("content")) for m in call.messages):
        return reply("done")
    if any(m["role"] == "tool" for m in call.messages):
        return reply(calls=[("compress_context", {})])
    return reply(calls=[("look", {})], prompt_tokens=cc.COMPACTION_AT_TOKENS)


@pytest.mark.asyncio
async def test_no_planted_credential_reaches_the_compaction_or_its_records(monkeypatch):
    from unify.common._async_tool import loop as loop_mod
    from unify.common.async_tool_loop import start_async_tool_loop
    from unify.common.llm_client import new_llm_client

    monkeypatch.setenv("OPENROUTER_API_KEY", TOKEN)
    monkeypatch.setenv("UNILLM_LLM_GATEWAY_URL", GATEWAY_URL)
    monkeypatch.setenv("UNILLM_LLM_GATEWAY_KEY", GATEWAY_KEY)
    events: list = []
    real = loop_mod.to_event_bus

    async def recording(msg, *args, **kwargs):
        events.append(msg)
        return await real(msg, *args, **kwargs)

    monkeypatch.setattr(loop_mod, "to_event_bus", recording)
    client = new_llm_client(
        h.MODEL,
        cache=False,
        reasoning_effort="low",
        extra_headers={"X-Planted": HEADER},
    )
    client.set_system_message("You are a scripted test agent.")
    model = ScriptedModel(
        actor=Always(_answer),
        compress_turn=Always(_answer),
        compression_fork=[reply(SUMMARY)],
    )
    with scripted(model):
        handle = start_async_tool_loop(
            client,
            "Look, then answer.",
            {"look": look},
            log_steps=False,
            timeout=30,
        )
        assert await asyncio.wait_for(handle.result(), 30) == "done"

    # the loop compacted through its fork summary and restarted from the summary
    assert "compression_fork" in model.kinds()
    assert any(SUMMARY in str(m.get("content")) for m in client.messages)

    seen = {
        "requests the model is sent": json.dumps(
            [
                {
                    k: c.request.get(k)
                    for k in ("messages", "tools", "tool_choice", "response_format")
                }
                for c in model.calls
            ],
            default=str,
        ),
        "the conversation after the restart": json.dumps(client.messages, default=str),
        "the loop's events": json.dumps(events, default=str),
    }
    session = getattr(client, "_unify_transcript", None)
    if session is not None and session.path.exists():
        seen["the session transcript"] = session.path.read_text(encoding="utf-8")
    for where, text in seen.items():
        for planted in PLANTED:
            assert planted not in text, (planted, where)
    assert events, "the loop published no events"
