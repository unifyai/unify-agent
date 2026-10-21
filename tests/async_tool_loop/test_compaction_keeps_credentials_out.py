"""The agent loop's compaction never carries a credential into what the model or the records see: with a provider
key, a gateway URL and key, and a header value planted in the environment and in the client's own configuration, a
loop driven past the compaction point by a scripted model compacts through its fork summary, and none of the planted
values appears in any request the model is sent (its messages, tools and tool choice: the summary request and the
restart included), in the conversation the loop continues from or in the session transcript. A teammate on the agent
record runs the same loop (``actor.act``), so this covers it too.
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


def execute_code(code: str) -> str:
    """Run code (the actor's kind of loop has no other tool)."""
    return "looked"


def _answer(call):
    if any(SUMMARY in str(m.get("content")) for m in call.messages):
        return reply("done")
    # this reply's prompt reaches the compaction point: the loop compacts before its next call
    return reply(calls=[("execute_code", {"code": "look()"})], prompt_tokens=cc.COMPACTION_AT_TOKENS)


@pytest.mark.asyncio
async def test_no_planted_credential_reaches_the_compaction_or_its_records(monkeypatch):
    from unify.common.async_tool_loop import start_async_tool_loop
    from unify.common.llm_client import new_llm_client

    monkeypatch.setenv("OPENROUTER_API_KEY", TOKEN)
    monkeypatch.setenv("UNILLM_LLM_GATEWAY_URL", GATEWAY_URL)
    monkeypatch.setenv("UNILLM_LLM_GATEWAY_KEY", GATEWAY_KEY)
    client = new_llm_client(
        h.MODEL,
        cache=False,
        reasoning_effort="low",
        extra_headers={"X-Planted": HEADER},
    )
    client.set_system_message("You are a scripted test agent.")
    model = ScriptedModel(
        actor=Always(_answer),
        compression_fork=[reply(SUMMARY)],
    )
    with scripted(model):
        handle = start_async_tool_loop(
            client,
            "Look, then answer.",
            {"execute_code": execute_code},
            log_steps=False,
            timeout=30,
            tools_in_code=True,
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
    }
    # the session transcript must exist here, so its check is never vacuous
    session = getattr(client, "_unify_transcript", None)
    assert (
        session is not None and session.path.exists()
    ), "no session transcript was written"
    seen["the session transcript"] = session.path.read_text(encoding="utf-8")
    assert (
        SUMMARY in seen["the session transcript"]
        or "look" in seen["the session transcript"]
    )
    for where, text in seen.items():
        for planted in PLANTED:
            assert planted not in text, (planted, where)
