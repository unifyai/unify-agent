"""Symbolic: under ``UNIFY_CACHE_DISCIPLINE`` compression is a fork of the conversation.

As shipped, compression hands the transcript to a separate compactor loop
with its own system prompt, which edits entries over several calls, and the
session restarts with a rewritten system prompt and an extra tool. None of
it is cached: one ScienceWorld compression cost 0.172 USD over 8 calls, and
the call after it had 0 cached tokens. With the switch on, the summary is
asked for with the last request sent plus one appended instruction, and the
session continues from its own system prompt, the same tool list and the
summary.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests import cache_discipline_helpers as h
from unify.common._async_tool import cache_discipline as cd
from unify.settings import SETTINGS


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", True)


def _dumps(messages: list[dict]) -> list[str]:
    return [json.dumps(m, default=str) for m in messages]


@pytest.mark.asyncio
async def test_the_summary_request_is_the_last_request_plus_one_message(on):
    result, counter, requests = await h.scenario_compress()
    assert result == "done"
    assert counter == {"execute_code": 1}
    assert len(requests) == 4
    last, summary = requests[1], requests[2]

    # Byte prefix: every message of the last request, unchanged, then one.
    assert _dumps(summary["messages"])[: len(last["messages"])] == _dumps(
        last["messages"],
    )
    assert len(summary["messages"]) == len(last["messages"]) + 1
    assert summary["messages"][-1] == {
        "role": "user",
        "content": cd.COMPRESSION_FORK_INSTRUCTION,
    }
    # Same tools; the forced choice of the context-full turn is sent as auto.
    assert h.request_bytes(summary)["tools"] == h.request_bytes(last)["tools"]
    assert last["tool_choice"] == "required"
    assert summary["tool_choice"] == "auto"
    assert summary["reasoning_effort"] == last["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_the_session_continues_from_the_summary_with_the_same_prefix(on):
    _result, _counter, requests = await h.scenario_compress()
    first, restarted = requests[0], requests[3]
    # The system prompt and the tool list survive the restart byte for byte.
    assert _dumps(restarted["messages"][:1]) == _dumps(first["messages"][:1])
    assert restarted["messages"][0]["content"] == "You are a scripted test agent."
    assert h.request_bytes(restarted)["tools"] == h.request_bytes(first)["tools"]
    assert "unpack_messages" not in h.request_bytes(restarted)["tools"]
    user = [m for m in restarted["messages"] if m["role"] == "user"]
    assert user[-1]["content"] == (
        "## Compressed Prior Context\n"
        f"{h.SUMMARY}\n\n"
        "Context was compressed. Continue from where you left off."
    )


@pytest.mark.asyncio
async def test_a_fork_that_returns_no_text_falls_back_to_the_compactor(on):
    replies = (
        h.COMPRESS_REPLIES[0],
        h.COMPRESS_REPLIES[1],
        # the fork calls a tool instead of summarising
        lambda: h.completion(calls=[("execute_code", {"code": "nope"})]),
        # the compactor as shipped, then the restarted session
        lambda: h.completion(content="compacted"),
        lambda: h.completion(content="done"),
    )
    result, counter, requests = await h.scenario_compress(replies)
    assert result == "done"
    assert counter == {"execute_code": 1}
    compactor = requests[3]
    assert compactor["messages"][0]["content"].startswith(
        "You are a context compactor",
    )


@pytest.mark.asyncio
async def test_without_a_recorded_request_the_fork_is_skipped(on):
    from unify.common.async_tool_loop import AsyncToolLoopHandle

    handle = SimpleNamespace(
        _client=h.new_client(),
        _log_label="test",
        _loop_id="test",
        _compression=SimpleNamespace(count=0),
    )
    assert cd.last_sent_request(handle._client) is None
    assert await AsyncToolLoopHandle._summarise_as_fork(handle, {"tools": {}}) is None
    assert handle._compression.count == 0


# ── the recorded request ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_dispatch_records_only_messages_tools_and_tool_choice(on):
    client = h.new_client()
    previous = cd.record_sent_request(
        client,
        [{"role": "user", "content": "x"}],
        {
            "tools": [{"type": "function", "function": {"name": "t"}}],
            "tool_choice": "auto",
            "api_key": "not-kept",  # pragma: allowlist secret
            "extra_headers": {"Authorization": "not-kept"},
        },
    )
    assert previous is None
    record = cd.last_sent_request(client)
    assert set(record) == {"messages", "tools", "tool_choice"}
    assert "not-kept" not in json.dumps(record)
    cd.restore_sent_request(client, previous)
    assert cd.last_sent_request(client) is None


@pytest.mark.asyncio
async def test_the_record_matches_what_was_sent_and_is_off_by_default(monkeypatch):
    from unify.common._async_tool.messages import generate_with_preprocess

    for switch in (False, True):
        monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", switch)
        client = h.new_client()
        client._messages.append({"role": "user", "content": "hello"})
        tools = [
            {
                "type": "function",
                "function": {"name": "t", "parameters": {"type": "object"}},
            },
        ]
        with h.scripted([lambda: h.completion(content="hi")]) as provider:
            await generate_with_preprocess(
                client,
                None,
                tools=tools,
                tool_choice="auto",
                return_full_completion=True,
                stateful=True,
            )
        record = cd.last_sent_request(client)
        if not switch:
            assert record is None
            continue
        sent = provider.requests[0]
        assert _dumps(record["messages"]) == _dumps(sent["messages"])
        assert record["tools"] == sent["tools"]
        assert record["tool_choice"] == sent["tool_choice"]


@pytest.mark.asyncio
async def test_a_dispatch_that_fails_puts_the_previous_record_back(on):
    from unify.common._async_tool.messages import generate_with_preprocess

    client = h.new_client()
    client._messages.append({"role": "user", "content": "one"})
    with h.scripted([lambda: h.completion(content="first")]):
        await generate_with_preprocess(
            client,
            None,
            return_full_completion=True,
            stateful=True,
        )
    first = cd.last_sent_request(client)
    client._messages.append({"role": "user", "content": "two"})

    def boom():
        raise RuntimeError("provider down")

    with h.scripted([boom]):
        with pytest.raises(RuntimeError):
            await generate_with_preprocess(
                client,
                None,
                return_full_completion=True,
                stateful=True,
            )
    assert cd.last_sent_request(client) is first


# ── the fork client ──────────────────────────────────────────────────────


def test_a_fork_keeps_model_and_effort_under_its_own_origin():
    from unify.common.llm_client import fork_llm_client, new_llm_client

    parent = new_llm_client(
        h.MODEL,
        cache=False,
        origin="CodeActActor.act",
        purpose="planning",
        reasoning_effort="high",
    )
    parent.set_system_message("sys")
    parent.set_reasoning_effort("low")  # copy() alone would restore "high"
    parent._messages.append({"role": "user", "content": "u"})

    snapshot = [{"role": "system", "content": "sys"}, {"role": "user", "content": "s"}]
    fork = fork_llm_client(
        parent,
        origin="StorageCheck",
        purpose="planning",
        messages=snapshot,
    )
    assert fork.endpoint == parent.endpoint
    assert fork.reasoning_effort == "low"
    assert fork.origin == "StorageCheck#purpose=planning"
    assert fork.system_message == "sys"
    assert fork.messages == snapshot and fork.messages is not snapshot
    fork._messages.append({"role": "user", "content": "fork only"})
    assert snapshot[-1]["content"] == "s"
    assert parent.messages[-1]["content"] == "u"
