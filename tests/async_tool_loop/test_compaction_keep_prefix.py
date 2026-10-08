"""Symbolic: ``UNIFY_COMPACTION_KEEP_PREFIX`` keeps the sent prefix through a compaction.

As shipped, a compaction asks for its summary as a fork of the last request
(a full cache hit) and then restarts the session from the system prompt and
the summary alone: the request's own words are replaced by the model's
paraphrase, and only the tools and the system prompt stay a cached prefix.

With the switch on, the restarted session starts with what it already sent,
byte for byte: the tools, the system prompt, the session's first user
message and every requester message of the current request, and only then
the summary, as one loop-authored message. The current request is read from
the session's own messages: it starts at the latest requester message (a
user message the loop did not author), with the requester messages just
before it that no model turn separates from it.

The model is scripted (``tests/scripted_model.py``) below unillm's request
building, so every request is recorded as the provider would receive it,
and nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests import cache_discipline_helpers as h
from tests.scripted_model import Always, ScriptedModel, reply, scripted
from unify.common._async_tool import cache_discipline as cd
from unify.common._async_tool.context_compression import (
    _COMPRESSED_HEADER,
    RESTART_NOTICE,
    current_request_messages,
    kept_prefix,
)
from unify.common._async_tool.messages import loop_user_notice
from unify.settings import SETTINGS

SYSTEM = "You are a scripted test agent."
SUMMARY = "Summary: looked once; the answer is 42."
CONTEXT = "Session context: the library is empty."
WAIT = 30


def canon(value) -> str:
    """Canonical JSON: the bytes compared."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


async def look() -> str:
    """Look again."""
    return "Nothing new."


@pytest.fixture
def keep_prefix(monkeypatch):
    def set_(value: str = "on") -> None:
        monkeypatch.setattr(SETTINGS, "UNIFY_COMPACTION_KEEP_PREFIX", value)

    set_()
    return set_


def _model(*, fork=None, compactor=None, actor=None) -> ScriptedModel:
    """Three requests; the third looks once, compresses on demand, answers."""
    scripts = {
        "actor": actor
        or [
            reply("first done"),
            reply("second done"),
            reply(calls=[("look", {})]),
            reply(calls=[("compress_context", {})]),
            reply("third done"),
        ],
        "compression_fork": fork or [reply(SUMMARY)],
    }
    if compactor is not None:
        scripts["compactor"] = compactor
    return ScriptedModel(**scripts)


async def _session(model, requests=("First", "Second", "Third"), **loop_kwargs):
    """A persistent session answering *requests* in turn; its result is
    driven in the background, as ``unify act --persist`` does, since the
    handle restarts a compacted loop from ``result()``."""
    from unify.common.async_tool_loop import start_async_tool_loop

    client = h.new_client(SYSTEM)
    with scripted(model):
        handle = start_async_tool_loop(
            client,
            requests[0],
            {"look": look},
            log_steps=False,
            timeout=WAIT,
            persist=True,
            **loop_kwargs,
        )
        driver = asyncio.ensure_future(handle.result())
        answers = [(await h._next_response(handle))["content"]]
        for text in requests[1:]:
            await handle.submit(text)
            answers.append((await h._next_response(handle))["content"])
        await handle.stop()
        await asyncio.wait_for(driver, WAIT)
    return client, handle, answers


def _around_compaction(model):
    """The last actor request before the compaction and the first after it."""
    at = next(
        i
        for i, kind in enumerate(model.kinds())
        if kind in ("compression_fork", "compactor")
    )
    before = [c for c in model.calls[:at] if c.kind == "actor"][-1]
    after = next(c for c in model.calls[at:] if c.kind == "actor")
    return before, after


def _user(messages, content):
    return next(
        m for m in messages if m["role"] == "user" and m.get("content") == content
    )


def _pointer(client) -> str:
    return client._unify_transcript.pointer_line()


def _summary_text(client, summary=SUMMARY) -> str:
    return f"{_COMPRESSED_HEADER}{summary}\n\n{RESTART_NOTICE}\n\n{_pointer(client)}"


# ── the switch ───────────────────────────────────────────────────────────


def test_the_switch_is_validated():
    from unify.settings import ProductionSettings

    assert ProductionSettings().UNIFY_COMPACTION_KEEP_PREFIX == ""
    assert (
        ProductionSettings(
            UNIFY_COMPACTION_KEEP_PREFIX="off",
        ).UNIFY_COMPACTION_KEEP_PREFIX
        == ""
    )
    assert (
        ProductionSettings(
            UNIFY_COMPACTION_KEEP_PREFIX=" On ",
        ).UNIFY_COMPACTION_KEEP_PREFIX
        == "on"
    )
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_COMPACTION_KEEP_PREFIX="yes")


# ── the current request, read from the session's messages ────────────────


def _requester(text, **extra):
    return {"role": "user", "content": text, **extra}


def _assistant(text=None, calls=None):
    return {"role": "assistant", "content": text, "tool_calls": calls}


def test_the_current_request_starts_at_the_latest_requester_message():
    first = _requester("First")
    second = _requester("Second", _interjection=True)
    third = _requester("Third", _interjection=True)
    history = [
        {"role": "system", "content": SYSTEM},
        first,
        _assistant("first done"),
        second,
        _assistant("second done"),
        third,
        _assistant(calls=[{"id": "c1"}]),
        {"role": "tool", "tool_call_id": "c1", "content": "x"},
        loop_user_notice("a notice"),
        _assistant(calls=[{"id": "c2"}]),
    ]
    assert current_request_messages(history) == [third]
    assert current_request_messages(history)[0] is third
    system, kept = kept_prefix(history)
    assert system == [history[0]] and system[0] is history[0]
    assert kept == [first, third]
    assert all(k is m for k, m in zip(kept, [first, third]))


def test_requester_messages_that_arrived_together_are_one_request():
    """Messages no model turn separates (a seeded batch, or several queued
    for one turn boundary) are one request; loop-authored ones between them
    are not kept."""
    first = _requester("First")
    a, b = _requester("A", _interjection=True), _requester("B", _interjection=True)
    notice = loop_user_notice("record block", _record_block=True)
    history = [
        {"role": "system", "content": SYSTEM},
        first,
        _assistant("done"),
        a,
        notice,
        b,
        _assistant(calls=[{"id": "c1"}]),
    ]
    assert current_request_messages(history) == [a, b]
    assert kept_prefix(history)[1] == [first, a, b]


def test_the_first_message_is_not_kept_twice():
    first = _requester("Only request")
    history = [
        {"role": "system", "content": SYSTEM},
        first,
        _assistant(calls=[{"id": "c1"}]),
        {"role": "tool", "tool_call_id": "c1", "content": "x"},
        # an earlier summary is loop-authored, so not a request
        loop_user_notice("## Compressed Prior Context\nold"),
    ]
    assert current_request_messages(history) == [first]
    assert kept_prefix(history) == ([history[0]], [first])


def test_without_a_requester_message_only_the_first_user_message_is_kept():
    notice = loop_user_notice("loop only")
    history = [{"role": "system", "content": SYSTEM}, notice]
    assert current_request_messages(history) == []
    assert kept_prefix(history) == ([history[0]], [notice])
    assert kept_prefix([]) == ([], [])


# ── (a) the byte prefix ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_after_a_compaction_the_request_starts_with_what_was_sent(keep_prefix):
    model = _model()
    client, _handle, answers = await _session(model)
    assert answers == ["first done", "second done", "third done"]
    assert model.kinds().count("compression_fork") == 1
    model.assert_used_up()

    fork = model.of("compression_fork")[0]
    before, after = _around_compaction(model)
    # The summary request is unchanged: the last request plus the instruction.
    assert canon(fork.messages[:-1]) == canon(before.messages)
    assert fork.messages[-1]["content"] == cd.COMPRESSION_FORK_INSTRUCTION

    sent = before.messages
    expected = [
        sent[0],  # the system prompt
        _user(sent, "First"),  # the session's first user message
        _user(sent, "Third"),  # the current request
    ]
    assert sent[0]["role"] == "system" and sent[0]["content"] == SYSTEM
    assert [canon(m) for m in after.messages[:3]] == [canon(m) for m in expected]
    # Then the summary, and nothing else.
    assert after.messages[3] == {"role": "user", "content": _summary_text(client)}
    assert len(after.messages) == 4
    # Same tools, in the same order.
    assert canon(after.request["tools"]) == canon(before.request["tools"])
    assert after.tool_names == before.tool_names
    # The earlier request and every answer are in the summary only.
    text = canon(after.messages)
    for gone in ("Second", "first done", "second done", "Nothing new."):
        assert gone not in text


@pytest.mark.asyncio
async def test_the_first_message_keeps_its_session_context(keep_prefix):
    """The first message opens with the session context; it is kept with it,
    and the summary is not given the context again."""
    model = _model()
    client, _handle, _answers = await _session(model, first_message_context=CONTEXT)
    before, after = _around_compaction(model)
    first = before.messages[1]
    assert first["content"] == f"{CONTEXT}\n\n---\n\nFirst"
    assert canon(after.messages[1]) == canon(first)
    assert after.messages[-1]["content"] == _summary_text(client)
    assert canon(after.messages).count(CONTEXT) == 1


# ── (b) the cache affinity key ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_cache_affinity_key_is_unchanged(keep_prefix, monkeypatch):
    sets = h.install_affinity_api(monkeypatch)
    model = _model()
    client, handle, _answers = await _session(model)
    # One key throughout: the session sets it before its first request, and
    # the compression fork (fork_llm_client) is given the same key, its
    # parent's; nothing sets another.
    assert sets
    key = sets[0][0]
    assert {k for k, _ in sets} == {key}
    assert client.cache_affinity == key
    # It still names the prefix the session has after the compaction.
    assert client.system_message == SYSTEM
    assert (
        cd.prefix_affinity_key(
            client.endpoint,
            client.system_message,
            handle._runtime_state.session_tools_schema,
        )
        == key
    )


# ── (c) a persistent session keeps only the current request ──────────────


@pytest.mark.asyncio
async def test_a_persistent_session_keeps_only_the_current_request(keep_prefix):
    model = _model(
        actor=[
            reply("first done"),
            reply("second done"),
            reply(calls=[("look", {})]),
            reply(calls=[("compress_context", {})]),
            reply("third done"),
            reply("fourth done"),
        ],
    )
    client, _handle, answers = await _session(
        model,
        requests=("First", "Second", "Third", "Fourth"),
    )
    assert answers == ["first done", "second done", "third done", "fourth done"]
    _before, after = _around_compaction(model)
    users = [m["content"] for m in after.messages if m["role"] == "user"]
    assert users == ["First", "Third", _summary_text(client)]
    # The next request follows the summary, and the prefix stays put.
    last = model.of("actor")[-1]
    assert [canon(m) for m in last.messages[:4]] == [canon(m) for m in after.messages]
    assert last.messages[-1] == {"role": "user", "content": "Fourth"}


# ── (d) off: the rebuild as shipped ──────────────────────────────────────


@pytest.mark.asyncio
async def test_off_the_rebuild_is_as_shipped(keep_prefix):
    keep_prefix("")
    model = _model()
    client, _handle, answers = await _session(model, first_message_context=CONTEXT)
    assert answers == ["first done", "second done", "third done"]
    _before, after = _around_compaction(model)
    # Pinned: the system prompt, then the summary given the session context.
    summary = (
        f"## Compressed Prior Context\n{SUMMARY}\n\n"
        "Context was compressed. Continue from where you left off."
        f"\n\n{_pointer(client)}"
    )
    assert after.messages == [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{CONTEXT}\n\n---\n\n{summary}"},
    ]
    restarted = [m for m in client.messages if m.get("role") == "user"][0]
    assert "_loop_authored" not in restarted


@pytest.mark.asyncio
async def test_off_the_fallback_rebuild_is_as_shipped(keep_prefix):
    keep_prefix("")
    model = _model(fork=[reply("")], compactor=[Always(reply("Compacted."))])
    client, _handle, _answers = await _session(model)
    before, after = _around_compaction(model)
    # The compactor's rebuild: system messages only, the compressed entries
    # in a system message of their own, and a restart notice.
    assert [m["role"] for m in after.messages] == ["system", "system", "user"]
    assert after.messages[1]["content"].startswith(_COMPRESSED_HEADER)
    assert "unpack_messages" in after.messages[1]["content"]
    assert after.messages[2]["content"] == RESTART_NOTICE
    assert client.system_message is None


# ── (e) the fallback compactor keeps the same prefix ─────────────────────


@pytest.mark.asyncio
async def test_the_fallback_compactor_keeps_the_same_prefix(keep_prefix):
    model = _model(fork=[reply("")], compactor=[Always(reply("Compacted."))])
    client, _handle, answers = await _session(model)
    assert answers == ["first done", "second done", "third done"]
    assert model.kinds().count("compactor") >= 1
    before, after = _around_compaction(model)
    sent = before.messages
    expected = [sent[0], _user(sent, "First"), _user(sent, "Third")]
    assert [canon(m) for m in after.messages[:3]] == [canon(m) for m in expected]
    assert len(after.messages) == 4
    summary = after.messages[3]
    assert summary["role"] == "user"
    assert summary["content"].startswith(_COMPRESSED_HEADER + "[")
    assert summary["content"].endswith(
        f"\n\n{RESTART_NOTICE}\n\n{_pointer(client)}",
    )
    assert "unpack_messages" not in summary["content"]
    assert canon(after.request["tools"]) == canon(before.request["tools"])
    assert client.system_message == SYSTEM


# ── (f) the marker stays inside the harness ──────────────────────────────


def _keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


@pytest.mark.asyncio
async def test_the_summary_is_loop_authored_but_the_mark_is_not_sent(keep_prefix):
    model = _model()
    client, _handle, _answers = await _session(model)
    summary = next(
        m
        for m in client.messages
        if m.get("role") == "user"
        and str(m.get("content")).startswith(_COMPRESSED_HEADER)
    )
    assert summary["_loop_authored"] is True
    # The kept requester message keeps its own internal mark too.
    third = next(m for m in client.messages if m.get("content") == "Third")
    assert third.get("_interjection") is True
    for call in model.calls:
        assert not [k for k in _keys(call.request["messages"]) if k.startswith("_")]


# ── edge cases ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_huge_first_message_is_kept_and_the_tail_compacted(keep_prefix):
    """A first message over the context threshold on its own is the request;
    it is kept, and only what followed it is summarised."""
    from unify.common.async_tool_loop import start_async_tool_loop

    huge = "data " * 50_000

    def actor(call):
        if any(SUMMARY in str(m.get("content")) for m in call.messages):
            return reply("done")
        if any(m["role"] == "tool" for m in call.messages):
            return reply(calls=[("compress_context", {})])
        return reply(calls=[("look", {})], prompt_tokens=900_000)

    model = ScriptedModel(
        actor=Always(actor),
        compress_turn=Always(actor),
        compression_fork=[reply(SUMMARY)],
    )
    client = h.new_client(SYSTEM)
    with scripted(model):
        handle = start_async_tool_loop(
            client,
            huge,
            {"look": look},
            log_steps=False,
            timeout=WAIT,
        )
        assert await asyncio.wait_for(handle.result(), WAIT) == "done"
    before, after = _around_compaction(model)
    assert canon(after.messages[:2]) == canon(before.messages[:2])
    assert after.messages[1]["content"] == huge
    assert after.messages[2] == {"role": "user", "content": _summary_text(client)}
    assert len(after.messages) == 3


@pytest.mark.asyncio
async def test_a_kept_prefix_over_the_threshold_is_not_compacted_again(keep_prefix):
    """When what a compaction kept is itself over the threshold, the first
    call after it is too, and the next compaction rebuilds as shipped from
    the summary alone: two compactions, not one per turn until the limit."""
    from unify.common.async_tool_loop import start_async_tool_loop

    huge = "data " * 50_000

    def actor(call):
        # The kept huge first message fills the context: the model looks once
        # and then compresses (the forced turn keeps the session's tools, so
        # it is an actor call); without it, it looks once and answers.
        full = any(m.get("content") == huge for m in call.messages)
        looked = any(m["role"] == "tool" for m in call.messages)
        if full:
            if looked:
                return reply(calls=[("compress_context", {})], prompt_tokens=900_000)
            return reply(calls=[("look", {})], prompt_tokens=900_000)
        if call.messages[-1]["role"] == "tool":
            return reply("done", prompt_tokens=1_000)
        return reply(calls=[("look", {})], prompt_tokens=1_000)

    model = ScriptedModel(
        actor=Always(actor),
        compress_turn=Always(actor),
        compression_fork=Always(lambda call: reply(SUMMARY)),
    )
    client = h.new_client(SYSTEM)
    with scripted(model):
        handle = start_async_tool_loop(
            client,
            huge,
            {"look": look},
            log_steps=False,
            timeout=WAIT,
            max_steps=60,
        )
        assert await asyncio.wait_for(handle.result(), WAIT) == "done"
    assert model.kinds().count("compression_fork") == 2
    assert handle._runtime_state.keep_prefix_fallbacks == 1
    # The first compaction kept the huge message; the second did not.
    forks = [i for i, k in enumerate(model.kinds()) if k == "compression_fork"]
    after_first = next(c for c in model.calls[forks[0] :] if c.kind == "actor")
    after_second = next(c for c in model.calls[forks[1] :] if c.kind == "actor")
    assert any(m.get("content") == huge for m in after_first.messages)
    assert not any(m.get("content") == huge for m in after_second.messages)


@pytest.mark.asyncio
async def test_a_record_block_queued_during_the_compaction_follows_the_summary(
    keep_prefix,
):
    blocks: list[str] = []

    def fork(call):
        blocks.append("Record: a helper finished.")
        return reply(SUMMARY)

    async def on_turn_boundary():
        return blocks.pop(0) if blocks else None

    model = _model(fork=[fork])
    client, _handle, _answers = await _session(model, on_turn_boundary=on_turn_boundary)
    _before, after = _around_compaction(model)
    users = [m["content"] for m in after.messages if m["role"] == "user"]
    assert users == [
        "First",
        "Third",
        _summary_text(client),
        "Record: a helper finished.",
    ]
    sent = [canon(c.request["messages"]) for c in model.calls]
    assert sum("Record: a helper finished." in s for s in sent) == 1


@pytest.mark.asyncio
async def test_the_transcript_gets_the_history_and_the_kept_context(keep_prefix):
    from unify import transcripts

    model = _model()
    client, _handle, _answers = await _session(model)
    session = client._unify_transcript
    pointer = session.pointer_line()
    transcripts.close(client)
    lines = [json.loads(line) for line in session.path.read_text().splitlines()]
    kinds = [line["type"] for line in lines]
    at = kinds.index("compaction")
    before = canon(lines[:at])
    for text in ("Second", "second done", "Nothing new."):
        assert text in before
    context = lines[at]["context"]
    assert [m.get("content") for m in context] == [
        SYSTEM,
        "First",
        "Third",
        _summary_text(client),
    ]
    assert pointer in context[-1]["content"]
    # Kept messages are not written again after the compaction line.
    after = lines[at + 1 :]
    assert not [
        ln
        for ln in after
        if ln["type"] == "message"
        and ln["message"].get("content") in ("First", "Third")
    ]
    assert any(
        ln["type"] == "message" and ln["message"].get("content") == "third done"
        for ln in after
    )
