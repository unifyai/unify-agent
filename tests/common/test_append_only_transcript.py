"""Unit tests for the sent-watermark append-only transcript invariant.

Covers the mechanism itself (is_mutable, generate_with_preprocess's watermark
advancement and dev-mode integrity assertion) plus the below-watermark gate
of insert_tool_message_after_assistant and the check_status pair it falls
back to. No LLM calls — these are symbolic tests of the
transcript-manipulation infrastructure, not of model behaviour.
"""

from __future__ import annotations

import asyncio

import pytest

from unify.common._async_tool.messages import (
    emit_completion_pair,
    generate_with_preprocess,
    insert_tool_message_after_assistant,
    is_mutable,
)
from unify.common._async_tool.tools_utils import (
    ToolCallMetadata,
    create_tool_call_message,
)

# ---------------------------------------------------------------------------
# Fakes — no LLM, no event bus, just enough surface for the transcript helpers
# ---------------------------------------------------------------------------


class _FakeClient:
    def __init__(self, messages=None):
        self.messages = list(messages) if messages else []

    def append_messages(self, value):
        self.messages += value
        return self


class _FakeMsgDispatcher:
    def __init__(self, client):
        self._client = client
        self.published: list = []

    async def append_msgs(self, msgs, origin=None, *, skip_event_bus=False, kind=None):
        self._client.append_messages(msgs)

    async def publish_to_event_bus(self, msgs, origin=None, kind=None):
        self.published.append(msgs)


class _FakeLogger:
    log_steps = False

    def debug(self, *a, **k):
        pass

    def info(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass


def _make_metadata(**overrides) -> ToolCallMetadata:
    defaults = dict(
        name="mytool",
        call_id="c1",
        call_dict={"function": {"arguments": "{}"}},
        call_idx=0,
        chat_context=None,
        assistant_msg={"role": "assistant", "content": None, "tool_calls": []},
        is_interjectable=False,
        tool_schema={},
        llm_arguments={},
        raw_arguments_json="{}",
    )
    defaults.update(overrides)
    return ToolCallMetadata(**defaults)


# ---------------------------------------------------------------------------
# 1. is_mutable — the one enforceable choke point
# ---------------------------------------------------------------------------


def test_is_mutable_respects_watermark_by_identity():
    client = _FakeClient([{"i": 0}, {"i": 1}, {"i": 2}])
    client._sent_watermark = 2

    assert is_mutable(client, client.messages[0]) is False
    assert is_mutable(client, client.messages[1]) is False
    assert is_mutable(client, client.messages[2]) is True

    # A structurally-identical-but-distinct dict is never confused for the
    # sent one — identity, not equality.
    assert is_mutable(client, {"i": 0}) is False

    # Absent from the transcript entirely — fails closed (immutable), not
    # open. A message missing from the transcript (e.g. a swapped-out
    # canonical log during a concurrent dispatch) must route through the
    # tail-append paths that actually reach the model, not be written into
    # a dict the transcript will never contain.
    assert is_mutable(client, {"unrelated": True}) is False


# ---------------------------------------------------------------------------
# 2. generate_with_preprocess — watermark set point + dev-mode hash assertion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watermark_advances_monotonically_on_pre_copy_length():
    client = _FakeClient([{"role": "user", "content": "hi"}])

    async def fake_generate(**kwargs):
        return "ok"

    client.generate = fake_generate

    await generate_with_preprocess(client, None)
    assert client._sent_watermark == 1

    client.messages.append({"role": "assistant", "content": "reply"})
    client.messages.append({"role": "user", "content": "follow-up"})
    await generate_with_preprocess(client, None)
    assert client._sent_watermark == 3

    # Monotonic: explicitly forcing a lower watermark (as if some caller
    # tried to regress it) never sticks — max() with the pre-copy length wins.
    client._sent_watermark = 1
    client._sent_watermark_hash = (
        None  # bypass the dev-mode check for this contrived probe
    )
    await generate_with_preprocess(client, None)
    assert client._sent_watermark == 3


@pytest.mark.asyncio
async def test_unanswered_dispatch_rolls_the_watermark_back():
    client = _FakeClient([{"role": "user", "content": "hi"}])
    started = asyncio.Event()

    async def hanging_generate(**kwargs):
        started.set()
        await asyncio.Event().wait()

    client.generate = hanging_generate

    task = asyncio.create_task(generate_with_preprocess(client, None))
    await started.wait()
    # Set before the await point, so nothing edits what the in-flight
    # request carries...
    assert client._sent_watermark == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # ...and undone when the request is cancelled without an answer.
    assert client._sent_watermark == 0
    assert client._sent_watermark_hash is None

    async def failing_generate(**kwargs):
        raise RuntimeError("provider error")

    client.generate = failing_generate
    with pytest.raises(RuntimeError, match="provider error"):
        await generate_with_preprocess(client, None)
    assert client._sent_watermark == 0


@pytest.mark.asyncio
async def test_dev_mode_assertion_fires_on_below_watermark_mutation():
    client = _FakeClient([{"role": "user", "content": "hi"}])

    async def fake_generate(**kwargs):
        return "ok"

    client.generate = fake_generate

    await generate_with_preprocess(client, None)
    assert client._sent_watermark == 1

    # Simulate the mid-history rewrite this invariant forbids: an
    # already-dispatched message mutated in place.
    client.messages[0]["content"] = "mutated after being sent"
    client.messages.append({"role": "assistant", "content": "reply"})

    with pytest.raises(
        AssertionError,
        match="Append-only transcript invariant violated",
    ):
        await generate_with_preprocess(client, None)


# ---------------------------------------------------------------------------
# 3. ensure_placeholders_for_pending — self-describing, always-bypass stub
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4. insert_tool_message_after_assistant — the watermark gate + escape hatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_below_watermark_routes_to_check_status_pair():
    asst_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "mytool", "arguments": "{}"},
            },
        ],
    }
    stub = create_tool_call_message(name="mytool", call_id="c1", content="pending")
    client = _FakeClient([asst_msg, stub])
    client._sent_watermark = 2  # both messages already dispatched
    dispatcher = _FakeMsgDispatcher(client)
    assistant_meta = {}

    reply = create_tool_call_message(
        name="mytool",
        call_id="c1",
        content="the real result",
    )
    await insert_tool_message_after_assistant(
        assistant_meta,
        asst_msg,
        reply,
        client,
        dispatcher,
    )

    # Below-watermark bytes are untouched — no splice happened.
    assert client.messages[0] is asst_msg
    assert client.messages[1] is stub
    assert stub["content"] == "pending"
    assert len(client.messages) == 4  # + synthetic assistant/tool check_status pair

    check_stub, check_reply = client.messages[2], client.messages[3]
    assert check_stub["role"] == "assistant"
    assert check_stub["tool_calls"][0]["function"]["name"] == "check_status_c1"
    assert check_reply["role"] == "tool"
    assert check_reply["tool_call_id"] == "c1_completed"
    assert check_reply["content"] == "the real result"


@pytest.mark.asyncio
async def test_insert_below_watermark_with_bypass_splices_adjacently():
    asst_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "wait", "arguments": "{}"},
            },
        ],
    }
    client = _FakeClient([{"role": "user", "content": "hi"}, asst_msg])
    dispatcher = _FakeMsgDispatcher(client)
    assistant_meta = {}

    async def fake_generate(**kwargs):
        return "ok"

    client.generate = fake_generate

    # A real prior dispatch, so both the watermark AND its stored hash
    # baseline reflect asst_msg genuinely having been sent — reproduces a
    # real backfill/restore splice after a genuine dispatch, not just a
    # hand-set watermark with no hash to violate.
    await generate_with_preprocess(client, None)
    assert client._sent_watermark == 2

    ack = create_tool_call_message(name="wait", call_id="c1", content="Acknowledged.")
    await insert_tool_message_after_assistant(
        assistant_meta,
        asst_msg,
        ack,
        client,
        dispatcher,
        bypass_watermark=True,
    )

    # Escape hatch: spliced directly after asst_msg despite being below-mark —
    # legality (an otherwise permanently-unanswered tool_calls entry) beats cache.
    assert client.messages[1] is asst_msg
    assert client.messages[2] is ack

    # The escape-hatch splice must re-baseline the stored watermark hash —
    # without it, the next dispatch's integrity check would read this
    # sanctioned, legal splice as an unsanctioned mutation and raise.
    await generate_with_preprocess(client, None)  # must not raise


@pytest.mark.asyncio
async def test_insert_above_watermark_splices_normally():
    asst_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "mytool", "arguments": "{}"},
            },
        ],
    }
    client = _FakeClient([asst_msg])
    client._sent_watermark = 0  # nothing dispatched yet
    dispatcher = _FakeMsgDispatcher(client)
    assistant_meta = {}

    reply = create_tool_call_message(name="mytool", call_id="c1", content="result")
    await insert_tool_message_after_assistant(
        assistant_meta,
        asst_msg,
        reply,
        client,
        dispatcher,
    )

    assert client.messages == [asst_msg, reply]


# ---------------------------------------------------------------------------
# 5. prune_wait_tool_call — below-mark leaves tool_calls untouched, acks instead
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6. record_progress — coalesce-then-freeze, separate from tool_reply_msg
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 7. record_clarification — separate tail message, never touches tool_reply_msg
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 8. emit_completion_pair — shape of the sole below-watermark result path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emit_completion_pair_shape():
    client = _FakeClient()
    dispatcher = _FakeMsgDispatcher(client)

    tool_msg = await emit_completion_pair("the result", "c1", dispatcher)

    assert len(client.messages) == 2
    stub, reply = client.messages
    assert stub["role"] == "assistant"
    assert stub["tool_calls"][0]["id"] == "c1_completed"
    assert stub["tool_calls"][0]["function"]["name"] == "check_status_c1"
    assert reply is tool_msg
    assert reply["tool_call_id"] == "c1_completed"
    assert reply["content"] == "the result"


# ---------------------------------------------------------------------------
# 9. multi-handle: FINAL per-child result vs shared placeholder freeze
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 10. process_completed_task — the placeholder-result-write site end to end
# ---------------------------------------------------------------------------
