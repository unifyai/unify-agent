"""Unit tests for the sent-watermark append-only transcript invariant.

Covers the mechanism itself (is_mutable, generate_with_preprocess's watermark
advancement and dev-mode integrity assertion) plus every below-watermark
gated call site: the placeholder result write, notification delivery,
clarification delivery, insert_tool_message_after_assistant's below-watermark
gate, prune_wait_tool_call's below-watermark ack path, and multi-handle
child-result delivery. No LLM calls — these are symbolic tests of the
transcript-manipulation infrastructure, not of model behaviour.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from unify.common._async_tool.loop import ToolLoopRuntimeState, _LoopToolFailureTracker
from unify.common._async_tool.messages import (
    emit_completion_pair,
    ensure_placeholders_for_pending,
    generate_with_preprocess,
    insert_tool_message_after_assistant,
    is_mutable,
    prune_wait_tool_call,
)
from unify.common._async_tool.tools_data import ToolsData, _MultiHandleState
from unify.common._async_tool.tools_utils import (
    ToolCallMetadata,
    create_tool_call_message,
)
from tests.async_helpers import _is_bare_pending_tool_placeholder

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


@pytest.mark.asyncio
async def test_pending_stub_is_self_describing_and_stays_bare_pending():
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
    dispatcher = _FakeMsgDispatcher(client)
    tools_data = ToolsData(tools={}, client=client, logger=_FakeLogger())

    task = asyncio.ensure_future(asyncio.sleep(0))
    info = _make_metadata(assistant_msg=asst_msg)
    tools_data.save_task(task, info)

    created = await ensure_placeholders_for_pending(
        tools_data=tools_data,
        assistant_meta={},
        client=client,
        msg_dispatcher=dispatcher,
    )
    assert created == ["c1"]

    stub = info.tool_reply_msg
    assert stub is not None
    parsed = json.loads(stub["content"])
    assert parsed["_placeholder"] == "pending"
    assert "check_status" in parsed["meta:status"]
    # The existing shared test helper (used by ~15 tests via
    # _wait_for_tool_result) must still classify this as a bare-pending
    # placeholder despite the added self-describing text.
    assert _is_bare_pending_tool_placeholder(stub) is True

    # First-ever reply to a call_id is always spliced directly after its
    # assistant message — bypasses the watermark gate on principle.
    assert client.messages[1] is stub

    await task


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


@pytest.mark.asyncio
async def test_prune_wait_below_mark_leaves_tool_calls_untouched_and_acks():
    asst_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "cwait",
                "type": "function",
                "function": {"name": "wait", "arguments": "{}"},
            },
        ],
    }
    client = _FakeClient([asst_msg])
    client._sent_watermark = 1  # asst_msg already dispatched
    dispatcher = _FakeMsgDispatcher(client)
    assistant_meta = {}

    await prune_wait_tool_call(
        asst_msg,
        "cwait",
        client=client,
        assistant_meta=assistant_meta,
        msg_dispatcher=dispatcher,
    )

    # The stale wait's tool_calls entry is untouched (no pop, no in-place edit) —
    # mutating it would have shifted every already-dispatched message after it.
    assert asst_msg["tool_calls"] == [
        {
            "id": "cwait",
            "type": "function",
            "function": {"name": "wait", "arguments": "{}"},
        },
    ]
    # Instead it's acknowledged via an appended reply for the same call_id.
    assert len(client.messages) == 2
    ack = client.messages[1]
    assert ack["role"] == "tool"
    assert ack["tool_call_id"] == "cwait"


@pytest.mark.asyncio
async def test_prune_wait_above_mark_still_removes_in_place():
    asst_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "cwait",
                "type": "function",
                "function": {"name": "wait", "arguments": "{}"},
            },
            {
                "id": "cother",
                "type": "function",
                "function": {"name": "other", "arguments": "{}"},
            },
        ],
    }
    client = _FakeClient([asst_msg])
    client._sent_watermark = 0  # nothing dispatched yet — free to mutate

    await prune_wait_tool_call(asst_msg, "cwait", client=client)

    assert [c["id"] for c in asst_msg["tool_calls"]] == ["cother"]
    assert client.messages == [asst_msg]


# ---------------------------------------------------------------------------
# 6. record_progress — coalesce-then-freeze, separate from tool_reply_msg
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_progress_coalesces_above_mark_then_freezes():
    client = _FakeClient()
    dispatcher = _FakeMsgDispatcher(client)
    tools_data = ToolsData(tools={}, client=client, logger=_FakeLogger())
    tools_data._visibility_guidance_injected = True  # covered separately below
    frozen_stub = {"role": "tool", "tool_call_id": "c1", "content": "stub"}
    info = _make_metadata(tool_reply_msg=frozen_stub)

    await tools_data.record_progress(info, "c1", "first update", dispatcher)
    assert len(client.messages) == 1
    msg1 = client.messages[0]
    assert msg1["role"] == "user"
    assert msg1["content"] == "[progress c1] first update"
    assert info.progress_msg is msg1

    # Still mutable (nothing dispatched) — coalesces into the same message.
    await tools_data.record_progress(info, "c1", "second update", dispatcher)
    assert len(client.messages) == 1
    assert client.messages[0] is msg1
    assert msg1["content"] == "[progress c1] second update"

    # The tool_reply_msg stub is never touched by progress delivery.
    assert frozen_stub["content"] == "stub"

    # Freeze it (simulate a dispatch), then a new notification starts a fresh
    # tail message rather than reaching back to mutate the sent one.
    client._sent_watermark = len(client.messages)
    await tools_data.record_progress(info, "c1", "third update", dispatcher)
    assert len(client.messages) == 2
    assert msg1["content"] == "[progress c1] second update"  # frozen, unchanged
    msg2 = client.messages[1]
    assert msg2 is not msg1
    assert msg2["content"] == "[progress c1] third update"
    assert info.progress_msg is msg2


# ---------------------------------------------------------------------------
# 7. record_clarification — separate tail message, never touches tool_reply_msg
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_clarification_coalesces_and_never_touches_stub():
    client = _FakeClient()
    dispatcher = _FakeMsgDispatcher(client)
    tools_data = ToolsData(tools={}, client=client, logger=_FakeLogger())
    tools_data._visibility_guidance_injected = True  # covered separately below
    frozen_stub = {"role": "tool", "tool_call_id": "c1", "content": "stub"}
    info = _make_metadata(tool_reply_msg=frozen_stub)

    await tools_data.record_clarification(info, "c1", "What color?", dispatcher)
    assert len(client.messages) == 1
    q_msg = client.messages[0]
    assert q_msg["role"] == "user"
    assert q_msg["content"].startswith("[clarification c1]")
    assert "What color?" in q_msg["content"]
    assert info.clarify_msg is q_msg
    assert frozen_stub["content"] == "stub"

    # Coalesces while mutable.
    await tools_data.record_clarification(info, "c1", "What color exactly?", dispatcher)
    assert len(client.messages) == 1
    assert "What color exactly?" in q_msg["content"]
    assert frozen_stub["content"] == "stub"


@pytest.mark.asyncio
async def test_visibility_guidance_injects_on_first_progress_message_not_just_interjection():
    """Most loops never get a real user interjection, so gating the
    guidance solely on that left every sub-agent and unattended task without
    it — the model would see [progress ...] under the
    standing "incorporate ALL user interjections" instruction with nothing
    telling it that prefix is exempt. Injection must fire on the first
    [progress]/[clarification] message too, and at most once overall.
    """
    client = _FakeClient()
    dispatcher = _FakeMsgDispatcher(client)
    tools_data = ToolsData(tools={}, client=client, logger=_FakeLogger())
    assert tools_data._visibility_guidance_injected is False

    info = _make_metadata()
    await tools_data.record_progress(info, "c1", "working…", dispatcher)

    assert len(client.messages) == 2
    guidance_msg, progress_msg = client.messages
    assert guidance_msg["role"] == "system"
    assert guidance_msg.get("_visibility_guidance") is True
    assert "[progress <call_id>]" in guidance_msg["content"]
    assert "[clarification <call_id>]" in guidance_msg["content"]
    assert 'steer(call_id=<call_id>, action="clarify"' in guidance_msg["content"]
    assert "[loop " not in guidance_msg["content"]  # dead prefix, dropped
    assert progress_msg["content"] == "[progress c1] working…"
    assert tools_data._visibility_guidance_injected is True

    # Injected at most once — a second status message (even a different
    # call_id / a clarification) does not re-inject.
    info2 = _make_metadata(call_id="c2")
    await tools_data.record_clarification(info2, "c2", "which one?", dispatcher)
    assert len(client.messages) == 3
    assert all(not m.get("_visibility_guidance") for m in client.messages[1:])


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


@pytest.mark.asyncio
async def test_multi_handle_child_result_mutable_then_frozen():
    placeholder = {"role": "tool", "tool_call_id": "parent", "content": "initial"}
    client = _FakeClient([placeholder])
    dispatcher = _FakeMsgDispatcher(client)
    tools_data = ToolsData(tools={}, client=client, logger=_FakeLogger())

    state = _MultiHandleState(
        parent_call_id="parent",
        parent_name="mytool",
        placeholder_msg=placeholder,
        template={"a": "[h0: steerable]", "b": "[h1: steerable]"},
        results={"h0": None, "h1": None},
    )

    # Still mutable: h0 completing rebuilds the shared placeholder in place.
    client._sent_watermark = 0
    state.results["h0"] = "h0 result"
    await state.record_child_result(
        "parenth0",
        "h0 result",
        tools_data=tools_data,
        msg_dispatcher=dispatcher,
    )
    assert len(client.messages) == 1
    assert "h0 result" in placeholder["content"]
    assert len(dispatcher.published) == 1

    # Now freeze the shared placeholder (simulate a dispatch) before h1 finishes.
    client._sent_watermark = len(client.messages)
    state.results["h1"] = "h1 result"
    await state.record_child_result(
        "parenth1",
        "h1 result",
        tools_data=tools_data,
        msg_dispatcher=dispatcher,
    )

    # The frozen shared placeholder is never rewritten again...
    assert "h1 result" not in placeholder["content"]
    # ...instead h1's own terminal result is delivered on its own call_id via
    # a per-child check_status pair.
    assert len(client.messages) == 3
    check_stub, check_reply = client.messages[1], client.messages[2]
    assert check_stub["tool_calls"][0]["function"]["name"] == "check_status_parenth1"
    assert check_reply["content"] == "h1 result"


# ---------------------------------------------------------------------------
# 10. process_completed_task — the placeholder-result-write site end to end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_completed_task_freezes_stub_below_mark_uses_check_status():
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
    dispatcher = _FakeMsgDispatcher(client)
    tools_data = ToolsData(tools={}, client=client, logger=_FakeLogger())
    assistant_meta = {}

    async def mytool():
        return "the final result"

    task = asyncio.create_task(mytool())
    info = _make_metadata(assistant_msg=asst_msg)
    tools_data.save_task(task, info)

    await ensure_placeholders_for_pending(
        tools_data=tools_data,
        assistant_meta=assistant_meta,
        client=client,
        msg_dispatcher=dispatcher,
    )
    stub = info.tool_reply_msg
    stub_content_at_dispatch = stub["content"]

    # Simulate the placeholder having been sent in a request before the tool
    # actually finishes — the realistic interrupt-mode shape below-watermark
    # gating must handle.
    client._sent_watermark = len(client.messages)

    await task
    tracker = _LoopToolFailureTracker(3, ToolLoopRuntimeState())
    needs_turn = await tools_data.process_completed_task(
        task=task,
        consecutive_failures=tracker,
        outer_handle_container=[None],
        assistant_meta=assistant_meta,
        msg_dispatcher=dispatcher,
    )

    assert needs_turn is True
    # The stub is never rewritten once below-mark — it stays exactly what it
    # was when it was self-describingly created.
    assert stub["content"] == stub_content_at_dispatch
    assert (
        "check_status" in stub_content_at_dispatch
        or "pending" in stub_content_at_dispatch
    )

    check_stub, check_reply = client.messages[-2], client.messages[-1]
    assert check_stub["tool_calls"][0]["function"]["name"] == "check_status_c1"
    assert "the final result" in check_reply["content"]


@pytest.mark.asyncio
async def test_process_completed_task_writes_in_place_above_mark():
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
    dispatcher = _FakeMsgDispatcher(client)
    tools_data = ToolsData(tools={}, client=client, logger=_FakeLogger())
    assistant_meta = {}

    async def mytool():
        return "quick result"

    task = asyncio.create_task(mytool())
    info = _make_metadata(assistant_msg=asst_msg)
    tools_data.save_task(task, info)

    await ensure_placeholders_for_pending(
        tools_data=tools_data,
        assistant_meta=assistant_meta,
        client=client,
        msg_dispatcher=dispatcher,
    )
    stub = info.tool_reply_msg

    # Nothing dispatched yet — free to edit in place.
    await task
    tracker = _LoopToolFailureTracker(3, ToolLoopRuntimeState())
    await tools_data.process_completed_task(
        task=task,
        consecutive_failures=tracker,
        outer_handle_container=[None],
        assistant_meta=assistant_meta,
        msg_dispatcher=dispatcher,
    )

    assert len(client.messages) == 2  # asst_msg + the one stub, edited in place
    assert client.messages[1] is stub
    assert "quick result" in stub["content"]


# ---------------------------------------------------------------------------
# 11. Wire-level: serialize consecutive requests, assert the common
#     byte-prefix covers everything except appended tail content.
#     Message-granular (each element individually
#     re-serialized), which is the correct proxy: a provider's own request
#     serialization concatenates discrete JSON message objects, so a changed
#     byte inside message i is exactly what shows up as a mismatch at index i.
# ---------------------------------------------------------------------------


def _serialize_request(msgs: list) -> list[str]:
    return [json.dumps(m, sort_keys=True, default=str) for m in msgs]


@pytest.mark.asyncio
async def test_wire_level_common_byte_prefix_across_consecutive_dispatches():
    client = _FakeClient([{"role": "user", "content": "start"}])
    dispatcher = _FakeMsgDispatcher(client)
    tools_data = ToolsData(tools={}, client=client, logger=_FakeLogger())
    tools_data._visibility_guidance_injected = True
    assistant_meta = {}

    captured: list[list[str]] = []

    async def fake_generate(**kwargs):
        captured.append(_serialize_request(client.messages))
        return "ok"

    client.generate = fake_generate

    # Turn 1: dispatch with just the user message.
    await generate_with_preprocess(client, None)

    # Between dispatches: a full async-tool cycle — schedule a tool
    # (placeholder spliced adjacently, always legal), a notification arrives
    # while it's still mutable (coalesces in place, so it never breaks *this*
    # prefix), then dispatch again — freezing the placeholder and the
    # progress message both.
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
    client.messages.append(asst_msg)
    task = asyncio.create_task(asyncio.sleep(0))
    info = _make_metadata(assistant_msg=asst_msg)
    tools_data.save_task(task, info)
    await ensure_placeholders_for_pending(
        tools_data=tools_data,
        assistant_meta=assistant_meta,
        client=client,
        msg_dispatcher=dispatcher,
    )
    await tools_data.record_progress(info, "c1", "starting…", dispatcher)

    # Turn 2: dispatch again.
    await generate_with_preprocess(client, None)
    await task

    prefix_1, prefix_2 = captured[0], captured[1]
    assert prefix_2[: len(prefix_1)] == prefix_1, (
        "turn 2's request must reproduce turn 1's byte-for-byte, with only "
        "new content appended after it"
    )

    # A notification arrives for the still-pending tool, now below the
    # watermark — must coalesce-then-freeze into a NEW tail message, never
    # rewrite the one already covered by prefix_2.
    await tools_data.record_progress(info, "c1", "still working…", dispatcher)
    # The tool finishes late — its placeholder is frozen, so the result must
    # arrive via check_status, not by rewriting anything prefix_2 covers.
    tracker = _LoopToolFailureTracker(3, ToolLoopRuntimeState())
    await tools_data.process_completed_task(
        task=task,
        consecutive_failures=tracker,
        outer_handle_container=[None],
        assistant_meta=assistant_meta,
        msg_dispatcher=dispatcher,
    )

    # Turn 3: dispatch again.
    await generate_with_preprocess(client, None)

    prefix_3 = captured[2]
    assert prefix_3[: len(prefix_2)] == prefix_2, (
        "turn 3's request must reproduce turn 2's byte-for-byte (including "
        "the frozen placeholder and progress message from turn 1/2), with "
        "only the new progress message, check_status pair, etc. appended"
    )
    # And transitively, turn 1 is still a prefix of turn 3.
    assert prefix_3[: len(prefix_1)] == prefix_1


# ---------------------------------------------------------------------------
# 12. Patient/legacy-blocking mode: a dispatch that returns *after* a tool
#     result should already have been consumed by the next turn must not
#     retroactively touch anything the prior (already-serialized) request
#     covered — the watermark/hash machinery doesn't know or care which mode
#     dispatched it, but this pins that patient mode's specific ordering
#     (finish the in-flight step, then ingest results, per loop.py's own
#     comment on why order matters there) doesn't accidentally bypass it.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_patient_mode_ordering_does_not_violate_watermark():
    client = _FakeClient([{"role": "user", "content": "start"}])
    dispatcher = _FakeMsgDispatcher(client)
    tools_data = ToolsData(tools={}, client=client, logger=_FakeLogger())
    tools_data._visibility_guidance_injected = True
    assistant_meta = {}

    async def fake_generate(**kwargs):
        return "ok"

    client.generate = fake_generate

    # Turn 1 dispatches with the tool already scheduled and placeholdered —
    # patient mode's defining trait is that the step is awaited to completion
    # even if a tool result becomes available while it's in flight.
    asst_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "slow_tool", "arguments": "{}"},
            },
        ],
    }
    client.messages.append(asst_msg)

    async def slow_tool():
        return "slow result"

    task = asyncio.create_task(slow_tool())
    info = _make_metadata(assistant_msg=asst_msg)
    tools_data.save_task(task, info)
    await ensure_placeholders_for_pending(
        tools_data=tools_data,
        assistant_meta=assistant_meta,
        client=client,
        msg_dispatcher=dispatcher,
    )

    await generate_with_preprocess(client, None)  # the "in-flight step"

    # Patient mode's own documented ordering: the step is awaited FIRST
    # (already happened above), THEN the result is ingested — never the
    # reverse, which loop.py's comment says would misattribute a synthetic
    # check_status pair's tool message as the step's own turn.
    await task
    tracker = _LoopToolFailureTracker(3, ToolLoopRuntimeState())
    await tools_data.process_completed_task(
        task=task,
        consecutive_failures=tracker,
        outer_handle_container=[None],
        assistant_meta=assistant_meta,
        msg_dispatcher=dispatcher,
    )

    # The result landed below the watermark (frozen by turn 1's dispatch) —
    # delivered via check_status, stub untouched — so a further dispatch
    # must not trip the integrity check.
    await generate_with_preprocess(client, None)  # must not raise


# ---------------------------------------------------------------------------
# 13. Interrupt-mode dispatch race: a tool completes *while* an LLM request
#     for the next turn is already in flight, so the loop cancels that
#     request (loop.py's `llm_task.cancel()` branch) and re-dispatches with
#     the fresh result. The unanswered request leaves nothing behind: the
#     result lands in its placeholder and the re-dispatch sends exactly what
#     it would have had the tool finished before any dispatch. A cached
#     replay, whose tools finish milliseconds earlier or later than the
#     recording's did, then reproduces the recorded request either way.
# ---------------------------------------------------------------------------


async def _request_after_tool_completes(*, superseded_dispatch: bool) -> list[str]:
    """The turn a finished tool earns, serialized, optionally with a dispatch
    in flight when the tool finished."""
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
    client = _FakeClient([{"role": "user", "content": "start"}, asst_msg])
    dispatcher = _FakeMsgDispatcher(client)
    tools_data = ToolsData(tools={}, client=client, logger=_FakeLogger())
    assistant_meta = {}
    finish = asyncio.Event()

    async def mytool():
        await finish.wait()
        return "the result"

    task = asyncio.create_task(mytool())
    info = _make_metadata(assistant_msg=asst_msg, notification_queue=asyncio.Queue())
    tools_data.save_task(task, info)
    await ensure_placeholders_for_pending(
        tools_data=tools_data,
        assistant_meta=assistant_meta,
        client=client,
        msg_dispatcher=dispatcher,
    )
    await tools_data.record_progress(info, "c1", "starting…", dispatcher)

    if superseded_dispatch:
        in_flight = asyncio.Event()

        async def hanging_generate(**kwargs):
            in_flight.set()
            await asyncio.Event().wait()

        client.generate = hanging_generate
        llm_task = asyncio.create_task(generate_with_preprocess(client, None))
        await in_flight.wait()

    # The tool reports progress just before it returns: the completion drains
    # that notification into the progress message the request carried.
    info.notification_queue.put_nowait({"message": "almost done"})
    finish.set()
    await task
    if superseded_dispatch:
        llm_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await llm_task

    tracker = _LoopToolFailureTracker(3, ToolLoopRuntimeState())
    await tools_data.process_completed_task(
        task=task,
        consecutive_failures=tracker,
        outer_handle_container=[None],
        assistant_meta=assistant_meta,
        msg_dispatcher=dispatcher,
    )

    captured: list[list[str]] = []

    async def fake_generate(**kwargs):
        captured.append(_serialize_request(client.messages))
        return "ok"

    client.generate = fake_generate
    await generate_with_preprocess(client, None)
    return captured[0]


@pytest.mark.asyncio
async def test_superseded_dispatch_leaves_the_next_request_unchanged():
    direct = await _request_after_tool_completes(superseded_dispatch=False)
    superseded = await _request_after_tool_completes(superseded_dispatch=True)

    assert superseded == direct
    assert not any("check_status_" in m for m in superseded)
    progress = [m for m in superseded if "[progress c1]" in m]
    assert len(progress) == 1 and "almost done" in progress[0]
    assert any("the result" in m for m in superseded)
