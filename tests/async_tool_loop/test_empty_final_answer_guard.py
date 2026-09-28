"""Loop-level tests for two finalization defects in the async tool loop.

Both are exercised with a fully scripted fake in place of
``generate_with_preprocess`` (no LLM calls, no network) driving a real
``start_async_tool_loop``.

1. A flag that guarantees "one more turn after a late-arriving tool result"
   must not survive past the dispatch that already carries that result in
   its prompt — otherwise a stale flag forces a spurious extra turn after
   the real answer, and that extra turn's empty content can silently
   replace it.
2. When a terminal turn's own content is empty, finalization must fall back
   to the last substantive assistant content already in the transcript
   instead of returning it verbatim; and when no substantive content exists
   anywhere, it must retry with a nudge before failing loudly, never
   returning a silent empty result.

In persist mode an empty turn is not finalized at all, but it still ends in
the wait state, and the response notification that marks it is sent with
empty content rather than withheld.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from unify.common.async_tool_loop import start_async_tool_loop
from unify.common.llm_client import new_llm_client
from unify.common._async_tool.messages import (
    extract_substantive_text,
    is_loop_authored_message,
    loop_user_notice,
)
from tests.helpers import _handle_project
from tests.async_helpers import _wait_for_tool_request, _wait_for_condition


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _asst_msg(calls: list[dict]) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": calls}


def _final_msg(content: str) -> dict:
    return {"role": "assistant", "content": content, "tool_calls": []}


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@_handle_project
async def test_stale_review_flag_does_not_force_extra_turn_after_answer(
    llm_config,
    monkeypatch,
) -> None:
    """A tool result ingested while an earlier, unrelated turn is in flight
    must not force a spurious extra turn once a later turn already produces
    the final answer."""
    client = new_llm_client(**llm_config)
    client.set_system_message("This turn is fully scripted by the test.")

    from unify.common._async_tool import loop as _loop

    turn_queue: asyncio.Queue = asyncio.Queue()
    dispatch_count = 0

    async def _fake_gwp(_client, _preprocess_msgs, **gen_kwargs):
        nonlocal dispatch_count
        dispatch_count += 1
        next_msg = await turn_queue.get()
        _client.messages.append(next_msg)
        return {"ok": True}

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    bg_release = asyncio.Event()

    async def bg_tool() -> str:
        await bg_release.wait()
        return "bg-done"

    bg_tool.__name__ = "bg_tool"
    bg_tool.__qualname__ = "bg_tool"

    async def quick_tool() -> str:
        return "quick-done"  # resolves on its own, no gate

    quick_tool.__name__ = "quick_tool"
    quick_tool.__qualname__ = "quick_tool"

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"bg_tool": bg_tool, "quick_tool": quick_tool},
        interrupt_llm_on_tool_completion=False,  # patient mode
        max_steps=40,
        timeout=30,
    )

    async def _dispatched_at_least(n: int) -> bool:
        return dispatch_count >= n

    # Turn 1 (the initial dispatch): call bg_tool, leaving it pending.
    await turn_queue.put(_asst_msg([_tool_call("call_bg", "bg_tool", {})]))
    await _wait_for_tool_request(client, "bg_tool")

    # With bg_tool pending and nothing new to report, the loop waits rather
    # than dispatching on its own — one explicit interject is needed to
    # force this next turn's dispatch. It runs before bg_tool's result is
    # ingested below, so it cannot be the thing that later clears the flag
    # under test; withholding this turn's content keeps the dispatch
    # genuinely in flight so bg_tool's completion races against it for real.
    await handle.interject("continue")
    await _wait_for_condition(lambda: _dispatched_at_least(2), poll=0.02, timeout=10.0)
    bg_release.set()
    await asyncio.sleep(0.3)  # let the completion be detected before proceeding

    # This turn's own content: a tool call that resolves on its own, so it
    # is itself a tool-call turn, followed by two more — each one's
    # completion (not an interject) is what wakes the loop for the next
    # dispatch, the same way capped_tool/quota_tool completions do in the
    # tools-bytes suite.
    await turn_queue.put(_asst_msg([_tool_call("call_quick1", "quick_tool", {})]))
    await turn_queue.put(_asst_msg([_tool_call("call_quick2", "quick_tool", {})]))
    await turn_queue.put(_asst_msg([_tool_call("call_quick3", "quick_tool", {})]))

    # The real final answer.
    await turn_queue.put(_final_msg("This is the complete final answer."))

    final = await asyncio.wait_for(handle.result(), timeout=30)
    assert final == "This is the complete final answer."
    assert dispatch_count == 5, (
        "expected exactly 5 dispatches (initial + 3 tool-call turns + the "
        f"answer); a stale flag would force an extra review turn, got "
        f"{dispatch_count}"
    )


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@_handle_project
async def test_late_result_review_turn_falls_back_to_substantive_answer(
    llm_config,
    monkeypatch,
) -> None:
    """A tool result that lands while the answer-producing turn is still in
    flight legitimately forces one more review turn. If that review turn
    comes back empty, finalization must fall back to the substantive answer
    already in the transcript rather than returning nothing."""
    client = new_llm_client(**llm_config)
    client.set_system_message("This turn is fully scripted by the test.")

    from unify.common._async_tool import loop as _loop

    turn_queue: asyncio.Queue = asyncio.Queue()
    dispatch_count = 0

    async def _fake_gwp(_client, _preprocess_msgs, **gen_kwargs):
        nonlocal dispatch_count
        dispatch_count += 1
        next_msg = await turn_queue.get()
        _client.messages.append(next_msg)
        return {"ok": True}

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    bg_release = asyncio.Event()

    async def bg_tool() -> str:
        await bg_release.wait()
        return "bg-done"

    bg_tool.__name__ = "bg_tool"
    bg_tool.__qualname__ = "bg_tool"

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"bg_tool": bg_tool},
        interrupt_llm_on_tool_completion=False,  # patient mode
        max_steps=40,
        timeout=30,
    )

    # With a tool pending and nothing new to report, the loop waits rather
    # than dispatching on its own — each subsequent turn below needs an
    # explicit interject to grant a new LLM step, same as every other
    # steering-driven scripted test in this suite.
    async def _next_turn(msg: dict, *, nudge: bool = True) -> None:
        await turn_queue.put(msg)
        if not nudge:
            return
        for _ in range(20):
            if turn_queue.qsize() == 0:
                return
            await asyncio.sleep(0.05)
        await handle.interject("continue")

    async def _dispatched_at_least(n: int) -> bool:
        return dispatch_count >= n

    # Turn 1: call bg_tool.
    await _next_turn(_asst_msg([_tool_call("call_bg", "bg_tool", {})]), nudge=False)
    await _wait_for_tool_request(client, "bg_tool")

    # Force the answer turn's dispatch, but withhold its content so
    # bg_tool's completion below races against it for real.
    await handle.interject("continue")
    await _wait_for_condition(lambda: _dispatched_at_least(2), poll=0.02, timeout=10.0)
    bg_release.set()
    await asyncio.sleep(0.3)

    # The answer turn's own content: the real, substantive answer.
    await _next_turn(_final_msg("Here is the complete answer."))

    # The forced review turn comes back with nothing to add.
    await _next_turn(_final_msg(""))

    final = await asyncio.wait_for(handle.result(), timeout=30)
    assert final == "Here is the complete answer."
    assert dispatch_count == 3, (
        f"expected exactly 3 dispatches (initial + answer + forced review), "
        f"got {dispatch_count}"
    )


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@_handle_project
async def test_true_empty_answer_retries_then_fails_loudly(
    llm_config,
    monkeypatch,
) -> None:
    """When no substantive assistant content exists anywhere in the
    transcript, the loop must retry with a nudge and, failing that, return a
    loud error naming the condition rather than a silent empty result."""
    client = new_llm_client(**llm_config)
    client.set_system_message("This turn is fully scripted by the test.")

    from unify.common._async_tool import loop as _loop

    turn_queue: asyncio.Queue = asyncio.Queue()
    dispatch_count = 0

    async def _fake_gwp(_client, _preprocess_msgs, **gen_kwargs):
        nonlocal dispatch_count
        dispatch_count += 1
        next_msg = await turn_queue.get()
        _client.messages.append(next_msg)
        return {"ok": True}

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={},
        interrupt_llm_with_interjections=False,  # legacy blocking mode; no racing needed
        max_steps=40,
        timeout=30,
    )

    await turn_queue.put(_final_msg(""))
    await turn_queue.put(_final_msg(""))

    final = await asyncio.wait_for(handle.result(), timeout=30)
    assert final is not None
    assert isinstance(final, str)
    assert final != ""
    assert (
        dispatch_count == 2
    ), f"expected exactly one nudge retry, got {dispatch_count} dispatches"

    nudge_count = sum(
        1
        for m in client.messages
        if m.get("role") == "user"
        and m.get("content") == "Produce your final answer as text."
    )
    assert nudge_count == 1


def test_extract_substantive_text_treats_whitespace_and_empty_blocks_as_empty() -> None:
    """Direct coverage of the text-extraction helper: whitespace-only
    content in either shape is empty, and a substantive content-block list
    yields its extracted text rather than the raw list."""
    assert extract_substantive_text(None) is None
    assert extract_substantive_text("") is None
    assert extract_substantive_text("   \n  ") is None
    assert extract_substantive_text([]) is None
    assert extract_substantive_text([{"type": "text", "text": ""}]) is None
    assert extract_substantive_text([{"type": "text", "text": "   \n  "}]) is None
    assert extract_substantive_text([{"type": "image", "url": "x"}]) is None

    assert extract_substantive_text("The answer is 42.") == "The answer is 42."
    assert (
        extract_substantive_text(
            [
                {"type": "text", "text": "Part one. "},
                {"type": "text", "text": "Part two."},
            ],
        )
        == "Part one. Part two."
    )


async def _quick_tool() -> str:
    return "quick-done"  # resolves on its own, no gate


_quick_tool.__name__ = "quick_tool"
_quick_tool.__qualname__ = "quick_tool"


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@_handle_project
async def test_whitespace_only_terminal_turn_falls_back_to_prior_answer(
    llm_config,
    monkeypatch,
) -> None:
    """A terminal turn whose content is whitespace-only must be treated as
    empty by the finalization guard, not returned verbatim."""
    client = new_llm_client(**llm_config)
    client.set_system_message("This turn is fully scripted by the test.")

    from unify.common._async_tool import loop as _loop

    turn_queue: asyncio.Queue = asyncio.Queue()

    async def _fake_gwp(_client, _preprocess_msgs, **gen_kwargs):
        next_msg = await turn_queue.get()
        _client.messages.append(next_msg)
        return {"ok": True}

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"quick_tool": _quick_tool},
        interrupt_llm_with_interjections=False,  # legacy blocking mode; no racing needed
        max_steps=40,
        timeout=30,
    )

    await turn_queue.put(
        {
            "role": "assistant",
            "content": "The answer is 42.",
            "tool_calls": [_tool_call("call_q", "quick_tool", {})],
        },
    )
    await turn_queue.put({"role": "assistant", "content": "   \n  ", "tool_calls": []})

    final = await asyncio.wait_for(handle.result(), timeout=30)
    assert final == "The answer is 42."


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@_handle_project
async def test_empty_content_block_list_falls_back_to_prior_answer(
    llm_config,
    monkeypatch,
) -> None:
    """A terminal turn whose content is a content-block list with only
    whitespace text must be treated as empty by the finalization guard."""
    client = new_llm_client(**llm_config)
    client.set_system_message("This turn is fully scripted by the test.")

    from unify.common._async_tool import loop as _loop

    turn_queue: asyncio.Queue = asyncio.Queue()

    async def _fake_gwp(_client, _preprocess_msgs, **gen_kwargs):
        next_msg = await turn_queue.get()
        _client.messages.append(next_msg)
        return {"ok": True}

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"quick_tool": _quick_tool},
        interrupt_llm_with_interjections=False,
        max_steps=40,
        timeout=30,
    )

    await turn_queue.put(
        {
            "role": "assistant",
            "content": "The answer is 42.",
            "tool_calls": [_tool_call("call_q", "quick_tool", {})],
        },
    )
    await turn_queue.put(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "   "}],
            "tool_calls": [],
        },
    )

    final = await asyncio.wait_for(handle.result(), timeout=30)
    assert final == "The answer is 42."


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@_handle_project
async def test_substantive_content_block_list_returns_extracted_text(
    llm_config,
    monkeypatch,
) -> None:
    """A terminal turn whose content is a substantive content-block list
    must return the extracted text, never the raw list."""
    client = new_llm_client(**llm_config)
    client.set_system_message("This turn is fully scripted by the test.")

    from unify.common._async_tool import loop as _loop

    turn_queue: asyncio.Queue = asyncio.Queue()

    async def _fake_gwp(_client, _preprocess_msgs, **gen_kwargs):
        next_msg = await turn_queue.get()
        _client.messages.append(next_msg)
        return {"ok": True}

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={},
        interrupt_llm_with_interjections=False,
        max_steps=40,
        timeout=30,
    )

    await turn_queue.put(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Part one. "},
                {"type": "text", "text": "Part two."},
            ],
            "tool_calls": [],
        },
    )

    final = await asyncio.wait_for(handle.result(), timeout=30)
    assert final == "Part one. Part two."
    assert isinstance(final, str)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@_handle_project
async def test_lifecycle_announcement_between_answer_and_empty_recovers(
    llm_config,
    monkeypatch,
) -> None:
    """A [steerable]/[askable] lifecycle announcement landing between the
    real answer and a later empty terminal turn must not block the
    walk-back from recovering the answer."""
    client = new_llm_client(**llm_config)
    client.set_system_message("This turn is fully scripted by the test.")

    from unify.common._async_tool import loop as _loop

    turn_queue: asyncio.Queue = asyncio.Queue()
    dispatch_count = 0

    async def _fake_gwp(_client, _preprocess_msgs, **gen_kwargs):
        nonlocal dispatch_count
        dispatch_count += 1
        next_msg = await turn_queue.get()
        _client.messages.append(next_msg)
        return {"ok": True}

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"quick_tool": _quick_tool},
        interrupt_llm_with_interjections=False,  # legacy blocking mode; no racing needed
        max_steps=40,
        timeout=30,
    )

    async def _dispatched_at_least(n: int) -> bool:
        return dispatch_count >= n

    # Turn 1: the real answer, alongside a tool call so the turn isn't
    # terminal yet and a second turn gets dispatched.
    await turn_queue.put(
        {
            "role": "assistant",
            "content": "The answer is 42.",
            "tool_calls": [_tool_call("call_q", "quick_tool", {})],
        },
    )

    # Wait for the second dispatch to start (quick_tool's completion has
    # already been ingested by then, since dispatch is gated on pending
    # tools draining), then splice in a lifecycle announcement the same
    # shape record_tool_completed_askable appends in production, before
    # handing the loop its empty terminal turn.
    await _wait_for_condition(lambda: _dispatched_at_least(2), poll=0.02, timeout=10.0)
    client.messages.append(
        loop_user_notice(
            "[askable call_q] quick_tool completed and is askable.",
            _lifecycle_msg=True,
        ),
    )
    await turn_queue.put(_final_msg(""))

    final = await asyncio.wait_for(handle.result(), timeout=30)
    assert final == "The answer is 42."


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@_handle_project
async def test_nudge_message_does_not_become_boundary_on_retry(
    llm_config,
    monkeypatch,
) -> None:
    """A nudge tail message must be recognized as loop-authored, not a
    genuine user turn boundary, so a retry's walk-back can still reach
    past it to a substantive answer."""
    client = new_llm_client(**llm_config)
    client.set_system_message("This turn is fully scripted by the test.")

    from unify.common._async_tool import loop as _loop

    turn_queue: asyncio.Queue = asyncio.Queue()
    dispatch_count = 0

    async def _fake_gwp(_client, _preprocess_msgs, **gen_kwargs):
        nonlocal dispatch_count
        dispatch_count += 1
        next_msg = await turn_queue.get()
        _client.messages.append(next_msg)
        return {"ok": True}

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"quick_tool": _quick_tool},
        interrupt_llm_with_interjections=False,
        max_steps=40,
        timeout=30,
    )

    async def _dispatched_at_least(n: int) -> bool:
        return dispatch_count >= n

    await turn_queue.put(
        {
            "role": "assistant",
            "content": "The answer is 42.",
            "tool_calls": [_tool_call("call_q", "quick_tool", {})],
        },
    )

    # Splice in a message shaped exactly like the loop's own nudge tail
    # message before handing over the empty terminal turn, to isolate
    # whether the marker alone (regardless of how it got there) is
    # correctly recognized as non-boundary.
    await _wait_for_condition(lambda: _dispatched_at_least(2), poll=0.02, timeout=10.0)
    client.messages.append(
        loop_user_notice("Produce your final answer as text.", _nudge_msg=True),
    )
    await turn_queue.put(_final_msg(""))

    final = await asyncio.wait_for(handle.result(), timeout=30)
    assert final == "The answer is 42."


def test_loop_user_notice_output_is_recognized_as_loop_authored() -> None:
    """Every message the constructor builds must satisfy the predicate it
    exists to feed — the pairing the rest of this file's loop-level probes
    depend on — while a hand-built user message without the marker (a
    stand-in for a genuine interjection) must not."""
    notice = loop_user_notice("Context window is nearly full.")
    assert notice["role"] == "user"
    assert notice["content"] == "Context window is nearly full."
    assert is_loop_authored_message(notice)

    # Extra purpose-specific markers pass through and still satisfy the
    # predicate via the constructor's own stamp.
    marked = loop_user_notice("some status", _lifecycle_msg=True)
    assert marked["_lifecycle_msg"] is True
    assert is_loop_authored_message(marked)

    genuine_interjection = {
        "role": "user",
        "content": "actually, do X instead",
        "_interjection": True,
    }
    assert not is_loop_authored_message(genuine_interjection)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@_handle_project
async def test_compression_threshold_notice_between_answer_and_empty_recovers(
    llm_config,
    monkeypatch,
) -> None:
    """A 'context window nearly full' notice landing between the real
    answer and a later empty terminal turn must not block the walk-back
    from recovering the answer."""
    client = new_llm_client(**llm_config)
    client.set_system_message("This turn is fully scripted by the test.")

    from unify.common._async_tool import loop as _loop

    turn_queue: asyncio.Queue = asyncio.Queue()
    dispatch_count = 0

    async def _fake_gwp(_client, _preprocess_msgs, **gen_kwargs):
        nonlocal dispatch_count
        dispatch_count += 1
        next_msg = await turn_queue.get()
        _client.messages.append(next_msg)
        return {"ok": True}

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"quick_tool": _quick_tool},
        interrupt_llm_with_interjections=False,  # legacy blocking mode; no racing needed
        max_steps=40,
        timeout=30,
    )

    async def _dispatched_at_least(n: int) -> bool:
        return dispatch_count >= n

    await turn_queue.put(
        {
            "role": "assistant",
            "content": "The answer is 42.",
            "tool_calls": [_tool_call("call_q", "quick_tool", {})],
        },
    )

    await _wait_for_condition(lambda: _dispatched_at_least(2), poll=0.02, timeout=10.0)
    client.messages.append(
        loop_user_notice(
            "Context window is nearly full. You must call `compress_context` now.",
        ),
    )
    await turn_queue.put(_final_msg(""))

    final = await asyncio.wait_for(handle.result(), timeout=30)
    assert final == "The answer is 42."


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@_handle_project
async def test_quota_pruning_notice_between_answer_and_empty_recovers(
    llm_config,
    monkeypatch,
) -> None:
    """A quota-pruning system notification landing between the real answer
    and a later empty terminal turn must not block the walk-back from
    recovering the answer."""
    client = new_llm_client(**llm_config)
    client.set_system_message("This turn is fully scripted by the test.")

    from unify.common._async_tool import loop as _loop

    turn_queue: asyncio.Queue = asyncio.Queue()
    dispatch_count = 0

    async def _fake_gwp(_client, _preprocess_msgs, **gen_kwargs):
        nonlocal dispatch_count
        dispatch_count += 1
        next_msg = await turn_queue.get()
        _client.messages.append(next_msg)
        return {"ok": True}

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"quick_tool": _quick_tool},
        interrupt_llm_with_interjections=False,
        max_steps=40,
        timeout=30,
    )

    async def _dispatched_at_least(n: int) -> bool:
        return dispatch_count >= n

    await turn_queue.put(
        {
            "role": "assistant",
            "content": "The answer is 42.",
            "tool_calls": [_tool_call("call_q", "quick_tool", {})],
        },
    )

    await _wait_for_condition(lambda: _dispatched_at_least(2), poll=0.02, timeout=10.0)
    client.messages.append(
        loop_user_notice(
            "System notification: The tool calls in your last response were "
            "blocked due to quota limits. Please modify your plan or conclude.",
        ),
    )
    await turn_queue.put(_final_msg(""))

    final = await asyncio.wait_for(handle.result(), timeout=30)
    assert final == "The answer is 42."


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@_handle_project
async def test_persist_empty_turn_still_surfaces_a_response(
    llm_config,
    monkeypatch,
) -> None:
    """In persist mode the response notification marks the wait state, so a
    turn that returns no content and no tool calls (what a model with nothing
    to say sends after a message that asks for nothing) surfaces an empty
    response rather than nothing. A driver waiting for the turn's response,
    such as ``unify act --jsonl``, would otherwise wait forever. The loop stays
    in its wait state and answers the next interjection as usual."""
    client = new_llm_client(**llm_config)
    client.set_system_message("This turn is fully scripted by the test.")

    from unify.common._async_tool import loop as _loop

    turn_queue: asyncio.Queue = asyncio.Queue()

    async def _fake_gwp(_client, _preprocess_msgs, **gen_kwargs):
        next_msg = await turn_queue.get()
        _client.messages.append(next_msg)
        return {"ok": True}

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={},
        persist=True,
        max_steps=40,
        timeout=30,
    )

    async def _next_response() -> dict:
        while True:
            notif = await asyncio.wait_for(handle.next_notification(), timeout=10)
            if isinstance(notif, dict) and notif.get("type") == "response":
                return notif

    await turn_queue.put(_final_msg("ready"))
    assert (await _next_response())["content"] == "ready"

    await handle.interject("Your submission was CORRECT.")
    await turn_queue.put({"role": "assistant", "content": None, "tool_calls": None})
    assert (await _next_response())["content"] == ""
    assert not handle.done()

    await handle.interject("Reply with the word done.")
    await turn_queue.put(_final_msg("done"))
    assert (await _next_response())["content"] == "done"

    await handle.stop()
    await asyncio.wait_for(handle.result(), timeout=30)
