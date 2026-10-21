"""Integration tests for context compression wiring in the async tool loop.

The tests that run a whole compression script the model (``loop_model``,
tests/async_tool_loop/conftest.py) and force the threshold with a trigger
the first ``add`` call fires. A compression has two paths, and each test
names the one it takes:

* as shipped (``FORK_SUMMARY``): the summary comes from a fork of the last
  request (a ``compression_fork`` request), and the compactor is never
  called. The fork's scripted answer resets the trigger.
* the fallback: the fork returns no text, so ``compress_and_rebuild`` runs
  the (patched) compactor, which resets the trigger and archives the raw
  transcript in ``raw_archives``.

The loops here are the actor's kind (``tools_in_code``): over the threshold
the loop compacts before its next model call, with no turn of its own, and
the session's tool list stays the same.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import unify.common._async_tool.loop as _loop_mod
import unify.common._async_tool.context_compression as _cc_mod
import unify.common.async_tool_loop as _atl_mod
from unify.common._async_tool.context_compression import (
    CompressedMessage,
    CompressedMessages,
)
from unify.common.async_tool_loop import start_async_tool_loop
from unify.common.llm_client import new_llm_client
from tests.scripted_model import reply

_SYS = (
    "You are in a test. Follow the steps exactly:\n"
    "1. Run `2 + 3` with execute_code.\n"
    "2. Run `10 + 20` with execute_code.\n"
    "3. Report both results."
)

SUMMARY = "Summary: added 2 + 3 = 5 and 10 + 20 = 30; report both."
FINAL = "5 and 30"
STOPPED = "processed stopped early, no result"

# The loop's turns up to the compaction: a sum, then a sum (its reply is the
# first over the threshold, since the first sum fired the trigger). The loops
# are the actor's kind, whose one tool is execute_code.
ADD_FIRST = reply(calls=[("execute_code", {"code": "2 + 3"})])
ADD_SECOND = reply(calls=[("execute_code", {"code": "10 + 20"})])


# ── Shared helpers ────────────────────────────────────────────────────────────


def _make_threshold_trigger():
    _fire = False

    def trigger():
        nonlocal _fire
        _fire = True

    def reset():
        nonlocal _fire
        _fire = False

    def check(n_tokens, threshold, max_input_tokens):
        return _fire

    return trigger, reset, check


async def _mock_compress(messages, endpoint, **kwargs):
    return CompressedMessages(
        messages=[
            CompressedMessage(
                content=json.dumps(
                    {
                        "role": m.get("role", "user"),
                        "content": f"[c] {str(m.get('content', ''))[:40]}",
                    },
                ),
            )
            for m in messages
        ],
    )


async def _compactor_not_called(messages, endpoint, **kwargs):
    raise AssertionError("the shipped path never calls the compactor")


def _make_add(trigger, calls=None):
    """The loops' ``execute_code``: it adds the two numbers its code names."""
    _triggered = False

    def execute_code(code: str) -> str:
        """Run code."""
        nonlocal _triggered
        a, b = (int(n) for n in code.split("+"))
        if calls is not None:
            calls.append((a, b))
        if not _triggered:
            _triggered = True
            trigger()
        return str(a + b)

    return execute_code


def _summary(then=None):
    """The fork's answer on the shipped path: run *then* (sync or async),
    then return the summary."""

    async def answer(call):
        if then is not None:
            done = then()
            if asyncio.iscoroutine(done):
                await done
        return reply(SUMMARY)

    return answer


def _text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return "" if content is None else str(content)


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_handle_state_preserved_after_compression(
    llm_config,
    monkeypatch,
    loop_model,
):
    """Events and queues on the handle must be the same objects after compression.

    Path: as shipped (the fork's summary)."""
    trigger, reset, check = _make_threshold_trigger()
    monkeypatch.setattr(_loop_mod, "context_over_threshold", check)
    monkeypatch.setattr(_cc_mod, "compress_messages", _compactor_not_called)

    model = loop_model(
        ADD_FIRST,
        ADD_SECOND,
        reply(FINAL),
        compression_fork=[_summary(reset)],
    )
    add = _make_add(trigger)
    client = new_llm_client(**llm_config)
    client.set_system_message(_SYS)

    handle = start_async_tool_loop(
        client=client,
        message="Add the numbers",
        tools={"execute_code": add},
        timeout=120,
        tools_in_code=True,
    )

    orig_queue = handle._queue
    orig_cancel = handle._cancel_event
    orig_stop = handle._stop_event

    result = await handle.result()

    assert handle._compression.count == 1
    assert handle._queue is orig_queue
    assert handle._cancel_event is orig_cancel
    assert handle._stop_event is orig_stop
    assert result == FINAL
    # The restarted loop continues from the summary.
    assert SUMMARY in "\n".join(_text(m) for m in model.calls[-1].messages)
    assert model.kinds() == ["actor", "actor", "compression_fork", "actor"]


@pytest.mark.asyncio
async def test_stop_carries_over_during_compression(
    llm_config,
    monkeypatch,
    loop_model,
):
    """A stop during the compression window ends the restarted loop before
    it asks the model anything.

    Path: as shipped; the stop comes while the fork writes its summary."""
    trigger, reset, check = _make_threshold_trigger()
    monkeypatch.setattr(_loop_mod, "context_over_threshold", check)
    monkeypatch.setattr(_cc_mod, "compress_messages", _compactor_not_called)

    handle_ref: dict = {}

    async def _stop_and_reset():
        await handle_ref["handle"].stop()
        reset()

    model = loop_model(
        ADD_FIRST,
        ADD_SECOND,
        compression_fork=[_summary(_stop_and_reset)],
    )
    add = _make_add(trigger)
    client = new_llm_client(**llm_config)
    client.set_system_message(_SYS)

    handle = start_async_tool_loop(
        client=client,
        message="Add the numbers",
        tools={"execute_code": add},
        timeout=120,
        tools_in_code=True,
    )
    handle_ref["handle"] = handle

    result = await handle.result()
    assert result == STOPPED
    assert handle._compression.count == 1
    assert handle._stop_event.is_set()
    # No turn after the fork.
    assert model.kinds() == ["actor", "actor", "compression_fork"]


@pytest.mark.asyncio
async def test_interjection_carries_over_during_compression(
    llm_config,
    monkeypatch,
    loop_model,
):
    """Interjection queued during compression is received by the new loop.

    Path: as shipped; the interjection comes while the fork writes its
    summary."""
    trigger, reset, check = _make_threshold_trigger()
    monkeypatch.setattr(_loop_mod, "context_over_threshold", check)
    monkeypatch.setattr(_cc_mod, "compress_messages", _compactor_not_called)

    handle_ref: dict = {}
    interjection_text = "URGENT: also compute 10 + 20"

    async def _interject_and_reset():
        await handle_ref["handle"]._queue.put(interjection_text)
        reset()

    model = loop_model(
        ADD_FIRST,
        ADD_SECOND,
        reply(FINAL),
        compression_fork=[_summary(_interject_and_reset)],
    )
    add = _make_add(trigger)
    client = new_llm_client(**llm_config)
    client.set_system_message(
        "You are in a test. Follow the steps exactly:\n"
        "1. Call `add` with a=2, b=3.\n"
        "2. Call `add` with a=10, b=20.\n"
        "3. Report both results.\n"
        "If you receive additional instructions, follow them.",
    )

    handle = start_async_tool_loop(
        client=client,
        message="Add the numbers",
        tools={"execute_code": add},
        timeout=120,
        tools_in_code=True,
    )
    handle_ref["handle"] = handle

    result = await handle.result()
    assert result == FINAL
    assert handle._compression.count == 1
    # The restarted loop's first request carries the summary and then the
    # interjection.
    restarted = "\n".join(_text(m) for m in model.calls[-1].messages)
    assert SUMMARY in restarted
    assert interjection_text in restarted
    assert restarted.index(SUMMARY) < restarted.index(interjection_text)
    assert all(
        interjection_text not in _text(m) for c in model.calls[:-1] for m in c.messages
    )
    assert model.kinds() == ["actor", "actor", "compression_fork", "actor"]


@pytest.mark.asyncio
async def test_compression_failure_returns_gracefully(monkeypatch):
    """When compression fails, result() returns gracefully.

    Compression first
    forks the conversation for a summary and uses ``compress_and_rebuild``
    only when the fork yields none, so both are made to fail. The model is
    scripted: the first turn fills the context, so the loop compacts.
    """
    from tests import cache_discipline_helpers as h

    async def _no_fork(self, cfg):
        return None

    async def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(_atl_mod.AsyncToolLoopHandle, "_summarise_as_fork", _no_fork)
    monkeypatch.setattr(_atl_mod, "compress_and_rebuild", _boom)

    async def execute_code(code: str) -> str:
        """Run code.

        Args:
            code: The code.
        """
        return f"ran {code}"

    replies = [
        lambda: h.completion(
            calls=[("execute_code", {"code": "1 + 2"})],
            prompt_tokens=900_000,
        ),
    ]
    with h.scripted(replies):
        handle = start_async_tool_loop(
            h.new_client(),
            "Add the numbers",
            {"execute_code": execute_code},
            log_steps=False,
            timeout=30,
            tools_in_code=True,
        )
        result = await asyncio.wait_for(handle.result(), 30)
    assert result == "processed stopped early, no result"
    assert handle._compression.count == 0


