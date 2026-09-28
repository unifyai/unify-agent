import pytest
import time
import json
import asyncio
import threading

from unify.common.async_tool_loop import (
    start_async_tool_loop,
    AsyncToolLoopHandle,
    SteerableToolHandle,
)
from unify.common.tool_spec import ToolSpec
from tests.helpers import _handle_project
from unify.common.llm_client import new_llm_client
from tests.async_helpers import (
    _wait_for_tool_request,
    _wait_for_tool_result,
    _wait_for_assistant_steer_action,
    _wait_for_tool_message_prefix,
    _wait_for_condition,
    _steer_call_action,
    make_gated_sync_tool,
    first_user_message,
    first_assistant_tool_call,
    last_plain_assistant_message,
    real_tool_messages,
)

pytestmark = pytest.mark.llm_call

# (prefix-based wait helpers moved to tests/async_tool_loop/async_helpers.py)


# ─────────────────────────────────────────────────────────────────────────────
#  Tools for the *inner* loop
# ─────────────────────────────────────────────────────────────────────────────


def inner_tool() -> str:  # noqa: D401 – simple value
    """Returns the literal string 'inner‑result'."""
    time.sleep(8)
    return "inner-result"


# ─────────────────────────────────────────────────────────────────────────────
#  Tool for the *outer* loop – spawns the nested loop and returns its handle
# ─────────────────────────────────────────────────────────────────────────────


def _make_outer_tool(llm_config: dict):
    """Factory to create outer_tool with a specific LLM config for nested clients."""

    async def outer_tool() -> AsyncToolLoopHandle:
        """Launch an **inner** async‑tool‑use loop and return its *handle*."""

        # brand‑new LLM client dedicated to the nested conversation
        inner_client = new_llm_client(**llm_config)
        inner_client.set_system_message(
            "You are running inside an automated test. "
            "ONLY do the following steps:\n"
            "1️⃣  Call `inner_tool` (no arguments).\n"
            "2️⃣  Wait for its response.\n"
            "3️⃣  Reply with exactly the single word 'done'.",
        )

        # Kick off the nested loop – **no interjectable_tools specified** on
        # purpose: the outer loop must deduce that from the returned handle.
        return start_async_tool_loop(
            client=inner_client,
            message="start",
            tools={"inner_tool": inner_tool},
            parent_chat_context=None,
            max_steps=10,
            timeout=120,
        )

    outer_tool.__name__ = "outer_tool"
    outer_tool.__qualname__ = "outer_tool"
    return outer_tool


# ─────────────────────────────────────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_nested_async_tool_loop(llm_config):
    """Full end-to-end check – no mocks, real network call to OpenAI."""

    # Outer client that drives the *first* loop
    expected_system = (
        "You are running inside an automated test. Perform the steps exactly:\n"
        "1\ufe0f\u20e3  Call `outer_tool` with no arguments.\n"
        "2\ufe0f\u20e3  Continue running this tool call, when given the option.\n"
        "3\ufe0f\u20e3  Once it is *completed*, respond with exactly 'all done'."
    )
    client = new_llm_client(**llm_config)
    client.set_system_message(expected_system)

    outer_tool = _make_outer_tool(llm_config)
    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"outer_tool": outer_tool},
        max_steps=10,
        timeout=240,
        time_awareness=False,
    )

    # Wait for the outer loop to finish.
    final_reply = await handle.result()

    # The assistant should complete (we don't assert exact text - that's eval, not symbolic)
    assert final_reply is not None, "Loop should complete with a response"

    # Use semantic/index-agnostic assertions to be robust to synthetic entries
    # that may be inserted for chronological ordering

    # 0. System message should be present (find it semantically)
    system_msgs = [m for m in client.messages if m.get("role") == "system"]
    # Filter to non-internal system messages (not _visibility_guidance, etc.)
    real_system_msgs = [
        m
        for m in system_msgs
        if not any(
            m.get(marker)
            for marker in ("_visibility_guidance", "_runtime_context", "_ctx_header")
        )
    ]
    assert real_system_msgs, "Expected a system message"
    assert real_system_msgs[0]["content"] == expected_system

    # 1. User message should be "start"
    user_msg = first_user_message(client.messages)
    assert user_msg["content"] == "start"

    # 2. Assistant should call outer_tool (use helper to find it)
    initial_call_msg, initial_tc = first_assistant_tool_call(
        client.messages,
        "outer_tool",
    )
    assert initial_tc["function"] == {
        "arguments": "{}",
        "name": "outer_tool",
    }

    # 3. Tool response for outer_tool should contain "done"
    # Use helper that filters out synthetic check_status tool messages
    outer_tool_results = real_tool_messages(client.messages, tool_name="outer_tool")
    assert outer_tool_results, "Expected a tool result for outer_tool"
    assert (
        outer_tool_results[0]["content"] == "done"
    ), "The placeholder for outer_tool should be updated with the inner loop's final result."

    # 4. Final assistant message should have content and no tool calls
    final_assistant_msg = last_plain_assistant_message(client.messages)
    assert (
        final_assistant_msg["content"] is not None
    ), "Final message should have content"
    assert (
        final_assistant_msg.get("tool_calls") is None
    ), "Final assistant message should not have tool calls"


@pytest.mark.asyncio
async def test_stop_nested_loop_calls_stop(llm_config, monkeypatch):
    """
    Launch `outer_tool`, then instruct the assistant to *stop* it via the
    dynamic helper.  The test passes only if that helper ends up calling
    `AsyncToolLoopHandle.stop()` exactly once.
    """

    # 1.  Instrument `AsyncToolLoopHandle.stop` so we can count invocations
    stop_called = {"count": 0}

    original_stop = AsyncToolLoopHandle.stop

    def patched_stop(self):
        stop_called["count"] += 1
        return original_stop(self)

    monkeypatch.setattr(
        AsyncToolLoopHandle,
        "stop",
        patched_stop,
        raising=True,
    )

    # 2.  Fire up the *outer* conversational loop
    client = new_llm_client(**llm_config)
    client.set_system_message(
        "You are running inside an automated test.\n"
        "1️⃣  Call `outer_tool` with no arguments.\n"
        '2️⃣  If the user later says **stop**, you MUST immediately call exactly once steer(call_id=<the id from the "[steerable ...]" announcement for your outer_tool call>, action="stop") to stop the running call. Do not wait for the inner tool to finish by itself.\n'
        "2️⃣b Immediately after that, call the `wait` helper to keep waiting if needed. Do not call any other helpers. You may call `wait` again if still waiting.\n"
        "3️⃣  Do not produce any other reply until the stop has taken effect.\n"
        "4️⃣  Only after you have called steer(action=\"stop\") and received the acknowledgement tool message (containing 'stopped successfully'), reply exactly the single line 'outer stopped'.",
    )

    # 2a. Gate the existing inner_tool so it cannot finish until we observe the stop helper
    finish_gate, gated_inner = make_gated_sync_tool(
        return_value="inner-result",
        timeout=60,
    )
    # Create outer_tool with the gated inner_tool captured in its closure
    outer_tool = _make_outer_tool(llm_config)
    # Patch inner_tool in the closure's globals
    outer_tool.__globals__["inner_tool"] = gated_inner

    outer_handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"outer_tool": outer_tool},
        max_steps=20,
        timeout=240,
    )

    # 3.  Interject: ask the assistant to stop the running tool call
    # Wait deterministically for `outer_tool` to be requested and the placeholder inserted
    await _wait_for_tool_request(client, "outer_tool")
    await _wait_for_tool_result(client, tool_name="outer_tool", min_results=1)
    await outer_handle.interject("stop")

    # Ensure the assistant actually invokes steer(action="stop") for the outer tool
    await _wait_for_assistant_steer_action(client, "stop")

    # Only now allow the inner tool to finish so the LLM's tool selection
    # happened while it was still running
    finish_gate.set()

    # 4.  Wait for completion & check outcomes
    final_reply = await outer_handle.result()

    # A. Loop completed (we don't assert exact text - that's eval, not symbolic)
    assert final_reply is not None, "Loop should complete with a response"

    # B. Our patched `stop()` *must* have been invoked once.
    assert (
        stop_called["count"] == 1
    ), "Nested AsyncToolLoopHandle.stop() was *not* invoked via stoplation"

    # C. Optional sanity – a tool message that confirms stoplation.
    assert any(
        m.get("role") == "tool"
        and "stop" in (m.get("name") or "")
        and "stopped successfully" in (m.get("content") or "").lower()
        for m in client.messages
    ), "No tool-message indicates the stoplation happened"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_reads_corrections",
    [False, True],
    ids=["plain_tool", "tool_with_own_interject_queue"],
)
async def test_interject_nested_handle(llm_config, tool_reads_corrections):
    """
    Verify that the outer loop can correctly interject the inner loop.

    Flow:
    1. Outer loop calls `outer_tool` which starts a nested async tool loop
    2. The nested loop starts a gated tool and waits
    3. Test sends interjection "forward: hello world" to the outer loop
    4. Outer loop uses steer(action="interject") to forward to nested loop
    5. Nested loop receives the interjection and includes it in its final response
    6. Gate is released, inner tool completes, and the result flows back up

    A tool may take its own ``_interject_queue`` for corrections to its body
    while it runs, as ``execute_code`` does. Once it has returned the handle,
    the forwarded interjection must still reach the handle.
    """

    # Gate to control when the inner tool completes
    inner_gate = asyncio.Event()

    async def gated_task() -> str:
        """Wait for the gate to be released, then return."""
        await asyncio.wait_for(inner_gate.wait(), timeout=120)
        return "task done"

    gated_task.__name__ = "gated_task"
    gated_task.__qualname__ = "gated_task"

    async def _start_nested_loop() -> AsyncToolLoopHandle:
        inner_client = new_llm_client(**llm_config)
        inner_client.set_system_message(
            "Call `gated_task` and wait for it. "
            "If you receive user messages while waiting, note them. "
            "When done, reply: 'Result: <task result>. Messages: <any user messages>'",
        )
        return start_async_tool_loop(
            client=inner_client,
            message="start",
            tools={"gated_task": gated_task},
        )

    # Outer tool: launches nested loop and returns its handle
    if tool_reads_corrections:

        async def outer_tool(
            _interject_queue: asyncio.Queue | None = None,
        ) -> AsyncToolLoopHandle:
            return await _start_nested_loop()

    else:

        async def outer_tool() -> AsyncToolLoopHandle:
            return await _start_nested_loop()

    outer_tool.__name__ = "outer_tool"
    outer_tool.__qualname__ = "outer_tool"

    # Top-level loop
    client = new_llm_client(**llm_config)
    client.set_system_message(
        "Call `outer_tool` (once only) to start a nested loop. "
        "When you receive 'forward: X', call steer(call_id=<the id from the "
        '"[steerable ...]" announcement for your outer_tool call>, '
        'action="interject", payload="X") to forward X to the nested loop. '
        "When outer_tool completes, reply with its result.",
    )

    top_handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"outer_tool": ToolSpec(fn=outer_tool, max_total_calls=1)},
        max_steps=20,
        timeout=240,
    )

    # Wait until the nested loop has started
    await _wait_for_tool_request(client, "outer_tool")
    await _wait_for_tool_result(client, tool_name="outer_tool", min_results=1)

    # Send the interjection to be forwarded
    await top_handle.interject("forward: hello world")

    # Wait for the outer LLM to call steer(action="interject") BEFORE releasing
    # the gate. This ensures the forwarded message reaches the inner loop while
    # gated_task is still running, so the inner loop can include "hello world"
    # in its response.
    await _wait_for_assistant_steer_action(client, "interject")

    # Only now release the gate, allowing gated_task to complete
    inner_gate.set()

    result = await top_handle.result()

    # Assertions
    msgs = client.messages

    # a) Outer loop called steer(action="interject") with "hello world"
    interject_call = next(
        (
            call
            for m in msgs
            if m.get("tool_calls")
            for call in m["tool_calls"]
            if _steer_call_action(call) == "interject"
        ),
        None,
    )
    assert interject_call is not None, 'steer(action="interject") was not called'

    args = json.loads(interject_call["function"]["arguments"]) or {}
    content = args.get("payload") or ""
    assert "hello world" in content.lower(), f"Wrong content forwarded: {content!r}"

    # b) The forwarded message reached the inner loop and appears in the
    #    outer_tool's completion result (check_status tool message).
    #    We check the tool result rather than the final LLM text because
    #    GPT-5.2 non-deterministically returns an empty final response
    #    instead of echoing the inner loop's output. steer's own
    #    acknowledgement quotes the payload whether or not it arrived, so it
    #    is no evidence.
    outer_tool_result = next(
        (
            m.get("content", "")
            for m in msgs
            if m.get("role") == "tool"
            and m.get("tool_call_id") != interject_call["id"]
            and "hello world" in str(m.get("content", "")).lower()
        ),
        None,
    )
    assert outer_tool_result is not None, (
        f"'hello world' not found in any tool result message. "
        f"Final LLM result: {result!r}"
    )


@pytest.mark.asyncio
async def test_clarification_nested_handle(llm_config):
    """
    Inner tool asks a question, outer loop surfaces it, assistant answers
    via steer(action="clarify"), inner loop receives the answer, outer loop
    completes.
    """
    exec_log = []

    # ── inner tool that *requires* clarification ─────────────────────────
    async def ask_colour(
        *,
        _clarification_up_q: asyncio.Queue[str],
        _clarification_down_q: asyncio.Queue[str],
    ) -> str:
        await _clarification_up_q.put("Which colour?")
        colour = await _clarification_down_q.get()
        exec_log.append(colour)
        return f"Chose {colour}"

    ask_colour.__name__ = "ask_colour"
    ask_colour.__qualname__ = "ask_colour"

    # ── outer tool launches a nested loop and *exposes the same queues* ──
    async def outer_tool() -> AsyncToolLoopHandle:
        up_q, down_q = asyncio.Queue(), asyncio.Queue()
        inner_client = new_llm_client(**llm_config)
        inner_client.set_system_message(
            "1️⃣  Call `ask_colour`.\n"
            "2️⃣  Wait for the clarification answer.\n."
            "3️⃣  Reply with **only** with 'done'.",
        )

        async def _ask_colour_wrapped() -> str:  # type: ignore[valid-type]
            return await ask_colour(
                _clarification_up_q=up_q,
                _clarification_down_q=down_q,
            )

        _ask_colour_wrapped.__name__ = "ask_colour"
        _ask_colour_wrapped.__qualname__ = "ask_colour"

        handle = start_async_tool_loop(
            client=inner_client,
            message="go",
            tools={"ask_colour": _ask_colour_wrapped},
            max_steps=10,
            timeout=60,
        )

        # Expose the same queues on the returned *handle* so the **outer** loop
        # can surface the clarification request and later push the answer down.
        handle.clarification_up_q = up_q
        handle.clarification_down_q = down_q

        return handle

    outer_tool.__name__ = "outer_tool"
    outer_tool.__qualname__ = "outer_tool"

    # ── top-level loop – the assistant must answer the clar request ——––
    client = new_llm_client(**llm_config)
    client.set_system_message(
        "Call `outer_tool` (once only). When it asks a question, answer with 'blue' via "
        'steer(call_id=<the id from the "[clarification ...]" tail message>, action="clarify", payload="blue").\n'
        "If waiting is still needed, call the `wait` helper; do not reply yet.\n"
        "Once outer_tool completes, say 'all done'.",
    )

    top_handle = start_async_tool_loop(
        client,
        message="start",
        tools={"outer_tool": ToolSpec(fn=outer_tool, max_total_calls=1)},
        max_steps=20,
        timeout=240,
    )

    final_reply = await top_handle.result()

    # Assertions ---------------------------------------------------------
    assert exec_log == ["blue"], "Inner loop must receive 'blue' from outer helper."


@pytest.mark.asyncio
async def test_notification_nested_handle(llm_config):
    """
    Inner tool emits notifications via ``notification_up_q``; the outer loop must
    surface these as notification events while continuing to completion.

    We assert that a notification event is observed via ``handle.next_notification()`` and
    that the conversation completes with the instructed final reply.
    """

    # ── inner tool that emits progress updates ───────────────────────────
    async def inner_progress(
        *,
        _notification_up_q: asyncio.Queue | None = None,
    ) -> str:
        if _notification_up_q is None:
            raise RuntimeError("notification queue missing")
        await _notification_up_q.put({"message": "Inner loop: preparing widget"})
        await asyncio.sleep(0)
        await _notification_up_q.put({"message": "Inner loop: halfway"})
        return "✅ inner finished"

    inner_progress.__name__ = "inner_progress"
    inner_progress.__qualname__ = "inner_progress"

    # ── outer tool launches a nested loop and bridges progress via parent's queue ──
    async def outer_tool(
        *,
        _notification_up_q: asyncio.Queue | None = None,
    ) -> AsyncToolLoopHandle:
        inner_client = new_llm_client(**llm_config)
        inner_client.set_system_message(
            "1️⃣  Call `inner_progress`.\n"
            "2️⃣  Surface any internal progress updates as they occur.\n"
            "3️⃣  Reply with exactly 'done'.",
        )

        async def inner_bridge() -> str:
            return await inner_progress(_notification_up_q=_notification_up_q)

        return start_async_tool_loop(
            client=inner_client,
            message="start",
            tools={"inner_progress": inner_bridge},
        )

    outer_tool.__name__ = "outer_tool"
    outer_tool.__qualname__ = "outer_tool"

    # ── top-level loop – must surface progress then finish ─────────────────
    client = new_llm_client(**llm_config)
    client.set_system_message(
        "You are running inside an automated test. Perform the steps exactly:\n"
        "1️⃣  Call `outer_tool` with no arguments.\n"
        "2️⃣  If any internal work makes progress, you may acknowledge it briefly but continue to completion.\n"
        "3️⃣  Once it is completed, respond with exactly 'outer done'.",
    )

    handle = start_async_tool_loop(
        client,
        message="start",
        tools={"outer_tool": outer_tool},
        max_steps=10,
        timeout=240,
    )

    # Receive a bubbled notification event from the INNER loop via the outer tool
    event = await asyncio.wait_for(handle.next_notification(), timeout=60)
    assert event["type"] == "notification"
    assert event["tool_name"] == "outer_tool"
    if isinstance(event.get("message"), str):
        assert any(k in event["message"].lower() for k in ["prepar", "halfway", "inner loop"])  # type: ignore[arg-type]

    # Finish
    final = await asyncio.wait_for(handle.result(), timeout=120)
    assert final is not None, "Loop should complete with a response"


@pytest.mark.asyncio
async def test_pause_nested_loop_calls_pause(llm_config):
    """
    Launch a nested loop, tell the assistant to *pause* it via the helper,
    and verify that `AsyncToolLoopHandle.pause()` is invoked exactly once.
    """
    pause_called = {"count": 0}

    async def dummy_long_job() -> AsyncToolLoopHandle:
        """
        Return a handle whose underlying coroutine will not complete until the
        pause helper has been invoked (deterministic gating).
        """
        pause_called_gate = asyncio.Event()

        async def _run():
            # Block completion until pause is called
            await pause_called_gate.wait()
            # Small tail to mimic finishing after pause
            await asyncio.sleep(0.1)
            return "done-after-pause"

        handle = AsyncToolLoopHandle(
            task=asyncio.create_task(_run()),
            interject_queue=asyncio.Queue(),
            cancel_event=asyncio.Event(),
            stop_event=asyncio.Event(),
        )

        # expose `.pause` and `.resume`
        async def _pause(self):  # noqa: D401
            pause_called["count"] += 1
            pause_called_gate.set()

        async def _resume(self):  # noqa: D401
            pass  # no-op for this test

        setattr(handle, "pause", _pause.__get__(handle, AsyncToolLoopHandle))
        setattr(handle, "resume", _resume.__get__(handle, AsyncToolLoopHandle))
        return handle

    dummy_long_job.__name__ = "dummy_long_job"
    dummy_long_job.__qualname__ = "dummy_long_job"

    # outer conversation --------------------------------------------------
    client = new_llm_client(**llm_config)
    client.set_system_message(
        "1️⃣  Call `dummy_long_job`.\n"
        "2️⃣  When the *user* says **pause**, you MUST immediately call exactly once "
        'steer(call_id=<the id from the "[steerable ...]" announcement for your '
        'dummy_long_job call>, action="pause"). Do NOT call any other helpers before this.\n'
        "2️⃣b If waiting is still needed, call the `wait` helper.\n"
        "3️⃣  Keep waiting for the job to finish and do not produce any other reply; then reply with 'paused done'.",
    )

    top = start_async_tool_loop(
        client=client,
        message="start",
        tools={"dummy_long_job": dummy_long_job},
        max_steps=20,
        timeout=240,
    )

    # Wait deterministically until the tool has been scheduled so the pause helper exists
    await _wait_for_tool_request(client, "dummy_long_job")
    await top.interject("pause")
    # Ensure the assistant actually invoked steer(action="pause") and we saw its ack
    await _wait_for_assistant_steer_action(client, "pause")
    await _wait_for_tool_message_prefix(client, "steer:pause ")

    final = await top.result()

    # assertions ----------------------------------------------------------
    assert final is not None, "Loop should complete with a response"
    assert pause_called["count"] == 1, "handle.pause() should be called exactly once"


@pytest.mark.asyncio
async def test_resume_nested_loop_calls_resume(llm_config):
    """
    Pause *and then* resume a running nested loop; ensure both helpers
    reach the corresponding `AsyncToolLoopHandle` methods once each.
    """
    counts = {"pause": 0, "resume": 0}

    async def dummy_job() -> AsyncToolLoopHandle:
        """Return a handle whose underlying coroutine can be paused / resumed."""

        # Deterministic gates so the job cannot finish until pause+resume helpers were called
        pause_called_gate = asyncio.Event()
        resume_called_gate = asyncio.Event()

        # ── internal pausable sleeper ─────────────────────────────────────────
        async def _run(timer: float, gate: asyncio.Event):
            remaining = timer
            step = 0.1  # seconds per loop-tick
            while remaining > 0:
                await gate.wait()  # block if paused
                await asyncio.sleep(step)
                remaining -= step
            # Block completion until both dynamic helpers were invoked
            await pause_called_gate.wait()
            await resume_called_gate.wait()

        gate = asyncio.Event()
        gate.set()  # start in *running* state
        task = asyncio.create_task(_run(8, gate))

        handle = AsyncToolLoopHandle(
            task=task,
            interject_queue=asyncio.Queue(),
            cancel_event=asyncio.Event(),
            stop_event=asyncio.Event(),
        )

        # Drive both the runner and dynamic-tool exposure from the same event
        handle._pause_event = gate  # let default .pause/.resume toggle the same gate

        # Count calls but preserve default semantics that flip _pause_event
        orig_pause = handle.pause
        orig_resume = handle.resume

        def _pause(self):
            counts["pause"] += 1
            pause_called_gate.set()
            return orig_pause()

        def _resume(self):
            counts["resume"] += 1
            resume_called_gate.set()
            return orig_resume()

        setattr(handle, "pause", _pause.__get__(handle, AsyncToolLoopHandle))
        setattr(handle, "resume", _resume.__get__(handle, AsyncToolLoopHandle))
        return handle

    dummy_job.__name__ = "dummy_job"
    dummy_job.__qualname__ = "dummy_job"

    client = new_llm_client(**llm_config)
    client.set_system_message(
        "You are running inside an automated test.\n"
        "1️⃣ Call `dummy_job`.\n"
        "2️⃣ When the user says 'hold on', immediately call exactly once steer(call_id=<the "
        'id from the "[steerable ...]" announcement for your dummy_job call>, action="pause"). '
        "Do NOT produce any status text or explanations.\n"
        "3️⃣ When the user then says 'resume', immediately call exactly once steer(call_id=<the "
        'same id>, action="resume"). Do NOT produce any status text or explanations.\n'
        "4️⃣ After the job completes, reply EXACTLY with: all done\n"
        "   - No other words, lines, punctuation, or explanations.\n"
        "   - Do not add any additional content before or after all done.\n",
    )

    h = start_async_tool_loop(
        client=client,
        message="start",
        tools={"dummy_job": ToolSpec(fn=dummy_job, max_total_calls=1)},
        max_steps=30,
        timeout=300,
    )

    # Ensure the tool has been scheduled so helpers exist
    await _wait_for_tool_request(client, "dummy_job")

    # Pause deterministically
    await h.interject("hold on")
    await _wait_for_assistant_steer_action(client, "pause")
    await _wait_for_tool_message_prefix(client, "steer:pause ")

    # Resume deterministically
    await h.interject("resume")
    await _wait_for_assistant_steer_action(client, "resume")
    await _wait_for_tool_message_prefix(client, "steer:resume ")

    final = await h.result()

    assert final is not None, "Loop should complete with a response"
    assert counts == {
        "pause": 1,
        "resume": 1,
    }, "pause/resume should each be called once"


@pytest.mark.asyncio
async def test_handle_pause_and_resume_freeze_and_unfreeze_loop(
    llm_config,
    monkeypatch,
):
    """
    • Call pause very early.
    • Wait three seconds while paused (tool finishes in the meantime).
    • Resume and ensure the outer loop *now* completes.
    • Verify pause/resume got invoked once each and that total duration
      exceeds the pause interval.
    """
    counts = {"pause": 0, "resume": 0}

    # ── 1.  Count invocations of the public API  ─────────────────────────
    original_pause = AsyncToolLoopHandle.pause
    original_resume = AsyncToolLoopHandle.resume

    def patched_pause(self):
        # Count only pauses invoked on the root outer handle; nested handles are propagated and should not increment here
        if getattr(self, "_is_root_handle", False):
            counts["pause"] += 1
        return original_pause(self)

    def patched_resume(self):
        # Count only resumes invoked on the root outer handle; nested handles are propagated and should not increment here
        if getattr(self, "_is_root_handle", False):
            counts["resume"] += 1
        return original_resume(self)

    monkeypatch.setattr(AsyncToolLoopHandle, "pause", patched_pause, raising=True)
    monkeypatch.setattr(AsyncToolLoopHandle, "resume", patched_resume, raising=True)

    # ── 2.  A very short tool (1 s) – proves that waiting is *because* of pause
    async def long_tool() -> AsyncToolLoopHandle:
        async def _run():
            await asyncio.sleep(1)  # completes quickly
            return "done-inside"

        return AsyncToolLoopHandle(
            task=asyncio.create_task(_run()),
            interject_queue=asyncio.Queue(),
            cancel_event=asyncio.Event(),
            stop_event=asyncio.Event(),
        )

    long_tool.__name__ = "long_tool"
    long_tool.__qualname__ = "long_tool"

    # ── 3.  Kick off outer loop ───────────────────────────────────────────
    client = new_llm_client(**llm_config)
    client.set_system_message(
        "1️⃣ Call `long_tool`.\n"
        "2️⃣ Wait for completion (use the `wait` helper if exposed) and do not produce any other reply.\n"
        "3️⃣ Reply with exactly **finished**.",
    )

    outer_handle = start_async_tool_loop(
        client,
        message="start",
        tools={"long_tool": long_tool},
        max_steps=25,
        timeout=300,
    )

    # ── 4.  Pause soon after launch; wait deterministically for the tool
    #        result placeholder to be appended while paused, then resume ───
    # Wait deterministically until the assistant has scheduled the tool
    await _wait_for_tool_request(client, "long_tool")
    await outer_handle.pause()

    # Ensure the tool result message for `long_tool` is appended while paused
    await _wait_for_tool_message_prefix(client, "long_tool")

    # Resume and finish
    await outer_handle.resume()
    final_reply = await outer_handle.result()

    # ── 5.  Assertions ───────────────────────────────────────────────────
    assert final_reply is not None, "Loop should complete with a response"

    # pause/resume each called exactly once
    assert counts == {"pause": 1, "resume": 1}

    # (Removed wall-clock sleep-based assertion; we rely on deterministic
    #  event waits above to validate paused behaviour.)


@pytest.mark.asyncio
async def test_handle_result_blocks_until_resume(llm_config):
    """
    `.result()` hangs while the loop is paused and unblocks immediately once
    `.resume()` is called.
    """

    async def noop_tool() -> str:
        await asyncio.sleep(0.2)
        return "ok"

    noop_tool.__name__ = "noop_tool"
    noop_tool.__qualname__ = "noop_tool"

    client = new_llm_client(**llm_config)
    client.set_system_message(
        "Call `noop_tool` then answer **only** with 'done'. Do not answer while the loop is paused or while tools are running; only answer after completion.",
    )

    h = start_async_tool_loop(
        client,
        message="go",
        tools={"noop_tool": noop_tool},
        timeout=120,
    )

    # pause almost immediately
    await h.pause()

    with pytest.raises(asyncio.TimeoutError):
        # Shield protects the inner task from the stoplation that
        # `wait_for` sends when the 1-second timeout expires.
        await asyncio.wait_for(asyncio.shield(h.result()), timeout=1)

    # resume – now it should finish quickly
    await h.resume()
    final = await asyncio.wait_for(h.result(), timeout=60)

    assert final is not None, "Loop should complete with a response"


@pytest.mark.asyncio
async def test_dynamic_handle_public_method(llm_config):
    """
    The inner tool returns a handle exposing a **public `.ask()` method**.
    The outer loop must call steer(action="ask") on it exactly once when
    the user asks "progress?", and finally reply with 'all done' after the
    long-running task completes.
    """

    progress_calls = {"count": 0}

    # ── tool that returns a handle with `.ask` ──────────────────────────
    ask_called_gate = asyncio.Event()

    async def long_compute() -> AsyncToolLoopHandle:
        """
        • Runs a 3-second dummy job in the background.
        • Provides `.ask()` so external callers can query the elapsed time.
        """

        async def _job():
            # Ensure the dynamic `ask_…` helper is invoked before the job can finish
            await ask_called_gate.wait()
            await asyncio.sleep(8)
            return "compute-done"

        handle = AsyncToolLoopHandle(
            task=asyncio.create_task(_job()),
            interject_queue=asyncio.Queue(),
            cancel_event=asyncio.Event(),
            stop_event=asyncio.Event(),
        )

        # public helper – gets exposed automatically
        async def _ask(self):
            progress_calls["count"] += 1
            # Signal that `ask_…` has been called so the job may proceed
            ask_called_gate.set()
            return f"1s elapsed"

        # Bind and expose
        setattr(handle, "ask", _ask.__get__(handle, AsyncToolLoopHandle))
        return handle

    long_compute.__name__ = "long_compute"
    long_compute.__qualname__ = "long_compute"

    # ── outer conversation that uses `long_compute` ────────────────────
    client = new_llm_client(**llm_config)
    client.set_system_message(
        "1️⃣  Call `long_compute`.\n"
        "2️⃣  When the *user* says 'progress?', you MUST immediately call exactly once "
        'steer(call_id=<the id from the "[steerable ...]" announcement for your '
        'long_compute call>, action="ask", payload="progress?"). Do not call any other '
        "helpers before this.\n"
        '2️⃣b After calling steer(action="ask"), if waiting is still needed, call the `wait` '
        "helper to keep waiting. You may call `wait` repeatedly while still waiting. Do not "
        "call any other helpers (no status checks). Do not reply to the user yet.\n"
        "3️⃣  Only once the computation finishes, answer **only** with 'all done'.",
    )

    top = start_async_tool_loop(
        client,
        message="start",
        tools={"long_compute": long_compute},
        max_steps=25,
        timeout=300,
    )

    # Wait deterministically until the assistant has launched the tool and a placeholder exists
    # so that steer(action="ask") is a valid, live target before we interject.
    await _wait_for_tool_request(client, "long_compute")
    await _wait_for_tool_result(client, tool_name="long_compute", min_results=1)

    # Now interject to trigger steer(action="ask")
    await top.interject("progress?")

    # Ensure the assistant actually invokes steer(action="ask")
    await _wait_for_assistant_steer_action(client, "ask")

    final_reply = await top.result()

    # ── Assertions ─────────────────────────────────────────────────────
    assert final_reply is not None, "Loop should complete with a response"
    assert progress_calls["count"] == 1, ".ask should be invoked exactly once"

    # Structural: find the steer(action="ask") tool_call id and its acknowledgement
    ask_call_id = next(
        (
            tc["id"]
            for m in client.messages
            if m.get("role") == "assistant"
            for tc in (m.get("tool_calls") or [])
            if _steer_call_action(tc) == "ask"
        ),
        None,
    )
    assert ask_call_id is not None, 'Assistant did not call steer(action="ask")'
    ask_ack = next(
        (
            m
            for m in client.messages
            if m.get("role") == "tool" and m.get("tool_call_id") == ask_call_id
        ),
        None,
    )
    assert ask_ack is not None, 'No acknowledgement found for steer(action="ask") call'


@pytest.mark.asyncio
async def test_outer_handle_stop_propagates_to_inner_loop_stop(llm_config):
    """
    Stopping the OUTER handle should propagate a stop down to any nested
    async tool loop handles that were returned by tools and are still running.
    """

    # Holder for the inner handle so we can instrument its stop()
    holder: dict[str, AsyncToolLoopHandle | None] = {"handle": None}
    stop_calls = {"count": 0}

    async def inner_long_job() -> str:
        # Sleep long enough that the outer stop can arrive while it's running
        await asyncio.sleep(10)
        return "inner-finished"

    inner_long_job.__name__ = "inner_long_job"
    inner_long_job.__qualname__ = "inner_long_job"

    async def outer_tool() -> AsyncToolLoopHandle:
        inner_client = new_llm_client(**llm_config)
        inner_client.set_system_message(
            "1️⃣ Call `inner_long_job`. 2️⃣ Wait for it to finish. 3️⃣ Reply 'done'.",
        )
        h = start_async_tool_loop(
            client=inner_client,
            message="start",
            tools={"inner_long_job": inner_long_job},
            max_steps=20,
            timeout=240,
        )

        # Wrap stop on the inner handle to count calls and allow graceful finish
        orig_stop = h.stop

        def _wrapped_stop(reason: str | None = None):  # type: ignore[no-redef]
            stop_calls["count"] += 1
            return orig_stop(reason)

        setattr(h, "stop", _wrapped_stop)
        holder["handle"] = h
        return h

    outer_tool.__name__ = "outer_tool"
    outer_tool.__qualname__ = "outer_tool"

    client = new_llm_client(**llm_config)
    client.set_system_message(
        "Call `outer_tool` with no arguments, then wait until it completes.",
    )

    outer = start_async_tool_loop(
        client=client,
        message="start",
        tools={"outer_tool": outer_tool},
        max_steps=20,
        timeout=240,
    )

    # Wait deterministically until the assistant has requested the outer tool
    await _wait_for_tool_request(client, "outer_tool")
    # Also wait until the tool placeholder for outer_tool has been inserted,
    # which happens when the inner handle has been created and adopted.
    await _wait_for_tool_result(client, tool_name="outer_tool", min_results=1)

    # Now stop the OUTER loop directly – should propagate to inner
    await outer.stop("test-stop")

    # The outer handle now returns a standardized notice instead of raising
    final = await outer.result()
    assert final == "processed stopped early, no result"

    # Assert that the inner handle's stop() was invoked at least once
    # (It might be called twice: once via the mirror message if processed, and once via the safety-net in cancel_pending_tasks)
    assert stop_calls["count"] >= 1, "inner handle stop() was not propagated"


@pytest.mark.asyncio
async def test_outer_handle_pause_propagates_to_inner_loop_pause(llm_config):
    """
    Pausing the OUTER handle should propagate a pause down to any nested
    async tool loop handles that were returned by tools and are still running.
    """

    pause_calls = {"count": 0}

    async def inner_long_job() -> str:
        await asyncio.sleep(6)
        return "inner-finished"

    inner_long_job.__name__ = "inner_long_job"
    inner_long_job.__qualname__ = "inner_long_job"

    async def outer_tool() -> AsyncToolLoopHandle:
        inner_client = new_llm_client(**llm_config)
        inner_client.set_system_message(
            "1️⃣ Call `inner_long_job`. 2️⃣ Wait for it to finish. 3️⃣ Reply 'done'.",
        )
        h = start_async_tool_loop(
            client=inner_client,
            message="start",
            tools={"inner_long_job": inner_long_job},
            max_steps=20,
            timeout=240,
        )

        # Wrap pause on the inner handle to count calls while preserving behaviour
        orig_pause = h.pause

        def _wrapped_pause():  # type: ignore[no-redef]
            pause_calls["count"] += 1
            return orig_pause()

        setattr(h, "pause", _wrapped_pause)
        return h

    outer_tool.__name__ = "outer_tool"
    outer_tool.__qualname__ = "outer_tool"

    client = new_llm_client(**llm_config)
    client.set_system_message(
        "Call `outer_tool` with no arguments, then wait until it completes.",
    )

    outer = start_async_tool_loop(
        client=client,
        message="start",
        tools={"outer_tool": outer_tool},
        max_steps=20,
        timeout=240,
    )

    await _wait_for_tool_request(client, "outer_tool")
    await _wait_for_tool_result(client, tool_name="outer_tool", min_results=1)

    # Pause outer; should propagate to inner handle if available – wait until observed
    await outer.pause()

    async def _paused_once():
        return pause_calls["count"] >= 1

    await _wait_for_condition(_paused_once, poll=0.05, timeout=60.0)
    await outer.resume()  # unfreeze outer so the test completes

    await outer.result()

    assert pause_calls["count"] == 1, "inner handle pause() was not propagated"


@pytest.mark.asyncio
async def test_outer_handle_resume_propagates_to_inner_loop_resume(llm_config):
    """
    Resuming the OUTER handle should propagate a resume down to any nested
    async tool loop handles that were returned by tools and are still running.
    """

    counts = {"pause": 0, "resume": 0}

    async def inner_long_job() -> str:
        # Long enough to pause then resume while running
        await asyncio.sleep(6)
        return "inner-finished"

    inner_long_job.__name__ = "inner_long_job"
    inner_long_job.__qualname__ = "inner_long_job"

    async def outer_tool() -> AsyncToolLoopHandle:
        inner_client = new_llm_client(**llm_config)
        inner_client.set_system_message(
            "1️⃣ Call `inner_long_job`. 2️⃣ Wait for it to finish. 3️⃣ Reply 'done'.",
        )
        h = start_async_tool_loop(
            client=inner_client,
            message="start",
            tools={"inner_long_job": inner_long_job},
            max_steps=20,
            timeout=240,
        )

        # Count pause/resume on the inner handle while preserving behaviour
        orig_pause = h.pause
        orig_resume = h.resume

        def _wrapped_pause():  # type: ignore[no-redef]
            counts["pause"] += 1
            return orig_pause()

        def _wrapped_resume():  # type: ignore[no-redef]
            counts["resume"] += 1
            return orig_resume()

        setattr(h, "pause", _wrapped_pause)
        setattr(h, "resume", _wrapped_resume)
        return h

    outer_tool.__name__ = "outer_tool"
    outer_tool.__qualname__ = "outer_tool"

    client = new_llm_client(**llm_config)
    client.set_system_message(
        "Call `outer_tool` with no arguments, then wait until it completes.",
    )

    outer = start_async_tool_loop(
        client=client,
        message="start",
        tools={"outer_tool": outer_tool},
        max_steps=20,
        timeout=240,
    )

    await _wait_for_tool_request(client, "outer_tool")
    await _wait_for_tool_result(client, tool_name="outer_tool", min_results=1)

    # Pause then resume outer; both should propagate – wait for each transition deterministically
    await outer.pause()

    async def _saw_pause():
        return counts["pause"] >= 1

    await _wait_for_condition(_saw_pause, poll=0.05, timeout=60.0)
    await outer.resume()

    async def _saw_resume():
        return counts["resume"] >= 1

    await _wait_for_condition(_saw_resume, poll=0.05, timeout=60.0)

    await outer.result()

    assert counts == {
        "pause": 1,
        "resume": 1,
    }, "pause/resume should each be propagated once"


@pytest.mark.asyncio
async def test_outer_handle_ask_propagates_to_inner_ask(llm_config):
    """
    ``outer_handle.ask()`` should propagate to the in-flight inner
    handle's ``ask()``, not just inspect the outer transcript.

    The generator calls ``get_seed`` then feeds the seed into
    ``generate_token``, which blocks on a gate. The outer transcript
    only shows "generator → (pending)" — the seed value is only visible
    inside the generator's transcript, so answering "what seed is being
    used?" requires delegating to the inner handle's ``ask()``.
    """

    ask_calls = {"count": 0}
    inner_gate = asyncio.Event()

    def get_seed() -> str:
        """Return the random seed used for token generation."""
        return "seed-7742"

    async def generate_token(seed: str) -> str:
        """Generate a token from the given seed. Blocks until approved.

        Parameters
        ----------
        seed : str
            The seed string returned by get_seed.
        """
        await asyncio.wait_for(inner_gate.wait(), timeout=120)
        return f"token-{seed}-xyz"

    generate_token.__name__ = "generate_token"
    generate_token.__qualname__ = "generate_token"

    async def generator() -> AsyncToolLoopHandle:
        inner_client = new_llm_client(**llm_config)
        inner_client.set_system_message(
            "You are a token generator. Perform these steps in order:\n"
            "1. Call `get_seed` to obtain the seed.\n"
            "2. Call `generate_token` passing the seed you received.\n"
            "3. Reply with **only** the token returned by `generate_token` — nothing else, no explanation.",
        )
        h = start_async_tool_loop(
            client=inner_client,
            message="start",
            tools={
                "get_seed": get_seed,
                "generate_token": generate_token,
            },
            max_steps=20,
            timeout=120,
        )

        orig_ask = h.ask

        async def _wrapped_ask(question, **kwargs):
            ask_calls["count"] += 1
            return await orig_ask(question, **kwargs)

        setattr(h, "ask", _wrapped_ask)
        return h

    generator.__name__ = "generator"
    generator.__qualname__ = "generator"

    client = new_llm_client(**llm_config)
    client.set_system_message(
        "Call `generator` with no arguments, then wait until it completes.",
    )

    outer_handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"generator": generator},
        max_steps=20,
        timeout=240,
    )

    await _wait_for_tool_request(client, "generator")
    await _wait_for_tool_result(client, tool_name="generator", min_results=1)

    # Wait until the generator's handle has been adopted and ask_* dynamic tools are available.
    async def _ask_tools_ready():
        try:
            return bool(outer_handle._task.get_ask_tools())
        except Exception:
            return False

    await _wait_for_condition(_ask_tools_ready, poll=0.05, timeout=10.0)

    # Call ask() on the OUTER handle — should propagate to inner handle's ask().
    # The outer transcript only shows "generator → (pending)"; the seed value
    # is only visible inside the generator's transcript.
    ask_handle = await outer_handle.ask(
        "What seed is the generator using?",
    )
    ask_result = await ask_handle.result()
    assert ask_result is not None, "ask() should return a result"

    inner_gate.set()
    await outer_handle.result()

    assert (
        ask_calls["count"] >= 1
    ), "inner handle ask() was not called — outer_handle.ask() did not propagate"


@pytest.mark.asyncio
async def test_outer_handle_ask_propagates_to_completed_inner_ask(llm_config):
    """
    After the solver completes, ``outer_handle.ask()`` should still
    propagate to the completed inner handle's ``ask()``.

    The solver performs opaque multi-step reasoning (fetch_data →
    compute_answer) invisible from the outer transcript — only the
    inner handle's transcript can answer "how did you get to 42?".
    """

    ask_calls = {"count": 0}

    def fetch_data() -> str:
        """Retrieve the raw dataset needed for the computation."""
        return "raw data: [10, 15, 17]"

    def compute_answer(data: str) -> str:
        """Compute the final numeric answer from the fetched data.

        Parameters
        ----------
        data : str
            The raw data string returned by fetch_data.
        """
        return "42"

    async def solver() -> AsyncToolLoopHandle:
        inner_client = new_llm_client(**llm_config)
        inner_client.set_system_message(
            "You are a solver. Perform these steps in order:\n"
            "1. Call `fetch_data` to get the raw dataset.\n"
            "2. Call `compute_answer` passing the data you received.\n"
            "3. Reply with **only** the numeric result returned by `compute_answer` — nothing else, no explanation.",
        )
        h = start_async_tool_loop(
            client=inner_client,
            message="start",
            tools={
                "fetch_data": fetch_data,
                "compute_answer": compute_answer,
            },
            max_steps=10,
            timeout=120,
        )

        orig_ask = h.ask

        async def _wrapped_ask(question, **kwargs):
            ask_calls["count"] += 1
            return await orig_ask(question, **kwargs)

        setattr(h, "ask", _wrapped_ask)
        return h

    solver.__name__ = "solver"
    solver.__qualname__ = "solver"

    client = new_llm_client(**llm_config)
    client.set_system_message(
        "Call `solver` with no arguments, then wait until it completes.\n"
        "Once it completes, reply with exactly 'all done'.",
    )

    outer_handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"solver": solver},
        max_steps=20,
        timeout=240,
    )

    final_reply = await outer_handle.result()
    assert final_reply is not None, "Outer loop should complete with a response"

    # Now ask *after* the loop has completed.  The outer transcript only
    # contains "solver → 42" — no trace of fetch_data / compute_answer.
    # Answering this meaningfully requires delegating to the inner handle.
    ask_handle = await outer_handle.ask(
        "What steps did the solver take to reach 42?",
    )
    ask_result = await ask_handle.result()
    assert ask_result is not None, "ask() should return a result"

    assert (
        ask_calls["count"] >= 1
    ), "inner handle ask() was not called — outer_handle.ask() did not propagate"


@pytest.mark.asyncio
@_handle_project
async def test_outer_stop_calls_inner_stop_on_cancel(llm_config):
    """
    Regression test for a hanging issue where outer_handle.stop()
    cancelled the wrapper task but failed to call inner_handle.stop().

    If inner_handle.stop() is not called, resources (like threads)
    might leak or block forever.
    """

    stop_called = {"count": 0}

    class ThreadedHandle(SteerableToolHandle):
        def __init__(self):
            self._done_event = threading.Event()

        async def ask(self, question: str, **kwargs):
            return "answer"

        def interject(self, message: str, **kwargs):
            pass

        def stop(self, reason: str | None = None, **kwargs):
            stop_called["count"] += 1
            self._done_event.set()

        def pause(self, **kwargs):
            pass

        def resume(self, **kwargs):
            pass

        def done(self) -> bool:
            return self._done_event.is_set()

        async def result(self) -> str:
            # Simulate a blocking thread wait - mimicking SimulatedActor
            await asyncio.to_thread(self._done_event.wait)
            return "done"

        # Event API stubs
        async def next_clarification(self):
            return {}

        async def next_notification(self):
            return {}

        async def answer_clarification(self, cid, ans):
            pass

    inner_handle = ThreadedHandle()

    async def outer_tool() -> SteerableToolHandle:
        return inner_handle

    client = new_llm_client(**llm_config)
    client.set_system_message(
        "You are running inside an automated test.\n"
        "1️⃣  Call `outer_tool` with no arguments.\n"
        "2️⃣  Wait for it to complete or be stopped.",
    )

    handle = start_async_tool_loop(
        client,
        message="start",
        tools={"outer_tool": outer_tool},
        timeout=30,
    )

    # Wait until tool is active
    await _wait_for_tool_request(client, "outer_tool")

    # Wait until the handle is likely adopted (placeholder inserted)
    async def _has_tool_placeholder():
        return any(
            m.get("role") == "tool" and m.get("name") == "outer_tool"
            for m in client.messages
        )

    await _wait_for_condition(_has_tool_placeholder, poll=0.05, timeout=5.0)

    # FORCE STOP the outer loop
    # We use _task.cancel() to simulate a hard cancellation (or the race condition where cancellation
    # is processed before the stop-mirror message). This ensures we verify the safety-net
    # in tools_data.cancel_pending_tasks is working.
    # Note: handle.stop() sets an event AND sends a mirror message; if we used that, the
    # mirror message might be processed first, masking the bug.
    handle._task.cancel()

    # Wait for cleanup
    try:
        await handle.result()
    except asyncio.CancelledError:
        pass

    # Wait briefly for propagation (threading event set is immediate but context switch needed)
    await asyncio.sleep(0.1)

    # Verify inner handle stop was called
    assert (
        stop_called["count"] >= 1
    ), "Inner handle stop() should have been called during outer cancellation"

    # Ensure the thread is unblocked
    assert inner_handle._done_event.is_set()

    # Clean up
    if not inner_handle._done_event.is_set():
        inner_handle._done_event.set()
