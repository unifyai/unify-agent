"""
Interjection behaviours for the async tool loop (live-handle path).

Covers:
- Injecting extra user messages that trigger additional tool calls.
- Preservation and placement of interjections relative to tool results.
- Pre-empting/interrupting LLM turns with interjections.
- Immediate placeholder insertion and backfill of missing tool replies.
"""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from unify.common._async_tool.messages import is_loop_authored_message
from unify.common.async_tool_loop import start_async_tool_loop
from tests.helpers import _handle_project
from unify.common.llm_client import new_llm_client
from tests.async_helpers import (
    _wait_for_tool_request,
    _wait_for_tool_result,
    _wait_for_condition,
    _wait_for_any_assistant_tool_call,
    real_assistant_tool_turns,
    find_assistant_tool_call,
    message_appears_before,
    is_user_interjection_containing,
    _is_synthetic_check_status_stub,
    _is_synthetic_check_status_tool_msg,
)

pytestmark = pytest.mark.llm_call


# --------------------------------------------------------------------------- #
#  TOOL IMPLEMENTATIONS                                                       #
# --------------------------------------------------------------------------- #
async def echo(txt: str) -> str:  # noqa: D401 – simple async tool
    await asyncio.sleep(0.50)
    return txt


async def slow() -> str:
    await asyncio.sleep(0.15)
    return "slow"


async def fast() -> str:
    await asyncio.sleep(0.05)
    return "fast"


def _first_with_tool_calls(msgs: List[dict]) -> int:
    return next(i for i, m in enumerate(msgs) if m.get("tool_calls"))


def _interjection_index(msgs: List[dict], snippet: str) -> int:
    """Return index of a user-role interjection whose content includes snippet.

    Interjections are now sent as simple user messages (not system messages)
    for Claude/Gemini compatibility. This looks for user messages after the
    first user message that contain the given snippet.
    """
    first_user_seen = False
    for i, m in enumerate(msgs):
        if m["role"] == "user":
            if not first_user_seen:
                first_user_seen = True
                continue
            # This is an interjection (user message after the first one)
            if snippet in m.get("content", ""):
                return i
    raise StopIteration(f"No interjection found containing snippet: {snippet}")


def _tool_indices(msgs: List[dict]) -> List[int]:
    return [i for i, m in enumerate(msgs) if m["role"] == "tool"]


def _are_contiguous(indices: List[int]) -> bool:
    return sorted(indices) == list(range(min(indices), max(indices) + 1))


def _is_internal_bookkeeping(msg: dict) -> bool:
    """Identify internal bookkeeping system messages that should be ignored for ordering checks.

    These are system messages injected by the async tool loop for internal purposes
    (e.g., visibility guidance, runtime context, semantic cache hints) that don't
    represent user-visible message ordering.
    """
    if _is_synthetic_check_status_stub(msg) or _is_synthetic_check_status_tool_msg(
        msg,
    ):
        return True
    # Lifecycle, progress and quota notices the loop appends in the user role
    # are bookkeeping too: only a genuine interjection is a turn boundary.
    if is_loop_authored_message(msg):
        return True
    if msg.get("role") != "system":
        return False
    # Check for known internal bookkeeping markers
    return any(
        msg.get(marker)
        for marker in (
            "_visibility_guidance",
            "_runtime_context",
            "_ctx_header",
        )
    )


def _effectively_adjacent(msgs: List[dict], idx1: int, idx2: int) -> bool:
    """Check if idx2 immediately follows idx1 when ignoring internal bookkeeping messages.

    Returns True if there are no non-bookkeeping messages between idx1 and idx2.
    This is useful for verifying message ordering while ignoring internal system
    messages that don't affect the semantic order of tool results and interjections.
    """
    if idx2 <= idx1:
        return False
    # Check that all messages between idx1 and idx2 are internal bookkeeping
    for i in range(idx1 + 1, idx2):
        if not _is_internal_bookkeeping(msgs[i]):
            return False
    return True


# --------------------------------------------------------------------------- #
#  TESTS                                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_interject_triggers_tool_and_result(llm_config):
    """
    Start with echo("A"), then interject to also echo("B"). Expect two tool
    calls and a final plain-text result.
    """
    client = new_llm_client(**llm_config)
    handle = start_async_tool_loop(
        client,
        message=("Use the `echo` tool to output the text 'A'."),
        tools={"echo": echo},
        interrupt_llm_with_interjections=False,
    )

    await _wait_for_tool_request(client, "echo")

    await handle.interject(
        "And also echo B please, the order of the echos doesn't matter."
        "Don't return until both have completed."
        "Use the 'continue' tool to continue a pending tool execution if needed.",
    )

    await handle.result()

    msgs = client.messages

    # Use real_assistant_tool_turns to filter out synthetic check_status_* entries
    # that may be inserted when tools complete after interjections arrive
    tool_turns = real_assistant_tool_turns(msgs)
    assert (
        len(tool_turns) >= 2
    ), f"Expected at least 2 real assistant tool turns, got {len(tool_turns)}"

    # Verify echo("A") was called (using find_assistant_tool_call for robustness)
    echo_a_result = find_assistant_tool_call(msgs, "echo", args_contain={"txt": "A"})
    assert echo_a_result is not None, "Expected echo('A') call not found"

    # Verify echo("B") was called
    echo_b_result = find_assistant_tool_call(msgs, "echo", args_contain={"txt": "B"})
    assert echo_b_result is not None, "Expected echo('B') call not found"

    # Verify ordering: echo("A") call → interjection → echo("B") call
    # Use index-agnostic ordering check
    echo_a_msg, _ = echo_a_result
    echo_b_msg, _ = echo_b_result

    assert message_appears_before(
        msgs,
        lambda m: m is echo_a_msg,
        is_user_interjection_containing("echo B"),
    ), "echo('A') should appear before the interjection"

    assert message_appears_before(
        msgs,
        is_user_interjection_containing("echo B"),
        lambda m: m is echo_b_msg,
    ), "Interjection should appear before echo('B')"

    # Verify the interjection message exists and has expected content
    interjection_found = any(
        m.get("role") == "user" and "echo B" in (m.get("content") or "") for m in msgs
    )
    assert interjection_found, "Interjection message with 'echo B' not found"


@pytest.mark.asyncio
@_handle_project
async def test_backfills_helper_call_reply(llm_config) -> None:
    """
    Pre-seed transcript with an assistant helper tool_call (e.g. wait).
    New behaviour: helper `wait` is pruned (no backfilled tool reply, no chat clutter).
    The pre-seeded assistant helper turn should be removed, and no tool reply should appear.
    """
    client = new_llm_client(**llm_config)

    helper_call_id = "call_TEST_HELPER"
    helper_name = "wait"
    assistant_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": helper_call_id,
                "type": "function",
                "function": {"name": helper_name, "arguments": "{}"},
            },
        ],
    }
    client.append_messages([assistant_msg])

    handle = start_async_tool_loop(
        client=client,
        message="Please proceed.",
        tools={},  # helpers are acknowledged during backfill without execution
    )

    # Allow the loop to process backfill/pruning deterministically by waiting until
    # the pre-seeded helper assistant turn is actually PRUNED from the transcript.
    async def _helper_pruned() -> bool:
        return assistant_msg not in (client.messages or [])

    await _wait_for_condition(_helper_pruned, poll=0.05, timeout=60.0)

    # The pre-seeded helper assistant turn should be pruned
    assert assistant_msg not in client.messages

    # No assistant message should contain the helper tool_call id
    assert not any(
        m.get("role") == "assistant"
        and m.get("tool_calls")
        and any(tc.get("id") == helper_call_id for tc in m["tool_calls"])
        for m in client.messages
    )

    # No tool reply should reference the helper call id
    assert not any(
        m.get("role") == "tool" and m.get("tool_call_id") == helper_call_id
        for m in client.messages
    )

    # Cleanly stop the loop
    await handle.stop()
    final2 = await handle.result()
    assert final2 == "processed stopped early, no result"

    assert handle.done()


@pytest.mark.asyncio
@_handle_project
async def test_patient_interjection_defers_turn(
    llm_config,
    monkeypatch,
) -> None:
    """
    A patient interjection (trigger_immediate_llm_turn=False) that arrives while the LLM
    is already thinking must trigger exactly one extra LLM turn after the current one
    completes, so the interjection is processed.

    NOTE: The non-cancellation guarantee is tested separately in
    test_patient_interjection_does_not_cancel_inflight_llm. This test focuses on
    the deferred turn semantics and message ordering.
    """
    client = new_llm_client(**llm_config)

    from unify.common._async_tool import loop as _loop

    llm_started = asyncio.Event()
    release_first = asyncio.Event()
    call_count = {"n": 0}
    orig_gwp = _loop.generate_with_preprocess

    async def _fake_gwp(_client, preprocess_msgs, **gen_kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            llm_started.set()
            # Wait until the test interjects in patient mode
            await release_first.wait()
            # First assistant turn (no tools)
            _client.messages.append(
                {"role": "assistant", "content": "first", "tool_calls": None},
            )
            return {"ok": True}
        # Second LLM turn – should occur due to deferred turn after patient interjection
        _client.messages.append(
            {"role": "assistant", "content": "second", "tool_calls": None},
        )
        return {"ok": True}

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    h = start_async_tool_loop(
        client=client,
        message="Say hello (no tools).",
        tools={},
        interrupt_llm_with_interjections=True,
        timeout=120,
        max_steps=10,
    )

    # Ensure first LLM thinking has begun, then interject in patient mode
    await asyncio.wait_for(llm_started.wait(), timeout=30.0)
    await h.interject("please consider this later", trigger_immediate_llm_turn=False)  # type: ignore[arg-type]
    # Allow the in-flight LLM to complete naturally
    release_first.set()

    final = await h.result()

    # There should be at least two assistant messages: the original and the deferred one
    # Filter out synthetic check_status stubs that may be inserted for chronological ordering
    assistant_msgs = [
        m
        for m in client.messages
        if m.get("role") == "assistant" and not _is_synthetic_check_status_stub(m)
    ]
    assert len(assistant_msgs) >= 2

    # Verify the user interjection message is present and appears between the two assistant turns
    # Interjections are now user messages (not system messages) for Claude/Gemini compatibility
    # Use index-agnostic ordering check
    assert message_appears_before(
        client.messages,
        lambda m: m is assistant_msgs[0],
        is_user_interjection_containing("please consider this later"),
    ), "First assistant message should appear before the interjection"

    assert message_appears_before(
        client.messages,
        is_user_interjection_containing("please consider this later"),
        lambda m: m is assistant_msgs[-1],
    ), "Interjection should appear before the last assistant message"

    # Final answer should be from the second turn
    assert isinstance(final, str) and final.strip()

    # Cleanup: restore original generator
    monkeypatch.setattr(_loop, "generate_with_preprocess", orig_gwp, raising=True)


@pytest.mark.asyncio
@_handle_project
async def test_patient_interjection_does_not_cancel_inflight_llm(
    llm_config,
    monkeypatch,
) -> None:
    """
    Patient interjection (trigger_immediate_llm_turn=False) must NOT cancel an
    in-flight LLM call that is actively doing work.

    This test simulates an LLM that continues doing async work after the
    interjection arrives. In patient mode, the LLM should complete naturally
    without being cancelled.

    The bug this catches: The cleanup code after asyncio.wait() unconditionally
    cancelled the LLM task before checking if the interjection was in patient mode.
    """
    client = new_llm_client(**llm_config)

    from unify.common._async_tool import loop as _loop

    llm_started = asyncio.Event()
    interjection_sent = asyncio.Event()
    was_cancelled = {"value": False}
    llm_completed_naturally = {"value": False}
    orig_gwp = _loop.generate_with_preprocess

    async def _fake_gwp(_client, preprocess_msgs, **gen_kwargs):
        try:
            llm_started.set()
            # Wait for the interjection to be sent, THEN continue working
            # This simulates an LLM that is actively processing when interjection arrives
            await interjection_sent.wait()
            # Simulate additional LLM work AFTER interjection arrives
            # In patient mode, this work should NOT be cancelled
            await asyncio.sleep(0.1)
            # Append a minimal assistant message the loop expects to see
            _client.messages.append(
                {
                    "role": "assistant",
                    "content": "completed naturally",
                    "tool_calls": None,
                },
            )
            llm_completed_naturally["value"] = True
            return {"ok": True}
        except asyncio.CancelledError:
            was_cancelled["value"] = True
            raise

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    h = start_async_tool_loop(
        client=client,
        message="Say hello (no tools).",
        tools={},
        interrupt_llm_with_interjections=True,
        timeout=120,
        max_steps=10,
    )

    # Ensure LLM thinking has begun
    await asyncio.wait_for(llm_started.wait(), timeout=30.0)

    # Send patient interjection - this should NOT cancel the LLM
    await h.interject("please consider this later", trigger_immediate_llm_turn=False)  # type: ignore[arg-type]

    # Signal that interjection was sent - LLM can now continue its work
    interjection_sent.set()

    final = await h.result()

    # The LLM should have completed naturally without cancellation
    assert (
        llm_completed_naturally["value"] is True
    ), "LLM should complete naturally in patient mode"
    assert (
        was_cancelled["value"] is False
    ), "patient interjection must NOT cancel in-flight LLM"
    assert isinstance(final, str) and final, "loop should complete with a final answer"

    # Cleanup: restore original generator
    monkeypatch.setattr(_loop, "generate_with_preprocess", orig_gwp, raising=True)


@pytest.mark.asyncio
@_handle_project
async def test_immediate_interjection_cancels_llm(llm_config, monkeypatch) -> None:
    """
    When the LLM is currently thinking, an immediate interjection
    (default behaviour) MUST cancel the in-flight LLM call.
    """
    client = new_llm_client(**llm_config)

    from unify.common._async_tool import loop as _loop

    llm_started = asyncio.Event()
    was_cancelled = {"value": False}
    call_count = {"n": 0}
    orig_gwp = _loop.generate_with_preprocess

    async def _fake_gwp(_client, preprocess_msgs, **gen_kwargs):
        call_count["n"] += 1
        # First call: simulate a long-running generation that must be cancelled
        if call_count["n"] == 1:
            llm_started.set()
            try:
                # Never completes unless cancelled
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                was_cancelled["value"] = True
                raise
        # Second call: complete immediately with a minimal assistant message
        _client.messages.append(
            {
                "role": "assistant",
                "content": "ok",
                "tool_calls": None,
            },
        )
        return {"ok": True}

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    h = start_async_tool_loop(
        client=client,
        message="Say hello (no tools).",
        tools={},
        interrupt_llm_with_interjections=True,
        timeout=120,
        max_steps=10,
    )

    # Ensure LLM thinking has begun, then interject in immediate mode (default)
    await asyncio.wait_for(llm_started.wait(), timeout=30.0)
    await h.interject("urgent change!")  # default: trigger_immediate_llm_turn=True

    final = await h.result()
    assert isinstance(final, str) and final, "loop should complete with a final answer"
    assert was_cancelled["value"] is True, "immediate interjection should cancel LLM"

    monkeypatch.setattr(_loop, "generate_with_preprocess", orig_gwp, raising=True)


@pytest.mark.asyncio
@_handle_project
async def test_interjections_processed_successfully(llm_config):
    """
    Fire two interjections (B, then C) and validate FIFO order and sufficient tool work.
    """
    client = new_llm_client(**llm_config)
    client.set_cache(False)
    handle = start_async_tool_loop(
        client,
        (
            "CRITICAL: You must follow these steps ONE AT A TIME. Do NOT skip ahead.\n\n"
            'STEP 1: Call `echo` with {"txt":"A"} and STOP. Do not call any other tools yet.\n\n'
            "STEP 2: WAIT for a user message. When you see a message like 'X please', "
            'call `echo` with {"txt": "X"} for that specific X value.\n\n'
            "STEP 3: Repeat step 2 for each new user message you receive.\n\n"
            "STEP 4: Only after you have received AND responded to multiple user interjections, "
            "and all echo calls have completed, reply with exactly: done\n\n"
            "ABSOLUTE RULES:\n"
            "- You MUST NOT call echo('B') or echo('C') until you ACTUALLY SEE "
            "a user message containing 'B please' or 'C please'.\n"
            "- Do NOT anticipate or predict future interjections.\n"
            "- Do NOT call multiple echo tools in a single response (except the first 'A').\n"
            "- Each user interjection must trigger exactly ONE new echo call.\n"
            "- Never include 'B' or 'C' in your text responses; only via tool calls."
        ),
        {"echo": echo},
    )

    # Wait for echo("A") to be requested (first echo call)
    await _wait_for_tool_request(client, "echo")
    await handle.interject("B please")

    # Wait for the NEXT echo request (echo("B")) using polling-based helper.
    # Can't use _wait_for_tool_request again since it only checks count >= 1.
    await _wait_for_any_assistant_tool_call(client, "echo")
    await handle.interject("C please")

    final = await handle.result()
    assert isinstance(final, str) and final.strip()

    msgs = client.messages
    # Interjections are now user messages (not system messages with wrapper content)
    # Find all user messages after the first one (which is the original request)
    first_user_seen = False
    interjection_contents = []
    for m in msgs:
        if m.get("role") == "user":
            if not first_user_seen:
                first_user_seen = True
                continue
            interjection_contents.append(m.get("content", ""))
    combined = "\n".join(interjection_contents)
    assert "B please" in combined and "C please" in combined
    assert combined.find("B please") < combined.find("C please")

    tool_msgs = [m for m in client.messages if m["role"] == "tool"]
    assert len(tool_msgs) >= 3


@pytest.mark.asyncio
@_handle_project
async def test_tool_result_precedes_interjection(llm_config):
    """
    Run `slow` once then reply "ACK". Interject while running.
    Expect: assistant → tool result → interjection.
    """
    client = new_llm_client(**llm_config)
    handle = start_async_tool_loop(
        client,
        (
            "Run the tool `slow` exactly once, "
            "then reply with the word ACK (nothing else)."
        ),
        {"slow": slow},
        interrupt_llm_with_interjections=False,
    )

    await _wait_for_tool_request(client, "slow")
    await handle.interject("thanks!")

    await handle.result()

    msgs = client.messages
    i_asst = _first_with_tool_calls(msgs)
    i_tool = _tool_indices(msgs)[0]
    i_user = _interjection_index(msgs, "thanks!")

    # Tool result should immediately follow assistant, interjection should
    # effectively follow tool result (ignoring internal bookkeeping messages)
    assert i_asst + 1 == i_tool
    assert _effectively_adjacent(msgs, i_tool, i_user)
    assert len(msgs[i_asst]["tool_calls"]) == 1


@pytest.mark.asyncio
@_handle_project
async def test_parallel_results_shift_interjection(llm_config):
    """
    Run both `fast` and `slow`, interject while they are running.
    Expect both tool results right after the assistant turn, interjection follows.
    """
    client = new_llm_client(**llm_config)
    client.set_cache(False)
    handle = start_async_tool_loop(
        client,
        (
            "Call the tools `fast` and `slow` both at the same time, "
            "then respond with ONLY the word DONE."
        ),
        {"fast": fast, "slow": slow},
        interrupt_llm_with_interjections=True,
    )

    await _wait_for_tool_request(client, "fast")
    await _wait_for_tool_request(client, "slow")

    await _wait_for_tool_result(client, min_results=2)

    await handle.interject("cheers!")
    await handle.result()

    msgs = client.messages
    i_asst = _first_with_tool_calls(msgs)
    tool_idxs = _tool_indices(msgs)[:2]
    i_user = _interjection_index(msgs, "cheers!")

    # Tool results should be contiguous and immediately follow assistant,
    # interjection should effectively follow last tool result
    assert _are_contiguous(tool_idxs)
    assert tool_idxs[0] == i_asst + 1
    assert _effectively_adjacent(msgs, max(tool_idxs), i_user)
    assert len(msgs[i_asst]["tool_calls"]) >= 2


@pytest.mark.asyncio
@_handle_project
async def test_interjection_stops_ongoing_llm(llm_config):
    """The first LLM generation is stopped once the user interjects."""
    client = new_llm_client(**llm_config)
    client.set_cache(False)
    handle = start_async_tool_loop(
        client,
        "Tell me something interesting about whales.",
        {},
    )

    # Interject immediately; the loop will pre-empt in-flight generation if running
    await handle.interject("Actually, make it about dolphins instead!")
    await handle.result()

    assistant_msgs = [m for m in client.messages if m.get("role") == "assistant"]
    assert len(assistant_msgs) == 1

    roles = [m["role"] for m in client.messages]
    # Now we expect 2 user messages: original request + interjection
    assert roles.count("user") == 2
    # The interjection about dolphins should be a user message
    assert any(
        m["role"] == "user" and "dolphins" in m.get("content", "")
        for m in client.messages
    )


# ─────────────────────────────────────────────────────────────────────────────
# Additional interjection behaviours: steerable tool roundtrip + invariants
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_interjectable_tool_roundtrip(llm_config) -> None:
    client = new_llm_client(**llm_config)

    exec_log: List[str] = []

    async def long_running(
        topic: str,
        *,
        _interject_queue: asyncio.Queue[str],
    ) -> str:
        """
        Wait up to 2 s for a steer; echo whichever topic we end up with.
        """
        try:
            steer = await asyncio.wait_for(_interject_queue.get(), timeout=60.0)
            exec_log.append(f"steered→{steer}")
            return f"Topic switched to: {steer}"
        except asyncio.TimeoutError:
            exec_log.append("no-steer")
            return f"Final topic: {topic}"

    long_running.__name__ = "long_running"
    long_running.__qualname__ = "long_running"

    handle = start_async_tool_loop(
        client=client,
        message=(
            "Follow STRICTLY these steps:\n"
            '1️⃣  Call `long_running` with `{ "topic": "cats" }`.\n'
            "2️⃣  WAIT for my next instruction.\n"
            "3️⃣  When I say “Actually, please switch to X instead.” "
            'call the helper `interject_<id>` with `{ "content": "X" }`.\n'
            "4️⃣  After the tool finishes, reply with ONE sentence "
            "mentioning the final topic.\n"
            "Do NOT add extra text between steps."
        ),
        tools={"long_running": long_running},
    )

    # Wait deterministically for the long_running tool to be requested
    await _wait_for_tool_request(client, "long_running")
    # Confirm that the request included the expected arguments once present
    cats_call_seen = any(
        m.get("role") == "assistant"
        and m.get("tool_calls")
        and any(
            tc.get("function", {}).get("name") == "long_running"
            and '"topic":"cats"'
            in (tc.get("function", {}).get("arguments") or "").replace(" ", "")
            for tc in (m.get("tool_calls") or [])
        )
        for m in (client.messages or [])
    )
    assert cats_call_seen, "LLM never called long_running with cats."

    await handle.interject("Actually, please switch to dogs instead.")

    final_answer: str = await handle.result()

    assert exec_log == ["steered→dogs"], "Tool must receive the 'dogs' steer."
    assert "dogs" in final_answer.lower(), "Assistant reply must mention dogs."

    assistant_msgs = [m for m in client.messages if m["role"] == "assistant"]
    assert assistant_msgs[-1]["tool_calls"] is None


@pytest.mark.asyncio
@_handle_project
async def test_immediate_interjection_has_reply(llm_config) -> None:
    """
    When an interjection arrives immediately after an assistant tool_calls turn,
    a tool placeholder must already be present to maintain API ordering.
    """
    client = new_llm_client(**llm_config)

    import time as _time

    def slow_tool(x: int) -> str:
        _time.sleep(0.2)
        return f"ok:{x}"

    slow_tool.__name__ = "slow_tool"
    slow_tool.__qualname__ = "slow_tool"

    handle = start_async_tool_loop(
        client=client,
        message=(
            "Follow these steps strictly:\n"
            '1) Call slow_tool with { "x": 1 }.\n'
            "2) WAIT for my next instruction.\n"
            "3) After my next instruction, reply with ONE word: done."
        ),
        tools={"slow_tool": slow_tool},
    )

    # Ensure the slow_tool call has been requested and capture its call id
    await _wait_for_tool_request(client, "slow_tool")
    call_id: str | None = None
    assistant_idx: int | None = None
    for i, m in enumerate(client.messages or []):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tc = m["tool_calls"][0]
            if tc.get("function", {}).get("name") == "slow_tool" and '"x":1' in (
                tc.get("function", {}).get("arguments") or ""
            ).replace(" ", ""):
                call_id = tc.get("id")
                assistant_idx = i
                break
    assert call_id is not None and assistant_idx is not None

    await handle.interject("finish")

    # Wait until the interjection user message appears; then assert placeholder adjacency
    # Interjections are now simple user messages (not system messages with wrapper)
    async def _saw_interjection_msg() -> bool:
        return any(
            m.get("role") == "user" and "finish" in (m.get("content") or "")
            for m in (client.messages or [])
        )

    await _wait_for_condition(_saw_interjection_msg, poll=0.05, timeout=60.0)
    assert (assistant_idx + 1) < len(client.messages)
    next_msg = client.messages[assistant_idx + 1]
    assert next_msg.get("role") == "tool" and next_msg.get("tool_call_id") == call_id

    final_ans = await handle.result()
    assert isinstance(final_ans, str) and len(final_ans) > 0


@pytest.mark.asyncio
@_handle_project
async def test_backfills_prior_assistant_reply(llm_config) -> None:
    """
    Pre-seed transcript with assistant tool_call but no tool reply.
    The loop must backfill a tool message directly after that assistant turn.
    """
    client = new_llm_client(**llm_config)

    def slow_tool(x: int) -> str:
        return f"ok:{x}"

    slow_tool.__name__ = "slow_tool"
    slow_tool.__qualname__ = "slow_tool"

    call_id = "call_TESTBACKFILL"
    assistant_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "slow_tool", "arguments": '{"x": 1}'},
            },
        ],
    }
    client.append_messages([assistant_msg])

    handle = start_async_tool_loop(
        client=client,
        message="Please proceed.",
        tools={"slow_tool": slow_tool},
    )

    # Wait until the loop backfills a tool message immediately after the assistant helper call
    async def _has_backfill_after_assistant() -> bool:
        msgs = client.messages or []
        has_assistant_with_call = any(
            (m.get("role") == "assistant")
            and (m.get("tool_calls"))
            and any(tc.get("id") == call_id for tc in (m.get("tool_calls") or []))
            for m in msgs
        )
        if not has_assistant_with_call:
            return False
        for i, m in enumerate(msgs):
            if (
                m.get("role") == "assistant"
                and (m.get("tool_calls"))
                and any(tc.get("id") == call_id for tc in (m.get("tool_calls") or []))
            ):
                if (i + 1) < len(msgs) and msgs[i + 1].get("role") == "tool":
                    return True
        return False

    await _wait_for_condition(_has_backfill_after_assistant, poll=0.05, timeout=60.0)
    # Locate the assistant turn and assert the next message is the tool backfill
    assistant_idx = next(
        i
        for i, m in enumerate(client.messages or [])
        if (m.get("role") == "assistant")
        and (m.get("tool_calls"))
        and any(tc.get("id") == call_id for tc in (m.get("tool_calls") or []))
    )
    next_msg = (client.messages or [])[assistant_idx + 1]
    assert next_msg.get("role") == "tool" and next_msg.get("tool_call_id") == call_id

    await handle.stop()
    final2 = await handle.result()
    assert final2 == "processed stopped early, no result"


# --------------------------------------------------------------------------- #
#  SYSTEM MESSAGE PRESERVATION WITH INTERJECTIONS                             #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_system_message_preserved_with_runtime_interjections(llm_config) -> None:
    """
    Verify that the original system message is preserved and followed when
    runtime system messages (e.g., visibility_guidance) are inserted between
    user messages due to interjections.

    This tests the scenario: user -> system(runtime) -> user

    The model should still see and follow the original system instructions
    despite the interleaved runtime system messages. This was a regression
    where the system message was being dropped for Claude models when
    preprocess_msgs was active.
    """
    # Use a very specific, verifiable instruction
    SYSTEM_INSTRUCTION = (
        "IMPORTANT: You must respond with EXACTLY the format 'ECHO: X' "
        "where X is the user's most recent message. No other text allowed."
    )

    client = new_llm_client(**llm_config)
    client.set_system_message(SYSTEM_INSTRUCTION)

    handle = start_async_tool_loop(
        client=client,
        message="first",
        tools={},
        max_consecutive_failures=1,
    )

    # Interject to create the user -> system(runtime) -> user pattern.
    # The visibility_guidance system message gets inserted between user messages.
    await handle.interject("second")

    final = await handle.result()

    # Verify the model followed the system instructions (saw the ECHO format requirement).
    # This would fail if the system message was dropped.
    assert (
        "ECHO" in final.upper()
    ), f"Model should follow system instruction to use ECHO format. Got: {final!r}"
    # The most recent message should be echoed
    assert (
        "second" in final.lower()
    ), f"Model should echo the interjected message 'second'. Got: {final!r}"


@pytest.mark.asyncio
@_handle_project
async def test_multiple_interjections_preserve_system_message(llm_config) -> None:
    """
    Verify system message preservation with multiple rapid interjections.

    Tests: user -> system(runtime) -> user -> user (multiple interjections)

    The model should still see the original system instructions after
    multiple interjections create a complex message interleaving pattern.
    """
    SYSTEM_INSTRUCTION = (
        "Count how many user messages you received and respond with ONLY "
        "a single digit number. No other text, punctuation, or explanation."
    )

    client = new_llm_client(**llm_config)
    client.set_system_message(SYSTEM_INSTRUCTION)

    handle = start_async_tool_loop(
        client=client,
        message="one",
        tools={},
        max_consecutive_failures=1,
    )

    await handle.interject("two")
    await handle.interject("three")

    final = await handle.result()

    # Model should count 3 messages if it sees the system instruction
    assert (
        "3" in final
    ), f"Model should count 3 user messages per system instruction. Got: {final!r}"
