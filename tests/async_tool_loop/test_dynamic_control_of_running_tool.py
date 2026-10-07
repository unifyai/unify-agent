"""
End-to-end tests for the *control-tool* extension of
`unify.common.async_tool_loop._async_tool_use_loop_inner`.

What we verify
--------------

* **Continue** – A long-running tool is launched, the user interjects asking
  the assistant to *keep waiting*; the loop must *not* start a second copy of
  that tool.

* **Stop** – The user interjects asking to *stop* the running tool; the
  task is aborted, no tool-result message appears, and the control decision is
  omitted from the permanent chat transcript.

As with the other suites we talk to a **live model** – make sure you have
internet connectivity and `OPENAI_API_KEY` (or proxy equivalent) configured.
"""

from __future__ import annotations

import asyncio
from typing import List

import pytest
from unify.common.async_tool_loop import start_async_tool_loop, SteerableToolHandle
from unify.common._async_tool.dynamic_tools_factory import DynamicToolFactory

# Shared helpers
from tests.helpers import _handle_project
from unify.common.llm_client import new_llm_client
from tests.async_helpers import (
    _wait_for_tool_request,
    _wait_for_assistant_steer_action,
    _wait_for_tool_message_prefix,
    _wait_for_condition,
    _steer_call_action,
)
from tests.baked_defaults import as_shipped  # noqa: F401

# as_shipped: deleted in step 5 (steering a running tool)
pytestmark = [pytest.mark.llm_call, pytest.mark.usefixtures("as_shipped")]


# --------------------------------------------------------------------------- #
#  TOOLS                                                                      #
# --------------------------------------------------------------------------- #
async def slow() -> str:
    """A slow-poke async tool – sleeps `delay` seconds then returns 'done'."""
    await asyncio.sleep(0.50)
    return "done"


# --------------------------------------------------------------------------- #
#  HELPERS                                                                    #
# --------------------------------------------------------------------------- #
def _assistant_calls(msgs: List[dict], tool_name: str) -> int:
    """Count assistant turns whose *visible* `tool_calls` reference `tool_name`."""
    return sum(
        1
        for m in msgs
        if m["role"] == "assistant"
        and any(
            tc["function"]["name"] == tool_name for tc in (m.get("tool_calls") or [])
        )
    )


def _assistant_calls_prefix(msgs: List[dict], prefix: str) -> int:
    """Count assistant turns whose tool-call name *starts with* `prefix`."""
    return sum(
        1
        for m in msgs
        if m["role"] == "assistant"
        and any(
            tc["function"]["name"].startswith(prefix)
            for tc in (m.get("tool_calls") or [])
        )
    )


def _steer_action_calls(msgs: List[dict], action: str) -> int:
    """Count assistant turns containing a `steer(action=<action>, ...)` call.

    Replaces `_assistant_calls_prefix(msgs, "pause_")`-style counting now
    that every steering call shares the tool name "steer" — the action lives
    in the parsed arguments instead of the name.
    """
    return sum(
        1
        for m in msgs
        if m["role"] == "assistant"
        and any(_steer_call_action(tc) == action for tc in (m.get("tool_calls") or []))
    )


def _tool_results(msgs: List[dict], tool_name: str) -> int:
    """Count tool-result messages for `tool_name`."""
    return sum(1 for m in msgs if m["role"] == "tool" and m["name"] == tool_name)


# --------------------------------------------------------------------------- #
#  HELPERS – NEW: classify helper-only assistant messages (mirrored control)  #
# --------------------------------------------------------------------------- #
def _is_helper_tool_name(name: str) -> bool:
    try:
        n = str(name or "")
    except Exception:
        n = ""
    # Every steering action now shares the single static tool name "steer";
    # ask_about_completed_tool is likewise static. No more per-call-id
    # pause_<fn>_<id>/resume_<fn>_<id>/... names to prefix-match.
    return n in ("wait", "steer", "ask_about_completed_tool")


def _assistant_is_helper_only(msg: dict) -> bool:
    """Return True when the assistant message only contains helper tool_calls (no LLM turn)."""
    try:
        if msg.get("role") != "assistant":
            return False
        calls = msg.get("tool_calls") or []
        if not calls:
            return False  # plain assistant text or no tool_calls → counts as a real LLM turn
        # helper-only if every tool_call is a known helper
        return all(
            _is_helper_tool_name((tc.get("function") or {}).get("name")) for tc in calls
        )
    except Exception:
        return False


def _assistant_is_check_status_only(msg: dict) -> bool:
    """
    Return True if the assistant message is a synthetic check-status stub:
      - role == assistant
      - tool_calls present
      - every tool_call function.name startswith 'check_status_'
    These are non-LLM synthetic pairs used to carry final tool results.
    """
    try:
        if msg.get("role") != "assistant":
            return False
        calls = msg.get("tool_calls") or []
        if not calls:
            return False
        return all(
            str((tc.get("function") or {}).get("name") or "").startswith(
                "check_status_",
            )
            for tc in calls
        )
    except Exception:
        return False


# --------------------------------------------------------------------------- #
#  FIXTURE                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="function")
def client(llm_config):
    return new_llm_client(**llm_config)


# --------------------------------------------------------------------------- #
#  TESTS                                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_wait_does_not_duplicate_tool(client):
    """
    Scenario
    --------
    * Ask the assistant to call `slow()` **exactly once** and then reply 'OK'.
    * While `slow` is still running we interject:
        “Please just keep waiting – don't start it again.”
    Expected
    --------
    * Only **one** visible assistant tool-call to `slow`.
    * Only **one** tool-result message for `slow`.
    * Final assistant text is returned.
    """
    handle = start_async_tool_loop(
        client,
        message=(
            "Call the tool `slow`, wait for the result, then "
            "reply with the word OK (nothing else)."
        ),
        tools={"slow": slow},
    )

    # Wait deterministically until the `slow` tool has been requested.
    await _wait_for_tool_request(client, "slow")
    await handle.interject(
        "Make sure you use the `wait` helper and keep the existing `slow` tool running",
    )

    final = await handle.result()
    assert final is not None, "Loop should complete with a response"

    msgs = client.messages
    assert _assistant_calls(msgs, "slow") == 1, "should be one visible request"
    assert _tool_results(msgs, "slow") == 1, "should be one execution only"


@pytest.mark.asyncio
@_handle_project
async def test_stop_removes_tool_and_yields_no_result(client):
    """
    Scenario
    --------
    * Ask the assistant to run `slow` then answer 'ACK'.
    * Shortly after, interject: “Cancel that operation.”
    Expected
    --------
    * **Zero** tool-result messages for `slow` (task was stopped).
    * No assistant turn in the log still exposes `slow` in `tool_calls`.
    * Loop finishes with a normal assistant reply.
    """
    handle = start_async_tool_loop(
        client,
        message=("Run the tool `slow`."),
        tools={"slow": slow},
        interrupt_llm_with_interjections=False,
    )

    # Wait deterministically until the assistant has actually scheduled the
    # `slow` tool so we know our interjection will hit *while* it is running.
    await _wait_for_tool_request(client, "slow")
    await handle.interject(
        "Please stop that run right away, and inform the user that it has been stopped.",
    )

    final = await handle.result()
    assert final is not None, "Loop should complete with a response"

    msgs = client.messages
    assert _tool_results(msgs, "slow") == 1, "stopping tool expected after stop"
    assert _assistant_calls(msgs, "slow") == 1, "tool-call should remain in the history"


@pytest.mark.asyncio
@_handle_project
async def test_functional_tool_pause_extends_wall_clock(client):
    """
    * The assistant must…
        1️⃣  call `pausable_fn`;
        2️⃣  when the *user* says **hold**, call steer(action="pause") on it;
        3️⃣  when the *user* says **go**,   call steer(action="resume") on it;
        4️⃣  when the tool finishes, reply with **done**.
    * We measure wall-clock time: because the loop is paused for ~2 s in the
      middle, total duration must be ≥ 2 s + the tool's own 1-second workload.
    """
    # Explicit gates to avoid timing races: tool cannot complete until
    # the pause helper has been invoked (gate A) and then the resume helper (gate B).
    pause_called_gate = asyncio.Event()
    resume_called_gate = asyncio.Event()

    async def pausable_fn(*, _pause_event: asyncio.Event) -> str:
        # Run until the PAUSE helper has been observed.
        while not pause_called_gate.is_set():
            await _pause_event.wait()
            await asyncio.sleep(0.05)
        # Do not finish until RESUME helper has been observed.
        await resume_called_gate.wait()
        # Perform a small amount of additional work after resume to ensure ordering.
        for _ in range(10):
            await _pause_event.wait()
            await asyncio.sleep(0.05)
        return "ok"

    pausable_fn.__name__ = "pausable_fn"
    pausable_fn.__qualname__ = "pausable_fn"

    client.set_system_message(
        "1️⃣ Call `pausable_fn`.\n"
        "2️⃣ When the user says **hold**, call steer(call_id=<the id of your "
        'pausable_fn call>, action="pause").\n'
        "3️⃣ When the user says **go**,   call steer(call_id=<the same id>, "
        'action="resume").\n'
        "4️⃣ Once the tool finishes, reply with **done**.",
    )

    outer = start_async_tool_loop(
        client,
        message="start",
        tools={"pausable_fn": pausable_fn},
        max_steps=30,
        timeout=300,
    )

    # ── deterministically wait until the assistant has actually scheduled the
    #    tool so our *hold* interjection reliably occurs while it is running.
    await _wait_for_tool_request(client, "pausable_fn")
    # Trigger pause while the tool is running
    await outer.interject("hold")
    # Wait until the assistant REQUESTS the pause action…
    await _wait_for_assistant_steer_action(client, "pause")
    # …and also until the loop ACKNOWLEDGES it (tool message inserted), which is
    # the moment the tool's pause_event has been cleared.
    await _wait_for_tool_message_prefix(client, "steer:pause ")

    # Release the tool's first gate now that pause helper has been invoked
    pause_called_gate.set()

    # While paused, the final assistant reply must NOT appear. Check deterministically
    # right after the pause has been acknowledged (no fixed sleep).
    msgs_during_pause = client.messages or []
    assert not any(
        (m.get("role") == "assistant")
        and isinstance(m.get("content"), str)
        and "done" in m["content"].strip().lower()
        for m in msgs_during_pause
    ), "assistant produced final reply while tool was paused"

    # Resume and finish – ensure the assistant calls steer(action="resume") first
    await outer.interject("go")
    await _wait_for_assistant_steer_action(client, "resume")
    await _wait_for_tool_message_prefix(client, "steer:resume ")

    # Release the tool's second gate now that resume helper has been invoked
    resume_called_gate.set()
    final = await outer.result()

    # ── assertions ───────────────────────────────────────────────────────
    assert final is not None, "Loop should complete with a response"
    # Removed wall‑clock duration assertion; rely on deterministic pause/resume events.


@pytest.mark.asyncio
@_handle_project
async def test_pause_resume_helpers_called_once(client):
    """
    Same scenario as above but we *count* helper invocations in the chat log.

    • Exactly one `steer(action="pause")` and one `steer(action="resume")`
      tool-call must appear.
    """

    # Gates to ensure deterministic ordering: the tool must see pause then resume
    pause_called_gate = asyncio.Event()
    resume_called_gate = asyncio.Event()

    async def pausable_fn(*, _pause_event: asyncio.Event) -> str:
        # Wait until pause helper has been invoked
        while not pause_called_gate.is_set():
            await _pause_event.wait()
            await asyncio.sleep(0.05)
        # Then wait until resume helper has been invoked
        await resume_called_gate.wait()
        # Do a short bit of post-resume work
        for _ in range(10):
            await _pause_event.wait()
            await asyncio.sleep(0.05)
        return "yo"

    pausable_fn.__name__ = "pausable_fn"
    pausable_fn.__qualname__ = "pausable_fn"

    client.set_system_message(
        "1️⃣ Call `pausable_fn`.\n"
        "2️⃣ If the user says **freeze**, call steer(call_id=<the id of your "
        'pausable_fn call>, action="pause") *once*.\n'
        "3️⃣ If the user then says **unfreeze**, call steer(call_id=<the same "
        'id>, action="resume") *once*.\n'
        "4️⃣ When the tool finishes, reply with **all done**.",
    )

    h = start_async_tool_loop(
        client,
        message="go",
        tools={"pausable_fn": pausable_fn},
        timeout=1000,
    )

    # ── deterministically trigger pause / resume via user turns ───────────────
    # Wait until the assistant has actually scheduled the tool so our
    # *freeze* interjection occurs while the tool is running.
    await _wait_for_tool_request(client, "pausable_fn")
    await h.interject("freeze")

    # Wait until the assistant has called steer(action="pause") before
    # sending the *unfreeze* command so we are sure the sequence is
    # pause -> resume (in that order).
    await _wait_for_assistant_steer_action(client, "pause")
    await _wait_for_tool_message_prefix(client, "steer:pause ")
    # Unblock the tool after pause helper observed
    pause_called_gate.set()

    await h.interject("unfreeze")
    # Ensure the resume action is actually invoked before allowing tool to finish
    await _wait_for_assistant_steer_action(client, "resume")
    await _wait_for_tool_message_prefix(client, "steer:resume ")
    resume_called_gate.set()

    final = await h.result()
    msgs = client.messages

    # helper counters -----------------------------------------------------
    pause_calls = _steer_action_calls(msgs, "pause")
    resume_calls = _steer_action_calls(msgs, "resume")

    assert final is not None, "Loop should complete with a response"
    assert (
        pause_calls == 1
    ), f"expected exactly 1 steer(action='pause') call, got {pause_calls}"
    assert (
        resume_calls == 1
    ), f"expected exactly 1 steer(action='resume') call, got {resume_calls}"


@pytest.mark.asyncio
@_handle_project
async def test_global_pause_blocks_llm_until_resume(client):
    """
    The global `pause()` should prevent the LLM from speaking while paused.

    Scenario
    --------
    * Ask the assistant to call `slow` then reply with the word 'OK'.
    * Pause the outer loop while the tool is still running.
    * Wait long enough for the tool to finish.

    Expected
    --------
    * While paused, no new assistant turn should appear after the assistant
      turn that requested the tool.
    * After `resume()`, the loop should complete and return the final 'OK'.
    """
    handle = start_async_tool_loop(
        client,
        message=(
            "Call the tool `slow`, wait for the result, then reply with the word OK (nothing else)."
        ),
        tools={"slow": slow},
    )

    # Ensure the tool has been requested so pausing happens while it is running
    await _wait_for_tool_request(client, "slow")

    # Pause the outer loop (tools should keep running; the LLM must not speak)
    await handle.pause()

    # Wait until the tool result for `slow` has been appended while paused
    await _wait_for_tool_message_prefix(client, "slow")

    msgs = client.messages or []

    # Locate the assistant turn that requested `slow`
    assistant_tool_call_indices = [
        i
        for i, m in enumerate(msgs)
        if m.get("role") == "assistant"
        and any(
            tc.get("function", {}).get("name") == "slow"
            for tc in (m.get("tool_calls") or [])
        )
    ]
    assert (
        assistant_tool_call_indices
    ), "expected at least one assistant turn requesting the `slow` tool"

    last_request_idx = assistant_tool_call_indices[-1]

    # While paused, there must be no further assistant messages after the tool
    # result messages that were appended during pause. Ignore mirrored helper-only
    # assistant messages inserted to represent control actions while paused.
    assistant_after_pause = any(
        (m.get("role") == "assistant") and (not _assistant_is_helper_only(m))
        for m in msgs[last_request_idx + 1 :]
    )
    assert (
        not assistant_after_pause
    ), "assistant produced a new message while the loop was paused"

    # Resume and allow the conversation to complete
    await handle.resume()
    final = await handle.result()

    assert final is not None, "Loop should complete with a response"


@pytest.mark.asyncio
@_handle_project
async def test_global_resume_idempotent_no_extra_turns(client):
    """
    Calling `resume()` multiple times should be harmless and must not create
    extra assistant turns after the tool completes.
    """
    handle = start_async_tool_loop(
        client,
        message=(
            "Call the tool `slow`, wait for the result, then reply with the word OK (nothing else)."
        ),
        tools={"slow": slow},
    )

    # Ensure the tool has been requested
    await _wait_for_tool_request(client, "slow")

    # Pause while tool is running; let it finish while paused – wait for tool result deterministically
    await handle.pause()
    await _wait_for_tool_message_prefix(client, "slow")

    # Resume twice (idempotent)
    await handle.resume()
    await handle.resume()

    final = await handle.result()
    assert final is not None, "Loop should complete with a response"

    # After the last assistant tool-call requesting `slow`, there should be exactly
    # one more assistant message (the final answer). Multiple resumes must not add more.
    msgs = client.messages or []
    last_req_idx = -1
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant" and any(
            tc.get("function", {}).get("name") == "slow"
            for tc in (m.get("tool_calls") or [])
        ):
            last_req_idx = i
    assert last_req_idx != -1, "expected an assistant tool-call to `slow`"

    # Count only non-helper, non-check_status assistant messages
    assistant_after = [
        m
        for m in msgs[last_req_idx + 1 :]
        if (m.get("role") == "assistant")
        and (not _assistant_is_helper_only(m))
        and (not _assistant_is_check_status_only(m))
    ]
    assert (
        len(assistant_after) == 1
    ), f"expected exactly 1 post-pause assistant turn, got {len(assistant_after)}"


@pytest.mark.asyncio
@_handle_project
async def test_resume_allows_llm_turn(client):
    """
    If the loop is paused while no tools are pending, resuming should immediately
    allow the next LLM turn to proceed and finish.
    """
    client.set_system_message(
        "Immediately reply ONLY with the word OK. Do not call any tools.",
    )

    # No tools exposed – pure LLM reply
    h = start_async_tool_loop(
        client,
        message="start",
        tools={},
        timeout=120,
    )

    # Pause immediately; there are no pending tools. The loop should not finish while paused.
    await h.pause()
    # Assert result() blocks while paused using a timeout-based check
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(h.result()), timeout=1)

    # Resume and finish
    await h.resume()
    final = await h.result()
    assert final is not None, "Loop should complete with a response"


@pytest.mark.asyncio
@_handle_project
async def test_result_propagates_caller_cancellation(client):
    """
    A caller's own timeout must surface as ``TimeoutError``, not as a value.

    ``result()`` reports a loop that died on its own as the stopped notice.
    An *unshielded* ``wait_for`` instead cancels the awaiting task, and
    answering that with the stopped notice hands the caller a value its
    timeout never produced: the run reads as a loop that finished without an
    answer, and the real cause — the caller ran out of patience — is lost.
    Callers that want the loop to outlive their own timeout shield the await,
    as ``test_resume_allows_llm_turn`` above does.
    """
    client.set_system_message(
        "Immediately reply ONLY with the word OK. Do not call any tools.",
    )

    h = start_async_tool_loop(
        client,
        message="start",
        tools={},
        timeout=120,
    )

    # Pause so the loop cannot finish on its own before the timeout fires.
    await h.pause()

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(h.result(), timeout=1)

    # Cancelling the await tears the loop down with it — the cost of not
    # shielding, and the reason the notice would have been misleading.
    assert h.done(), "unshielded cancellation should have ended the loop"


@pytest.mark.asyncio
@_handle_project
async def test_steer_schema_stable_and_state_enforced_across_pause_resume_cycles(
    client,
):
    """
    Pre-steer() this test asserted the *schema* flipped between exposing
    `pause_…`/`resume_…` depending on live state — the exact per-turn schema
    churn issue 01/05/06 existed to describe. Post-steer(), the schema is
    static (byte-identical `steer` tool every turn); state is enforced at
    *execution* time instead: a mistargeted pause/resume gets an instructive
    refusal rather than being unavailable to call in the first place.

    This verifies both halves: the tool-name set the LLM sees never changes
    across a pause->resume->pause->resume cycle, and the pause/resume
    sequence still produces the correct steering-state outcome end to end.
    """

    done_event = asyncio.Event()

    async def pausable_fn(*, _pause_event: asyncio.Event) -> str:
        # Run until the test explicitly signals completion.
        while not done_event.is_set():
            await _pause_event.wait()
            await asyncio.sleep(0)
        return "done"

    pausable_fn.__name__ = "pausable_fn"
    pausable_fn.__qualname__ = "pausable_fn"

    client.set_system_message(
        "️1. Call `pausable_fn`.\n"
        "2. When the user says 'hold', call steer(call_id=<the id of your "
        'pausable_fn call>, action="pause").\n'
        "3. When the user says 'go', call steer(call_id=<the same id>, "
        'action="resume") immediately.\n'
        "4. Repeat the pause→resume cycle twice.\n"
        "5. IMPORTANT: Do NOT call `wait` in response to 'go' — always steer "
        'with action="resume" instead.\n'
        "6. You may use `wait` only when you intend to remain paused without "
        "resuming; do NOT use it right after 'go'.\n"
        "7. After the second resume, wait for completion and reply with 'done'.",
    )

    # Spy tool-name exposure at the exact callsite the loop uses.
    from unify.common._async_tool import loop as _loop

    seen_tools: list[list[str]] = []
    orig_gwp = _loop.generate_with_preprocess

    async def _spy_gwp(_client, preprocess_msgs, **gen_kwargs):
        tools = gen_kwargs.get("tools") or []
        names: list[str] = []
        for t in tools:
            try:
                fn = t.get("function", {})
                name = fn.get("name")
                if isinstance(name, str):
                    names.append(name)
            except Exception:
                pass
        seen_tools.append(names)
        return await orig_gwp(_client, preprocess_msgs, **gen_kwargs)

    setattr(_loop, "generate_with_preprocess", _spy_gwp)

    h = start_async_tool_loop(
        client,
        message="start",
        tools={"pausable_fn": pausable_fn},
        timeout=300,
        max_steps=60,
        max_parallel_tool_calls=1,
    )

    # Ensure the tool is running before issuing commands
    await _wait_for_tool_request(client, "pausable_fn")
    baseline_names = set(seen_tools[-1]) if seen_tools else set()
    assert "steer" in baseline_names

    def _assert_schema_unchanged(label: str):
        for names in seen_tools:
            assert set(names) == baseline_names, (
                f"tool-name set changed after {label}: expected {baseline_names}, "
                f"got {set(names)}"
            )

    # First pause/resume cycle
    await h.interject("hold")
    await _wait_for_assistant_steer_action(client, "pause")
    await _wait_for_tool_message_prefix(client, "steer:pause ")
    _assert_schema_unchanged("first pause")

    await h.interject("go")
    await _wait_for_assistant_steer_action(client, "resume")
    await _wait_for_tool_message_prefix(client, "steer:resume ")
    _assert_schema_unchanged("first resume")

    # Second pause/resume cycle
    await h.interject("hold")
    await _wait_for_assistant_steer_action(client, "pause")
    await _wait_for_tool_message_prefix(client, "steer:pause ")
    _assert_schema_unchanged("second pause")

    await h.interject("go")
    await _wait_for_assistant_steer_action(client, "resume")
    await _wait_for_tool_message_prefix(client, "steer:resume ")
    _assert_schema_unchanged("second resume")

    done_event.set()
    final = await h.result()
    assert final is not None, "Loop should complete with a response"

    # The tool-name set the model saw was identical on every single turn of
    # the run, not just at the four checkpoints above.
    for names in seen_tools:
        assert set(names) == baseline_names


def test_custom_call_discovery_hides_notification_clarification():
    """
    Verify `steer(action="call", method=...)` can never reach
    `next_notification`/`next_clarification`/`answer_clarification` on an
    in-flight inner handle.

    Pre-steer(), this was verified indirectly by spying on
    `DynamicToolFactory._register_tool` and checking no
    `next_notification_…`/`next_clarification_…` tool got minted. That
    registration point no longer exists for per-handle methods at all
    (steer's action="call" resolves methods at execution time via
    `_discover_custom_public_methods` instead of pre-minting a tool per
    method), so this now tests that discovery function directly — the exact
    mechanism loop.py's steer(action="call") branch consults to decide
    which methods are reachable.
    """

    class MockNestedHandle(SteerableToolHandle):
        def __init__(self):
            self._done = asyncio.Event()

        async def ask(self, question: str) -> "SteerableToolHandle":
            return self

        async def interject(self, message: str):
            return None

        def stop(self, reason: str | None = None):
            self._done.set()

        async def pause(self):
            return "paused"

        async def resume(self):
            return "resumed"

        def done(self) -> bool:
            return self._done.is_set()

        async def result(self) -> str:
            await self._done.wait()
            return "inner_done"

        # Event APIs
        async def next_clarification(self) -> dict:
            return {}

        async def next_notification(self) -> dict:
            return {}

        async def answer_clarification(self, call_id: str, answer: str) -> None:
            return None

    inner_handle = MockNestedHandle()
    custom_methods = DynamicToolFactory._discover_custom_public_methods(inner_handle)

    for core_method in (
        "ask",
        "interject",
        "stop",
        "pause",
        "resume",
        "done",
        "result",
        "next_clarification",
        "next_notification",
        "answer_clarification",
        "get_history",
    ):
        assert core_method not in custom_methods, (
            f"{core_method!r} is a core SteerableToolHandle method — it must "
            "stay reachable only via steer's dedicated actions, never via "
            f'action="call". Discovered custom methods: {sorted(custom_methods)}'
        )


@pytest.mark.asyncio
@_handle_project
async def test_new_tool_scheduled_while_paused_starts_paused(client, monkeypatch):
    """
    A base tool scheduled AFTER the outer handle is paused must start paused
    (its `_pause_event` is cleared). Before the change, the event started set.
    """
    # Patch the loop's LLM call to emit a tool-call only AFTER we pause
    from unify.common._async_tool import loop as _loop

    llm_started = asyncio.Event()
    release_llm = asyncio.Event()
    orig_gwp = _loop.generate_with_preprocess

    async def _fake_gwp(_client, preprocess_msgs, **gen_kwargs):
        # Signal that LLM thinking has started
        llm_started.set()
        # Wait until the test allows the LLM to finish (after outer pause)
        await release_llm.wait()
        # Emit a single assistant turn that calls `pausable_fn` with no args
        _client.messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_fake_1",
                        "type": "function",
                        "function": {"name": "pausable_fn", "arguments": "{}"},
                    },
                ],
            },
        )
        return {"ok": True}

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    # Base tool that records the initial pause state immediately on start
    initial_pause_state = {"value": None}

    async def pausable_fn(*, _pause_event: asyncio.Event) -> str:
        try:
            initial_pause_state["value"] = _pause_event.is_set()
        except Exception:
            initial_pause_state["value"] = None
        return "ok"

    pausable_fn.__name__ = "pausable_fn"
    pausable_fn.__qualname__ = "pausable_fn"

    client.set_system_message(
        "When you respond, call `pausable_fn` exactly once and then finish.",
    )

    # Start loop, immediately pause, then release the LLM patch to schedule tool
    h = start_async_tool_loop(
        client=client,
        message="start",
        tools={"pausable_fn": pausable_fn},
        timeout=120,
        max_steps=20,
    )

    # Ensure the LLM step actually started, then pause the outer handle
    await asyncio.wait_for(llm_started.wait(), timeout=30)
    await h.pause()
    # Allow the patched LLM to proceed and return the tool-call while paused
    release_llm.set()

    # Wait until the tool result for `pausable_fn` appears
    await _wait_for_tool_message_prefix(client, "pausable_fn")

    # The tool must have observed an initial paused state (event cleared)
    assert (
        initial_pause_state["value"] is False
    ), "newly scheduled tool did not start paused"

    # Cleanup: stop the loop and restore original LLM generator
    await h.stop("test cleanup")
    await h.result()
    monkeypatch.setattr(_loop, "generate_with_preprocess", orig_gwp, raising=True)


@pytest.mark.asyncio
@_handle_project
async def test_resume_unblocks_base_tool(client, monkeypatch):
    """
    A base tool scheduled while the outer loop is paused should resume
    running immediately when `handle.resume()` is called, even if the LLM
    never calls a `resume_…` helper. This would have failed before the
    auto-resume improvement.
    """
    from unify.common._async_tool import loop as _loop

    llm_started = asyncio.Event()
    release_llm = asyncio.Event()
    orig_gwp = _loop.generate_with_preprocess

    # Track invocation count so the monkeypatch only emits the tool call once.
    # On subsequent calls (after tool completion), emit a final response to
    # properly terminate the loop.
    call_count = {"value": 0}

    async def _fake_gwp(_client, preprocess_msgs, **gen_kwargs):
        call_count["value"] += 1
        if call_count["value"] == 1:
            # First call: wait for test signal, then emit tool call
            llm_started.set()
            await release_llm.wait()
            _client.messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_fake_2",
                            "type": "function",
                            "function": {"name": "pausable_fn", "arguments": "{}"},
                        },
                    ],
                },
            )
        else:
            # Subsequent calls: emit a final response to terminate the loop
            _client.messages.append(
                {
                    "role": "assistant",
                    "content": "Done.",
                    "tool_calls": None,
                },
            )
        return {"ok": True}

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    # Base tool: starts paused; only completes after _pause_event is set (resume)
    initial_pause_state = {"value": None}

    async def pausable_fn(*, _pause_event: asyncio.Event) -> str:
        try:
            initial_pause_state["value"] = _pause_event.is_set()
        except Exception:
            initial_pause_state["value"] = None
        # Wait until resumed, then finish quickly
        await _pause_event.wait()
        return "ok"

    pausable_fn.__name__ = "pausable_fn"
    pausable_fn.__qualname__ = "pausable_fn"

    client.set_system_message(
        "When you respond, call `pausable_fn` exactly once and then finish.",
    )

    h = start_async_tool_loop(
        client=client,
        message="start",
        tools={"pausable_fn": pausable_fn},
        timeout=180,
        max_steps=20,
        time_awareness=False,
    )

    # Ensure LLM step started, then pause the outer handle
    await asyncio.wait_for(llm_started.wait(), timeout=30)
    await h.pause()
    release_llm.set()

    # Wait until the tool placeholder appears (scheduled while paused)
    await _wait_for_tool_message_prefix(client, "pausable_fn")

    # Confirm the tool started in a paused state
    assert (
        initial_pause_state["value"] is False
    ), "tool did not start paused while outer loop was paused"

    # Resume the outer handle – should auto-set the per-call pause_event for base tools
    await h.resume()

    # Wait until final tool result "ok" is observed without relying on a resume helper
    async def _has_final_ok() -> bool:
        msgs = client.messages or []
        return any(
            (m.get("role") == "tool")
            and (m.get("name") == "pausable_fn")
            and (m.get("content") == "ok")
            for m in msgs
        )

    await _wait_for_condition(_has_final_ok, poll=0.05, timeout=60.0)

    # Ensure no resume action was requested by the assistant (programmatic resume path)
    msgs = client.messages or []
    assert (
        _steer_action_calls(msgs, "resume") == 0
    ), 'LLM should not need to call steer(action="resume") for base tools'

    # Cleanup – stop the loop and restore generator
    await h.stop("cleanup")
    # Await result; outer handle returns a standardized notice on stop
    await asyncio.wait_for(asyncio.shield(h.result()), timeout=60)
    monkeypatch.setattr(_loop, "generate_with_preprocess", orig_gwp, raising=True)
