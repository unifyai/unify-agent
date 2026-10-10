import asyncio
import pytest
import unillm
from typing import Dict, Callable

from unify.common.async_tool_loop import start_async_tool_loop
from unify.common.tool_spec import ToolSpec
from unify.common.llm_client import new_llm_client

pytestmark = pytest.mark.llm_call


# small helper: pre-seed an assistant tool_call so preflight backfill schedules it immediately
def _preseed_tool_call(
    client: "unillm.AsyncUnify",
    *,
    call_id: str,
    tool_name: str,
    args_json: str,
) -> None:
    client.append_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": args_json},
                    },
                ],
            },
        ],
    )


# ── 1. max_steps safeguard ────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_max_steps_exceeded(llm_config):
    client = new_llm_client(**llm_config)
    # The conversation will contain at least USER + ASSISTANT = 2 messages,
    # so max_steps=1 must raise.
    handle = start_async_tool_loop(
        client,
        message="hello",
        tools={},
        max_steps=1,
        timeout=5,
        raise_on_limit=True,
    )
    with pytest.raises(RuntimeError, match="max_steps"):
        await handle.result()


# ── 2. timeout safeguard (guards against hung tools, not slow LLM) ───────────
@pytest.mark.asyncio
async def test_timeout_exceeded(llm_config):
    """Timeout triggers when a TOOL hangs beyond the timeout threshold.
    The timeout is activity-based and guards against hung tools, not slow LLM.

    The tool takes 5s, much longer than the LLM response (~1s). After the LLM
    finishes, the timer resets, then we enter 'tools only' mode where the
    0.5s timeout is enforced. The hung tool triggers the timeout."""

    async def slow_tool():
        await asyncio.sleep(5)  # Hangs much longer than timeout
        return "done"

    client = new_llm_client(**llm_config)
    # Preseed a tool call so it starts immediately
    _preseed_tool_call(
        client,
        call_id="call_slow_tool",
        tool_name="slow_tool",
        args_json="{}",
    )

    handle = start_async_tool_loop(
        client,
        message="run slow_tool",
        tools={"slow_tool": slow_tool},
        timeout=0.5,  # Tool takes 5s, timeout is 0.5s (after LLM responds)
        max_steps=100,
        raise_on_limit=True,
    )
    with pytest.raises(asyncio.TimeoutError):
        await handle.result()


@pytest.mark.asyncio
async def test_llm_failure_exception_omits_full_client_messages(
    llm_config,
    monkeypatch,
):
    """LLM failures should not embed full client.messages in exception text."""
    from unify.common._async_tool import loop as _loop

    client = new_llm_client(**llm_config)
    huge_content = "x" * 20000
    client.append_messages([{"role": "user", "content": huge_content}])

    async def _failing_generate_with_preprocess(_client, preprocess_msgs, **gen_kwargs):
        raise RuntimeError("synthetic llm failure for regression")

    monkeypatch.setattr(
        _loop,
        "generate_with_preprocess",
        _failing_generate_with_preprocess,
        raising=True,
    )

    handle = start_async_tool_loop(
        client,
        message="trigger llm failure path",
        tools={},
        timeout=5,
        max_steps=10,
    )

    with pytest.raises(Exception) as exc_info:
        await handle.result()

    error_text = str(exc_info.value)
    assert "LLM call failed:" in error_text
    assert "Messages at the time" not in error_text
    assert "synthetic llm failure for regression" in error_text
    assert huge_content[:200] not in error_text
    assert len(error_text) < 2000


# ── 3 & 4. graceful early-exit when limits hit (NO raise) ──────────────────
class _ToolCallingDriver:
    """Deprecated stub removed – tests now instruct the real LLM instead."""

    def __init__(self, client: unillm.AsyncUnify):
        self._client = client
        self._orig = client.generate

    async def __call__(self, **kwargs):
        return await self._orig(**kwargs)


# ── 7. pruning over-quota tool calls (hidden quotas) ────────────────────────
class _MultiCallDriver:
    def __init__(self, client: unillm.AsyncUnify):
        self._client = client
        self._orig = client.generate

    async def __call__(self, **kwargs):
        return await self._orig(**kwargs)


@pytest.mark.asyncio
async def test_prunes_over_quota_tool_calls(llm_config, monkeypatch):
    """When `max_total_calls` is 2, only two calls are scheduled; extras are pruned."""

    counter = {"n": 0}

    async def short_tool():
        counter["n"] += 1
        return "ok"

    client = new_llm_client(**llm_config)
    # Instruct the real LLM to attempt three calls; the loop will prune to 2
    client.set_system_message(
        "You are running inside an automated test. Your FIRST assistant message "
        "MUST contain exactly three parallel tool_calls to `short_tool` "
        "(three separate entries in the tool_calls array of one message). "
        "Do not call the tool once per turn. After emitting those three "
        "tool_calls, finish shortly.",
    )

    handle = start_async_tool_loop(
        client,
        message="start",
        tools={
            # hidden per-loop quota of 2 total calls
            "short_tool": ToolSpec(fn=short_tool, max_total_calls=2),
        },
        prune_tool_duplicates=False,  # allow multiple identical tool_calls for this test
        max_parallel_tool_calls=3,
        timeout=60,
        max_steps=100,
        raise_on_limit=False,
        enable_compression=False,
    )

    await handle.result()

    # Tool ran exactly twice
    assert counter["n"] == 2

    # The first assistant message with `short_tool` requests should be
    # pruned to two entries.
    first_asst_with_short_tool = next(
        m
        for m in client.messages
        if m.get("role") == "assistant"
        and m.get("tool_calls")
        and any(
            tc.get("function", {}).get("name") == "short_tool" for tc in m["tool_calls"]
        )
    )
    short_tool_entries = [
        tc
        for tc in first_asst_with_short_tool["tool_calls"]
        if tc.get("function", {}).get("name") == "short_tool"
    ]
    assert len(short_tool_entries) == 2, (
        f"Expected short_tool to be pruned to 2 entries in the first "
        f"assistant message that requested it, got {len(short_tool_entries)}"
    )
    assert all(
        tc.get("function", {}).get("name") == "short_tool"
        for tc in first_asst_with_short_tool["tool_calls"]
    )


# ── 8. pruning over-quota tool calls across serial turns ────────────────────
class _SerialCallsDriver:
    def __init__(self, client: unillm.AsyncUnify):
        self._client = client
        self._orig = client.generate

    async def __call__(self, **kwargs):
        return await self._orig(**kwargs)


@pytest.mark.asyncio
async def test_prunes_over_quota_serial_calls(llm_config, monkeypatch):
    """When three serial turns each request a call, quota=2 prunes the third."""

    counter = {"n": 0}

    async def short_tool():
        counter["n"] += 1
        return "ok"

    client = new_llm_client(**llm_config)
    # Instruct the model how to conclude after the allowed calls
    client.set_system_message(
        "You are part of an automated test. If tools are available, request the tool `short_tool` exactly once per turn. "
        "After at most two such tool calls have been made, do not request any more tools and reply exactly with the word 'done'.",
    )
    # Let real LLM drive; instruction above ensures at most two calls then 'done'

    handle = start_async_tool_loop(
        client,
        message="start",
        tools={
            "short_tool": ToolSpec(fn=short_tool, max_total_calls=2),
        },
        # Force exactly one tool call per turn for the first two turns
        tool_policy=lambda step, tls: ("required", tls) if step < 2 else ("auto", tls),
        timeout=60,
        max_steps=100,
        raise_on_limit=False,
        enable_compression=False,
    )

    await handle.result()

    # Tool ran exactly twice
    assert counter["n"] == 2

    # Across all assistant messages, only two tool_calls remain after pruning
    # Count only `short_tool` calls — the tool under test. The async tool
    # loop auto-exposes infrastructure tools like `compress_context` for
    # context-window management, and LLMs (Claude-Opus especially) sometimes
    # request them mid-conversation. Those calls aren't subject to
    # short_tool's max_total_calls=2 quota and shouldn't count toward
    # this test's pruning assertion — which is specifically verifying that
    # the quota'd tool was pruned, not that the loop emitted no other
    # tool calls at all.
    short_tool_calls = 0
    for m in client.messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                if tc.get("function", {}).get("name") == "short_tool":
                    short_tool_calls += 1
    assert (
        short_tool_calls == 2
    ), f"Expected short_tool to be pruned to 2 calls, got {short_tool_calls}"


# helper factory: returns an async tool that notes cancellation -------------
def _make_long_tool(cancel_flag: dict):
    async def long_tool(seconds: int):
        try:
            await asyncio.sleep(seconds)
            return "finished"
        except asyncio.CancelledError:
            cancel_flag["cancelled"] = True
            raise

    return long_tool


@pytest.mark.asyncio
async def test_timeout_graceful_termination(llm_config):
    """No exception; pending tool is cancelled when timeout hits."""
    cancel_flag = {}
    client = new_llm_client(**llm_config)
    # Instruct real LLM to call the tool once and keep running; timeout will stop it
    client.set_system_message(
        'You are running inside an automated test. In your FIRST assistant turn, call `long_tool` with {"seconds": 5}. '
        "Keep waiting afterwards.",
    )
    _preseed_tool_call(
        client,
        call_id="call_preseed_timeout",
        tool_name="long_tool",
        args_json='{"seconds": 5}',
    )

    handle = start_async_tool_loop(
        client,
        message="go",
        tools={"long_tool": _make_long_tool(cancel_flag)},
        timeout=0.5,  # real small timeout – tool is already scheduled via backfill
        max_steps=100,
        raise_on_limit=False,
    )
    result = await handle.result()
    assert "Terminating early" in result
    assert cancel_flag.get("cancelled", False)


@pytest.mark.asyncio
async def test_max_steps_graceful_termination():
    """No exception; a call the limit comes before is answered, not run.

    The step limit counts messages and applies at a turn boundary, so a
    call that is already running runs to completion (the timeout and stop()
    bound it, see test_timeout_graceful_termination). Here the limit is
    reached before the preseeded call starts: it is never run, and the
    transcript the loop ends with answers it. No model call is made.
    """
    from tests import cache_discipline_helpers as h

    cancel_flag = {}
    started: list = []
    long_tool = _make_long_tool(cancel_flag)

    async def tracked_long_tool(seconds: int):
        started.append(seconds)
        return await long_tool(seconds)

    with h.scripted(()):
        client = h.new_client()
        _preseed_tool_call(
            client,
            call_id="call_preseed_steps",
            tool_name="long_tool",
            args_json='{"seconds": 5}',
        )
        handle = start_async_tool_loop(
            client,
            message="go",
            tools={"long_tool": tracked_long_tool},
            max_steps=2,  # the preseeded call and the request reach it
            timeout=5,
            raise_on_limit=False,
        )
        result = await handle.result()
    assert result.startswith("🔚 Terminating early: max_steps (2) exceeded")
    assert started == []
    (reply,) = [
        m
        for m in client.messages
        if m.get("role") == "tool" and m.get("tool_call_id") == "call_preseed_steps"
    ]
    assert reply["content"].startswith("Not run: max_steps (2) exceeded")


# ─────────────────────────────────────────────────────────────────────────────
# 5. tool_policy behaviour
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_default_policy_returns_immediately(llm_config):
    """With ``tool_policy=None`` the loop should accept the LLM's first
    answer (no tools) and finish without touching *any* tools."""

    async def noop_tool():  # pragma: no cover – should never be called
        raise RuntimeError("tool should not have been invoked")

    client = new_llm_client(**llm_config)
    handle = start_async_tool_loop(
        client,
        message="You are part of a test. Do *not* call any tools, just return to the user immediately",
        tools={"noop_tool": noop_tool},
        # default → no tool_policy passed
    )
    await handle.result()


@pytest.mark.asyncio
async def test_policy_forces_single_tool_invocation(llm_config):
    """A custom ``tool_policy`` can replicate the old
    ``minimum_tool_turns=1`` semantics by forcing a *required* tool call on the
    first turn only."""

    flag = {"called": False}

    async def dummy_tool():
        flag["called"] = True
        return "ok"

    client = new_llm_client(**llm_config)
    handle = start_async_tool_loop(
        client,
        message="You are part of a test. Do *not* call any tools, just return to the user immediately",
        tools={"dummy_tool": dummy_tool},
        tool_policy=lambda i, tls: ("required", tls) if i < 1 else ("auto", tls),
    )
    await handle.result()

    # The loop had to wait for the tool to finish and therefore should return
    # the *final* assistant content.
    assert flag["called"] is True


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Advanced tool_policy scenarios
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_policy_shows_then_hides_tool(llm_config):
    """
    Step 0: tool is required → LLM calls it exactly once.
    Step 1+: tools hidden → LLM produces text → loop ends.

    Symbolic test: verifies tool_policy correctly gates tool visibility
    across turns. The "required" choice on step 0 removes eval variance.
    """

    call_log: list[str] = []

    async def observed_tool():
        call_log.append("invoked")
        return "ok"

    def require_first_then_hide(step: int, tools: Dict[str, Callable]):
        if step == 0:
            return "required", tools
        return "auto", {}

    client = new_llm_client(**llm_config)
    handle = start_async_tool_loop(
        client,
        "Call the observed_tool when it is available.",
        {"observed_tool": observed_tool},
        tool_policy=require_first_then_hide,
        max_parallel_tool_calls=1,
    )
    await handle.result()

    assert call_log == ["invoked"]  # exactly one call, on step 0


@pytest.mark.asyncio
async def test_policy_two_required_then_auto(llm_config):
    """
    Verify that tool_policy correctly transitions tool_choice from
    'required' to 'auto' after two steps. This is a symbolic test that
    checks infrastructure behavior, not LLM decision-making.
    """

    tool_choice_history = []  # Track what was actually sent to the API
    counter = {"n": 0}

    async def counting_tool():
        counter["n"] += 1
        # Return a clear "done" signal so LLMs don't feel compelled to continue
        if counter["n"] >= 2:
            return f"call {counter['n']} - COMPLETE. Task finished. Do not call again."
        return f"call {counter['n']}"

    def first_two_required(step: int, tools: Dict[str, Callable]):
        choice = "required" if step < 2 else "auto"
        tool_choice_history.append((step, choice))
        return (choice, tools)

    client = new_llm_client(**llm_config)
    handle = start_async_tool_loop(
        client,
        "Call the 'counting_tool' when required. Stop immediately when the tool indicates the task is complete.",
        {"counting_tool": counting_tool},
        tool_policy=first_two_required,
    )
    await handle.result()

    # Symbolic assertion: policy was invoked correctly for at least the first few steps
    assert len(tool_choice_history) >= 2
    assert tool_choice_history[0] == (0, "required")
    assert tool_choice_history[1] == (1, "required")
    if len(tool_choice_history) > 2:
        assert tool_choice_history[2] == (2, "auto")

    # The tool was called at least twice (guaranteed by "required")
    assert counter["n"] >= 2


# ─────────────────────────────────────────────────────────────────────────────
# 7. tool_policy with called_tools history (3-arg policy)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_policy_called_tools_history_gates_on_both(llm_config):
    """A 3-arg tool_policy receives the history of previously called tools and
    can gate on *multiple* required tools being called (order-independent)
    before opening up the full tool set.

    The policy requires both ``tool_a`` and ``tool_b`` to have been invoked
    before any other tool becomes visible.  This is a symbolic test: the LLM
    is instructed which tools to call and in which order.
    """

    call_log: list[str] = []

    async def tool_a():
        call_log.append("a")
        return "tool_a done"

    async def tool_b():
        call_log.append("b")
        return "tool_b done"

    async def tool_final():
        call_log.append("final")
        return "COMPLETE — do not call any more tools."

    REQUIRED = {"tool_a", "tool_b"}
    policy_log: list[tuple] = []

    def gate_policy(
        step: int,
        tools: Dict[str, Callable],
        called_tools: list[str],
    ):
        called_set = set(called_tools)
        satisfied = REQUIRED.issubset(called_set)
        policy_log.append((step, list(called_tools), satisfied))
        if not satisfied:
            gate = {k: v for k, v in tools.items() if k in REQUIRED}
            return "required", gate
        return "auto", tools

    client = new_llm_client(**llm_config)
    handle = start_async_tool_loop(
        client,
        (
            "You are part of a test. Follow these steps exactly:\n"
            "1. Call `tool_a` first.\n"
            "2. Call `tool_b` next.\n"
            "3. Call `tool_final` once it becomes available.\n"
            "4. Then stop and respond to the user.\n"
            "Do NOT skip any steps."
        ),
        {"tool_a": tool_a, "tool_b": tool_b, "tool_final": tool_final},
        tool_policy=gate_policy,
        timeout=60,
    )
    await handle.result()

    # Both gate tools were called before the final tool
    assert "a" in call_log
    assert "b" in call_log

    # The policy received called_tools history and transitioned correctly:
    # step 0 should have empty history and not be satisfied
    assert policy_log[0][1] == []
    assert policy_log[0][2] is False

    # After tool_a is called, the history should contain it
    assert "tool_a" in policy_log[1][1]

    # At some point the policy must have seen both tools and become satisfied
    satisfied_entries = [e for e in policy_log if e[2] is True]
    assert len(satisfied_entries) >= 1


@pytest.mark.asyncio
async def test_policy_legacy_two_arg_still_works(llm_config):
    """Existing 2-arg policies continue to work unchanged when called_tools
    tracking is active (backward compatibility)."""

    flag = {"called": False}

    async def dummy_tool():
        flag["called"] = True
        return "ok"

    def legacy_policy(step: int, tools: Dict[str, Callable]):
        if step < 1:
            return "required", tools
        return "auto", tools

    client = new_llm_client(**llm_config)
    handle = start_async_tool_loop(
        client,
        message="You are part of a test. Do *not* call any tools, just return to the user immediately",
        tools={"dummy_tool": dummy_tool},
        tool_policy=legacy_policy,
    )
    await handle.result()

    assert flag["called"] is True


# ── 9. timeout resets after LLM response (activity-based) ─────────────────────
@pytest.mark.asyncio
async def test_timeout_resets_after_llm_response(llm_config, monkeypatch):
    """Timeout is activity-based: a slow LLM call should NOT trigger timeout
    as long as the LLM eventually responds. The timeout only guards against
    hung tools, not slow LLM inference."""

    client = new_llm_client(**llm_config)

    # Inject delay into the LLM call that exceeds the timeout value.
    # If timeout were measured from loop start, this would fail.
    # With activity-based timeout (reset after LLM response), it should pass.
    orig_generate = client.generate
    call_count = {"n": 0}

    async def _slow_generate(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First LLM call: delay longer than the timeout
            await asyncio.sleep(0.3)
        return await orig_generate(**kwargs)

    monkeypatch.setattr(client, "generate", _slow_generate, raising=True)

    handle = start_async_tool_loop(
        client,
        message="Reply with 'done'.",
        tools={},
        timeout=0.2,  # Timeout shorter than the LLM delay
        max_steps=100,
        raise_on_limit=True,  # Would raise if timeout triggered incorrectly
    )

    # Should NOT raise TimeoutError - the LLM response resets the timer
    result = await handle.result()
    assert "done" in result.lower() or len(result) > 0


@pytest.mark.skip(reason="Will only pass once we support the responses API")
@pytest.mark.asyncio
async def test_max_parallel_tool_calls(llm_config):
    X = 2  # allowed concurrent tool calls per LLM turn
    Y = 5  # requested by the model

    counter = {"n": 0}

    async def short(i: int) -> str:
        counter["n"] += 1
        await asyncio.sleep(0.01)
        return f"ok-{i}"

    short.__name__ = "short"
    short.__qualname__ = "short"

    client = new_llm_client(**llm_config)

    prompt = (
        "You are part of a test. In a single assistant turn, call the tool `short(i: int)` "
        f"exactly {Y} times in parallel with i = 1..{Y}. Make ALL tool calls in one message. "
        "Do not make any further tool calls in later turns. If the platform limits how many "
        "tool calls you can make per turn, issue only as many as allowed and then finish by "
        "replying 'ok'."
    )

    handle = start_async_tool_loop(
        client,
        message=prompt,
        tools={"short": short},
        max_parallel_tool_calls=X,
        prune_tool_duplicates=False,
        timeout=30,
        max_steps=100,
        raise_on_limit=True,
    )

    await handle.result()

    # The first assistant turn must not request more than X tool calls
    first_asst_with_calls = next(
        m
        for m in client.messages
        if m.get("role") == "assistant" and m.get("tool_calls")
    )
    assert 1 <= len(first_asst_with_calls["tool_calls"]) <= X

    # The tool should have been invoked exactly as many times as requested in that turn
    assert counter["n"] == len(first_asst_with_calls["tool_calls"])
