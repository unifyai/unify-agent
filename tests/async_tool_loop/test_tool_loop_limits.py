import asyncio
import pytest
import unillm

from tests.scripted_model import ScriptedModel, reply, scripted
from unify.common.async_tool_loop import start_async_tool_loop
from unify.common.llm_client import new_llm_client


def _tool_reply(client, call_id: str) -> str:
    (msg,) = [
        m
        for m in client.messages
        if m.get("role") == "tool" and m.get("tool_call_id") == call_id
    ]
    return str(msg["content"])


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


# ── 8. pruning over-quota tool calls across serial turns ────────────────────
class _SerialCallsDriver:
    def __init__(self, client: unillm.AsyncUnify):
        self._client = client
        self._orig = client.generate

    async def __call__(self, **kwargs):
        return await self._orig(**kwargs)


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


# ── 9. timeout resets after LLM response (activity-based) ─────────────────────
@pytest.mark.asyncio
async def test_timeout_resets_after_llm_response(llm_config, monkeypatch):
    """Timeout is activity-based: a slow LLM call should NOT trigger timeout
    as long as the LLM eventually responds. The timeout only guards against
    hung tools, not slow LLM inference."""

    # The scripted model call takes longer than the timeout value.
    # If timeout were measured from loop start, this would fail.
    # With activity-based timeout (reset after LLM response), it should pass.
    async def _slow_reply(_call):
        await asyncio.sleep(0.3)
        return reply("done")

    model = ScriptedModel(query_llm=[_slow_reply])
    with scripted(model):
        client = new_llm_client(**llm_config)
        handle = start_async_tool_loop(
            client,
            message="Reply with 'done'.",
            tools={},
            timeout=0.2,  # Timeout shorter than the LLM delay
            max_steps=100,
        )

        # Should NOT end at the timeout - the LLM response resets the timer
        result = await asyncio.wait_for(handle.result(), 30)
    assert result == "done"
    assert model.kinds() == ["query_llm"]


