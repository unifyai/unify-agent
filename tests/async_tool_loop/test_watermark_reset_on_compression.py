"""Forced-compression watermark reset.

A compression rebuild is a deliberate full-cache sacrifice: the transcript it
replaces no longer exists, so nothing in the rebuilt one was ever dispatched.
The sent watermark must reset to 0 (not merely end up smaller than the old
transcript length), and the loop must keep running cleanly afterward — proving
the reset didn't leave a stale hash the very next dispatch would trip over.

The model is scripted (``loop_model``, tests/async_tool_loop/conftest.py),
and the compression takes the shipped path: the summary comes from a fork of
the last request, whose scripted answer resets the threshold trigger, and the
compactor is never called.
"""

from __future__ import annotations

import pytest

import unify.common._async_tool.loop as _loop_mod
import unify.common._async_tool.context_compression as _cc_mod
import unify.common.async_tool_loop as _atl_mod
from unify.common.async_tool_loop import start_async_tool_loop
from unify.common.llm_client import new_llm_client
from tests.helpers import _handle_project
from tests.scripted_model import reply

_SYS = (
    "You are in a test. Follow the steps exactly:\n"
    "1. Call `add` with a=2, b=3.\n"
    "2. Call `add` with a=10, b=20.\n"
    "3. Report both results."
)


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


SUMMARY = "Summary: added 2 + 3 = 5 and 10 + 20 = 30; report both."
FINAL = "5 and 30"


async def _compactor_not_called(messages, endpoint, **kwargs):
    raise AssertionError("the shipped path never calls the compactor")


def _make_add(trigger):
    _triggered = False

    def add(a: int, b: int) -> str:
        """Add two numbers."""
        nonlocal _triggered
        if not _triggered:
            _triggered = True
            trigger()
        return str(a + b)

    return add


@pytest.mark.asyncio
@_handle_project
async def test_watermark_resets_to_zero_after_forced_compression(
    llm_config,
    monkeypatch,
    loop_model,
):
    trigger, reset, check = _make_threshold_trigger()
    monkeypatch.setattr(_loop_mod, "context_over_threshold", check)
    monkeypatch.setattr(_cc_mod, "compress_messages", _compactor_not_called)

    def _summary(call):
        reset()
        return reply(SUMMARY)

    # add, add (the first reply over the threshold, since the first add fired
    # the trigger), the turn that must compress, the fork's summary, and the
    # restarted loop's answer.
    model = loop_model(
        reply(calls=[("add", {"a": 2, "b": 3})]),
        reply(calls=[("add", {"a": 10, "b": 20})]),
        reply(calls=[("compress_context", {})]),
        reply(FINAL),
        compression_fork=[_summary],
    )

    # The loop keeps running after a compression rebuild — it immediately
    # dispatches a "continue from where you left off" turn, which correctly
    # re-advances the watermark past 0 again. So the watermark must be
    # observed right at reset time, not after handle.result() returns (by
    # then several more legitimate turns may have happened).
    watermarks_at_reset: list[int | None] = []
    watermarks_before: list[int | None] = []
    _orig_restart = _atl_mod.AsyncToolLoopHandle._restart_with_compressed_context

    async def _spy_restart(self, *a, **kw):
        watermarks_before.append(getattr(self._client, "_sent_watermark", None))
        await _orig_restart(self, *a, **kw)
        watermarks_at_reset.append(getattr(self._client, "_sent_watermark", None))

    monkeypatch.setattr(
        _atl_mod.AsyncToolLoopHandle,
        "_restart_with_compressed_context",
        _spy_restart,
    )

    add = _make_add(trigger)
    client = new_llm_client(**llm_config)
    client.set_system_message(_SYS)

    # Pre-compression: dispatch at least one real request so the watermark is
    # meaningfully non-zero and the dev-mode hash is populated — otherwise a
    # reset-to-zero would be indistinguishable from "never touched".
    handle = start_async_tool_loop(
        client=client,
        message="Add the numbers",
        tools={"add": add},
        timeout=120,
        max_parallel_tool_calls=1,
    )

    result = await handle.result()

    assert (
        handle._compression.count == 1
    ), "compression never fired — test setup is broken"
    assert result == FINAL

    # The rebuild is a deliberate full-cache sacrifice: nothing in the new
    # (compressed) transcript was ever dispatched, so the watermark must have
    # been exactly 0 right after the reset — not merely "smaller than before".
    assert watermarks_at_reset, "compression never restarted the loop — spy never fired"
    assert watermarks_at_reset == [0], watermarks_at_reset
    assert watermarks_before[0] > 0, watermarks_before

    # The loop continued normally after the reset: if the reset had left a
    # stale watermark/hash pair, the very next dispatch's dev-mode integrity
    # assertion in generate_with_preprocess would have raised, and
    # handle.result() above would have surfaced that as an exception instead
    # of a clean result. The restarted loop did send that request.
    assert model.kinds() == ["actor", "actor", "actor", "compression_fork", "actor"]
