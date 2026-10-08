"""Extra compression tools: store_skills alongside compress_context at 70%.

When the context window hits the 70% threshold and ``extra_compression_tools``
is configured, the loop exposes those tools alongside ``compress_context``
(with ``tool_choice="required"``).  The prompt guides the LLM to call
``store_skills`` first **if** the trajectory contains unstored skills worth
preserving, then ``compress_context``.  The LLM may skip ``store_skills``
if it judges there is nothing new to store.

This test monkeypatches ``context_over_threshold`` to simulate reaching the
70% threshold after the model has run two cells, and verifies that the
turn that must compress offers exactly those tools with the call required,
and that both calls run, ``store_skills`` first. The model is scripted
(``tests/cache_discipline_helpers``): which turn reaches the threshold no
longer depends on a live model's choices, and nothing leaves the process.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests import cache_discipline_helpers as h
from unify.actor.code_act_actor import CodeActActor
from unify.common.async_tool_loop import AsyncToolLoopHandle
from unify.function_manager.function_manager import FunctionManager


class _StubGuidanceManager:
    """Minimal GuidanceManager stand-in with the methods the actor registers."""

    def search(self, references=None, k=10):
        return []

    def filter(self, filter=None, offset=0, limit=100):
        return []

    def get_guidance(self, *, guidance_id: int):
        """Fetch one guidance entry by id."""
        return {"details": {"guidance_id": guidance_id}}

    def add_guidance(self, *, title, content, function_ids=None):
        return {"details": {"guidance_id": 1}}

    def update_guidance(
        self,
        *,
        guidance_id,
        title=None,
        content=None,
        function_ids=None,
    ):
        return {"details": {"guidance_id": guidance_id}}

    def delete_guidance(self, *, guidance_id):
        return {"deleted": True}

    def reconcile_dependencies(self, *, guidance_ids=None):
        """Refresh structured link debt for related functions."""
        return {"outcome": "checked", "details": {"guidance_ids": guidance_ids or []}}


def _make_delayed_threshold(trigger_after: int = 2):
    """Build a ``context_over_threshold`` replacement that triggers the 70%
    threshold only after *trigger_after* checks (giving the LLM enough
    turns to build a meaningful trajectory)."""
    _check_count = [0]

    def _fake(n_tokens: int, threshold: float, max_input_tokens: int) -> bool:
        if threshold >= 0.7:
            _check_count[0] += 1
            return _check_count[0] > trigger_after
        return False

    return _fake


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_compress_threshold_exposes_extra_tools_and_compress_is_called():
    """When the 70% threshold fires with ``extra_compression_tools``, the
    loop exposes ``store_skills`` + ``compress_context`` (required) on that
    turn only, and a turn calling both runs ``store_skills`` first."""

    fm = FunctionManager(include_primitives=False)
    gm = _StubGuidanceManager()

    actor = CodeActActor(
        function_manager=fm,
        guidance_manager=gm,
        timeout=120,
    )

    mock_proactive_handle = MagicMock()

    async def _quick_storage_result():
        return "Stored 1 function: test_helper"

    mock_proactive_handle.result = _quick_storage_result
    mock_proactive_handle.done = MagicMock(return_value=True)

    async def _mock_restart(self):
        async def _done():
            return "Context compressed. Continuing from where you left off."

        self._task = asyncio.create_task(_done())

    def _cell(code: str):
        return lambda: h.completion(
            calls=[("execute_code", {"thought": "Running it.", "code": code})],
        )

    replies = [
        _cell(
            "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
        ),
        _cell("print(fib(10))"),
        lambda: h.completion(
            calls=[
                ("store_skills", {"request": "Store the iterative fib function."}),
                ("compress_context", {}),
            ],
        ),
    ]

    with (
        h.scripted(replies) as provider,
        patch(
            "unify.common._async_tool.loop.context_over_threshold",
            _make_delayed_threshold(trigger_after=1),
        ),
        patch(
            "unify.actor.code_act_actor._start_proactive_storage_loop",
            return_value=mock_proactive_handle,
        ),
        patch(
            "unify.actor.code_act_actor._start_storage_check_loop",
            return_value=None,
        ),
        patch.object(
            AsyncToolLoopHandle,
            "_restart_with_compressed_context",
            _mock_restart,
        ),
        patch(
            "unify.actor.code_act_actor.publish_manager_method_event",
            new_callable=AsyncMock,
        ),
    ):
        try:
            handle = await actor.act(
                "Write a Python function that computes the Fibonacci sequence "
                "iteratively. Execute it with n=10 and show me the result.",
                can_store=True,
                persist=False,
                clarification_enabled=False,
            )
            result = await asyncio.wait_for(handle.result(), timeout=120)

            # Only the turn that must compress offers the compression tools.
            names = [
                sorted(t["function"]["name"] for t in r["tools"] or [])
                for r in h.session_requests(provider.requests)
            ]
            assert names == [
                ["execute_code"],
                ["execute_code"],
                ["compress_context", "store_skills"],
            ]
            assert provider.requests[2]["tool_choice"] == "required"

            history = handle.get_history()
            tool_names: list[str] = []
            for msg in history:
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        tool_names.append(tc["function"]["name"])

            assert tool_names == [
                "execute_code",
                "execute_code",
                "store_skills",
                "compress_context",
            ]
            answered = {m.get("name") for m in history if m.get("role") == "tool"}
            assert {"store_skills", "compress_context"} <= answered
            assert result == "Context compressed. Continuing from where you left off."
        finally:
            try:
                await actor.close()
            except Exception:
                pass
