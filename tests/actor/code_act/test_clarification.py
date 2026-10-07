import asyncio

import pytest

from unify.actor.code_act_actor import CodeActActor
from tests.baked_defaults import as_shipped  # noqa: F401

# as_shipped: deleted in step 5 (the clarification channel, replaced by the agent record)
pytestmark = [pytest.mark.llm_call, pytest.mark.usefixtures("as_shipped")]


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_clarification_flow_with_caller_queues():
    """Caller-supplied queues still receive top-level request_clarification."""
    clarification_up_q = asyncio.Queue()
    clarification_down_q = asyncio.Queue()

    planner = CodeActActor()
    active_task = None
    try:
        active_task = await planner.act(
            (
                "Before doing anything else, call request_clarification exactly once "
                "to ask what recipe I want, then wait for my answer. Do not search yet."
            ),
            _clarification_up_q=clarification_up_q,
            _clarification_down_q=clarification_down_q,
        )

        question = await asyncio.wait_for(clarification_up_q.get(), timeout=60)
        assert question.strip(), "Expected a non-empty clarification question"
        await clarification_down_q.put("chocolate cake")

        await active_task.stop("Test complete")
        final_result = await active_task.result()
        assert not str(final_result).startswith(
            "Error:",
        ), f"Unexpected error: {final_result}"
    finally:
        if active_task and not active_task.done():
            try:
                await active_task.stop("Test cleanup")
            except Exception:
                pass
        try:
            await planner.close()
        except Exception:
            pass
        await asyncio.sleep(0.5)


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_clarification_without_caller_queues_uses_handle_api():
    """CM path: omitted queues must still surface via handle.next_clarification()."""
    planner = CodeActActor()
    active_task = None
    try:
        active_task = await planner.act(
            (
                "Before doing anything else, call request_clarification exactly once "
                "with the question 'Which recipe should I search for?', then wait "
                "for the answer. Do not search the web yet."
            ),
        )

        clar = await asyncio.wait_for(active_task.next_clarification(), timeout=60)
        assert isinstance(clar, dict)
        call_id = str(clar.get("call_id") or "")
        assert call_id, f"Expected call_id on clarification event, got: {clar!r}"
        assert clar.get("tool_name") == "request_clarification"
        question = str(clar.get("question") or "")
        assert "recipe" in question.lower(), f"Unexpected question: {question!r}"

        await active_task.answer_clarification(call_id, "chocolate cake")

        await active_task.stop("Test complete")
        final_result = await active_task.result()
        assert not str(final_result).startswith(
            "Error:",
        ), f"Unexpected error: {final_result}"
    finally:
        if active_task and not active_task.done():
            try:
                await active_task.stop("Test cleanup")
            except Exception:
                pass
        try:
            await planner.close()
        except Exception:
            pass
        await asyncio.sleep(0.5)
