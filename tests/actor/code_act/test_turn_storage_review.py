"""Tests for mid-session (turn-boundary) storage reviews.

By default a persistent session is reviewed once, when it ends, over the
whole trajectory. ``_StorageCheckHandle`` can opt into reviewing at each
completed turn (``type="response"`` notification) that ran tools
(``UNIFY_TURN_STORAGE_REVIEWS``), recording the summary for the final
review and leaving a transcript note in the live loop so the next request
can execute what was stored.

All tests here are symbolic infrastructure tests: the inner handle and the
review loop are mocked; what is under test is the trigger, the gating, the
coalescing, and the plumbing of summaries and notes.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unify.actor.code_act_actor import (
    _CURRENT_AGENT_CONTEXT,
    AgentContext,
    _StorageCheckHandle,
)


def _make_inner_handle(
    notifications: "asyncio.Queue[dict]",
    result_future: "asyncio.Future[str]",
    messages: list[dict],
) -> MagicMock:
    inner = MagicMock()

    async def _result():
        return await result_future

    inner.result = _result

    async def _next_notification():
        return await notifications.get()

    inner.next_notification = _next_notification

    mock_client = MagicMock()
    mock_client.messages = messages
    inner._client = mock_client

    mock_task = MagicMock()
    mock_task.get_ask_tools = MagicMock(return_value={})
    mock_task.get_completed_tool_metadata = MagicMock(return_value={})
    inner._task = mock_task

    inner._queue = asyncio.Queue()
    return inner


def _mock_actor() -> MagicMock:
    actor = MagicMock()
    actor.function_manager = MagicMock()
    actor.guidance_manager = MagicMock()
    return actor


def _mock_review_handle(summary: str) -> MagicMock:
    review = MagicMock()

    async def _result():
        return summary

    review.result = _result
    return review


@pytest.fixture(autouse=True)
def _fresh_agent_context():
    """Isolate each test from the ContextVar's shared default AgentContext."""
    token = _CURRENT_AGENT_CONTEXT.set(AgentContext())
    try:
        yield
    finally:
        _CURRENT_AGENT_CONTEXT.reset(token)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_turn_review_runs_on_response_with_tool_activity():
    """A turn boundary with completed tool activity triggers a live-session
    review; its summary is recorded for the final review and delivered to
    the live loop as a transcript note."""
    notifications: asyncio.Queue[dict] = asyncio.Queue()
    result_future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
    messages = [
        {"role": "user", "content": "file the week 1 spend report"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "content": "report filed"},
        {"role": "assistant", "content": "Filed week 1."},
    ]
    inner = _make_inner_handle(notifications, result_future, messages)

    ctx = AgentContext()
    ctx_token = _CURRENT_AGENT_CONTEXT.set(ctx)
    try:
        with (
            patch(
                "unify.actor.code_act_actor._start_storage_check_loop",
            ) as mock_loop,
            patch(
                "unify.actor.code_act_actor.publish_manager_method_event",
                new_callable=AsyncMock,
            ),
        ):
            mock_loop.return_value = _mock_review_handle(
                "Stored function 7 `file_weekly_report(week)`.",
            )

            handle = _StorageCheckHandle(
                inner=inner,
                actor=_mock_actor(),
                turn_reviews_enabled=True,
            )

            await notifications.put({"type": "response", "content": "Filed week 1."})

            async def _review_done() -> bool:
                task = handle._turn_review_task
                return task is not None and task.done()

            for _ in range(200):
                if await _review_done():
                    break
                await asyncio.sleep(0.01)
            assert await _review_done(), "turn review task should have completed"

            assert mock_loop.call_count == 1
            kwargs = mock_loop.call_args.kwargs
            assert kwargs["live_session"] is True
            assert kwargs["original_result"] == "Filed week 1."

            assert ctx.proactive_storage_summaries == [
                "Stored function 7 `file_weekly_report(week)`.",
            ]

            note = inner._queue.get_nowait()
            assert "_transcript_note" in note
            assert "Stored function 7" in note["_transcript_note"]["text"]

            compact = inner._queue.get_nowait()
            assert compact == {
                "_compact_transcript": {"reviewed_messages": 4},
            }

            relayed = []
            while not handle._notification_q.empty():
                relayed.append(handle._notification_q.get_nowait())
            kinds = [n.get("type") for n in relayed if isinstance(n, dict)]
            assert "turn_storage_review_complete" in kinds

            # Session ends: the final Phase-2 review sees the turn summary
            # as a prior pass.
            result_future.set_result("session over")
            for _ in range(300):
                if handle.done():
                    break
                await asyncio.sleep(0.01)
            assert handle.done()
            assert mock_loop.call_count == 2
            final_kwargs = mock_loop.call_args.kwargs
            assert final_kwargs.get("live_session", False) is False
            assert final_kwargs["proactive_summaries"] == [
                "Stored function 7 `file_weekly_report(week)`.",
            ]
    finally:
        _CURRENT_AGENT_CONTEXT.reset(ctx_token)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_turn_review_skipped_without_tool_activity():
    """A pure-conversation turn (no completed tool results) does not start
    a review — there is nothing new to distill."""
    notifications: asyncio.Queue[dict] = asyncio.Queue()
    result_future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi!"},
    ]
    inner = _make_inner_handle(notifications, result_future, messages)

    with (
        patch(
            "unify.actor.code_act_actor._start_storage_check_loop",
        ) as mock_loop,
        patch(
            "unify.actor.code_act_actor.publish_manager_method_event",
            new_callable=AsyncMock,
        ),
    ):
        mock_loop.return_value = None
        handle = _StorageCheckHandle(
            inner=inner,
            actor=_mock_actor(),
            turn_reviews_enabled=True,
        )

        await notifications.put({"type": "response", "content": "Hi!"})

        async def _boundary_settled() -> bool:
            task = handle._turn_review_task
            return task is not None and task.done()

        for _ in range(200):
            if await _boundary_settled():
                break
            await asyncio.sleep(0.01)
        assert await _boundary_settled()
        assert mock_loop.call_count == 0
        assert inner._queue.empty()

        result_future.set_result("done")
        for _ in range(300):
            if handle.done():
                break
            await asyncio.sleep(0.01)
        assert handle.done()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_turn_reviews_disabled_by_default():
    """Without ``turn_reviews_enabled`` (one-shot acts, task runs), a
    response notification never schedules a review."""
    notifications: asyncio.Queue[dict] = asyncio.Queue()
    result_future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
    messages = [
        {"role": "user", "content": "do work"},
        {"role": "tool", "content": "worked"},
    ]
    inner = _make_inner_handle(notifications, result_future, messages)

    with (
        patch(
            "unify.actor.code_act_actor._start_storage_check_loop",
        ) as mock_loop,
        patch(
            "unify.actor.code_act_actor.publish_manager_method_event",
            new_callable=AsyncMock,
        ),
    ):
        mock_loop.return_value = None
        handle = _StorageCheckHandle(inner=inner, actor=_mock_actor())

        await notifications.put({"type": "response", "content": "done"})
        await asyncio.sleep(0.2)
        assert handle._turn_review_task is None

        result_future.set_result("done")
        for _ in range(300):
            if handle.done():
                break
            await asyncio.sleep(0.01)
        assert handle.done()
        # Phase 2 (the ordinary post-run review) still ran once.
        assert mock_loop.call_count == 1


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_turn_boundaries_coalesce_while_review_in_flight():
    """A boundary that arrives mid-review coalesces into one re-run against
    the then-current trajectory — reviews never run concurrently."""
    notifications: asyncio.Queue[dict] = asyncio.Queue()
    result_future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
    messages = [
        {"role": "user", "content": "week 1"},
        {"role": "tool", "content": "filed 1"},
    ]
    inner = _make_inner_handle(notifications, result_future, messages)

    release_first = asyncio.Event()
    call_count = 0

    def _make_review(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        review = MagicMock()
        if call_count == 1:

            async def _result():
                await release_first.wait()
                return "first summary"

        else:

            async def _result():
                return "second summary"

        review.result = _result
        return review

    with (
        patch(
            "unify.actor.code_act_actor._start_storage_check_loop",
            side_effect=_make_review,
        ) as mock_loop,
        patch(
            "unify.actor.code_act_actor.publish_manager_method_event",
            new_callable=AsyncMock,
        ),
    ):
        handle = _StorageCheckHandle(
            inner=inner,
            actor=_mock_actor(),
            turn_reviews_enabled=True,
        )

        await notifications.put({"type": "response", "content": "filed week 1"})

        for _ in range(200):
            if mock_loop.call_count >= 1:
                break
            await asyncio.sleep(0.01)
        assert mock_loop.call_count == 1

        # Second turn completes while the first review is still running.
        messages.append({"role": "tool", "content": "filed 2"})
        await notifications.put({"type": "response", "content": "filed week 2"})
        await asyncio.sleep(0.1)
        assert mock_loop.call_count == 1, "reviews must not run concurrently"

        release_first.set()

        async def _task_done() -> bool:
            task = handle._turn_review_task
            return task is not None and task.done()

        for _ in range(300):
            if await _task_done():
                break
            await asyncio.sleep(0.01)
        assert await _task_done()
        assert mock_loop.call_count == 2, "pending boundary re-runs exactly once"

        result_future.set_result("done")
        for _ in range(300):
            if handle.done():
                break
            await asyncio.sleep(0.01)
        assert handle.done()


def test_turn_reviews_are_opt_in():
    """A persistent session distils once, at its end; per-turn reviews are a setting."""
    from unify.settings import SETTINGS

    assert SETTINGS.UNIFY_TURN_STORAGE_REVIEWS is False


def test_session_end_review_frames_the_whole_session():
    """Ending a session with ``SESSION_ENDED`` reviews it as finished work, not as an
    interrupted one."""
    from unify.actor.code_act_actor import SESSION_ENDED, _start_storage_check_loop

    prompts: list[str] = []
    client = MagicMock()
    client.set_system_message = lambda text: prompts.append(text)
    with (
        patch("unify.actor.code_act_actor.new_llm_client", return_value=client),
        patch(
            "unify.actor.code_act_actor.start_async_tool_loop",
            return_value=MagicMock(),
        ),
        patch(
            "unify.actor.code_act_actor._build_storage_tools",
            return_value={},
        ),
    ):
        for reason in (SESSION_ENDED, "user cancelled"):
            _start_storage_check_loop(
                trajectory=[
                    {"role": "user", "content": "do work"},
                    {"role": "tool", "content": "worked"},
                ],
                actor=_mock_actor(),
                original_result="done",
                stop_reason=reason,
            )
    ended, cancelled = prompts
    assert "Session End" in ended and "explicitly stopped" not in ended
    assert "explicitly stopped" in cancelled and "user cancelled" in cancelled
