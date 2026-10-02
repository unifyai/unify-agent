"""Cancellation ownership at native handle hand-off boundaries.

Uses a real client without generation and event-triggered native scheduling.
"""

import asyncio

import pytest

from unify.common.async_tool_loop import SteerableToolHandle
from unify.common.llm_client import new_llm_client
from unify.common._async_tool import message_dispatcher as dispatch_module
from unify.common._async_tool.context_tracker import LoopContextState
from unify.common._async_tool.loop import (
    LoopLogger,
    _LoopToolFailureTracker,
    ToolLoopRuntimeState,
)
from unify.common._async_tool.loop_config import LoopConfig
from unify.common._async_tool.message_dispatcher import LoopMessageDispatcher
from unify.common._async_tool.messages import ensure_placeholders_for_pending
from unify.common._async_tool.timeout_timer import TimeoutTimer
from unify.common._async_tool.tools_data import ToolsData


class OwnedHandle(SteerableToolHandle):
    def __init__(self):
        self.stops = 0
        self.started = asyncio.Event()
        self.finished = asyncio.Event()
        self.result_task = None
        self.result_tasks = []

    async def result(self):
        self.result_task = asyncio.current_task()
        self.result_tasks.append(self.result_task)
        self.started.set()
        await self.finished.wait()
        return "done"

    def stop(self, reason=None, **kwargs):
        self.stops += 1
        self.finished.set()

    def done(self):
        return self.finished.is_set()

    async def ask(self, question, **kwargs):
        return "running"

    def interject(self, message, **kwargs):
        pass

    def pause(self, **kwargs):
        pass

    def resume(self, **kwargs):
        pass

    async def next_clarification(self):
        return {}

    async def next_notification(self):
        return {}

    async def answer_clarification(self, cid, answer):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["returned", "draining", "adopting", "adopted"])
@pytest.mark.parametrize("composite", [False, True, "aliased", "stop_error"])
async def test_cancellation_stops_returned_handle_at_each_boundary(
    monkeypatch,
    boundary,
    composite,
):
    child = OwnedHandle()
    sibling = OwnedHandle()
    children = [child, sibling] if composite else [child]
    if composite == "stop_error":

        def failing_stop(reason=None, **kwargs):
            child.stops += 1
            child.finished.set()
            raise ValueError("owned stop failure")

        child.stop = failing_stop

    async def make_handle():
        """Return a live owned operation supporting cancellation."""
        if composite == "aliased":
            return {"first": child, "alias": child, "second": sibling}
        return {"first": child, "second": sibling} if composite else child

    client = new_llm_client("openai/gpt-6-luna@openrouter", stateful=True)
    cfg = LoopConfig("cancellation-ownership", None, [])
    dispatcher = LoopMessageDispatcher(
        client,
        cfg,
        TimeoutTimer(None, None, False, client),
    )
    tools = ToolsData(
        {"make_handle": make_handle},
        client=client,
        logger=LoopLogger(cfg, False),
    )
    meta = {}
    client.append_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "owned",
                        "type": "function",
                        "function": {
                            "name": "make_handle",
                            "arguments": "{}",
                        },
                    },
                ],
            },
        ],
    )
    await tools.schedule_base_tool_call(
        client.messages[-1],
        name="make_handle",
        args_json="{}",
        call_id="owned",
        call_idx=0,
        context_state=LoopContextState(),
        propagate_chat_context=False,
        assistant_meta=meta,
        msg_dispatcher=dispatcher,
    )
    factory_task = next(iter(tools.pending))
    await factory_task
    await ensure_placeholders_for_pending(
        tools_data=tools,
        assistant_meta=meta,
        client=client,
        msg_dispatcher=dispatcher,
    )
    client._sent_watermark = len(client.messages)
    publication = asyncio.Event()
    release = asyncio.Event()
    processing = None

    async def held_publication(*args, **kwargs):
        publication.set()
        await release.wait()

    try:
        if boundary != "returned":
            if boundary in ("draining", "adopting"):
                monkeypatch.setattr(dispatch_module, "to_event_bus", held_publication)
            if boundary == "draining":
                tools.info[factory_task].notification_queue = asyncio.Queue()
                tools.info[factory_task].notification_queue.put_nowait(
                    {"message": "completed factory"},
                )
            processing = asyncio.create_task(
                tools.process_completed_task(
                    factory_task,
                    _LoopToolFailureTracker(5, ToolLoopRuntimeState()),
                    None,
                    meta,
                    dispatcher,
                ),
            )
            if boundary in ("draining", "adopting"):
                await asyncio.wait_for(publication.wait(), 5)
                if boundary == "adopting":
                    await asyncio.wait_for(child.started.wait(), 5)
                processing.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await processing
            else:
                await processing
                await asyncio.wait_for(child.started.wait(), 5)
        await tools.cancel_pending_tasks()

        for item in children:
            assert item.stops == 1, "returned handle lost its cancellation owner"
            assert item.finished.is_set()
            assert all(
                task.done() for task in item.result_tasks
            ), "nested result task survived parent cleanup"
    finally:
        release.set()
        for item in children:
            item.finished.set()
        if processing is not None and not processing.done():
            processing.cancel()
            await asyncio.gather(processing, return_exceptions=True)
        for item in children:
            for task in item.result_tasks:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        await tools.cancel_pending_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "cyclic",
        "cancel_return",
        "stop_cancel",
        "pydantic",
        "container_accessor",
        "class_accessor",
    ],
)
async def test_cleanup_handles_adversarial_return_and_stop(case):
    children = [OwnedHandle(), OwnedHandle()]
    started = asyncio.Event()
    release = asyncio.Event()

    async def operation():
        """Return supported owned resources, including cancellation hand-off."""
        started.set()
        if case == "cancel_return":
            try:
                await release.wait()
            except asyncio.CancelledError:
                return children
        if case == "cyclic":
            value = [children]
            value.append(value)
            return value
        if case == "pydantic":
            from typing import Any
            from pydantic import BaseModel

            class StoredResources(BaseModel):
                resources: Any

                def __getattribute__(self, name):
                    if name == "resources":
                        raise AssertionError(
                            "computed accessor must not run during cleanup",
                        )
                    return super().__getattribute__(name)

            return StoredResources(resources=children)
        if case == "container_accessor":

            class StoredList(list):
                def __iter__(self):
                    raise AssertionError("custom iteration must not run during cleanup")

            return StoredList(children)
        if case == "class_accessor":

            class UnsupportedLeaf:
                @property
                def __class__(self):
                    raise AssertionError(
                        "virtual class accessor must not run during cleanup",
                    )

            return [children, UnsupportedLeaf()]
        return children

    if case == "stop_cancel":

        async def stop_cancelled(reason=None, **kwargs):
            children[0].stops += 1
            children[0].finished.set()
            raise asyncio.CancelledError("child stop cancelled")

        children[0].stop = stop_cancelled

    client = new_llm_client("openai/gpt-6-luna@openrouter", stateful=True)
    cfg = LoopConfig("cancellation-adversary", None, [])
    tools = ToolsData(
        {"operation": operation},
        client=client,
        logger=LoopLogger(cfg, False),
    )
    client.append_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "owned",
                        "type": "function",
                        "function": {
                            "name": "operation",
                            "arguments": "{}",
                        },
                    },
                ],
            },
        ],
    )
    await tools.schedule_base_tool_call(
        client.messages[-1],
        name="operation",
        args_json="{}",
        call_id="owned",
        call_idx=0,
        context_state=LoopContextState(),
        propagate_chat_context=False,
        assistant_meta={},
        msg_dispatcher=None,
    )
    factory = next(iter(tools.pending))
    await started.wait()
    if case != "cancel_return":
        await factory
    try:
        await tools.cancel_pending_tasks()
        assert [child.stops for child in children] == [1, 1]
        assert factory.done()
        assert not tools.pending
    finally:
        release.set()
        for child in children:
            child.finished.set()
        factory.cancel()
        await asyncio.gather(factory, return_exceptions=True)
