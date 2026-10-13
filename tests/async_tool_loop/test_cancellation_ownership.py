"""Ownership of the handles a tool call returns.

The loop adopts no handle a call returns (a turn's calls run to completion),
so the call's end is where such a handle stops being owned: a handle still
running when its call's result is taken is stopped then, and a call
cancelled before its result is taken has every handle it returned stopped
by ``cancel_pending_tasks``. Uses a real client without generation.
"""

import asyncio

import pytest

from unify.common.async_tool_loop import ToolLoopHandle
from unify.common.llm_client import new_llm_client
from unify.common._async_tool import tools_data as tools_data_module
from unify.common._async_tool.context_tracker import LoopContextState
from unify.common._async_tool.loop import (
    LoopLogger,
    _LoopToolFailureTracker,
    ToolLoopRuntimeState,
)
from unify.common._async_tool.loop_config import LoopConfig
from unify.common._async_tool.message_dispatcher import LoopMessageDispatcher
from unify.common._async_tool.timeout_timer import TimeoutTimer
from unify.common._async_tool.tools_data import ToolsData


class OwnedHandle(ToolLoopHandle):
    def __init__(self):
        self.stops = 0
        self.finished = asyncio.Event()

    async def result(self):
        await self.finished.wait()
        return "done"

    def stop(self, reason=None, **kwargs):
        self.stops += 1
        self.finished.set()

    def done(self):
        return self.finished.is_set()

    async def submit(self, text):
        pass


def _call_message(name: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "owned",
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            },
        ],
    }


async def _scheduled(tools_fn: dict, name: str):
    client = new_llm_client("openai/gpt-6-luna@openrouter", stateful=True)
    cfg = LoopConfig("cancellation-ownership", None, [])
    dispatcher = LoopMessageDispatcher(
        client,
        cfg,
        TimeoutTimer(None, None, False, client),
    )
    tools = ToolsData(tools_fn, client=client, logger=LoopLogger(cfg, False))
    client.append_messages([_call_message(name)])
    meta: dict = {}
    task = await tools.schedule_base_tool_call(
        client.messages[-1],
        name=name,
        args_json="{}",
        call_id="owned",
        call_idx=0,
        context_state=LoopContextState(),
        propagate_chat_context=False,
        assistant_meta=meta,
        msg_dispatcher=dispatcher,
    )
    return client, tools, task, meta, dispatcher


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["completed", "cancelled_unread"])
@pytest.mark.parametrize("composite", [False, True, "aliased", "stop_error"])
async def test_a_returned_handle_is_stopped_once_its_call_ends(boundary, composite):
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
        """Return a live owned operation."""
        if composite == "aliased":
            return {"first": child, "alias": child, "second": sibling}
        return {"first": child, "second": sibling} if composite else child

    client, tools, task, meta, dispatcher = await _scheduled(
        {"make_handle": make_handle},
        "make_handle",
    )
    await task
    try:
        if boundary == "completed":
            # The call's result is taken: its handles are stopped then.
            await tools.process_completed_task(
                task,
                _LoopToolFailureTracker(5, ToolLoopRuntimeState()),
                None,
                meta,
                dispatcher,
            )
            assert any(m.get("role") == "tool" for m in client.messages)
        else:
            # Cancelled before the result was taken.
            await tools.cancel_pending_tasks()
        for item in children:
            assert item.stops == 1, "returned handle lost its owner"
            assert item.finished.is_set()
        # Nothing stops them a second time.
        await tools.cancel_pending_tasks()
        assert [item.stops for item in children] == [1] * len(children)
        assert not tools.pending
    finally:
        for item in children:
            item.finished.set()


@pytest.mark.asyncio
async def test_a_handle_already_done_is_not_stopped():
    child = OwnedHandle()
    child.finished.set()

    async def make_handle():
        """Return a finished handle."""
        return child

    _client, tools, task, meta, dispatcher = await _scheduled(
        {"make_handle": make_handle},
        "make_handle",
    )
    await task
    await tools.process_completed_task(
        task,
        _LoopToolFailureTracker(5, ToolLoopRuntimeState()),
        None,
        meta,
        dispatcher,
    )
    assert child.stops == 0


@pytest.mark.asyncio
async def test_a_returned_handle_whose_stop_hangs_is_left_after_the_grace(
    monkeypatch,
):
    monkeypatch.setattr(tools_data_module, "_RETURNED_STOP_GRACE_S", 0.05)
    child = OwnedHandle()
    hang = asyncio.Event()

    async def stuck_stop(reason=None, **kwargs):
        child.stops += 1
        await hang.wait()

    child.stop = stuck_stop

    async def make_handle():
        """Return a handle whose stop never ends."""
        return child

    client, tools, task, meta, dispatcher = await _scheduled(
        {"make_handle": make_handle},
        "make_handle",
    )
    await task
    try:
        await asyncio.wait_for(
            tools.process_completed_task(
                task,
                _LoopToolFailureTracker(5, ToolLoopRuntimeState()),
                None,
                meta,
                dispatcher,
            ),
            5,
        )
        assert child.stops == 1
        assert any(m.get("role") == "tool" for m in client.messages)
        assert tools._abandoned, "the hanging stop is held until it ends"
    finally:
        hang.set()
        await asyncio.sleep(0)


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


@pytest.mark.asyncio
async def test_a_cell_waits_for_the_handles_it_spawned_returned_or_not():
    """In process, a handle a cell spawned and returned as its last
    expression is awaited with the others before the call ends: nothing
    adopts it afterwards, so nothing would own it."""
    from unify.actor.execution.session import _await_orphan_sandbox_handles

    returned, dropped = OwnedHandle(), OwnedHandle()
    waiting = asyncio.create_task(
        _await_orphan_sandbox_handles(spawned=[returned, dropped]),
    )
    await asyncio.sleep(0.05)
    assert not waiting.done()
    returned.finished.set()
    await asyncio.sleep(0.05)
    assert not waiting.done()
    dropped.finished.set()
    await asyncio.wait_for(waiting, 5)
