"""
tests/conversation_manager/core/test_explicit_retirement.py
===========================================================

An explicit shutdown is a retirement.

When the process is told to retire (``stop_async``, a benchmark scenario
ending, a sandbox exiting), the session must retire in seconds: run the
``_request_shutdown`` sequence, discard in-flight actions rather than waiting
on them, and return. A fresh boot over the same durable world — in the same
process — must not block on anything the previous instance left behind: not
a parked persist session, not a lock a frozen task still holds on a dead
event loop.

These tests cover the "asked to shut down" path plus the reboot shape the
colleague benchmark runs (teardown-plus-boot between weekly scenarios), where
a retirement that waits on parked work stalls for whatever outer timeout
happens to fire — 600s step ceilings, 900s drains — while a rebooted week
sits idle.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import pytest


def _make_cm(event_broker, stop_event):
    from unify.legacy.conversation_manager.conversation_manager import (
        ConversationManager,
    )

    return ConversationManager(event_broker, stop_event)


@pytest.fixture
def event_broker():
    from unify.legacy.conversation_manager.in_memory_event_broker import (
        create_in_memory_event_broker,
        reset_in_memory_event_broker,
    )

    reset_in_memory_event_broker()
    broker = create_in_memory_event_broker()
    yield broker
    reset_in_memory_event_broker()


class _PromptHandle:
    """A real-actor-shaped handle that stops when asked."""

    def __init__(self):
        self.stop_reason = None

    async def stop(self, reason=None, **kwargs) -> None:
        self.stop_reason = reason


class _DeafHandle:
    """A handle whose stop never returns — a parked session mid-hung-call."""

    async def stop(self, reason=None, **kwargs) -> None:
        await asyncio.Event().wait()


class _SimulatedHandle:
    """A simulated-actor-shaped handle: completion is triggered, not awaited."""

    def __init__(self):
        self.completed = False

    def trigger_completion(self) -> None:
        self.completed = True


class TestExplicitStopIsARetirement:
    @pytest.mark.asyncio
    async def test_stop_async_runs_the_retirement_sequence(
        self,
        event_broker,
    ):
        """stop_async retires through _request_shutdown, in seconds.

        Reason recorded, session_end path taken, stop set, broker closed —
        and control returns promptly even with a parked action in flight.
        """
        import unify.legacy.conversation_manager as cm_mod

        stop_event = asyncio.Event()
        cm = _make_cm(event_broker, stop_event)
        parked = _PromptHandle()
        cm.in_flight_actions[1] = {"handle": parked, "handle_actions": []}

        cm_mod._conversation_manager = cm
        started = time.monotonic()
        try:
            await asyncio.wait_for(
                cm_mod.stop_async(reason="scenario end"),
                timeout=30.0,
            )
        finally:
            cm_mod.reset()
        elapsed = time.monotonic() - started

        assert elapsed < 20.0, f"explicit retirement took {elapsed:.1f}s"
        assert cm.shutdown_reason == "scenario end"
        assert stop_event.is_set()
        assert event_broker._closed, "retirement closes the broker"
        assert parked.stop_reason == "session retired"
        assert cm.in_flight_actions == {}
        assert cm.completed_actions == {}

    @pytest.mark.asyncio
    async def test_an_internal_exit_keeps_its_recorded_reason(self, event_broker):
        """A stop_async after an internal exit already decided must not relabel it."""
        import unify.legacy.conversation_manager as cm_mod

        stop_event = asyncio.Event()
        cm = _make_cm(event_broker, stop_event)
        await cm._request_shutdown("internal_exit", "Internal exit requested")

        cm_mod._conversation_manager = cm
        try:
            await cm_mod.stop_async(reason="cleanup")
        finally:
            cm_mod.reset()

        assert cm.shutdown_reason == "internal_exit"

    @pytest.mark.asyncio
    async def test_cleanup_abandons_a_handle_that_ignores_stop(self, event_broker):
        """A parked session that cannot stop is discarded, not waited on.

        The in-process retirement must be no slower because a handle is deaf.
        The grace period is 5s, so the whole cleanup stays bounded in seconds.
        """
        stop_event = asyncio.Event()
        cm = _make_cm(event_broker, stop_event)
        cm.in_flight_actions[1] = {"handle": _DeafHandle(), "handle_actions": []}
        cm.in_flight_actions[2] = {"handle": _SimulatedHandle(), "handle_actions": []}
        simulated = cm.in_flight_actions[2]["handle"]

        started = time.monotonic()
        await asyncio.wait_for(cm.cleanup(), timeout=30.0)
        elapsed = time.monotonic() - started

        assert elapsed < 20.0, f"cleanup took {elapsed:.1f}s with a deaf handle"
        assert simulated.completed
        assert cm.in_flight_actions == {}
        assert cm.completed_actions == {}


class TestEventBusSurvivesAnInProcessReboot:
    """The process-global bus must not chain a successor to a dead loop."""

    @pytest.mark.asyncio
    async def test_callbacks_parked_on_a_dead_loop_are_not_awaited(self):
        """A successor's join must not wait on a predecessor's frozen callbacks.

        A callback task on a loop that died mid-run never completes, and its
        done-callback never removes it from the bus. A successor that awaited
        it would block until some outer timeout fired (600s step ceilings —
        the "idle-waits in ten-minute quanta").
        """
        from unify.events.event_bus import EventBus

        bus = EventBus()
        dead = asyncio.new_event_loop()
        parked = dead.create_future()
        dead.close()
        setattr(parked, "_eb_seq", 1)
        setattr(parked, "_eb_root_seq", 1)
        bus._callback_seq = 1
        bus._callback_futures.add(parked)

        await asyncio.wait_for(bus.ajoin_callbacks(), timeout=1.0)

        assert parked not in bus._callback_futures

    @pytest.mark.asyncio
    async def test_callbacks_on_a_live_loop_are_kept(self):
        """Pruning is for dead loops only; a running loop keeps its callbacks."""
        import threading

        from unify.events.event_bus import EventBus

        bus = EventBus()
        owner = asyncio.new_event_loop()
        thread = threading.Thread(target=owner.run_forever, daemon=True)
        thread.start()
        running = threading.Event()
        owner.call_soon_threadsafe(running.set)
        assert running.wait(timeout=5), "owner loop failed to start"
        try:
            live = owner.create_future()
            setattr(live, "_eb_seq", 1)
            setattr(live, "_eb_root_seq", 1)
            bus._callback_seq = 1
            bus._callback_futures.add(live)

            # This loop is not the owner, so the join has nothing of its own
            # to wait for and must return at once — without dropping the
            # owner's live callback.
            await asyncio.wait_for(bus.ajoin_callbacks(), timeout=1.0)

            assert live in bus._callback_futures
        finally:
            owner.call_soon_threadsafe(owner.stop)
            thread.join(timeout=5)
            owner.close()

    @pytest.mark.asyncio
    async def test_operations_queue_bound_to_a_dead_loop_is_replaced(self):
        """A successor's operations listener must not die on the old queue.

        The module-level queue binds to the loop that first awaits it; a
        rebooted session's listener then dies on its first ``get`` with
        "bound to a different event loop" — silently, after logging that it
        started — and every queued operation (EventBus persistence among
        them) accumulates unprocessed forever.
        """
        from unify.legacy.conversation_manager.domains import managers_utils

        old_queue = asyncio.Queue()
        dead = asyncio.new_event_loop()

        async def bind_and_abandon():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(old_queue.get(), timeout=0.01)

        await asyncio.to_thread(dead.run_until_complete, bind_and_abandon())
        dead.close()

        original_queue = managers_utils._operations_queue
        original_lock = managers_utils._init_lock
        original_loop = managers_utils._module_loop
        managers_utils._operations_queue = old_queue
        managers_utils._module_loop = dead
        try:
            await managers_utils.queue_operation(asyncio.sleep, 0)

            assert managers_utils._operations_queue is not old_queue
            assert managers_utils._module_loop is asyncio.get_running_loop()
            # The new queue accepts and serves work on this loop.
            item = await asyncio.wait_for(
                managers_utils._operations_queue.get(),
                timeout=1.0,
            )
            assert item[0] is asyncio.sleep
        finally:
            managers_utils._operations_queue = original_queue
            managers_utils._init_lock = original_lock
            managers_utils._module_loop = original_loop


class TestRebootOverTheSameWorld:
    @pytest.mark.asyncio
    async def test_teardown_plus_boot_completes_in_seconds(self):
        """Boot, retire explicitly, boot again — the benchmark's weekly shape.

        The whole point of the retirement contract: the second boot over the
        same durable world proceeds promptly, with nothing inherited from the
        first session — no open broker, no in-flight registry.
        """
        from unify.legacy.conversation_manager import (
            get_conversation_manager,
            start_async,
            stop_async,
        )
        from unify.legacy.conversation_manager.event_broker import reset_event_broker

        reset_event_broker()
        cm1 = await start_async()
        cm1.in_flight_actions[1] = {"handle": _PromptHandle(), "handle_actions": []}

        started = time.monotonic()
        await asyncio.wait_for(stop_async(reason="scenario end"), timeout=60.0)
        # Deliberately no reset_event_broker() here: retirement itself must
        # leave nothing for a successor to trip over.
        cm2 = await asyncio.wait_for(
            start_async(),
            timeout=60.0,
        )
        elapsed = time.monotonic() - started

        try:
            assert cm2 is not cm1
            assert get_conversation_manager() is cm2
            assert elapsed < 45.0, (
                f"teardown+boot took {elapsed:.1f}s — the reboot shape must "
                "complete in seconds, not outer-timeout quanta"
            )
            assert cm1.shutdown_reason == "scenario end"
            assert cm1.stop.is_set()
            assert cm1.in_flight_actions == {}
            assert not cm2.stop.is_set()
        finally:
            await stop_async(reason="test cleanup")
            reset_event_broker()
