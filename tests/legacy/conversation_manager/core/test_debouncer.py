"""
Tests for the Debouncer utility.

The Debouncer manages async task execution with queue-of-2 debouncing:
- Pending task replacement (debouncing)
- Running tasks always complete; only the pending slot is replaced
- At most 1 running + 1 pending

These tests verify the Debouncer's behavior independent of the
ConversationManager integration.

NOTE: Debouncer tests are inherently timing-sensitive since the debouncer
manages task timing. Some fixed sleeps are intentional to test the debounce
behavior (e.g., simulating LLM thinking time). Where possible, we use
event-based synchronization for task coordination.
"""

import asyncio
import time as _time

import pytest


async def _wait_for_condition(
    predicate,
    *,
    timeout: float = 5.0,
    poll: float = 0.02,
) -> bool:
    """Poll predicate() until True or timeout. Returns whether condition was met."""
    start = _time.perf_counter()
    while _time.perf_counter() - start < timeout:
        if predicate():
            return True
        await asyncio.sleep(poll)
    return False


class TestDebouncerQueueOfTwo:
    """Tests that the running task is never cancelled by new submissions."""

    @pytest.mark.asyncio
    async def test_preserves_running_task(self):
        """
        Verify that the running task completes when a new submission arrives.

        New submissions replace the pending task, not the running one.
        """
        from unify.legacy.conversation_manager.domains.utils import Debouncer

        debouncer = Debouncer()

        execution_log = []
        task1_started = asyncio.Event()
        task1_can_complete = asyncio.Event()

        async def task1():
            """First task - signals when started, waits for permission to complete."""
            execution_log.append("task1:started")
            task1_started.set()
            try:
                await asyncio.wait_for(task1_can_complete.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                execution_log.append("task1:timeout")
                return
            except asyncio.CancelledError:
                execution_log.append("task1:cancelled")
                raise
            execution_log.append("task1:completed")

        async def task2():
            """Second task."""
            execution_log.append("task2:started")
            await asyncio.sleep(0.1)
            execution_log.append("task2:completed")

        # Submit first task
        await debouncer.submit(task1)

        # Wait for first task to start
        await asyncio.wait_for(task1_started.wait(), timeout=2.0)

        # Second submission must not cancel task1
        await debouncer.submit(task2)

        # Wait for task2 to be queued (pending)
        await _wait_for_condition(lambda: debouncer.pending_task is not None)

        # Now allow task1 to complete
        task1_can_complete.set()

        # Wait for both tasks to complete (poll instead of fixed sleep)
        await _wait_for_condition(lambda: "task1:completed" in execution_log)
        await _wait_for_condition(lambda: "task2:completed" in execution_log)

        # KEY ASSERTION: Task 1 should have COMPLETED, not been cancelled
        assert "task1:completed" in execution_log, (
            f"Task 1 should have completed!\n"
            f"  Execution log: {execution_log}\n"
            f"\n"
            f"The running task should be allowed to complete.\n"
            f"Only the pending task should be replaced."
        )

        assert "task1:cancelled" not in execution_log, (
            f"Task 1 was cancelled by a newer submission!\n"
            f"  Execution log: {execution_log}"
        )


class TestDebouncerCancellationPropagation:
    """
    Tests for cancellation propagation behavior.

    In Python 3.11+, cancelling a task that is awaiting another task will
    also cancel the inner task. The Debouncer must use asyncio.shield()
    to protect running tasks from this propagation when a pending task
    is cancelled (debounced).
    """

    @pytest.mark.asyncio
    async def test_pending_cancellation_does_not_cancel_running_task(self):
        """
        REGRESSION TEST: Cancelling a pending task must NOT cancel the running task.

        This tests the asyncio.shield() fix for Python 3.11+ behavior where
        cancelling a task that awaits another task propagates the cancellation.

        Scenario:
        1. Task A starts running
        2. Task B is submitted (pending, waiting for A)
        3. Task C is submitted (cancels pending B)
        4. Task A must continue and complete (not be cancelled by B's cancellation)

        Before fix: Task A would be cancelled when Task B was cancelled
        After fix: Task A completes normally, Task C runs after
        """
        from unify.legacy.conversation_manager.domains.utils import Debouncer

        debouncer = Debouncer()
        results = []

        async def slow_task(task_id):
            """A task that takes 0.3s to complete."""
            try:
                results.append(f"task{task_id}:started")
                await asyncio.sleep(0.3)
                results.append(f"task{task_id}:completed")
            except asyncio.CancelledError:
                results.append(f"task{task_id}:cancelled")
                raise

        # Submit task 0 - it starts running
        await debouncer.submit(slow_task, args=(0,))
        # Wait for task 0 to start
        await _wait_for_condition(lambda: "task0:started" in results)

        # Submit task 1 - becomes pending, waiting for task 0
        await debouncer.submit(slow_task, args=(1,))
        # Wait for task 1 to be queued as pending
        await _wait_for_condition(lambda: debouncer.pending_task is not None)

        # Submit task 2 - this cancels pending task 1 (debounce)
        # The bug: task 1's cancellation would propagate to task 0
        await debouncer.submit(slow_task, args=(2,))

        # Wait for task 0 and task 2 to complete (poll instead of fixed sleep)
        await _wait_for_condition(
            lambda: "task0:completed" in results and "task2:completed" in results,
        )

        # Task 0 MUST have completed (not cancelled)
        assert "task0:completed" in results, (
            f"Task 0 should have completed!\n"
            f"  Results: {results}\n"
            f"\n"
            f"This regression indicates the asyncio.shield() fix is missing.\n"
            f"In Python 3.11+, cancelling a pending task that awaits the running\n"
            f"task will propagate the cancellation unless shield() is used."
        )

        assert "task0:cancelled" not in results, (
            f"Task 0 was cancelled by pending task cancellation!\n"
            f"  Results: {results}\n"
            f"\n"
            f"The Debouncer must use asyncio.shield() to protect the running\n"
            f"task from cancellation propagation when pending tasks are cancelled."
        )

        # Task 2 should also complete (it was the final pending task)
        assert "task2:completed" in results, (
            f"Task 2 should have completed!\n" f"  Results: {results}"
        )

    @pytest.mark.asyncio
    async def test_rapid_submissions_preserve_running_task(self):
        """
        Test rapid submissions don't cause unexpected cancellations.

        Simulates the voice mode scenario where rapid user utterances
        trigger many submissions in quick succession.
        """
        from unify.legacy.conversation_manager.domains.utils import Debouncer

        debouncer = Debouncer()
        results = []

        async def task(task_id):
            try:
                results.append(f"task{task_id}:started")
                await asyncio.sleep(0.5)  # Simulates LLM thinking time
                results.append(f"task{task_id}:completed")
            except asyncio.CancelledError:
                results.append(f"task{task_id}:cancelled")
                raise

        # Rapid submissions (faster than task completion time)
        # NOTE: The 0.1s delay between submissions is intentional - it simulates
        # rapid user utterances in voice mode (faster than LLM thinking time)
        for i in range(5):
            await debouncer.submit(task, args=(i,))
            if i < 4:
                await asyncio.sleep(
                    0.1,
                )  # 0.1s between submissions (intentional timing)

        # Wait for final tasks to complete (poll instead of fixed 2s sleep)
        # Queue-of-2: task0 runs to completion, then task4 runs
        await _wait_for_condition(
            lambda: sum(1 for r in results if "completed" in r) >= 2,
            timeout=5.0,
        )

        # Count results
        completed = sum(1 for r in results if "completed" in r)
        cancelled = sum(1 for r in results if "cancelled" in r)

        # Queue-of-2 with shielding:
        # - Task 0 completes (first running task)
        # - Task 4 completes (final pending task after task 0 finishes)
        # - No tasks should be cancelled
        assert completed >= 2, (
            f"Expected at least 2 completions (first + final task)\n"
            f"  Completed: {completed}, Cancelled: {cancelled}\n"
            f"  Results: {results}"
        )

        assert cancelled == 0, (
            f"No tasks should be cancelled during rapid submissions!\n"
            f"  Completed: {completed}, Cancelled: {cancelled}\n"
            f"  Results: {results}"
        )


class TestDebouncerPendingReplacement:
    """Any new submit replaces the pending slot; origin does not matter."""

    @pytest.mark.asyncio
    async def test_non_user_can_replace_pending_user(self):
        """An actor/system submit replaces a pending user utterance."""
        from unify.legacy.conversation_manager.domains.utils import Debouncer

        debouncer = Debouncer(name="TestCM")

        results = []
        task0_started = asyncio.Event()

        async def slow_task(label):
            if label == "running":
                task0_started.set()
            try:
                results.append(f"{label}:started")
                await asyncio.sleep(0.5)
                results.append(f"{label}:completed")
            except asyncio.CancelledError:
                results.append(f"{label}:cancelled")
                raise

        await debouncer.submit(slow_task, args=("running",))
        await asyncio.wait_for(task0_started.wait(), timeout=2.0)

        await debouncer.submit(slow_task, args=("user",))
        await _wait_for_condition(lambda: debouncer.pending_task is not None)

        await debouncer.submit(slow_task, args=("actor",))

        await _wait_for_condition(
            lambda: "actor:completed" in results,
            timeout=5.0,
        )

        assert "actor:completed" in results, (
            f"Actor task should have replaced the pending user task.\n"
            f"  Results: {results}"
        )
        assert "user:started" not in results, (
            f"User task should have been replaced by actor task.\n"
            f"  Results: {results}"
        )

    @pytest.mark.asyncio
    async def test_user_can_replace_pending_non_user(self):
        """A user submission replaces a pending non-user submission."""
        from unify.legacy.conversation_manager.domains.utils import Debouncer

        debouncer = Debouncer()

        results = []
        task0_started = asyncio.Event()

        async def slow_task(label):
            if label == "running":
                task0_started.set()
            try:
                results.append(f"{label}:started")
                await asyncio.sleep(0.3)
                results.append(f"{label}:completed")
            except asyncio.CancelledError:
                results.append(f"{label}:cancelled")
                raise

        await debouncer.submit(slow_task, args=("running",))
        await asyncio.wait_for(task0_started.wait(), timeout=2.0)

        await debouncer.submit(slow_task, args=("actor",))
        await debouncer.submit(slow_task, args=("user",))

        await _wait_for_condition(
            lambda: "user:completed" in results,
            timeout=5.0,
        )

        assert "user:completed" in results
        assert "actor:started" not in results, (
            f"Actor task should have been replaced by user task.\n"
            f"  Results: {results}"
        )

    @pytest.mark.asyncio
    async def test_newer_submit_replaces_older_pending(self):
        """A newer submission replaces an older pending submission."""
        from unify.legacy.conversation_manager.domains.utils import Debouncer

        debouncer = Debouncer()

        results = []
        task0_started = asyncio.Event()

        async def slow_task(label):
            if label == "running":
                task0_started.set()
            try:
                results.append(f"{label}:started")
                await asyncio.sleep(0.3)
                results.append(f"{label}:completed")
            except asyncio.CancelledError:
                results.append(f"{label}:cancelled")
                raise

        await debouncer.submit(slow_task, args=("running",))
        await asyncio.wait_for(task0_started.wait(), timeout=2.0)

        await debouncer.submit(slow_task, args=("first",))
        await debouncer.submit(slow_task, args=("second",))

        await _wait_for_condition(
            lambda: "second:completed" in results,
            timeout=5.0,
        )

        assert "second:completed" in results
        assert "first:started" not in results, (
            f"Older pending task should have been replaced by newer submit.\n"
            f"  Results: {results}"
        )


class TestDebouncerCancelByTurn:
    """``cancel_run_by_turn`` cancels exactly the run a given turn spawned,
    wherever it sits, and no-ops otherwise. This is what lets the fast brain
    drop its own turn's slow-brain run without harming unrelated runs."""

    @staticmethod
    def _meta(turn_id, **extra):
        return {"turn_id": turn_id, **extra}

    @pytest.mark.asyncio
    async def test_cancels_matching_running(self):
        from unify.legacy.conversation_manager.domains.utils import Debouncer

        debouncer = Debouncer()
        log = []
        started = asyncio.Event()

        async def run():
            log.append("started")
            started.set()
            try:
                await asyncio.sleep(5.0)
                log.append("completed")
            except asyncio.CancelledError:
                log.append("cancelled")
                raise

        await debouncer.submit(run, trace_meta=self._meta(1))
        await asyncio.wait_for(started.wait(), timeout=2.0)

        assert await debouncer.cancel_run_by_turn(1) is True
        assert "cancelled" in log and "completed" not in log

    @pytest.mark.asyncio
    async def test_unknown_turn_is_noop(self):
        from unify.legacy.conversation_manager.domains.utils import Debouncer

        debouncer = Debouncer()
        log = []
        started = asyncio.Event()

        async def run():
            log.append("started")
            started.set()
            await asyncio.sleep(0.3)
            log.append("completed")

        await debouncer.submit(run, trace_meta=self._meta(1))
        await asyncio.wait_for(started.wait(), timeout=2.0)

        # Different turn id -> nothing cancelled; the run completes.
        assert await debouncer.cancel_run_by_turn(999) is False
        await _wait_for_condition(lambda: "completed" in log)
        assert "completed" in log

    @pytest.mark.asyncio
    async def test_none_turn_is_noop(self):
        from unify.legacy.conversation_manager.domains.utils import Debouncer

        debouncer = Debouncer()
        started = asyncio.Event()

        async def run():
            started.set()
            await asyncio.sleep(0.3)

        await debouncer.submit(run, trace_meta=self._meta(None))
        await asyncio.wait_for(started.wait(), timeout=2.0)
        assert await debouncer.cancel_run_by_turn(None) is False

    @pytest.mark.asyncio
    async def test_tool_committed_running_is_spared(self):
        from unify.legacy.conversation_manager.domains.utils import Debouncer

        debouncer = Debouncer()
        log = []
        started = asyncio.Event()

        async def run():
            log.append("started")
            started.set()
            await asyncio.sleep(0.3)
            log.append("completed")

        await debouncer.submit(
            run,
            trace_meta=self._meta(1, tool_commit_started="true"),
        )
        await asyncio.wait_for(started.wait(), timeout=2.0)

        # Already speaking -> spared even though the turn id matches.
        assert await debouncer.cancel_run_by_turn(1) is False
        await _wait_for_condition(lambda: "completed" in log)
        assert "completed" in log

    @pytest.mark.asyncio
    async def test_cancels_matching_pending_leaving_running(self):
        from unify.legacy.conversation_manager.domains.utils import Debouncer

        debouncer = Debouncer()
        log = []
        run1_started = asyncio.Event()

        async def run1():
            log.append("run1:started")
            run1_started.set()
            await asyncio.sleep(0.5)
            log.append("run1:completed")

        async def run2():
            log.append("run2:started")

        # run1 (turn 1) running; run2 (turn 2) queued behind it.
        await debouncer.submit(run1, trace_meta=self._meta(1))
        await asyncio.wait_for(run1_started.wait(), timeout=2.0)
        await debouncer.submit(run2, trace_meta=self._meta(2))

        # Cancel the PENDING turn 2 -> run1 keeps going and completes; run2 never runs.
        assert await debouncer.cancel_run_by_turn(2) is True
        await _wait_for_condition(lambda: "run1:completed" in log)
        assert "run1:completed" in log
        assert "run2:started" not in log

    @pytest.mark.asyncio
    async def test_cancels_matching_running_promotes_pending(self):
        from unify.legacy.conversation_manager.domains.utils import Debouncer

        debouncer = Debouncer()
        log = []
        run1_started = asyncio.Event()

        async def run1():
            log.append("run1:started")
            run1_started.set()
            try:
                await asyncio.sleep(5.0)
                log.append("run1:completed")
            except asyncio.CancelledError:
                log.append("run1:cancelled")
                raise

        async def run2():
            log.append("run2:started")

        # run1 (turn 1) running; run2 (turn 2) queued behind it.
        await debouncer.submit(run1, trace_meta=self._meta(1))
        await asyncio.wait_for(run1_started.wait(), timeout=2.0)
        await debouncer.submit(run2, trace_meta=self._meta(2))

        # Cancel the RUNNING turn 1 -> pending turn 2 auto-promotes and runs.
        assert await debouncer.cancel_run_by_turn(1) is True
        await _wait_for_condition(lambda: "run2:started" in log)
        assert "run1:cancelled" in log
        assert "run2:started" in log
        assert "run1:completed" not in log
