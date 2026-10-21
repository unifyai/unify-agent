import asyncio
import traceback

from unify.common.diagnostic_logging import staging_diagnostics_enabled
from unify.common.startup_timing import log_startup_timing
from unify.logger import LOGGER


def log_task_exc(task: asyncio.Task) -> None:
    try:
        task.result()  # re-raises if failed
    except asyncio.CancelledError:
        pass
    except Exception as e:
        if staging_diagnostics_enabled():
            task_name = task.get_name()
            trace_meta = getattr(task, "_trace_meta", {}) or {}
            LOGGER.exception(
                "Slow-brain task failed task=%s trace_meta=%s error=%s",
                task_name,
                trace_meta,
                e,
            )
            LOGGER.error(
                "Slow-brain task traceback text task=%s trace_meta=%s:\n%s",
                task_name,
                trace_meta,
                traceback.format_exc(),
            )
            return
        LOGGER.exception("Slow-brain task failed: %s", e)


class Debouncer:
    def __init__(self, name: str | None = None):
        self.running_task: asyncio.Task = None
        self.pending_task: asyncio.Task = None
        self._name = name
        self._pending_label: str = ""
        # Trace meta of the queued (pending) run, so it can be matched/cancelled
        # by turn id while still pending (before it promotes to running).
        self._pending_trace_meta: dict = {}
        self.was_queued: bool = False
        self.running_task_started_at: float = 0.0
        self.running_task_trace_meta: dict = {}

    async def submit(
        self,
        async_fn,
        args: tuple = None,
        kwargs: dict = None,
        delay=0,
        label: str = "",
        trace_meta: dict | None = None,
    ):
        args, kwargs = args or (), kwargs or {}

        had_pending = self.pending_task is not None and not self.pending_task.done()
        has_running = self.running_task is not None and not self.running_task.done()
        old_label = self._pending_label

        await self._cancel_pending()

        if self._name:
            new_type = f"{label} " if label else ""
            if had_pending and has_running:
                old_type = f" (replacing {old_label})" if old_label else ""
                LOGGER.info(
                    f"🚦 [{self._name}] {new_type}request queued{old_type}",
                )
            elif has_running:
                LOGGER.info(
                    f"🚦 [{self._name}] {new_type}request queued",
                )

        async def wait_for_running_task():
            if delay > 0:
                log_startup_timing(
                    LOGGER,
                    "⏱️ [StartupTiming] debouncer.%s sleeping delay=%.2fs label=%s",
                    self._name or "unknown",
                    delay,
                    label or "-",
                )
                await asyncio.sleep(delay)
            queued = self.running_task is not None and not self.running_task.done()
            wait_t0 = asyncio.get_event_loop().time()
            try:
                # Wait for any currently running task to complete.
                # Use asyncio.shield() to protect the running task from being
                # cancelled if THIS pending task is cancelled. In Python 3.11+,
                # cancelling a task that awaits another task will also cancel
                # the inner task - shield() prevents this propagation.
                if self.running_task and not self.running_task.done():
                    await asyncio.shield(self.running_task)
            except asyncio.CancelledError:
                # CancelledError can come from two sources:
                # 1. The running task was cancelled externally
                # 2. This pending task was cancelled (debounced by a newer submit)
                #
                # In case 1, we should proceed to create a new running task.
                # In case 2, we should NOT proceed - let the newer pending task handle it.
                if self.running_task and self.running_task.cancelled():
                    pass
                else:
                    raise
            self.was_queued = queued
            self.running_task_started_at = asyncio.get_event_loop().time()
            self.running_task_trace_meta = trace_meta or {}
            log_startup_timing(
                LOGGER,
                (
                    "⏱️ [StartupTiming] debouncer.%s starting_task "
                    "label=%s queued=%s wait_for_running=%.2fs"
                ),
                self._name or "unknown",
                label or "-",
                queued,
                self.running_task_started_at - wait_t0,
            )
            self.running_task = asyncio.create_task(async_fn(*args, **kwargs))
            if label:
                self.running_task.set_name(f"{self._name or 'Debouncer'}:{label}")
            self.running_task._trace_meta = dict(self.running_task_trace_meta)
            self.running_task.add_done_callback(log_task_exc)
            self.pending_task = None
            self._pending_label = ""
            self._pending_trace_meta = {}

        self.pending_task = asyncio.create_task(wait_for_running_task())
        self._pending_label = label
        self._pending_trace_meta = trace_meta or {}

    async def _cancel_pending(self):
        if self.pending_task and not self.pending_task.done():
            self.pending_task.cancel()
            try:
                await self.pending_task
            except asyncio.CancelledError:
                pass

    @staticmethod
    def _is_tool_committed(meta: dict) -> bool:
        return (
            str((meta or {}).get("tool_commit_started", "")).strip().lower() == "true"
        )

    async def cancel_run_by_turn(self, turn_id) -> bool:
        """Cancel exactly the run spawned by ``turn_id``, wherever it sits.

        Only that turn's slow-brain run is dropped, never a prior
        still-thinking run or an unrelated run.

        - Pending match -> cancel the pending wrapper; the running task (a
          different turn) keeps going.
        - Running match -> cancel it (unless it is already in tool commit, i.e.
          replying, in which case it is spared); the pending wrapper then
          auto-promotes to running.
        - No match (already debounced out / not ours) -> no-op.

        Returns True iff a task was cancelled. ``turn_id is None`` never matches.
        """
        if turn_id is None:
            return False

        if (
            self.pending_task is not None
            and not self.pending_task.done()
            and self._pending_trace_meta.get("turn_id") == turn_id
        ):
            self.pending_task.cancel()
            try:
                await self.pending_task
            except asyncio.CancelledError:
                pass
            self.pending_task = None
            self._pending_label = ""
            self._pending_trace_meta = {}
            if self._name:
                LOGGER.info(
                    f"🚦 [{self._name}] cancelled queued run for turn {turn_id}",
                )
            return True

        if (
            self.running_task is not None
            and not self.running_task.done()
            and self.running_task_trace_meta.get("turn_id") == turn_id
        ):
            if self._is_tool_committed(self.running_task_trace_meta):
                # Already producing its reply; leave it to finish.
                return False
            self.running_task.cancel()
            try:
                await self.running_task
            except asyncio.CancelledError:
                pass
            if self._name:
                LOGGER.info(
                    f"🚦 [{self._name}] cancelled running run for turn {turn_id}",
                )
            return True

        return False
