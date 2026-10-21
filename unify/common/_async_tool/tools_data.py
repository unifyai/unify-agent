import asyncio
import json
import traceback
import time


from typing import (
    Dict,
    Set,
    Any,
    Optional,
    TYPE_CHECKING,
)
from .tools_utils import ToolCallMetadata, create_tool_call_message
from .messages import (
    insert_tool_message_after_assistant,
    _normalise_kwargs_for_bound_method,
    is_mutable,
)
from ..tool_spec import normalise_tools
from ..llm_helpers import method_to_schema
from .formatting import serialize_tool_content, sanitize_tool_msg_for_logging
from contextlib import suppress
from ..tool_errors import ToolInputError

if TYPE_CHECKING:
    from .loop import LoopLogger, _LoopToolFailureTracker
    from .message_dispatcher import LoopMessageDispatcher


def _failure_text(exc: BaseException) -> str:
    """The text a caller reads for *exc*.

    A refusal already says which argument to change, so the traceback would only
    bury it. Anything else is unexpected, and there the frames are the point.
    """
    if isinstance(exc, ToolInputError):
        return exc.as_tool_result()
    return traceback.format_exc()


def _unknown_arguments_refusal(
    tool_name: str,
    tool_schema: dict,
    unknown: dict,
) -> ToolInputError:
    """Refuse a call that passed arguments *tool_name* has no parameter for.

    The parameters listed are the ones the tool's schema shows the model, so a
    misnamed argument can be matched to the name it was meant to be.
    """
    quoted = [repr(k) for k in unknown]
    named = (
        quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} or {quoted[-1]}"
    )
    params = ", ".join(tool_schema["function"]["parameters"]["properties"])
    suggestion = (
        f"Call {tool_name} again using only its own parameters: {params}."
        if params
        else f"Call {tool_name} again with no arguments."
    )
    return ToolInputError(
        f"{tool_name} was not run: it has no parameter named {named}.",
        suggestion=suggestion,
    )


async def _raise(exc: BaseException) -> None:
    """Run as a tool call's task, so *exc* reaches the model as its result."""
    raise exc


def _memory_result_hook():
    """UNIFY_MEMORY=on with a request run: memory v2's hook for a finished tool
    call (``hooks.tool_result``); None otherwise, and then the loop never enters
    it, so with memory v2 off a call is processed exactly as without it."""
    from unify.memory.integration import hooks

    return hooks.result_hook()


def _record_failure(
    tracker: Any,
    *,
    exc: BaseException,
    tool_name: str,
    args: Any,
) -> None:
    """Tally a failed call. Never raises — the caller decides when to stop.

    Refusals and unexpected exceptions are counted differently: a refusal is
    how a caller converges on an argspec, so only repetition ends the loop,
    while consecutive unexpected exceptions mean something is broken and a
    few is already too many. Stopping is left to ``tracker.stop_reason()`` at
    the end of the call so the failure reaches the transcript first — a loop
    that aborts before recording why is one nobody can diagnose.
    """
    if isinstance(exc, ToolInputError):
        tracker.note_refusal(
            tool_name=tool_name,
            args=args,
            message=exc.message,
        )
        return

    tracker.increment_failures()


class ToolsData:
    def __init__(
        self,
        tools,
        *,
        client,
        logger: "LoopLogger",
        call_counts: Optional[Dict[str, int]] = None,
    ):
        self._client = client
        # Calls abandoned after a bounded cancel, held until they end.
        self._abandoned: Set[asyncio.Task] = set()
        self._logger = logger
        self.normalized = normalise_tools(tools)
        self.pending: Set[asyncio.Task] = set()
        self.info: Dict[asyncio.Task, ToolCallMetadata] = {}
        self.call_counts: Dict[str, int] = (
            call_counts if call_counts is not None else {}
        )
        self.completed_results: Dict[str, str] = {}

    def _quota_count(self, task_name: str) -> int:
        return self.call_counts.get(task_name, 0)

    def _mutable(self, msg: dict) -> bool:
        """True when *msg* has not yet been included in any dispatched request."""
        return is_mutable(self._client, msg)

    def has_exceeded_quota_for_tool(self, task_name: str) -> bool:
        if task_name not in self.normalized:
            return False

        limit = self.normalized[task_name].max_total_calls
        return limit is not None and self._quota_count(task_name) >= limit

    def save_task(self, coro, metadata: ToolCallMetadata):
        self.pending.add(coro)
        self.info[coro] = metadata

    def pop_task(self, coro: asyncio.Task) -> ToolCallMetadata:
        self.pending.discard(coro)
        return self.info.pop(coro, None)

    async def cancel_pending_tasks(self, *, grace: Optional[float] = None) -> set:
        """Cancel every pending call and wait for each to stop.

        With *grace*, a call still running that long after its cancel (a
        tool that ignores cancellation) is abandoned: it is dropped from the
        loop's bookkeeping and left to end on its own, and the wait ends.
        Returns the abandoned tasks.
        """
        pending = list(self.pending)
        abandoned: set = set()
        for task in pending:
            task.cancel()
        waiting = asyncio.gather(*pending, return_exceptions=True)
        if grace is None:
            await waiting
        else:
            await asyncio.wait({asyncio.ensure_future(waiting)}, timeout=grace)
        for task in pending:
            if not task.done():
                abandoned.add(task)
                # Held until it ends, so it is not garbage-collected while
                # it runs; its result is never read.
                self._abandoned.add(task)
                task.add_done_callback(self._abandoned.discard)
            self.pending.discard(task)
            self.info.pop(task, None)
        if abandoned:
            self._logger.error(
                f"{len(abandoned)} call(s) still running {grace:g}s after their "
                "cancel were abandoned: "
                + ", ".join(sorted(t.get_name() for t in abandoned)),
                prefix="⚠️",
            )
        return abandoned

    async def cancel_pending_tasks_with_reply(
        self,
        content: str,
        *,
        assistant_meta,
        msg_dispatcher,
        grace: Optional[float] = None,
    ) -> list[str]:
        """Cancel every pending call and give each *content* as its final reply.

        For a loop that goes on after the cancellation (a persistent request
        ended at its step limit, a cancelled request): an unanswered call would be run again before
        the next request. The reply is delivered as a completed result would be, and is
        remembered as one, so the call is not run again. Returns the call ids.
        *grace* is as for :meth:`cancel_pending_tasks`; the reply to an
        abandoned call says so.
        """
        by_task = {task: self.info.get(task) for task in list(self.pending)}
        abandoned = await self.cancel_pending_tasks(grace=grace)
        answered: list[str] = []
        for task, info in by_task.items():
            if info is None or info.call_id in self.completed_results:
                continue
            reply = content
            if task in abandoned:
                reply += (
                    f" It was still running {grace:g}s after the cancel and was "
                    "abandoned; whatever it still does is not reported."
                )
            self.completed_results[info.call_id] = reply
            await insert_tool_message_after_assistant(
                assistant_meta,
                info.assistant_msg,
                create_tool_call_message(info.name, info.call_id, reply),
                self._client,
                msg_dispatcher,
                bypass_watermark=True,
            )
            answered.append(info.call_id)
        return answered

    def prune_over_quota_tool_calls(self, asst_msg: dict) -> None:
        """Remove, in place, the tool_calls of asst_msg that would exceed the
        per-tool quota. Calls that are not executed must not remain in the
        history without a response, or the provider rejects the request.

        Only safe while asst_msg is still mutable (not yet included in a
        dispatched request): an in-place tool_calls edit below the sent
        watermark would shift every already-dispatched message that follows.
        Both call sites only reach a message at preflight (watermark 0) or
        the current turn's own message (index == watermark); the assertion
        makes that a stated invariant instead of caller discipline.
        """
        tcs = asst_msg.get("tool_calls")
        if not tcs:
            return
        if not is_mutable(self._client, asst_msg):
            # Logged explicitly: the callers (preflight repair, the
            # persist-mode branch) wrap this in suppress/except-pass, which
            # would otherwise swallow the raise along with the failure.
            _msg = (
                "prune_over_quota_tool_calls: asst_msg is already below the "
                "sent watermark; an in-place tool_calls edit would mutate "
                "already-dispatched bytes."
            )
            self._logger.error(_msg, prefix="🚨")
            raise ValueError(_msg)

        # Count locally across this batch; self.call_counts itself is only
        # incremented when a call is scheduled.
        temp_counts = self.call_counts.copy()

        valid_tcs = []
        for tc in tcs:
            try:
                name = tc.get("function", {}).get("name")

                if name not in self.normalized:
                    # Unknown tools are kept (handled by execution/error logic)
                    valid_tcs.append(tc)
                    continue

                spec = self.normalized[name]
                limit = spec.max_total_calls
                current = temp_counts.get(name, 0)

                if limit is not None and current >= limit:
                    continue

                temp_counts[name] = current + 1
                valid_tcs.append(tc)
            except Exception:
                # Malformed tool call, keep it
                valid_tcs.append(tc)

        asst_msg["tool_calls"] = valid_tcs

        # An assistant message with neither content nor tool_calls is
        # rejected by the API; a placeholder also tells the model why.
        has_content = bool(asst_msg.get("content"))
        if not valid_tcs and not has_content:
            asst_msg["content"] = "(Tool calls were removed due to quota limits)"

    # Shared by the main dispatch path and backfill.
    async def schedule_base_tool_call(
        self,
        asst_msg: dict,
        *,
        name: str,
        args_json: Any,
        call_id: str,
        call_idx: int,
        assistant_meta,
        msg_dispatcher: Optional["LoopMessageDispatcher"] = None,
    ) -> Optional[asyncio.Task]:
        """Start one call of tool *name*; returns its task, or ``None`` when
        the tool is unknown or its call limit is reached (not started)."""
        if name not in self.normalized:
            return None

        fn = self.normalized[name].fn

        # Over-quota calls should already be pruned from the assistant
        # message; one that slipped through is not started.
        with suppress(Exception):
            lim = self.normalized[name].max_total_calls
            if lim is not None and self.call_counts.get(name, 0) >= lim:
                return None

        with suppress(Exception):
            call_args = (
                json.loads(args_json)
                if isinstance(args_json, str)
                else (args_json or {})
            )
        if "call_args" not in locals():
            call_args = {}

        allowed_call_args, unknown_call_args = _normalise_kwargs_for_bound_method(
            fn,
            call_args,
        )
        merged_kwargs = dict(allowed_call_args)

        tool_schema = method_to_schema(fn, name)

        # An argument the tool has no parameter for fails the call instead of
        # being dropped. Dropped, it would leave the tool to run on its
        # defaults and return a plausible result for a request the model never
        # made, with nothing to tell the model its argument was ignored. The
        # refusal names the tool's parameters, so the model can reissue the
        # call; ToolInputError makes it a refusal, which ends the loop only
        # when it repeats.
        if unknown_call_args:
            coro = _raise(
                _unknown_arguments_refusal(name, tool_schema, unknown_call_args),
            )
        # Argument binding for an async fn happens synchronously at coroutine
        # creation, so a model omitting a required argument raises TypeError
        # here — outside the task machinery that turns failures into tool
        # results. Convert it into a task-level failure so the model sees the
        # error and self-corrects instead of the whole trajectory dying. The
        # sync branch is already safe: asyncio.to_thread defers binding into
        # the task.
        elif asyncio.iscoroutinefunction(fn):
            try:
                coro = fn(**merged_kwargs)
            except TypeError as bind_exc:
                coro = _raise(bind_exc)
        else:
            coro = asyncio.to_thread(fn, **merged_kwargs)

        call_dict = {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": args_json},
        }

        t = asyncio.create_task(coro, name=f"ToolCall_{name}")
        metadata = ToolCallMetadata(
            name=name,
            call_id=call_id,
            assistant_msg=asst_msg,
            call_dict=call_dict,
            call_idx=call_idx,
            # Debug helpers for failure logging. The arguments keep any the
            # tool has no parameter for: refusal tallies fingerprint them, and
            # two calls that differ only there are different calls.
            tool_schema=tool_schema,
            llm_arguments={**allowed_call_args, **unknown_call_args},
            raw_arguments_json=args_json,
        )
        self.save_task(t, metadata)

        if self._logger.log_steps:
            self._logger.info(
                f"{name} - {call_id}",
                prefix=f"🛠️  ToolCall Scheduled",
            )

        # The quota counter moves only once scheduling has succeeded.
        with suppress(Exception):
            self.call_counts[name] = self.call_counts.get(name, 0) + 1

        # Ensure assistant meta exists for deterministic insertion ordering
        assistant_meta.setdefault(id(asst_msg), {"results_count": 0})
        return t

    async def process_completed_task(
        self,
        task: asyncio.Task,
        consecutive_failures: "_LoopToolFailureTracker",
        assistant_meta,
        msg_dispatcher,
    ) -> bool:
        info = self.info[task]
        try:
            return await self._process_completed_task(
                task,
                consecutive_failures,
                assistant_meta,
                msg_dispatcher,
            )
        except asyncio.CancelledError:
            # Completion temporarily removes the task from the pending set;
            # it goes back, so the loop's cleanup still sees it.
            self.save_task(task, info)
            raise

    async def _process_completed_task(
        self,
        task: asyncio.Task,
        consecutive_failures: "_LoopToolFailureTracker",
        assistant_meta,
        msg_dispatcher,
    ) -> bool:
        """Deal with a finished tool *task* exactly once: pop its bookkeeping
        (``pending`` / ``info``), serialise success or exception into
        ``result``, insert the tool message after its assistant message,
        publish it to the event bus, record the payload in
        ``completed_results`` for post-hoc lookups and enforce the
        *max_consecutive_failures* safety valve.
        """
        import time as _pct_time

        _pct_t0 = _pct_time.perf_counter()

        def _pct_ms():
            return f"{(_pct_time.perf_counter() - _pct_t0) * 1000:.0f}ms"

        info: ToolCallMetadata = self.pop_task(task)
        name = info.name
        call_id = info.call_id

        _pickup_delay = _pct_time.perf_counter() - info.scheduled_time
        self._logger.debug(
            f"⏱️ [ToolsData.process_completed +{_pct_ms()}] {name} ({call_id}) "
            f"total_elapsed={_pickup_delay:.2f}s",
        )

        # UNIFY_MEMORY=on with a request run: the hook noting a code cell's
        # structured result, by call id, for the request's use record. None
        # otherwise, and then it is never entered; it never fails the call.
        _mv2_note = None
        _returned = False
        try:
            _mv2_note = _memory_result_hook()
            raw = task.result()
            _returned = True
            if _mv2_note is not None:
                with suppress(Exception):
                    _mv2_note(name, call_id, raw)
            result = serialize_tool_content(tool_name=name, payload=raw, is_final=True)
            consecutive_failures.reset_failures()
        except Exception as exc:
            result = _failure_text(exc)
            if _mv2_note is not None and not _returned:  # the call itself raised
                with suppress(Exception):
                    _mv2_note(name, call_id, None, raised=exc)
            _record_failure(
                consecutive_failures,
                exc=exc,
                tool_name=name,
                args=info.llm_arguments,
            )
            if self._logger.log_steps:
                self._logger.error(
                    f"Error: {name} failed "
                    f"(attempt {consecutive_failures.current_failures}/{consecutive_failures.max_failures}):\n{result}",
                    prefix="❌",
                )
                # The exact schema and arguments the LLM saw for this call, to
                # diagnose docstring/argspec mismatches behind tool misuse.
                with suppress(Exception):
                    debug_payload = {
                        "tool_name": name,
                        "call_id": call_id,
                        "llm_function_schema": info.tool_schema,
                        "llm_arguments": info.llm_arguments,
                        "raw_arguments_json": info.raw_arguments_json,
                    }
                    self._logger.error(
                        f"FAILED TOOL SCHEMA (as given to LLM):\n{json.dumps(debug_payload, indent=2)}",
                        prefix="🧩",
                    )

        # Remembered so later lookups can answer instantly.
        self.completed_results[call_id] = result

        tool_msg = create_tool_call_message(name, call_id, result)
        # First-ever reply to this call_id — legality requires strict
        # adjacency, so this always bypasses the watermark gate (see
        # insert_tool_message_after_assistant's escape hatch).
        await insert_tool_message_after_assistant(
            assistant_meta,
            info.assistant_msg,
            tool_msg,
            self._client,
            msg_dispatcher,
            bypass_watermark=True,
        )

        if self._logger.log_steps:
            # Exactly what was inserted, with base64 data URLs redacted.
            try:
                safe_for_logs = sanitize_tool_msg_for_logging(tool_msg)
                self._logger.info(
                    f"{json.dumps(safe_for_logs, indent=4)}",
                    prefix=f"✅  ToolCall Completed [{time.perf_counter() - info.scheduled_time:.2f}s]",
                )
            except Exception:
                pass

        stop_reason = consecutive_failures.stop_reason()
        if stop_reason:
            if self._logger.log_steps:
                self._logger.error(f"Aborting: {stop_reason}", prefix="🚨")
            raise RuntimeError(stop_reason)

        # A final result, success or failure: the LLM may need to react.
        return True
