import asyncio
import inspect
import json
import traceback
import time


from typing import (
    Dict,
    Set,
    Tuple,
    Any,
    Optional,
    TYPE_CHECKING,
)
from .tools_utils import ToolCallMetadata, create_tool_call_message
from .messages import (
    insert_tool_message_after_assistant,
    _normalise_kwargs_for_bound_method,
    apply_llm_soft_required_defaults,
    is_mutable,
)
from ..tool_spec import normalise_tools
from ..llm_helpers import method_to_schema
from .formatting import serialize_tool_content, sanitize_tool_msg_for_logging
from contextlib import suppress
from .propagation_mode import ChatContextPropagation
from ..tool_errors import ToolInputError


from .context_tracker import LoopContextState

if TYPE_CHECKING:  # TODO: remove once dependencies are fixed
    from .loop import LoopLogger, _LoopToolFailureTracker
    from .message_dispatcher import LoopMessageDispatcher
    from .time_context import TimeContext


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


def _returned_handles_for_cleanup(value):
    """Find handles in supported stored values without evaluating accessors.

    Cycles and aliases are traversed once. Arbitrary iterables and computed
    attributes are not part of the returned-resource ownership contract.
    """
    from pydantic import BaseModel
    from unify.common.async_tool_loop import SteerableToolHandle, ToolLoopHandle

    pending = [value]
    visited = set()
    handles = []
    while pending:
        node = pending.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        node_type = type(node)
        if issubclass(node_type, (SteerableToolHandle, ToolLoopHandle)):
            handles.append(node)
        elif issubclass(node_type, dict):
            pending.extend(dict.values(node))
        elif issubclass(node_type, list):
            pending.extend(list.__iter__(node))
        elif issubclass(node_type, tuple):
            pending.extend(tuple.__iter__(node))
        elif issubclass(node_type, BaseModel):
            fields = inspect.getattr_static(node_type, "__pydantic_fields__", {})
            stored = BaseModel.__dict__["__dict__"].__get__(node)
            if type(fields) is dict and type(stored) is dict:
                pending.extend(stored[name] for name in fields if name in stored)
    return handles


def compute_context_injection(
    *,
    args: dict,
    propagate_chat_context: ChatContextPropagation,
    context_state: LoopContextState,
    client_messages: list,
    call_id: str,
    accepts_parent_ctx: bool,
    accepts_parent_ctx_cont: bool,
    target_context_opted_in: Optional[bool] = None,
    is_continuation_only: bool = False,
) -> Tuple[dict, bool]:
    """Compute the parent-chat-context kwargs for one tool call.

    Shared by base tool dispatch and dynamic tool dispatch so
    ``include_parent_chat_context`` and ``include_parent_chat_context_cont``
    are handled identically.

    ``args`` is the tool call's arguments and is mutated: the two control
    params are popped. ``propagate_chat_context`` is the loop's mode (ALWAYS,
    NEVER or LLM_DECIDES), ``context_state`` its context tracker,
    ``client_messages`` the current conversation (``_ctx_header`` messages are
    filtered out) and ``call_id`` identifies the call for context tracking.
    ``accepts_parent_ctx`` / ``accepts_parent_ctx_cont`` say whether the target
    function accepts ``_parent_chat_context`` / ``_parent_chat_context_cont``.
    ``target_context_opted_in`` is, for steering tools, whether the target
    tool initially opted into context; ``None`` means a fresh tool call, not
    steering. ``is_continuation_only`` computes only the continuation context
    (for interject_*) instead of the full initial context (base tools and
    ask_*).

    Returns ``(extra_kwargs, context_opted_in)``: the context params to
    inject, and the opt-in decision.
    """
    extra_kwargs: dict = {}

    # Initial context injection is opt-in: an omitted
    # include_parent_chat_context means no parent context.
    llm_include_ctx = args.pop("include_parent_chat_context", False)
    llm_include_ctx_cont = args.pop("include_parent_chat_context_cont", True)

    should_inject_ctx = False

    if is_continuation_only:
        if target_context_opted_in:
            if propagate_chat_context == ChatContextPropagation.ALWAYS:
                should_inject_ctx = True
            elif propagate_chat_context == ChatContextPropagation.LLM_DECIDES:
                should_inject_ctx = llm_include_ctx_cont
            # NEVER mode: should_inject_ctx stays False
    else:
        if accepts_parent_ctx or accepts_parent_ctx_cont:
            if propagate_chat_context == ChatContextPropagation.ALWAYS:
                should_inject_ctx = True
            elif propagate_chat_context == ChatContextPropagation.NEVER:
                should_inject_ctx = False
            elif propagate_chat_context == ChatContextPropagation.LLM_DECIDES:
                should_inject_ctx = llm_include_ctx

    if should_inject_ctx:
        cur_msgs = [m for m in client_messages if not m.get("_ctx_header")]

        if is_continuation_only:
            _, ctx_cont = context_state.compute_context_for_inner_tool(
                call_id,
                cur_msgs,
            )
            if ctx_cont and accepts_parent_ctx_cont:
                extra_kwargs["_parent_chat_context_cont"] = ctx_cont
        else:
            parent_ctx, parent_ctx_cont = context_state.compute_context_for_inner_tool(
                call_id,
                cur_msgs,
            )
            if parent_ctx is not None and accepts_parent_ctx:
                extra_kwargs["_parent_chat_context"] = parent_ctx
            if parent_ctx_cont is not None and accepts_parent_ctx_cont:
                extra_kwargs["_parent_chat_context_cont"] = parent_ctx_cont

    return extra_kwargs, should_inject_ctx


class ToolsData:
    def __init__(
        self,
        tools,
        *,
        client,
        logger: "LoopLogger",
        time_ctx: "Optional[TimeContext]" = None,
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
        self.clarification_channels: Dict[
            str,
            Tuple[asyncio.Queue[str], asyncio.Queue[str]],
        ] = {}
        self.completed_results: Dict[str, str] = {}
        # Tool name for every completed tool (steerable or not), keyed by call_id.
        self._completed_tool_names: Dict[str, str] = {}
        # Time context for inline timing annotations on tool results
        self._time_ctx: Optional["TimeContext"] = time_ctx

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

    def has_exceeded_concurrent_limit_for_tool(self, task_name: str) -> bool:
        if task_name not in self.normalized:
            return False

        limit = self.normalized[task_name].max_concurrent
        return limit is not None and self.active_count(task_name) >= limit

    def save_task(self, coro, metadata: ToolCallMetadata):
        self.pending.add(coro)
        self.info[coro] = metadata

    def pop_task(self, coro: asyncio.Task) -> ToolCallMetadata:
        self.pending.discard(coro)
        info = self.info.pop(coro, None)
        if info is not None:
            self.clarification_channels.pop(info.call_id, None)
        return info

    def active_count(self, task_name: str) -> int:
        return sum(1 for _t, _inf in self.info.items() if _inf.name == task_name)

    async def cancel_pending_tasks(self, *, grace: Optional[float] = None) -> set:
        """Cancel every pending call and wait for each to stop.

        With *grace*, a call still running that long after its cancel (a
        tool that ignores cancellation) is abandoned: it is dropped from the
        loop's bookkeeping and left to end on its own, and the wait ends.
        Returns the abandoned tasks.
        """
        pending = list(self.pending)
        stopped = set()
        abandoned: set = set()

        async def stop_handle(handle):
            result = handle.stop("loop cancelled")
            if inspect.isawaitable(result):
                await result

        def stop_tasks():
            handles = []
            for task in pending:
                info = self.info.get(task)
                if info and info.handle is not None:
                    handles.append(info.handle)
                if task.done() and not task.cancelled() and task.exception() is None:
                    try:
                        handles.extend(_returned_handles_for_cleanup(task.result()))
                    except Exception as exc:
                        # Malformed returned data cannot abandon sibling cleanup.
                        self._logger.error(
                            f"Returned handle inspection failed: {type(exc).__name__}",
                            prefix="⚠️",
                        )
            stops = []
            for handle in handles:
                if id(handle) in stopped:
                    continue
                stopped.add(id(handle))
                stops.append(asyncio.create_task(stop_handle(handle)))
            return stops

        async def join_stops(stops):
            for result in await asyncio.gather(*stops, return_exceptions=True):
                if isinstance(result, BaseException):
                    self._logger.error(
                        f"Owned handle stop failed: {type(result).__name__}",
                        prefix="⚠️",
                    )

        async def bounded(waiting) -> bool:
            """Await *waiting*, at most *grace* seconds; whether it ended."""
            if grace is None:
                await waiting
                return True
            future = asyncio.ensure_future(waiting)
            done, _ = await asyncio.wait({future}, timeout=grace)
            return bool(done)

        stops = stop_tasks()
        for task in pending:
            task.cancel()
        # Stop and cancel concurrently: an async stop can await its owned task.
        # A child's stop failure must not cancel sibling cleanup operations.
        if await bounded(
            asyncio.gather(*pending, join_stops(stops), return_exceptions=True),
        ):
            # A factory may catch cancellation and return a newly owned handle.
            await bounded(join_stops(stop_tasks()))
        for task in pending:
            if not task.done():
                abandoned.add(task)
                # Held until it ends, so it is not garbage-collected while
                # it runs; its result is never read.
                self._abandoned.add(task)
                task.add_done_callback(self._abandoned.discard)
            self.pending.discard(task)
            info = self.info.pop(task, None)
            if info is not None:
                self.clarification_channels.pop(info.call_id, None)
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

        For a loop that goes on after the cancellation (UNIFY_STEP_CAP_REPLY,
        a cancelled request): an unanswered call would be run again before
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
            self._completed_tool_names[info.call_id] = info.name
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
        context_state: LoopContextState,
        propagate_chat_context,
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

        sig = inspect.signature(fn)
        params = sig.parameters
        has_varkw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

        # Parsed before context injection, which pops include_parent_chat_context.
        with suppress(Exception):
            call_args = (
                json.loads(args_json)
                if isinstance(args_json, str)
                else (args_json or {})
            )
        if "call_args" not in locals():
            call_args = {}

        sig_accepts_parent_ctx = "_parent_chat_context" in params or has_varkw
        sig_accepts_parent_ctx_cont = "_parent_chat_context_cont" in params or has_varkw

        ctx_extra_kwargs, context_opted_in = compute_context_injection(
            args=call_args,
            propagate_chat_context=propagate_chat_context,
            context_state=context_state,
            client_messages=self._client.messages,
            call_id=call_id,
            accepts_parent_ctx=sig_accepts_parent_ctx,
            accepts_parent_ctx_cont=sig_accepts_parent_ctx_cont,
            is_continuation_only=False,
        )

        extra_kwargs: dict = dict(ctx_extra_kwargs)

        sig_accepts_clar_qs = (
            "_clarification_up_q" in params and "_clarification_down_q" in params
        ) or has_varkw
        sig_accepts_progress = "_notification_up_q" in params or has_varkw

        clar_up_q: Optional[asyncio.Queue[str]] = None
        clar_down_q: Optional[asyncio.Queue[str]] = None
        if sig_accepts_clar_qs:
            clar_up_q = asyncio.Queue()
            clar_down_q = asyncio.Queue()
            extra_kwargs["_clarification_up_q"] = clar_up_q
            extra_kwargs["_clarification_down_q"] = clar_down_q

        progress_q: Optional[asyncio.Queue[dict]] = None
        if sig_accepts_progress:
            progress_q = asyncio.Queue()
            extra_kwargs["_notification_up_q"] = progress_q

        filtered_extras = {
            k: v for k, v in extra_kwargs.items() if k in params or has_varkw
        }
        allowed_call_args, unknown_call_args = _normalise_kwargs_for_bound_method(
            fn,
            call_args,
        )
        merged_kwargs = {**allowed_call_args, **filtered_extras}

        # Backfill advisory args advertised as required but safe to default
        # (e.g. execute_code's `thought`): the schema keeps its strong
        # `required` signal without a model omission raising TypeError.
        merged_kwargs = apply_llm_soft_required_defaults(fn, merged_kwargs)

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
            is_interjectable=False,
            chat_context=extra_kwargs.get("_parent_chat_context"),
            clar_up_queue=clar_up_q,
            clar_down_queue=clar_down_q,
            notification_queue=progress_q,
            # Debug helpers for failure logging. The arguments keep any the
            # tool has no parameter for: refusal tallies fingerprint them, and
            # two calls that differ only there are different calls.
            tool_schema=tool_schema,
            llm_arguments={**allowed_call_args, **unknown_call_args},
            raw_arguments_json=args_json,
            context_opted_in=context_opted_in,
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

        if clar_up_q is not None:
            self.clarification_channels[call_id] = (
                clar_up_q,
                clar_down_q,
            )

        # Ensure assistant meta exists for deterministic insertion ordering
        assistant_meta.setdefault(id(asst_msg), {"results_count": 0})
        return t

    async def process_completed_task(
        self,
        task: asyncio.Task,
        consecutive_failures: "_LoopToolFailureTracker",
        outer_handle_container,
        assistant_meta,
        msg_dispatcher,
    ) -> bool:
        info = self.info[task]
        try:
            return await self._process_completed_task(
                task,
                consecutive_failures,
                outer_handle_container,
                assistant_meta,
                msg_dispatcher,
            )
        except asyncio.CancelledError:
            # Completion temporarily removes the factory from the pending set.
            # Preserve ownership of its returned handles for loop cleanup.
            self.save_task(task, info)
            raise

    async def _process_completed_task(
        self,
        task: asyncio.Task,
        consecutive_failures: "_LoopToolFailureTracker",
        outer_handle_container,
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

        # Notifications that arrived just before completion still reach the
        # handle (they never enter the transcript), so none is lost.
        q = info.notification_queue
        if q is not None:
            while True:
                try:
                    payload = q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                with suppress(Exception):
                    outer = (
                        outer_handle_container[0] if outer_handle_container else None
                    )
                    if outer is not None and hasattr(outer, "_notification_q"):
                        event_payload = (
                            payload
                            if isinstance(payload, dict)
                            else {"message": str(payload)}
                        )
                        await outer._notification_q.put(
                            {
                                "type": "notification",
                                "call_id": call_id,
                                "tool_name": name,
                                **event_payload,
                            },
                        )

        try:
            raw = task.result()
            result = serialize_tool_content(tool_name=name, payload=raw, is_final=True)

            if self._time_ctx is not None and not info.is_dynamic:
                result = self._time_ctx.wrap_result(result, info.scheduled_time)

            consecutive_failures.reset_failures()
        except Exception as exc:
            result = _failure_text(exc)
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
        self._completed_tool_names[call_id] = name

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
