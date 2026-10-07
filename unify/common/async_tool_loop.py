import asyncio
import unillm
from abc import ABC, abstractmethod
from contextlib import suppress
from typing import (
    Optional,
    Awaitable,
    Dict,
    Callable,
    Tuple,
    Any,
    Union,
    TYPE_CHECKING,
)
from ..logger import LOGGER
from unify.common.hierarchical_logger import ICONS
from .llm_helpers import short_id
from .llm_client import fork_llm_client
from ._async_tool import cache_discipline as _cache_discipline
from ._async_tool import bound_request as _bound_request
from unify import transcripts
from ._async_tool.loop_config import TOOL_LOOP_LINEAGE
from ._async_tool.loop import ToolLoopRuntimeState, async_tool_loop_inner
from ._async_tool.propagation_mode import ChatContextPropagation
from ._async_tool.context_compression import (
    _COMPRESSED_HEADER,
    _COMPRESSION_SIGNAL,
    CompressionState,
    compress_and_rebuild,
)

if TYPE_CHECKING:
    from unillm.types import PromptCacheParam


# ── Handles ──────────────────────────────────────────────────────────────────


class SteerableToolHandle(ABC):
    """Abstract base class for steerable tool handles.

    Defines the full steering surface: query (``ask``, ``interject``),
    lifecycle (``stop``, ``pause``, ``resume``), completion (``done``,
    ``result``), and event APIs (``next_clarification``,
    ``next_notification``, ``answer_clarification``).

    Context parameters
    ------------------
    Steering methods accept plumbing parameters that are hidden from LLM tool
    schemas by their underscore prefix and injected by orchestrating code:

    - ``_parent_chat_context_cont`` (for ``interject``): continuation of the
      parent conversation since this loop started, injected into the ongoing
      conversation as an incremental update.

    - ``_parent_chat_context`` (for ``ask``): full context snapshot for the
      fresh inspection loop ``ask`` spawns, injected according to the LLM's
      ``include_parent_chat_context`` choice.

    Signature extension contract
    ----------------------------
    Derived classes may extend any steering method signature with additional
    keyword arguments specific to their domain. The signatures defined here
    are the minimum universal contract every handle accepts, so callers that
    hold a reference typed as ``SteerableToolHandle`` may safely pass only
    these base parameters.

    When dispatching a steering call to a handle whose concrete type is
    unknown, use ``forward_handle_call`` (from
    ``unify.common._async_tool.messages``): it introspects the target
    method's signature, filters out kwargs the target does not accept, and
    applies positional fallbacks.
    """

    @abstractmethod
    def __init__(
        self,
    ) -> None:
        pass

    @abstractmethod
    async def ask(
        self,
        question: str,
        *,
        _parent_chat_context: list[dict] | None = None,
    ) -> "SteerableToolHandle":
        """Ask about status/progress if the task is still running, or the retrospective process/method if it has completed.

        Read-only — does not modify the task. This operation is asynchronous:
        it returns immediately and the answer appears in the task's history on
        the next turn.
        """

    @abstractmethod
    async def interject(
        self,
        message: str,
        *,
        _parent_chat_context_cont: list[dict] | None = None,
    ) -> None:
        """Provide additional information or instructions to the running task.

        Use this to give the task new context, correct its approach, or add
        requirements mid-flight without stopping or restarting it.
        """

    @abstractmethod
    async def stop(
        self,
        reason: Optional[str] = None,
    ) -> None:
        """Stop this tool, cancelling any pending work.

        While any tools are still running you cannot end the conversation;
        stop or wait for all in-flight tools to complete, then respond.
        """

    async def cancel_request(self, reason: Optional[str] = None) -> bool:
        """End the running request of a persistent task, keeping the task.

        For a host (``unify act --jsonl``'s ``{"cancel": true}``); not a
        steering action. Returns whether the cancel was delivered; a handle
        that serves no persistent requests has none to cancel.
        """
        return False

    @abstractmethod
    async def pause(self) -> Optional[str]:
        """Pause this task temporarily without cancelling it.

        In-flight operations continue executing, but no new actions are taken
        until resumed.
        """

    @abstractmethod
    async def resume(self) -> Optional[str]:
        """Resume a task that was previously paused.

        Any work that completed while paused will be processed before the
        task continues.
        """

    @abstractmethod
    def done(self) -> Awaitable[bool] | bool:
        """Check if this task has completed."""

    @abstractmethod
    def result(self) -> Awaitable[str] | str:
        """Wait for the assistant's *final* reply."""

    @abstractmethod
    async def next_clarification(self) -> dict:
        """Await the next clarification event pushed by a running tool."""

    @abstractmethod
    async def next_notification(self) -> dict:
        """Await the next notification pushed by a running tool."""

    @abstractmethod
    async def answer_clarification(self, call_id: str, answer: str) -> None:
        """Answer a clarification question that the task is waiting on.

        Provide the call_id from the clarification request and the answer text.
        No-op if the tool already finished.
        """

    def get_history(self) -> list[dict]:
        """The loop's conversational history, in the LLM client's message
        format. Empty for handles without an LLM client."""
        return []


class ToolLoopHandle(ABC):
    """What a caller holds for a running tool loop: no steering.

    ``result()`` waits for the final answer, ``done()`` says whether the loop
    has ended, ``stop(reason)`` ends it (cancelling the model call or tool
    call in flight), and ``submit(text)`` queues the requester's next message,
    which the loop appends at its next turn boundary (a persistent session's
    next request). Nothing a caller does interrupts a model call or a tool
    call except ``stop``. The event methods carry what a running call sends
    the requester: progress notifications, the turn responses of a persistent
    session, and the questions a cell asks with ``request_clarification``.
    """

    @abstractmethod
    def result(self) -> Awaitable[Any]:
        """Wait for the loop's final answer."""

    @abstractmethod
    def done(self) -> bool:
        """Whether the loop has ended."""

    @abstractmethod
    async def stop(self, reason: Optional[str] = None) -> None:
        """End the loop, cancelling the model call or tool call in flight."""

    @abstractmethod
    async def submit(self, text: str) -> None:
        """Queue *text* for the next turn boundary (never mid-turn)."""

    async def cancel_request(self, reason: Optional[str] = None) -> bool:
        """End the running request of a persistent session, keeping the
        session (``unify act --jsonl``'s ``{"cancel": true}``). Returns whether
        the cancel was delivered."""
        return False

    async def next_notification(self) -> dict:
        """Await the next notification or persistent-session response."""
        raise NotImplementedError

    async def next_clarification(self) -> dict:
        """Await the next question a running call asks the requester."""
        raise NotImplementedError

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        """Answer the question call *call_id* asked; a no-op once it ended."""
        return None

    def get_history(self) -> list[dict]:
        """The loop's conversational history, in the LLM client's message
        format. Empty for handles without an LLM client."""
        return []


class AsyncToolLoopHandle(ToolLoopHandle):
    """Returned by ``start_async_tool_loop``: waits for, stops and feeds the
    running loop."""

    def __init__(
        self,
        *,
        task: asyncio.Task,
        interject_queue: asyncio.Queue[dict | str],
        cancel_event: asyncio.Event,
        stop_event: asyncio.Event,
        client: "unillm.AsyncUnify | None" = None,
        loop_id: str = "",
        response_format: Optional[Any] = None,
        runtime_state: Optional[ToolLoopRuntimeState] = None,
    ):
        self._task = task
        self._queue = interject_queue
        self._cancel_event = cancel_event
        self._stop_event = stop_event
        self._client = client
        self._loop_id: str = loop_id
        # Log label with the 4-hex suffix, set by the inner loop once it
        # builds its LoopConfig; the bare loop_id until then.
        self._log_label: str = loop_id
        self._loop_cfg: Optional[Any] = None
        # When set, result() parses the loop's raw JSON string into this
        # Pydantic model.
        self._response_format: Optional[Any] = response_format

        self._clar_q: asyncio.Queue[dict] = asyncio.Queue()
        self._notification_q: asyncio.Queue[dict] = asyncio.Queue()
        # call_id -> (question queue, answer queue) of the calls that can ask
        # the requester; the running loop keeps it current.
        self._clarification_channels: dict = {}

        self._compression = CompressionState()
        self._loop_config: Optional[dict] = None
        self._runtime_state = runtime_state or ToolLoopRuntimeState()

    async def submit(self, text: str) -> None:
        """Queue *text*, appended as a user message at the next boundary.

        While a model call or a tool call runs it waits; a persistent session
        parked between requests takes it as its next request.
        """
        _label = getattr(self, "_log_label", None) or self._loop_id
        LOGGER.debug(f"{ICONS['interjection']} [{_label}] Message submitted: {text}")
        # put_nowait registers the message synchronously, before this
        # coroutine yields; otherwise a fast loop can finish its turn and
        # exit before seeing the queued item.
        self._queue.put_nowait({"message": text})

    async def stop(self, reason: Optional[str] = None, **kwargs) -> None:
        # Idempotent: a second stop neither logs nor re-signals.
        if self._cancel_event.is_set():
            return
        _label = getattr(self, "_log_label", None) or self._loop_id
        suffix = f" – reason: {reason}" if reason else ""
        LOGGER.info(f"{ICONS['stop_requested']} [{_label}] Stop requested{suffix}")
        with suppress(Exception):
            self._cancel_event.set()
        with suppress(Exception):
            self._stop_event.set()

    async def cancel_request(self, reason: Optional[str] = None) -> bool:
        # The loop takes the cancel at its next boundary or, while a tool
        # call runs, at once (the call is cancelled and answered as such); a
        # model call in flight runs to its end first. A persistent loop ends
        # the request; a parked loop has none running, and a loop that is not
        # persistent is stopped with stop(). Nothing after a stop.
        if self._task.done() or self._cancel_event.is_set():
            return False
        _label = getattr(self, "_log_label", None) or self._loop_id
        LOGGER.info(
            f"{ICONS['interjection']} [{_label}] Cancel of the request requested",
        )
        self._queue.put_nowait({"_cancel_request": {"reason": reason}})
        return True

    def done(self) -> bool:
        return self._task.done()

    async def result(self):
        """Return the final answer once the conversation loop completes.

        When *response_format* was supplied to ``start_async_tool_loop``, the
        raw JSON string produced by the inner loop is parsed into a Pydantic
        model instance.

        If the inner loop returns ``_COMPRESSION_SIGNAL``, the handle
        compresses the context and starts a new loop transparently, as many
        times as needed; callers always receive the final real result.
        """
        _stopped_notice = "processed stopped early, no result"
        while True:
            try:
                raw = await self._task
            except asyncio.CancelledError:
                # Only a loop that died on its own is reported as stopped. A
                # cancellation aimed at the caller — an enclosing ``wait_for``
                # or ``asyncio.timeout``, a task group tearing down — has to
                # propagate, otherwise the caller's timeout is answered with a
                # value and reads as a loop that finished with no answer.
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
                return _stopped_notice

            if raw is _COMPRESSION_SIGNAL:
                try:
                    await self._restart_with_compressed_context()
                except Exception as exc:
                    LOGGER.error(
                        f"Context compression failed: {type(exc).__name__}: {exc}",
                    )
                    return _stopped_notice
                continue

            if self._response_format is not None and isinstance(raw, str):
                try:
                    from unify.common._async_tool.response_format import (
                        try_normalize_response_format,
                    )

                    normalized = try_normalize_response_format(self._response_format)
                    if normalized is not None:
                        return normalized.parse_result(raw)
                except Exception:
                    pass
            return raw

    async def _restart_with_compressed_context(self) -> None:
        """Compress context and start a new loop iteration.

        ``compress_and_rebuild`` does the data transformation; this method
        handles the loop lifecycle: replacing client messages, creating a new
        ``asyncio.Task``, and swapping the task reference so the handle's
        methods target the new loop.
        """
        cfg = self._loop_config
        if cfg is None:
            raise RuntimeError(
                "Cannot compress: loop config was not stored on the handle.",
            )

        # UNIFY_STEP_CAP_COMPACT: the loop compacted the context itself at
        # its step limit, before it ended; otherwise it is compacted here.
        at_step_limit = self._runtime_state.step_cap_compacted
        self._runtime_state.step_cap_compacted = None
        compacted = at_step_limit or await self._compact_context(cfg)
        n_archived, restart_messages, restart_tools, restart_message, forked = compacted
        self._client._messages = restart_messages
        if not forked:
            self._client._system_message = None

        # A compression rebuild is a deliberate full-cache sacrifice: the
        # transcript it replaces no longer exists, so nothing in the new one
        # was ever dispatched. Reset explicitly rather than relying on the
        # rebuilt list happening to be shorter than the old watermark.
        self._client._sent_watermark = 0
        self._client._sent_watermark_hash = None
        if at_step_limit is not None:
            # The step limit counts from the compacted conversation.
            self._runtime_state.message_count_offset = 0
        else:
            self._runtime_state.message_count_offset += n_archived - len(
                self._client._messages,
            )

        outer_handle_container: list = [None]
        _parent = cfg["parent_lineage"] or TOOL_LOOP_LINEAGE.get([])
        _lineage = [*_parent, cfg.get("loop_id", "compressed")]

        inner_kwargs = {
            k: v for k, v in cfg.items() if k not in ("parent_lineage", "tools")
        }
        # Parent context was captured in the first loop pass (often embedded
        # in the compressed system messages); re-injecting the full blob on
        # every restart could immediately re-trigger compression.
        inner_kwargs["parent_chat_context"] = None
        cfg["parent_chat_context"] = None

        async def _loop_wrapper():
            return await async_tool_loop_inner(
                self._client,
                restart_message,
                restart_tools,
                lineage=_lineage,
                interject_queue=self._queue,
                cancel_event=self._cancel_event,
                stop_event=self._stop_event,
                outer_handle_container=outer_handle_container,
                **inner_kwargs,
            )

        new_task = asyncio.create_task(_loop_wrapper(), name="ToolUseLoop")
        self._task = new_task
        outer_handle_container[0] = self

        LOGGER.info(
            f"{ICONS.get('completed', '✓')} [{self._log_label}] "
            f"Context compressed (pass #{self._compression.count}), "
            f"archived {n_archived} messages, new loop started.",
        )

    async def _compact_context(
        self,
        cfg: Optional[dict] = None,
    ) -> tuple[int, list[dict], dict, str, bool]:
        """Compress the context for a restart, leaving the transcript as it is.

        Returns ``(messages archived, restart messages, restart tools,
        restart message, forked)``; ``forked`` is ``True`` when the summary
        came from a fork (``UNIFY_CACHE_DISCIPLINE``), which keeps the
        client's system message. Raises when compression fails.
        """
        cfg = cfg if cfg is not None else self._loop_config
        if cfg is None:
            raise RuntimeError(
                "Cannot compress: loop config was not stored on the handle.",
            )
        n_archived = len(self._client.messages)
        forked = (
            await self._summarise_as_fork(cfg) if _cache_discipline.enabled() else None
        )
        if forked is not None:
            restart_messages, restart_tools, restart_message = forked
            return n_archived, restart_messages, restart_tools, restart_message, True
        result = await compress_and_rebuild(
            self._compression,
            self._client.messages,
            self._client.endpoint,
            dict(cfg["tools"]),
        )
        return (
            n_archived,
            result.system_msgs,
            result.tools,
            "Context was compressed. Continue from where you left off.",
            False,
        )

    async def _summarise_as_fork(
        self,
        cfg: dict,
    ) -> Optional[tuple[list[dict], dict, str]]:
        """Under UNIFY_CACHE_DISCIPLINE, compress by forking the conversation.

        The summary request is the last request this loop sent, unchanged,
        plus one appended instruction, so the provider serves everything but
        that instruction from its cache. A forced tool choice is sent as
        ``auto``, since the reply has to be text. The session then restarts
        from its own system prompt and the summary, with the same tools, so
        the fixed tool list and the system prompt stay a cached prefix.

        Returns ``(messages, tools, first_message)`` for the restart, or
        ``None`` -- and logs why -- when there is no recorded request or the
        fork yields no summary; the caller then compresses as shipped.
        """
        label = getattr(self, "_log_label", None) or self._loop_id
        last = _cache_discipline.last_sent_request(self._client)
        if last is None or not last.get("messages"):
            LOGGER.info(
                f"[{label}] compression fork skipped: no request was recorded; "
                "compressing as shipped",
            )
            return None
        request: dict[str, Any] = {
            "messages": [
                *last["messages"],
                {
                    "role": "user",
                    "content": _cache_discipline.COMPRESSION_FORK_INSTRUCTION,
                },
            ],
            "stateful": False,
            "return_full_completion": True,
        }
        if last.get("tools"):
            tool_choice = last.get("tool_choice")
            if _cache_discipline.is_forced_tool_choice(tool_choice):
                tool_choice = "auto"
            request["tools"] = last["tools"]
            request["tool_choice"] = tool_choice
        if cfg.get("prompt_caching") is not None:
            request["prompt_caching"] = cfg["prompt_caching"]
        try:
            fork = fork_llm_client(self._client, origin="compress_context")
            completion = await fork.generate(**request)
            _cache_discipline.log_cache_use(fork, completion, label=label)
            content = completion.choices[0].message.content
        except Exception as exc:
            LOGGER.warning(
                f"[{label}] compression fork failed ({type(exc).__name__}: "
                f"{exc}); compressing as shipped",
            )
            return None
        summary = _cache_discipline.completion_text(content)
        if not summary:
            LOGGER.info(
                f"[{label}] compression fork returned no summary text; "
                "compressing as shipped",
            )
            return None
        self._compression.count += 1
        history = self._client.messages or []
        system = [
            m for m in history[:1] if isinstance(m, dict) and m.get("role") == "system"
        ]
        restart_message = (
            f"{_COMPRESSED_HEADER}{summary}\n\n"
            "Context was compressed. Continue from where you left off."
        )
        # UNIFY_TRANSCRIPTS: the history the summary replaces goes to disk
        # first, and the summary names the file, so the model can read back
        # what it left out.
        session = transcripts.session_for_messages(self._client.messages)
        if session is not None:
            session.sync()
            restart_message += "\n\n" + session.pointer_line()
            transcripts.record_compaction(
                session,
                archived=len(history),
                pass_number=self._compression.count,
                context=[*system, {"role": "user", "content": restart_message}],
            )
        return system, dict(cfg["tools"]), restart_message

    def get_history(self) -> list[dict]:
        """The full LLM conversation history including assistant reasoning,
        tool calls and tool outputs; empty when no client is available."""
        if self._client is not None:
            return self._client.messages
        return []

    # ── bottom-up event APIs ---------------------------------------------------
    async def next_clarification(self) -> dict:
        """Await the next question a running call asks the requester."""
        return await self._clar_q.get()

    async def next_notification(self) -> dict:
        """Await the next notification pushed by a running call, or a
        persistent session's turn response."""
        return await self._notification_q.get()

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        """Put *answer* on the answer queue of the call that asked; a no-op
        when that call is no longer running."""
        channel = self._clarification_channels.get(call_id)
        if channel is None:
            LOGGER.info(
                f"[{self._log_label}] Clarification answer for {call_id!r} "
                "dropped: no running call asked it",
            )
            return
        await channel[1].put(answer)


def start_async_tool_loop(
    client: unillm.AsyncUnify,
    message: str | dict | list[str | dict],
    tools: Dict[str, Callable],
    *,
    loop_id: Optional[str] = None,
    parent_lineage: Optional[list[str]] = None,
    max_consecutive_failures: int = 3,
    prune_tool_duplicates=True,
    propagate_chat_context: ChatContextPropagation = ChatContextPropagation.LLM_DECIDES,
    parent_chat_context: Optional[list[dict]] = None,
    caller_description: Optional[str] = None,
    log_steps: Union[bool, str] = True,
    max_steps: Optional[int] = None,
    timeout: Optional[int] = None,
    raise_on_limit: bool = False,
    tool_policy: Optional[
        Union[
            Callable[
                [int, Dict[str, Callable]],
                Union[
                    Tuple[str, Dict[str, Callable]],
                    Tuple[str, Dict[str, Callable], Dict[str, Any]],
                ],
            ],
            Callable[
                [int, Dict[str, Callable], list[str]],
                Union[
                    Tuple[str, Dict[str, Callable]],
                    Tuple[str, Dict[str, Callable], Dict[str, Any]],
                ],
            ],
        ]
    ] = None,
    preprocess_msgs: Optional[Callable[[list[dict]], list[dict]]] = None,
    response_format: Optional[Any] = None,
    max_parallel_tool_calls: Optional[int] = None,
    persist: bool = False,
    prompt_caching: Optional["PromptCacheParam"] = None,
    time_awareness: bool = False,
    enable_compression: bool = True,
    extra_compression_tools: Optional[list[str]] = None,
    clarification_queues: Optional[Tuple["asyncio.Queue", "asyncio.Queue"]] = None,
    on_clarification_request: Optional[Callable[[str], Any]] = None,
    on_clarification_answer: Optional[Callable[[str], Any]] = None,
    on_notify: Optional[Callable[[str], Any]] = None,
    fixed_tools_schema: Optional[list[dict]] = None,
    first_message_context: Optional[str] = None,
    compression_tools_on_demand: bool = False,
    reply_channel: bool = False,
    on_turn_boundary: Optional[Callable[[], Awaitable[Optional[str]]]] = None,
    bind_request: bool = False,
) -> AsyncToolLoopHandle:
    """
    Run ``async_tool_loop_inner`` in its own task and return a handle for it.

    The loop runs one model turn at a time and each tool call to completion,
    in call order; see ``async_tool_loop_inner`` for every parameter.

    Parameters
    ----------
    log_steps : bool | str, default True
        Controls verbosity of step logging to `LOGGER`:
          - False: no logging
          - True: log everything except system messages
          - "full": log everything including system messages

    first_message_context : str | None, default None
        Text that opens the loop's first user message, and the message that
        restarts it after compression (see ``async_tool_loop_inner``).

    compression_tools_on_demand, reply_channel, bind_request : bool
        See ``async_tool_loop_inner``; the defaults are as shipped.

    on_turn_boundary : optional coroutine function
        See ``async_tool_loop_inner``; ``None`` (as shipped) adds nothing.

    timeout : int | None, default None
        Activity-based timeout in seconds. When ``None`` (default), no
        timeout is enforced.

    raise_on_limit : bool, default False
        If ``True``, raises ``asyncio.TimeoutError`` or ``RuntimeError``
        when the timeout or max_steps limit is exceeded. If ``False``,
        the loop terminates gracefully with a summary message.

    persist : bool, default False
        If ``True``, the loop does not terminate when the LLM produces content
        without tool calls. It parks until the next message arrives through
        ``handle.submit()`` and takes it as its next request. The loop only
        terminates when stopped via ``handle.stop()``.

    time_awareness : bool, default False
        If ``True``, a time-context system message is injected into the
        conversation and tool results carry their timing.
    """
    # One stable loop_id shared by the handle and the inner loop.
    if loop_id is not None:
        client.set_origin(loop_id)
    loop_id = loop_id if loop_id is not None else short_id()
    interject_queue: asyncio.Queue[dict | str] = asyncio.Queue()
    cancel_event = asyncio.Event()
    stop_event = asyncio.Event()
    runtime_state = ToolLoopRuntimeState()
    # UNIFY_BIND_REQUEST=on: the loop's current request, kept by the handle
    # so a loop restarted after compression keeps it.
    request_slot = _bound_request.new_slot(bind_request)

    # Mutable container through which the inner loop reaches the outer handle
    # once it exists.
    outer_handle_container: list = [None]

    _parent = (
        parent_lineage if parent_lineage is not None else TOOL_LOOP_LINEAGE.get([])
    )
    _lineage = [*_parent, loop_id]

    async def _loop_wrapper():
        return await async_tool_loop_inner(
            client,
            message,
            tools,
            loop_id=loop_id,
            lineage=_lineage,
            interject_queue=interject_queue,
            cancel_event=cancel_event,
            stop_event=stop_event,
            max_consecutive_failures=max_consecutive_failures,
            prune_tool_duplicates=prune_tool_duplicates,
            propagate_chat_context=propagate_chat_context,
            parent_chat_context=parent_chat_context,
            caller_description=caller_description,
            log_steps=log_steps,
            max_steps=max_steps,
            timeout=timeout,
            raise_on_limit=raise_on_limit,
            tool_policy=tool_policy,
            preprocess_msgs=preprocess_msgs,
            outer_handle_container=outer_handle_container,
            response_format=response_format,
            max_parallel_tool_calls=max_parallel_tool_calls,
            persist=persist,
            prompt_caching=prompt_caching,
            time_awareness=time_awareness,
            enable_compression=enable_compression,
            extra_compression_tools=extra_compression_tools,
            clarification_queues=clarification_queues,
            on_clarification_request=on_clarification_request,
            on_clarification_answer=on_clarification_answer,
            on_notify=on_notify,
            fixed_tools_schema=fixed_tools_schema,
            first_message_context=first_message_context,
            runtime_state=runtime_state,
            compression_tools_on_demand=compression_tools_on_demand,
            reply_channel=reply_channel,
            on_turn_boundary=on_turn_boundary,
            bind_request=request_slot,
        )

    task = asyncio.create_task(_loop_wrapper(), name="ToolUseLoop")

    handle = AsyncToolLoopHandle(
        task=task,
        interject_queue=interject_queue,
        cancel_event=cancel_event,
        stop_event=stop_event,
        client=client,
        loop_id=loop_id,
        response_format=response_format,
        runtime_state=runtime_state,
    )

    # _restart_with_compressed_context re-creates the loop from this config
    # with identical settings after compression.
    handle._loop_config = {
        "loop_id": loop_id,
        "parent_lineage": list(_parent),
        "tools": dict(tools),
        "max_consecutive_failures": max_consecutive_failures,
        "prune_tool_duplicates": prune_tool_duplicates,
        "propagate_chat_context": propagate_chat_context,
        "parent_chat_context": parent_chat_context,
        "caller_description": caller_description,
        "log_steps": log_steps,
        "max_steps": max_steps,
        "timeout": timeout,
        "raise_on_limit": raise_on_limit,
        "tool_policy": tool_policy,
        "preprocess_msgs": preprocess_msgs,
        "response_format": response_format,
        "max_parallel_tool_calls": max_parallel_tool_calls,
        "persist": persist,
        "prompt_caching": prompt_caching,
        "time_awareness": time_awareness,
        "enable_compression": enable_compression,
        "extra_compression_tools": extra_compression_tools,
        "clarification_queues": clarification_queues,
        "on_clarification_request": on_clarification_request,
        "on_clarification_answer": on_clarification_answer,
        "on_notify": on_notify,
        "runtime_state": runtime_state,
        "fixed_tools_schema": fixed_tools_schema,
        "first_message_context": first_message_context,
        "compression_tools_on_demand": compression_tools_on_demand,
        "reply_channel": reply_channel,
        "on_turn_boundary": on_turn_boundary,
        "bind_request": request_slot,
    }

    with suppress(Exception):
        handle._lineage = list(_lineage)  # type: ignore[attr-defined]

    # The inner loop reaches the handle through this container.
    outer_handle_container[0] = handle
    return handle
