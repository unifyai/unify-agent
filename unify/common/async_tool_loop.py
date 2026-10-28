import asyncio
import unillm
from abc import ABC, abstractmethod
from contextlib import suppress
from typing import (
    Optional,
    Awaitable,
    Dict,
    Callable,
    Any,
    TYPE_CHECKING,
)
from ..logger import LOGGER
from unify.common.hierarchical_logger import ICONS
from .llm_helpers import short_id
from .llm_client import count_requester_message, fork_llm_client
from ._async_tool import cache_discipline as _cache_discipline
from unify import transcripts
from ._async_tool.loop_config import TOOL_LOOP_LINEAGE
from ._async_tool.loop import (
    ToolLoopRuntimeState,
    async_tool_loop_inner,
)
from ._async_tool.context_compression import (
    _COMPRESSED_HEADER,
    _COMPRESSION_SIGNAL,
    FORK_SUMMARY,
    CompressionState,
    compress_and_rebuild,
)

if TYPE_CHECKING:
    from unillm.types import PromptCacheParam


# ── Handles ──────────────────────────────────────────────────────────────────


class ToolLoopHandle(ABC):
    """What a caller holds for a running tool loop.

    ``result()`` waits for the final answer, ``done()`` says whether the loop
    has ended, ``stop(reason)`` ends it (cancelling the model call or tool
    call in flight), and ``submit(text)`` queues the requester's next message,
    which the loop appends at its next turn boundary (a persistent session's
    next request). Only ``stop`` (and, for a persistent session, its request's
    ``cancel_request``) interrupts a model call or a tool call.
    ``next_notification()`` carries the turn responses of a persistent session.
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

    @abstractmethod
    async def cancel_request(self, reason: Optional[str] = None) -> bool:
        """End the running request of a persistent session, keeping the
        session (``unify act --jsonl``'s ``{"cancel": true}``). Returns whether
        the cancel was delivered."""

    @abstractmethod
    async def next_notification(self) -> dict:
        """Await the next turn response of a persistent session."""


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

        self._notification_q: asyncio.Queue[dict] = asyncio.Queue()

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

    async def stop(self, reason: Optional[str] = None) -> None:
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
        # A persistent loop takes the cancel at once: the model call or tool
        # call in flight is cancelled (a call is answered as cancelled) and
        # the request ends in its response. A parked loop has no request
        # running, and a loop that is not persistent is stopped with stop().
        # Nothing after a stop.
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

        ``_compact_context`` makes the restart; this method handles the loop
        lifecycle: replacing client messages, creating a new
        ``asyncio.Task``, and swapping the task reference so the handle's
        methods target the new loop.
        """
        cfg = self._loop_config
        if cfg is None:
            raise RuntimeError(
                "Cannot compress: loop config was not stored on the handle.",
            )

        n_archived, restart_messages, restart_tools, restart_message, forked = (
            await self._compact_context(cfg)
        )
        self._client._messages = restart_messages
        if not forked:
            self._client._system_message = None

        # A compression rebuild is a deliberate full-cache sacrifice: the
        # transcript it replaces no longer exists, so nothing in the new one
        # was ever dispatched. Reset explicitly rather than relying on the
        # rebuilt list happening to be shorter than the old watermark.
        self._client._sent_watermark = 0
        self._client._sent_watermark_hash = None
        self._runtime_state.message_count_offset += n_archived - len(
            self._client._messages,
        )

        outer_handle_container: list = [None]
        _parent = cfg["parent_lineage"]
        _lineage = [*_parent, cfg.get("loop_id", "compressed")]

        inner_kwargs = {
            k: v for k, v in cfg.items() if k not in ("parent_lineage", "tools")
        }

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
        cfg: dict,
    ) -> tuple[int, list[dict], dict, str, bool]:
        """Compress the context for a restart, leaving the transcript as it is.

        Returns ``(messages archived, restart messages, restart tools,
        restart message, forked)``; ``forked`` is ``True`` when the summary
        came from a fork (``FORK_SUMMARY``), which keeps the client's system
        message. Raises when compression fails.
        """
        n_archived = len(self._client.messages)
        forked = await self._summarise_as_fork(cfg) if FORK_SUMMARY else None
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
        """Compress by forking the conversation (``FORK_SUMMARY``).

        The summary request is the last request this loop sent, unchanged,
        plus one appended instruction, so the provider serves everything but
        that instruction from its cache. The session then restarts
        from its own system prompt and the summary, with the same tools, so
        the fixed tool list and the system prompt stay a cached prefix.

        Returns ``(messages, tools, first_message)`` for the restart, or
        ``None`` -- and logs why -- when there is no recorded request or the
        fork yields no summary; the caller then rebuilds with the fallback compressor.
        """
        label = getattr(self, "_log_label", None) or self._loop_id
        last = _cache_discipline.last_sent_request(self._client)
        # The actor's loop compacted before its next model call, so what
        # followed the last request (its reply and the cells'
        # results) is sent after it, unchanged.
        state = getattr(self, "_runtime_state", None)
        tail = list(getattr(state, "in_code_tail", None) or [])
        if state is not None:
            state.in_code_tail = None
        if last is None or not last.get("messages"):
            LOGGER.info(
                f"[{label}] compression fork skipped: no request was recorded; "
                "compressing with the fallback compressor",
            )
            return None
        request: dict[str, Any] = {
            "messages": [
                *last["messages"],
                *tail,
                {
                    "role": "user",
                    "content": _cache_discipline.compression_fork_instruction(),
                },
            ],
            "stateful": False,
            "return_full_completion": True,
        }
        if last.get("tools"):
            request["tools"] = last["tools"]
            request["tool_choice"] = last.get("tool_choice")
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
                f"{exc}); compressing with the fallback compressor",
            )
            return None
        summary = _cache_discipline.completion_text(content)
        if not summary:
            LOGGER.info(
                f"[{label}] compression fork returned no summary text; "
                "compressing with the fallback compressor",
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
        # The history the summary replaces goes to disk first, and the summary names the file, so the model can read back
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

    async def next_notification(self) -> dict:
        """Await the next turn response of a persistent session."""
        return await self._notification_q.get()


def start_async_tool_loop(
    client: unillm.AsyncUnify,
    message: str | dict | list[dict],
    tools: Dict[str, Callable],
    *,
    loop_id: Optional[str] = None,
    log_steps: bool | str = True,
    max_steps: Optional[int] = None,
    timeout: Optional[int] = None,
    preprocess_msgs: Optional[Callable[[list[dict]], list[dict]]] = None,
    response_format: Optional[Any] = None,
    persist: bool = False,
    prompt_caching: Optional["PromptCacheParam"] = None,
    first_message_context: Optional[str] = None,
    on_turn_boundary: Optional[Callable[[], Awaitable[Optional[str]]]] = None,
    tools_in_code: bool = False,
) -> AsyncToolLoopHandle:
    """
    Run ``async_tool_loop_inner`` in its own task and return a handle for it.

    The loop runs one model turn at a time and each tool call to completion,
    in call order; see ``async_tool_loop_inner`` for every parameter. A loop
    that compacts its context (``tools_in_code``) is restarted by the handle
    from the summary, with the same parameters.
    """
    # One stable loop_id shared by the handle and the inner loop.
    if loop_id is not None:
        client.set_origin(loop_id)
    loop_id = loop_id if loop_id is not None else short_id()
    interject_queue: asyncio.Queue[dict | str] = asyncio.Queue()
    cancel_event = asyncio.Event()
    stop_event = asyncio.Event()
    runtime_state = ToolLoopRuntimeState()
    # The request the actor's loop starts with is the requester's first
    # message, counted for the request metadata headers (a restart after
    # compression is not started here, so it counts nothing).
    if tools_in_code:
        count_requester_message(client)

    # Mutable container through which the inner loop reaches the outer handle
    # once it exists.
    outer_handle_container: list = [None]

    _parent = TOOL_LOOP_LINEAGE.get([])
    _lineage = [*_parent, loop_id]
    settings = {
        "loop_id": loop_id,
        "log_steps": log_steps,
        "max_steps": max_steps,
        "timeout": timeout,
        "preprocess_msgs": preprocess_msgs,
        "response_format": response_format,
        "persist": persist,
        "prompt_caching": prompt_caching,
        "runtime_state": runtime_state,
        "first_message_context": first_message_context,
        "on_turn_boundary": on_turn_boundary,
        "tools_in_code": tools_in_code,
    }

    async def _loop_wrapper():
        return await async_tool_loop_inner(
            client,
            message,
            tools,
            lineage=_lineage,
            interject_queue=interject_queue,
            cancel_event=cancel_event,
            stop_event=stop_event,
            outer_handle_container=outer_handle_container,
            **settings,
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
        "parent_lineage": list(_parent),
        "tools": dict(tools),
        **settings,
    }

    with suppress(Exception):
        handle._lineage = list(_lineage)  # type: ignore[attr-defined]

    # The inner loop reaches the handle through this container.
    outer_handle_container[0] = handle
    return handle
