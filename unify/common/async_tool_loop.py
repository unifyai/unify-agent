import asyncio
import unillm
import functools
import json
import time
from contextlib import suppress
from typing import (
    Optional,
    Awaitable,
    Dict,
    Callable,
    Tuple,
    Any,
    Union,
    Type,
    TYPE_CHECKING,
)
from ..logger import LOGGER
from unify.common.hierarchical_logger import ICONS
from .llm_helpers import short_id
from .llm_client import fork_llm_client, new_llm_client
from ._async_tool import cache_discipline as _cache_discipline
from unify import transcripts
from ._async_tool.loop_config import TOOL_LOOP_LINEAGE, _PENDING_LOOP_SUFFIX
from ._async_tool.event_bus_util import to_event_bus
from ..events.types.tool_loop import ToolLoopKind
from ._async_tool.loop import ToolLoopRuntimeState, async_tool_loop_inner
from ._async_tool.propagation_mode import ChatContextPropagation
from ._async_tool.context_compression import (
    _COMPRESSED_HEADER,
    _COMPRESSION_SIGNAL,
    CompressionState,
    compress_and_rebuild,
)
from .context_dump import make_messages_safe_for_context_dump
from ._async_tool.transcript_ops import (
    _replace_runtime_parent_context,
    _transform_inner_roles,
    build_digest,
    make_read_child_message_tool,
)
from ._async_tool.multi_handle import (
    MultiHandleCoordinator,
    MultiRequestHandle,
)
from ._async_tool.tagging import tag_message_with_request

if TYPE_CHECKING:
    from unillm.types import PromptCacheParam

_STEERING_ACTION_KIND: dict[str, str] = {
    "pause": ToolLoopKind.STEERING_PAUSE,
    "resume": ToolLoopKind.STEERING_RESUME,
    "stop": ToolLoopKind.STEERING_STOP,
}


# ── Inspection prompt ────────────────────────────────────────────────────────
# ``ask()`` spawns a read-only loop whose system message carries the inspected
# transcript in one of two shapes: a completed handle embeds a byte-stable
# digest (built once, cached, so repeat asks hit the provider's prefix cache)
# plus a drill-down tool; a still-running handle embeds a live snapshot.

_DIGEST_SYSTEM_HEADER = (
    "You are inspecting a COMPLETED tool-use conversation to answer a question "
    "about it.\n\n"
    "## Digest\n"
    "The JSON below is a mechanically-built, compact summary of the completed "
    "run (not the full transcript): the original request, each tool call in "
    "execution order (its name, its `thought` argument when the tool supplied "
    "one, a preview of its result, the result's size in bytes, and the message "
    "`idx` it corresponds to), source URLs seen across the run, and the final "
    "result. On a long run, the middle turns may be replaced by a single "
    '`{"elided": true, ...}` marker giving their count and idx range — those '
    "turns still exist and are still reachable, just not listed individually.\n\n"
    "If the digest does not contain enough detail to answer precisely, call "
    "`read_child_message(idx)` with the `idx` of any turn (or any idx inside "
    "an elided range) to fetch that exact message from the completed "
    "transcript verbatim (compact-serialized, capped at 32KB)."
)

_LIVE_SYSTEM_HEADER = (
    "You are inspecting a running tool-use conversation to answer a question about it."
)

_LIVE_TRANSCRIPT_DESCRIPTION = (
    "This is the transcript of the tool/loop you are being asked about. "
    "Messages use 'inner_user' and 'inner_assistant' roles to clearly "
    "distinguish them from your current conversation. "
    "Use this to answer the user's question about the current state or progress."
)

_LIVE_TRANSCRIPT_PARENT_NOTE = (
    " Note: this is separate from the Parent Chat Context that may "
    "appear below — that context shows the broader conversation that "
    "led to this request, while this transcript is what you are "
    "answering questions about."
)

_INNER_LOOP_TOOLS_HINT = (
    "You have access to `ask_*` tools that query inner tool loops for detailed information. "
    "Each inner tool loop has its own transcript that may contain details NOT visible in the "
    "context above. If that context does not contain enough information "
    "to answer the question — for example if a tool's result only shows a placeholder or "
    "summary — you MUST call the corresponding `ask_*` tool to get details from that "
    "tool's own internal context. Only answer directly from the context above when it clearly "
    "contains the specific information being asked about."
)

_INSPECTION_FOOTER = (
    "Answer the user's follow-up question using the context above and any tools exposed to you.",
    "Do not ask the user questions or request clarification. If information is missing,",
    "state what is known and, if helpful, briefly note assumptions. Respond in a single, concise paragraph.",
)


def _inflight_status_entry(elapsed: float) -> dict:
    """Final transcript entry for a loop that is waiting on an LLM response."""
    return {
        "role": "system",
        "_loop_status": True,
        "content": (
            "STATUS: the inspected loop is currently waiting on "
            f"an in-flight LLM request that started {elapsed:.0f} "
            "seconds ago and has not returned yet. The transcript "
            "ends here because the model is still thinking, not "
            "because a tool is stuck or anything failed. Answer "
            "progress questions from this fact first; only reach "
            "for inspection tools if the question is about "
            "something this does not explain."
        ),
    }


def _inspection_prompt(context_parts: list[str], *, has_inner_tools: bool) -> str:
    """System message for the inspection loop: the transcript context, a hint
    about ``ask_*`` tools when inner loops are reachable, and the answer
    instructions. Whether inner tools exist is fixed once a handle has
    completed, so the digest branch stays byte-stable across asks."""
    parts = [*context_parts]
    if has_inner_tools:
        parts.extend(["", "## Inner Loop Tools", _INNER_LOOP_TOOLS_HINT])
    parts.extend(["", *_INSPECTION_FOOTER])
    return "\n".join(parts)


def _inspection_failed_answer(exc: Exception) -> str:
    """Answer returned when the inspection loop itself fails.

    A read-only inspection must never surface as the inspected work failing:
    the inspection loop runs with ``max_consecutive_failures=1``, so one flaky
    introspection tool would otherwise raise out of ``result()`` and be
    mistaken for the parent task crashing.
    """
    return (
        "I couldn't determine the progress right now — the read-only "
        f"inspection step itself failed ({type(exc).__name__}: {exc}). "
        "This does NOT mean the underlying task failed; it may still be "
        "running. Do not treat this as a task failure or stop the task on "
        "account of it — just try checking again shortly."
    )


# ── Handles ──────────────────────────────────────────────────────────────────
from abc import ABC, abstractmethod


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


class AsyncToolLoopHandle(SteerableToolHandle):
    """Returned by ``start_async_tool_loop``: steers the running loop and
    answers read-only questions about it."""

    def __init__(
        self,
        *,
        task: asyncio.Task,
        interject_queue: asyncio.Queue[dict | str],
        cancel_event: asyncio.Event,
        stop_event: asyncio.Event,
        pause_event: Optional[asyncio.Event] = None,
        client: "unillm.AsyncUnify | None" = None,
        loop_id: str = "",
        initial_user_message: Optional[Any] = None,
        response_format: Optional[Any] = None,
        runtime_state: Optional[ToolLoopRuntimeState] = None,
    ):
        self._task = task
        self._queue = interject_queue
        self._cancel_event = cancel_event
        self._stop_event = stop_event
        # "running" ⇢ Event set, "paused" ⇢ Event cleared
        self._pause_event = pause_event or asyncio.Event()
        self._client = client
        self._pause_event.set()
        self._loop_id: str = loop_id
        # Log label with the 4-hex suffix, set by the inner loop once it
        # builds its LoopConfig; the bare loop_id until then.
        self._log_label: str = loop_id
        self._loop_cfg: Optional[Any] = None
        # Only the top-level handle emits the public stop log; nested and
        # adopted handles keep False to avoid duplicate logging.
        self._is_root_handle: bool = False
        # When set, result() parses the loop's raw JSON string into this
        # Pydantic model.
        self._response_format: Optional[Any] = response_format

        # What the end user would see: the original prompt, interjections and
        # ask questions (user), and ask answers (assistant).
        self._user_visible_history: list[dict] = []
        if initial_user_message:
            self._user_visible_history.append(
                {"role": "user", "content": initial_user_message},
            )

        self._clar_q: asyncio.Queue[dict] = asyncio.Queue()
        self._notification_q: asyncio.Queue[dict] = asyncio.Queue()

        self._compression = CompressionState()
        self._loop_config: Optional[dict] = None
        self._runtime_state = runtime_state or ToolLoopRuntimeState()

        # digest()'s cached text and the sanitized transcript snapshot it was
        # built from, which backs the read_child_message(idx) drill-down.
        self._digest_cache: Optional[str] = None
        self._digest_messages: Optional[list[dict]] = None

    def _append_user_visible_user(
        self,
        message: str,
        _parent_chat_context_cont: list[dict] | None,
    ) -> None:
        with suppress(Exception):
            if _parent_chat_context_cont is not None:
                self._user_visible_history.append(
                    {
                        "role": "user",
                        "content": {
                            "message": message,
                            "_parent_chat_context_continued": _parent_chat_context_cont,
                        },
                    },
                )
            else:
                self._user_visible_history.append(
                    {"role": "user", "content": message},
                )

    def _append_user_visible_assistant(self, message: str) -> None:
        with suppress(Exception):
            self._user_visible_history.append(
                {"role": "assistant", "content": message},
            )

    async def _emit_steering_event(self, action: str, content: str = "") -> None:
        cfg = getattr(self, "_loop_cfg", None)
        if cfg is None:
            return
        msg = {
            "role": "system",
            "_steering": True,
            "_steering_action": action,
            "content": content,
        }
        _kind = _STEERING_ACTION_KIND.get(action)
        with suppress(Exception):
            await to_event_bus(msg, cfg, kind=_kind)

    async def _mirror(self, method: str, kwargs: dict) -> None:
        """Queue a synthetic helper tool_call so the inner loop records this
        steering call in its transcript without an LLM step."""
        try:
            await self._queue.put({"_mirror": {"method": method, "kwargs": kwargs}})
        except Exception:
            pass

    # ── ask: read-only inspection ------------------------------------------------
    async def ask(
        self,
        question: str,
        *,
        _parent_chat_context: list[dict] | None = None,
        _propagate_chat_context: ChatContextPropagation = ChatContextPropagation.LLM_DECIDES,
        _return_reasoning_steps: bool = False,
        **kwargs,
    ) -> "SteerableToolHandle":
        _label = getattr(self, "_log_label", None) or self._loop_id
        LOGGER.info(f"{ICONS['clarification']} [{_label}] Ask requested: {question}")

        self._append_user_visible_user(question, _parent_chat_context)

        # done() picks the inspection shape: a completed handle gets the
        # cached digest, a still-running one the live snapshot. No await
        # precedes this check, so the task cannot finish between the check
        # and the branch it selects.
        completed = self.done()

        parent_chat_context_safe = make_messages_safe_for_context_dump(
            _parent_chat_context,
        )
        ask_tools, completed_askable_tools = self._inspection_tool_surface()

        inspection_client = new_llm_client(self._client_model())
        # An inspection belongs to the run it inspects, so it runs at that
        # run's effort -- not at the "high" a client named for a model gets.
        inspected_effort = self._client_reasoning_effort()
        if inspected_effort:
            inspection_client.set_reasoning_effort(inspected_effort)
        inspection_tools: dict = dict(ask_tools)

        if completed:
            # digest() must run first: it populates self._digest_messages,
            # which the drill-down tool captures.
            context_parts = [_DIGEST_SYSTEM_HEADER, "", self.digest()]
            inspection_tools["read_child_message"] = (
                self._make_read_child_message_tool()
            )
        else:
            context_parts = self._live_inspection_context(bool(_parent_chat_context))

        inspection_client.set_system_message(
            _inspection_prompt(context_parts, has_inner_tools=bool(ask_tools)),
        )

        helper_handle = await self._start_inspection_loop(
            inspection_client,
            question,
            inspection_tools,
            completed_askable_tools,
            parent_chat_context=(
                parent_chat_context_safe if _parent_chat_context else None
            ),
            propagate_chat_context=_propagate_chat_context,
            return_reasoning_steps=_return_reasoning_steps,
        )
        await self._mirror("ask", {"question": question, **(kwargs or {})})
        return helper_handle

    def _client_model(self) -> str | None:
        with suppress(Exception):
            if self._client is not None:
                return self._client.endpoint
        return None

    def _client_reasoning_effort(self) -> str | None:
        with suppress(Exception):
            if self._client is not None:
                effort = self._client.reasoning_effort
                return effort if isinstance(effort, str) and effort else None
        return None

    def _inspection_tool_surface(self) -> tuple[dict, dict]:
        """Snapshot the inspected loop's ``ask_*`` tools and its completed-
        askable registry.

        The inspection loop needs both: the ``ask_*`` tools let it propagate
        a question into a still-running inner handle, and the registry is
        required because the inspected transcript announces its finished
        steerable calls as "[askable <call_id>]", so the inspection loop's
        own ``ask_about_completed_tool`` must resolve those ids to the same
        handles rather than to its (empty) registry.
        """
        ask_tools: dict = {}
        with suppress(Exception):
            ask_tools = getattr(self._task, "get_ask_tools", lambda: {})()
        completed_askable_tools: dict = {}
        with suppress(Exception):
            completed_askable_tools = getattr(
                self._task,
                "get_completed_tool_metadata",
                lambda: {},
            )()
        return ask_tools, completed_askable_tools

    def _transcript_snapshot(self) -> list[dict]:
        """This loop's messages so far, deep-copied with image blobs redacted."""
        return make_messages_safe_for_context_dump(
            list(getattr(self._client, "messages", None) or []),
        )

    def _live_inspection_context(self, has_parent_context: bool) -> list[str]:
        """System-message parts carrying a still-running loop's transcript."""
        snapshot = self._transcript_snapshot()

        # Fresh parent context reaches the inspection loop through its own
        # header, so the stale copy embedded in the transcript is reduced to
        # a pointer. The structural marker stays so the model still sees
        # that, and where, the loop received parent context.
        if has_parent_context:
            snapshot = _replace_runtime_parent_context(snapshot)
        snapshot = _transform_inner_roles(snapshot)

        # A snapshot dead-ends silently while the loop waits on an LLM
        # response — there is no message for "the model has not answered
        # yet", so inspectors misread the pause as a stall and reach for
        # progress tools that cannot help. Surface the in-flight window
        # (stamped by ``generate_with_preprocess``) as an explicit final entry.
        with suppress(Exception):
            inflight_since = (
                getattr(self._client, "_llm_inflight_since", None)
                if self._client is not None
                else None
            )
            if inflight_since:
                elapsed = max(0.0, time.time() - float(inflight_since))
                snapshot = [*snapshot, _inflight_status_entry(elapsed)]

        description = _LIVE_TRANSCRIPT_DESCRIPTION
        if has_parent_context:
            description += _LIVE_TRANSCRIPT_PARENT_NOTE

        return [
            _LIVE_SYSTEM_HEADER,
            "",
            "## Inspected Loop Transcript",
            description,
            "",
            json.dumps(snapshot, separators=(",", ":")),
        ]

    async def _start_inspection_loop(
        self,
        inspection_client: "unillm.AsyncUnify",
        question: str,
        tools: dict,
        completed_askable_tools: dict,
        *,
        parent_chat_context: list[dict] | None,
        propagate_chat_context: ChatContextPropagation,
        return_reasoning_steps: bool,
    ) -> "SteerableToolHandle":
        """Start the stand-alone read-only loop that answers *question*.

        The returned handle's ``result()`` records the answer in the
        user-visible history, publishes the outgoing boundary event, and
        returns ``(answer, inspection_messages)`` when
        *return_reasoning_steps* is set.
        """
        from ..events.manager_event_logging import (
            new_call_id,
            publish_manager_method_event,
        )
        from secrets import token_hex

        # Loop identifier for logs, e.g. "Question(CodeActActor.act)".
        parent_label: str = "unknown"
        with suppress(Exception):
            parent_label = (
                getattr(self, "_log_label", None)
                or getattr(self, "_loop_id", "unknown")
                or "unknown"
            )
        loop_id_label = f"Question({parent_label})"

        # The inspection loop is a sibling of this loop, not a child: it
        # shares this loop's parent lineage so the action tree places it at
        # the same nesting level.
        parent_hierarchy = list(getattr(self, "_log_hierarchy", None) or [])
        sibling_lineage = parent_hierarchy[:-1] if len(parent_hierarchy) > 1 else []

        # Boundary ManagerMethod events give every ask() its own node in the
        # action tree, mirroring the execute_code/execute_function wrappers
        # in CodeActActor.
        call_id = new_call_id()
        suffix = token_hex(2)
        manager = (self._loop_id or "").split(".")[0] or "unknown"
        hierarchy = [*sibling_lineage, f"{loop_id_label}({suffix})"]

        await publish_manager_method_event(
            call_id,
            manager,
            "ask",
            phase="incoming",
            display_label="Answering question",
            question=question,
            hierarchy=hierarchy,
        )

        # The inner LoopConfig picks up the same suffix as the boundary event.
        suffix_token = _PENDING_LOOP_SUFFIX.set(suffix)
        try:
            helper_handle = start_async_tool_loop(
                inspection_client,
                question,
                tools,
                completed_askable_tools=completed_askable_tools,
                loop_id=loop_id_label,
                parent_lineage=sibling_lineage,
                parent_chat_context=parent_chat_context,
                propagate_chat_context=propagate_chat_context,
                prune_tool_duplicates=False,
                interrupt_llm_with_interjections=False,
                max_consecutive_failures=1,
            )
        finally:
            _PENDING_LOOP_SUFFIX.reset(suffix_token)

        original_result = helper_handle.result

        async def _result():
            try:
                answer = await original_result()
            except Exception as exc:
                answer = _inspection_failed_answer(exc)
            self._append_user_visible_assistant(answer)
            await publish_manager_method_event(
                call_id,
                manager,
                "ask",
                phase="outgoing",
                display_label="Answering question",
                answer=answer if isinstance(answer, str) else str(answer),
                hierarchy=hierarchy,
            )
            if return_reasoning_steps:
                return answer, inspection_client.messages
            return answer

        helper_handle.result = _result  # type: ignore[attr-defined]
        return helper_handle

    def digest(self) -> str:
        """Return a compact, byte-stable digest of this handle's completed run.

        Built once, mechanically, from the transcript and the task's result
        (see ``build_digest``) and cached, so every subsequent call — and
        every completed-``ask()`` that embeds it — returns byte-identical
        text, which is what lets a second question about the same handle hit
        the provider's prefix cache.

        If this handle's loop compressed its context, ``self._client.messages``
        already reflects the post-compression state by the time this can run:
        the wrapper task that adopted this handle (``ToolsData.adopt_nested``)
        awaits ``self.result()`` to completion, which resolves any compression
        restarts, before the handle is ever reachable as "completed".

        Raises
        ------
        asyncio.InvalidStateError
            If called before the handle has completed. ``ask()`` only reaches
            this on its completed branch; the guard exists so a premature
            direct call fails loudly instead of being mislabelled — without
            it, reading ``self._task.result()`` on a running task raises its
            own ``InvalidStateError``, which would be rendered as a false
            "this run ended with an error" outcome.
        """
        if self._digest_cache is not None:
            return self._digest_cache

        if not self.done():
            raise asyncio.InvalidStateError(
                "digest() is only available after completion; use the "
                "live-snapshot ask() path for a still-running handle.",
            )

        self._digest_messages = self._transcript_snapshot()

        request = None
        if self._user_visible_history:
            request = self._user_visible_history[0].get("content")
            if isinstance(request, dict):
                request = request.get("message")

        self._digest_cache = build_digest(
            self._digest_messages,
            request=request,
            final_result=self._final_result_text(),
        )
        return self._digest_cache

    def _final_result_text(self) -> Optional[str]:
        """The completed task's result as digest text, or ``None`` when the
        task holds no result yet.

        Reading the task directly, rather than guessing the answer's shape in
        the transcript, keeps the digest truthful for structured-output loops
        (whose answer went through a tool) and for runs that errored or were
        stopped.
        """
        try:
            raw = self._task.result()
        except asyncio.CancelledError:
            return "(this run was stopped before producing a result)"
        except asyncio.InvalidStateError:
            # digest()'s done() guard rules this out; re-raise rather than
            # let the generic handler below mislabel a not-actually-finished
            # task as one that errored.
            raise
        except Exception as exc:
            return f"(this run ended with an error: {type(exc).__name__}: {exc})"
        # `_COMPRESSION_SIGNAL` means no one has driven this task's result()
        # through a compression restart yet. The digest should be unreachable
        # in that state (see digest()), but fall back to the transcript rather
        # than surface the sentinel if it somehow is.
        if raw is _COMPRESSION_SIGNAL:
            return None
        if isinstance(raw, str):
            return raw
        return json.dumps(raw, separators=(",", ":"), default=str)

    def _make_read_child_message_tool(self) -> Callable:
        """Drill-down tool over the snapshot ``digest()`` built; ``ask()``
        calls ``digest()`` first."""
        return make_read_child_message_tool(self._digest_messages or [])

    # ── steering ---------------------------------------------------------------
    @functools.wraps(SteerableToolHandle.interject, updated=())
    async def interject(
        self,
        message: str,
        *,
        _parent_chat_context_cont: list[dict] | None = None,
        trigger_immediate_llm_turn: bool = True,
        **kwargs,
    ) -> None:
        _label = getattr(self, "_log_label", None) or self._loop_id
        LOGGER.debug(
            f"{ICONS['interjection']} [{_label}] Interject requested: {message}",
        )
        self._append_user_visible_user(message, _parent_chat_context_cont)

        payload = {
            "message": message,
            "_parent_chat_context_continued": _parent_chat_context_cont,
            "trigger_immediate_llm_turn": trigger_immediate_llm_turn,
            "suppress_response_notification": kwargs.get(
                "suppress_response_notification",
                False,
            ),
        }
        # put_nowait registers the interjection synchronously, before this
        # coroutine yields; otherwise a fast loop can finish its turn and
        # exit before seeing the queued item.
        self._queue.put_nowait(payload)

        await self._mirror("interject", {"message": message, **(kwargs or {})})

    @functools.wraps(SteerableToolHandle.stop, updated=())
    async def stop(
        self,
        reason: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Idempotent: a second stop neither logs nor re-signals.
        if self._cancel_event.is_set():
            return

        await self._emit_steering_event("stop", reason or "")
        # Un-pause so the inner loop can observe the stop immediately.
        with suppress(Exception):
            self._pause_event.set()
        # The mirror must be queued before cancel/stop are signalled: the
        # inner loop exits after processing it.
        try:
            self._queue.put_nowait(
                {
                    "_mirror": {
                        "method": "stop",
                        "kwargs": {
                            "reason": reason,
                            **(kwargs or {}),
                        },
                    },
                },
            )
        except Exception:
            pass
        with suppress(Exception):
            self._cancel_event.set()
        with suppress(Exception):
            self._stop_event.set()

    def _set_base_tool_pause_events(self, running: bool) -> None:
        """Toggle the pause events of base (non-steerable) tools directly.

        Steerable handles are paused and resumed through the mirror path,
        which synthesizes helper tool_calls in the transcript so the outer
        LLM sees that the inner tool was paused. Base tools have no handle —
        only a raw pause_event — and the mirror's dispatch would reach them
        too, but toggling here removes the latency window before the next
        loop iteration drains the mirror.
        """
        with suppress(Exception):
            task_info = getattr(self._task, "task_info", {})
            items = task_info.items() if isinstance(task_info, dict) else []
            for _t, _inf in items:
                if getattr(_inf, "handle", None) is not None:
                    continue
                ev = getattr(_inf, "pause_event", None)
                if ev is not None and hasattr(ev, "set" if running else "clear"):
                    with suppress(Exception):
                        if running:
                            ev.set()
                        else:
                            ev.clear()

    @functools.wraps(SteerableToolHandle.pause, updated=())
    async def pause(self, **kwargs) -> None:
        _label = getattr(self, "_log_label", None) or self._loop_id
        LOGGER.info(f"{ICONS['pause']} [{_label}] Pause requested")
        await self._emit_steering_event("pause")
        self._set_base_tool_pause_events(running=False)
        self._pause_event.clear()
        # The inner loop dispatches the mirrored pause to all children,
        # steerable and base alike.
        await self._mirror("pause", dict(kwargs or {}))

    @functools.wraps(SteerableToolHandle.resume, updated=())
    async def resume(self, **kwargs) -> None:
        _label = getattr(self, "_log_label", None) or self._loop_id
        LOGGER.info(f"{ICONS['resume']} [{_label}] Resume requested")
        await self._emit_steering_event("resume")
        self._set_base_tool_pause_events(running=True)
        self._pause_event.set()
        await self._mirror("resume", dict(kwargs or {}))

    @functools.wraps(SteerableToolHandle.done, updated=())
    def done(self) -> bool:
        return self._task.done()

    @functools.wraps(SteerableToolHandle.result, updated=())
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
        ``asyncio.Task``, and swapping the task reference so steering methods
        target the new loop.
        """
        cfg = self._loop_config
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
            self._client._messages = restart_messages
        else:
            result = await compress_and_rebuild(
                self._compression,
                self._client.messages,
                self._client.endpoint,
                dict(cfg["tools"]),
            )
            self._client._messages = result.system_msgs
            self._client._system_message = None
            restart_tools = result.tools
            restart_message = (
                "Context was compressed. Continue from where you left off."
            )

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
                pause_event=self._pause_event,
                outer_handle_container=outer_handle_container,
                **inner_kwargs,
            )

        new_task = asyncio.create_task(_loop_wrapper(), name="ToolUseLoop")

        with suppress(Exception):
            setattr(new_task, "task_info", {})
            setattr(new_task, "clarification_channels", {})
            setattr(new_task, "get_ask_tools", lambda: {})
            setattr(new_task, "get_completed_tool_metadata", lambda: {})

        self._task = new_task
        outer_handle_container[0] = self

        LOGGER.info(
            f"{ICONS.get('completed', '✓')} [{self._log_label}] "
            f"Context compressed (pass #{self._compression.count}), "
            f"archived {n_archived} messages, new loop started.",
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
    @functools.wraps(SteerableToolHandle.next_clarification, updated=())
    async def next_clarification(self) -> dict:
        """Await the next clarification event pushed by a running tool."""
        return await self._clar_q.get()

    @functools.wraps(SteerableToolHandle.next_notification, updated=())
    async def next_notification(self) -> dict:
        """Await the next notification pushed by a running tool."""
        return await self._notification_q.get()

    @functools.wraps(SteerableToolHandle.answer_clarification, updated=())
    async def answer_clarification(self, call_id: str, answer: str) -> None:
        await self._mirror("clarify", {"call_id": call_id, "answer": answer})


def start_async_tool_loop(
    client: unillm.AsyncUnify,
    message: str | dict | list[str | dict],
    tools: Dict[str, Callable],
    *,
    loop_id: Optional[str] = None,
    parent_lineage: Optional[list[str]] = None,
    max_consecutive_failures: int = 3,
    prune_tool_duplicates=True,
    interrupt_llm_with_interjections: bool = True,
    interrupt_llm_on_tool_completion: bool = True,
    propagate_chat_context: ChatContextPropagation = ChatContextPropagation.LLM_DECIDES,
    parent_chat_context: Optional[list[dict]] = None,
    caller_description: Optional[str] = None,
    log_steps: Union[bool, str] = True,
    max_steps: Optional[int] = None,
    timeout: Optional[int] = None,
    raise_on_limit: bool = False,
    include_class_in_dynamic_tool_names: bool = False,
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
    handle_cls: Optional[Type[AsyncToolLoopHandle]] = None,
    persist: bool = False,
    multi_handle: bool = False,
    prompt_caching: Optional["PromptCacheParam"] = None,
    time_awareness: bool = False,
    extra_ask_tools: Optional[Dict[str, Callable]] = None,
    completed_askable_tools: Optional[Dict[str, dict]] = None,
    enable_compression: bool = True,
    extra_compression_tools: Optional[list[str]] = None,
    clarification_queues: Optional[Tuple["asyncio.Queue", "asyncio.Queue"]] = None,
    on_clarification_request: Optional[Callable[[str], Any]] = None,
    on_clarification_answer: Optional[Callable[[str], Any]] = None,
    on_notify: Optional[Callable[[str], Any]] = None,
    fixed_tools_schema: Optional[list[dict]] = None,
    first_message_context: Optional[str] = None,
    steering_tools: bool = True,
    compression_tools_on_demand: bool = False,
) -> AsyncToolLoopHandle:
    """
    Run ``async_tool_loop_inner`` in its own task and return a handle for
    live interaction.

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

    steering_tools, compression_tools_on_demand : bool
        See ``async_tool_loop_inner``; the defaults are as shipped.

    timeout : int | None, default None
        Activity-based timeout in seconds. When ``None`` (default), no
        timeout is enforced.

    raise_on_limit : bool, default False
        If ``True``, raises ``asyncio.TimeoutError`` or ``RuntimeError``
        when the timeout or max_steps limit is exceeded. If ``False``,
        the loop terminates gracefully with a summary message.

    interrupt_llm_on_tool_completion : bool, default True
        Whether a background tool finishing mid-reasoning cancels the in-flight
        LLM step so the next one starts with that result. Keeping this ``True``
        is what guarantees the model never reasons on stale context.

        Set it ``False`` only for loops that fan out to many short tools whose
        completions are staggered: there, each completion kills a reasoning step
        the provider has already billed, so a batch of N tools discards up to
        N-1 paid steps. With it ``False`` the step is allowed to finish and be
        used, and the tool result reaches the model on the following turn
        instead. That trades a turn of freshness for the cost of the step, which
        is only worth it where nobody is waiting on the latency.
        ``UNIFY_BATCH_WAKE`` turns it ``False`` for every loop.

    persist : bool, default False
        If ``True``, the loop does not terminate when the LLM produces content
        without tool calls. Instead, it blocks waiting for the next interjection
        via ``handle.interject()``. When an interjection arrives, the LLM is
        granted another turn. This enables a single persistent loop that can
        process multiple events over time. The loop only terminates when
        explicitly stopped via ``handle.stop()`` or cancelled.

    multi_handle : bool, default False
        If ``True``, enables multi-handle mode where the loop can serve multiple
        concurrent requests. Each request is identified by a request_id, and the
        LLM uses ``final_answer(request_id, answer)`` to complete specific requests.
        The returned handle supports ``add_request(message)`` to add new requests
        to the running loop. Interjections are tagged with request IDs so the LLM
        knows which request they belong to. The loop terminates when all requests
        are completed/cancelled (unless persist=True).

    time_awareness : bool, default True
        If ``True``, a time-context system message is injected into the
        conversation and updated after each tool completion, giving the LLM
        awareness of wall-clock time and tool execution durations.  If
        ``False``, the time-context table is omitted entirely.
    """
    # One stable loop_id shared by the handle and the inner loop.
    if loop_id is not None:
        client.set_origin(loop_id)
    loop_id = loop_id if loop_id is not None else short_id()
    interject_queue: asyncio.Queue[dict | str] = asyncio.Queue()
    cancel_event = asyncio.Event()
    stop_event = asyncio.Event()
    pause_event = asyncio.Event()
    pause_event.set()  # start un-paused
    runtime_state = ToolLoopRuntimeState()

    # Mutable container through which the inner loop reaches the outer handle
    # once it exists.
    outer_handle_container: list = [None]

    _parent = (
        parent_lineage if parent_lineage is not None else TOOL_LOOP_LINEAGE.get([])
    )
    _lineage = [*_parent, loop_id]

    multi_handle_coordinator: MultiHandleCoordinator | None = None
    if multi_handle:
        # clarification_channels is set on the task later; the coordinator
        # starts with a placeholder dict that is swapped once the task has it.
        _clarification_channels_ref: dict = {}
        multi_handle_coordinator = MultiHandleCoordinator(
            interject_queue=interject_queue,
            clarification_channels=_clarification_channels_ref,
            persist=persist,
        )
        multi_handle_coordinator.register_request()

    async def _loop_wrapper():
        try:
            return await async_tool_loop_inner(
                client,
                (
                    message
                    if not multi_handle
                    else tag_message_with_request(
                        message if isinstance(message, str) else str(message),
                        0,
                    )
                ),
                tools,
                loop_id=loop_id,
                lineage=_lineage,
                interject_queue=interject_queue,
                cancel_event=cancel_event,
                stop_event=stop_event,
                pause_event=pause_event,
                max_consecutive_failures=max_consecutive_failures,
                prune_tool_duplicates=prune_tool_duplicates,
                interrupt_llm_with_interjections=interrupt_llm_with_interjections,
                interrupt_llm_on_tool_completion=interrupt_llm_on_tool_completion,
                propagate_chat_context=propagate_chat_context,
                parent_chat_context=parent_chat_context,
                caller_description=caller_description,
                log_steps=log_steps,
                max_steps=max_steps,
                timeout=timeout,
                raise_on_limit=raise_on_limit,
                include_class_in_dynamic_tool_names=include_class_in_dynamic_tool_names,
                tool_policy=tool_policy,
                preprocess_msgs=preprocess_msgs,
                outer_handle_container=outer_handle_container,
                response_format=response_format,
                max_parallel_tool_calls=max_parallel_tool_calls,
                persist=persist,
                multi_handle_coordinator=multi_handle_coordinator,
                prompt_caching=prompt_caching,
                time_awareness=time_awareness,
                extra_ask_tools=extra_ask_tools,
                completed_askable_tools=completed_askable_tools,
                enable_compression=enable_compression,
                extra_compression_tools=extra_compression_tools,
                clarification_queues=clarification_queues,
                on_clarification_request=on_clarification_request,
                on_clarification_answer=on_clarification_answer,
                on_notify=on_notify,
                fixed_tools_schema=fixed_tools_schema,
                first_message_context=first_message_context,
                runtime_state=runtime_state,
                steering_tools=steering_tools,
                compression_tools_on_demand=compression_tools_on_demand,
            )
        except asyncio.CancelledError:
            raise

    task = asyncio.create_task(_loop_wrapper(), name="ToolUseLoop")

    # Introspection surfaces are available on the wrapper task immediately;
    # the inner loop rebinds them to its live dicts once running.
    try:  # pragma: no cover
        setattr(task, "task_info", {})  # asyncio.Task -> ToolCallMetadata
        setattr(task, "clarification_channels", {})  # call_id -> (up_q, down_q)
        setattr(task, "get_ask_tools", lambda: {})  # snapshot of ask_* dynamic tools
        setattr(
            task,
            "get_completed_tool_metadata",
            lambda: {},
        )  # completed tool metadata with handle refs
    except Exception:
        pass

    # The handle's initial user message, from whichever input form was given.
    init_content = None
    if isinstance(message, dict):
        init_content = message.get("content")
    elif isinstance(message, list):
        for m in message:
            if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
                init_content = m["content"]
                break
            if isinstance(m, str):
                init_content = m
                break
    else:
        init_content = message

    HandleType = handle_cls or AsyncToolLoopHandle
    handle = HandleType(
        task=task,
        interject_queue=interject_queue,
        cancel_event=cancel_event,
        stop_event=stop_event,
        pause_event=pause_event,
        client=client,
        loop_id=loop_id,
        initial_user_message=init_content,
        response_format=response_format,
    )
    handle._runtime_state = runtime_state

    # _restart_with_compressed_context re-creates the loop from this config
    # with identical settings after compression.
    handle._loop_config = {
        "loop_id": loop_id,
        "parent_lineage": list(_parent),
        "tools": dict(tools),
        "max_consecutive_failures": max_consecutive_failures,
        "prune_tool_duplicates": prune_tool_duplicates,
        "interrupt_llm_with_interjections": interrupt_llm_with_interjections,
        "interrupt_llm_on_tool_completion": interrupt_llm_on_tool_completion,
        "propagate_chat_context": propagate_chat_context,
        "parent_chat_context": parent_chat_context,
        "caller_description": caller_description,
        "log_steps": log_steps,
        "max_steps": max_steps,
        "timeout": timeout,
        "raise_on_limit": raise_on_limit,
        "include_class_in_dynamic_tool_names": include_class_in_dynamic_tool_names,
        "tool_policy": tool_policy,
        "preprocess_msgs": preprocess_msgs,
        "response_format": response_format,
        "max_parallel_tool_calls": max_parallel_tool_calls,
        "persist": persist,
        "multi_handle_coordinator": multi_handle_coordinator,
        "prompt_caching": prompt_caching,
        "time_awareness": time_awareness,
        "extra_ask_tools": extra_ask_tools,
        "completed_askable_tools": completed_askable_tools,
        "enable_compression": enable_compression,
        "extra_compression_tools": extra_compression_tools,
        "clarification_queues": clarification_queues,
        "on_clarification_request": on_clarification_request,
        "on_clarification_answer": on_clarification_answer,
        "on_notify": on_notify,
        "runtime_state": runtime_state,
        "fixed_tools_schema": fixed_tools_schema,
        "first_message_context": first_message_context,
        "steering_tools": steering_tools,
        "compression_tools_on_demand": compression_tools_on_demand,
    }

    with suppress(Exception):
        handle._lineage = list(_lineage)  # type: ignore[attr-defined]

    # The root handle is the one that emits the public stop log.
    with suppress(Exception):
        handle._is_root_handle = True  # type: ignore[attr-defined]

    # Lets the inner coroutine switch steering to the outer handle when a
    # nested handle requests pass-through behaviour.
    outer_handle_container[0] = handle

    if multi_handle and multi_handle_coordinator is not None:
        # The task attribute is set after creation, so the coordinator's
        # channel reference is swapped in here.
        try:
            multi_handle_coordinator._clarification_channels = getattr(
                task,
                "clarification_channels",
                {},
            )
        except Exception:
            pass

        request_handle = MultiRequestHandle(
            request_id=0,
            coordinator=multi_handle_coordinator,
            loop_id=loop_id,
        )

        state = multi_handle_coordinator.registry.get(0)
        if state:
            state.handle_ref = request_handle

        return request_handle  # type: ignore[return-value]

    return handle
