import asyncio
import inspect
import json
import traceback
import dataclasses
import time


from typing import (
    Callable,
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
    emit_completion_pair,
    is_mutable,
    loop_user_notice,
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


# Sentinel for bare top-level handles (no label needed).
_HANDLE_SENTINEL = "<steerable handle — now in-flight>"

# Tells the model what the end-user can and cannot see, so it does not
# mistake automatically-appended [progress]/[clarification] tail messages
# for a real user interjection it is told elsewhere to "consider and
# incorporate". Injected on demand (see _ensure_visibility_guidance_injected
# and the loop's own interjection-triggered call into it) rather than at
# loop start, so the model stays on task until a trigger actually fires.
USER_VISIBILITY_GUIDANCE = (
    "## User Visibility Context\n"
    "IMPORTANT: The end-user who initiated this conversation can ONLY see:\n"
    "1. Their original request and any follow-up messages they send (interjections)\n"
    "2. Any notifications you emit (status updates, progress indicators, etc.)\n"
    "3. Any clarification requests you send asking for more information\n"
    "4. Your FINAL plain-text response at the end of this tool-use session\n\n"
    "The user CANNOT see:\n"
    "- Any intermediate tool calls you make\n"
    "- Any tool results or outputs\n"
    "- Any assistant messages that include tool_calls\n\n"
    "When the user sends follow-up messages (interjections) during your tool-use "
    "session, these appear as regular user messages. Consider and incorporate ALL "
    "user interjections in your final response. Later interjections should override "
    "earlier ones if there are any conflicting comments or requests.\n\n"
    "EXCEPTION: user-role messages prefixed with `[progress <call_id>]` are NOT "
    "interjections from the user — they are status updates a running tool appends "
    "automatically to report its in-flight progress. Do not treat them as requests "
    "to incorporate or respond to.\n\n"
    "user-role messages prefixed with `[clarification <call_id>]` are also not from "
    "the user — they are a pending tool asking you a question it needs answered to "
    'continue. Answer them by calling steer(call_id=<call_id>, action="clarify", '
    "payload=<answer>), not by responding to the user or treating the question "
    "itself as a request.\n\n"
    "user-role messages prefixed with `[steerable <call_id>]` or `[askable <call_id>]` "
    "are also not from the user — they are lifecycle announcements the loop appends "
    "automatically when a call becomes steerable or, once it finishes, becomes askable "
    "via `ask_about_completed_tool`. Do not treat them as requests to incorporate or "
    "respond to; they exist only so you know which call_id to pass to `steer` or "
    "`ask_about_completed_tool`."
)


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


def _handle_label_sentinel(label: str) -> str:
    """Labeled sentinel for a handle inside a composite return."""
    return f"[{label}: steerable]"


@dataclasses.dataclass
class _MultiHandleState:
    """Shared state for multiple handles spawned from one tool return."""

    parent_call_id: str
    parent_name: str
    placeholder_msg: dict
    template: Any  # cleaned structure with labeled sentinels
    results: dict  # label -> raw result (None while pending)

    async def record_child_result(
        self,
        child_call_id: str,
        child_content: str,
        *,
        tools_data: "ToolsData",
        msg_dispatcher: "LoopMessageDispatcher",
    ) -> None:
        """Merge one child's terminal result into the shared placeholder.

        A multi-handle child completes exactly once (success or error), so
        every call here is that child's final result. While the shared
        placeholder is still mutable it is rebuilt in place with every
        result resolved so far (free, since nothing has been dispatched).
        Once sent it is frozen for good — it cannot keep reporting new
        children without rewriting already-sent bytes — so this child's
        result is delivered instead on its own synthesized call_id via a
        per-child check_status pair.
        """
        if tools_data._mutable(self.placeholder_msg):
            updated = _rebuild_multi_handle_content(self.template, self.results)
            all_done = all(v is not None for v in self.results.values())
            self.placeholder_msg["content"] = serialize_tool_content(
                tool_name=self.parent_name,
                payload=updated,
                is_final=all_done,
            )
            await msg_dispatcher.publish_to_event_bus([self.placeholder_msg])
            return
        await emit_completion_pair(child_content, child_call_id, msg_dispatcher)


def _rebuild_multi_handle_content(template, results):
    """Replace labeled sentinels in *template* with completed results."""

    def _walk(node):
        if isinstance(node, str):
            for label, raw in results.items():
                if node == _handle_label_sentinel(label) and raw is not None:
                    return raw
            return node
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(v) for v in node]
        if isinstance(node, tuple):
            return tuple(_walk(v) for v in node)
        try:
            from pydantic import BaseModel

            if isinstance(node, BaseModel):
                updates = {}
                changed = False
                for field_name in node.model_fields:
                    val = getattr(node, field_name)
                    walked = _walk(val)
                    if walked is not val:
                        updates[field_name] = walked
                        changed = True
                return node.model_copy(update=updates) if changed else node
        except ImportError:
            pass
        return node

    return _walk(template)


def _extract_nested_handle(obj):
    """Walk *obj* (dict / list / tuple / Pydantic model) looking for ``SteerableToolHandle`` instances.

    Returns ``(result, cleaned_obj)`` where:

    - **Bare top-level handle**: ``result`` is the handle itself, ``cleaned_obj``
      is :data:`_HANDLE_SENTINEL`.
    - **Handles nested in containers** (one or more): ``result`` is a list of
      ``(handle, label)`` tuples (``"h0"``, ``"h1"``, …), ``cleaned_obj`` is a
      copy of *obj* with handles replaced by labeled sentinels.
    - **No handles**: ``(None, obj)`` unchanged.
    """
    from unify.common.async_tool_loop import SteerableToolHandle

    if isinstance(obj, SteerableToolHandle):
        return obj, _HANDLE_SENTINEL

    found: list = []
    counter = [0]

    def _walk(node):
        if isinstance(node, SteerableToolHandle):
            label = f"h{counter[0]}"
            counter[0] += 1
            found.append((node, label))
            return _handle_label_sentinel(label)

        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}

        if isinstance(node, list):
            return [_walk(v) for v in node]

        if isinstance(node, tuple):
            return tuple(_walk(v) for v in node)

        # Pydantic BaseModel: walk public field values so handles inside
        # e.g. ExecutionResult.result are detected.
        try:
            from pydantic import BaseModel

            if isinstance(node, BaseModel):
                changed = False
                updates = {}
                for field_name in node.model_fields:
                    val = getattr(node, field_name)
                    cleaned_val = _walk(val)
                    if cleaned_val is not val:
                        updates[field_name] = cleaned_val
                        changed = True
                if changed:
                    return node.model_copy(update=updates)
                return node
        except ImportError:
            pass

        return node

    cleaned = _walk(obj)

    if found:
        return found, cleaned

    return None, obj


def _returned_handles_for_cleanup(value):
    """Find handles in supported stored values without evaluating accessors.

    Cycles and aliases are traversed once. Arbitrary iterables and computed
    attributes are not part of the returned-resource ownership contract.
    """
    from pydantic import BaseModel
    from unify.common.async_tool_loop import SteerableToolHandle

    pending = [value]
    visited = set()
    handles = []
    while pending:
        node = pending.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        node_type = type(node)
        if issubclass(node_type, SteerableToolHandle):
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
        extra_ask_tools: "Optional[Dict[str, Callable]]" = None,
        completed_askable_tools: Optional[Dict[str, dict]] = None,
        call_counts: Optional[Dict[str, int]] = None,
    ):
        self._client = client
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
        # Callback for refreshing dynamic helpers when a handle is adopted
        self._on_handle_adopted: Optional[Callable[[asyncio.Task], None]] = None
        # Time context for inline timing annotations on tool results
        self._time_ctx: Optional["TimeContext"] = time_ctx
        # DynamicToolFactory.live_ask_fns for the current turn: per-call `ask`
        # closures kept only to seed recursive inspection-loop tool schemas
        # (get_ask_tools()); never part of the outer loop's own visible
        # schema, which holds only the static wait/steer/ask_about_completed_tool
        # surface.
        self._live_ask_fns_ref: Optional[Dict[str, Callable]] = None
        self._completed_ask_handles: Dict[str, Callable] = {}
        self._task_ask_keys: Dict[asyncio.Task, str] = {}
        # Metadata for completed steerable tools, keyed by call_id; each entry
        # is {"name": str, "call_id": str, "ask_fn": Callable, "handle": Any}.
        # An inspection loop seeds this with the registry of the loop it
        # inspects: that transcript announces "[askable <call_id>]" ids from
        # the inspected loop's namespace, so ask_about_completed_tool must
        # resolve them here too.
        self._completed_askable_tools: Dict[str, dict] = (
            dict(completed_askable_tools) if completed_askable_tools else {}
        )
        # Caller-supplied ask tools injected at construction time (e.g.
        # domain-specific read-only tools for handle.ask() inspection loops).
        self._extra_ask_tools: Dict[str, Callable] = (
            dict(extra_ask_tools) if extra_ask_tools else {}
        )
        # Shared with the loop's own interjection-triggered injection so the
        # guidance lands at most once, whichever trigger — a user interjection
        # or the first [progress]/[clarification] message — fires first.
        self._visibility_guidance_injected: bool = False
        # UNIFY_LIFECYCLE_NOTICES, read once so a loop never changes mid-way:
        # off, the "[steerable ...]" and "[askable ...]" announcements are not
        # appended (nothing already in the transcript is touched).
        from unify.settings import SETTINGS

        self._lifecycle_notices: bool = bool(SETTINGS.UNIFY_LIFECYCLE_NOTICES)

    def get_ask_tools(self) -> Dict[str, Callable]:
        """Snapshot of the currently available ``ask_*`` dynamic tools.

        Merges three sources with increasing precedence: completed ask
        handles < extra_ask_tools < live ask closures. Used solely to seed a
        *recursive* inspection loop's own tool schema
        (SteerableToolHandle.ask()) so it can propagate a question into a
        still-nested grandchild; never merged into the outer loop's own
        visible schema.
        """
        result = dict(self._completed_ask_handles)
        result.update(self._extra_ask_tools)
        live = self._live_ask_fns_ref
        if live and isinstance(live, dict):
            result.update(live)
        return result

    @staticmethod
    def _pretty_tool_payload(tool_name: str, payload: Any) -> str:
        # Non-final serialization, for progress/notification placeholders.
        return serialize_tool_content(
            tool_name=tool_name,
            payload=payload,
            is_final=False,
        )

    def _quota_count(self, task_name: str) -> int:
        return self.call_counts.get(task_name, 0)

    def _mutable(self, msg: dict) -> bool:
        """True when *msg* has not yet been included in any dispatched request."""
        return is_mutable(self._client, msg)

    async def _ensure_visibility_guidance_injected(
        self,
        msg_dispatcher: "LoopMessageDispatcher",
    ) -> None:
        """Inject the user-visibility guidance before the first status-shaped
        tail message a user could mistake for an interjection.

        Shares its flag with the loop's own interjection-triggered injection
        so the guidance lands exactly once, whichever trigger — a real user
        interjection, or the first ``[progress]``/``[clarification]``
        message — fires first. Most loops never see a user interjection, so
        gating solely on that would leave every sub-agent and unattended task
        without the guidance that says these messages are not requests.

        The check-await-set pattern assumes one coroutine calls this per
        ``ToolsData`` instance at a time: the loop drives one turn at a time
        even with concurrent tools in flight, since notification/clarification
        handling and the interjection drain never run concurrently with each
        other. Two truly concurrent callers could both read the flag as
        ``False`` before either sets it and double-inject — harmless, since
        the system message is idempotent.
        """
        if self._visibility_guidance_injected:
            return
        await msg_dispatcher.append_msgs(
            [
                {
                    "role": "system",
                    "_visibility_guidance": True,
                    "content": USER_VISIBILITY_GUIDANCE,
                },
            ],
        )
        self._visibility_guidance_injected = True

    async def record_progress(
        self,
        info: "ToolCallMetadata",
        call_id: str,
        pretty: str,
        msg_dispatcher: "LoopMessageDispatcher",
    ) -> None:
        """Coalesce-then-freeze progress delivery.

        Progress lands as ``[progress <call_id>]``-prefixed user-role tail
        messages tracked in ``info.progress_msg``, separate from
        ``info.tool_reply_msg`` so the final result never shares a slot with
        transient progress text. While the current progress message is still
        above the sent watermark it is edited in place, coalescing a burst
        into one message; once dispatched it is frozen and the next
        notification starts a fresh tail message.
        """
        await self._ensure_visibility_guidance_injected(msg_dispatcher)
        content = f"[progress {call_id}] {pretty}"
        existing = info.progress_msg
        if existing is not None and self._mutable(existing):
            existing["content"] = content
            return
        new_msg = loop_user_notice(content, _progress_msg=True)
        await msg_dispatcher.append_msgs([new_msg])
        info.progress_msg = new_msg

    async def record_clarification(
        self,
        info: "ToolCallMetadata",
        call_id: str,
        question_text: str,
        msg_dispatcher: "LoopMessageDispatcher",
    ) -> None:
        """Coalesce-then-freeze clarification-question delivery.

        Mirrors ``record_progress`` but is tracked in ``info.clarify_msg``
        and prefixed ``[clarification <call_id>]`` so the model knows it
        wants a reply via ``steer(call_id=<call_id>, action="clarify",
        payload=<answer>)``, unlike a status-only ``[progress ...]`` message.
        ``info.tool_reply_msg`` (the pending stub) is never touched: the
        final result lands there, or on ``clarify_placeholder`` once the
        model answers, never on this tail message.
        """
        await self._ensure_visibility_guidance_injected(msg_dispatcher)
        content = (
            f"[clarification {call_id}] Tool incomplete, please answer the "
            f"following to continue tool execution via "
            f'steer(call_id="{call_id}", action="clarify", payload=<answer>):\n'
            f"{question_text}"
        )
        existing = info.clarify_msg
        if existing is not None and self._mutable(existing):
            existing["content"] = content
            return
        new_msg = loop_user_notice(content, _clarify_msg=True)
        await msg_dispatcher.append_msgs([new_msg])
        info.clarify_msg = new_msg

    @staticmethod
    def _describe_custom_methods(handle: Any, call_id: str) -> str:
        """Render a handle's custom methods (beyond the core steering surface)
        as a short listing: name, signature, one-line docstring. Custom
        `action="call"` methods are validated at execution time rather than
        exposed as tools of their own, so this listing is the model's only
        description of them. Returns "" when there are none.
        """
        with suppress(Exception):
            # Imported at call time: dynamic_tools_factory imports this
            # module at top level, so a module-level import would be circular.
            from .dynamic_tools_factory import DynamicToolFactory

            custom_methods = DynamicToolFactory._discover_custom_public_methods(
                handle,
            )
            lines = []
            for meth_name, bound in sorted(custom_methods.items()):
                try:
                    sig = inspect.signature(bound)
                except Exception:
                    sig = "(...)"
                doc = (inspect.getdoc(bound) or "").strip().splitlines()
                first_line = doc[0] if doc else ""
                suffix = f" — {first_line}" if first_line else ""
                lines.append(f"  - {meth_name}{sig}{suffix}")
            if lines:
                return (
                    f' Custom methods reachable via steer(call_id="{call_id}", '
                    'action="call", method=<name>, payload=<JSON object>):\n'
                    + "\n".join(lines)
                )
        return ""

    async def record_tool_started(
        self,
        info: "ToolCallMetadata",
        msg_dispatcher: "LoopMessageDispatcher",
    ) -> None:
        """Announce that a call is now live and steerable via ``steer``.

        One-shot, append-only tail message (same shape as record_progress /
        record_clarification, minus coalescing — a call starts exactly
        once). The tool schema is static, so the transcript is the only
        place this signal can live.

        Deliberately carries no argument payload — the adjacent assistant
        `tool_calls` entry already has the full arguments; duplicating them
        here would freeze a second copy into the prefix forever.
        Custom-method discoverability lives in `record_tool_capability_delta`,
        so a call that never gets a handle never pays for that either.
        Not appended under ``UNIFY_LIFECYCLE_NOTICES=0``.
        """
        if not self._lifecycle_notices:
            return
        await self._ensure_visibility_guidance_injected(msg_dispatcher)
        content = f"[steerable {info.call_id}] {info.name} started."
        await msg_dispatcher.append_msgs(
            [loop_user_notice(content, _lifecycle_msg=True)],
        )

    async def record_tool_capability_delta(
        self,
        info: "ToolCallMetadata",
        msg_dispatcher: "LoopMessageDispatcher",
    ) -> None:
        """Announce that a call already covered by `record_tool_started`
        just widened its steer() surface (a handle was adopted).

        Not a re-announcement: no arguments, no restatement of "started",
        only which of interject/pause/ask became available plus any custom
        methods the handle exposes. Not appended under
        ``UNIFY_LIFECYCLE_NOTICES=0``.
        """
        if not self._lifecycle_notices:
            return
        handle = info.handle
        caps = []
        if handle is not None:
            if hasattr(handle, "interject"):
                caps.append("interject")
            if hasattr(handle, "pause") or hasattr(handle, "resume"):
                caps.append("pause")
            if hasattr(handle, "ask"):
                caps.append("ask")
        if not caps:
            return
        await self._ensure_visibility_guidance_injected(msg_dispatcher)
        content = f"[steerable {info.call_id}] now supports {'/'.join(caps)}."
        if handle is not None:
            content += self._describe_custom_methods(handle, info.call_id)
        await msg_dispatcher.append_msgs(
            [loop_user_notice(content, _lifecycle_msg=True)],
        )

    async def record_tool_completed_askable(
        self,
        call_id: str,
        name: str,
        msg_dispatcher: "LoopMessageDispatcher",
    ) -> None:
        """Announce that a completed call's trajectory is now askable.

        An appended tail message rather than a live listing in
        ``ask_about_completed_tool``'s docstring, which would churn the
        schema on every completion; that docstring stays frozen. Carries no
        argument payload, same as ``record_tool_started``: the adjacent
        assistant `tool_calls` entry already has the full arguments, and a
        copy here would freeze into the prefix forever. Not appended under
        ``UNIFY_LIFECYCLE_NOTICES=0``.
        """
        if not self._lifecycle_notices:
            return
        await self._ensure_visibility_guidance_injected(msg_dispatcher)
        content = (
            f"[askable {call_id}] {name} completed and is askable via "
            f'ask_about_completed_tool(tool_id="{call_id}", question=...).'
        )
        await msg_dispatcher.append_msgs(
            [loop_user_notice(content, _lifecycle_msg=True)],
        )

    def resolve_call_id(
        self,
        call_id: str,
    ) -> Tuple[Optional[asyncio.Task], Optional["ToolCallMetadata"]]:
        """Exact-match lookup of the live (pending) task for *call_id*.

        steer() targets calls by their real id verbatim; suffix/endswith
        matching would only add ambiguity.

        Also requires ``not t.done()``: a task can briefly sit in
        ``self.pending``/``self.info`` after its coroutine has finished but
        before ``process_completed_task`` has popped it. A call in that
        window must resolve as "not live" so steer() routes it to the same
        instructive "already completed" error as any other finished call
        rather than dispatching against it as if it were still running.
        """
        for t in self.pending:
            if t.done():
                continue
            inf = self.info.get(t)
            if inf is not None and inf.call_id == call_id:
                return t, inf
        return None, None

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
        # Retain this task's ask_* handle so handle.ask() can still propagate
        # after completion.
        info = self.info.get(coro)
        ask_name = self._task_ask_keys.pop(coro, None)
        if ask_name is not None:
            dt = self._live_ask_fns_ref
            if dt and isinstance(dt, dict) and ask_name in dt:
                ask_fn = dt[ask_name]
                self._completed_ask_handles[ask_name] = ask_fn
                # Store metadata for the ask_about_completed_tool dispatcher.
                if info is not None:
                    call_id = info.call_id
                    self._completed_askable_tools[call_id] = {
                        "name": info.name,
                        "call_id": call_id,
                        "ask_fn": ask_fn,
                        "handle": info.handle,
                    }
        self.pending.discard(coro)
        return self.info.pop(coro, None)

    def active_count(self, task_name: str) -> int:
        return sum(1 for _t, _inf in self.info.items() if _inf.name == task_name)

    async def cancel_pending_tasks(self):
        pending = list(self.pending)
        stopped = set()

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

        stops = stop_tasks()
        for task in pending:
            task.cancel()
        # Stop and cancel concurrently: an async stop can await its owned task.
        # A child's stop failure must not cancel sibling cleanup operations.
        await asyncio.gather(*pending, join_stops(stops), return_exceptions=True)
        # A factory may catch cancellation and return a newly owned handle.
        await join_stops(stop_tasks())
        for task in pending:
            self.pending.discard(task)
            self.info.pop(task, None)

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
        initial_paused: bool = False,
    ) -> None:
        if name not in self.normalized:
            return

        fn = self.normalized[name].fn

        # Over-quota calls should already be pruned from the assistant
        # message; skip silently if one slipped through.
        with suppress(Exception):
            lim = self.normalized[name].max_total_calls
            if lim is not None and self.call_counts.get(name, 0) >= lim:
                return

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

        sig_accepts_interject_q = "_interject_queue" in params or has_varkw
        sig_accepts_pause_event = "_pause_event" in params or has_varkw
        sig_accepts_clar_qs = (
            "_clarification_up_q" in params and "_clarification_down_q" in params
        ) or has_varkw
        sig_accepts_progress = "_notification_up_q" in params or has_varkw

        pause_ev: Optional[asyncio.Event] = None
        if sig_accepts_pause_event:
            pause_ev = asyncio.Event()
            if initial_paused:
                pause_ev.clear()  # start paused
            else:
                pause_ev.set()  # start running
            extra_kwargs["_pause_event"] = pause_ev

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

        sub_q: Optional[asyncio.Queue[str]] = None
        if sig_accepts_interject_q:
            sub_q = asyncio.Queue()
            extra_kwargs["_interject_queue"] = sub_q

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
            is_interjectable=sig_accepts_interject_q,
            interject_queue=sub_q,
            chat_context=extra_kwargs.get("_parent_chat_context"),
            clar_up_queue=clar_up_q,
            clar_down_queue=clar_down_q,
            notification_queue=progress_q,
            pause_event=pause_ev,
            # Debug helpers for failure logging. The arguments keep any the
            # tool has no parameter for: refusal tallies fingerprint them, and
            # two calls that differ only there are different calls.
            tool_schema=tool_schema,
            llm_arguments={**allowed_call_args, **unknown_call_args},
            raw_arguments_json=args_json,
            # Track context opt-in for steering method context propagation
            context_opted_in=context_opted_in,
        )
        self.save_task(t, metadata)

        if self._logger.log_steps:
            self._logger.info(
                f"{name} - {call_id}",
                prefix=f"🛠️  ToolCall Scheduled",
            )

        # Announce steerability so the model has an explicit call_id pointer
        # for steer() later — without this, models reliably hallucinate a
        # plausible-looking id instead of reading the real one back from
        # their own earlier tool_calls entry.
        if msg_dispatcher is not None:
            with suppress(Exception):
                await self.record_tool_started(metadata, msg_dispatcher)

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
        ``result``, patch or insert the tool message, publish it to the event
        bus, record the payload in ``completed_results`` for post-hoc lookups
        and enforce the *max_consecutive_failures* safety valve.
        """
        import time as _pct_time

        _pct_t0 = _pct_time.perf_counter()

        def _pct_ms():
            return f"{(_pct_time.perf_counter() - _pct_t0) * 1000:.0f}ms"

        info: ToolCallMetadata = self.pop_task(task)
        name = info.name
        call_id = info.call_id

        # Announce retrospective askability now that pop_task has (possibly)
        # promoted this call_id into _completed_askable_tools.
        askable_entry = self._completed_askable_tools.get(call_id)
        if askable_entry is not None:
            with suppress(Exception):
                await self.record_tool_completed_askable(
                    call_id,
                    askable_entry["name"],
                    msg_dispatcher,
                )

        _pickup_delay = _pct_time.perf_counter() - info.scheduled_time
        self._logger.debug(
            f"⏱️ [ToolsData.process_completed +{_pct_ms()}] {name} ({call_id}) "
            f"total_elapsed={_pickup_delay:.2f}s",
        )

        # Drain notifications that arrived just before completion, so a
        # fast-finishing tool's progress events are not lost.
        try:
            q = info.notification_queue
        except Exception:
            q = None
        if q is not None:
            while True:
                try:
                    payload = q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                except Exception:
                    break

                pretty = self._pretty_tool_payload(name, payload)

                # Coalesce-then-freeze into a separate [progress <call_id>]
                # tail message — never the tool_reply_msg placeholder, which
                # must stay byte-frozen once sent (see record_progress).
                await self.record_progress(info, call_id, pretty, msg_dispatcher)

                # Forward a programmatic notification event to the outer handle.
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

        self._logger.debug(
            f"⏱️ [ToolsData.process_completed +{_pct_ms()}] {name} notification drain done",
        )

        try:
            raw = task.result()

            # Multi-handle child: each completes independently and the shared
            # placeholder is rebuilt with the newly resolved result.
            mh_state = getattr(info, "_multi_handle_state", None)
            if mh_state is not None:
                label = info._multi_handle_label
                mh_state.results[label] = raw
                child_content = serialize_tool_content(
                    tool_name=name,
                    payload=raw,
                    is_final=True,
                )
                await mh_state.record_child_result(
                    call_id,
                    child_content,
                    tools_data=self,
                    msg_dispatcher=msg_dispatcher,
                )
                self.completed_results[call_id] = child_content
                self._completed_tool_names[call_id] = name
                consecutive_failures.reset_failures()
                if self._logger.log_steps:
                    self._logger.info(
                        f"{name} [{label}] - {call_id}",
                        prefix="✅  MultiHandle Child Completed",
                    )
                return True

            # Bare handle: the tool returned a SteerableToolHandle directly.
            from unify.common.async_tool_loop import SteerableToolHandle

            if isinstance(raw, SteerableToolHandle):
                await self.adopt_nested(
                    info,
                    raw,
                    msg_dispatcher=msg_dispatcher,
                    assistant_meta=assistant_meta,
                    outer_handle_container=outer_handle_container,
                )
                return False  # ⬅️  no LLM turn required

            # Composite return: one or more handles nested inside a
            # dict/list/tuple alongside intermediate data, surfaced to the
            # LLM as progress while each handle is steered independently.
            nested_handles, cleaned = _extract_nested_handle(raw)
            if nested_handles is not None:
                await self.adopt_multi_nested(
                    info,
                    nested_handles,
                    cleaned,
                    msg_dispatcher=msg_dispatcher,
                    assistant_meta=assistant_meta,
                    outer_handle_container=outer_handle_container,
                )
                return True  # ⬅️  LLM turn required — intermediate content to process

            # Plain (non-handle) result.
            result = serialize_tool_content(tool_name=name, payload=raw, is_final=True)

            if self._time_ctx is not None and not info.is_dynamic:
                result = self._time_ctx.wrap_result(result, info.scheduled_time)

            consecutive_failures.reset_failures()
        except Exception as exc:
            # Multi-handle child error: update the shared placeholder and return.
            mh_state = getattr(info, "_multi_handle_state", None)
            if mh_state is not None:
                label = info._multi_handle_label
                error_tb = _failure_text(exc)
                mh_state.results[label] = f"[{label}: error]\n{error_tb}"
                await mh_state.record_child_result(
                    call_id,
                    error_tb,
                    tools_data=self,
                    msg_dispatcher=msg_dispatcher,
                )
                self.completed_results[call_id] = error_tb
                self._completed_tool_names[call_id] = name
                if self._logger.log_steps:
                    self._logger.error(
                        f"{name} [{label}] - {call_id}\n{error_tb}",
                        prefix="❌  MultiHandle Child Failed",
                    )
                _record_failure(
                    consecutive_failures,
                    exc=exc,
                    tool_name=name,
                    args=info.llm_arguments,
                )
                mh_stop = consecutive_failures.stop_reason()
                if mh_stop:
                    raise RuntimeError(mh_stop)
                return True

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

        self._logger.debug(
            f"⏱️ [ToolsData.process_completed +{_pct_ms()}] {name} result obtained",
        )

        asst_msg = info.assistant_msg
        clarify_ph = info.clarify_placeholder
        tool_reply_msg = info.tool_reply_msg

        # Placeholder handling under the sent-watermark invariant: while still
        # mutable (not yet dispatched) it is updated in place, free of cache
        # cost; once sent the stub is never rewritten — it was self-describing
        # from the start (see ensure_placeholders_for_pending) — and the result
        # is delivered solely via an appended check_status pair.
        placeholder = clarify_ph or tool_reply_msg

        if placeholder is not None:
            if self._mutable(placeholder):
                placeholder["content"] = result
                tool_msg = placeholder
                # Placeholder insertion skipped the event bus; publish now
                # that the content is final.
                await msg_dispatcher.publish_to_event_bus([tool_msg])
            else:
                tool_msg = await emit_completion_pair(
                    result,
                    call_id,
                    msg_dispatcher,
                )
        else:
            tool_msg = create_tool_call_message(name, call_id, result)
            # First-ever reply to this call_id — legality requires strict
            # adjacency, so this always bypasses the watermark gate (see
            # insert_tool_message_after_assistant's escape hatch).
            await insert_tool_message_after_assistant(
                assistant_meta,
                asst_msg,
                tool_msg,
                self._client,
                msg_dispatcher,
                bypass_watermark=True,
            )

        self._logger.debug(
            f"⏱️ [ToolsData.process_completed +{_pct_ms()}] {name} tool message emitted",
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

    async def adopt_nested(
        self,
        info: "ToolCallMetadata",
        child_handle,
        *,
        msg_dispatcher,
        assistant_meta,
        outer_handle_container,
        intermediate_content: Any = None,
    ) -> None:
        """Adopt a child SteerableToolHandle returned by a tool into this loop.

        Creates or updates the single placeholder tool message, schedules the
        child's result as a nested task with inherited metadata, and wires
        clarification channels. When *intermediate_content* is given (a
        composite return with the handle nested inside data), the placeholder
        carries that data formatted as a progress notification so the LLM can
        react to it while steering continues.
        """
        if hasattr(child_handle, "interject"):
            info.is_interjectable = True

        h_up_q = getattr(child_handle, "clarification_up_q", info.clar_up_queue)
        h_down_q = getattr(child_handle, "clarification_down_q", info.clar_down_queue)
        if (h_up_q is not None) ^ (h_down_q is not None):
            raise AttributeError(
                f"Handle returned by tool {info.name!r} exposes only one of "
                "'clarification_up_q' / 'clarification_down_q'. Both are required (or neither).",
            )

        if inspect.iscoroutinefunction(child_handle.result):
            nested_coro = child_handle.result()
        else:
            nested_coro = asyncio.to_thread(child_handle.result)
        nested_task = asyncio.create_task(nested_coro)

        if intermediate_content is not None:
            placeholder_content = serialize_tool_content(
                tool_name=info.name,
                payload=intermediate_content,
                is_final=False,
            )
        else:
            placeholder_content = json.dumps(
                {"_placeholder": "nested_start"},
                indent=4,
            )

        ph = info.tool_reply_msg
        metadata = dataclasses.replace(
            info,
            handle=child_handle,
            is_interjectable=hasattr(child_handle, "interject"),
            interject_queue=None,
            tool_reply_msg=ph,
            clar_up_queue=h_up_q,
            clar_down_queue=h_down_q,
            notification_queue=info.notification_queue,
        )
        # Own the child before publishing anything that can suspend adoption.
        self.save_task(nested_task, metadata)
        if ph is None:
            ph = create_tool_call_message(
                name=info.name,
                call_id=info.call_id,
                content=placeholder_content,
            )
            metadata.tool_reply_msg = ph
            await insert_tool_message_after_assistant(
                assistant_meta,
                info.assistant_msg,
                ph,
                self._client,
                msg_dispatcher,
                skip_event_bus=True,  # Nested placeholder; final result published when nested task completes
                bypass_watermark=True,  # first-ever reply to this call_id — legality, not caching
            )
            info.tool_reply_msg = ph
        elif self._mutable(ph):
            # Nothing has dispatched ph yet, so editing it in place to say
            # "now running as a nested handle" is free; observers waiting on
            # the tool result key off this edit to know adoption happened.
            ph["content"] = placeholder_content
        else:
            # ph is already frozen (dispatched). This update is a transient
            # status marker, not the call's terminal result — that still
            # lands on `ph` via process_completed_task's own gate when the
            # nested task finishes — so it goes through the same
            # coalesce-then-freeze progress delivery as tool notifications.
            pretty = (
                placeholder_content
                if isinstance(placeholder_content, str)
                else json.dumps(placeholder_content)
            )
            await self.record_progress(info, info.call_id, pretty, msg_dispatcher)

        if h_up_q is not None:
            self.clarification_channels[info.call_id] = (h_up_q, h_down_q)
        # Refresh dynamic helpers immediately, now that a handle is available.
        if self._on_handle_adopted is not None:
            with suppress(Exception):
                self._on_handle_adopted(nested_task)
        # Only the capability delta: record_tool_started already covered
        # call_id discovery when this call was scheduled.
        with suppress(Exception):
            await self.record_tool_capability_delta(metadata, msg_dispatcher)

    async def adopt_multi_nested(
        self,
        info: "ToolCallMetadata",
        handles: list,
        intermediate_content: Any,
        *,
        msg_dispatcher,
        assistant_meta,
        outer_handle_container,
    ) -> None:
        """Adopt multiple handles from a single tool's composite return.

        One placeholder for the parent call_id carries the intermediate
        content with labeled sentinels (``[h0: steerable]``, …); each handle
        is scheduled as an independent child task with a synthesized call_id
        and progressively updates that placeholder via
        :class:`_MultiHandleState`.
        """
        parent_call_id = info.call_id

        placeholder_content = serialize_tool_content(
            tool_name=info.name,
            payload=intermediate_content,
            is_final=False,
        )
        ph = info.tool_reply_msg
        if ph is None:
            ph = create_tool_call_message(
                name=info.name,
                call_id=parent_call_id,
                content=placeholder_content,
            )
            await insert_tool_message_after_assistant(
                assistant_meta,
                info.assistant_msg,
                ph,
                self._client,
                msg_dispatcher,
                skip_event_bus=True,
                bypass_watermark=True,  # first-ever reply to this call_id — legality, not caching
            )
        else:
            # `ph` becomes `state.placeholder_msg`, the single shared slot
            # `_MultiHandleState.record_child_result` keeps managing
            # (mutate-if-mutable, else per-child check_status) for the rest
            # of this call. If it has already been dispatched, leave it
            # frozen rather than forking a second, disconnected view that
            # record_child_result never touches.
            if self._mutable(ph):
                ph["content"] = placeholder_content

        state = _MultiHandleState(
            parent_call_id=parent_call_id,
            parent_name=info.name,
            placeholder_msg=ph,
            template=intermediate_content,
            results={label: None for _, label in handles},
        )

        for handle, label in handles:
            # The label is appended without an underscore so the 8-char
            # safe_call_id carries both parent uniqueness and the label, and
            # two parents that each have an h0 cannot collide.
            synth_call_id = f"{parent_call_id}{label}"

            if inspect.iscoroutinefunction(handle.result):
                nested_coro = handle.result()
            else:
                nested_coro = asyncio.to_thread(handle.result)
            nested_task = asyncio.create_task(nested_coro)

            h_up_q = getattr(handle, "clarification_up_q", None)
            h_down_q = getattr(handle, "clarification_down_q", None)

            metadata = dataclasses.replace(
                info,
                call_id=synth_call_id,
                handle=handle,
                is_interjectable=hasattr(handle, "interject"),
                # As in adopt_nested: the parent tool's queue died with it.
                interject_queue=None,
                tool_reply_msg=ph,
                clar_up_queue=h_up_q,
                clar_down_queue=h_down_q,
                notification_queue=None,
                # A clean slate per child: two children inheriting the
                # parent's progress/clarify message reference would coalesce
                # their updates onto one tail message.
                progress_msg=None,
                clarify_msg=None,
                _multi_handle_state=state,
                _multi_handle_label=label,
            )
            self.save_task(nested_task, metadata)

            if h_up_q is not None:
                self.clarification_channels[synth_call_id] = (h_up_q, h_down_q)

            if self._on_handle_adopted is not None:
                with suppress(Exception):
                    self._on_handle_adopted(nested_task)

            # Announce this child's synthesized call_id: steer() needs it
            # verbatim and, unlike the single-handle case, it differs from
            # the original call's id, so nothing else in the transcript
            # carries it. The child has its handle at birth, so the
            # capability delta fires right after as part of this one
            # announcement.
            with suppress(Exception):
                await self.record_tool_started(metadata, msg_dispatcher)
                await self.record_tool_capability_delta(metadata, msg_dispatcher)
