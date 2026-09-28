import asyncio
import unillm
import hashlib
import json
import inspect
import copy
from dataclasses import dataclass, field

from typing import (
    Dict,
    Union,
    Callable,
    Tuple,
    Any,
    Set,
    Optional,
    TYPE_CHECKING,
)
from contextlib import suppress
from pydantic import BaseModel

from ...logger import LOGGER
from ..tool_spec import ToolSpec, normalise_tools
from .propagation_mode import ChatContextPropagation
from .context_tracker import LoopContextState
from .utils import maybe_await, get_handle_paused_state
from .event_bus_util import to_event_bus
from ...events.types.tool_loop import ToolLoopKind
from .messages import (
    find_unreplied_assistant_entries,
    generate_with_preprocess,
    acknowledge_helper_call,
)
from .message_dispatcher import LoopMessageDispatcher
from .tools_utils import (
    create_tool_call_message,
    ToolCallMetadata,
)
from ..llm_helpers import (
    DEFAULT_TOOL_SCHEMA_STRICT,
    method_to_schema,
    _dumps,
    short_id,
)
from .loop_config import (
    LoopConfig,
    TOOL_LOOP_LINEAGE,
)
from .timeout_timer import TimeoutTimer
from .messages import (
    insert_tool_message_after_assistant,
    ensure_placeholders_for_pending,
    forward_handle_call,
    schedule_missing_for_message,
    is_mutable,
    is_loop_authored_message,
    loop_user_notice,
    extract_substantive_text,
    compact_reviewed_messages,
    strip_reasoning_payloads,
    _rebaseline_watermark_hash,
)
from .tools_data import (
    ToolsData,
    compute_context_injection,
)
from .dynamic_tools_factory import DynamicToolFactory
from .time_context import create_time_context, TimeContext
from .context_compression import (
    compress_context,
    _COMPRESSION_SIGNAL,
    context_over_threshold,
)
from .response_format import (
    NormalizedResponseFormat,
    normalize_response_format,
)
from ..context_dump import make_messages_safe_for_context_dump
from ...common.hierarchical_logger import ICONS

if TYPE_CHECKING:
    from .multi_handle import MultiHandleCoordinator
    from unillm.types import PromptCacheParam


@dataclass
class ToolLoopRuntimeState:
    call_counts: Dict[str, int] = field(default_factory=dict)
    called_tools: list[str] = field(default_factory=list)
    step_index: int = 0
    consecutive_failures: int = 0
    message_count_offset: int = 0
    # Refused calls, tallied two ways so that working out an argspec is free
    # while repeating a rejected call is not. Both persist for the run: a call
    # refused identically three times is not being learned from, and a success
    # elsewhere does not change that.
    refusals_by_call: Dict[str, int] = field(default_factory=dict)
    refusals_by_complaint: Dict[str, int] = field(default_factory=dict)
    pending_stop_reason: Optional[str] = None


def _parse_tool_policy_result(
    result: Any,
) -> Tuple[str, Dict[str, Callable], bool]:
    """Normalize a ``tool_policy`` return value.

    Accepted shapes:
      - ``(mode, tools)``
      - ``(mode, tools, {"eager": bool, ...})``

    ``eager=True`` means: after the model schedules tool calls on this turn,
    immediately grant another LLM turn (without waiting for those tools to
    finish) for as long as subsequent policy evaluations keep returning
    ``eager=True``.  Default is ``False`` (wait for tool results).

    While ``eager=True``, the loop also withholds ``compress_context`` from the
    visible tool schema (except on the forced over-threshold compression path)
    so gated required policies cannot be bypassed by compressing context.
    """
    if not isinstance(result, (tuple, list)) or len(result) < 2:
        raise TypeError(
            f"tool_policy must return (mode, tools[, opts]), got {type(result)!r}",
        )
    mode, tools = result[0], result[1]
    eager = False
    if len(result) >= 3:
        opts = result[2]
        if isinstance(opts, dict):
            eager = bool(opts.get("eager", False))
        else:
            eager = bool(opts)
    return str(mode), tools, eager


def _is_cache_miss_error(exc: BaseException | None) -> bool:
    """True when *exc* (or anything in its cause/context chain) is a
    read-only LLM-cache miss (``unillm.caching.CacheMissError``)."""
    from unillm.caching import CacheMissError

    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, CacheMissError):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ if exc.__cause__ is not None else exc.__context__
    return False


def prune_duplicate_tool_calls(tool_calls: list) -> tuple[list, set[str]]:
    """Remove duplicate tool calls from a list.

    Returns (unique_calls, pruned_call_ids) where pruned_call_ids contains
    the IDs of calls that were removed as duplicates.
    """
    seen: Set[tuple[str, str]] = set()
    unique_calls: list = []
    pruned_ids: set[str] = set()
    for call in tool_calls:
        _fn = call.get("function") or {}
        _args = _fn.get("arguments", "")
        _args_str = _args if isinstance(_args, str) else json.dumps(_args)
        sig = (_fn.get("name", ""), _args_str)
        if sig not in seen:
            seen.add(sig)
            unique_calls.append(call)
        else:
            pruned_ids.add(call.get("id", ""))
    return unique_calls, pruned_ids


def _transform_context_roles(messages: list[dict]) -> list[dict]:
    """
    Rewrite 'user'/'assistant' roles to 'outer_user'/'outer_assistant' so parent
    context reads as system-provided history rather than injected user content.
    """
    transformed = []
    for msg in messages:
        new_msg = dict(msg)
        role = new_msg.get("role", "")
        if role == "user":
            new_msg["role"] = "outer_user"
        elif role == "assistant":
            new_msg["role"] = "outer_assistant"
        transformed.append(new_msg)
    return transformed


def _sort_completed_tasks_by_call_id(
    tasks: Set[asyncio.Task],
    tools_data: "ToolsData",
) -> list[asyncio.Task]:
    """Sort completed tasks by call_id for deterministic processing order."""
    return sorted(
        tasks,
        key=lambda t: (
            tools_data.info.get(t).call_id if tools_data.info.get(t) else ""
        ),
    )


class LoopLogger:
    def __init__(self, cfg: LoopConfig, log_steps: bool | str) -> None:
        self._label = cfg.label
        self._log_steps = log_steps
        self._first_llm_logged = False
        self._defer_after_first_llm: list[tuple[str, str]] = []
        self._thinking_emitted = False

    @property
    def log_steps(self):
        return self._log_steps

    @property
    def log_label(self):
        return self._label

    def info(self, msg, prefix=""):
        txt = f"{prefix} [{self._label}] {msg}"
        LOGGER.info(txt)

    def debug(self, msg, prefix=""):
        txt = f"{prefix} [{self._label}] {msg}"
        LOGGER.debug(txt)

    def error(self, msg, prefix=""):
        txt = f"{prefix} [{self._label}] {msg}"
        LOGGER.error(txt)

    def begin_thinking(self) -> None:
        self._thinking_emitted = False
        if not self._first_llm_logged:
            self._first_llm_logged = True
            for p, m in self._defer_after_first_llm:
                self.info(m, prefix=p)
            self._defer_after_first_llm.clear()

    def emit_thinking_with_path(self, path) -> None:
        self._thinking_emitted = True
        self.info(f"LLM thinking… → {path}", prefix=ICONS["llm_thinking"])

    def emit_thinking_fallback(self) -> None:
        if not self._thinking_emitted:
            self._thinking_emitted = True
            self.info("LLM thinking…", prefix=ICONS["llm_thinking"])

    def defer_after_first_llm(self, msg: str, prefix: str = "") -> None:
        if self._first_llm_logged:
            self.info(msg, prefix=prefix)
        else:
            self._defer_after_first_llm.append((prefix, msg))


class _LoopToolFailureTracker:
    def __init__(
        self,
        max_consecutive_failures: int,
        runtime_state: ToolLoopRuntimeState,
    ):
        self._runtime_state = runtime_state
        self._max_consecutive_failures = max_consecutive_failures

    @property
    def current_failures(self):
        return self._runtime_state.consecutive_failures

    @property
    def max_failures(self):
        return self._max_consecutive_failures

    def has_exceeded_failures(self) -> bool:
        return (
            self._runtime_state.consecutive_failures >= self._max_consecutive_failures
        )

    def increment_failures(self):
        self._runtime_state.consecutive_failures += 1

    def reset_failures(self):
        self._runtime_state.consecutive_failures = 0

    # ── refused calls ──────────────────────────────────────────────────────
    #
    # A refusal is not a fault the way an unexpected exception is: converging on
    # an argspec means being told what is wrong and trying again, so counting
    # refusals against `max_consecutive_failures` would abort exactly the
    # behaviour the messages exist to produce. What is never progress is
    # repetition, so refusals are counted by what repeats rather than by how
    # many there are.

    # The same call, refused and sent again unchanged. Nothing was read.
    IDENTICAL_CALL_LIMIT = 3
    # The same complaint, however the arguments are dressed. Something is being
    # varied, but not the part the refusal is about.
    SAME_COMPLAINT_LIMIT = 6

    def note_refusal(self, *, tool_name: str, args: Any, message: str) -> Optional[str]:
        """Record a refused call; return why to stop, or ``None`` to continue."""
        state = self._runtime_state

        call_key = f"{tool_name}::{_fingerprint(args)}"
        state.refusals_by_call[call_key] = state.refusals_by_call.get(call_key, 0) + 1
        seen = state.refusals_by_call[call_key]
        if seen >= self.IDENTICAL_CALL_LIMIT:
            return self._stop(
                f"{tool_name} was called with the same arguments and refused "
                f"{seen} times, so the refusal is not being read. Last refusal: "
                f"{message}",
            )

        complaint_key = f"{tool_name}::{_fingerprint(message)}"
        state.refusals_by_complaint[complaint_key] = (
            state.refusals_by_complaint.get(complaint_key, 0) + 1
        )
        same_complaint = state.refusals_by_complaint[complaint_key]
        if same_complaint >= self.SAME_COMPLAINT_LIMIT:
            return self._stop(
                f"{tool_name} was refused {same_complaint} times for the same "
                f"reason while the arguments varied around it, so the part being "
                f"varied is not the part at fault. Refusal: {message}",
            )

        return None

    def _stop(self, reason: str) -> str:
        self._runtime_state.pending_stop_reason = reason
        return reason

    def stop_reason(self) -> Optional[str]:
        """Why the loop should end now, or ``None`` to keep going."""
        if self._runtime_state.pending_stop_reason is not None:
            return self._runtime_state.pending_stop_reason
        if self.has_exceeded_failures():
            return "Aborted after too many consecutive tool failures."
        return None


def _fingerprint(value: Any) -> str:
    """Stable short digest of *value*, so tallies cost no memory of their own."""
    try:
        rendered = json.dumps(value, sort_keys=True, default=repr)
    except (TypeError, ValueError):
        rendered = repr(value)
    return hashlib.sha256(rendered.encode("utf-8", "replace")).hexdigest()[:16]


async def async_tool_loop_inner(
    client: unillm.AsyncUnify,
    message: str | dict | list[str | dict],
    tools: Dict[str, Union[Callable, ToolSpec]],
    *,
    loop_id: Optional[str] = None,
    lineage: Optional[list[str]] = None,
    interject_queue: asyncio.Queue[dict | str],
    cancel_event: asyncio.Event,
    stop_event: asyncio.Event | None = None,
    pause_event: asyncio.Event,
    max_consecutive_failures: int = 3,
    prune_tool_duplicates: bool = True,
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
    outer_handle_container: Optional[list] = None,
    response_format: Optional[Any] = None,
    max_parallel_tool_calls: Optional[int] = None,
    persist: bool = False,
    multi_handle_coordinator: Optional["MultiHandleCoordinator"] = None,
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
    runtime_state: Optional[ToolLoopRuntimeState] = None,
) -> str:
    r"""
    Run an interactive function-calling dialogue between an LLM and a set of
    Python callables until the model yields a final plain-text answer.

    Every tool call the model requests runs in its own ``asyncio.Task``, so
    long-running calls advance in parallel and the loop only ever waits for
    the first one to finish. Setting ``cancel_event`` cancels and awaits every
    task, then re-raises ``asyncio.CancelledError``; pushing onto
    ``interject_queue`` injects a user turn before the next LLM step without
    disturbing running tools. Exceptions inside tools are serialised and shown
    to the model; ``max_consecutive_failures`` back-to-back crashes abort the
    loop with ``RuntimeError``. Transport lives outside the loop: the event
    bus lets a UI or logger observe every message.

    Parameters
    ----------
    client : ``unillm.AsyncUnify``
        Pre-initialised client providing ``append_messages`` and ``generate``;
        every token sent to or received from the LLM flows through it.

    message : ``str | dict | list[str | dict]``
        The first user prompt, or a batch of already-structured messages that
        seed the conversation before unresolved tool calls are backfilled.

    tools : ``dict[str, Callable]``
        ``name → function`` for every callable the LLM may invoke. Each must be
        fully type-hinted with a concise docstring; both are converted to a
        tool schema via :pyfunc:`method_to_schema`.

    interject_queue : ``asyncio.Queue[str | dict]``
        Channel through which the outer application pushes additional user
        turns at any time. A dict payload has the shape
        ``{"message": str, "_parent_chat_context_continued": list[dict]}``.

    cancel_event : ``asyncio.Event``
        Set by the outer caller to request graceful shutdown: the loop cancels
        every running task and propagates ``asyncio.CancelledError``.

    max_consecutive_failures : ``int``, default ``3``
        After this many back-to-back tool exceptions the loop raises
        ``RuntimeError`` rather than crash-and-retry indefinitely.

    prune_tool_duplicates : ``bool``, default ``True``
        Drop model-requested tool calls with identical ``function.name`` and
        argument JSON, in place, before they reach chat history or scheduling.

    interrupt_llm_with_interjections : ``bool``, default ``True``
        When ``True`` an in-flight ``client.generate`` is cancelled the moment
        a new user turn arrives so the assistant can pivot immediately; when
        ``False`` the loop waits for the model to finish first.

    propagate_chat_context : ``ChatContextPropagation``, default ``LLM_DECIDES``
        Whether a filtered snapshot of this loop's conversation (genuine user
        turns and substantive assistant text only) is threaded into child
        tools that accept a ``_parent_chat_context`` keyword argument.
        ``ALWAYS`` injects on every such call, ``NEVER`` on none, and
        ``LLM_DECIDES`` exposes an ``include_parent_chat_context`` parameter
        the model may set to ``true`` (omission means no context). The
        ``_parent_chat_context`` argument itself is injected automatically and
        never exposed to the LLM.

    tool_policy : ``Callable | None``, default ``None``
        Dynamically controls tool exposure and whether a tool call is required
        on a given turn. Receives the turn index (from ``0``) and the full
        ``{name → callable}`` mapping, plus optionally the list of previously
        called tool names as a third positional argument. Returns
        ``(policy, tools)`` or ``(policy, tools, {"eager": bool})``: ``policy``
        is ``"auto"`` or ``"required"`` (fed straight into ``tool_choice``) and
        ``tools`` is the possibly-filtered mapping of base tools visible that
        turn. With ``eager=True`` the loop grants another LLM turn immediately
        after scheduling tool calls, without waiting for them, for as long as
        the policy keeps returning ``eager=True``; eager turns also withhold
        ``compress_context`` from the visible schema (forced over-threshold
        compression still applies). Omitting ``eager`` keeps the
        wait-for-results behaviour.

    parent_chat_context : ``list[dict] | None``
        Chat history passed from an outer loop. When a tool call opts into
        context (or ``propagate_chat_context`` is ``ALWAYS``), the filtered
        snapshot is forwarded to that inner tool on its first call and later
        calls receive only the messages added since, to avoid token waste.

    log_steps : ``bool | str``, default ``True``
        Step logging to ``LOGGER``: ``False`` for none, ``True`` for everything
        except system messages, ``"full"`` for everything.

    timeout : ``int | None``, default ``None``
        Activity-based timeout in seconds; the timer resets after each
        observable event (LLM response, tool completion, interjection). It
        guards against hung user-defined tools, not slow LLM inference:
        providers have their own timeouts, and an in-flight LLM call is
        awaited before the timeout is checked. ``None`` disables it.

    raise_on_limit : ``bool``, default ``False``
        If ``True``, exceeding the timeout or ``max_steps`` raises
        ``asyncio.TimeoutError`` or ``RuntimeError``; if ``False`` the loop
        terminates gracefully with a summary message.

    persist : ``bool``, default ``False``
        If ``True``, content without tool calls does not end the loop; it
        blocks on ``interject_queue`` and grants the LLM another turn when an
        interjection arrives, so one loop can process many events over time.
        The loop then ends only via ``cancel_event`` or ``stop_event``.

    time_awareness : ``bool``, default ``False``
        If ``True``, a time-context system message is injected at the start of
        the conversation and refreshed after each tool completion, giving the
        LLM wall-clock time and tool execution durations. If ``False`` the
        time-context table is omitted and no tool timing is tracked.

    Returns
    -------
    str
        The assistant's final plain-text reply after every tool result has
        been fed back into the conversation.
    """
    cfg = LoopConfig(loop_id, lineage, TOOL_LOOP_LINEAGE.get([]))
    # The outer handle shares the loop's resolved label so steering logs
    # (stop/pause/resume/interject/ask) line up with the tool loop's, and the
    # resolved lineage so event payloads carry the full parent->child stack
    # even when emitted outside the tool loop ContextVar scope.
    with suppress(Exception):
        if outer_handle_container and outer_handle_container[0] is not None:
            setattr(outer_handle_container[0], "_log_label", cfg.label)
            setattr(outer_handle_container[0], "_log_hierarchy", list(cfg.lineage))
            setattr(outer_handle_container[0], "_loop_cfg", cfg)
    logger = LoopLogger(cfg, log_steps)

    # When UNILLM_LOG_DIR is set each LLM call writes a request+response file.
    # The pending callback fires at the start of generate() (before inference),
    # so the "LLM thinking…" line can carry the log file path.
    if log_steps:
        client.set_on_log_file_pending(
            lambda path: logger.emit_thinking_with_path(path),
        )

    time_ctx: Optional[TimeContext] = create_time_context() if time_awareness else None
    _token = TOOL_LOOP_LINEAGE.set(cfg.lineage)

    def _apply_reasoning_model_compat(gen_kwargs: dict, tool_choice: str) -> Callable:
        """Return the effective preprocess callable. Provider-specific thinking
        mode compliance lives in unillm's provider preprocessing, so the loop
        itself stays provider-agnostic."""
        return preprocess_msgs

    stop_event = stop_event or asyncio.Event()

    _initial_user_message = copy.deepcopy(message)

    # Normalize response_format once. LLM-supplied nested tool args may pass a
    # JSON Schema dict / JSON string rather than a Pydantic class; accept those
    # so final_response can be injected. Unsupported values disable structured
    # mode rather than forcing tool_choice=required with no escape hatch.
    _rf_norm: Optional[NormalizedResponseFormat] = None
    if response_format is not None:
        try:
            _rf_norm = normalize_response_format(response_format)
        except Exception as _exc:  # noqa: BLE001
            logger.error(
                f"response_format normalization failed ({_exc!r}); "
                f"continuing without structured-output mode.",
            )
            _rf_norm = None

    # Tell the model up-front when structured output is expected so it can plan
    # with the final JSON shape in mind; enforcement happens through the
    # response-submission tool during the loop. The hint goes into a separate
    # system message appended below, never into the caller's original.
    _response_format_hint: str | None = None
    if _rf_norm is not None:
        _response_format_hint = (
            "## Response Format\n"
            "NOTE: After completing all tool calls, submit your final answer via "
            "the response tool as JSON that conforms to the following schema. "
            "Do NOT include any extra keys or commentary.\n"
            + json.dumps(_rf_norm.answer_json_schema, indent=2)
        )

    runtime_state = runtime_state or ToolLoopRuntimeState()

    # ── runtime guards ────────────────────────────────────────────────────
    # A run with no step ceiling ends only when the model chooses to stop, so
    # one that never converges keeps calling tools — and billing — forever.
    # Fall back to the configured ceiling when a caller expresses no opinion,
    # which bounds every entry point at once; an explicit ``max_steps`` still
    # wins for callers that legitimately need more.
    if max_steps is None:
        from unify.settings import SETTINGS as _SETTINGS

        configured_max_steps = _SETTINGS.UNIFY_MAX_TOOL_LOOP_STEPS
        max_steps = configured_max_steps if configured_max_steps > 0 else None

    timer: TimeoutTimer = TimeoutTimer(
        timeout=timeout,
        max_steps=max_steps,
        raise_on_limit=raise_on_limit,
        client=client,
        message_count_offset=runtime_state.message_count_offset,
    )
    _msg_dispatcher = LoopMessageDispatcher(client, cfg, timer)
    parent_chat_context_safe = make_messages_safe_for_context_dump(parent_chat_context)

    if log_steps:
        if log_steps == "full":
            if parent_chat_context_safe:
                from .utils import format_json_for_log

                logger.info(
                    f"Parent Context: {format_json_for_log(parent_chat_context_safe)}",
                    prefix=ICONS["tool_seeding"],
                )
            logger.info(
                f"System Message: {client.system_message}",
                prefix=ICONS["system_message"],
            )
        # A seeded batch is logged per item below, not here.
        if not isinstance(message, list):
            logger.info(f"Request: {message}", prefix=ICONS["request"])

    import time as _setup_time

    _setup_t0 = _setup_time.perf_counter()

    def _setup_elapsed() -> str:
        return f"{(_setup_time.perf_counter() - _setup_t0) * 1000:.0f}ms"

    # ── Runtime-context system header ─────────────────────────────────────
    # One system message at the start of the conversation says who the "user"
    # is (which manager is calling this loop) and, for nested loops, what the
    # broader conversation is. ``_runtime_context=True`` identifies it later;
    # ``_ctx_header=True`` marks it for filtering when forwarding to inner tools.

    # The parent caller is the second-to-last lineage entry (the last is this
    # loop's own id).
    _effective_caller_description = caller_description
    if _effective_caller_description is None and lineage and len(lineage) >= 2:
        try:
            parent_label = lineage[-2]
            # "ClassName.method" or "ClassName.method(id)" → "ClassName"
            parent_class = parent_label.split(".")[0].split("(")[0]
            for prefix in ("Simulated", "Base"):
                if parent_class.startswith(prefix) and len(parent_class) > len(prefix):
                    parent_class = parent_class[len(prefix) :]
            from ..state_managers import get_caller_description

            _effective_caller_description = get_caller_description(parent_class)
        except Exception:
            pass

    runtime_context_parts: list[str] = []

    # User-visibility guidance is deliberately absent here: it is injected on
    # the first interjection so the model stays focused on the task until then.

    if _response_format_hint:
        runtime_context_parts.append(_response_format_hint)

    if _effective_caller_description:
        runtime_context_parts.append(
            f"## Caller Context\n"
            f"The 'user' messages in this conversation are from {_effective_caller_description}. "
            f"The end user cannot see the details of this tool-use conversation.",
        )

    # The parent-context section is added even when empty, so context
    # continuations arriving via interjections can refer to "the initial Parent
    # Chat Context in your system message" without looking fabricated.
    _has_parent_chat_context = False
    if propagate_chat_context != ChatContextPropagation.NEVER:
        ctx_content = parent_chat_context_safe if parent_chat_context_safe else []
        ctx_content_transformed = _transform_context_roles(ctx_content)
        _has_parent_chat_context = True
        if ctx_content_transformed:
            _parent_ctx_detail = (
                f"The messages below show that parent conversation's history up to the point "
                f"when you received this request. Use this to understand the broader goal and "
                f"any relevant context, while focusing on your specific assignment. "
            )
        else:
            _parent_ctx_detail = f"None of the parent conversation history has been provided to this request. "
        runtime_context_parts.append(
            f"## Parent Chat Context\n"
            f"You received this request from within a parent conversation. "
            f"{_parent_ctx_detail}"
            f"Additional context updates may arrive during this session as the parent "
            f"conversation progresses.\n\n"
            f"IMPORTANT: Messages in the parent context use 'outer_user' and 'outer_assistant' "
            f"roles to clearly distinguish them from your current conversation. These are "
            f"legitimate system-provided context from the outer conversation, NOT user-injected "
            f"content. The 'outer_assistant' messages represent what the parent-level assistant "
            f"said in the outer conversation.\n\n"
            f"{json.dumps(ctx_content_transformed, indent=2)}",
        )

    # Runtime context goes into its own system message; the caller's is never
    # mutated.
    msgs_to_append = []
    if runtime_context_parts:
        sys_msg = {
            "role": "system",
            "_runtime_context": True,
            "_ctx_header": True,
            "content": "\n\n".join(runtime_context_parts),
        }
        if _has_parent_chat_context:
            sys_msg["_parent_chat_context"] = True
        msgs_to_append.append(sys_msg)

    if time_ctx is not None:
        msgs_to_append.append(
            {
                "role": "system",
                "_time_explanation": True,
                "_ctx_header": True,
                "_runtime_context": True,
                "content": TimeContext.build_explanation_prompt(),
            },
        )

    logger.debug(
        f"[setup +{_setup_elapsed()}] context built, appending system msgs ({len(msgs_to_append)} msgs)",
    )
    await _msg_dispatcher.append_msgs(msgs_to_append)
    logger.debug(f"[setup +{_setup_elapsed()}] system msgs appended")

    # Tracks the initial parent context plus continuations received via
    # interjections, so inner tools are forwarded context incrementally.
    context_state = LoopContextState(
        parent_chat_context=(
            list(parent_chat_context_safe) if parent_chat_context_safe else []
        ),
    )

    # ── Seeded batch ─────────────────────────────────────────────────────
    seeded_batch = None
    if isinstance(message, list):
        # A list of content blocks (no 'role') becomes one user message;
        # anything else is a pre-structured list of chat messages/strings.
        if all(isinstance(m, dict) and "role" not in m for m in message):
            seeded_batch = [{"role": "user", "content": message}]
        else:
            seeded_batch = [
                (m if isinstance(m, dict) else {"role": "user", "content": m})
                for m in message
            ]

        logger.debug(
            f"[setup +{_setup_elapsed()}] appending seeded batch ({len(seeded_batch)} msgs)",
        )
        await _msg_dispatcher.append_msgs(seeded_batch)
        logger.debug(f"[setup +{_setup_elapsed()}] seeded batch appended")

    # ── Loop-owned tools, when the caller opted in ────────────────────────
    if clarification_queues is not None:
        from ..llm_helpers import make_request_clarification_tool

        _clar_up_q, _clar_down_q = clarification_queues
        tools["request_clarification"] = make_request_clarification_tool(
            _clar_up_q,
            _clar_down_q,
            on_request=on_clarification_request,
            on_answer=on_clarification_answer,
        )

    if on_notify is not None:
        from ..llm_helpers import make_send_notification_tool

        tools["send_notification"] = make_send_notification_tool(on_notify=on_notify)

    # ToolsData must exist before the preflight backfill below can schedule.
    logger.debug(f"[setup +{_setup_elapsed()}] initialising ToolsData")
    tools_data: ToolsData = ToolsData(
        tools,
        client=client,
        logger=logger,
        time_ctx=time_ctx,
        extra_ask_tools=extra_ask_tools,
        completed_askable_tools=completed_askable_tools,
        call_counts=runtime_state.call_counts,
    )
    logger.debug(
        f"[setup +{_setup_elapsed()}] ToolsData ready ({len(tools_data.normalized)} tools)",
    )

    _alias_lookup = {
        name: spec.display_label
        for name, spec in tools_data.normalized.items()
        if spec.display_label
    }
    cfg.tool_alias_lookup = _alias_lookup or None

    consecutive_failures = _LoopToolFailureTracker(
        max_consecutive_failures,
        runtime_state,
    )
    assistant_meta: Dict[int, Dict[str, Any]] = {}

    _max_input_tokens = unillm.get_max_input_tokens(client.endpoint)
    _over_threshold = False
    _full_completion: Any = None

    # Whether tool_policy accepts a third positional arg (called_tools
    # history), computed once to avoid per-turn introspection.
    _policy_accepts_history = False
    if tool_policy is not None:
        with suppress(Exception):
            _sig = inspect.signature(tool_policy)
            _positional_kinds = (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            _n_positional = sum(
                1 for p in _sig.parameters.values() if p.kind in _positional_kinds
            )
            _policy_accepts_history = _n_positional >= 3
    # Outer handles introspect running nested handles through attributes on
    # this Task: task_info (used by ask/stop helpers), clarification_channels
    # (so handle-level methods route answers without involving the LLM),
    # get_ask_tools (so handle.ask() reaches inner handles) and completed
    # tool metadata including handle refs.
    with suppress(Exception):
        _self_task = asyncio.current_task()
        if _self_task is not None:
            setattr(_self_task, "task_info", tools_data.info)  # type: ignore[attr-defined]
            setattr(
                _self_task,
                "clarification_channels",
                tools_data.clarification_channels,
            )
            setattr(_self_task, "get_ask_tools", tools_data.get_ask_tools)  # type: ignore[attr-defined]
            setattr(_self_task, "get_completed_tool_metadata", lambda: dict(tools_data._completed_askable_tools))  # type: ignore[attr-defined]

    # Preflight repair: backfill pre-existing assistant tool_calls without
    # replies, oldest → newest. Each entry is repaired inside its own
    # try/except: prune_over_quota_tool_calls may raise (a below-watermark
    # mutation refused on a resumed client whose watermark carried over), and
    # one blanket suppress around the whole loop would silently abandon every
    # entry after the one that raised instead of just skipping it.
    logger.debug(f"[setup +{_setup_elapsed()}] preflight repair start")
    with suppress(Exception):
        unreplied = find_unreplied_assistant_entries(client)
        if unreplied:
            for entry in unreplied:
                try:
                    amsg = entry["assistant_msg"]
                    tools_data.prune_over_quota_tool_calls(amsg)
                    if prune_tool_duplicates and amsg.get("tool_calls"):
                        unique, pruned = prune_duplicate_tool_calls(amsg["tool_calls"])
                        if pruned:
                            amsg["tool_calls"] = unique
                            entry["missing"] = [
                                cid for cid in entry["missing"] if cid not in pruned
                            ]
                    missing_ids = set(entry["missing"])
                    if not missing_ids:
                        continue
                    await schedule_missing_for_message(
                        amsg,
                        missing_ids,
                        tools_data=tools_data,
                        context_state=context_state,
                        propagate_chat_context=propagate_chat_context,
                        assistant_meta=assistant_meta,
                        client=client,
                        msg_dispatcher=_msg_dispatcher,
                    )
                except Exception as exc:
                    logger.error(
                        f"Preflight repair failed for one assistant entry; "
                        f"continuing with the rest: {exc}",
                        prefix="🚨",
                    )

    # ── Steering: target selection + per-child dispatch ──────────────────
    def _select_steering_targets(
        method: str,
        payload: dict | None,
    ) -> list[Tuple[asyncio.Task, "ToolCallMetadata"]]:
        """
        Choose which child tool calls receive a steering signal:
          - clarify: the specified call_id only (exact or suffix match)
          - pause/resume/stop: all children
          - interject/ask/custom: not auto-forwarded to children
        """
        base = str(method or "").lower().strip()
        payload = payload or {}
        selected: list[Tuple[asyncio.Task, ToolCallMetadata]] = []
        if base == "clarify":
            try:
                target_call_id = payload.get("call_id")
            except Exception:
                target_call_id = None
            if isinstance(target_call_id, str) and target_call_id:
                for t, inf in list(tools_data.info.items()):
                    try:
                        if str(inf.call_id) == target_call_id or str(
                            inf.call_id,
                        ).endswith(target_call_id):
                            selected.append((t, inf))
                            break
                    except Exception:
                        continue
            return selected
        if base in ("pause", "resume", "stop"):
            for t, inf in list(tools_data.info.items()):
                try:
                    # Included even before a handle is adopted, so pause/resume
                    # can still toggle pause_event.
                    selected.append((t, inf))
                except Exception:
                    continue
            return selected
        return selected

    async def _dispatch_steering_to_child(
        method: str,
        payload: dict | None,
        inf: "ToolCallMetadata",
    ) -> None:
        """
        Execute one steering operation on a single child:
          - interject: prefer the private interject_queue; else handle.interject(...)
          - ask: not forwarded here (see below)
          - pause/resume: handle.pause()/resume() when available; else toggle pause_event
          - stop: handle.stop(...)
          - clarify: put the answer onto the clarification down-queue (by call_id)
          - default: best-effort generic forward to the handle
        """
        base = str(method or "").lower().strip()
        args = dict(payload or {})
        h = getattr(inf, "handle", None)
        if base == "interject":
            try:
                new_text = args.get("content") if isinstance(args, dict) else None
                if new_text is None and isinstance(args, dict):
                    new_text = args.get("message")
            except Exception:
                new_text = None
            iq = getattr(inf, "interject_queue", None)
            if iq is not None:
                _ctx_cont = (
                    args.get("_parent_chat_context_cont")
                    if isinstance(args, dict)
                    else None
                )
                if _ctx_cont is None:
                    # Bare text when there is no continuation context: many
                    # simple tools declare `_interject_queue` and just
                    # `await _interject_queue.get()` expecting the raw string,
                    # so wrapping unconditionally would break them.
                    await iq.put(new_text)
                else:
                    # Same payload shape as AsyncToolLoopHandle.interject
                    # (unify/common/async_tool_loop.py), so a call routed
                    # through this queue shortcut carries the same
                    # continuation context as one routed through
                    # handle.interject() below.
                    await iq.put(
                        {
                            "message": new_text,
                            "_parent_chat_context_continued": _ctx_cont,
                            "trigger_immediate_llm_turn": (
                                args.get("trigger_immediate_llm_turn", True)
                                if isinstance(args, dict)
                                else True
                            ),
                            "suppress_response_notification": (
                                args.get("suppress_response_notification", False)
                                if isinstance(args, dict)
                                else False
                            ),
                        },
                    )
                return
            if h is not None:
                await forward_handle_call(  # type: ignore[name-defined]
                    h,
                    "interject",
                    args if isinstance(args, dict) else {},
                    fallback_positional_keys=["content", "message"],
                )
            return
        if base == "ask":
            # The outer ask() starts a dedicated inspection loop and injects
            # ask_* tool calls that adopt and run nested ask handles;
            # forwarding here would duplicate those calls.
            return
        if base == "pause":
            if h is not None and hasattr(h, "pause"):
                await forward_handle_call(  # type: ignore[name-defined]
                    h,
                    "pause",
                    args if isinstance(args, dict) else {},
                )
                return
            ev = getattr(inf, "pause_event", None)
            if ev is not None:
                ev.clear()
            return
        if base == "resume":
            if h is not None and hasattr(h, "resume"):
                await forward_handle_call(  # type: ignore[name-defined]
                    h,
                    "resume",
                    args if isinstance(args, dict) else {},
                )
                return
            ev = getattr(inf, "pause_event", None)
            if ev is not None:
                ev.set()
            return
        if base == "stop":
            if h is not None and hasattr(h, "stop"):
                await forward_handle_call(  # type: ignore[name-defined]
                    h,
                    "stop",
                    args if isinstance(args, dict) else {},
                    fallback_positional_keys=["reason"],
                )
            return
        if base == "clarify":
            with suppress(Exception):
                _cid = str(inf.call_id)
                _clar_map = tools_data.clarification_channels
                # Exact id first, then suffix lookup.
                if _cid in _clar_map:
                    down_q = _clar_map[_cid][1]
                else:
                    down_q = None
                    for k, (_u, _d) in list(_clar_map.items()):
                        if str(k).endswith(_cid[-6:]):
                            down_q = _d
                            break
                if down_q is not None:
                    await down_q.put((args or {}).get("answer"))
            return
        # Best-effort generic forward: strip the control keys, then try the
        # original name, its aliases and finally the base name.
        if h is not None:
            try:
                args.pop("_custom", None)
                aliases = list(args.pop("_aliases", []) or [])
            except Exception:
                aliases = []
            try:
                fb_keys = tuple(args.pop("_fallback", ()) or ())
            except Exception:
                fb_keys = ()
            try:
                original_name = str(method or "")
            except Exception:
                original_name = base
            candidates: list[str] = []
            if original_name:
                candidates.append(original_name)
            for nm in aliases:
                if isinstance(nm, str) and nm:
                    candidates.append(nm)
            if base and base not in candidates:
                candidates.append(base)
            for nm in candidates:
                try:
                    attr = getattr(h, nm, None)
                    if not callable(attr):
                        continue
                    await forward_handle_call(  # type: ignore[name-defined]
                        h,
                        nm,
                        args if isinstance(args, dict) else {},
                        fallback_positional_keys=fb_keys,
                    )
                    return
                except Exception:
                    continue

    async def _synthesize_mirrored_helper_calls(
        method: str,
        payload: dict | None = None,
    ) -> None:
        """
        Append an assistant message whose `steer` tool_calls mirror a steering
        command, ack each one immediately, then forward the steering to the
        target child handles. No LLM step is involved.
        """
        payload = payload or {}
        # "_inject_only" records the mirror without dispatching, so child
        # steering already performed elsewhere is not executed twice.
        inject_only = False
        try:
            inject_only = bool(payload.get("_inject_only"))
        except Exception:
            inject_only = False

        # Banner-deferral sentinels carry no tool acks.
        base_name = ""
        try:
            base_name = str(method or "").lower().strip()
        except Exception:
            base_name = ""
        if base_name == "_banner_after_first_llm":
            text = ""
            prefix = ""
            try:
                text = str((payload or {}).get("text") or "")
                prefix = str((payload or {}).get("prefix") or "")
            except Exception:
                text, prefix = "", ""
            if text:
                try:
                    logger.defer_after_first_llm(text, prefix=prefix)
                except Exception:
                    pass
            return

        # The stop log (and any chained banner) is deferred until after the
        # first LLM thinking line.
        if base_name == "stop":
            reason_txt = ""
            try:
                r = payload.get("reason")
                if isinstance(r, str) and r:
                    reason_txt = r
            except Exception:
                reason_txt = ""
            suffix = f" – reason: {reason_txt}" if reason_txt else ""
            try:
                logger.defer_after_first_llm(
                    f"Stop requested{suffix}",
                    prefix=ICONS["stop_requested"],
                )
            except Exception:
                pass
            try:
                banner = payload.get("_after_first_llm_banner")
                if isinstance(banner, dict):
                    btxt = str(banner.get("text") or "")
                    bpf = str(banner.get("prefix") or "")
                    if btxt:
                        logger.defer_after_first_llm(btxt, prefix=bpf)
            except Exception:
                pass

        targets: list[Tuple[asyncio.Task, ToolCallMetadata]] = _select_steering_targets(
            method,
            payload if isinstance(payload, dict) else {},
        )
        if not targets:
            return

        base = str(method or "").lower().strip()

        def _steer_payload_for(base_action: str) -> Optional[str]:
            if base_action == "interject":
                return payload.get("message") or payload.get("content")
            if base_action == "ask":
                return payload.get("question")
            if base_action == "stop":
                return payload.get("reason")
            if base_action == "clarify":
                return payload.get("answer")
            return None  # pause/resume carry no payload

        # One assistant message with one `steer` tool_call per target, in the
        # same structured-args shape the LLM itself emits, so programmatic
        # steering is acked and dispatched through the same `steer`
        # schema/transcript convention rather than a parallel one. The full
        # forward kwargs (minus control keys) are kept aside for dispatch.
        tool_calls = []
        args_by_id: dict[str, Any] = {}
        for _t, inf in targets:
            try:
                try:
                    forward_args = dict(payload or {})
                except Exception:
                    forward_args = {}
                for _k in ("_custom", "_aliases", "_fallback"):
                    try:
                        forward_args.pop(_k, None)
                    except Exception:
                        pass

                steer_args: dict[str, Any] = {
                    "call_id": inf.call_id,
                    "action": base,
                }
                _pl = _steer_payload_for(base)
                if _pl is not None:
                    steer_args["payload"] = _pl

                call_id = f"mirror_{short_id(6)}"
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "steer",
                            "arguments": json.dumps(steer_args),
                        },
                    },
                )
                args_by_id[call_id] = (forward_args, inf)
            except Exception:
                continue
        if not tool_calls:
            return

        assistant_msg = {"role": "assistant", "content": "", "tool_calls": tool_calls}
        await _msg_dispatcher.append_msgs([assistant_msg])
        with suppress(Exception):
            await to_event_bus(assistant_msg, cfg, kind=ToolLoopKind.STEERING_HELPER)
        assistant_meta[id(assistant_msg)] = {"results_count": 0}

        # Ack each call, then forward the steering to its target handle.
        for call in tool_calls:
            try:
                cid = call.get("id")
                if not isinstance(cid, str):
                    continue
                args, inf = args_by_id.get(cid, (None, None))
                with suppress(Exception):
                    await acknowledge_helper_call(  # type: ignore[name-defined]
                        assistant_msg,
                        cid,
                        "steer",
                        call["function"].get("arguments", "{}"),
                        assistant_meta=assistant_meta,
                        client=client,
                        msg_dispatcher=_msg_dispatcher,
                    )
                if (not inject_only) and (inf is not None):
                    await _dispatch_steering_to_child(base, args, inf)
            except Exception:
                continue

    # ── Initial user message (single-message path) ──────────────────────
    if seeded_batch is None:
        if isinstance(message, dict):
            initial_user_msg = message
        else:
            initial_user_msg = {"role": "user", "content": message}
        if time_ctx is not None and isinstance(initial_user_msg.get("content"), str):
            initial_user_msg["content"] = time_ctx.prefix_user_message(
                initial_user_msg["content"],
            )
        await _msg_dispatcher.append_msgs([initial_user_msg])

    async def _handle_limit_reached(reason: str) -> str:
        """
        Terminate gracefully when *timeout* or *max_steps* is exceeded and
        `raise_on_limit` is *False*: stop every pending tool (via
        handle.stop() when available), cancel the tasks, and append a short
        assistant notice.
        """
        for task in list(tools_data.pending):
            with suppress(Exception):
                inf = tools_data.info.get(task)
                if inf is not None and inf.handle is not None and hasattr(inf.handle, "stop"):  # type: ignore[attr-defined]
                    await maybe_await(inf.handle.stop())
            if not task.done():
                task.cancel()
        await asyncio.gather(*tools_data.pending, return_exceptions=True)
        tools_data.pending.clear()

        notice = {
            "role": "assistant",
            "content": f"🔚 Terminating early: {reason}",
        }
        await _msg_dispatcher.append_msgs([notice])
        if log_steps:
            logger.info(f"Early exit – {reason}", prefix=ICONS["early_exit"])
        return notice["content"]

    async def _handle_clarification(
        src_task: asyncio.Task,
        question_payload: Any,
    ) -> None:
        question_text = ""
        try:
            if isinstance(question_payload, dict):
                question_text = question_payload.get("question", "")
            else:
                question_text = str(question_payload)
        except Exception:
            question_text = str(question_payload)

        call_id = tools_data.info[src_task].call_id
        tool_name = tools_data.info[src_task].name

        tools_data.info[src_task].waiting_for_clarification = True

        # Coalesce-then-freeze into a [clarification <call_id>] tail message,
        # never the tool_reply_msg pending stub, which stays byte-frozen once
        # sent. The model answers off this tail message via steer(clarify).
        await tools_data.record_clarification(
            tools_data.info[src_task],
            call_id,
            question_text,
            _msg_dispatcher,
        )

        try:
            logger.info(
                f"Clarification requested – {tool_name}: {question_text}",
                prefix=ICONS["clarification"],
            )
        except Exception:
            pass

        # Programmatic clarification event for the outer handle.
        with suppress(Exception):
            outer = outer_handle_container[0] if outer_handle_container else None
            if outer is not None and hasattr(outer, "_clar_q"):
                await outer._clar_q.put(
                    {
                        "type": "clarification",
                        "call_id": call_id,
                        "tool_name": tool_name,
                        "question": question_text,
                    },
                )

    async def _handle_notification(src_task: asyncio.Task, payload: Any) -> None:
        call_id = tools_data.info[src_task].call_id
        tool_name = tools_data.info[src_task].name

        pretty = ToolsData._pretty_tool_payload(tool_name, payload)

        try:
            if isinstance(payload, dict):
                _msg_txt = str(
                    payload.get("message") or payload.get("status") or payload,
                )
            else:
                _msg_txt = str(payload)
            logger.info(
                f"Notification from {tool_name}: {_msg_txt}",
                prefix=ICONS["notification"],
            )
        except Exception:
            pass

        # Coalesce-then-freeze into a separate [progress <call_id>] tail
        # message, never the tool_reply_msg placeholder, which must stay
        # byte-frozen once sent: rewriting a placeholder in place, mid-history,
        # breaks the cached prompt prefix on every turn a sub-agent runs.
        await tools_data.record_progress(
            tools_data.info[src_task],
            call_id,
            pretty,
            _msg_dispatcher,
        )

        # Programmatic notification event for the outer handle.
        with suppress(Exception):
            outer = outer_handle_container[0] if outer_handle_container else None
            if outer is not None and hasattr(outer, "_notification_q"):
                event_payload = (
                    payload if isinstance(payload, dict) else {"message": str(payload)}
                )
                await outer._notification_q.put(
                    {
                        "type": "notification",
                        "call_id": call_id,
                        "tool_name": tool_name,
                        **event_payload,
                    },
                )

    # True whenever the LLM must get an immediate turn before the loop waits
    # again (user interjection, clarification answer, etc.).
    llm_turn_required = False
    # A patient interjection (trigger_immediate_llm_turn=False) arriving while
    # the LLM is already thinking earns exactly one extra LLM step after the
    # current one, unless another event triggers a turn anyway.
    deferred_llm_turn = False
    # Bounded retries for a terminal turn that returns empty content with no
    # substantive answer anywhere else in the conversation to fall back on.
    _empty_final_answer_retries = 0
    _MAX_EMPTY_FINAL_ANSWER_RETRIES = 1

    logger.debug(f"[setup +{_setup_elapsed()}] entering main loop")

    try:
        while True:
            # ── Pause gate ───────────────────────────────────────────────
            # Tool completions and cancellation are still handled while
            # paused; the LLM never speaks.
            if not pause_event.is_set():
                # Mirror steering sentinels are processed immediately so
                # control signals (pause/resume/stop) still reach child
                # handles without waiting for resume.
                try:
                    while True:
                        try:
                            _extra = interject_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        if isinstance(_extra, dict) and "_mirror" in _extra:
                            _ms = _extra.get("_mirror") or {}
                            _m = _ms.get("method")
                            _kw = _ms.get("kwargs") or {}
                            if isinstance(_m, str) and _m:
                                try:
                                    merged = dict(_kw if isinstance(_kw, dict) else {})
                                except Exception:
                                    merged = {}
                                try:
                                    if _ms.get("_custom"):
                                        merged["_custom"] = True
                                except Exception:
                                    pass
                                try:
                                    if "_aliases" in _ms:
                                        merged["_aliases"] = list(
                                            _ms.get("_aliases") or [],
                                        )
                                except Exception:
                                    pass
                                try:
                                    if "_fallback" in _ms:
                                        merged["_fallback"] = list(
                                            _ms.get("_fallback") or [],
                                        )
                                except Exception:
                                    pass
                                await _synthesize_mirrored_helper_calls(_m, merged)
                            continue
                        else:
                            # Non-mirror entries wait until resume.
                            await interject_queue.put(_extra)
                            break
                except Exception:
                    pass
                # Unreplied assistant tool_calls are scheduled while paused so
                # base tools start in the paused state and placeholders appear.
                with suppress(Exception):
                    if True:
                        if unreplied := find_unreplied_assistant_entries(client):
                            last_problem = unreplied[-1]
                            amsg = last_problem["assistant_msg"]
                            missing_ids = set(last_problem["missing"])
                            if id(amsg) not in assistant_meta:
                                await schedule_missing_for_message(
                                    amsg,
                                    missing_ids,
                                    tools_data=tools_data,
                                    context_state=context_state,
                                    propagate_chat_context=propagate_chat_context,
                                    assistant_meta=assistant_meta,
                                    client=client,
                                    msg_dispatcher=_msg_dispatcher,
                                    initial_paused=True,
                                )
                                await ensure_placeholders_for_pending(
                                    tools_data=tools_data,
                                    assistant_meta=assistant_meta,
                                    client=client,
                                    msg_dispatcher=_msg_dispatcher,
                                    time_ctx=time_ctx,
                                )
                # Let pending tool tasks finish, or wait until the loop is
                # resumed / cancelled. Each waiter is a Task because
                # asyncio.wait() requires them.
                if tools_data.pending:
                    pause_waiter = asyncio.create_task(
                        pause_event.wait(),
                        name="PauseEventWait",
                    )
                    cancel_waiter = asyncio.create_task(
                        cancel_event.wait(),
                        name="CancelEventWait",
                    )
                    waiters = tools_data.pending | {
                        pause_waiter,
                        cancel_waiter,
                    }

                    done, _ = await asyncio.wait(
                        waiters,
                        timeout=0.1,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # Unused waiters must not dangle.
                    for w in (pause_waiter, cancel_waiter):
                        if w not in done and not w.done():
                            w.cancel()
                            await asyncio.gather(w, return_exceptions=True)

                    for t in _sort_completed_tasks_by_call_id(
                        done & tools_data.pending,
                        tools_data,
                    ):
                        await tools_data.process_completed_task(
                            task=t,
                            consecutive_failures=consecutive_failures,
                            outer_handle_container=outer_handle_container,
                            assistant_meta=assistant_meta,
                            msg_dispatcher=_msg_dispatcher,
                        )
                    if cancel_event.is_set():
                        # The mirrored stop has already reached children.
                        raise asyncio.CancelledError
                    continue  # remain paused: do not allow the LLM to speak while paused
                else:
                    # Nothing running: schedule any missing tool replies from
                    # the last assistant turn, then idle until resumed or
                    # cancelled.
                    with suppress(Exception):
                        if unreplied := find_unreplied_assistant_entries(client):
                            last_problem = unreplied[-1]
                            amsg = last_problem["assistant_msg"]
                            missing_ids = set(last_problem["missing"])
                            if id(amsg) not in assistant_meta:
                                await schedule_missing_for_message(
                                    amsg,
                                    missing_ids,
                                    tools_data=tools_data,
                                    context_state=context_state,
                                    propagate_chat_context=propagate_chat_context,
                                    assistant_meta=assistant_meta,
                                    client=client,
                                    msg_dispatcher=_msg_dispatcher,
                                    initial_paused=True,
                                )
                                await ensure_placeholders_for_pending(
                                    tools_data=tools_data,
                                    assistant_meta=assistant_meta,
                                    client=client,
                                    msg_dispatcher=_msg_dispatcher,
                                    time_ctx=time_ctx,
                                )
                    done, _ = await asyncio.wait(
                        {
                            asyncio.create_task(
                                pause_event.wait(),
                                name="PauseEventWait",
                            ),
                            asyncio.create_task(
                                cancel_event.wait(),
                                name="CancelEventWait",
                            ),
                        },
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if pause_event.is_set():
                        continue  # back to main loop, un-paused

                    if cancel_event.is_set():
                        # The mirrored stop has already reached children.
                        raise asyncio.CancelledError
                    continue  # top-of-loop, still paused

            if timer.has_exceeded_time():
                return await _handle_limit_reached(
                    f"timeout ({timeout}s) exceeded",
                )

            if timer.has_exceeded_msgs():
                return await _handle_limit_reached(
                    f"max_steps ({max_steps}) exceeded",
                )

            # Outstanding assistant tool_calls missing replies are repaired
            # before any new user interjection is appended. Only the latest
            # such assistant message is considered, and only once.
            with suppress(Exception):
                if unreplied := find_unreplied_assistant_entries(client):
                    last_problem = unreplied[-1]
                    amsg = last_problem["assistant_msg"]
                    missing_ids = set(last_problem["missing"])
                    if id(amsg) not in assistant_meta:
                        await schedule_missing_for_message(
                            amsg,
                            missing_ids,
                            tools_data=tools_data,
                            context_state=context_state,
                            propagate_chat_context=propagate_chat_context,
                            assistant_meta=assistant_meta,
                            client=client,
                            msg_dispatcher=_msg_dispatcher,
                        )

            # ── Drain queued interjections ───────────────────────────────
            # This must run before waiting on tool completion so a fast
            # typist can still get a question in while long-running tools
            # are in flight.
            _suppress_persist_response = False
            _had_interjections = False
            while True:
                try:
                    extra = interject_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                _is_sentinel = isinstance(extra, dict) and (
                    "_mirror" in extra
                    or "_transcript_note" in extra
                    or "_compact_transcript" in extra
                    or extra.get("_replay")
                )
                # Transcript-note sentinel: a background process (e.g. a
                # storage review) leaves a loop-authored note in the
                # transcript without granting an LLM turn — the model reads
                # it whenever it next speaks.
                if isinstance(extra, dict) and "_transcript_note" in extra:
                    try:
                        _note = str(
                            (extra.get("_transcript_note") or {}).get("text") or "",
                        )
                        if _note:
                            await _msg_dispatcher.append_msgs(
                                [loop_user_notice(_note)],
                            )
                    except Exception:
                        pass
                    continue
                # Transcript-compaction sentinel: a completed storage review
                # has consolidated the covered turns, so their raw tool
                # payloads shed their bulk. Processed only at drain points —
                # never mid-dispatch.
                if isinstance(extra, dict) and "_compact_transcript" in extra:
                    try:
                        _n = int(
                            (extra.get("_compact_transcript") or {}).get(
                                "reviewed_messages",
                            )
                            or 0,
                        )
                        if _n > 0:
                            compact_reviewed_messages(client, _n)
                    except Exception:
                        pass
                    continue
                if not _is_sentinel:
                    if not _had_interjections:
                        _had_interjections = True
                        _suppress_persist_response = True
                    if not isinstance(extra, dict) or not extra.get(
                        "suppress_response_notification",
                        False,
                    ):
                        _suppress_persist_response = False

                # "_llm_turn" lets an interjection choose how it schedules the
                # LLM: "none", "deferred", or the default immediate turn
                # (which also clears any prior deferral).
                llm_policy = "immediate"
                try:
                    if isinstance(extra, dict):
                        llm_policy = str(extra.get("_llm_turn") or "immediate")
                except Exception:
                    llm_policy = "immediate"
                if llm_policy == "none":
                    pass
                elif llm_policy == "deferred":
                    try:
                        deferred_llm_turn = True
                    except Exception:
                        pass
                else:
                    llm_turn_required = True
                    try:
                        deferred_llm_turn = False
                    except Exception:
                        pass
                # Mirrored steering sentinel: synthesize helper tool_calls now.
                try:
                    if isinstance(extra, dict) and "_mirror" in extra:
                        _ms = extra.get("_mirror") or {}
                        _m = _ms.get("method")
                        _kw = _ms.get("kwargs") or {}
                        if isinstance(_m, str) and _m:
                            try:
                                merged = dict(_kw if isinstance(_kw, dict) else {})
                            except Exception:
                                merged = {}
                            try:
                                if _ms.get("_custom"):
                                    merged["_custom"] = True
                            except Exception:
                                pass
                            try:
                                if "_aliases" in _ms:
                                    merged["_aliases"] = list(_ms.get("_aliases") or [])
                            except Exception:
                                pass
                            try:
                                if "_fallback" in _ms:
                                    merged["_fallback"] = list(
                                        _ms.get("_fallback") or [],
                                    )
                            except Exception:
                                pass
                            await _synthesize_mirrored_helper_calls(_m, merged)
                            continue
                except Exception:
                    pass
                # Replay sentinel: grant the next LLM turn without appending
                # any message, preserving transcript fidelity after resume.
                try:
                    if isinstance(extra, dict) and extra.get("_replay"):
                        llm_turn_required = True
                        continue
                except Exception:
                    pass
                # User-visible history lives on the outer handle; fall back to
                # the original user prompt if it is unavailable.
                history_lines: list[str] = []
                try:
                    outer_handle = (
                        outer_handle_container[0] if outer_handle_container else None
                    )
                    uvh = (
                        getattr(outer_handle, "_user_visible_history", [])
                        if outer_handle
                        else []
                    )
                    for _m in uvh:
                        role = _m.get("role")
                        _content = _m.get("content")
                        if isinstance(_content, dict):
                            _text = str(_content.get("message", "")).strip()
                        else:
                            _text = str(_content or "").strip()
                        if role in ("user", "assistant") and _text:
                            history_lines.append(f"{role}: {_text}")
                except Exception:
                    try:
                        first_user = next(
                            (
                                m.get("content", "")
                                for m in client.messages
                                if m.get("role") == "user"
                            ),
                            "",
                        )
                        if first_user:
                            history_lines = [f"user: {first_user}"]
                    except Exception:
                        history_lines = []

                # Dict interjections may carry continued parent context.
                # Interjections are sent as user messages (not system) for
                # broad provider compatibility; user-visibility context sits
                # in the topmost system message.
                if isinstance(extra, dict):
                    _msg_text = str(extra.get("message", "")).strip()
                    _ctx_cont = extra.get(
                        "_parent_chat_context_continued",
                    ) or extra.get(
                        "_parent_chat_context_continuted",
                    )
                else:
                    _msg_text = str(extra)
                    _ctx_cont = None

                try:
                    logger.info(
                        f"Interjection received: {_msg_text}",
                        prefix=ICONS["interjection"],
                    )
                except Exception:
                    pass

                if _ctx_cont:
                    _ctx_cont = make_messages_safe_for_context_dump(_ctx_cont)
                    context_state.receive_context_continuation(_ctx_cont)
                    # Only inner handles that opted into context initially
                    # receive continuations.
                    for task, info in tools_data.info.items():
                        if info.interject_queue is not None and info.context_opted_in:
                            with suppress(Exception):
                                info.interject_queue.put_nowait(
                                    {
                                        "message": "",  # Empty message, just context update
                                        "_parent_chat_context_continued": _ctx_cont,
                                        "_context_only": True,  # Flag to indicate context-only update
                                    },
                                )
                                context_state.mark_cont_forwarded_to_tool(info.call_id)

                # The first interjection injects user-visibility guidance so the
                # model understands why a user message appears mid-execution
                # and what the user can and cannot see. record_progress and
                # record_clarification share the same flag on tools_data, so
                # the injection is paid for once whichever event comes first.
                await tools_data._ensure_visibility_guidance_injected(_msg_dispatcher)

                # A context continuation goes in a separate user message tagged
                # _ctx_header, so the current LLM sees it but it is filtered
                # out when building cur_msgs for inner tool forwarding.
                msgs_to_append: list[dict] = []
                if _ctx_cont:
                    ctx_cont_transformed = _transform_context_roles(_ctx_cont)
                    ctx_cont_content = (
                        "## Parent Chat Context (continued)\n"
                        "This is the next incremental chunk of the parent conversation since the "
                        "last context update (either the initial Parent Chat Context in your system "
                        "message, or the previous continued context chunk). These messages arrived "
                        "while you have been working on this request and may be relevant. Use this "
                        "to stay informed of any updates or new information from the parent conversation. "
                        "As explained in the system message, 'outer_user' and 'outer_assistant' roles "
                        "indicate messages from the parent conversation.\n\n"
                        f"{json.dumps(ctx_cont_transformed, indent=2)}"
                    )
                    msgs_to_append.append(
                        loop_user_notice(ctx_cont_content, _ctx_header=True),
                    )
                if _msg_text:
                    _user_content = (
                        time_ctx.prefix_user_message(_msg_text)
                        if time_ctx is not None
                        else _msg_text
                    )
                    msgs_to_append.append(
                        {
                            "role": "user",
                            "_interjection": True,
                            "content": _user_content,
                        },
                    )
                if msgs_to_append:
                    await _msg_dispatcher.append_msgs(msgs_to_append)
                with suppress(Exception):
                    if outer_handle:
                        outer_handle._user_visible_history.append(
                            {
                                "role": "user",
                                "content": (
                                    {
                                        "message": _msg_text,
                                        "_parent_chat_context_continued": _ctx_cont,
                                    }
                                    if isinstance(extra, dict) and _ctx_cont
                                    else _msg_text
                                ),
                            },
                        )

            # ── A. Wait for a tool completion, cancellation, interjection,
            #       clarification or notification ────────────────────────
            # Skipped entirely when the model already needs to speak.
            if tools_data.pending and not llm_turn_required:
                interject_w = asyncio.create_task(
                    interject_queue.get(),
                    name="InterjectQueueGet",
                )
                cancel_waiter = asyncio.create_task(
                    cancel_event.wait(),
                    name="CancelEventWait",
                )
                clar_waiters: Dict[asyncio.Task, asyncio.Task] = {}
                notif_waiters: Dict[asyncio.Task, asyncio.Task] = {}
                for _t in tools_data.pending:
                    # A task already awaiting an answer has
                    # waiting_for_clarification set; only new questions are
                    # listened for.
                    info = tools_data.info[_t]
                    if info.waiting_for_clarification:
                        continue

                    if info.clar_up_queue is not None:
                        w = asyncio.create_task(
                            info.clar_up_queue.get(),
                            name="ClarificationQueueGet",
                        )
                        clar_waiters[w] = _t

                    if info.notification_queue is not None:
                        pw = asyncio.create_task(
                            info.notification_queue.get(),
                            name="NotificationQueueGet",
                        )
                        notif_waiters[pw] = _t
                waiters = (
                    tools_data.pending
                    | set(clar_waiters)
                    | set(notif_waiters)
                    | {cancel_waiter, interject_w}
                )

                if timer.has_exceeded_time():
                    return await _handle_limit_reached(
                        f"timeout ({timeout}s) exceeded",
                    )

                done, _ = await asyncio.wait(
                    waiters,
                    timeout=timer.remaining_time(),
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Nothing completed means the wait itself timed out.
                if not done:
                    if raise_on_limit:
                        raise asyncio.TimeoutError(
                            f"Loop exceeded {timeout}s wall-clock limit",
                        )
                    else:
                        return await _handle_limit_reached(
                            f"timeout ({timeout}s) exceeded",
                        )

                # Unused auxiliary waiters must be cancelled and awaited,
                # otherwise a lingering queue getter consumes the next
                # interjection invisibly.
                for aux in (
                    interject_w,
                    cancel_waiter,
                    *clar_waiters.keys(),
                    *notif_waiters.keys(),
                ):
                    if aux not in done and not aux.done():
                        aux.cancel()
                        await asyncio.gather(aux, return_exceptions=True)

                if interject_w in done:
                    # Re-queued so the drain at the top handles it.
                    await interject_queue.put(interject_w.result())
                    continue  # → loop, will be processed in 0.

                if cancel_waiter in done:
                    # Cancellation wins; the mirrored stop is the only
                    # propagation path to children.
                    raise asyncio.CancelledError  # cancellation wins

                # A clarification request from a child gets the assistant an
                # immediate turn; notifications from the same tick are
                # ingested first.
                if done & clar_waiters.keys():
                    for cw in done & clar_waiters.keys():
                        await _handle_clarification(clar_waiters[cw], cw.result())

                    if done & notif_waiters.keys():
                        for pw in done & notif_waiters.keys():
                            await _handle_notification(notif_waiters[pw], pw.result())

                    llm_turn_required = True
                    continue

                # A progress notification also earns an immediate LLM turn.
                if done & notif_waiters.keys():
                    for pw in done & notif_waiters.keys():
                        await _handle_notification(notif_waiters[pw], pw.result())
                    llm_turn_required = True

                needs_turn = False
                # Helper waiters are excluded; only real tool tasks complete.
                _completed_tools = done & tools_data.pending
                if _completed_tools:
                    logger.debug(
                        f"⏱️ [ToolLoop] {len(_completed_tools)} tool task(s) completed, "
                        f"{len(tools_data.pending) - len(_completed_tools)} still pending",
                    )
                for task in _sort_completed_tasks_by_call_id(
                    _completed_tools,
                    tools_data,
                ):
                    if await tools_data.process_completed_task(
                        task=task,
                        consecutive_failures=consecutive_failures,
                        outer_handle_container=outer_handle_container,
                        assistant_meta=assistant_meta,
                        msg_dispatcher=_msg_dispatcher,
                    ):
                        needs_turn = True

                if needs_turn:
                    llm_turn_required = True
                if tools_data.pending:
                    continue  # jump to top-of-loop

            # ── B. Wait for remaining tools before asking the LLM again,
            #       unless the model already deserves a turn ───────────────
            if tools_data.pending and not llm_turn_required:
                await ensure_placeholders_for_pending(
                    tools_data=tools_data,
                    assistant_meta=assistant_meta,
                    client=client,
                    msg_dispatcher=_msg_dispatcher,
                    time_ctx=time_ctx,
                )
                continue  # still waiting for other tool tasks

            # ── C. Build this turn's toolkit ─────────────────────────────
            # Rebuilt fresh every turn so concurrency changes (tasks
            # finishing, stopping, …) are reflected in what the LLM sees.

            # Tool policy and tool subset for this turn. Eager policies (e.g.
            # discovery-first gates) keep the model on a narrow required
            # subset; tracking that keeps compress_context out of the schema
            # as an escape hatch.
            logger.debug(
                f"[setup +{_setup_elapsed()}] tool policy eval (step={runtime_state.step_index})",
            )
            _policy_eager = False
            if tool_policy is not None:
                _tools_snapshot = {n: s.fn for n, s in tools_data.normalized.items()}
                try:
                    if _policy_accepts_history:
                        _policy_result = tool_policy(
                            runtime_state.step_index,
                            _tools_snapshot,
                            list(runtime_state.called_tools),
                        )
                    else:
                        _policy_result = tool_policy(
                            runtime_state.step_index,
                            _tools_snapshot,
                        )
                    tool_choice_mode, filtered, _policy_eager = (
                        _parse_tool_policy_result(
                            _policy_result,
                        )
                    )
                except Exception as _e:  # never abort the loop on mis-behaving policies
                    logger.error(
                        f"tool_policy raised on turn {runtime_state.step_index}: {_e!r}",
                    )
                    tool_choice_mode, filtered = "auto", _tools_snapshot
                    _policy_eager = False
                policy_tools_norm = normalise_tools(filtered)
            else:
                tool_choice_mode = "auto"
                policy_tools_norm = tools_data.normalized

            # When tools are in-flight, force tool_choice=required so the LLM
            # must call a real tool (steer, wait, ask_about_completed_tool,
            # etc.) rather than ending the loop. The response tool stays in
            # the schema but is refused at execution time while anything is
            # pending (see the steer()/response-tool execution branches
            # below), so "required" still only leaves live options.
            _has_pending_tools = bool(tools_data.pending)
            if _has_pending_tools and tool_choice_mode != "required":
                tool_choice_mode = "required"

            logger.debug(
                f"[setup +{_setup_elapsed()}] building tool schemas ({len(policy_tools_norm)} tools)",
            )
            _compress_schema = (
                method_to_schema(compress_context, "compress_context")
                if enable_compression
                else None
            )

            if _over_threshold and enable_compression:
                if _has_pending_tools:
                    # Over threshold, pending tools → no base tools, no
                    # compress_context (can't compress mid-flight). Only the
                    # static surface (wait, steer, ask_about_completed_tool)
                    # remains visible.
                    visible_base_tools_schema = []
                    _threshold_msg = (
                        "Context window is nearly full. "
                        "You cannot start new tools. Wait for in-flight tools to complete and then call "
                        "`compress_context` to free up context."
                    )
                    if log_steps == "full":
                        logger.info(
                            f"Context over threshold (pending in-flight tools): {_threshold_msg}",
                            prefix=ICONS["summarize"],
                        )
                    await _msg_dispatcher.append_msgs(
                        [loop_user_notice(_threshold_msg)],
                    )
                else:
                    # Over threshold, no pending → compress_context plus
                    # any caller-specified extra compression tools (pulled
                    # from the full tool set so policy gates are bypassed).
                    visible_base_tools_schema = [_compress_schema]
                    if extra_compression_tools:
                        visible_base_tools_schema.extend(
                            method_to_schema(
                                spec.fn,
                                name,
                                expose_context_control=(
                                    propagate_chat_context
                                    == ChatContextPropagation.LLM_DECIDES
                                ),
                                has_parent_context=bool(parent_chat_context),
                            )
                            for name, spec in tools_data.normalized.items()
                            if name in extra_compression_tools
                        )
                    tool_choice_mode = "required"
                    _threshold_msg = (
                        "Context window is nearly full. "
                        "You must call `compress_context` now."
                    )
                    if log_steps == "full":
                        logger.info(
                            f"Context over threshold (no pending): {_threshold_msg}",
                            prefix=ICONS["summarize"],
                        )
                    await _msg_dispatcher.append_msgs(
                        [loop_user_notice(_threshold_msg)],
                    )
            else:
                # Schema constancy beats schema minimalism: tools stay visible
                # even while saturated on max_concurrent/max_total_calls —
                # a saturated call is refused at execution time instead (see
                # has_exceeded_concurrent_limit_for_tool / prune_over_quota_tool_calls),
                # so hitting the cap never changes what the model can see.
                visible_base_tools_schema = [
                    method_to_schema(
                        spec.fn,
                        name,
                        expose_context_control=(
                            propagate_chat_context == ChatContextPropagation.LLM_DECIDES
                        ),
                        has_parent_context=bool(parent_chat_context),
                    )
                    for name, spec in policy_tools_norm.items()
                ]
                # compress_context stays out of eager gated turns so required
                # discovery/tool policies cannot be satisfied by compressing;
                # forced over-threshold compression above still applies.
                if _compress_schema is not None and not _policy_eager:
                    visible_base_tools_schema.append(_compress_schema)

            # The response-submission tool is in the schema whenever
            # response_format is set, regardless of in-flight tools: masking
            # it in and out on every pending<->idle transition would break
            # the prompt prefix each time. It means "end the current turn"
            # (the tool-call analogue of a bare text response); calling it
            # while tools are pending is refused at execution time, the same
            # schema-constant-but-execution-gated pattern as concurrency/quota
            # saturation and steer().
            #
            #   persist=True  → "send_response"  (signals turn completion,
            #                    loop continues waiting for next interjection)
            #   persist=False → "final_response"  (terminates the loop)
            _response_tool_name = "send_response" if persist else "final_response"
            # "Ready" means present in the schema (response_format configured
            # and injection succeeded), not safe to call right now; safety is
            # enforced by the pending-tools refusal in the execution branch.
            _structured_response_tool_ready = False

            if _rf_norm is not None:
                if persist:
                    _response_tool_desc = (
                        "Submit your structured response for the current "
                        "request in the required JSON format. This signals "
                        "that you have completed the current work and are "
                        "ready for the next instruction. Do not use this "
                        "for progress updates — those should be sent via "
                        "notifications while work is still ongoing."
                    )
                else:
                    _response_tool_desc = (
                        "Submit your final response in the required JSON "
                        "format. The response can be a complete result, a "
                        "partial result, or a message indicating you cannot "
                        "proceed (e.g., 'I cannot help with that.'). "
                        "Calling this tool terminates the conversation."
                    )
                try:
                    _answer_schema = _rf_norm.answer_json_schema

                    visible_base_tools_schema.append(
                        {
                            "type": "function",
                            "strict": DEFAULT_TOOL_SCHEMA_STRICT,
                            "function": {
                                "name": _response_tool_name,
                                "description": _response_tool_desc,
                                "parameters": {
                                    "type": "object",
                                    "properties": {"answer": _answer_schema},
                                    "required": ["answer"],
                                },
                            },
                        },
                    )
                    _structured_response_tool_ready = True
                except Exception as _injection_exc:  # noqa: BLE001
                    logger.error(
                        f"Failed to inject {_response_tool_name} tool: {_injection_exc!r}",
                    )

            # Only force tool use for structured output once the response tool
            # is actually available. Forcing required without final_response /
            # send_response creates an inescapable tool-call loop.
            if _structured_response_tool_ready and tool_choice_mode != "required":
                tool_choice_mode = "required"

            # Multi-handle mode: `final_response` takes a request_id and is
            # always available (tools may be shared), unlike response_format
            # mode; `ask_user_clarification` routes questions to one request.
            if multi_handle_coordinator is not None:
                visible_base_tools_schema.append(
                    {
                        "type": "function",
                        "strict": DEFAULT_TOOL_SCHEMA_STRICT,
                        "function": {
                            "name": "final_response",
                            "description": (
                                "Submit the final response for a specific request. "
                                "Use this to complete a request when you have the result. "
                                "Each request must be answered exactly once."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "request_id": {
                                        "type": "integer",
                                        "description": "The ID of the request being answered (from [Request N] tag).",
                                    },
                                    "answer": {
                                        "type": "string",
                                        "description": "The final answer text for this request.",
                                    },
                                },
                                "required": ["request_id", "answer"],
                            },
                        },
                    },
                )
                visible_base_tools_schema.append(
                    {
                        "type": "function",
                        "strict": DEFAULT_TOOL_SCHEMA_STRICT,
                        "function": {
                            "name": "ask_user_clarification",
                            "description": (
                                "Ask a specific user for clarification. Use this when you need "
                                "more information from the user who submitted a particular request."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "request_id": {
                                        "type": "integer",
                                        "description": "The ID of the request whose user should receive the question.",
                                    },
                                    "question": {
                                        "type": "string",
                                        "description": "The clarification question to ask the user.",
                                    },
                                },
                                "required": ["request_id", "question"],
                            },
                        },
                    },
                )

            # Yield so just-scheduled tool tasks can run (especially those
            # that immediately return a SteerableToolHandle), so dynamic
            # helpers are generated with the handle's docstrings.
            logger.debug(f"[setup +{_setup_elapsed()}] yielding (asyncio.sleep(0))")
            await asyncio.sleep(0)
            logger.debug(f"[setup +{_setup_elapsed()}] resumed after yield")

            for task in list(tools_data.pending):
                if task.done():
                    with suppress(Exception):
                        await tools_data.process_completed_task(
                            task=task,
                            consecutive_failures=consecutive_failures,
                            outer_handle_container=outer_handle_container,
                            assistant_meta=assistant_meta,
                            msg_dispatcher=_msg_dispatcher,
                        )

            dynamic_tool_factory = DynamicToolFactory(tools_data)
            dynamic_tool_factory.generate()
            dynamic_tools = dynamic_tool_factory.dynamic_tools

            # A handle adopted mid-loop refreshes capability bookkeeping
            # (is_interjectable, clarification queue wiring, live-ask
            # closures). No outer-visible tools are minted per handle:
            # steer()/wait/ask_about_completed_tool are static.
            def _refresh_helpers_for_task(task: asyncio.Task) -> None:
                with suppress(Exception):
                    dynamic_tool_factory._refresh_task_capabilities(task)

            tools_data._on_handle_adopted = _refresh_helpers_for_task

            # `wait` stays in the schema, byte-stable, even while a
            # clarification is pending; the deadlock interlock is enforced at
            # execution time (see the `lname_cf == "wait"` branch below).

            # Every pending call needs a placeholder tool reply before the
            # assistant speaks again.
            logger.debug(f"[setup +{_setup_elapsed()}] ensure_placeholders start")
            await ensure_placeholders_for_pending(
                tools_data=tools_data,
                assistant_meta=assistant_meta,
                client=client,
                msg_dispatcher=_msg_dispatcher,
                time_ctx=time_ctx,
            )
            logger.debug(f"[setup +{_setup_elapsed()}] ensure_placeholders done")

            # Dynamic helpers join the visible toolkit. In LLM_DECIDES mode,
            # dynamic tools accepting _parent_chat_context (the ask_* tools)
            # expose include_parent_chat_context so the model can opt out of
            # context for inspection loops, and steering methods
            # (ask/interject) on tools that opted into context expose
            # include_parent_chat_context_cont.
            _expose_ctx_cont_control = (
                propagate_chat_context == ChatContextPropagation.LLM_DECIDES
            )
            tmp_tools = visible_base_tools_schema + [
                method_to_schema(
                    fn,
                    include_class_name=include_class_in_dynamic_tool_names,
                    expose_context_control=_expose_ctx_cont_control,
                    has_parent_context=bool(parent_chat_context),
                    expose_context_cont_control=(
                        _expose_ctx_cont_control
                        and getattr(fn, "__supports_context_propagation__", False)
                        and getattr(fn, "__context_opted_in__", False)
                    ),
                )
                for fn in dynamic_tools.values()
            ]

            # ── D. Ask the LLM what to do next ───────────────────────────
            logger.debug(
                f"[setup +{_setup_elapsed()}] ready for LLM call (step={runtime_state.step_index}, {len(tmp_tools)} tools)",
            )
            if log_steps:
                logger.begin_thinking()

            await to_event_bus(
                {"role": "assistant", "_thinking_in_flight": True},
                cfg,
            )

            # Set only by patient mode below, to keep hold of the assistant
            # message this step produced.
            _patient_asst_msg: Optional[dict] = None

            if interrupt_llm_with_interjections:
                # ––––– pre-emptive mode: the LLM step races the pending
                # tools, interjections, cancellation, clarifications and
                # notifications –––––––––––––––––––––––––––––––––––––––––
                _gen_kwargs = {
                    "return_full_completion": True,
                    "tools": tmp_tools,
                    "tool_choice": tool_choice_mode,
                    "stateful": True,
                    "prompt_caching": prompt_caching,
                }
                if max_parallel_tool_calls is not None:
                    _gen_kwargs["parallel_tool_calls"] = max_parallel_tool_calls > 1
                elif _policy_eager:
                    # Discovery-first (and other eager gates) expose multiple
                    # required tools that must be callable in one assistant turn.
                    _gen_kwargs["parallel_tool_calls"] = True

                # The prompt this dispatch sends is about to be snapshotted
                # from the current transcript, so it provably contains every
                # result ingested so far — the obligation deferred_llm_turn
                # exists to enforce is satisfied by this dispatch alone.
                # Clearing here, not when the step completes, is what keeps
                # a result that lands *during* this same dispatch's flight
                # correctly deferred to the turn after it: the set sites run
                # after this point, so they still land after the clear.
                deferred_llm_turn = False

                llm_task = asyncio.create_task(
                    generate_with_preprocess(
                        client,
                        _apply_reasoning_model_compat(_gen_kwargs, tool_choice_mode),
                        **_gen_kwargs,
                    ),
                    name="LLMGenerate",
                )
                interject_w = asyncio.create_task(
                    interject_queue.get(),
                    name="InterjectQueueGet",
                )
                cancel_waiter = asyncio.create_task(
                    cancel_event.wait(),
                    name="CancelEventWait",
                )

                pending_snapshot = set(tools_data.pending)
                clar_waiters2: Dict[asyncio.Task, asyncio.Task] = {}
                notif_waiters2: Dict[asyncio.Task, asyncio.Task] = {}
                for _t in pending_snapshot:
                    _inf = tools_data.info[_t]
                    # Only new clarification requests are listened for.
                    if (
                        _inf is not None
                        and not getattr(_inf, "waiting_for_clarification", False)
                        and _inf.clar_up_queue is not None
                    ):
                        cw2 = asyncio.create_task(
                            _inf.clar_up_queue.get(),
                            name="ClarificationQueueGet",
                        )
                        clar_waiters2[cw2] = _t
                    if _inf is not None and _inf.notification_queue is not None:
                        pw2 = asyncio.create_task(
                            _inf.notification_queue.get(),
                            name="NotificationQueueGet",
                        )
                        notif_waiters2[pw2] = _t

                done, _ = await asyncio.wait(
                    pending_snapshot
                    | set(clar_waiters2.keys())
                    | set(notif_waiters2.keys())
                    | {llm_task, interject_w, cancel_waiter},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if log_steps:
                    logger.emit_thinking_fallback()

                # Only the auxiliary waiters are cancelled here. llm_task is
                # deliberately left alone: each branch below decides.
                # - Tool finished → cancel (needs new context), unless
                #   ``interrupt_llm_on_tool_completion`` is False
                # - Immediate interjection → cancel (user wants a response now)
                # - Patient interjection → do not cancel (let it finish)
                # - Clarification/notification → cancel (surface the event)
                # - Cancellation requested → cancel (explicit stop)
                for tsk in (
                    interject_w,
                    cancel_waiter,
                    *clar_waiters2.keys(),
                    *notif_waiters2.keys(),
                ):
                    if tsk not in done and not tsk.done():
                        tsk.cancel()
                await asyncio.gather(
                    interject_w,
                    cancel_waiter,
                    *clar_waiters2.keys(),
                    *notif_waiters2.keys(),
                    return_exceptions=True,
                )

                # A tool finished before the LLM answered.
                if done & pending_snapshot:
                    logger.debug(
                        f"⏱️ [ToolLoop] tool(s) finished during LLM race: "
                        f"{len(done & pending_snapshot)} completed",
                    )
                    if not interrupt_llm_on_tool_completion:
                        # Patient mode: the reasoning step already sent its
                        # prompt to the provider, which bills it whether or not
                        # the answer is collected, so discarding it to re-ask
                        # with the tool result costs a whole step and buys only
                        # latency. Let it finish and be used instead.
                        #
                        # The results still have to be ingested here: leaving the
                        # task pending makes section F read it as work in flight,
                        # and ``cancel_pending_tasks`` drops it without
                        # processing, losing the very result this branch fired
                        # for. ``deferred_llm_turn`` then guarantees a further
                        # turn, so the model always sees these results before it
                        # can conclude; it reasons one step behind, never without.
                        #
                        # Order matters. The step is awaited first so its own
                        # assistant message can be captured, because ingesting a
                        # result whose placeholder is no longer at the tail
                        # appends a synthetic assistant/tool status pair, and the
                        # loop would otherwise mistake that pair's tool message
                        # for this step's turn.
                        deferred_llm_turn = True
                        await asyncio.gather(llm_task, return_exceptions=True)
                        _patient_asst_msg = client.messages[-1]
                        completed_snapshot = {
                            task for task in pending_snapshot if task.done()
                        }
                        for task in _sort_completed_tasks_by_call_id(
                            completed_snapshot,
                            tools_data,
                        ):
                            await tools_data.process_completed_task(
                                task=task,
                                consecutive_failures=consecutive_failures,
                                outer_handle_container=outer_handle_container,
                                assistant_meta=assistant_meta,
                                msg_dispatcher=_msg_dispatcher,
                            )
                    else:
                        # Cancel the half-finished reasoning step, then handle
                        # each newly-finished task exactly as branch A does.
                        if not llm_task.done():
                            llm_task.cancel()
                        for aux in (interject_w, cancel_waiter):
                            if aux not in done and not aux.done():
                                aux.cancel()
                        await asyncio.gather(
                            llm_task,
                            interject_w,
                            cancel_waiter,
                            return_exceptions=True,
                        )
                        needs_turn = False
                        for task in _sort_completed_tasks_by_call_id(
                            done & pending_snapshot,
                            tools_data,
                        ):
                            if await tools_data.process_completed_task(
                                task=task,
                                consecutive_failures=consecutive_failures,
                                outer_handle_container=outer_handle_container,
                                assistant_meta=assistant_meta,
                                msg_dispatcher=_msg_dispatcher,
                            ):
                                needs_turn = True

                        if needs_turn:  # assistant speaks only if needed
                            llm_turn_required = True
                        continue

                # The user interjected. Immediate unless the interjection
                # itself says otherwise.
                if interject_w in done:
                    _payload = None
                    try:
                        _payload = interject_w.result()
                    except Exception:
                        _payload = None
                    _immediate = True
                    try:
                        if isinstance(_payload, dict):
                            _immediate = bool(
                                _payload.get("trigger_immediate_llm_turn", True),
                            )
                    except Exception:
                        _immediate = True
                    # Re-queued for the main drain path.
                    await interject_queue.put(_payload)
                    if _immediate:
                        if not llm_task.done():
                            llm_task.cancel()
                            await asyncio.gather(llm_task, return_exceptions=True)
                        continue  # top of loop
                    # Patient: let the in-flight LLM call finish and schedule
                    # exactly one subsequent LLM turn after it.
                    deferred_llm_turn = True
                    if not llm_task.done():
                        await asyncio.gather(llm_task, return_exceptions=True)

                # A clarification bubbled up while the LLM was thinking:
                # cancel the step, surface the request, restart so the next
                # assistant turn can ingest it.
                if done & set(clar_waiters2.keys()):
                    if not llm_task.done():
                        llm_task.cancel()
                        await asyncio.gather(llm_task, return_exceptions=True)
                    for cw in done & set(clar_waiters2.keys()):
                        await _handle_clarification(clar_waiters2[cw], cw.result())
                    llm_turn_required = True
                    continue

                # Likewise for a notification.
                if done & set(notif_waiters2.keys()):
                    if not llm_task.done():
                        llm_task.cancel()
                        await asyncio.gather(llm_task, return_exceptions=True)
                    for pw in done & set(notif_waiters2.keys()):
                        await _handle_notification(notif_waiters2[pw], pw.result())
                    llm_turn_required = True
                    continue

                # Cancellation only escalates when the flag is actually set.
                if cancel_waiter in done:
                    if cancel_event.is_set():
                        if not llm_task.done():
                            llm_task.cancel()
                            await asyncio.gather(llm_task, return_exceptions=True)
                        raise asyncio.CancelledError

                # The LLM finished.
                if llm_task.cancelled():
                    raise asyncio.CancelledError
                if llm_task.exception():
                    # Cached-replay determinism: a read-only cache miss while
                    # tools are still in flight means the live run never
                    # consumed this step — a tool completion (or steering
                    # event) superseded it mid-call and the loop re-issued the
                    # turn with updated context, so no entry was ever
                    # recorded. Mirror that outcome: discard the step and fall
                    # back to the tool-wait block, which grants a fresh turn
                    # once the superseding event lands. With nothing in flight
                    # the miss is genuinely fatal and propagates.
                    if _is_cache_miss_error(llm_task.exception()) and (
                        tools_data.pending
                    ):
                        llm_turn_required = False
                        continue
                    try:
                        llm_task.result()
                    except Exception as e:
                        raise Exception(
                            f"LLM call failed: {type(e).__name__}: {e}",
                        ) from e

                    if done & set(clar_waiters2.keys()):
                        for cw in done & set(clar_waiters2.keys()):
                            await _handle_clarification(clar_waiters2[cw], cw.result())
                        llm_turn_required = True

                    if done & set(notif_waiters2.keys()):
                        for pw in done & set(notif_waiters2.keys()):
                            await _handle_notification(notif_waiters2[pw], pw.result())
                        llm_turn_required = True

                _full_completion = llm_task.result()

            else:
                # ––––– blocking mode: the LLM step runs to completion –––––
                try:
                    _gen_kwargs = {
                        "return_full_completion": True,
                        "tools": tmp_tools,
                        "tool_choice": tool_choice_mode,
                        "stateful": True,
                        "prompt_caching": prompt_caching,
                    }
                    if max_parallel_tool_calls is not None:
                        _gen_kwargs["parallel_tool_calls"] = max_parallel_tool_calls > 1
                    elif _policy_eager:
                        _gen_kwargs["parallel_tool_calls"] = True

                    # See the matching comment at the interrupt-mode dispatch
                    # above: clearing here (not at step completion) means the
                    # prompt this dispatch is about to snapshot provably
                    # contains everything ingested so far.
                    deferred_llm_turn = False

                    _full_completion = await generate_with_preprocess(
                        client,
                        _apply_reasoning_model_compat(_gen_kwargs, tool_choice_mode),
                        **_gen_kwargs,
                    )
                    if log_steps:
                        logger.emit_thinking_fallback()
                except Exception as e:
                    raise Exception(
                        f"LLM call failed: {type(e).__name__}: {e}",
                    ) from e

            # Normally the step's assistant message is the tail. Patient mode
            # ingests tool results after it lands, which can append a synthetic
            # status pair on top, so it captures the message itself.
            msg = (
                _patient_asst_msg
                if _patient_asst_msg is not None
                else client.messages[-1]
            )
            await to_event_bus(msg, cfg)

            # Update context threshold from the LLM response usage data.
            if enable_compression:
                with suppress(Exception):
                    _usage = getattr(_full_completion, "usage", None)
                    if (
                        _usage
                        and getattr(_usage, "prompt_tokens", None)
                        and _max_input_tokens
                    ):
                        _over_threshold = context_over_threshold(
                            _usage.prompt_tokens,
                            0.7,
                            _max_input_tokens,
                        )

            # The activity timeout catches hung tools, not slow inference
            # (providers have their own timeouts), so an LLM response resets it.
            timer.reset()

            if log_steps:
                with suppress(Exception):
                    from .utils import format_llm_response_for_log

                    logger.info(
                        format_llm_response_for_log(msg),
                        prefix=ICONS["llm_response"],
                    )

            if timer.has_exceeded_time():
                return await _handle_limit_reached(
                    f"timeout ({timeout}s) exceeded",
                )

            llm_turn_required = False
            runtime_state.step_index += 1

            # ── E. Launch any new tool calls ─────────────────────────────
            # Each call's arguments are JSON-parsed once here, the loop-owned
            # tools (response submission, compress_context, wait, steer,
            # ask_about_completed_tool) are handled inline, and every other
            # call is scheduled as a Task whose metadata lets its result be
            # inserted at the right chronological position. Control then
            # jumps back to branch A to wait for the first completion.
            _persist_response_emitted = False
            _persist_response_content = None  # captured by send_response for surfacing

            if msg["tool_calls"]:
                # Both mutations below edit msg["tool_calls"] in place, which
                # is safe only while msg is still mutable: an edit below the
                # sent watermark would mutate already-dispatched bytes. msg is
                # this turn's freshly-generated message (index == watermark,
                # nothing has dispatched it), so this always holds; it is
                # checked up front so the invariant is stated rather than
                # dependent on which mutation happens to run first.
                if not is_mutable(client, msg):
                    logger.error(
                        "persist-mode tool_calls pruning: msg is already "
                        "below the sent watermark; an in-place edit would "
                        "mutate already-dispatched bytes.",
                        prefix="🚨",
                    )
                    raise ValueError(
                        "persist-mode tool_calls pruning: msg is already "
                        "below the sent watermark; an in-place edit would "
                        "mutate already-dispatched bytes.",
                    )

                # De-duplication runs before quota pruning: quota accounting
                # should count unique calls, not raw duplicate occurrences — a
                # tool called identically 3x against a max_total_calls=2 limit
                # spends 1 unit of quota, not 3.
                if prune_tool_duplicates:
                    unique, _ = prune_duplicate_tool_calls(msg["tool_calls"])
                    if len(unique) != len(msg["tool_calls"]):
                        msg["tool_calls"] = unique

                # Over-quota calls are always removed before any scheduling,
                # regardless of the de-duplication setting.
                tools_data.prune_over_quota_tool_calls(msg)

                # If pruning removed every call and left the placeholder
                # notice, a user turn prompts the model to continue; without
                # it strict models reject the assistant->assistant history.
                # The 'user' role keeps alternation valid for all providers.
                if not msg.get(
                    "tool_calls",
                ) and "(Tool calls were removed due to quota limits)" in str(
                    msg.get("content") or "",
                ):
                    sys_notice = loop_user_notice(
                        "System notification: The tool calls in your last response "
                        "were blocked due to quota limits. Please modify your plan "
                        "or conclude.",
                    )
                    await _msg_dispatcher.append_msgs([sys_notice])

                for idx, call in enumerate(msg["tool_calls"]):  # capture index
                    name = call["function"]["name"]
                    runtime_state.called_tools.append(name)

                    # Arguments arrive as a JSON string or a dict. A model can
                    # emit invalid JSON — most often truncated, because
                    # generation ran to the output-token cap mid-object. That
                    # is recoverable for this one call, so it is surfaced back
                    # to the model the same way an unavailable tool is (below)
                    # rather than aborting the whole turn; repetition ends the
                    # loop via the refusal tally.
                    _raw_args = call["function"]["arguments"]
                    if isinstance(_raw_args, str):
                        try:
                            args = json.loads(_raw_args)
                        except ValueError as exc:
                            logger.error(
                                "Malformed tool-call arguments for %s (%d chars): %s",
                                name,
                                len(_raw_args),
                                exc,
                            )
                            refusal = (
                                f"⚠️ Error: the arguments for '{name}' were not "
                                f"valid JSON ({exc}). They may have been cut off "
                                "mid-object. Re-issue the call with complete, "
                                "well-formed JSON arguments, keeping each value "
                                "in the type the target expects."
                            )
                            consecutive_failures.note_refusal(
                                tool_name=name,
                                args=_raw_args,
                                message=refusal,
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                create_tool_call_message(
                                    name=name,
                                    call_id=call["id"],
                                    content=refusal,
                                ),
                                client,
                                _msg_dispatcher,
                            )
                            stop_reason = consecutive_failures.stop_reason()
                            if stop_reason:
                                raise RuntimeError(stop_reason)
                            continue
                    else:
                        args = _raw_args if isinstance(_raw_args, dict) else {}

                    # Response-submission tool (send_response in persist mode,
                    # final_response otherwise).
                    _is_response_tool = (
                        name in ("final_response", "send_response")
                        and _rf_norm is not None
                    )
                    if _is_response_tool:
                        if tools_data.pending:
                            # Execution-time refusal (schema presence is
                            # unconditional; see the injection comment above).
                            # The exits are named explicitly: this fires under
                            # tool_choice="required" (has_pending_tools forces
                            # it), so a refusal with no way out would be a
                            # retry loop.
                            tool_msg = create_tool_call_message(
                                name=name,
                                call_id=call["id"],
                                content=(
                                    f"⚠️ Cannot call '{name}': "
                                    f"{len(tools_data.pending)} tool call(s) still "
                                    "running. Call `wait` to let them finish, or "
                                    'steer(call_id=<id>, action="stop") one of '
                                    "them if it's no longer needed — do not retry "
                                    f"'{name}' until nothing is pending."
                                ),
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )
                            continue
                        try:
                            payload = (
                                args.get("answer") if isinstance(args, dict) else None
                            )
                            if payload is None:
                                raise ValueError("Missing 'answer' in tool arguments.")

                            validated_payload = _rf_norm.validate(payload)
                            if isinstance(validated_payload, BaseModel):
                                payload_for_return = validated_payload.model_dump(
                                    mode="json",
                                )
                            else:
                                payload_for_return = validated_payload

                            tool_msg = create_tool_call_message(
                                name=name,
                                call_id=call["id"],
                                content=_dumps(payload_for_return, indent=4),
                            )

                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )

                            if persist:
                                # The current turn's response; the loop goes on.
                                _persist_response_emitted = True
                                _persist_response_content = json.dumps(
                                    payload_for_return,
                                )
                                break  # exit the for-loop over tool_calls
                            return json.dumps(payload_for_return)
                        except Exception as _exc:
                            tool_msg = create_tool_call_message(
                                name=name,
                                call_id=call["id"],
                                content=(
                                    "⚠️ Validation failed – proceeding with standard formatting step.\n"
                                    + str(_exc)
                                ),
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )
                            continue

                    # A response tool call with no response_format configured is
                    # only reachable when the model hallucinates it.
                    _is_generic_response = (
                        name in ("final_response", "send_response")
                        and _rf_norm is None
                        and multi_handle_coordinator is None
                    )
                    if _is_generic_response:
                        answer = args.get("answer") if isinstance(args, dict) else None
                        if answer is None:
                            answer = str(args) if args else ""

                        if tools_data.pending and not persist:
                            logger.info(
                                f"{name} called while {len(tools_data.pending)} "
                                f"task(s) are in-flight. Auto-cancelling to terminate.",
                                prefix=ICONS["auto_cancel"],
                            )
                            await tools_data.cancel_pending_tasks()

                        tool_msg = create_tool_call_message(
                            name=name,
                            call_id=call["id"],
                            content=answer,
                        )

                        await insert_tool_message_after_assistant(
                            assistant_meta,
                            msg,
                            tool_msg,
                            client,
                            _msg_dispatcher,
                        )

                        if persist:
                            _persist_response_emitted = True
                            _persist_response_content = answer
                            break
                        return answer

                    # Multi-handle response tool.
                    _is_multi_response = (
                        name == "final_response"
                        and multi_handle_coordinator is not None
                    )
                    if _is_multi_response:
                        try:
                            request_id = args.get("request_id")
                            answer = args.get("answer")

                            if request_id is None:
                                raise ValueError(
                                    "Missing 'request_id' in tool arguments.",
                                )
                            if answer is None:
                                raise ValueError("Missing 'answer' in tool arguments.")

                            request_id = int(request_id)

                            error_msg = multi_handle_coordinator.validate_request_id(
                                request_id,
                            )
                            if error_msg:
                                tool_msg = create_tool_call_message(
                                    name=name,
                                    call_id=call["id"],
                                    content=f"⚠️ Error: {error_msg}",
                                )
                                await insert_tool_message_after_assistant(
                                    assistant_meta,
                                    msg,
                                    tool_msg,
                                    client,
                                    _msg_dispatcher,
                                )
                                continue

                            multi_handle_coordinator.complete_request(
                                request_id,
                                str(answer),
                            )

                            tool_msg = create_tool_call_message(
                                name=name,
                                call_id=call["id"],
                                content=f"Request {request_id} completed successfully.",
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )

                            logger.info(
                                f"Request {request_id} completed with answer: {answer[:100]}{'...' if len(answer) > 100 else ''}",
                                prefix=ICONS["completed"],
                            )

                            # should_terminate() at the next iteration ends the
                            # loop once every request is done.
                            continue

                        except Exception as _exc:
                            tool_msg = create_tool_call_message(
                                name=name,
                                call_id=call["id"],
                                content=f"⚠️ Error processing {name}: {_exc}",
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )
                            continue

                    # Multi-handle `ask_user_clarification` tool.
                    if (
                        name == "ask_user_clarification"
                        and multi_handle_coordinator is not None
                    ):
                        try:
                            request_id = args.get("request_id")
                            question = args.get("question")

                            if request_id is None:
                                raise ValueError(
                                    "Missing 'request_id' in tool arguments.",
                                )
                            if question is None:
                                raise ValueError(
                                    "Missing 'question' in tool arguments.",
                                )

                            request_id = int(request_id)

                            multi_handle_coordinator.route_clarification_to_request(
                                request_id,
                                {
                                    "type": "clarification",
                                    "request_id": request_id,
                                    "question": str(question),
                                },
                            )

                            tool_msg = create_tool_call_message(
                                name="ask_user_clarification",
                                call_id=call["id"],
                                content=f"Clarification question sent to request {request_id}. Waiting for user response.",
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )
                            continue

                        except Exception as _exc:
                            tool_msg = create_tool_call_message(
                                name="ask_user_clarification",
                                call_id=call["id"],
                                content=f"⚠️ Error: {_exc}",
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )
                            continue

                    if name == "compress_context":
                        tool_msg = create_tool_call_message(
                            name=name,
                            call_id=call["id"],
                            content=(
                                "Compression initiated. Ending current loop "
                                "to restart with compressed context."
                            ),
                        )
                        await insert_tool_message_after_assistant(
                            assistant_meta,
                            msg,
                            tool_msg,
                            client,
                            _msg_dispatcher,
                        )
                        return _COMPRESSION_SIGNAL

                    # Static helpers: `wait` acknowledges without scheduling;
                    # `steer` dispatches on structured args (stop/interject/
                    # pause/resume/clarify/call/ask).
                    lname = str(name or "").strip()
                    lname_cf = lname.casefold()

                    if lname_cf == "wait" and any(
                        getattr(_inf, "waiting_for_clarification", False)
                        for _inf in tools_data.info.values()
                    ):
                        # Execution-time deadlock guard: `wait` is always in the
                        # schema (stable-schema design), so waiting on a tool
                        # that is itself waiting for an answer is refused here.
                        _pending_clar_ids = [
                            _inf.call_id
                            for _inf in tools_data.info.values()
                            if getattr(_inf, "waiting_for_clarification", False)
                        ]
                        tool_msg = create_tool_call_message(
                            name="wait",
                            call_id=call["id"],
                            content=(
                                "⚠️ Refused: a clarification is pending on "
                                f"{_pending_clar_ids} — answer it via "
                                'steer(call_id=<id>, action="clarify", payload=<answer>) '
                                "before waiting."
                            ),
                        )
                        await insert_tool_message_after_assistant(
                            assistant_meta,
                            msg,
                            tool_msg,
                            client,
                            _msg_dispatcher,
                        )
                        continue

                    if lname_cf == "wait":
                        # With pending tools the wait call is pruned to avoid
                        # transcript clutter; the loop waits for them anyway.
                        if tools_data.pending:
                            try:
                                logger.info(
                                    "Assistant chose `wait` – no-op; not persisting to transcript.",
                                    prefix=ICONS["wait"],
                                )
                            except Exception:
                                pass

                            with suppress(Exception):
                                from .messages import (
                                    prune_wait_tool_call as _prune_wait,
                                )

                                await _prune_wait(
                                    msg,
                                    call["id"],
                                    client=client,
                                    assistant_meta=assistant_meta,
                                    msg_dispatcher=_msg_dispatcher,
                                )

                            # The assistant message containing this wait() was
                            # published to the EventBus before it could be
                            # inspected, so a matching tool result lets the
                            # frontend resolve the pending tool-call row.
                            with suppress(Exception):
                                await to_event_bus(
                                    create_tool_call_message(
                                        "wait",
                                        call["id"],
                                        "",
                                    ),
                                    cfg,
                                )

                            # No immediate LLM turn after a wait: the loop now
                            # waits for pending tools or interjections.
                            continue

                        # With no pending tools, pruning would loop forever on
                        # the cache (same conversation → same cached response).
                        # A factual tool response changes the conversation
                        # state, prescribes nothing, and stays accurate even if
                        # interjections arrive later.
                        try:
                            logger.info(
                                "Assistant called `wait` with no pending tools.",
                                prefix=ICONS["wait"],
                            )
                        except Exception:
                            pass

                        tool_msg = create_tool_call_message(
                            name="wait",
                            call_id=call["id"],
                            content="No tasks are currently running.",
                        )
                        await insert_tool_message_after_assistant(
                            assistant_meta,
                            msg,
                            tool_msg,
                            client,
                            _msg_dispatcher,
                        )
                        continue

                    elif lname_cf == "steer":
                        # Unified steering dispatcher keyed on (call_id, action).
                        # `args` is the parsed form from above (malformed JSON
                        # was already refused), so it is reused directly.
                        _target_call_id = args.get("call_id")
                        _action = str(args.get("action") or "").strip().lower()
                        _payload = args.get("payload")
                        _method = args.get("method")

                        async def _steer_reply(content: str) -> None:
                            _tm = create_tool_call_message(
                                name="steer",
                                call_id=call["id"],
                                content=content,
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                _tm,
                                client,
                                _msg_dispatcher,
                            )

                        if not isinstance(_target_call_id, str) or not _target_call_id:
                            await _steer_reply(
                                "⚠️ steer() requires a string `call_id` identifying "
                                "which call to steer.",
                            )
                            continue

                        tgt_task, tgt_info = tools_data.resolve_call_id(
                            _target_call_id,
                        )

                        if tgt_task is None or tgt_info is None:
                            # "Already completed" and "never existed" are told
                            # apart so the model can self-correct.
                            _completed_name = tools_data._completed_tool_names.get(
                                _target_call_id,
                            )
                            if _completed_name is not None:
                                if _action == "ask":
                                    _err = (
                                        f"call_id={_target_call_id!r} "
                                        f"({_completed_name}) has already completed "
                                        "and is no longer live. Use "
                                        f'ask_about_completed_tool(tool_id="{_target_call_id}", '
                                        "question=...) instead."
                                    )
                                else:
                                    _err = (
                                        f"Cannot {_action or '<missing action>'} "
                                        f"call_id={_target_call_id!r} ({_completed_name}): "
                                        "it has already completed."
                                    )
                            else:
                                _err = (
                                    f"No live call found for call_id={_target_call_id!r}. "
                                    "It may never have existed, or is mistyped — check "
                                    "the call_id shown on the original tool call or on its "
                                    "[steerable ...]/[progress ...]/[clarification ...] "
                                    "tail messages."
                                )
                            await _steer_reply(f"⚠️ {_err}")
                            continue

                        _handle = tgt_info.handle
                        _orig_fn = tgt_info.name
                        _orig_arg_json = tgt_info.call_dict["function"]["arguments"]
                        _pretty_name = (
                            f"steer:{_action or '?'} {_orig_fn}({_orig_arg_json})"
                        )

                        if _action == "stop":
                            with suppress(Exception):
                                await _dispatch_steering_to_child(
                                    "stop",
                                    {"reason": _payload} if _payload else {},
                                    tgt_info,
                                )
                            if not tgt_task.done():
                                tgt_task.cancel()
                            tools_data.pop_task(tgt_task)
                            tool_msg = create_tool_call_message(
                                name=_pretty_name,
                                call_id=call["id"],
                                content=f"The call [{_target_call_id}] has been stopped successfully.",
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )
                            with suppress(Exception):
                                await to_event_bus(
                                    create_tool_call_message(
                                        _orig_fn,
                                        _target_call_id,
                                        json.dumps({"status": "stopped"}),
                                    ),
                                    cfg,
                                )
                            continue

                        elif _action == "interject":
                            if not tgt_info.is_interjectable:
                                await _steer_reply(
                                    f"⚠️ call_id={_target_call_id!r} ({_orig_fn}) does "
                                    "not accept interjections.",
                                )
                                continue

                            # _parent_chat_context_cont is forwarded when the
                            # target's own interject() accepts it and the
                            # target originally opted into context; steer's
                            # include_parent_context field is the opt-out.
                            _interject_accepts_ctx_cont = False
                            if _handle is not None and hasattr(_handle, "interject"):
                                with suppress(Exception):
                                    _ij_sig = inspect.signature(_handle.interject)
                                    _ij_has_varkw = any(
                                        p.kind == inspect.Parameter.VAR_KEYWORD
                                        for p in _ij_sig.parameters.values()
                                    )
                                    _interject_accepts_ctx_cont = (
                                        "_parent_chat_context_cont"
                                        in _ij_sig.parameters
                                        or _ij_has_varkw
                                    )

                            _interject_extra_kwargs, _ = compute_context_injection(
                                args={
                                    "include_parent_chat_context_cont": args.get(
                                        "include_parent_context",
                                        True,
                                    ),
                                },
                                propagate_chat_context=propagate_chat_context,
                                context_state=context_state,
                                client_messages=client.messages,
                                call_id=f"interject_{_target_call_id}_{call['id']}",
                                accepts_parent_ctx=False,
                                accepts_parent_ctx_cont=_interject_accepts_ctx_cont,
                                target_context_opted_in=tgt_info.context_opted_in,
                                is_continuation_only=True,
                            )
                            _interject_payload: Dict[str, Any] = {"content": _payload}
                            if "_parent_chat_context_cont" in _interject_extra_kwargs:
                                _interject_payload["_parent_chat_context_cont"] = (
                                    _interject_extra_kwargs["_parent_chat_context_cont"]
                                )

                            with suppress(Exception):
                                await _dispatch_steering_to_child(
                                    "interject",
                                    _interject_payload,
                                    tgt_info,
                                )
                            tool_msg = create_tool_call_message(
                                name=_pretty_name,
                                call_id=call["id"],
                                content=f'Guidance "{_payload}" forwarded to the running tool.',
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )
                            continue

                        elif _action in ("pause", "resume"):
                            _cap = (
                                _handle is not None
                                and hasattr(
                                    _handle,
                                    "pause" if _action == "pause" else "resume",
                                )
                            ) or (tgt_info.pause_event is not None)
                            if not _cap:
                                await _steer_reply(
                                    f"⚠️ call_id={_target_call_id!r} ({_orig_fn}) cannot "
                                    f"be {_action}d.",
                                )
                                continue
                            _paused_state = (
                                get_handle_paused_state(_handle)
                                if _handle is not None
                                else None
                            )
                            if (
                                _paused_state is None
                                and tgt_info.pause_event is not None
                                and hasattr(tgt_info.pause_event, "is_set")
                            ):
                                with suppress(Exception):
                                    _paused_state = not tgt_info.pause_event.is_set()
                            if _action == "pause" and _paused_state is True:
                                await _steer_reply(
                                    f"⚠️ call_id={_target_call_id!r} ({_orig_fn}) is "
                                    "already paused.",
                                )
                                continue
                            if _action == "resume" and _paused_state is not True:
                                await _steer_reply(
                                    f"⚠️ call_id={_target_call_id!r} ({_orig_fn}) is not "
                                    "currently paused.",
                                )
                                continue
                            with suppress(Exception):
                                await _dispatch_steering_to_child(
                                    _action,
                                    {},
                                    tgt_info,
                                )
                            _past = "paused" if _action == "pause" else "resumed"
                            tool_msg = create_tool_call_message(
                                name=_pretty_name,
                                call_id=call["id"],
                                content=f"The call [{_target_call_id}] has been {_past} successfully.",
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )
                            with suppress(Exception):
                                await to_event_bus(
                                    create_tool_call_message(
                                        _orig_fn,
                                        _target_call_id,
                                        json.dumps({"status": _past}),
                                    ),
                                    cfg,
                                )
                            continue

                        elif _action == "clarify":
                            if tgt_info.clar_up_queue is None or not getattr(
                                tgt_info,
                                "waiting_for_clarification",
                                False,
                            ):
                                await _steer_reply(
                                    f"⚠️ call_id={_target_call_id!r} ({_orig_fn}) has no "
                                    "pending clarification to answer right now.",
                                )
                                continue
                            with suppress(Exception):
                                await _dispatch_steering_to_child(
                                    "clarify",
                                    {"answer": _payload},
                                    tgt_info,
                                )
                                tgt_info.waiting_for_clarification = False
                            tool_reply_msg = create_tool_call_message(
                                name=_pretty_name,
                                call_id=call["id"],
                                content=(
                                    f"Clarification answer sent upstream: {_payload!r}\n"
                                    "⏳ Waiting for the original tool to finish…"
                                ),
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_reply_msg,
                                client,
                                _msg_dispatcher,
                            )
                            # The tool's eventual final result lands on this
                            # reply, not the tool_reply_msg pending stub nor the
                            # [clarification <call_id>] tail message that carried
                            # the question (see record_clarification).
                            tgt_info.clarify_placeholder = tool_reply_msg
                            continue

                        elif _action == "ask":
                            if _handle is None or not hasattr(_handle, "ask"):
                                await _steer_reply(
                                    f"⚠️ call_id={_target_call_id!r} ({_orig_fn}) has no "
                                    "ask capability.",
                                )
                                continue

                            _ask_extra_kwargs, _ask_ctx_opted_in = (
                                compute_context_injection(
                                    args={
                                        "include_parent_chat_context": args.get(
                                            "include_parent_context",
                                            False,
                                        ),
                                    },
                                    propagate_chat_context=propagate_chat_context,
                                    context_state=context_state,
                                    client_messages=client.messages,
                                    call_id=f"ask_{_target_call_id}_{call['id']}",
                                    accepts_parent_ctx=True,
                                    accepts_parent_ctx_cont=False,
                                    is_continuation_only=False,
                                )
                            )
                            _ask_kwargs: Dict[str, Any] = {"question": _payload}
                            if "_parent_chat_context" in _ask_extra_kwargs:
                                _ask_kwargs["_parent_chat_context"] = _ask_extra_kwargs[
                                    "_parent_chat_context"
                                ]

                            async def _do_ask(_h=_handle, _kw=_ask_kwargs):
                                return await forward_handle_call(
                                    _h,
                                    "ask",
                                    _kw,
                                    fallback_positional_keys=["question"],
                                )

                            _steer_call_dict = {
                                "id": call["id"],
                                "type": "function",
                                "function": {
                                    "name": "steer",
                                    "arguments": call["function"]["arguments"],
                                },
                            }
                            _t = asyncio.create_task(
                                _do_ask(),
                                name="ToolCall_steer_ask",
                            )
                            tools_data.save_task(
                                _t,
                                ToolCallMetadata(
                                    name=f"{_orig_fn}.ask",
                                    call_id=call["id"],
                                    assistant_msg=msg,
                                    call_dict=_steer_call_dict,
                                    call_idx=idx,
                                    is_interjectable=False,
                                    is_dynamic=True,
                                    chat_context=_ask_extra_kwargs.get(
                                        "_parent_chat_context",
                                    ),
                                    pause_event=None,
                                    tool_schema={
                                        "type": "function",
                                        "function": {"name": "steer"},
                                    },
                                    llm_arguments=_ask_kwargs,
                                    raw_arguments_json=(_payload or ""),
                                    context_opted_in=_ask_ctx_opted_in,
                                ),
                            )
                            continue

                        elif _action == "call":
                            if not _method:
                                await _steer_reply(
                                    '⚠️ action="call" requires a `method` name.',
                                )
                                continue
                            _custom_methods = (
                                DynamicToolFactory._discover_custom_public_methods(
                                    _handle,
                                )
                                if _handle is not None
                                else {}
                            )
                            if _method not in _custom_methods:
                                await _steer_reply(
                                    f"⚠️ No custom method {_method!r} on "
                                    f"call_id={_target_call_id!r} ({_orig_fn}). "
                                    f"Available: {sorted(_custom_methods)}",
                                )
                                continue

                            _bound = _custom_methods[_method]
                            try:
                                _parsed_payload = (
                                    json.loads(_payload) if _payload else {}
                                )
                                if not isinstance(_parsed_payload, dict):
                                    raise ValueError(
                                        "payload must be a JSON object string",
                                    )
                                inspect.signature(_bound).bind(**_parsed_payload)
                            except Exception as _val_exc:
                                await _steer_reply(
                                    f"⚠️ Invalid payload for method={_method!r}: "
                                    f"{_val_exc}. Expected a JSON object string "
                                    f"matching signature {inspect.signature(_bound)}.",
                                )
                                continue

                            _write_only = set(
                                getattr(_handle, "write_only_methods", None) or [],
                            ) | set(
                                getattr(_handle, "write_only_tools", None) or [],
                            )

                            async def _invoke_custom(
                                _m=_method,
                                _h=_handle,
                                _kw=_parsed_payload,
                            ):
                                return await forward_handle_call(_h, _m, _kw)

                            if _method in _write_only:
                                tool_msg = create_tool_call_message(
                                    name=_pretty_name,
                                    call_id=call["id"],
                                    content=(
                                        f"Operation {_method!r} acknowledged and "
                                        "forwarded to the running tool."
                                    ),
                                )
                                await insert_tool_message_after_assistant(
                                    assistant_meta,
                                    msg,
                                    tool_msg,
                                    client,
                                    _msg_dispatcher,
                                )
                                with suppress(Exception):
                                    asyncio.create_task(
                                        _invoke_custom(),
                                        name=f"ToolCall_steer_call_{_method}",
                                    )
                                continue

                            _steer_call_dict = {
                                "id": call["id"],
                                "type": "function",
                                "function": {
                                    "name": "steer",
                                    "arguments": call["function"]["arguments"],
                                },
                            }
                            _t = asyncio.create_task(
                                _invoke_custom(),
                                name=f"ToolCall_steer_call_{_method}",
                            )
                            tools_data.save_task(
                                _t,
                                ToolCallMetadata(
                                    name=f"{_orig_fn}.{_method}",
                                    call_id=call["id"],
                                    assistant_msg=msg,
                                    call_dict=_steer_call_dict,
                                    call_idx=idx,
                                    is_interjectable=False,
                                    is_dynamic=True,
                                    chat_context=None,
                                    pause_event=None,
                                    tool_schema={
                                        "type": "function",
                                        "function": {"name": "steer"},
                                    },
                                    llm_arguments=_parsed_payload,
                                    raw_arguments_json=(_payload or "{}"),
                                    context_opted_in=False,
                                ),
                            )
                            continue

                        else:
                            await _steer_reply(
                                f"⚠️ Unknown action {_action!r}. Valid actions: "
                                "stop, interject, pause, resume, clarify, call, ask.",
                            )
                            continue

                    # Over-quota calls were already pruned above; this guards
                    # the remainder.
                    if tools_data.has_exceeded_quota_for_tool(name):
                        continue

                    # At the per-tool concurrency cap the call is refused with
                    # a tool-error message and never scheduled.
                    if tools_data.has_exceeded_concurrent_limit_for_tool(name):
                        tool_msg = create_tool_call_message(
                            name=name,
                            call_id=call["id"],
                            content=(
                                f"⚠️ Cannot start '{name}': "
                                f"max_concurrent={tools_data.normalized[name].max_concurrent} "
                                "already reached — at capacity. Use `wait` for an "
                                "existing call to finish, or "
                                'steer(call_id=<id>, action="stop") one before retrying.'
                            ),
                        )
                        await insert_tool_message_after_assistant(
                            assistant_meta,
                            msg,
                            tool_msg,
                            client,
                            _msg_dispatcher,
                        )
                        continue

                    elif lname_cf == "ask_about_completed_tool":
                        # The docstring is frozen; askable ids arrive via
                        # "[askable <call_id>]" tail messages
                        # (ToolsData.record_tool_completed_askable).
                        _tool_id = (
                            args.get("tool_id") if isinstance(args, dict) else None
                        )
                        _question = (
                            args.get("question") if isinstance(args, dict) else None
                        )

                        _entry = (
                            tools_data._completed_askable_tools.get(_tool_id)
                            if isinstance(_tool_id, str)
                            else None
                        )
                        if _entry is None:
                            _completed_name = (
                                tools_data._completed_tool_names.get(_tool_id)
                                if isinstance(_tool_id, str)
                                else None
                            )
                            if _completed_name is not None:
                                _err = (
                                    f"Cannot ask about tool_id={_tool_id!r} "
                                    f"({_completed_name}). This tool completed "
                                    "successfully but was not steerable — it "
                                    "executed as a direct function call with no "
                                    "inner reasoning trajectory to inspect. Its "
                                    "result is already visible in the outer "
                                    "transcript above."
                                )
                            else:
                                _available = list(
                                    tools_data._completed_askable_tools.keys(),
                                )
                                _err = (
                                    f"No tool found with tool_id={_tool_id!r}. "
                                    "This ID does not match any completed tool "
                                    "call. Available tool_ids for retrospective "
                                    f"inspection: {_available}"
                                )
                            tool_msg = create_tool_call_message(
                                name="ask_about_completed_tool",
                                call_id=call["id"],
                                content=f"⚠️ {_err}",
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )
                            continue

                        _completed_handle = _entry.get("handle")
                        _aact_extra_kwargs, _aact_ctx_opted_in = (
                            compute_context_injection(
                                args={},
                                propagate_chat_context=propagate_chat_context,
                                context_state=context_state,
                                client_messages=client.messages,
                                call_id=f"ask_{_tool_id}_{call['id']}",
                                accepts_parent_ctx=True,
                                accepts_parent_ctx_cont=False,
                                is_continuation_only=False,
                            )
                        )
                        _aact_kwargs: Dict[str, Any] = {"question": _question}
                        if "_parent_chat_context" in _aact_extra_kwargs:
                            _aact_kwargs["_parent_chat_context"] = _aact_extra_kwargs[
                                "_parent_chat_context"
                            ]

                        async def _do_completed_ask(
                            _h=_completed_handle,
                            _kw=_aact_kwargs,
                        ):
                            return await forward_handle_call(
                                _h,
                                "ask",
                                _kw,
                                fallback_positional_keys=["question"],
                            )

                        _aact_call_dict = {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": "ask_about_completed_tool",
                                "arguments": call["function"]["arguments"],
                            },
                        }
                        _t = asyncio.create_task(
                            _do_completed_ask(),
                            name="ToolCall_ask_about_completed_tool",
                        )
                        tools_data.save_task(
                            _t,
                            ToolCallMetadata(
                                name=f"{_entry.get('name', '?')}.ask",
                                call_id=call["id"],
                                assistant_msg=msg,
                                call_dict=_aact_call_dict,
                                call_idx=idx,
                                is_interjectable=False,
                                is_dynamic=True,
                                chat_context=_aact_extra_kwargs.get(
                                    "_parent_chat_context",
                                ),
                                pause_event=None,
                                tool_schema={
                                    "type": "function",
                                    "function": {"name": "ask_about_completed_tool"},
                                },
                                llm_arguments=_aact_kwargs,
                                raw_arguments_json=call["function"]["arguments"],
                                context_opted_in=_aact_ctx_opted_in,
                            ),
                        )
                        continue

                    # A tool that does not exist or was not visible this turn
                    # (hallucinated, or hidden by tool_policy) gets an error
                    # tool response so the transcript stays valid; an
                    # unresolved tool_call would make later LLM calls fail.
                    if name not in policy_tools_norm:
                        tool_msg = create_tool_call_message(
                            name=name,
                            call_id=call["id"],
                            content=(
                                f"⚠️ Error: Tool '{name}' is not available. "
                                "The tool may have been removed or does not exist. "
                                "Please proceed without using this tool."
                            ),
                        )
                        await insert_tool_message_after_assistant(
                            assistant_meta,
                            msg,
                            tool_msg,
                            client,
                            _msg_dispatcher,
                        )
                        continue

                    await tools_data.schedule_base_tool_call(
                        msg,
                        name=name,
                        args_json=call["function"]["arguments"],
                        call_id=call["id"],
                        call_idx=idx,
                        context_state=context_state,
                        propagate_chat_context=propagate_chat_context,
                        assistant_meta=assistant_meta,
                        msg_dispatcher=_msg_dispatcher,
                        initial_paused=not pause_event.is_set(),
                    )

                if _persist_response_emitted:
                    pass  # fall through to section F → persist wait
                else:
                    assistant_meta[id(msg)] = {
                        "results_count": 0,
                    }

                    # Placeholder tool replies go in immediately so API
                    # ordering holds even if an interjection arrives instantly.
                    try:
                        await ensure_placeholders_for_pending(
                            assistant_msg=msg,
                            tools_data=tools_data,
                            assistant_meta=assistant_meta,
                            client=client,
                            msg_dispatcher=_msg_dispatcher,
                            time_ctx=time_ctx,
                        )
                    except Exception as _ph_exc:
                        logger.error(
                            f"Failed to insert immediate placeholders: {_ph_exc!r}",
                        )

                    # Eager policies: if gates are still unsatisfied after the
                    # calls just scheduled, grant another LLM turn now
                    # (overlapping in-flight tools) instead of waiting. The
                    # re-evaluation uses the updated called_tools so eagerness
                    # ends as soon as the policy stops requesting it, and only
                    # runs when this turn was already eager: non-eager policies
                    # must not get an extra same-step callback.
                    if tool_policy is not None and _policy_eager:
                        try:
                            _eager_snapshot = {
                                n: s.fn for n, s in tools_data.normalized.items()
                            }
                            if _policy_accepts_history:
                                _eager_result = tool_policy(
                                    runtime_state.step_index,
                                    _eager_snapshot,
                                    list(runtime_state.called_tools),
                                )
                            else:
                                _eager_result = tool_policy(
                                    runtime_state.step_index,
                                    _eager_snapshot,
                                )
                            _, _, _still_eager = _parse_tool_policy_result(
                                _eager_result,
                            )
                            if _still_eager:
                                llm_turn_required = True
                        except Exception as _eager_exc:
                            logger.error(
                                f"tool_policy eager re-check failed: {_eager_exc!r}",
                            )

                    continue  # finished scheduling tools, back to the very top

            # ── F. No new tool calls ─────────────────────────────────────
            # Three cases reach here:
            #   • pending non-empty, not all blocked on clarification →
            #     older tools are still in flight (persist mode loops back
            #     to wait; otherwise they are cancelled and the text wins).
            #   • pending non-empty, all blocked on clarification → the LLM
            #     ended without answering; the blocked tasks are cancelled
            #     so the loop exits instead of deadlocking.
            #   • pending empty → a plain assistant message; return it.
            if tools_data.pending:
                blocked_on_clar = [
                    t
                    for t in tools_data.pending
                    if getattr(
                        tools_data.info.get(t),
                        "waiting_for_clarification",
                        False,
                    )
                ]
                not_blocked = [
                    t for t in tools_data.pending if t not in blocked_on_clar
                ]

                if blocked_on_clar and not not_blocked:
                    logger.info(
                        f"LLM returned content while {len(blocked_on_clar)} task(s) "
                        f"await clarification. Cancelling blocked tasks to exit.",
                        prefix=ICONS["auto_cancel"],
                    )
                    for t in blocked_on_clar:
                        t.cancel()
                    await asyncio.gather(*blocked_on_clar, return_exceptions=True)
                    for t in blocked_on_clar:
                        tools_data.pending.discard(t)
                    # Fall through to return the final answer
                else:
                    if persist:
                        # A bare text response never cancels in-flight tools
                        # in persist mode; branch A races tool completions,
                        # notifications, interjections and cancellation.
                        continue
                    # A text-only response with tools in flight is a valid
                    # termination signal.
                    logger.info(
                        f"LLM returned text-only response while {len(not_blocked)} "
                        f"task(s) are in-flight. Auto-cancelling to terminate.",
                        prefix=ICONS["auto_cancel"],
                    )
                    await tools_data.cancel_pending_tasks()
                    # Fall through to return the final answer

            # A patient interjection from the last LLM step, or anything still
            # queued, is processed before returning.
            try:
                if deferred_llm_turn or not interject_queue.empty():
                    deferred_llm_turn = False
                    continue  # drain interjections at top-of-loop; grants one extra LLM turn
            except Exception:
                pass

            if timer.has_exceeded_time():
                return await _handle_limit_reached(
                    f"timeout ({timeout}s) exceeded",
                )

            if timer.has_exceeded_msgs():
                return await _handle_limit_reached(
                    f"max_steps ({max_steps}) exceeded",
                )

            final_content = extract_substantive_text(msg["content"])

            # An empty/whitespace-only terminal turn must never override a
            # substantive answer already in the transcript: a model with
            # nothing left to add can still return empty content on a later
            # turn. Multi-handle and the plain return both read final_content
            # after this point, so it is resolved once here. Persist mode is
            # exempt because it never finalizes here: an empty turn surfaces
            # an empty response and re-enters the persist wait, and with
            # response_format the turn's answer is the response-tool payload
            # rather than text, so the nudge/loud-fail below would inject
            # spurious turns and then end a loop that only an explicit stop
            # may end.
            if final_content is None and not persist:
                _substantive_content = None
                for _hist_msg in reversed(client.messages):
                    _hist_role = _hist_msg.get("role")
                    if _hist_role == "user" and not is_loop_authored_message(_hist_msg):
                        # A genuine user turn boundary (not a loop-authored
                        # status message): an answer past it belongs to a
                        # different question.
                        break
                    if _hist_role != "assistant":
                        continue
                    _hist_content = extract_substantive_text(_hist_msg.get("content"))
                    if _hist_content is not None:
                        _substantive_content = _hist_content
                        break

                if _substantive_content is not None:
                    final_content = _substantive_content
                elif _empty_final_answer_retries < _MAX_EMPTY_FINAL_ANSWER_RETRIES:
                    # No substantive answer anywhere in this cycle either: the
                    # model gets a bounded number of chances before a loud
                    # failure. The nudge is appended at the tail (nothing
                    # dispatched is touched) and marked loop-authored so it can
                    # never pass for a genuine user turn boundary above.
                    _empty_final_answer_retries += 1
                    await _msg_dispatcher.append_msgs(
                        [
                            loop_user_notice(
                                "Produce your final answer as text.",
                                _nudge_msg=True,
                            ),
                        ],
                    )
                    continue
                else:
                    # Retries exhausted: fail loudly rather than return an
                    # empty result silently.
                    notice = {
                        "role": "assistant",
                        "content": (
                            "No final answer was produced: the model returned "
                            "empty content after "
                            f"{_MAX_EMPTY_FINAL_ANSWER_RETRIES} nudge attempt(s), "
                            "with no substantive answer anywhere in this "
                            "conversation."
                        ),
                    }
                    await _msg_dispatcher.append_msgs([notice])
                    logger.error(
                        "Empty final answer after "
                        f"{_MAX_EMPTY_FINAL_ANSWER_RETRIES} nudge attempt(s); no "
                        "substantive assistant content exists in this conversation.",
                        prefix=ICONS["llm_error"],
                    )
                    return notice["content"]

            # ── Multi-handle mode ────────────────────────────────────────
            if multi_handle_coordinator is not None:
                if multi_handle_coordinator.should_terminate():
                    # All requests completed/cancelled and persist=False.
                    logger.info(
                        "Multi-handle mode: all requests completed, terminating loop.",
                        prefix=ICONS["completed"],
                    )
                    multi_handle_coordinator.close()
                    return final_content
                else:
                    logger.info(
                        f"Multi-handle mode: {multi_handle_coordinator.registry.pending_count()} request(s) still pending.",
                        prefix=ICONS["pending"],
                    )
                    continue

            # ── Persist mode: wait for the next interjection ─────────────
            if persist:
                # The turn-complete response reaches the outer handle so the
                # ConversationManager can tell "response (awaiting input)"
                # from in-progress "notification" events. It marks the wait
                # state, so it is sent even when the turn produced no text: a
                # model with nothing to say after a message that asks for
                # nothing returns empty content, and a driver waiting for the
                # response (`unify act --jsonl`) would otherwise wait forever.
                _response_to_surface = (
                    _persist_response_content
                    if _persist_response_content is not None
                    else final_content
                )
                _outer = outer_handle_container[0] if outer_handle_container else None
                if (
                    _outer is not None
                    and hasattr(_outer, "_notification_q")
                    and not _suppress_persist_response
                ):
                    await _outer._notification_q.put(
                        {
                            "type": "response",
                            "content": _response_to_surface or "",
                        },
                    )
                _persist_response_content = None
                _persist_response_emitted = False

                # A parked turn's chain of thought is never consulted again:
                # the next dispatch starts from a fresh user interjection, so
                # provider reasoning payloads (encrypted blobs, reasoning
                # summaries) on every completed assistant message are pure
                # re-billed bulk from here on. Shed them now rather than
                # waiting for a storage review to cover the span — reviews
                # lag turns, and the lag is paid on every call in between.
                try:
                    _shed = 0
                    for _m in client.messages or []:
                        if isinstance(_m, dict) and _m.get("role") == "assistant":
                            _shed += strip_reasoning_payloads(_m)
                    if _shed:
                        _rebaseline_watermark_hash(client)
                except Exception:
                    pass

                logger.info(
                    "Persist mode: waiting for next interjection...",
                    prefix=ICONS["pause"],
                )
                try:
                    from ...events.manager_event_logging import (
                        publish_persist_session_phase,
                    )

                    await publish_persist_session_phase(_outer, "awaiting_input")
                except Exception:
                    pass
                while True:
                    cancel_waiter = asyncio.create_task(
                        cancel_event.wait(),
                        name="PersistCancelWait",
                    )
                    interject_waiter = asyncio.create_task(
                        interject_queue.get(),
                        name="PersistInterjectWait",
                    )
                    done, pending = await asyncio.wait(
                        {cancel_waiter, interject_waiter},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for p in pending:
                        p.cancel()
                        await asyncio.gather(p, return_exceptions=True)

                    if cancel_event.is_set():
                        raise asyncio.CancelledError

                    if interject_waiter not in done:
                        continue

                    interjection = interject_waiter.result()

                    # Transcript-note sentinels append the loop-authored note
                    # and stay in persist wait; the model reads it on its next
                    # granted turn.
                    if (
                        isinstance(interjection, dict)
                        and "_transcript_note" in interjection
                    ):
                        try:
                            _note = str(
                                (interjection.get("_transcript_note") or {}).get(
                                    "text",
                                )
                                or "",
                            )
                            if _note:
                                await _msg_dispatcher.append_msgs(
                                    [loop_user_notice(_note)],
                                )
                        except Exception:
                            pass
                        continue

                    # Transcript-compaction sentinels: the covered turns were
                    # consolidated by a storage review; shed their raw tool
                    # payloads and stay in persist wait.
                    if (
                        isinstance(interjection, dict)
                        and "_compact_transcript" in interjection
                    ):
                        try:
                            _n = int(
                                (interjection.get("_compact_transcript") or {}).get(
                                    "reviewed_messages",
                                )
                                or 0,
                            )
                            if _n > 0:
                                compact_reviewed_messages(client, _n)
                        except Exception:
                            pass
                        continue

                    # Mirror sentinels are transcript-only (no user message).
                    # Process them in-place and stay in persist wait — resuming
                    # the full loop would trigger an LLM call with a trailing
                    # assistant message, which strict models reject.
                    if isinstance(interjection, dict) and "_mirror" in interjection:
                        try:
                            _ms = interjection.get("_mirror") or {}
                            _m = _ms.get("method")
                            _kw = _ms.get("kwargs") or {}
                            if isinstance(_m, str) and _m:
                                merged = dict(_kw if isinstance(_kw, dict) else {})
                                for _key in ("_custom", "_aliases", "_fallback"):
                                    if _key in _ms:
                                        merged[_key] = _ms[_key]
                                await _synthesize_mirrored_helper_calls(_m, merged)
                        except Exception:
                            pass
                        continue

                    # A real interjection goes back on the queue for the
                    # normal drain path.
                    try:
                        await interject_queue.put(interjection)
                        logger.info(
                            "Persist mode: interjection received, resuming loop",
                            prefix=ICONS["resume"],
                        )
                        from ...events.manager_event_logging import (
                            publish_persist_session_phase,
                        )

                        await publish_persist_session_phase(_outer, "resumed")
                    except Exception:
                        pass
                    break

                timer.reset()
                continue  # Back to top of loop to process the interjection

            # final_content is non-empty here (or the loop returned earlier
            # with a loud error).
            return final_content  # DONE!

    except asyncio.CancelledError:  # graceful shutdown
        # Every running tool task is cancelled and awaited first so each can
        # release resources; only then is the same CancelledError re-raised,
        # preserving asyncio semantics for upstream callers.
        await tools_data.cancel_pending_tasks()
        raise
    finally:
        with suppress(Exception):
            TOOL_LOOP_LINEAGE.reset(_token)
