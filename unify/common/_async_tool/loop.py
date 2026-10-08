import asyncio
import unillm
import hashlib
import json
import inspect
import copy
from dataclasses import dataclass, field

from typing import (
    Awaitable,
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
from .event_bus_util import to_event_bus
from .messages import (
    find_unreplied_assistant_entries,
    generate_with_preprocess,
)
from .message_dispatcher import LoopMessageDispatcher
from .tools_utils import create_tool_call_message
from ..llm_helpers import (
    DEFAULT_TOOL_SCHEMA_STRICT,
    method_to_schema,
    _dumps,
)
from .loop_config import (
    LoopConfig,
    TOOL_LOOP_LINEAGE,
)
from .timeout_timer import TimeoutTimer
from .messages import (
    insert_tool_message_after_assistant,
    is_mutable,
    is_loop_authored_message,
    loop_user_notice,
    extract_substantive_text,
)
from .tools_data import ToolsData
from . import cache_discipline as _cache_discipline
from . import loop_stop as _loop_stop_mod
from . import cell_reply as _cell_reply
from . import bound_request as _bound_request
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
    # The tool list this session advertises, fixed at
    # its first call and kept across compression restarts.
    session_tools_schema: Optional[list] = None
    # UNIFY_STEP_CAP_REPLY=last_word: the tool-less turns given at the step
    # limit, and how many of them gave no answer (the draft was used).
    step_cap_last_word_turns: int = 0
    step_cap_last_word_fallbacks: int = 0
    # Replies dropped and sent again because their choice carried a
    # provider error.
    provider_error_retries: int = 0
    # UNIFY_REPLY_CHANNEL=code+text: turns a cell's reply() ended, and those
    # whose text was a computed value rather than a string literal.
    replies_from_cell: int = 0
    replies_from_value: int = 0
    # UNIFY_STEP_CAP_COMPACT: compactions at the step limit, in all and in
    # the current request, the ones that failed or ran out of time, and the
    # compacted context a loop leaves its handle to restart from
    # (``AsyncToolLoopHandle._compact_context``'s result).
    step_cap_compactions: int = 0
    step_cap_compactions_in_request: int = 0
    step_cap_compaction_failures: int = 0
    step_cap_compacted: Optional[tuple] = None
    # Model calls cancelled after they were sent, by cause ("stop", "cancel":
    # the requester's cancel of the request). The provider bills what it
    # received; unillm keeps such a call running and reports its charge
    # when the answer arrives (``_bill_abandoned_call``), so the run's meter
    # counts it. This is the count of those calls.
    cancelled_turns: int = 0
    cancelled_turns_by_cause: Dict[str, int] = field(default_factory=dict)
    # UNIFY_LOOP_STOP: requests ended for making no progress.
    loop_stops: int = 0
    # UNIFY_COMPACTION_KEEP_PREFIX=on: a keep-prefix restart waits for its
    # first measured call (``keep_prefix_unmeasured``). If that call is still
    # over the compression threshold, what the restart kept is itself too
    # large, and the session's next compaction rebuilds as shipped, from the
    # summary alone (``keep_prefix_shipped_next``), so it cannot compact the
    # same prefix again and again. Both read the session's own usage only.
    keep_prefix_unmeasured: bool = False
    keep_prefix_shipped_next: bool = False
    keep_prefix_fallbacks: int = 0


# How long a cancelled request waits for its running calls to stop before it
# abandons them and still ends in its response (a tool that swallows the
# cancellation would otherwise hold the response for as long as it runs).
_CANCEL_GRACE_S = 2.0

# UNIFY_STEP_CAP_COMPACT: how many times one request may be compacted at the
# step limit; at its next limit the limit stops it as without the switch.
STEP_CAP_COMPACTIONS = 2

# A reply whose choice carries a provider error is sent again this many
# times, after 1 s then 2 s (UniLLM's transient retry already ran inside
# the call: the provider answered HTTP 200, so it saw no failure).
_PROVIDER_ERROR_RETRIES = 2
_PROVIDER_ERROR_BACKOFF_S = 1.0


def _completion_provider_error(completion: Any) -> Optional[str]:
    """The provider error a completion's first choice carries, if any.

    OpenRouter reports an upstream failure part-way through a reply as HTTP
    200 with ``finish_reason: "error"`` and ``choices[0].error`` (e.g. 502
    "Stream ended before a terminal response event"). LiteLLM maps the
    finish reason to ``stop`` and keeps the original under the choice's
    ``provider_specific_fields``; the content is null, partial, or (after
    UniLLM's postprocessing) the reasoning summary.
    """
    try:
        choice = completion.choices[0]
    except Exception:
        return None
    fields = getattr(choice, "provider_specific_fields", None)
    if not isinstance(fields, dict):
        fields = {}
    error = fields.get("error") or getattr(choice, "error", None)
    if fields.get("native_finish_reason") != "error" and not error:
        return None
    if isinstance(error, dict):
        return f"{error.get('code', '?')}: {error.get('message', 'error')}"
    return str(error or "finish_reason error")


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
    so gated required policies cannot be bypassed by compressing context, and
    asks for parallel tool calls. ``{"gated": True}`` asks for those two
    without the eager turn (see ``_policy_gates_turn``).
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


def _policy_gates_turn(result: Any) -> bool:
    """Whether a ``tool_policy`` result gates the turn it is evaluated for.

    A gated turn withholds ``compress_context`` from the visible schema and
    asks for parallel tool calls, so the required calls are made together
    and cannot be bypassed by compressing. Eager results gate their turn;
    ``{"gated": True}`` gates it without the eager turn.
    """
    _, _, eager = _parse_tool_policy_result(result)
    opts = result[2] if len(result) >= 3 else None
    return eager or (isinstance(opts, dict) and bool(opts.get("gated", False)))


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


def _requeue_at_front(queue: asyncio.Queue, item: Any) -> None:
    """Put *item*, just taken off *queue*, back at its head, ahead of anything
    queued after it, so the next reader sees the queue in its original order."""
    behind = []
    while not queue.empty():
        behind.append(queue.get_nowait())
    queue.put_nowait(item)
    for later in behind:
        queue.put_nowait(later)


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
        self.flush_deferred()

    def flush_deferred(self) -> None:
        """Emit the lines held back until the first LLM thinking line.

        A loop that ends before its first LLM step never reaches that line,
        so the loop also calls this on the way out.
        """
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


def with_first_message_context(msg: dict, context: str) -> dict:
    """*msg* as a new message whose content opens with *context*.

    Text content becomes ``context``, a rule, then the text; a list of
    content blocks gets *context* as a leading text block. Any other content
    is left as it is.
    """
    content = msg.get("content")
    if isinstance(content, str):
        content = f"{context}\n\n---\n\n{content}"
    elif isinstance(content, list):
        content = [{"type": "text", "text": context}, *content]
    else:
        return msg
    return {**msg, "content": content}


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
    max_consecutive_failures: int = 3,
    prune_tool_duplicates: bool = True,
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
    outer_handle_container: Optional[list] = None,
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
    runtime_state: Optional[ToolLoopRuntimeState] = None,
    fixed_tools_schema: Optional[list[dict]] = None,
    first_message_context: Optional[str] = None,
    compression_tools_on_demand: bool = False,
    reply_channel: bool = False,
    on_turn_boundary: Optional[Callable[[], Awaitable[Optional[str]]]] = None,
    bind_request: Any = False,
) -> str:
    r"""
    Run a function-calling dialogue between an LLM and a set of Python
    callables until the model yields a final plain-text answer.

    One model turn runs at a time, and no turn starts while a tool call runs.
    A turn's tool calls run in call order, each to completion, and their
    results are appended in that order, each with its own ``tool_call_id``.
    No message interrupts a model call or a tool call: a message pushed onto
    ``interject_queue`` (``handle.submit``) is appended at the next turn
    boundary. Setting ``cancel_event`` (``handle.stop``) cancels the model
    call or the tool call in flight and ends the loop with
    ``asyncio.CancelledError``; in a persistent loop the requester's cancel
    of the request (``handle.cancel_request``) cancels them and ends the
    request, keeping the session. Exceptions inside tools are
    serialised and shown to the model; ``max_consecutive_failures``
    back-to-back crashes abort the loop with ``RuntimeError``.

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
        The requester's next messages (``{"message": str}`` or a string), read
        only at a turn boundary, plus the loop's own sentinels: a persistent
        loop's request cancel (``_cancel_request``), a storage review's note
        (``_transcript_note``) and its compaction (``_compact_transcript``).

    cancel_event : ``asyncio.Event``
        Set by the outer caller to stop the loop: the model call or tool call
        in flight is cancelled and ``asyncio.CancelledError`` propagates.

    max_consecutive_failures : ``int``, default ``3``
        After this many back-to-back tool exceptions the loop raises
        ``RuntimeError`` rather than crash-and-retry indefinitely.

    prune_tool_duplicates : ``bool``, default ``True``
        Drop model-requested tool calls with identical ``function.name`` and
        argument JSON, in place, before they reach chat history or scheduling.

    propagate_chat_context : ``ChatContextPropagation``, default ``LLM_DECIDES``
        Whether a filtered snapshot of this loop's conversation (genuine user
        turns and substantive assistant text only) is threaded into child
        tools that accept a ``_parent_chat_context`` keyword argument.
        ``ALWAYS`` injects on every such call, ``NEVER`` on none, and
        ``LLM_DECIDES`` exposes an ``include_parent_chat_context`` parameter
        the model may set to ``true`` (omission means no context).

    tool_policy : ``Callable | None``, default ``None``
        Controls tool exposure and whether a tool call is required on a given
        turn. Receives the turn index (from ``0``) and the full
        ``{name → callable}`` mapping, plus optionally the list of previously
        called tool names as a third positional argument. Returns
        ``(policy, tools)`` or ``(policy, tools, opts)``: ``policy`` is
        ``"auto"`` or ``"required"`` (fed straight into ``tool_choice``) and
        ``tools`` is the possibly-filtered mapping of tools visible that
        turn. ``{"eager": True}`` or ``{"gated": True}`` gate the turn:
        ``compress_context`` is withheld from it (forced over-threshold
        compression still applies) and parallel tool calls are asked for.

    parent_chat_context : ``list[dict] | None``
        Chat history passed from an outer caller, shown in the runtime
        context and forwarded to tools that opt into context.

    log_steps : ``bool | str``, default ``True``
        Step logging to ``LOGGER``: ``False`` for none, ``True`` for everything
        except system messages, ``"full"`` for everything.

    timeout : ``int | None``, default ``None``
        Activity-based timeout in seconds; the timer resets after each LLM
        response and each appended message. A tool call that runs past it is
        cancelled. ``None`` disables it.

    raise_on_limit : ``bool``, default ``False``
        If ``True``, exceeding the timeout or ``max_steps`` raises
        ``asyncio.TimeoutError`` or ``RuntimeError``; if ``False`` the loop
        terminates gracefully with a summary message.

    persist : ``bool``, default ``False``
        If ``True``, content without tool calls does not end the loop: the
        turn's response goes to the handle and the loop parks until the next
        message arrives on ``interject_queue``, so one loop serves many
        requests. The loop then ends only via ``cancel_event``.

    time_awareness : ``bool``, default ``False``
        If ``True``, a time-context system message is injected at the start of
        the conversation and tool results carry their timing.

    fixed_tools_schema : ``list[dict] | None``
        The exact tool list to send on every call instead of one built from
        ``tools`` -- a fork sends its parent's list so its requests extend
        the parent's. A listed tool this loop does not implement is refused
        when called.

    first_message_context : ``str | None``
        Text that opens the first user message of this loop (the request, or
        the first user message of a seeded batch), separated from it by a
        rule. A compressed session restarts with it again, since its first
        message is then a new one. ``None`` sends the message as given.

    compression_tools_on_demand : ``bool``, default ``False``
        ``True`` (the core tool surface): ``compress_context`` and the
        ``extra_compression_tools`` are offered only on the turn that must
        compress, and are otherwise left out of the tool list
        (the session's fixed list included), so the list holds
        only the caller's tools.

    reply_channel : ``bool``, default ``False``
        ``True`` (the actor's task loop, ``UNIFY_REPLY_CHANNEL=code+text``): a
        cell's ``reply(text)`` ends the turn with that text as the reply, as
        a text reply would, without another model call (``cell_reply.py``).
        Ignored while the switch is off and in a loop whose answer is a
        response tool's.

    on_turn_boundary : optional coroutine function, default ``None``
        Called just before each model call, after every tool result (and any
        footer) is appended. A non-empty string it returns is appended as one
        loop-authored user message. Never called while a model call or a tool
        runs, nor after a turn has ended (the agent record).

    bind_request : ``bool`` or ``RequestSlot``, default ``False``
        ``True`` (the actor's task loop, ``UNIFY_BIND_REQUEST=on``): the
        requester's latest message is the current request a cell reads as
        ``request`` (``bound_request.py``); the handle passes its own slot,
        which a restart after compression keeps. Ignored while the switch is
        off; any other loop's cells have no ``request``.

    Returns
    -------
    str
        The assistant's final plain-text reply after every tool result has
        been fed back into the conversation.
    """
    cfg = LoopConfig(loop_id, lineage, TOOL_LOOP_LINEAGE.get([]))
    # The outer handle shares the loop's resolved label so its logs (stop,
    # submit, cancel) line up with the tool loop's, and the resolved lineage
    # so event payloads carry the full parent->child stack even when emitted
    # outside the tool loop ContextVar scope.
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
    # UNIFY_REPLY_CHANNEL=code+text: this loop's slot for a cell's reply,
    # ``None`` for a loop that takes none; nothing is set while it is off.
    _reply_token = _cell_reply.bind(
        reply_channel and _rf_norm is None,
    )
    _reply_slot = _cell_reply.current() if _reply_token is not None else None
    # UNIFY_BIND_REQUEST=on: this loop's current request, which its cells
    # read as ``request``; ``None`` for a loop that answers no requester.
    _request_token = _bound_request.bind(bind_request)
    _request_slot = _bound_request.current() if _request_token is not None else None

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
    # UNIFY_STEP_CAP_REPLY: reaching max_steps in a persistent loop ends the
    # request, not the loop, and every stop at the limit quotes the latest
    # draft of the request's answer ("draft"), or the answer the model gives
    # in one tool-less turn at the limit ("last_word"). Off: as shipped.
    from unify.settings import SETTINGS as _CAP_SETTINGS

    _step_cap_mode = _CAP_SETTINGS.step_cap_reply()
    _step_cap_reply = bool(_step_cap_mode)
    _step_cap_last_word = _step_cap_mode == "last_word"
    # UNIFY_STEP_CAP_COMPACT=on: at max_steps a loop that can compress its
    # context compacts it, and the request goes on from the compacted
    # context (at most STEP_CAP_COMPACTIONS times a request). Off: as shipped.
    _step_cap_compact = (
        getattr(_CAP_SETTINGS, "UNIFY_STEP_CAP_COMPACT", "") == "on"
        and bool(enable_compression)
        and bool(max_steps)
        and not raise_on_limit
    )
    # UNIFY_LOOP_STOP: the no-progress calls in a row of the current request.
    # A stop ends the request as the step limit does; with
    # UNIFY_STEP_CAP_REPLY off it takes the last word, so it always replies.
    # Only in a task loop that answers a requester with text and that no
    # other loop started (as UNIFY_PROMPT_ACCURACY's
    # test for a parent): never in a sub-agent, a review or its fork.
    _loop_stop = (
        _loop_stop_mod.Tracker(_loop_stop_mod.threshold())
        if _loop_stop_mod.enabled()
        and reply_channel
        and _rf_norm is None
        and parent_chat_context is None
        and len(cfg.lineage) < 2
        else None
    )

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
    # UNIFY_PROMPT_ACCURACY: not to a loop that has no parent -- one no other
    # loop started (its lineage is its own id) and given no parent context --
    # which no parent conversation exists for, and no continuation can reach.
    _no_parent = parent_chat_context is None and len(cfg.lineage) < 2
    _has_parent_chat_context = False
    if propagate_chat_context != ChatContextPropagation.NEVER and not _no_parent:
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
    # UNIFY_BIND_REQUEST=on: the request as the requester wrote it, without
    # the session context put before it below.
    _bound_request.record_first(_request_slot, message)
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

        if first_message_context:
            for index, msg in enumerate(seeded_batch):
                if msg.get("role") == "user":
                    seeded_batch[index] = with_first_message_context(
                        msg,
                        first_message_context,
                    )
                    break

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
        call_counts=runtime_state.call_counts,
    )
    logger.debug(
        f"[setup +{_setup_elapsed()}] ToolsData ready ({len(tools_data.normalized)} tools)",
    )
    # The handle answers a running call's question on that call's own queue.
    with suppress(Exception):
        if outer_handle_container and outer_handle_container[0] is not None:
            setattr(
                outer_handle_container[0],
                "_clarification_channels",
                tools_data.clarification_channels,
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
    # ── Initial user message (single-message path) ──────────────────────
    if seeded_batch is None:
        if isinstance(message, dict):
            initial_user_msg = message
        else:
            initial_user_msg = {"role": "user", "content": message}
        if first_message_context:
            initial_user_msg = with_first_message_context(
                initial_user_msg,
                first_message_context,
            )
        if time_ctx is not None and isinstance(initial_user_msg.get("content"), str):
            initial_user_msg["content"] = time_ctx.prefix_user_message(
                initial_user_msg["content"],
            )
        await _msg_dispatcher.append_msgs([initial_user_msg])

    async def _handle_limit_reached(
        reason: str,
        draft: Optional[str] = None,
        *,
        last_word: bool = False,
        stop: Optional[_loop_stop_mod.Stop] = None,
    ) -> str:
        """
        Terminate gracefully when *timeout* or *max_steps* is exceeded and
        `raise_on_limit` is *False*: append a short assistant notice,
        followed by *draft* when one is given. Every call has been answered
        by then (``_interrupt_turn`` answers the ones a limit interrupts).

        With *last_word* (UNIFY_STEP_CAP_REPLY=last_word at max_steps) the
        pending calls are answered as cancelled, the model is given one
        tool-less turn, and its answer, when it gives one, takes the place
        of *draft*. A loop stop (UNIFY_LOOP_STOP) passes its *stop* texts.
        """
        if last_word:
            await tools_data.cancel_pending_tasks_with_reply(
                (
                    stop.cancelled
                    if stop is not None
                    else f"Cancelled: the step limit ({reason}) ended the request "
                    "before this call finished."
                ),
                assistant_meta=assistant_meta,
                msg_dispatcher=_msg_dispatcher,
            )
            draft = await _last_word(reason, stop) or draft
        await tools_data.cancel_pending_tasks(grace=_CANCEL_GRACE_S)
        # A call the limit came before (one a seeded or resumed transcript
        # carried) is answered too, so the transcript the loop ends with
        # leaves no call unanswered.
        for entry in find_unreplied_assistant_entries(client):
            for call in entry["assistant_msg"].get("tool_calls") or []:
                if call.get("id") in entry["missing"] and not _call_answered(
                    call.get("id"),
                ):
                    await _answer_call(
                        entry["assistant_msg"],
                        call,
                        f"Not run: {reason} before this call started.",
                    )

        notice = {
            "role": "assistant",
            "content": f"🔚 Terminating early: {reason}"
            + (f"\n\nBest current answer:\n{draft}" if draft is not None else ""),
        }
        await _msg_dispatcher.append_msgs([notice])
        if log_steps:
            logger.info(f"Early exit – {reason}", prefix=ICONS["early_exit"])
        return notice["content"]

    def _request_draft() -> Optional[str]:
        """The latest reply text drafted for the current request, if any.

        Walks back to the request's own message, a genuine user turn (a
        loop-authored notice is not one), as the empty-final-answer guard
        does: text before it answered something else.
        """
        for _msg in reversed(client.messages or []):
            _role = _msg.get("role")
            if _role == "user" and not is_loop_authored_message(_msg):
                return None
            if _role == "assistant":
                _text = extract_substantive_text(_msg.get("content"))
                if _text:
                    return _text
        return None

    async def _last_word(
        reason: str,
        stop: Optional[_loop_stop_mod.Stop] = None,
    ) -> Optional[str]:
        """UNIFY_STEP_CAP_REPLY=last_word: one tool-less turn at the step limit.

        Every call of the request has been answered. A loop-authored notice
        says the limit is reached and asks for the best answer now; the
        model is then called once with no tools offered (no tool_choice is
        forced). Returns the reply's text, or ``None`` when the call fails,
        is stopped, runs past the loop's timeout or gives no text: the
        caller then uses the draft, so the request never ends silent. A
        loop stop (UNIFY_LOOP_STOP) passes its *stop* texts.
        """
        runtime_state.step_cap_last_word_turns += 1
        await _msg_dispatcher.append_msgs(
            [
                loop_user_notice(
                    (
                        stop.notice
                        if stop is not None
                        else f"The step limit for this request is reached ({reason}): "
                        "no more tools can be called for it. Reply now with your "
                        "best answer to the request."
                    ),
                ),
            ],
        )
        # The record a fork (storage review, compression) is built from
        # stays the last request that offered the session's tools.
        _recorded = _cache_discipline.last_sent_request(client)
        call = asyncio.create_task(
            generate_with_preprocess(
                client,
                preprocess_msgs,
                return_full_completion=True,
                stateful=True,
                prompt_caching=prompt_caching,
            ),
            name="StepCapLastWord",
        )
        stopped = asyncio.create_task(cancel_event.wait(), name="CancelEventWait")
        text: Optional[str] = None
        why = "no text"
        try:
            done, _ = await asyncio.wait(
                {call, stopped},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if call not in done:
                why = "stopped" if stopped in done else f"no reply in {timeout}s"
                call.cancel()
                await asyncio.gather(call, return_exceptions=True)
            elif call.exception() is not None:
                why = f"{type(call.exception()).__name__}: {call.exception()}"
            else:
                reply = (client.messages or [{}])[-1]
                if reply.get("role") == "assistant":
                    # No tools were offered; a call the reply names anyway
                    # would be run after the limit, so it is dropped.
                    if reply.get("tool_calls") and is_mutable(client, reply):
                        reply.pop("tool_calls", None)
                    text = extract_substantive_text(reply.get("content"))
        finally:
            stopped.cancel()
            await asyncio.gather(stopped, return_exceptions=True)
            _cache_discipline.restore_sent_request(client, _recorded)
        if text is None:
            runtime_state.step_cap_last_word_fallbacks += 1
        logger.info(
            (stop.label if stop is not None else "Step limit")
            + " – last word: "
            + ("answered" if text is not None else f"none ({why}); using the draft"),
            prefix=ICONS["early_exit"],
        )
        return text

    async def _end_request_at_step_limit(
        stop: Optional[_loop_stop_mod.Stop] = None,
    ) -> None:
        """UNIFY_STEP_CAP_REPLY in a persistent loop: end the request at max_steps.

        The pending calls are cancelled and answered as such, the reply says
        the request stopped at the step limit and quotes its latest draft,
        and the loop parks for the next request, whose steps are counted
        from its own message. The reply also stays in the transcript, so
        the model reads on the next request where the last one stopped.
        With ``last_word`` the model first gets one tool-less turn, and its
        answer is quoted in place of the draft; the transcript then holds
        the notice and that answer instead of the reply.

        UNIFY_LOOP_STOP ends a request the same way, with its *stop* texts
        and reply mode; it logs its own line.
        """
        reason = (
            stop.reason if stop is not None else f"max_steps ({max_steps}) exceeded"
        )
        draft = _request_draft()
        await tools_data.cancel_pending_tasks_with_reply(
            (
                stop.cancelled
                if stop is not None
                else f"Cancelled: the step limit ({reason}) ended the request before "
                "this call finished."
            ),
            assistant_meta=assistant_meta,
            msg_dispatcher=_msg_dispatcher,
        )
        last_word = stop.last_word if stop is not None else _step_cap_last_word
        answer = await _last_word(reason, stop) if last_word else None
        if answer is not None:
            draft = answer
        content = (
            stop.headline
            if stop is not None
            else f"🔚 Stopped at the step limit: {reason}, so this request ended "
            "before it was finished. The session is still open: the next "
            "message starts a new request."
        )
        if draft is not None:
            content += f"\n\nBest current answer:\n{draft}"
        else:
            content += "\n\nNo reply text was drafted for this request."
        if answer is None:
            await _msg_dispatcher.append_msgs(
                [{"role": "assistant", "content": content}],
            )
        if log_steps and stop is None:
            logger.info(
                f"Step limit – {reason}; waiting for the next request",
                prefix=ICONS["early_exit"],
            )
        _outer = outer_handle_container[0] if outer_handle_container else None
        if _outer is not None and hasattr(_outer, "_notification_q"):
            await _outer._notification_q.put({"type": "response", "content": content})
        await _park_until_next_request()
        runtime_state.step_cap_compactions_in_request = 0
        timer.reset()
        # Counting per request is UNIFY_STEP_CAP_REPLY's; a loop stop with
        # it off keeps counting the whole loop, as shipped.
        if _step_cap_reply:
            timer.start_request()

    def _can_compact_at_step_limit() -> bool:
        """UNIFY_STEP_CAP_COMPACT: whether this loop's handle can compact it.

        The handle must be this loop's own (it shares the runtime state), as
        it restarts the loop from what the compaction returns.
        """
        if not _step_cap_compact:
            return False
        _outer = outer_handle_container[0] if outer_handle_container else None
        return getattr(_outer, "_runtime_state", None) is runtime_state and callable(
            getattr(_outer, "_compact_context", None),
        )

    def _drop_queued_cancels() -> int:
        """Take every queued request cancel off the queue, keeping the order
        of the rest; how many were dropped."""
        kept, dropped = [], 0
        while not interject_queue.empty():
            item = interject_queue.get_nowait()
            if isinstance(item, dict) and "_cancel_request" in item:
                dropped += 1
            else:
                kept.append(item)
        for item in kept:
            interject_queue.put_nowait(item)
        return dropped

    def _request_cancel_queued() -> bool:
        """Whether the requester's cancel of the request waits in the queue."""
        return any(
            isinstance(item, dict) and "_cancel_request" in item
            for item in list(getattr(interject_queue, "_queue", None) or ())
        )

    async def _until_request_cancel() -> None:
        while not _request_cancel_queued():
            await asyncio.sleep(0.05)

    async def _compact_at_step_limit() -> str:
        """UNIFY_STEP_CAP_COMPACT: compact the context at max_steps.

        Returns ``"compacted"`` when the handle's context compression (the
        one a full context gets) returned a compacted context; the caller
        then ends this loop with the compression signal, and the handle
        restarts it from that context, the request going on with its steps
        counted from there. Calls still running are cancelled and answered
        as such first, since a restarted loop cannot collect them.

        ``"defer"``: the turn ends without a model call first (a cell's
        reply() is waiting to be taken, or the requester's cancel of the
        request is queued), so nothing is compacted for it; the caller goes
        on with the step. ``"limit"``: the limit applies as without the
        switch, because the loop cannot compact, the request was compacted
        STEP_CAP_COMPACTIONS times, or the compaction failed or ran past the
        loop's timeout. A stop of the loop during the compaction cancels it
        and stops the loop.
        """
        if not _can_compact_at_step_limit():
            return "limit"
        if (_reply_slot is not None and _reply_slot.replied) or (
            persist and _request_cancel_queued()
        ):
            return "defer"
        if runtime_state.step_cap_compactions_in_request >= STEP_CAP_COMPACTIONS:
            return "limit"
        runtime_state.step_cap_compactions_in_request += 1
        reason = f"max_steps ({max_steps}) exceeded"
        await tools_data.cancel_pending_tasks_with_reply(
            f"Cancelled: the step limit ({reason}) was reached before this "
            "call finished.",
            assistant_meta=assistant_meta,
            msg_dispatcher=_msg_dispatcher,
        )
        logger.info(
            f"Step limit – {reason}; compacting the conversation "
            f"({runtime_state.step_cap_compactions_in_request} of "
            f"{STEP_CAP_COMPACTIONS} for this request)",
            prefix=ICONS["early_exit"],
        )
        _outer = outer_handle_container[0]
        compaction = asyncio.create_task(
            _outer._compact_context(),
            name="StepCapCompaction",
        )
        stopped = asyncio.create_task(cancel_event.wait(), name="CancelEventWait")
        watchers = [stopped]
        if persist:
            watchers.append(
                asyncio.create_task(
                    _until_request_cancel(),
                    name="RequestCancelWait",
                ),
            )
        try:
            done, _ = await asyncio.wait(
                {compaction, *watchers},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for task in (compaction, *watchers):
                if not task.done():
                    task.cancel()
            await asyncio.gather(compaction, *watchers, return_exceptions=True)
        if stopped in done:
            raise asyncio.CancelledError
        if compaction in done and not compaction.cancelled():
            if compaction.exception() is None:
                runtime_state.step_cap_compacted = compaction.result()
                runtime_state.step_cap_compactions += 1
                logger.info(
                    "Step limit – compacted; the request goes on",
                    prefix=ICONS["early_exit"],
                )
                return "compacted"
            why = f"{type(compaction.exception()).__name__}: {compaction.exception()}"
        elif any(task in done for task in watchers[1:]):
            logger.info(
                "Step limit – compaction stopped: the requester cancelled the "
                "request",
                prefix=ICONS["early_exit"],
            )
            return "defer"
        else:
            why = f"no result in {timeout}s"
        runtime_state.step_cap_compaction_failures += 1
        logger.info(
            f"Step limit – compaction failed ({why}); stopping at the limit",
            prefix=ICONS["early_exit"],
        )
        return "limit"

    async def _end_request_on_cancel(reason: Optional[str]) -> None:
        """A persistent loop's requester cancelled the running request.

        The step in flight was already cancelled by the wake-up that brought
        the cancel. The pending calls are cancelled and answered as such (so
        none is scheduled again), the transcript says the request was
        cancelled, the request ends in its response (the text drafted for it
        so far, marked cancelled) and the loop parks for the next request.
        """
        draft = _request_draft()
        await tools_data.cancel_pending_tasks_with_reply(
            "Cancelled: the requester cancelled the request before this call "
            "finished.",
            assistant_meta=assistant_meta,
            msg_dispatcher=_msg_dispatcher,
            grace=_CANCEL_GRACE_S,
        )
        notice = (
            "🔚 Cancelled: the requester cancelled this request before it was "
            "finished. The session is still open: the next message starts a "
            "new request."
        )
        if reason:
            notice += f"\n\nReason given: {reason}"
        await _msg_dispatcher.append_msgs([{"role": "assistant", "content": notice}])
        logger.info(
            "Request cancelled by the requester; waiting for the next request",
            prefix=ICONS["early_exit"],
        )
        _outer = outer_handle_container[0] if outer_handle_container else None
        if _outer is not None and hasattr(_outer, "_notification_q"):
            await _outer._notification_q.put(
                {"type": "response", "content": draft or "", "cancelled": True},
            )
        await _park_until_next_request()
        runtime_state.step_cap_compactions_in_request = 0
        timer.reset()
        if _step_cap_reply:
            timer.start_request()

    async def _park_until_next_request() -> None:
        """Park a persistent loop until its next request arrives.

        Returns when an interjection that starts a request is back on the
        queue, or when the loop is stopped; the caller then resumes at the
        top of the loop, which processes either.
        """
        _outer = outer_handle_container[0] if outer_handle_container else None
        # UNIFY_REPLY_CHANNEL=code+text: the next request starts with no reply.
        if _reply_slot is not None:
            _reply_slot.clear()

        # A cancel still queued was sent for the request that has just
        # ended (it came after that request's last call), so it has nothing
        # to cancel; a message queued around it is the next request.
        _dropped = _drop_queued_cancels()
        if _dropped:
            logger.info(
                f"Persist mode: {_dropped} cancel(s) ignored, the request "
                "they were sent for has ended",
                prefix=ICONS["pause"],
            )

        # A parked turn keeps the reasoning payloads of its assistant
        # messages: shedding them would rewrite every sent message, so the
        # next call would start cold.

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

            # Whatever the waiter took goes back to the head of the
            # queue; the check after the drain ends the loop.
            if cancel_event.is_set():
                if interject_waiter in done:
                    _requeue_at_front(
                        interject_queue,
                        interject_waiter.result(),
                    )
                break

            interjection = interject_waiter.result()

            # A cancel with no request running has nothing to end: the
            # last request already ended in its response.
            if isinstance(interjection, dict) and "_cancel_request" in interjection:
                logger.info(
                    "Persist mode: cancel ignored, no request is running",
                    prefix=ICONS["pause"],
                )
                continue

            # Transcript-note sentinels append the loop-authored note
            # and stay in persist wait; the model reads it on its next
            # granted turn.
            if isinstance(interjection, dict) and "_transcript_note" in interjection:
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
            # consolidated by a storage review. Sent messages are never
            # edited, so the reviewed span keeps its bytes; stay in persist
            # wait.
            if isinstance(interjection, dict) and "_compact_transcript" in interjection:
                continue

            # The next request goes back to the head of the queue, ahead of
            # anything sent after it, so the drain takes them in order.
            try:
                _requeue_at_front(interject_queue, interjection)
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

    def _outer_handle() -> Any:
        return outer_handle_container[0] if outer_handle_container else None

    async def _deliver_clarification(info: Any, question_payload: Any) -> None:
        """A running call asks the requester a question.

        It goes to the handle (``next_clarification``), whose
        ``answer_clarification`` puts the answer on the call's own queue; the
        call waits for it. The model gets no turn while the call runs.
        """
        try:
            if isinstance(question_payload, dict):
                question_text = question_payload.get("question", "")
            else:
                question_text = str(question_payload)
        except Exception:
            question_text = str(question_payload)
        with suppress(Exception):
            logger.info(
                f"Clarification requested – {info.name}: {question_text}",
                prefix=ICONS["clarification"],
            )
        outer = _outer_handle()
        if outer is not None and hasattr(outer, "_clar_q"):
            await outer._clar_q.put(
                {
                    "type": "clarification",
                    "call_id": info.call_id,
                    "tool_name": info.name,
                    "question": question_text,
                },
            )

    async def _deliver_notification(info: Any, payload: Any) -> None:
        """A running call's progress notification, for the handle only
        (``next_notification``); it never enters the transcript."""
        with suppress(Exception):
            if isinstance(payload, dict):
                _msg_txt = str(
                    payload.get("message") or payload.get("status") or payload,
                )
            else:
                _msg_txt = str(payload)
            logger.info(
                f"Notification from {info.name}: {_msg_txt}",
                prefix=ICONS["notification"],
            )
        outer = _outer_handle()
        if outer is not None and hasattr(outer, "_notification_q"):
            event_payload = (
                payload if isinstance(payload, dict) else {"message": str(payload)}
            )
            await outer._notification_q.put(
                {
                    "type": "notification",
                    "call_id": info.call_id,
                    "tool_name": info.name,
                    **event_payload,
                },
            )

    async def _answer_call(msg: dict, call: dict, content: str) -> None:
        """Give one tool call of *msg* *content* as its reply."""
        await insert_tool_message_after_assistant(
            assistant_meta,
            msg,
            create_tool_call_message(
                name=call["function"]["name"],
                call_id=call["id"],
                content=content,
            ),
            client,
            _msg_dispatcher,
        )

    async def _run_tool_call(msg: dict, call: dict, idx: int) -> Optional[str]:
        """Run one base tool call to completion and append its result.

        While it runs, only a stop (``cancel_event``), the call's own
        clarification and notification queues, the loop's timeout and, in a
        persistent loop, the requester's cancel of the request are watched;
        a message for the session stays queued for the next boundary.
        Returns ``None`` once the result is appended, or ``"cancel_request"``
        or ``"timeout"`` when the call was interrupted (the caller cancels
        it if it still runs, and answers it). A stop raises
        ``asyncio.CancelledError``.
        """
        task = await tools_data.schedule_base_tool_call(
            msg,
            name=call["function"]["name"],
            args_json=call["function"]["arguments"],
            call_id=call["id"],
            call_idx=idx,
            context_state=context_state,
            propagate_chat_context=propagate_chat_context,
            assistant_meta=assistant_meta,
            msg_dispatcher=_msg_dispatcher,
        )
        if task is None:
            await _answer_call(
                msg,
                call,
                f"⚠️ Error: '{call['function']['name']}' was not run: its call "
                "limit is reached.",
            )
            return None
        info = tools_data.info[task]
        while not task.done():
            if timer.has_exceeded_time():
                return "timeout"
            watchers: Dict[str, asyncio.Task] = {
                "cancel": asyncio.create_task(
                    cancel_event.wait(),
                    name="CancelEventWait",
                ),
            }
            if info.clar_up_queue is not None:
                watchers["clarification"] = asyncio.create_task(
                    info.clar_up_queue.get(),
                    name="ClarificationQueueGet",
                )
            if info.notification_queue is not None:
                watchers["notification"] = asyncio.create_task(
                    info.notification_queue.get(),
                    name="NotificationQueueGet",
                )
            if persist:
                watchers["cancel_request"] = asyncio.create_task(
                    _until_request_cancel(),
                    name="RequestCancelWait",
                )
            try:
                await asyncio.wait(
                    {task, *watchers.values()},
                    timeout=timer.remaining_time(),
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                for watcher in watchers.values():
                    if not watcher.done():
                        watcher.cancel()
                await asyncio.gather(*watchers.values(), return_exceptions=True)

            def _took(kind: str) -> bool:
                watcher = watchers.get(kind)
                return (
                    watcher is not None
                    and watcher.done()
                    and not watcher.cancelled()
                    and watcher.exception() is None
                )

            # What a queue waiter took is delivered even when a stop came
            # with it, so nothing a call sent is lost.
            if _took("clarification"):
                await _deliver_clarification(info, watchers["clarification"].result())
            if _took("notification"):
                await _deliver_notification(info, watchers["notification"].result())
            if cancel_event.is_set():
                raise asyncio.CancelledError
            if _took("cancel_request"):
                return "cancel_request"
        # A cancel that came as the call ended (a cell interrupted for it,
        # say) still wins: the call is answered as cancelled, as one still
        # running would be, rather than with what the interruption left.
        if persist and _request_cancel_queued():
            return "cancel_request"
        await tools_data.process_completed_task(
            task=task,
            consecutive_failures=consecutive_failures,
            outer_handle_container=outer_handle_container,
            assistant_meta=assistant_meta,
            msg_dispatcher=_msg_dispatcher,
        )
        return None

    async def _interrupt_turn(
        msg: dict,
        rest: list[dict],
        reason: str,
    ) -> None:
        """A call of *msg* was interrupted: cancel it and answer it, then
        answer the calls of the turn that never ran (*rest*)."""
        await tools_data.cancel_pending_tasks_with_reply(
            f"Cancelled: {reason} before this call finished.",
            assistant_meta=assistant_meta,
            msg_dispatcher=_msg_dispatcher,
            grace=_CANCEL_GRACE_S,
        )
        for later in rest:
            if _call_answered(later.get("id")):
                continue
            await _answer_call(
                msg,
                later,
                f"Not run: {reason} before this call started.",
            )

    def _call_answered(call_id: Optional[str]) -> bool:
        if not isinstance(call_id, str):
            return True
        if call_id in tools_data.completed_results:
            return True
        return any(
            m.get("role") == "tool" and m.get("tool_call_id") == call_id
            for m in client.messages or []
        )

    _INTERRUPT_REASONS = {
        "cancel_request": "the requester cancelled the request",
    }

    def _interrupt_reason(status: str) -> str:
        return _INTERRUPT_REASONS.get(status) or f"the timeout ({timeout}s) was reached"

    async def _run_missing_calls(amsg: dict, missing_ids: set) -> Optional[str]:
        """Run, in call order, the calls of *amsg* that have no reply yet.

        A call to a tool this loop does not have is answered with an error.
        Returns the status of an interruption (see ``_run_tool_call``), with
        the interrupted call and the ones after it answered, or ``None``.
        """
        calls = [
            c
            for c in (amsg.get("tool_calls") or [])
            if isinstance(c, dict) and c.get("id") in missing_ids
        ]
        for position, call in enumerate(calls):
            if _call_answered(call.get("id")):
                continue
            name = (call.get("function") or {}).get("name")
            if name not in tools_data.normalized:
                await _answer_call(
                    amsg,
                    call,
                    f"⚠️ Error: Tool '{name}' is not available. "
                    "The tool may have been removed or does not exist. "
                    "Please proceed without using this tool.",
                )
                continue
            idx = (amsg.get("tool_calls") or []).index(call)
            status = await _run_tool_call(amsg, call, idx)
            if status is not None:
                await _interrupt_turn(
                    amsg,
                    calls[position + 1 :],
                    _interrupt_reason(status),
                )
                return status
        return None

    async def _repair_unreplied(entries: list) -> Optional[str]:
        """Run the unanswered calls of *entries*, oldest first; see
        ``_run_missing_calls``. Each entry is repaired inside its own
        try/except: prune_over_quota_tool_calls may raise (a below-watermark
        mutation refused on a resumed client whose watermark carried over),
        and one failure must not abandon the entries after it."""
        for entry in entries:
            amsg = entry["assistant_msg"]
            if id(amsg) in assistant_meta:
                continue
            try:
                tools_data.prune_over_quota_tool_calls(amsg)
                if prune_tool_duplicates and amsg.get("tool_calls"):
                    unique, pruned = prune_duplicate_tool_calls(amsg["tool_calls"])
                    if pruned:
                        amsg["tool_calls"] = unique
                        entry["missing"] = [
                            cid for cid in entry["missing"] if cid not in pruned
                        ]
            except Exception as exc:
                logger.error(
                    f"Repair failed for one assistant entry; continuing with "
                    f"the rest: {exc}",
                    prefix="🚨",
                )
                continue
            assistant_meta.setdefault(id(amsg), {"results_count": 0})
            status = await _run_missing_calls(amsg, set(entry["missing"]))
            if status is not None:
                return status
        return None

    def _queued_message() -> bool:
        """Whether a message for the session (not a loop sentinel) is queued."""
        return any(
            not (
                isinstance(item, dict)
                and (
                    "_cancel_request" in item
                    or "_transcript_note" in item
                    or "_compact_transcript" in item
                )
            )
            for item in list(getattr(interject_queue, "_queue", None) or ())
        )

    # Consecutive replies of this turn dropped for a provider error.
    _provider_error_retries = 0
    # Bounded retries for a terminal turn that returns empty content with no
    # substantive answer anywhere else in the conversation to fall back on.
    _empty_final_answer_retries = 0
    _MAX_EMPTY_FINAL_ANSWER_RETRIES = 1

    logger.debug(f"[setup +{_setup_elapsed()}] entering main loop")

    try:
        while True:
            if timer.has_exceeded_time():
                return await _handle_limit_reached(
                    f"timeout ({timeout}s) exceeded",
                )

            if timer.has_exceeded_msgs():
                # UNIFY_STEP_CAP_COMPACT: compact and go on, or take the
                # step that ends the turn first; else the limit applies.
                _at_limit = (
                    await _compact_at_step_limit() if _step_cap_compact else "limit"
                )
                if _at_limit == "compacted":
                    return _COMPRESSION_SIGNAL
                if _at_limit == "limit":
                    if _step_cap_reply and persist:
                        await _end_request_at_step_limit()
                        _persist_response_content = None
                        _persist_response_emitted = False
                        continue
                    return await _handle_limit_reached(
                        f"max_steps ({max_steps}) exceeded",
                        _request_draft() if _step_cap_reply else None,
                        last_word=_step_cap_last_word,
                    )

            # Assistant tool_calls missing replies (a seeded batch or a
            # resumed transcript) are run, in call order, before anything
            # queued is appended. Each assistant message is repaired once.
            _repair_status = await _repair_unreplied(
                find_unreplied_assistant_entries(client),
            )
            if _repair_status == "timeout":
                return await _handle_limit_reached(
                    f"timeout ({timeout}s) exceeded",
                )

            # ── Turn boundary: drain the queue ──────────────────────────
            # Nothing runs here, so every message the requester sent since
            # the last boundary is appended now, in the order it was sent.
            _cancel_request: Optional[dict] = None
            while True:
                try:
                    extra = interject_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                # The requester cancelled the running request. What was
                # queued after the cancel stays queued for the next request.
                if isinstance(extra, dict) and "_cancel_request" in extra:
                    _cancel_request = dict(extra.get("_cancel_request") or {})
                    break
                # Transcript-note sentinel: a background process (e.g. a
                # storage review) leaves a loop-authored note in the
                # transcript; the model reads it whenever it next speaks.
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
                # has consolidated the covered turns. Sent messages are never
                # edited, so the reviewed span keeps its bytes.
                if isinstance(extra, dict) and "_compact_transcript" in extra:
                    continue

                if isinstance(extra, dict):
                    _msg_text = str(extra.get("message", "")).strip()
                else:
                    _msg_text = str(extra)

                try:
                    logger.info(
                        f"Interjection received: {_msg_text}",
                        prefix=ICONS["interjection"],
                    )
                except Exception:
                    pass

                if _msg_text:
                    # UNIFY_BIND_REQUEST=on: the requester's message is now
                    # the current request.
                    _bound_request.record(_request_slot, _msg_text)
                    _user_content = (
                        time_ctx.prefix_user_message(_msg_text)
                        if time_ctx is not None
                        else _msg_text
                    )
                    await _msg_dispatcher.append_msgs(
                        [
                            {
                                "role": "user",
                                "_interjection": True,
                                "content": _user_content,
                            },
                        ],
                    )

            # A stop ends the loop here: any further turn would be sent only
            # to be thrown away.
            if cancel_event.is_set():
                raise asyncio.CancelledError

            # A cancelled request ends here and the loop waits for the next.
            # Only a persistent loop serves more than one request; a loop
            # that is not persistent is ended with stop().
            if _cancel_request is not None:
                if persist:
                    await _end_request_on_cancel(_cancel_request.get("reason"))
                    _persist_response_content = None
                    _persist_response_emitted = False
                    continue
                logger.info(
                    "Cancel ignored: the loop is not persistent (stop() ends it)",
                    prefix=ICONS["pause"],
                )

            # ── C. Build this turn's toolkit ─────────────────────────────
            # Tool policy and tool subset for this turn. Gated policies (e.g.
            # discovery-first gates) keep the model on a narrow required
            # subset; tracking that keeps compress_context out of the schema
            # as an escape hatch.
            logger.debug(
                f"[setup +{_setup_elapsed()}] tool policy eval (step={runtime_state.step_index})",
            )
            _policy_gated = False
            _policy_mask_rules: Dict[str, str] = {}
            _policy_mask_default: Optional[str] = None
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
                    tool_choice_mode, filtered, _ = _parse_tool_policy_result(
                        _policy_result,
                    )
                    _policy_gated = _policy_gates_turn(_policy_result)
                    _policy_mask_rules, _policy_mask_default = (
                        _cache_discipline.policy_mask_rules(_policy_result)
                    )
                except Exception as _e:  # never abort the loop on mis-behaving policies
                    logger.error(
                        f"tool_policy raised on turn {runtime_state.step_index}: {_e!r}",
                    )
                    tool_choice_mode, filtered = "auto", _tools_snapshot
                    _policy_gated = False
                policy_tools_norm = normalise_tools(filtered)
            else:
                tool_choice_mode = "auto"
                policy_tools_norm = tools_data.normalized

            logger.debug(
                f"[setup +{_setup_elapsed()}] building tool schemas ({len(policy_tools_norm)} tools)",
            )
            _compress_schema = (
                method_to_schema(compress_context, "compress_context")
                if enable_compression
                else None
            )

            if _over_threshold and enable_compression:
                # Over threshold → compress_context plus any caller-specified
                # extra compression tools (pulled from the full tool set so
                # policy gates are bypassed).
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
                # even past max_total_calls — an over-quota call is refused at
                # execution time instead (see prune_over_quota_tool_calls), so
                # hitting the cap never changes what the model can see.
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
                    if not (
                        compression_tools_on_demand
                        and name in (extra_compression_tools or ())
                    )
                ]
                # compress_context stays out of gated turns so required
                # discovery/tool policies cannot be satisfied by compressing;
                # forced over-threshold compression above still applies.
                # compression_tools_on_demand: only that forced turn has it.
                if (
                    _compress_schema is not None
                    and not _policy_gated
                    and not compression_tools_on_demand
                ):
                    visible_base_tools_schema.append(_compress_schema)

            # The response-submission tool is in the schema whenever
            # response_format is set. It means "end the current turn" (the
            # tool-call analogue of a bare text response).
            #
            #   persist=True  → "send_response"  (signals turn completion,
            #                    loop continues waiting for next interjection)
            #   persist=False → "final_response"  (terminates the loop)
            _response_tool_name = "send_response" if persist else "final_response"
            # "Ready" means present in the schema (response_format configured
            # and injection succeeded).
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

            tmp_tools = list(visible_base_tools_schema)

            # What this turn assembled is what it allows; what it advertises
            # is the session's fixed list. A call
            # to anything outside the allowed set is refused below, with the
            # rule that masks it, so the list never changes mid-session.
            _turn_available = frozenset(
                _cache_discipline.schema_names(tmp_tools),
            )
            if (
                runtime_state.session_tools_schema is None
                and fixed_tools_schema is not None
            ):
                runtime_state.session_tools_schema = copy.deepcopy(
                    list(fixed_tools_schema),
                )
            if runtime_state.session_tools_schema is None:
                runtime_state.session_tools_schema = (
                    _cache_discipline.build_session_schema(
                        base_schemas={
                            name: method_to_schema(
                                spec.fn,
                                name,
                                expose_context_control=(
                                    propagate_chat_context
                                    == ChatContextPropagation.LLM_DECIDES
                                ),
                                has_parent_context=bool(parent_chat_context),
                            )
                            for name, spec in tools_data.normalized.items()
                            if not (
                                compression_tools_on_demand
                                and name in (extra_compression_tools or ())
                            )
                        },
                        compress_schema=(
                            None if compression_tools_on_demand else _compress_schema
                        ),
                        turn_schemas=[
                            schema
                            for schema in tmp_tools
                            if not compression_tools_on_demand
                            or _cache_discipline.schema_name(schema)
                            not in {
                                "compress_context",
                                *(extra_compression_tools or ()),
                            }
                        ],
                    )
                )
            # compression_tools_on_demand: the turn that must compress
            # sends its own list (the session's list has no compression
            # tools); compression starts a new prefix anyway.
            _on_demand_turn = (
                compression_tools_on_demand and _over_threshold and enable_compression
            )
            if not _on_demand_turn:
                tmp_tools = runtime_state.session_tools_schema
            # Set once, before the first request: the key names the
            # prefix (model, system prompt, this fixed list) unless a
            # key is already set, as a fork's is.
            _cache_discipline.ensure_cache_affinity(client, tmp_tools)
            _session_tool_names = frozenset(
                _cache_discipline.schema_names(tmp_tools),
            )
            _turn_available = _turn_available & _session_tool_names
            _turn_mask_rules = dict(_policy_mask_rules)
            _turn_mask_default = _policy_mask_default
            if _over_threshold and enable_compression:
                _turn_mask_rules = {}
                _turn_mask_default = (
                    "the context window is nearly full, so "
                    "`compress_context` has to be called now"
                )
            elif _policy_gated:
                _turn_mask_rules.setdefault(
                    "compress_context",
                    "compression waits until this turn's required calls "
                    "have been made",
                )

            # ── D. Ask the LLM what to do next ───────────────────────────
            # A stop that landed while this turn was being built ends the
            # loop rather than dispatching a turn only to cancel it.
            if cancel_event.is_set():
                raise asyncio.CancelledError

            # UNIFY_REPLY_CHANNEL=code+text: a cell called reply(); its text is
            # this step's assistant message and no model is asked. None
            # while the switch is off.
            _cell_reply_msg = _reply_slot.take() if _reply_slot is not None else None

            # UNIFY_LOOP_STOP: K model calls in a row that made no progress
            # end the request here, as the step limit would, instead of
            # making another call. Before the boundary hook, so a record
            # block is never consumed by a call that is not made.
            if (
                _loop_stop is not None
                and _cell_reply_msg is None
                and _loop_stop.observe(
                    client.messages or [],
                    in_flight=bool(tools_data.pending),
                )
            ):
                _stop = _loop_stop_mod.Stop(
                    k=_loop_stop.k,
                    last_word=_step_cap_mode != "draft",
                )
                runtime_state.loop_stops += 1
                logger.info(
                    f"Loop stop – {_stop.reason}; ending the request with "
                    + ("the model's last word" if _stop.last_word else "its draft"),
                    prefix=ICONS["early_exit"],
                )
                _loop_stop.count = 0
                if persist:
                    await _end_request_at_step_limit(_stop)
                    _persist_response_content = None
                    _persist_response_emitted = False
                    continue
                return await _handle_limit_reached(
                    _stop.reason,
                    _request_draft(),
                    last_word=_stop.last_word,
                    stop=_stop,
                )

            # The agent record: what is new in the shared record for this
            # agent, appended after this turn's tool results. Reached only when
            # a model call is about to be made, so never after a turn has ended
            # (a cell's reply() above ends it with no model call).
            if on_turn_boundary is not None and _cell_reply_msg is None:
                _record_block = await on_turn_boundary()
                if _record_block:
                    await _msg_dispatcher.append_msgs(
                        [loop_user_notice(_record_block, _record_block=True)],
                    )

            logger.debug(
                f"[setup +{_setup_elapsed()}] ready for LLM call (step={runtime_state.step_index}, {len(tmp_tools)} tools)",
            )
            if log_steps and _cell_reply_msg is None:
                logger.begin_thinking()

            if _cell_reply_msg is None:
                await to_event_bus(
                    {"role": "assistant", "_thinking_in_flight": True},
                    cfg,
                )

            if _cell_reply_msg is not None:
                # The reply goes in as the model's text reply would, and the
                # step goes on from here exactly as for one (section F).
                client.append_messages([_cell_reply_msg])
                _full_completion = None
                runtime_state.replies_from_cell += 1
                if _cell_reply_msg[_cell_reply.FROM_VALUE_KEY]:
                    runtime_state.replies_from_value += 1
                logger.info(
                    "A cell's reply() ends the turn; no model call",
                    prefix=ICONS["completed"],
                )
            else:
                # The model call runs to completion; only a stop, or in a
                # persistent loop the requester's cancel of the request,
                # cancels it.
                _gen_kwargs = {
                    "return_full_completion": True,
                    "tools": tmp_tools,
                    "tool_choice": tool_choice_mode,
                    "stateful": True,
                    "prompt_caching": prompt_caching,
                }
                if max_parallel_tool_calls is not None:
                    _gen_kwargs["parallel_tool_calls"] = max_parallel_tool_calls > 1
                elif _policy_gated:
                    # Discovery-first (and other gated turns) expose multiple
                    # required tools that must be callable in one assistant turn.
                    _gen_kwargs["parallel_tool_calls"] = True

                llm_task = asyncio.create_task(
                    generate_with_preprocess(
                        client,
                        _apply_reasoning_model_compat(
                            _gen_kwargs,
                            tool_choice_mode,
                        ),
                        **_gen_kwargs,
                    ),
                    name="LLMGenerate",
                )
                cancel_waiter = asyncio.create_task(
                    cancel_event.wait(),
                    name="CancelEventWait",
                )
                watchers = [cancel_waiter]
                if persist:
                    watchers.append(
                        asyncio.create_task(
                            _until_request_cancel(),
                            name="RequestCancelWait",
                        ),
                    )
                try:
                    await asyncio.wait(
                        {llm_task, *watchers},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    for watcher in watchers:
                        if not watcher.done():
                            watcher.cancel()
                    await asyncio.gather(*watchers, return_exceptions=True)
                    if not llm_task.done():
                        # A stop, the requester's cancel of the request, or
                        # the loop's own cancellation. The unanswered
                        # dispatch leaves the transcript as it was; its cost
                        # still reaches the run's meter through unillm.
                        _cause = (
                            "stop"
                            if cancel_event.is_set()
                            else "cancel" if _request_cancel_queued() else "loop"
                        )
                        runtime_state.cancelled_turns += 1
                        runtime_state.cancelled_turns_by_cause[_cause] = (
                            runtime_state.cancelled_turns_by_cause.get(_cause, 0) + 1
                        )
                        llm_task.cancel()
                        await asyncio.gather(llm_task, return_exceptions=True)

                if log_steps:
                    logger.emit_thinking_fallback()

                if llm_task.cancelled():
                    if not cancel_event.is_set() and _request_cancel_queued():
                        # The drain at the top ends the request.
                        continue
                    raise asyncio.CancelledError
                if llm_task.exception() is not None:
                    try:
                        llm_task.result()
                    except Exception as e:
                        raise Exception(
                            f"LLM call failed: {type(e).__name__}: {e}",
                        ) from e
                _full_completion = llm_task.result()
                # A stop that came with the answer still wins: nothing the
                # answer asks for is run.
                if cancel_event.is_set():
                    raise asyncio.CancelledError

            # A reply the provider marked as failed (OpenRouter's HTTP 200
            # with a choice error, which LiteLLM maps to finish_reason
            # "stop") is not the model's turn: it is dropped and the same
            # turn is sent again, a bounded number of times.
            _provider_error = _completion_provider_error(_full_completion)
            if _provider_error is not None:
                _failed = (client.messages or [None])[-1]
                if (
                    _provider_error_retries < _PROVIDER_ERROR_RETRIES
                    and isinstance(_failed, dict)
                    and _failed.get("role") == "assistant"
                    and is_mutable(client, _failed)
                ):
                    _provider_error_retries += 1
                    runtime_state.provider_error_retries += 1
                    client.messages.pop()
                    _backoff = _PROVIDER_ERROR_BACKOFF_S * 2 ** (
                        _provider_error_retries - 1
                    )
                    logger.error(
                        f"Provider error in the model's reply ({_provider_error}); "
                        f"sending the turn again in {_backoff:g}s (retry "
                        f"{_provider_error_retries}/{_PROVIDER_ERROR_RETRIES})",
                        prefix=ICONS["early_exit"],
                    )
                    with suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(cancel_event.wait(), _backoff)
                    continue
                logger.error(
                    f"Provider error in the model's reply ({_provider_error}); "
                    "retries spent, the reply is kept as it is",
                    prefix=ICONS["early_exit"],
                )
            _provider_error_retries = 0

            # The step's assistant message is the tail.
            msg = client.messages[-1]
            # UNIFY_REPLY_CHANNEL=code+text: a text reply is recorded as one.
            if (
                _reply_slot is not None
                and not msg.get("tool_calls")
                and _cell_reply.SOURCE_KEY not in msg
            ):
                _cell_reply.stamp(msg, source="text", from_value=False)
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
                        if runtime_state.keep_prefix_unmeasured:
                            runtime_state.keep_prefix_unmeasured = False
                            if _over_threshold:
                                runtime_state.keep_prefix_shipped_next = True
                                logger.info(
                                    "the first call after a keep-prefix "
                                    "compaction is still over the threshold "
                                    f"({_usage.prompt_tokens} prompt tokens); "
                                    "the next compaction rebuilds as shipped",
                                )

            with suppress(Exception):
                _cache_discipline.log_cache_use(
                    client,
                    _full_completion,
                    label=cfg.label,
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

            runtime_state.step_index += 1

            # ── E. Run the turn's tool calls ─────────────────────────────
            # Each call's arguments are JSON-parsed once here, the loop-owned
            # tools (response submission, compress_context) are handled
            # inline, and every other call runs to completion before the
            # next one starts, so the results land in call order.
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

                assistant_meta.setdefault(id(msg), {"results_count": 0})
                _turn_calls = list(msg["tool_calls"])
                _interrupted: Optional[str] = None
                for idx, call in enumerate(_turn_calls):  # capture index
                    name = call["function"]["name"]

                    # The step limit counts every message, so this turn's
                    # own message or its earlier calls' results can reach
                    # it: the calls from that point are not run, and the
                    # limit applies at the top of the loop as at any other
                    # boundary.
                    if timer.has_exceeded_msgs():
                        _limit = f"max_steps ({max_steps}) exceeded"
                        for later in _turn_calls[idx:]:
                            if not _call_answered(later.get("id")):
                                await _answer_call(
                                    msg,
                                    later,
                                    f"Not run: the step limit ({_limit}) was "
                                    "reached before this call started.",
                                )
                        break

                    # The session's tool list is fixed, so a tool this turn
                    # does not allow is refused here, by rule, instead of
                    # having been left out of the request. A refused call is
                    # not recorded as called: it must not satisfy a policy
                    # gate.
                    if name not in _turn_available:
                        await _answer_call(
                            msg,
                            call,
                            _cache_discipline.masked_tool_refusal(
                                name,
                                advertised=_session_tool_names,
                                available=_turn_available,
                                rule=(_turn_mask_rules.get(name) or _turn_mask_default),
                            ),
                        )
                        continue

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
                                f"Malformed tool-call arguments for {name} "
                                f"({len(_raw_args)} chars): {exc}",
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
                            await _answer_call(msg, call, refusal)
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

                            await _answer_call(
                                msg,
                                call,
                                _dumps(payload_for_return, indent=4),
                            )

                            if persist:
                                # The current turn's response; the loop goes on.
                                _persist_response_emitted = True
                                _persist_response_content = json.dumps(
                                    payload_for_return,
                                )
                                _interrupted = "response"
                                break  # exit the for-loop over tool_calls
                            return json.dumps(payload_for_return)
                        except Exception as _exc:
                            await _answer_call(
                                msg,
                                call,
                                "⚠️ Validation failed – proceeding with standard formatting step.\n"
                                + str(_exc),
                            )
                            continue

                    # A response tool call with no response_format configured is
                    # only reachable when the model hallucinates it.
                    _is_generic_response = (
                        name in ("final_response", "send_response") and _rf_norm is None
                    )
                    if _is_generic_response:
                        answer = args.get("answer") if isinstance(args, dict) else None
                        if answer is None:
                            answer = str(args) if args else ""

                        await _answer_call(msg, call, answer)

                        if persist:
                            _persist_response_emitted = True
                            _persist_response_content = answer
                            _interrupted = "response"
                            break
                        return answer

                    if name == "compress_context":
                        await _answer_call(
                            msg,
                            call,
                            "Compression initiated. Ending current loop "
                            "to restart with compressed context.",
                        )
                        return _COMPRESSION_SIGNAL

                    # Over-quota calls were already pruned above; this guards
                    # the remainder.
                    if tools_data.has_exceeded_quota_for_tool(name):
                        await _answer_call(
                            msg,
                            call,
                            f"⚠️ Error: '{name}' was not run: its call limit is "
                            "reached.",
                        )
                        continue

                    # A tool that does not exist or was not visible this turn
                    # (hallucinated, or hidden by tool_policy) gets an error
                    # tool response so the transcript stays valid; an
                    # unresolved tool_call would make later LLM calls fail.
                    if name not in policy_tools_norm:
                        await _answer_call(
                            msg,
                            call,
                            f"⚠️ Error: Tool '{name}' is not available. "
                            "The tool may have been removed or does not exist. "
                            "Please proceed without using this tool.",
                        )
                        continue

                    _status = await _run_tool_call(msg, call, idx)
                    if _status is not None:
                        await _interrupt_turn(
                            msg,
                            _turn_calls[idx + 1 :],
                            _interrupt_reason(_status),
                        )
                        _interrupted = _status
                        break

                if _interrupted == "response":
                    # The turn's response is sent; a call after it never runs.
                    for later in _turn_calls[idx + 1 :]:
                        if not _call_answered(later.get("id")):
                            await _answer_call(
                                msg,
                                later,
                                "Not run: the response was sent before this "
                                "call started.",
                            )
                elif _interrupted == "timeout":
                    return await _handle_limit_reached(
                        f"timeout ({timeout}s) exceeded",
                    )
                else:
                    # Every call has its result (a cancelled request is ended
                    # at the next boundary): back to the very top.
                    continue

            # ── F. No new tool calls ─────────────────────────────────────
            # A plain assistant message (or the response a persistent loop's
            # response tool sent): nothing is running. A message the
            # requester sent during the model call is part of the request
            # in a loop that ends here, so the model gets a turn for it; a
            # persistent loop answers this request first and takes the
            # message as its next one when it parks.
            if not persist and _queued_message():
                continue

            if timer.has_exceeded_time():
                return await _handle_limit_reached(
                    f"timeout ({timeout}s) exceeded",
                )

            # UNIFY_STEP_CAP_COMPACT: a reply given at the limit is the
            # request's answer; a loop that can compact goes on to give it.
            if timer.has_exceeded_msgs() and not _can_compact_at_step_limit():
                if _step_cap_reply and persist:
                    await _end_request_at_step_limit()
                    _persist_response_content = None
                    _persist_response_emitted = False
                    continue
                return await _handle_limit_reached(
                    f"max_steps ({max_steps}) exceeded",
                    _request_draft() if _step_cap_reply else None,
                    last_word=_step_cap_last_word,
                )

            final_content = extract_substantive_text(msg["content"])

            # An empty/whitespace-only terminal turn must never override a
            # substantive answer already in the transcript: a model with
            # nothing left to add can still return empty content on a later
            # turn. The persist response and the plain return both read
            # final_content after this point, so it is resolved once here.
            # Persist mode is exempt because it never finalizes here: an
            # empty turn surfaces an empty response and re-enters the
            # persist wait, and with
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
                if _outer is not None and hasattr(_outer, "_notification_q"):
                    await _outer._notification_q.put(
                        {
                            "type": "response",
                            "content": _response_to_surface or "",
                        },
                    )
                _persist_response_content = None
                _persist_response_emitted = False

                await _park_until_next_request()
                runtime_state.step_cap_compactions_in_request = 0

                timer.reset()
                if _step_cap_reply:
                    timer.start_request()
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
        _cell_reply.unbind(_reply_token)
        _bound_request.unbind(_request_token)
        # A loop stopped before its first LLM step still logs its stop.
        if log_steps:
            logger.flush_deferred()
        with suppress(Exception):
            TOOL_LOOP_LINEAGE.reset(_token)
