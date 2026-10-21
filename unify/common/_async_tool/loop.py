import asyncio
import unillm
import hashlib
import json
import copy
from dataclasses import dataclass, field

from typing import (
    Awaitable,
    Dict,
    Union,
    Callable,
    Any,
    Set,
    Optional,
    TYPE_CHECKING,
)
from contextlib import suppress

from ...logger import LOGGER
from ..tool_spec import ToolSpec
from unify import transcripts
from .messages import (
    find_unreplied_assistant_entries,
    generate_with_preprocess,
)
from .message_dispatcher import LoopMessageDispatcher
from .tools_utils import create_tool_call_message
from ..llm_client import count_requester_message
from ..llm_helpers import method_to_schema
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
from . import cell_reply as _cell_reply
from .context_compression import (
    COMPACTION_THRESHOLD,
    _COMPRESSION_SIGNAL,
    context_over_threshold,
)
from .response_format import (
    NormalizedResponseFormat,
    normalize_response_format,
)
from ...common.hierarchical_logger import ICONS

if TYPE_CHECKING:
    from unillm.types import PromptCacheParam


@dataclass
class ToolLoopRuntimeState:
    call_counts: Dict[str, int] = field(default_factory=dict)
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
    # The actor's loop (``tools_in_code``) compacts by itself before the next
    # model call. ``in_code_tail_from`` is where the messages after the last
    # request start (its reply first); the restart's summary request is that
    # request plus those messages (``in_code_tail``), so the summary covers
    # the whole conversation.
    in_code_tail_from: Optional[int] = None
    in_code_tail: Optional[list] = None


# How long a cancelled request waits for its running calls to stop before it
# abandons them and still ends in its response (a tool that swallows the
# cancellation would otherwise hold the response for as long as it runs).
_CANCEL_GRACE_S = 2.0

# Back-to-back tool exceptions after which the loop raises ``RuntimeError``
# rather than crash-and-retry indefinitely.
MAX_CONSECUTIVE_FAILURES = 3

async def _memory_pause(client: Any, *, compacting: bool = False, new_message: bool = False) -> None:
    """UNIFY_MEMORY_IN_EPISODE=on: a turn boundary of the request's actor, where the pauses due start their segment
    fork (``hooks.turn_boundary``: the marked ones, a compaction's, a new message's). Off, or for another loop, or
    with no request run: returns at once and nothing runs. Never raises."""
    try:
        from unify.memory.integration import hooks as _mv2

        if not _mv2.in_episode_active():
            return
        await asyncio.to_thread(
            _mv2.turn_boundary,
            client,
            compacting=compacting,
            new_message=new_message,
        )
    except Exception as exc:  # noqa: BLE001 - a pause never stops the loop
        LOGGER.warning(f"memory: a pause was not taken ({type(exc).__name__})")


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


#: The turn's text once a withheld reply was sent the retries and still withheld.
WITHHELD_REPLY = (
    "The provider withheld the model's reply to this turn (finish_reason "
    "content_filter) {attempts} times in a row, so this turn has no answer."
)


def _usage_counts(completion: Any) -> tuple[Optional[int], bool]:
    """The completion's output tokens (None when not reported) and whether its
    reported usage is all zero (every count given is 0, and at least one is)."""
    usage = getattr(completion, "usage", None)
    if usage is None:
        return None, False

    def _count(name: str) -> Optional[int]:
        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    output = _count("completion_tokens")
    if output is None:
        output = _count("output_tokens")
    counts = [
        c
        for c in (
            _count("prompt_tokens"),
            _count("completion_tokens"),
            _count("total_tokens"),
            _count("input_tokens"),
            _count("output_tokens"),
        )
        if c is not None
    ]
    return output, bool(counts) and all(c == 0 for c in counts)


def _completion_withheld(completion: Any) -> bool:
    """Whether the provider withheld a completion's reply: its first choice
    finished with ``content_filter`` (as LiteLLM maps it, or OpenRouter's own
    ``native_finish_reason``) with no tool call, and the reply is a stand-in
    rather than the model's: its text is blank, or it reports no output
    tokens, or its usage is all zero. (A provider's stand-in text is not
    matched: a filtered reply with output tokens and text is the model's
    turn.)"""
    try:
        choice = completion.choices[0]
    except Exception:
        return False
    fields = getattr(choice, "provider_specific_fields", None)
    if not isinstance(fields, dict):
        fields = {}
    if "content_filter" not in (
        getattr(choice, "finish_reason", None),
        fields.get("native_finish_reason"),
    ):
        return False
    message = getattr(choice, "message", None)
    if getattr(message, "tool_calls", None):
        return False
    if not str(getattr(message, "content", None) or "").strip():
        return True
    output, all_zero = _usage_counts(completion)
    return output == 0 or all_zero


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


def _unknown_tool_text(name: str) -> str:
    """The answer to a call of a tool the session does not offer."""
    return f"Tool '{name}' does not exist; `execute_code` is your one tool, and `tools.list()` lists the rest."


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
    message: str | dict | list[dict],
    tools: Dict[str, Union[Callable, ToolSpec]],
    *,
    loop_id: Optional[str] = None,
    lineage: Optional[list[str]] = None,
    interject_queue: asyncio.Queue[dict | str],
    cancel_event: asyncio.Event,
    stop_event: asyncio.Event | None = None,
    log_steps: Union[bool, str] = True,
    max_steps: Optional[int] = None,
    timeout: Optional[int] = None,
    preprocess_msgs: Optional[Callable[[list[dict]], list[dict]]] = None,
    outer_handle_container: Optional[list] = None,
    response_format: Optional[Any] = None,
    persist: bool = False,
    prompt_caching: Optional["PromptCacheParam"] = None,
    runtime_state: Optional[ToolLoopRuntimeState] = None,
    first_message_context: Optional[str] = None,
    on_turn_boundary: Optional[Callable[[], Awaitable[Optional[str]]]] = None,
    tools_in_code: bool = False,
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
    request, keeping the session. Exceptions inside tools are serialised and
    shown to the model; :data:`MAX_CONSECUTIVE_FAILURES` back-to-back crashes
    abort the loop with ``RuntimeError``. Duplicate tool calls in one turn
    (same name and arguments) are dropped before they run.

    Parameters
    ----------
    client : ``unillm.AsyncUnify``
        Pre-initialised client providing ``append_messages`` and ``generate``;
        every token sent to or received from the LLM flows through it.

    message : ``str | dict | list[dict]``
        The first user prompt: text, a user message, or a list of content
        blocks (one user message).

    tools : ``dict[str, Callable]``
        ``name → function`` for every callable the LLM may invoke. Each must be
        fully type-hinted with a concise docstring; both are converted to a
        tool schema via :pyfunc:`method_to_schema`.

    interject_queue : ``asyncio.Queue[str | dict]``
        The requester's next messages (``{"message": str}`` or a string), read
        only at a turn boundary, plus a persistent loop's request cancel
        (``_cancel_request``).

    cancel_event : ``asyncio.Event``
        Set by the outer caller to stop the loop: the model call or tool call
        in flight is cancelled and ``asyncio.CancelledError`` propagates.

    log_steps : ``bool | str``, default ``True``
        Step logging to ``LOGGER``: ``False`` for none, ``True`` for everything
        except system messages, ``"full"`` for everything.

    max_steps : ``int | None``
        The step limit, a count of messages; ``None`` reads
        ``UNIFY_MAX_TOOL_LOOP_STEPS``. In a persistent loop it counts the
        messages of the current request, and reaching it ends that request
        with a reply that quotes the request's latest draft; any other loop
        ends with that reply.

    timeout : ``int | None``, default ``None``
        Activity-based timeout in seconds; the timer resets after each LLM
        response and each appended message. A tool call that runs past it is
        cancelled. ``None`` disables it.

    persist : ``bool``, default ``False``
        If ``True``, content without tool calls does not end the loop: the
        turn's response goes to the handle and the loop parks until the next
        message arrives on ``interject_queue``, so one loop serves many
        requests. The loop then ends only via ``cancel_event``.

    first_message_context : ``str | None``
        Text that opens the first user message of this loop, separated from it
        by a rule. ``None`` sends the message as given.

    on_turn_boundary : optional coroutine function, default ``None``
        Called just before each model call, after every tool result is
        appended. A non-empty string it returns is appended as one
        loop-authored user message. Never called while a model call or a tool
        runs, nor after a turn has ended (the agent record).

    tools_in_code : ``bool``, default ``False``
        ``True`` for the actor's and its teammates' loop, which answers a
        requester: its only model-API tool is ``execute_code``
        (``tool_registry.json_tools``); each requester message is counted for
        the request metadata headers; the loop compacts by itself when the
        prompt reaches the compaction point, before the next model call; and a
        cell's ``tools.answer(answer)`` gives the turn's answer, also for a
        request with a response format. Any other loop (the context
        compactor's) has the JSON tools it is given and never compacts.

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

    _token = TOOL_LOOP_LINEAGE.set(cfg.lineage)

    stop_event = stop_event or asyncio.Event()

    # Normalize response_format once. Unsupported values disable structured
    # mode rather than ask for an answer shape nothing can check.
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
    # with the final JSON shape in mind. The hint goes into a separate system
    # message appended below, never into the caller's original.
    _response_format_hint: str | None = None
    if _rf_norm is not None:
        _response_format_hint = (
            "## Response Format\n"
            "NOTE: After completing all tool calls, submit your final answer "
            "with `tools.answer(answer)` in a cell, as JSON that conforms to "
            "the following schema. Do NOT include any extra keys or "
            "commentary.\n" + json.dumps(_rf_norm.answer_json_schema, indent=2)
        )

    runtime_state = runtime_state or ToolLoopRuntimeState()
    fixed_tools_schema: Optional[list] = None
    _reply_token = None
    _reply_slot: Optional[_cell_reply.ReplySlot] = None
    if tools_in_code:
        from unify.common import tool_registry as _tool_registry

        # The request's one model-API tool, the same bytes for every agent.
        fixed_tools_schema = _tool_registry.json_tools("actor")
        # This loop's slot for the turn's answer, which a cell's tools.answer fills.
        _reply_token = _cell_reply.bind()
        _reply_slot = _cell_reply.current()

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
        client=client,
        message_count_offset=runtime_state.message_count_offset,
    )
    _msg_dispatcher = LoopMessageDispatcher(client, cfg, timer)

    if log_steps:
        if log_steps == "full":
            logger.info(
                f"System Message: {client.system_message}",
                prefix=ICONS["system_message"],
            )
        # A list of content blocks is not logged whole.
        if not isinstance(message, list):
            logger.info(f"Request: {message}", prefix=ICONS["request"])

    import time as _setup_time

    _setup_t0 = _setup_time.perf_counter()

    def _setup_elapsed() -> str:
        return f"{(_setup_time.perf_counter() - _setup_t0) * 1000:.0f}ms"

    # ── Response format: its own system message ──────────────────────────
    # ``_runtime_context=True`` identifies it later; ``_ctx_header=True`` marks
    # it as the loop's own header.
    if _response_format_hint:
        await _msg_dispatcher.append_msgs(
            [
                {
                    "role": "system",
                    "_runtime_context": True,
                    "_ctx_header": True,
                    "content": _response_format_hint,
                },
            ],
        )

    # ToolsData must exist before the preflight backfill below can schedule.
    logger.debug(f"[setup +{_setup_elapsed()}] initialising ToolsData")
    tools_data: ToolsData = ToolsData(
        tools,
        client=client,
        logger=logger,
        call_counts=runtime_state.call_counts,
    )
    logger.debug(
        f"[setup +{_setup_elapsed()}] ToolsData ready ({len(tools_data.normalized)} tools)",
    )

    consecutive_failures = _LoopToolFailureTracker(
        MAX_CONSECUTIVE_FAILURES,
        runtime_state,
    )
    assistant_meta: Dict[int, Dict[str, Any]] = {}

    _max_input_tokens = unillm.get_max_input_tokens(client.endpoint)
    _over_threshold = False
    _full_completion: Any = None

    # ── The first user message ───────────────────────────────────────────
    if isinstance(message, dict):
        initial_user_msg = message
    elif isinstance(message, list):
        # A list of content blocks is one user message.
        initial_user_msg = {"role": "user", "content": message}
    else:
        initial_user_msg = {"role": "user", "content": message}
    if first_message_context:
        initial_user_msg = with_first_message_context(
            initial_user_msg,
            first_message_context,
        )
    await _msg_dispatcher.append_msgs([initial_user_msg])

    async def _handle_limit_reached(reason: str, draft: Optional[str] = None) -> str:
        """
        Terminate gracefully when *timeout* or *max_steps* is exceeded: append
        a short assistant notice, followed by *draft* when one is given. Every
        call has been answered by then (``_interrupt_turn`` answers the ones a
        limit interrupts).
        """
        await tools_data.cancel_pending_tasks(grace=_CANCEL_GRACE_S)
        await _answer_calls_not_run(reason)

        notice = {
            "role": "assistant",
            "content": f"🔚 Terminating early: {reason}"
            + (f"\n\nBest current answer:\n{draft}" if draft is not None else ""),
        }
        await _msg_dispatcher.append_msgs([notice])
        if log_steps:
            logger.info(f"Early exit – {reason}", prefix=ICONS["early_exit"])
        return notice["content"]

    async def _answer_calls_not_run(reason: str) -> None:
        """Answer each call the limit came before (one a seeded or resumed
        transcript carried, or one a model call that ran into the timeout
        made), so the transcript leaves no call unanswered."""
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

    async def _end_request_at_step_limit() -> None:
        """A persistent loop's request ends at max_steps.

        The pending calls are cancelled and answered as such, the reply says
        the request stopped at the step limit and quotes its latest draft,
        and the loop parks for the next request, whose steps are counted
        from its own message. The reply also stays in the transcript, so
        the model reads on the next request where the last one stopped.
        """
        reason = f"max_steps ({max_steps}) exceeded"
        draft = _request_draft()
        await tools_data.cancel_pending_tasks_with_reply(
            f"Cancelled: the step limit ({reason}) ended the request before "
            "this call finished.",
            assistant_meta=assistant_meta,
            msg_dispatcher=_msg_dispatcher,
        )
        content = (
            f"🔚 Stopped at the step limit: {reason}, so this request ended "
            "before it was finished. The session is still open: the next "
            "message starts a new request."
        )
        if draft is not None:
            content += f"\n\nBest current answer:\n{draft}"
        else:
            content += "\n\nNo reply text was drafted for this request."
        await _msg_dispatcher.append_msgs(
            [{"role": "assistant", "content": content}],
        )
        if log_steps:
            logger.info(
                f"Step limit – {reason}; waiting for the next request",
                prefix=ICONS["early_exit"],
            )
        _outer = outer_handle_container[0] if outer_handle_container else None
        if _outer is not None and hasattr(_outer, "_notification_q"):
            await _outer._notification_q.put({"type": "response", "content": content})
        await _park_until_next_request()
        timer.reset()
        timer.start_request()

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
        timer.reset()
        timer.start_request()

    async def _park_until_next_request() -> None:
        """Park a persistent loop until its next request arrives.

        Returns when an interjection that starts a request is back on the
        queue, or when the loop is stopped; the caller then resumes at the
        top of the loop, which processes either.
        """
        _outer = outer_handle_container[0] if outer_handle_container else None
        # The next request starts with no answer.
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

            # The next request goes back to the head of the queue, ahead of
            # anything sent after it, so the drain takes them in order.
            _requeue_at_front(interject_queue, interjection)
            logger.info(
                "Persist mode: interjection received, resuming loop",
                prefix=ICONS["resume"],
            )
            break

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

        While it runs, only a stop (``cancel_event``), the loop's timeout
        and, in a persistent loop, the requester's cancel of the request are
        watched; a message for the session stays queued for the next
        boundary.
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
        while not task.done():
            if timer.has_exceeded_time():
                return "timeout"
            watchers: Dict[str, asyncio.Task] = {
                "cancel": asyncio.create_task(
                    cancel_event.wait(),
                    name="CancelEventWait",
                ),
            }
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
                    _unknown_tool_text(name),
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
                if amsg.get("tool_calls"):
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
        """Whether a message for the session (not a request cancel) is queued."""
        return any(
            not (isinstance(item, dict) and "_cancel_request" in item)
            for item in list(getattr(interject_queue, "_queue", None) or ())
        )

    # Consecutive replies of this turn dropped for a provider error, and
    # those the provider withheld (content_filter).
    _provider_error_retries = 0
    _content_filter_retries = 0
    # Bounded retries for a terminal turn that returns empty content with no
    # substantive answer anywhere else in the conversation to fall back on.
    _empty_final_answer_retries = 0
    _MAX_EMPTY_FINAL_ANSWER_RETRIES = 1

    logger.debug(f"[setup +{_setup_elapsed()}] entering main loop")

    try:
        while True:
            if timer.has_exceeded_time():
                return await _handle_limit_reached(f"timeout ({timeout}s) exceeded")

            if timer.has_exceeded_msgs():
                if persist:
                    await _end_request_at_step_limit()
                    continue
                return await _handle_limit_reached(
                    f"max_steps ({max_steps}) exceeded",
                    _request_draft(),
                )

            # Assistant tool_calls missing replies (a resumed transcript) are
            # run, in call order, before anything queued is appended. Each
            # assistant message is repaired once.
            _repair_status = await _repair_unreplied(
                find_unreplied_assistant_entries(client),
            )
            if _repair_status == "timeout":
                return await _handle_limit_reached(f"timeout ({timeout}s) exceeded")

            # UNIFY_MEMORY_IN_EPISODE=on: every tool call is answered here, so the pauses marked during the turn (a
            # checkpoint the actor called, an in-episode signal) start their segment fork now
            await _memory_pause(client)

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
                    # One more requester message, in the loop that answers a
                    # requester (the request metadata headers count them).
                    if tools_in_code:
                        count_requester_message(client)
                    # UNIFY_MEMORY_IN_EPISODE=on: a new message is a pause; its segment fork runs before the message
                    # joins the conversation
                    await _memory_pause(client, new_message=True)
                    await _msg_dispatcher.append_msgs(
                        [{"role": "user", "_interjection": True, "content": _msg_text}],
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
                    continue
                logger.info(
                    "Cancel ignored: the loop is not persistent (stop() ends it)",
                    prefix=ICONS["pause"],
                )

            # The actor's loop: over the compaction point it compacts now,
            # before the next model call, with no forced turn: the handle
            # summarises (the last request and what followed it) and restarts.
            if tools_in_code and _over_threshold:
                # UNIFY_MEMORY_IN_EPISODE=on: a compaction is a pause; its segment fork runs before it
                await _memory_pause(client, compacting=True)
                tail_from = runtime_state.in_code_tail_from
                runtime_state.in_code_tail = (
                    copy.deepcopy(list(client.messages or [])[tail_from:])
                    if tail_from is not None
                    else None
                )
                runtime_state.in_code_tail_from = None
                logger.info(
                    "Context over the compaction point: compacting before the "
                    "next model call",
                    prefix=ICONS["summarize"],
                )
                return _COMPRESSION_SIGNAL

            # ── C. This turn's toolkit ───────────────────────────────────
            # The session's tool list is fixed at its first call (the actor's
            # is ``execute_code`` alone), so the requests keep one cached prefix.
            if runtime_state.session_tools_schema is None:
                runtime_state.session_tools_schema = (
                    copy.deepcopy(list(fixed_tools_schema))
                    if fixed_tools_schema is not None
                    else [
                        method_to_schema(tools_data.normalized[name].fn, name)
                        for name in sorted(tools_data.normalized)
                    ]
                )
            tmp_tools = runtime_state.session_tools_schema
            # Set once, before the first request: the key names the
            # prefix (model, system prompt, this fixed list) unless a
            # key is already set, as a fork's is.
            _cache_discipline.ensure_cache_affinity(client, tmp_tools)
            _session_tool_names = frozenset(
                _cache_discipline.schema_names(tmp_tools),
            )

            # ── D. Ask the LLM what to do next ───────────────────────────
            # A stop that landed while this turn was being built ends the
            # loop rather than dispatching a turn only to cancel it.
            if cancel_event.is_set():
                raise asyncio.CancelledError

            # A cell's tools.answer gave the turn's answer: it is this step's
            # assistant message and no model is asked.
            _cell_reply_msg = _reply_slot.take() if _reply_slot is not None else None

            # The agent record: what is new in the shared record for this
            # agent, appended after this turn's tool results. Reached only when
            # a model call is about to be made, so never after a turn has ended
            # (a cell's answer above ends it with no model call).
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
                transcripts.observe(
                    {"role": "assistant", "_thinking_in_flight": True},
                    cfg,
                )

            if _cell_reply_msg is not None:
                # The answer goes in as the model's text reply would, and the
                # step goes on from here exactly as for one (section F).
                client.append_messages([_cell_reply_msg])
                _full_completion = None
                logger.info(
                    "A cell's answer ends the turn; no model call",
                    prefix=ICONS["completed"],
                )
            else:
                # The model call runs to completion; only a stop, or in a
                # persistent loop the requester's cancel of the request,
                # cancels it.
                _gen_kwargs = {
                    "return_full_completion": True,
                    "tools": tmp_tools,
                    "tool_choice": "auto",
                    "stateful": True,
                    "prompt_caching": prompt_caching,
                }

                llm_task = asyncio.create_task(
                    generate_with_preprocess(
                        client,
                        preprocess_msgs,
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

            # A reply the provider withheld (finish_reason "content_filter"
            # with no text and no tool call: an HTTP 200 with nothing charged)
            # is not the model's turn either: it is sent again in the same
            # way, counted apart. Once the retries are spent the turn ends
            # with a text that says so, never an empty answer.
            if _completion_withheld(_full_completion):
                _failed = (client.messages or [None])[-1]
                _owned = (
                    isinstance(_failed, dict)
                    and _failed.get("role") == "assistant"
                    and is_mutable(client, _failed)
                )
                if _owned and _content_filter_retries < _PROVIDER_ERROR_RETRIES:
                    _content_filter_retries += 1
                    client.messages.pop()
                    _backoff = _PROVIDER_ERROR_BACKOFF_S * 2 ** (
                        _content_filter_retries - 1
                    )
                    logger.error(
                        "The provider withheld the model's reply (content_filter); "
                        f"sending the turn again in {_backoff:g}s (retry "
                        f"{_content_filter_retries}/{_PROVIDER_ERROR_RETRIES})",
                        prefix=ICONS["early_exit"],
                    )
                    with suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(cancel_event.wait(), _backoff)
                    continue
                if _owned:
                    _failed["content"] = WITHHELD_REPLY.format(
                        attempts=_content_filter_retries + 1,
                    )
                    logger.error(
                        "The provider withheld the model's reply (content_filter); "
                        "retries spent, the turn ends with a text saying so",
                        prefix=ICONS["early_exit"],
                    )
            _content_filter_retries = 0

            # The step's assistant message is the tail.
            msg = client.messages[-1]
            transcripts.observe(msg, cfg)

            # The actor's loop: the prompt's size decides whether it compacts
            # before its next model call.
            if tools_in_code:
                with suppress(Exception):
                    _usage = getattr(_full_completion, "usage", None)
                    # the window may be unknown: the fixed compaction point then applies alone
                    if _usage and getattr(_usage, "prompt_tokens", None):
                        _over_threshold = context_over_threshold(
                            _usage.prompt_tokens,
                            COMPACTION_THRESHOLD,
                            _max_input_tokens,
                        )
                        # the reply just appended is the first message the
                        # last request did not carry
                        runtime_state.in_code_tail_from = len(client.messages or []) - 1

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
                return await _handle_limit_reached(f"timeout ({timeout}s) exceeded")

            runtime_state.step_index += 1

            # ── E. Run the turn's tool calls ─────────────────────────────
            # Each call's arguments are JSON-parsed once here, and every call
            # runs to completion before the next one starts, so the results
            # land in call order.

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

                    # A tool the session's fixed list does not hold is refused
                    # here, by rule.
                    if name not in _session_tool_names:
                        await _answer_call(
                            msg,
                            call,
                            _cache_discipline.masked_tool_refusal(
                                name,
                                advertised=_session_tool_names,
                            ),
                        )
                        continue

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
                            json.loads(_raw_args)
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

                    # A tool the loop does not run gets an error tool response
                    # so the transcript stays valid; an unresolved tool_call
                    # would make later LLM calls fail.
                    if name not in tools_data.normalized:
                        await _answer_call(
                            msg,
                            call,
                            _unknown_tool_text(name),
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

                if _interrupted == "timeout":
                    return await _handle_limit_reached(f"timeout ({timeout}s) exceeded")
                # Every call has its result (a cancelled request is ended at
                # the next boundary): back to the very top.
                continue

            # ── F. No new tool calls ─────────────────────────────────────
            # A plain assistant message: nothing is running. A message the
            # requester sent during the model call is part of the request
            # in a loop that ends here, so the model gets a turn for it; a
            # persistent loop answers this request first and takes the
            # message as its next one when it parks.
            if not persist and _queued_message():
                continue

            if timer.has_exceeded_time():
                return await _handle_limit_reached(f"timeout ({timeout}s) exceeded")

            if timer.has_exceeded_msgs():
                if persist:
                    await _end_request_at_step_limit()
                    continue
                return await _handle_limit_reached(
                    f"max_steps ({max_steps}) exceeded",
                    _request_draft(),
                )

            final_content = extract_substantive_text(msg["content"])

            # An empty/whitespace-only terminal turn must never override a
            # substantive answer already in the transcript: a model with
            # nothing left to add can still return empty content on a later
            # turn. The persist response and the plain return both read
            # final_content after this point, so it is resolved once here.
            # Persist mode is exempt because it never finalizes here: an
            # empty turn surfaces an empty response and re-enters the
            # persist wait, so the nudge/loud-fail below would inject
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
                # The turn-complete response reaches the outer handle. It
                # marks the wait state, so it is sent even when the turn
                # produced no text: a model with nothing to say after a
                # message that asks for nothing returns empty content, and a
                # driver waiting for the response (`unify act --jsonl`) would
                # otherwise wait forever.
                _outer = outer_handle_container[0] if outer_handle_container else None
                if _outer is not None and hasattr(_outer, "_notification_q"):
                    await _outer._notification_q.put(
                        {"type": "response", "content": final_content or ""},
                    )

                await _park_until_next_request()
                timer.reset()
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
        # A loop stopped before its first LLM step still logs its stop.
        if log_steps:
            logger.flush_deferred()
        with suppress(Exception):
            TOOL_LOOP_LINEAGE.reset(_token)
