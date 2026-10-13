import asyncio
import collections
import contextlib
import json
import traceback
from datetime import datetime
from typing import Any, Optional

from unify.logger import LOGGER
from unify.common.hierarchical_logger import DEFAULT_ICON
from unify.common.startup_timing import log_startup_timing
from unify.common.diagnostic_logging import staging_diagnostics_enabled
from unify.manager_registry import SingletonABCMeta
from unify.common.async_tool_loop import SteerableToolHandle
from unify.common.hierarchical_logger import SessionLogger
from unify.legacy.conversation_manager.domains.chat_history import ChatHistory
from unify.legacy.conversation_manager.domains.brain import build_brain_spec
from unify.legacy.conversation_manager.domains.brain_action_tools import (
    ConversationManagerBrainActionTools,
)
from unify.legacy.conversation_manager.domains.brain_tools import (
    ConversationManagerBrainTools,
)
from unify.legacy.conversation_manager.domains.event_handlers import EventHandler
from unify.legacy.conversation_manager.domains.renderer import Renderer
from unify.legacy.conversation_manager.events import Event, OpenSlowBrainTurn
from unify.common.prompt_helpers import now as prompt_now

from unify.common.llm_client import new_slow_brain_llm_client
from unify.common.single_shot import ToolExecution, single_shot_tool_decision
from unify.events.manager_event_logging import _EVENT_SOURCE
from unify.legacy.conversation_manager.domains.notifications import NotificationBar
from unify.legacy.conversation_manager.domains.utils import Debouncer

from unify.actor.base import BaseActor

RECENT_TOOL_EXECUTIONS_LIMIT = 20
RECENT_TOOL_PREVIEW_CHARS = 500
# Upper bound a slow-brain turn holds for boot hydration before rendering
# anyway. Hydration is one chat-table read (seconds); a hold that outlives
# this bound means hydration is stuck, and an eager reply from the
# pre-hydration view beats indefinite silence.
BOOT_HYDRATION_MAX_WAIT_SECONDS = 30.0
SLOW_BRAIN_FAILURE_REPLY_THROTTLE_SECONDS = 600
SLOW_BRAIN_FAILURE_RESPONSE = (
    "I hit a technical problem and couldn't respond just now. Please try "
    "again in a moment."
)
# Self-scheduled wait(delay) polling backoff. Timer wakes are full-priced
# slow-brain turns; a model that busy-polls a long-running act ("check
# again in 10 seconds", repeatedly) burns tokens with zero information
# gain — external events wake the brain immediately regardless, so only
# the self-scheduled timer needs a floor. A small budget of fast polls
# per window stays free; beyond it the effective delay doubles per extra
# poll up to the cap.
WAIT_POLL_WINDOW_SECONDS = 600.0
WAIT_POLL_FREE_BUDGET = 5
WAIT_POLL_MIN_CLAMPED_DELAY_SECONDS = 60
WAIT_POLL_MAX_CLAMPED_DELAY_SECONDS = 600

COMMISSIONING_MUTATION_TOOL_NAMES = frozenset(
    {
        "act",
    },
)
COMMISSIONING_OUTBOUND_FOLLOWUP_EVENTS = frozenset(
    {
        "UnifyMessageSent",
    },
)


ACT_FOLLOWUP_ARGUMENT_DEFAULTS: dict[str, Any] = {
    "response_format": None,
    "persist": False,
    "include_conversation_context": True,
}


def _log_slow_brain_single_shot_failure(
    *,
    run_id: str,
    request_id: str,
    origin_event_name: str,
    message_count: int,
    tool_count: int,
    state_chars: int,
) -> None:
    if not staging_diagnostics_enabled():
        return
    LOGGER.exception(
        (
            "Slow-brain single-shot failed "
            "run_id=%s request_id=%s origin_event=%s "
            "message_count=%d tool_count=%d state_chars=%d"
        ),
        run_id,
        request_id or "-",
        origin_event_name or "-",
        message_count,
        tool_count,
        state_chars,
    )
    LOGGER.error(
        "Slow-brain single-shot traceback text:\n%s",
        traceback.format_exc(),
    )


def _format_tool_thoughts_for_log(tools: list[ToolExecution]) -> str:
    parts: list[str] = []
    for tool_exec in tools:
        thoughts = getattr(tool_exec, "thoughts", None)
        if isinstance(thoughts, str) and thoughts.strip():
            parts.append(f"[{tool_exec.name}] {thoughts.strip()}")
    return " | ".join(parts)


class ConversationManager(metaclass=SingletonABCMeta):
    def __init__(
        self,
        event_broker,
        stop: asyncio.Event,
    ):
        # initialization state
        self.initialized: bool = False
        # Open ⇒ slow-brain turns may render. ``init_conv_manager`` closes it
        # for the window between boot and global-thread hydration; hydration
        # completion (restored, empty, or failed) reopens it. Steady-state
        # turns therefore never wait on it. See ``_run_llm`` for why a turn
        # must not render from a pre-hydration view.
        self._hydration_gate: asyncio.Event = asyncio.Event()
        self._hydration_gate.set()
        self.ready_for_brain: bool = True
        self.loop = asyncio.get_event_loop()

        # shutdown
        self.shutdown_reason: str | None = None
        self.stop = stop

        self.event_broker = event_broker

        # the actor that carries out actions
        self.actor: BaseActor | None = None

        self.debouncer = Debouncer(name="ConversationManager")

        # renderer
        self.prompt_renderer = Renderer()

        # state
        # The conversation with the user, persisted to the chat table.
        self.chat_history = ChatHistory()
        # The brain's own LLM message list: one state snapshot and one
        # assistant reply per turn, in memory only.
        self.brain_messages: list[dict] = []
        self.notifications_bar = NotificationBar()
        self.in_flight_actions: dict[
            int,
            dict,
        ] = (
            {}
        )  # dict[int, {"handle": SteerableTool, "query": str, "calling_id": str|None, ...}]
        self.completed_actions: dict[
            int,
            dict,
        ] = {}  # Finished actions, kept for post-completion ask() queries
        self._pending_steering_tasks: set[asyncio.Task] = (
            set()
        )  # Background tasks from async steering ops (e.g., ask_*)
        self.last_snapshot = prompt_now(as_string=False)
        self._current_snapshot = None
        self._current_state_snapshot = (
            None  # Fresh rendered state for tools during _run_llm
        )
        self._current_snapshot_state = (
            None  # SnapshotState with element tracking for incremental diff computation
        )

        # ask handles (for Actor actions)
        self.active_ask_handle: Optional["SteerableToolHandle"] = None

        # LLM run requests recorded during event handling (production path).
        # In step() mode, requests are recorded via a contextvar instead.
        self._pending_llm_requests: list[tuple[float, bool]] = []
        self._pending_llm_request_meta: list[dict[str, Any]] = []
        self._current_event_trace: dict[str, str] | None = None
        self._event_trace_seq: int = 0
        self._llm_request_seq: int = 0
        self._llm_run_seq: int = 0
        self._llm_gen: int = 0
        self._active_llm_trace_meta: dict[str, Any] | None = None
        self._recent_tool_executions: list[dict[str, Any]] = []
        self._recent_commissioning_successes: dict[str, int] = {}

        # Hierarchical session logger for consistent nested logging
        self._session_logger = SessionLogger("ConversationManager")
        self._session_logger.debug(
            "session_start",
            "ConversationManager session initialized",
        )

    def snapshot(self):
        self._current_snapshot = prompt_now(as_string=False)
        # Track how many notifications were present at snapshot time.
        # Any notifications appended while the LLM is running (e.g., an action that
        # completes very quickly) must remain visible for at least the NEXT LLM run.
        # Otherwise, `commit()` would immediately drop them and the LLM would never
        # see the result, which can cause repeated duplicate actions.
        self._snapshot_notif_count = len(self.notifications_bar.notifications)
        return self._current_snapshot

    def commit(self):
        self.last_snapshot = self._current_snapshot
        notifs = self.notifications_bar.notifications
        snap_n = int(getattr(self, "_snapshot_notif_count", 0) or 0)
        # Keep:
        # - pinned notifications
        # - notifications that were appended AFTER the last snapshot was taken
        #   (these arrived during the LLM run and must be shown next turn)
        self.notifications_bar.notifications = [
            n for i, n in enumerate(notifs) if n.pinned or i >= snap_n
        ]

    @staticmethod
    def _tool_result_is_error(result: Any) -> bool:
        return isinstance(result, dict) and "error_kind" in result

    @staticmethod
    def _preview_value(
        value: Any,
        *,
        max_chars: int = RECENT_TOOL_PREVIEW_CHARS,
    ) -> str:
        try:
            rendered = json.dumps(value, sort_keys=True, default=str)
        except Exception:
            rendered = repr(value)
        if len(rendered) <= max_chars:
            return rendered
        return rendered[: max_chars - 3] + "..."

    @staticmethod
    def _normalize_followup_tool_args(
        tool_name: str,
        tool_args: dict[str, Any] | None,
    ) -> dict[str, Any]:
        normalized = dict(tool_args or {})
        if tool_name == "act":
            for key, default_value in ACT_FOLLOWUP_ARGUMENT_DEFAULTS.items():
                normalized.setdefault(key, default_value)
        return normalized

    @classmethod
    def _commissioning_tool_fingerprint(
        cls,
        tool_name: str,
        tool_args: dict[str, Any] | None,
    ) -> str:
        stable_args = json.dumps(
            cls._normalize_followup_tool_args(tool_name, tool_args),
            sort_keys=True,
            default=str,
        )
        return f"{tool_name}:{stable_args}"

    def _is_immediate_commissioning_followup(self, origin_event_name: str) -> bool:
        return origin_event_name in COMMISSIONING_OUTBOUND_FOLLOWUP_EVENTS

    def suppress_duplicate_commissioning_tool(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Suppress immediate duplicate commissioning calls from outbound follow-ups."""
        if tool_name not in COMMISSIONING_MUTATION_TOOL_NAMES:
            return None
        trace_meta = self._active_llm_trace_meta or {}
        origin_event_name = str(trace_meta.get("origin_event_name") or "")
        if not self._is_immediate_commissioning_followup(origin_event_name):
            return None
        fingerprint = self._commissioning_tool_fingerprint(tool_name, tool_args)
        last_success_gen = self._recent_commissioning_successes.get(fingerprint)
        if last_success_gen != self._llm_gen - 1:
            return None
        return {
            "error_kind": "duplicate_suppressed",
            "message": (
                "Skipped duplicate commissioning tool call from immediate outbound "
                "follow-up event."
            ),
            "details": {
                "tool_name": tool_name,
                "origin_event_name": origin_event_name,
            },
        }

    def _record_recent_tool_executions(
        self,
        *,
        tools: list[Any],
        trace_meta: dict[str, Any],
    ) -> None:
        origin_event_name = str(trace_meta.get("origin_event_name") or "")
        for tool_exec in tools:
            tool_name = str(getattr(tool_exec, "name", ""))
            tool_args = getattr(tool_exec, "args", {}) or {}
            tool_result = getattr(tool_exec, "result", None)
            self._recent_tool_executions.append(
                {
                    "generation": self._llm_gen,
                    "origin_event_name": origin_event_name,
                    "tool_name": tool_name,
                    "args_preview": self._preview_value(tool_args),
                    "result_preview": self._preview_value(tool_result),
                },
            )
            if (
                tool_name in COMMISSIONING_MUTATION_TOOL_NAMES
                and not self._tool_result_is_error(tool_result)
            ):
                fingerprint = self._commissioning_tool_fingerprint(tool_name, tool_args)
                self._recent_commissioning_successes[fingerprint] = self._llm_gen
        if len(self._recent_tool_executions) > RECENT_TOOL_EXECUTIONS_LIMIT:
            self._recent_tool_executions = self._recent_tool_executions[
                -RECENT_TOOL_EXECUTIONS_LIMIT:
            ]
        for fingerprint, generation in list(
            self._recent_commissioning_successes.items(),
        ):
            if generation < self._llm_gen - 1:
                del self._recent_commissioning_successes[fingerprint]

    def get_recent_transcript(
        self,
        max_messages: int | None = None,
    ) -> tuple[list[dict], datetime | None]:
        """The tail of the conversation as role/content turns.

        Args:
            max_messages: Maximum number of messages to return. None for all.

        Returns:
            A tuple of (conversation_turns, last_message_timestamp) where:
            - conversation_turns: List of {"role": "user"|"assistant", "content": str}
            - last_message_timestamp: Timestamp of the last message, or None
        """
        messages = self.chat_history.recent(max_messages)
        conversation_turns = [
            {"role": message.role, "content": message.content.strip()}
            for message in messages
        ]
        last_message_timestamp = messages[-1].timestamp if messages else None
        return conversation_turns, last_message_timestamp

    def _preprocess_messages(
        self,
        messages: str | dict | list,
    ) -> str | dict | list:
        """Keep only the latest state snapshot from message history.

        ConversationManager renders a full state snapshot each turn. We keep only the
        latest snapshot when calling the model, while preserving any system messages
        and user interjections.
        """
        if isinstance(messages, str):
            return messages
        if isinstance(messages, dict):
            return messages
        if not isinstance(messages, list):
            return messages

        try:
            # Find all state snapshot messages
            state_indices = [
                i
                for i, m in enumerate(messages)
                if isinstance(m, dict) and m.get("_cm_state_snapshot") is True
            ]
            if not state_indices:
                return messages

            # Keep only the latest state snapshot and non-state messages
            last_state = messages[state_indices[-1]]
            kept: list[dict] = []
            for m in messages:
                if not isinstance(m, dict):
                    continue
                role = m.get("role")
                if role == "system":
                    kept.append(m)
                elif role == "user" and not m.get("_cm_state_snapshot"):
                    kept.append(m)

            kept.append(last_state)
            return kept
        except Exception:
            return messages

    async def run_llm(
        self,
        delay: float = 0,
        trace_meta: dict[str, str] | None = None,
    ):
        await self.debouncer.submit(
            self._run_llm_with_failure_notification,
            kwargs={"trace_meta": trace_meta or {}},
            delay=delay,
            label=(trace_meta or {}).get("origin_event_name", ""),
            trace_meta=trace_meta,
        )

    async def _run_llm_with_failure_notification(
        self,
        trace_meta: dict[str, str] | None = None,
    ) -> list[str] | None:
        """Wrap ``_run_llm`` so a hard failure reaches the user.

        A failed slow-brain turn would otherwise only produce a
        ``log_task_exc`` line in the logs, leaving the user in silence and
        re-sending the same message. This wrapper sends a throttled apology
        to the chat, then re-raises so the failure log is preserved.
        """
        try:
            return await self._run_llm(trace_meta=trace_meta)
        except asyncio.CancelledError:
            raise
        except Exception:
            with contextlib.suppress(Exception):
                await self._send_slow_brain_failure_reply()
            raise

    async def _send_slow_brain_failure_reply(self) -> None:
        """Tell the user their message hit a hard failure (throttled).

        Only a conversation the user has spoken in gets the apology: a turn
        that failed with nothing inbound has nobody waiting on it.
        """
        if not any(m.role == "user" for m in self.chat_history.recent()):
            return
        now = self.loop.time()
        last_sent = getattr(self, "_slow_brain_failure_reply_sent_at", None)
        if (
            last_sent is not None
            and now - last_sent < SLOW_BRAIN_FAILURE_REPLY_THROTTLE_SECONDS
        ):
            return
        self._slow_brain_failure_reply_sent_at = now
        tools = ConversationManagerBrainActionTools(self)
        await tools.send_unify_message(content=SLOW_BRAIN_FAILURE_RESPONSE)

    def _clamp_wait_poll_delay(self, delay: int) -> int:
        """Apply escalating backoff to repeated self-scheduled wait polls.

        Keeps the first ``WAIT_POLL_FREE_BUDGET`` timer wakes per window at
        the model's requested cadence, then doubles the enforced minimum
        per extra poll (capped). External events are unaffected — they wake
        the brain immediately whether or not a timer is pending.
        """
        polls = getattr(self, "_wait_poll_times", None)
        if polls is None:
            polls = self._wait_poll_times = collections.deque(maxlen=64)
        now = self.loop.time()
        polls.append(now)
        recent = sum(1 for t in polls if now - t <= WAIT_POLL_WINDOW_SECONDS)
        excess = recent - WAIT_POLL_FREE_BUDGET
        if excess <= 0:
            return delay
        floor = min(
            WAIT_POLL_MIN_CLAMPED_DELAY_SECONDS * (2 ** (excess - 1)),
            WAIT_POLL_MAX_CLAMPED_DELAY_SECONDS,
        )
        if delay < floor:
            self._session_logger.info(
                "wait",
                f"Wait-poll backoff: raising delay {delay}s -> {floor}s "
                f"({recent} timer polls in the last "
                f"{int(WAIT_POLL_WINDOW_SECONDS)}s)",
            )
            return floor
        return delay

    async def request_llm_run(
        self,
        delay=0,
        is_user_origin: bool = False,
    ) -> str:
        """Request an LLM run.

        The request is recorded and later scheduled by the event loop after
        the current event is handled.
        """
        self._llm_request_seq += 1
        request_id = f"llmreq-{self._llm_request_seq:06d}"
        event_trace = self._current_event_trace or {}
        request_meta = {
            "request_id": request_id,
            "origin_event_id": event_trace.get("event_id", ""),
            "origin_event_name": event_trace.get("event_name", ""),
            "is_user_origin": is_user_origin,
        }
        self._pending_llm_requests.append((delay, is_user_origin))
        self._pending_llm_request_meta.append(request_meta)
        log_startup_timing(
            LOGGER,
            (
                "⏱️ [StartupTiming] first_reply.request_llm_run queued "
                "request_id=%s origin_event=%s delay=%s "
                "is_user_origin=%s pending=%d ready_for_brain=%s"
            ),
            request_id,
            request_meta["origin_event_name"] or "-",
            delay,
            is_user_origin,
            len(self._pending_llm_requests),
            self.ready_for_brain,
        )
        self._session_logger.debug(
            "llm_queue",
            (
                f"Queued slow-brain run request_id={request_id} "
                f"origin_event_id={request_meta['origin_event_id'] or '-'} "
                f"origin_event={request_meta['origin_event_name'] or '-'} "
                f"delay={delay} is_user_origin={is_user_origin}"
            ),
        )
        return request_id

    async def flush_llm_requests(self) -> None:
        """Schedule any pending LLM runs recorded during event handling."""
        if not self._pending_llm_requests:
            return
        if not self.ready_for_brain:
            return

        requests = self._pending_llm_requests
        metas = self._pending_llm_request_meta

        # Prefer the newest user-origin request; fall back to the newest overall.
        selected_idx = len(requests) - 1
        for i in range(len(requests) - 1, -1, -1):
            if requests[i][1]:  # is_user_origin
                selected_idx = i
                break

        dropped_requests = len(requests) - 1
        delay, is_user_origin = requests[selected_idx]
        selected_meta = dict(metas[selected_idx]) if metas else {}

        self._pending_llm_requests.clear()
        self._pending_llm_request_meta.clear()

        self._llm_run_seq += 1
        run_id = f"llmrun-{self._llm_run_seq:06d}"
        selected_meta["run_id"] = run_id
        selected_meta["dropped_requests"] = str(dropped_requests)
        log_startup_timing(
            LOGGER,
            (
                "⏱️ [StartupTiming] first_reply.flush_llm_requests dispatch "
                "run_id=%s request_id=%s origin_event=%s dropped=%d delay=%s "
                "is_user_origin=%s"
            ),
            run_id,
            selected_meta.get("request_id", "-"),
            selected_meta.get("origin_event_name", "-") or "-",
            dropped_requests,
            delay,
            is_user_origin,
        )

        self._session_logger.debug(
            "llm_thinking",
            (
                f"Dispatching slow-brain run_id={run_id} "
                f"request_id={selected_meta.get('request_id', '-')} "
                f"origin_event_id={selected_meta.get('origin_event_id', '-') or '-'} "
                f"origin_event={selected_meta.get('origin_event_name', '-') or '-'} "
                f"dropped_requests={dropped_requests} delay={delay} "
                f"is_user_origin={is_user_origin}"
            ),
        )
        log_startup_timing(
            LOGGER,
            (
                "⏱️ [StartupTiming] first_reply.run_llm_submitted "
                "run_id=%s request_id=%s origin_event=%s was_queued=%s"
            ),
            run_id,
            selected_meta.get("request_id", "-") or "-",
            selected_meta.get("origin_event_name", "-") or "-",
            self.debouncer.was_queued,
        )
        await self.run_llm(
            delay=delay,
            trace_meta=selected_meta,
        )

    async def _open_slow_brain_follow_on_turn(
        self,
        *,
        origin_run_id: str,
        previous_tools: list[str],
    ) -> None:
        """Schedule another slow-brain turn when the prior turn omitted wait."""
        if not self.ready_for_brain:
            return

        event = OpenSlowBrainTurn(
            origin_run_id=origin_run_id,
            previous_tools=list(previous_tools),
        )
        await EventHandler.handle_event(event, self)
        await self.flush_llm_requests()

    async def _run_llm(self, trace_meta: dict[str, str] | None = None) -> list[str]:
        """Run a single LLM decision and return all tool names that were called."""
        import time as _rl_time

        # Hold the turn while boot hydration is still landing. A reply
        # rendered from a pre-hydration view answers with confident ignorance
        # about a conversation whose history is seconds from appearing, and
        # the post-init follow-up turn cannot reliably repair a wrong first
        # answer already sent. Serving during init is otherwise unchanged:
        # the gate reopens the moment hydration resolves, well before manager
        # init finishes, so pre-init replies still happen — just never from
        # an empty view of a non-empty conversation.
        if not self._hydration_gate.is_set():
            _gate_t0 = _rl_time.perf_counter()
            try:
                await asyncio.wait_for(
                    self._hydration_gate.wait(),
                    timeout=BOOT_HYDRATION_MAX_WAIT_SECONDS,
                )
            except asyncio.TimeoutError:
                LOGGER.warning(
                    f"{DEFAULT_ICON} [ConversationManager] Boot hydration "
                    f"still pending after {BOOT_HYDRATION_MAX_WAIT_SECONDS:.0f}s "
                    "— rendering this turn without hydrated history",
                )
            log_startup_timing(
                LOGGER,
                "⏱️ [StartupTiming] first_reply.hydration_gate_wait duration=%.2fs",
                _rl_time.perf_counter() - _gate_t0,
            )

        _preamble_t0 = _rl_time.perf_counter()
        _last_preamble_step = _preamble_t0

        def _ms_since_start() -> str:
            return f"{(_rl_time.perf_counter() - _preamble_t0) * 1000:.0f}ms"

        def _mark_preamble_step() -> float:
            nonlocal _last_preamble_step
            now = _rl_time.perf_counter()
            elapsed_ms = (now - _last_preamble_step) * 1000
            _last_preamble_step = now
            return elapsed_ms

        trace_meta = trace_meta or {}

        self._llm_gen += 1
        run_id = trace_meta.get("run_id", "llmrun-unknown")
        request_id = trace_meta.get("request_id", "")
        origin_event_id = trace_meta.get("origin_event_id", "")
        origin_event_name = trace_meta.get("origin_event_name", "")
        self._session_logger.debug(
            "llm_thinking",
            (
                f"Slow-brain run started run_id={run_id} "
                f"request_id={request_id or '-'} "
                f"origin_event_id={origin_event_id or '-'} "
                f"origin_event={origin_event_name or '-'} "
                f"was_queued={self.debouncer.was_queued}"
            ),
        )
        _run_metadata_ms = _mark_preamble_step()

        self.snapshot()
        _snapshot_ms = _mark_preamble_step()

        log_startup_timing(
            LOGGER,
            (
                "⏱️ [StartupTiming] llm_preamble.setup.detail "
                "run_id=%s total=%.0fms metadata=%.0fms snapshot=%.0fms "
                "conversation=%d brain_messages=%d"
            ),
            run_id,
            (_rl_time.perf_counter() - _preamble_t0) * 1000,
            _run_metadata_ms,
            _snapshot_ms,
            len(self.chat_history.recent()),
            len(self.brain_messages),
        )

        _t0 = _rl_time.perf_counter()
        snapshot_state = self.prompt_renderer.render_state(
            self.chat_history,
            self.notifications_bar,
            self.in_flight_actions,
            self.completed_actions,
            self.last_snapshot,
            recent_tool_executions=self._recent_tool_executions,
        )
        _render_ms = (_rl_time.perf_counter() - _t0) * 1000

        _t0 = _rl_time.perf_counter()
        brain_spec = build_brain_spec(self, snapshot_state=snapshot_state)
        _brain_spec_ms = (_rl_time.perf_counter() - _t0) * 1000

        _t0 = _rl_time.perf_counter()
        input_message = brain_spec.state_message()
        _state_message_ms = (_rl_time.perf_counter() - _t0) * 1000
        system_prompt = brain_spec.system_prompt

        self._current_state_snapshot = input_message

        self._current_snapshot_state = snapshot_state

        reason = trace_meta.get("origin_event_name", "")
        self._session_logger.debug(
            "llm_thinking",
            f"LLM thinking... ({reason})" if reason else "LLM thinking...",
        )

        _t0 = _rl_time.perf_counter()
        _tools_step_t0 = _t0
        brain_tools = ConversationManagerBrainTools(self)
        _brain_tools_init_ms = (_rl_time.perf_counter() - _tools_step_t0) * 1000
        _tools_step_t0 = _rl_time.perf_counter()
        action_tools = ConversationManagerBrainActionTools(self)
        _action_tools_init_ms = (_rl_time.perf_counter() - _tools_step_t0) * 1000
        _tools_step_t0 = _rl_time.perf_counter()
        brain_tool_dict = brain_tools.as_tools()
        _brain_tools_ms = (_rl_time.perf_counter() - _tools_step_t0) * 1000
        _tools_step_t0 = _rl_time.perf_counter()
        action_tool_dict = action_tools.as_tools()
        _action_tools_ms = (_rl_time.perf_counter() - _tools_step_t0) * 1000
        _tools_step_t0 = _rl_time.perf_counter()
        steering_tool_dict = action_tools.build_action_steering_tools()
        _steering_tools_ms = (_rl_time.perf_counter() - _tools_step_t0) * 1000
        _tools_step_t0 = _rl_time.perf_counter()
        tools = {
            **brain_tool_dict,
            **action_tool_dict,
            **steering_tool_dict,
        }
        _tools_merge_ms = (_rl_time.perf_counter() - _tools_step_t0) * 1000
        _tools_ms = (_rl_time.perf_counter() - _t0) * 1000
        log_startup_timing(
            LOGGER,
            (
                "⏱️ [StartupTiming] llm_preamble.tools.detail "
                "run_id=%s total=%.0fms brain_init=%.0fms action_init=%.0fms "
                "brain_tools=%.0fms action_tools=%.0fms steering=%.0fms "
                "merge=%.0fms "
                "brain_tool_count=%d action_tool_count=%d steering_tool_count=%d "
                "total_tool_count=%d"
            ),
            run_id,
            _tools_ms,
            _brain_tools_init_ms,
            _action_tools_init_ms,
            _brain_tools_ms,
            _action_tools_ms,
            _steering_tools_ms,
            _tools_merge_ms,
            len(brain_tool_dict),
            len(action_tool_dict),
            len(steering_tool_dict),
            len(tools),
        )

        _t0 = _rl_time.perf_counter()
        _client_step_t0 = _t0
        client = new_slow_brain_llm_client(
            origin="ConversationManager",
            # Slow brain pins "high" explicitly so an assistant-level or
            # SLOW_BRAIN_REASONING_EFFORT override is the only thing that
            # can change it. When the assistant carries a default-model
            # effort override, it takes priority inside
            # new_slow_brain_llm_client().
            reasoning_effort="high",
        )
        _new_client_ms = (_rl_time.perf_counter() - _client_step_t0) * 1000
        _client_step_t0 = _rl_time.perf_counter()
        if hasattr(client, "_pending_thinking_log"):
            parts = [
                p
                for p in [reason, "from queue" if self.debouncer.was_queued else ""]
                if p
            ]
            suffix = f" ({', '.join(parts)})" if parts else ""
            client._pending_thinking_log.set_thinking_context(suffix)
        _thinking_context_ms = (_rl_time.perf_counter() - _client_step_t0) * 1000
        _client_step_t0 = _rl_time.perf_counter()
        client.set_system_message(system_prompt.to_list())
        _set_system_ms = (_rl_time.perf_counter() - _client_step_t0) * 1000
        _client_step_t0 = _rl_time.perf_counter()
        client.set_prompt_caching(["system"])
        _prompt_caching_ms = (_rl_time.perf_counter() - _client_step_t0) * 1000
        _client_step_t0 = _rl_time.perf_counter()
        messages = self._preprocess_messages(self.brain_messages + [input_message])
        _preprocess_messages_ms = (_rl_time.perf_counter() - _client_step_t0) * 1000
        _client_ms = (_rl_time.perf_counter() - _t0) * 1000
        log_startup_timing(
            LOGGER,
            (
                "⏱️ [StartupTiming] llm_preamble.client.detail "
                "run_id=%s total=%.0fms new_client=%.0fms thinking_context=%.0fms "
                "set_system=%.0fms prompt_caching=%.0fms preprocess_messages=%.0fms "
                "state_message=%.0fms brain_messages=%d "
                "message_count=%d system_parts=%d state_chars=%d"
            ),
            run_id,
            _client_ms,
            _new_client_ms,
            _thinking_context_ms,
            _set_system_ms,
            _prompt_caching_ms,
            _preprocess_messages_ms,
            _state_message_ms,
            len(self.brain_messages),
            len(messages),
            len(system_prompt.to_list()),
            len(brain_spec.state_prompt),
        )

        _source_token = _EVENT_SOURCE.set("ConversationManager")

        _rl_t0 = _rl_time.perf_counter()

        def _rl_ms() -> str:
            return f"{(_rl_time.perf_counter() - _rl_t0) * 1000:.0f}ms"

        self._session_logger.debug(
            "perf",
            (
                f"[_run_llm preamble={_ms_since_start()}] "
                f"render_state={_render_ms:.0f}ms brain_spec={_brain_spec_ms:.0f}ms "
                f"tools={_tools_ms:.0f}ms client={_client_ms:.0f}ms | "
                f"calling single_shot_tool_decision ({len(tools)} tools, {len(messages)} msgs)"
            ),
        )
        log_startup_timing(
            LOGGER,
            (
                "⏱️ [StartupTiming] first_reply.llm_preamble "
                "run_id=%s duration=%s render_state=%.0fms brain_spec=%.0fms "
                "tools=%.0fms client=%.0fms tool_count=%d message_count=%d"
            ),
            run_id,
            _ms_since_start(),
            _render_ms,
            _brain_spec_ms,
            _tools_ms,
            _client_ms,
            len(tools),
            len(messages),
        )
        self._active_llm_trace_meta = trace_meta

        try:
            try:
                result = await single_shot_tool_decision(
                    client,
                    messages,
                    tools,
                    tool_choice="required" if tools else "auto",
                    inject_tool_thoughts=True,
                    on_tool_execution_start=lambda: self._mark_tool_commit_started(
                        trace_meta,
                        run_id,
                    ),
                )
            except Exception:
                _log_slow_brain_single_shot_failure(
                    run_id=run_id,
                    request_id=request_id,
                    origin_event_name=origin_event_name,
                    message_count=len(messages),
                    tool_count=len(tools),
                    state_chars=len(input_message),
                )
                raise
        finally:
            self._active_llm_trace_meta = None
            if hasattr(client, "_pending_thinking_log"):
                client._pending_thinking_log.emit_fallback()
            _EVENT_SOURCE.reset(_source_token)
        tool_names = [t.name for t in result.tools]
        self._session_logger.debug(
            "perf",
            f"[_run_llm +{_rl_ms()}] single_shot returned tools={tool_names}",
        )
        log_startup_timing(
            LOGGER,
            "⏱️ [StartupTiming] first_reply.single_shot duration=%s run_id=%s tools=%s",
            _rl_ms(),
            run_id,
            tool_names,
        )
        self._record_recent_tool_executions(
            tools=result.tools,
            trace_meta=trace_meta,
        )

        thoughts_summary = _format_tool_thoughts_for_log(result.tools)

        llm_response_msg = f"run_id={run_id}"
        if thoughts_summary:
            preview = (
                f"{thoughts_summary[:100]}..."
                if len(thoughts_summary) > 100
                else thoughts_summary
            )
            llm_response_msg += f" thoughts: {preview}"
        if tool_names:
            llm_response_msg += f" | actions: {tool_names}"
        self._session_logger.debug("llm_response", llm_response_msg)

        self._session_logger.debug(
            "perf",
            f"[_run_llm +{_rl_ms()}] tools executed, committing",
        )
        self.commit()
        log_startup_timing(
            LOGGER,
            "⏱️ [StartupTiming] first_reply.commit completed run_id=%s elapsed=%s",
            run_id,
            _rl_ms(),
        )
        self._session_logger.debug("state_update", "Committing state")

        # Clear the temporary state snapshots now that tools have executed
        self._current_state_snapshot = None
        self._current_snapshot_state = None

        # Record the turn in the brain's own message list
        assistant_content = result.text_response or ""
        self.brain_messages.append(input_message)
        self.brain_messages.append({"role": "assistant", "content": assistant_content})

        # If the LLM called wait(delay=N), schedule a delayed follow-up turn.
        for tool_exec in result.tools:
            if tool_exec.name == "wait":
                delay = (tool_exec.args or {}).get("delay")
                msg = (
                    f"Decided to wait {delay} seconds"
                    if delay is not None
                    else "Decided to wait"
                )
                self._session_logger.info("wait", msg)
                if delay is not None:
                    delay = self._clamp_wait_poll_delay(delay)
                    await self.run_llm(delay=delay)
                break

        if "wait" not in tool_names:
            await self._open_slow_brain_follow_on_turn(
                origin_run_id=run_id,
                previous_tools=tool_names,
            )

        self._session_logger.debug(
            "perf",
            f"[_run_llm +{_rl_ms()}] post-processing done",
        )
        log_startup_timing(
            LOGGER,
            "⏱️ [StartupTiming] first_reply.post_processing completed run_id=%s elapsed=%s",
            run_id,
            _rl_ms(),
        )
        self._session_logger.debug(
            "llm_response",
            (f"Slow-brain run completed run_id={run_id} " f"tools={tool_names or '-'}"),
        )

        return tool_names

    def _mark_tool_commit_started(
        self,
        trace_meta: dict[str, str] | None,
        run_id: str,
    ) -> None:
        if trace_meta is not None:
            trace_meta["tool_commit_started"] = "true"
        running_meta = getattr(self.debouncer, "running_task_trace_meta", None)
        if isinstance(running_meta, dict) and running_meta.get("run_id") == run_id:
            running_meta["tool_commit_started"] = "true"
        self._session_logger.debug(
            "llm_thinking",
            f"Slow-brain run entered tool commit run_id={run_id}",
        )

    async def wait_for_events(self):
        async with self.event_broker.pubsub() as pubsub:
            await pubsub.psubscribe(
                "app:comms:*",
                "app:actor:*",
                "app:logging:message_logged",
                "app:managers:output",
            )

            # A retired session stops listening: once `stop` is set the broker
            # has been (or is about to be) closed, so nothing new can arrive,
            # and an in-process successor must not find this loop still
            # holding the old session's machinery.
            while not self.stop.is_set():
                msg = await pubsub.get_message(
                    timeout=2,
                    ignore_subscribe_messages=True,
                )

                if not msg:
                    continue
                # process events
                event = Event.from_json(msg["data"])
                channel = msg.get("channel", "")
                self._event_trace_seq += 1
                event_id = f"evt-{self._event_trace_seq:06d}"
                event_name = event.__class__.__name__
                self._current_event_trace = {
                    "event_id": event_id,
                    "event_name": event_name,
                }
                if event.__class__.loggable:
                    self._session_logger.debug(
                        "event_trace",
                        (
                            f"Processing event_id={event_id} "
                            f"event={event_name} channel={channel or '-'}"
                        ),
                    )
                try:
                    _event_t0 = self.loop.time()
                    await EventHandler.handle_event(
                        event,
                        self,
                    )
                    log_startup_timing(
                        LOGGER,
                        "⏱️ [StartupTiming] event.handle_event duration=%.2fs event_id=%s event=%s channel=%s",
                        self.loop.time() - _event_t0,
                        event_id,
                        event_name,
                        channel or "-",
                    )
                    _flush_t0 = self.loop.time()
                    await self.flush_llm_requests()
                    log_startup_timing(
                        LOGGER,
                        "⏱️ [StartupTiming] event.flush_llm_requests duration=%.2fs event_id=%s event=%s",
                        self.loop.time() - _flush_t0,
                        event_id,
                        event_name,
                    )
                except Exception as exc:
                    LOGGER.error(
                        f"⚠️ [EventLoop] Unhandled error processing "
                        f"event_id={event_id} event={event_name} "
                        f"channel={channel or '-'}: {exc}",
                        exc_info=True,
                    )
                finally:
                    self._current_event_trace = None

    async def _request_shutdown(self, reason: str, log_str: str) -> None:
        """Signal the session to wind down, from any reason the caller recognises.

        One sequence for every exit so a new one cannot half-implement it:
        record the reason, log ``session_end``, set ``stop`` and close the
        broker so nothing new can arrive. Whether the session *should* stop
        is the caller's question.
        """

        self.shutdown_reason = reason
        LOGGER.info(f"{DEFAULT_ICON} {log_str}")
        self._session_logger.info("session_end", log_str)
        self.stop.set()
        await self.event_broker.aclose()

    async def _retire_in_flight_actions(self) -> None:
        """Discard the in-flight action registry.

        A retirement must not wait on work that may never finish — a
        persist-mode act parked for an interjection, or a provider call that
        has hung. Every handle gets one stop request under a single shared
        grace period; whatever ignores it is abandoned along with the
        registry.
        """
        stops = []
        for handle_data in list(self.in_flight_actions.values()):
            handle = handle_data.get("handle")
            if handle is None:
                continue
            if hasattr(handle, "trigger_completion"):
                handle.trigger_completion()
            else:
                # Stopping an already-finished handle is a no-op, so no
                # done() probe — its signature varies across handle types.
                stops.append(
                    asyncio.ensure_future(handle.stop(reason="session retired")),
                )
        if stops:
            done, pending = await asyncio.wait(stops, timeout=5.0)
            for task in done:
                if not task.cancelled() and task.exception() is not None:
                    self._session_logger.info(
                        "session_end",
                        f"In-flight action stop failed during retirement: "
                        f"{task.exception()!r}",
                    )
            for task in pending:
                task.cancel()
            if pending:
                self._session_logger.info(
                    "session_end",
                    f"Abandoned {len(pending)} in-flight action(s) that did "
                    "not stop within the retirement grace period",
                )
        self.in_flight_actions.clear()
        self.completed_actions.clear()

    async def cleanup(self):
        """Retire in-flight actions and stop.

        The conversation needs no flush: every message was written to the
        chat table as it arrived.
        """
        await self._retire_in_flight_actions()
        self.stop.set()

    async def stop_in_flight_action_by_calling_id(
        self,
        calling_id: str,
        *,
        reason: str = "",
    ) -> bool:
        """Stop the in-flight act whose ManagerMethod ``calling_id`` matches.

        External stop requests identify roots by EventBus ``calling_id``
        (UUID). CM steering uses integer ``handle_id``; this bridges the two.
        """
        if not calling_id:
            return False

        for handle_id, handle_data in list(self.in_flight_actions.items()):
            handle = handle_data.get("handle")
            stored = handle_data.get("calling_id") or getattr(
                handle,
                "_manager_call_id",
                None,
            )
            if stored != calling_id:
                continue

            stop_reason = reason or "Stopped by request."
            handle_data.setdefault("handle_actions", []).append(
                {
                    "action_name": f"stop_{handle_id}",
                    "query": stop_reason,
                    "timestamp": prompt_now(),
                },
            )
            if handle is not None:
                await handle.stop(reason=stop_reason)
            stopped = self.in_flight_actions.pop(handle_id, None)
            if stopped is not None:
                self.completed_actions[handle_id] = stopped
            return True
        return False
