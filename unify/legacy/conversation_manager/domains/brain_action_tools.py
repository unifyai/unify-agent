"""
Brain action tools for ConversationManager.

Context Propagation:
- When `act` is called, the current state snapshot is passed to Actor via _parent_chat_context
- For `interject` operations, only the incremental diff from the initial snapshot is sent
  via _parent_chat_context_cont, avoiding duplication of unchanged state
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from functools import wraps
import inspect
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel as _BaseModel
from pydantic import create_model as _create_model

from unify.common.plain_text import (
    PLACEHOLDER_CONTENT_ERROR,
    is_placeholder_outbound_content,
    normalize_outbound_plain_text,
)
from unify.common.prompt_helpers import now as prompt_now
from unify.workspace import get_local_root

from unify.legacy.conversation_manager.domains import managers_utils
from unify.legacy.conversation_manager.events import (
    ActorHandleStarted,
    ActorHandleResponse,
    UnifyMessageSent,
)
from unify.common._async_tool.dynamic_tools_factory import DynamicToolFactory
from unify.common._async_tool.utils import get_handle_paused_state
from unify.legacy.conversation_manager.task_actions import (
    OPERATION_MAP,
    safe_call_id_suffix,
)
from unify.legacy.conversation_manager.domains.renderer import (
    SnapshotState,
    compute_snapshot_diff,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from unify.legacy.conversation_manager.conversation_manager import (
        ConversationManager,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Schema dict → Pydantic model conversion
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA_TYPE_MAP: dict[str, type] = {
    "string": str,
    "str": str,
    "integer": int,
    "int": int,
    "number": float,
    "float": float,
    "boolean": bool,
    "bool": bool,
}


def _resolve_schema_type(schema: Any, name_hint: str) -> type:
    """Resolve a single schema value to a Python type.

    Handles:
    - String type names (``"string"``, ``"integer"``, …)
    - Nested dicts (recursively creates a child Pydantic model)
    - Lists where the first element defines the item schema
    """
    if isinstance(schema, str):
        return _SCHEMA_TYPE_MAP.get(schema.lower(), str)
    if isinstance(schema, dict):
        return schema_dict_to_pydantic(schema, name_hint)
    if isinstance(schema, list) and len(schema) > 0:
        item_type = _resolve_schema_type(schema[0], f"{name_hint}Item")
        return list[item_type]  # type: ignore[valid-type]
    return str  # fallback for unrecognised shapes


def schema_dict_to_pydantic(
    schema: dict,
    model_name: str = "ResponseFormat",
) -> type[_BaseModel]:
    """Convert a simplified schema dict to a dynamic Pydantic model.

    The schema uses a concise, LLM-friendly format:

    - **String values** are type names: ``"string"``, ``"integer"``,
      ``"number"``, ``"boolean"`` (shorthand ``"str"``, ``"int"``, etc.
      also accepted).
    - **Dict values** define nested object schemas (recursively converted).
    - **List values** define array types; the first element is the item
      schema.

    Examples::

        # Flat
        {"email": "string", "age": "integer"}

        # Nested with array
        {"contacts": [{"name": "string", "phone": "string"}], "total": "integer"}
    """
    fields: dict[str, tuple[type, ...]] = {}
    for field_name, field_schema in schema.items():
        field_type = _resolve_schema_type(
            field_schema,
            f"{model_name}{field_name.title()}",
        )
        fields[field_name] = (field_type, ...)
    return _create_model(model_name, **fields)


_next_handle_id = 0


# Pattern matching <in_flight_actions>...</in_flight_actions> sections.
# These contain CM-level steering tools that should not be exposed to the Actor.
_IN_FLIGHT_ACTIONS_PATTERN = re.compile(
    r"<in_flight_actions>.*?</in_flight_actions>\s*",
    re.DOTALL,
)

# Completed-action (and any other) <steering_tools> blocks list CM-only
# ``*_action(handle_id=N, ...)`` invocations that are not in the Actor's scope.
_STEERING_TOOLS_PATTERN = re.compile(
    r"<steering_tools>.*?</steering_tools>\s*",
    re.DOTALL,
)


def _filter_cm_state_for_actor(state_snapshot: dict) -> dict:
    """Filter CM state snapshot before passing to Actor as parent context.

    The CM state snapshot contains <in_flight_actions> with <steering_tools>
    listing CM-level invocations (stop_action, pause_action, interject_action,
    ask_action, ...) for each action. <completed_actions> keeps the same
    <steering_tools> surface for post-hoc ask_action calls. These are CM
    brain tools that exist only in the CM's tool surface.

    If passed verbatim to the Actor, the Actor LLM may interpret these tool
    names as callable functions and generate code like:
        await stop_action(handle_id=1)
    This causes NameError since these tools don't exist in the Actor's scope.

    This function strips the <in_flight_actions> section and any remaining
    <steering_tools> blocks while preserving other useful context
    (notifications, active_conversations, completed action results/history).

    Args:
        state_snapshot: The CM state snapshot dict with "content" key.

    Returns:
        A filtered copy of the snapshot with CM-only tool listings removed.
    """
    if not state_snapshot:
        return state_snapshot

    content = state_snapshot.get("content", "")
    if not content:
        return state_snapshot

    content = _IN_FLIGHT_ACTIONS_PATTERN.sub("", content)
    content = _STEERING_TOOLS_PATTERN.sub("", content)
    return {**state_snapshot, "content": content}


# Largest file the chat accepts as a single attachment.
_MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024


def _attachment_path(filepath: str) -> str | dict[str, Any]:
    """Resolve a file the assistant wants to attach to a chat message.

    Relative paths resolve against the workspace root. Returns the path to
    put on the message — workspace-relative when the file lives inside the
    workspace, absolute otherwise — or an ``{"error": ...}`` payload when the
    file cannot be attached.
    """
    root = Path(get_local_root()).resolve()
    path = Path(filepath).expanduser()
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        return {"error": f"File not found: {filepath}"}
    size_bytes = path.stat().st_size
    if size_bytes > _MAX_ATTACHMENT_BYTES:
        size_mb = size_bytes / (1024 * 1024)
        return {
            "error": f"File too large: {size_mb:.1f}MB exceeds 25MB attachment limit.",
        }
    resolved = path.resolve()
    if resolved.is_relative_to(root):
        return resolved.relative_to(root).as_posix()
    return str(resolved)


# Whether the outbound send in progress is the slow brain's own direct tool
# call. The sent-message event such a send publishes must not wake the slow
# brain again: it already knows what it just said.
_SLOW_BRAIN_DIRECT_OUTBOUND: ContextVar[bool] = ContextVar(
    "slow_brain_direct_outbound",
    default=False,
)


def slow_brain_direct_outbound_active() -> bool:
    return _SLOW_BRAIN_DIRECT_OUTBOUND.get()


def slow_brain_direct_comms(method):
    @wraps(method)
    async def wrapper(self, *args, **kwargs):
        token = _SLOW_BRAIN_DIRECT_OUTBOUND.set(True)
        try:
            return await method(self, *args, **kwargs)
        finally:
            _SLOW_BRAIN_DIRECT_OUTBOUND.reset(token)

    return wrapper


class ConversationManagerBrainActionTools:
    """Side-effecting tools for the Main CM Brain."""

    def __init__(self, cm: "ConversationManager"):
        self._cm = cm
        self._event_broker = cm.event_broker

    @slow_brain_direct_comms
    async def send_unify_message(
        self,
        *,
        content: str,
        attachment_filepath: str | None = None,
    ) -> dict[str, Any]:
        """Send a chat message to the user.

        Write plain text: prose as continuous lines that reflow naturally, a
        blank line between paragraphs, and each bullet or numbered item on
        its own line.

        Parameters
        ----------
        content : str
            Message body to send.
        attachment_filepath : str | None, optional
            Path of one file to attach, relative to the workspace root.

        Returns
        -------
        dict[str, Any]
            ``{"status": "ok"}`` on success, or an ``{"error": ...}`` payload
            describing why the message was not sent.
        """
        content = normalize_outbound_plain_text(content)
        if is_placeholder_outbound_content(content):
            return {"error": PLACEHOLDER_CONTENT_ERROR}

        attachments: list[str] = []
        if attachment_filepath:
            attachment = _attachment_path(attachment_filepath)
            if isinstance(attachment, dict):
                return attachment
            attachments.append(attachment)

        event = UnifyMessageSent(content=content, attachments=attachments)
        event.suppress_slow_brain_wake = slow_brain_direct_outbound_active()
        await self._event_broker.publish(UnifyMessageSent.topic, event.to_json())
        return {"status": "ok"}

    async def act(
        self,
        *,
        query: str,
        response_format: Optional[dict] = None,
        persist: bool = False,
        include_conversation_context: bool = True,
        llm_profile: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Engage with the workspace, stored skills, and the world beyond the chat.

        This is the all-purpose method for any work that requires computing,
        reading, retrieving, manipulating, or acting on information. Use ``act``
        liberally — if it cannot help, it will simply report back. There is no
        penalty for speculative delegation.

        **Capabilities include:**

        - **Retrieval**: Read files in the workspace (attachments the user sent
          arrive there as paths), look up stored procedures and functions,
          fetch data from the web and APIs through code
        - **Action**: Write and run code, produce files, call stored functions,
          store new functions and procedures for next time
        - **Combined**: Find information and act on it (e.g., "read the attached
          CSV and chart the monthly totals")

        **When uncertain, call ``act``**: If you need information you don't have,
        call ``act`` to find it. If ``act`` can't find it, it will tell you, and
        you can then ask the user.

        Args:
            query: Natural language request specifying what to do or find,
                written in English whatever language the conversation is
                in; the actor works in English. Quote the user's own words
                only where they are the payload (a name, an address, the
                text of a message to send). Quote attachment paths from the
                conversation verbatim so the actor can open the files.
            response_format: An optional structured schema describing the shape of
                the result you need back.  When provided, the action is required to
                return a JSON object conforming to this schema (via a dedicated
                ``final_response`` tool) instead of free-form text.

                The schema uses a concise format where keys are field names and
                values describe their types:

                - Type strings: ``"string"``, ``"integer"``, ``"number"``,
                  ``"boolean"`` (shorthand ``"str"``, ``"int"``, etc. also work).
                - Nested objects: use a dict value, e.g.
                  ``{"address": {"city": "string", "zip": "string"}}``.
                - Arrays: use a single-element list whose element defines the item
                  schema, e.g. ``[{"name": "string", "email": "string"}]``.

                **Examples:**

                - Simple flat fields::

                      {"email": "string", "phone": "string"}

                - Nested with array::

                      {"contacts": [{"name": "string", "email": "string"}],
                       "total_count": "integer"}

                When omitted (the default), the action returns free-form text and
                the result is whatever the actor decides to report.
            persist: If True, the action runs as a **persistent session** that does
                not self-complete.  The actor stays alive after each response and
                waits for the next ``interject`` before continuing.  Use this for
                long-running interactive sessions (e.g. multi-step procedures with
                a tight feedback loop between conversation and action).

                **Key differences from the default (persist=False):**

                - The action will **never** complete on its own.  You must
                  explicitly call ``stop_action`` to end the session.
                - Intermediate responses from the actor appear as **response**
                  events in the action's history (marked ``awaiting_input``).
                  Each response means the actor has finished its current turn
                  and is waiting for your next instruction via ``interject_action``.
                - Progress updates (notifications) may still arrive while the
                  actor is working, before it sends a response.

                The default (False) is a one-shot task: the actor works until
                done and the result arrives as an ``ActorResult``.
            include_conversation_context: Whether to pass the current conversation
                state to the action. When ``true`` (default), the action receives
                the full rendered conversation snapshot — messages, notifications,
                and in-flight actions — helping it understand the broader context.
                Set ``false`` when the action is self-contained and the query
                alone provides all necessary information (e.g. simple lookups
                or factual questions). Subsequent steering calls
                (interject, ask) on this action will also skip context forwarding.
            llm_profile: Optional curated LLM profile for this action. Leave
                unset for the default actor profile, normally
                ``openai/gpt-5.6-sol@openrouter`` at high reasoning effort. Use
                ``gpt_5_5_low``, ``gpt_5_5_medium``, or ``gpt_5_5_high`` only
                when the task or the user's wording warrants the GPT-5.5
                family specifically. Requests to "use all of your thinking effort"
                or similar explicitly select ``gpt_5_5_high``.

                Escalate the profile when retrying an action that shows
                concrete evidence of model/tool-use struggle, rather than
                retrying the same default profile repeatedly. Good escalation
                signals include a previous ``ActorResult`` ending in a
                tool-schema or tool-call formatting error, the same failed
                step recurring after a retry, repeated execution mistakes
                without new information being gathered, clear user frustration,
                or the user's explicit request for stronger/premium reasoning.
                Do not escalate solely because an action has been running for
                a long time; long-running data, coding, or browser work can be
                normal. If restarting after one of the concrete failure
                signals above, preserve the user's task and set
                ``llm_profile`` to ``gpt_5_5_medium`` or ``gpt_5_5_high``
                depending on difficulty and urgency.
        """
        global _next_handle_id

        cm = self._cm

        suppression = cm.suppress_duplicate_commissioning_tool(
            tool_name="act",
            tool_args={
                "query": query,
                "response_format": response_format,
                "persist": persist,
                "include_conversation_context": include_conversation_context,
                "llm_profile": llm_profile,
            },
        )
        if suppression is not None:
            return suppression

        # Pass the fresh rendered state snapshot as context for the Actor,
        # unless the LLM opted out.
        parent_context = None
        if include_conversation_context:
            parent_context = (
                [_filter_cm_state_for_actor(cm._current_state_snapshot)]
                if cm._current_state_snapshot
                else None
            )

        # Convert the LLM-provided schema dict into a Pydantic model that the
        # Actor's async tool loop uses for structured output validation.
        pydantic_response_format = None
        if response_format is not None:
            pydantic_response_format = schema_dict_to_pydantic(response_format)

        handle_id = _next_handle_id
        _next_handle_id += 1

        handle = await cm.actor.act(
            query,
            _parent_chat_context=parent_context,
            response_format=pydantic_response_format,
            persist=persist,
            llm_profile=llm_profile,
        )

        initial_snapshot_state: SnapshotState | None = None
        if hasattr(cm, "_current_snapshot_state"):
            initial_snapshot_state = cm._current_snapshot_state

        cm.in_flight_actions[handle_id] = {
            "handle": handle,
            "query": query,
            "persist": persist,
            "llm_profile": llm_profile,
            "action_type": "act",
            "calling_id": getattr(handle, "_manager_call_id", None),
            "handle_actions": [
                {
                    "action_name": "act_started",
                    "query": query,
                    "timestamp": prompt_now(),
                },
            ],
            "initial_snapshot_state": initial_snapshot_state,
            "context_opted_in": include_conversation_context,
        }
        asyncio.create_task(
            managers_utils.actor_watch_result(
                handle_id,
                handle,
                action_type="act",
            ),
        )
        asyncio.create_task(
            managers_utils.actor_watch_notifications(handle_id, handle),
        )
        asyncio.create_task(
            managers_utils.actor_watch_clarifications(handle_id, handle),
        )

        await self._event_broker.publish(
            f"app:actor:actor_started_handle_{handle_id}",
            ActorHandleStarted(
                handle_id=handle_id,
                action_name="act",
                query=query,
                response_format=response_format,
            ).to_json(),
        )

        return {"status": "acting", "query": query}

    async def wait(
        self,
        delay: int | None = None,
    ) -> dict[str, Any]:
        """
        Wait for more input without taking any action.

        Call this tool when I have nothing left to say or do this turn:
        - After I already answered the user's latest message and they should
          have the last word
        - When there are no NEW inbound messages or completion events to handle
        - After starting a long-running action whose outcome I will handle on
          the next event
        - To let a natural exchange end

        Do NOT call this tool when:
        - The user just sent a message I have not answered yet
        - The user asks a question, expresses confusion, or checks whether I
          am still here ("hello?", "what next?", "are you ignoring me?")

        The user should usually have the last word after I answer — not while
        they are waiting on me. Do not send unprompted "anything else?" filler,
        but do not leave an unanswered message on the chat thread.

        Parameters
        ----------
        delay : int | None
            Seconds to wait before automatically waking up for another thinking
            turn.  When ``None`` (the default), wait indefinitely until the next
            external event (new message, action completion, etc.).  When set to a
            positive integer, the system schedules a follow-up thinking turn after
            that many seconds — useful for probing a long-running action or
            revisiting a situation after a reasonable interval.
        """
        return {"status": "waiting", "delay": delay}

    def as_tools(self) -> dict[str, "Callable[..., Any]"]:
        """Return the static tools dict for start_async_tool_loop."""
        tools: dict[str, Callable[..., Any]] = {
            "send_unify_message": self.send_unify_message,
            "wait": self.wait,
        }
        if self._cm.initialized:
            tools["act"] = self.act
        return tools

    def build_action_steering_tools(self) -> dict[str, "Callable[..., Any]"]:
        """Fixed-name steering tools that address actions by ``handle_id``.

        The tool set must stay constant across brain runs: tool definitions
        precede messages in provider prompt-cache keys, so a schema that
        changes with the in-flight action set re-bills the entire static
        prompt on every action transition. Targets arrive as ``handle_id``
        (the ``<action id='N'>`` shown in the ``in_flight_actions`` pane)
        and are resolved at call time; a stale or unknown id gets a
        corrective error instead of a vanished tool.
        """
        cm = self._cm

        def _in_flight(handle_id: Any) -> tuple[Optional[int], Optional[dict]]:
            try:
                hid = int(handle_id)
            except (TypeError, ValueError):
                return None, None
            return hid, (cm.in_flight_actions or {}).get(hid)

        def _missing(handle_id: Any, operation: str) -> dict[str, Any]:
            ids = sorted((cm.in_flight_actions or {}).keys())
            return {
                "status": "error",
                "operation": operation,
                "message": (
                    f"No in-flight action with handle_id={handle_id!r}. "
                    f"In flight now: {ids if ids else 'none'} — use the id "
                    "from the in_flight_actions pane."
                ),
            }

        def _delegate(
            op_name: str,
            hid: int,
            handle_data: dict,
            call_id: str | None = None,
        ) -> "Callable[..., Any]":
            op = OPERATION_MAP[op_name]
            return self._make_steering_tool(
                hid,
                handle_data.get("handle"),
                op.name,
                op.param_name,
                op.get_docstring(),
                handle_data.get("query", ""),
                call_id,
            )

        async def interject_action(handle_id: int, message: str) -> dict[str, Any]:
            """Send guidance into a running action without stopping it.

            Args:
                handle_id: The action's id from the in_flight_actions pane.
                message: The guidance or correction to deliver.
            """
            hid, handle_data = _in_flight(handle_id)
            if handle_data is None:
                return _missing(handle_id, "interject")
            return await _delegate("interject", hid, handle_data)(message=message)

        async def stop_action(handle_id: int, reason: str = "") -> dict[str, Any]:
            """Terminate a running action immediately.

            Args:
                handle_id: The action's id from the in_flight_actions pane.
                reason: Optional reason recorded with the stop.
            """
            hid, handle_data = _in_flight(handle_id)
            if handle_data is None:
                return _missing(handle_id, "stop")
            return await _delegate("stop", hid, handle_data)(reason=reason)

        async def pause_action(handle_id: int) -> dict[str, Any]:
            """Pause a running action; resume it later with resume_action.

            Args:
                handle_id: The action's id from the in_flight_actions pane.
            """
            hid, handle_data = _in_flight(handle_id)
            if handle_data is None:
                return _missing(handle_id, "pause")
            if get_handle_paused_state(handle_data.get("handle")) is True:
                return {
                    "status": "ok",
                    "operation": "pause",
                    "message": f"Action {hid} is already paused.",
                }
            return await _delegate("pause", hid, handle_data)()

        async def resume_action(handle_id: int) -> dict[str, Any]:
            """Resume a paused action.

            Args:
                handle_id: The action's id from the in_flight_actions pane.
            """
            hid, handle_data = _in_flight(handle_id)
            if handle_data is None:
                return _missing(handle_id, "resume")
            if get_handle_paused_state(handle_data.get("handle")) is not True:
                return {
                    "status": "ok",
                    "operation": "resume",
                    "message": f"Action {hid} is not paused.",
                }
            return await _delegate("resume", hid, handle_data)()

        async def ask_action(handle_id: int, question: str) -> dict[str, Any]:
            """Ask a question of a running or completed action.

            Args:
                handle_id: The action's id from the in_flight_actions or
                    completed_actions pane.
                question: What to ask about the action's work or results.
            """
            hid, handle_data = _in_flight(handle_id)
            if handle_data is None and hid is not None:
                handle_data = (cm.completed_actions or {}).get(hid)
            if handle_data is None:
                return _missing(handle_id, "ask")
            return await _delegate("ask", hid, handle_data)(question=question)

        async def answer_clarification_action(
            handle_id: int,
            answer: str,
            call_id: str = "",
        ) -> dict[str, Any]:
            """Answer a pending clarification a running action has raised.

            Args:
                handle_id: The action's id from the in_flight_actions pane.
                answer: The answer to give the action.
                call_id: The clarification's call id (shown in the pane).
                    May be omitted when exactly one clarification is pending.
            """
            hid, handle_data = _in_flight(handle_id)
            if handle_data is None:
                return _missing(handle_id, "answer_clarification")
            pending = [
                a
                for a in handle_data.get("handle_actions", [])
                if a.get("action_name") == "clarification_request"
                and not a.get("response")
            ]
            if not pending:
                return {
                    "status": "error",
                    "operation": "answer_clarification",
                    "message": f"Action {hid} has no pending clarification.",
                }
            chosen = None
            if call_id:
                for clar in pending:
                    cid = clar.get("call_id", "")
                    if cid == call_id or safe_call_id_suffix(cid) == call_id:
                        chosen = clar
                        break
                if chosen is None:
                    return {
                        "status": "error",
                        "operation": "answer_clarification",
                        "message": (
                            f"No pending clarification with call_id={call_id!r} "
                            f"on action {hid}; pending: "
                            f"{[c.get('call_id', '') for c in pending]}"
                        ),
                    }
            elif len(pending) == 1:
                chosen = pending[0]
            else:
                return {
                    "status": "error",
                    "operation": "answer_clarification",
                    "message": (
                        f"Action {hid} has {len(pending)} pending "
                        "clarifications; pass call_id to pick one: "
                        f"{[c.get('call_id', '') for c in pending]}"
                    ),
                }
            return await _delegate(
                "answer_clarification",
                hid,
                handle_data,
                chosen.get("call_id", ""),
            )(answer=answer)

        return {
            "interject_action": interject_action,
            "stop_action": stop_action,
            "pause_action": pause_action,
            "resume_action": resume_action,
            "ask_action": ask_action,
            "answer_clarification_action": answer_clarification_action,
        }

    @staticmethod
    def _extract_tool_param_value(
        *,
        kwargs: dict[str, Any],
        primary_name: str,
        aliases: tuple[str, ...] = (),
    ) -> Any:
        """Extract a tool parameter value from kwargs using primary name then aliases."""
        if not primary_name:
            return ""
        for name in (primary_name, *aliases):
            if name in kwargs:
                return kwargs.get(name, "")
        return ""

    def _make_steering_tool(
        self,
        handle_id: int,
        handle: Any,
        operation: str,
        param_name: str,
        docstring: str,
        query: str,
        call_id: str | None = None,
    ) -> "Callable[..., Any]":
        """Create a closure for an action steering operation.

        ``ask`` also serves completed actions: the handle keeps its
        trajectory after ``ActorResult`` moves it to ``completed_actions``,
        so the closure resolves the action's record from either registry at
        call time.
        """

        cm = self._cm
        # Use cm.event_broker to ensure the same broker is used throughout
        # (important for test patching)
        event_broker = cm.event_broker

        async def steering_tool(
            **kwargs: Any,
        ) -> dict[str, Any]:
            param_aliases: tuple[str, ...] = ()
            if operation == "ask":
                param_aliases = tuple(
                    name for name in ("question", "query") if name != param_name
                )
            param_value = self._extract_tool_param_value(
                kwargs=kwargs,
                primary_name=param_name,
                aliases=param_aliases,
            )

            handle_data = cm.in_flight_actions.get(
                handle_id,
            ) or cm.completed_actions.get(handle_id)

            result = ""
            try:
                match operation:
                    case "ask":
                        # Record action with pending status - result will arrive async
                        if handle_data:
                            handle_data["handle_actions"].append(
                                {
                                    "action_name": f"ask_{handle_id}",
                                    "query": param_value,
                                    "status": "pending",
                                    "timestamp": prompt_now(),
                                },
                            )

                        # Capture values for the closure.
                        # Use the fresh rendered state snapshot (set by _run_llm before tools execute).
                        # Only pass context if the original action opted in.
                        _handle = handle
                        _param_value = param_value
                        _handle_id = handle_id
                        _ctx_opted_in = (
                            handle_data.get("context_opted_in", True)
                            if handle_data
                            else True
                        )
                        _parent_context = None
                        if _ctx_opted_in and cm._current_state_snapshot:
                            _parent_context = [cm._current_state_snapshot]

                        # Spawn background task to perform ask and emit result
                        async def _perform_ask_and_emit():
                            try:
                                # Start the ask operation (does the LLM roundtrip)
                                ask_handle = await _handle.ask(
                                    _param_value,
                                    _parent_chat_context=_parent_context,
                                )
                                # Await the result
                                ask_result = await ask_handle.result()
                            except Exception as e:
                                ask_result = f"Error: {e}"
                            # Emit ActorHandleResponse event to wake brain
                            await event_broker.publish(
                                f"app:actor:handle_response_{_handle_id}",
                                ActorHandleResponse(
                                    handle_id=_handle_id,
                                    action_name="ask",
                                    query=_param_value,
                                    response=ask_result,
                                    call_id="",
                                ).to_json(),
                            )

                        task = asyncio.create_task(_perform_ask_and_emit())
                        cm._pending_steering_tasks.add(task)
                        task.add_done_callback(
                            cm._pending_steering_tasks.discard,
                        )

                        # Return immediately - brain will be woken when result arrives
                        return {
                            "status": "ok",
                            "operation": "ask",
                            "result": (
                                "Query submitted. You will receive another turn "
                                "when the answer is ready."
                            ),
                        }

                    case "interject":
                        if handle_data:
                            handle_data["handle_actions"].append(
                                {
                                    "action_name": f"{operation}_{handle_id}",
                                    "query": param_value,
                                    "timestamp": prompt_now(),
                                },
                            )

                        # Only compute and send context diffs if the original
                        # action opted into conversation context.
                        parent_context_cont = None
                        _interject_ctx_opted_in = (
                            handle_data.get("context_opted_in", True)
                            if handle_data
                            else True
                        )

                        if _interject_ctx_opted_in:
                            initial_snapshot = (
                                handle_data.get("initial_snapshot_state")
                                if handle_data
                                else None
                            )
                            current_snapshot = getattr(
                                cm,
                                "_current_snapshot_state",
                                None,
                            )

                            if current_snapshot is not None:
                                diff_content = compute_snapshot_diff(
                                    initial_snapshot,
                                    current_snapshot,
                                )
                                if diff_content:
                                    parent_context_cont = [
                                        {
                                            "role": "user",
                                            "content": diff_content,
                                            "_cm_context_diff": True,
                                        },
                                    ]
                            elif cm._current_state_snapshot:
                                parent_context_cont = [cm._current_state_snapshot]

                        await handle.interject(
                            param_value,
                            _parent_chat_context_cont=parent_context_cont,
                        )
                        result = "Interjected successfully"
                    case "stop":
                        if handle_data:
                            handle_data["handle_actions"].append(
                                {
                                    "action_name": f"{operation}_{handle_id}",
                                    "query": param_value,
                                    "timestamp": prompt_now(),
                                },
                            )
                        await handle.stop(reason=param_value or None)
                        stopped = cm.in_flight_actions.pop(handle_id, None)
                        if stopped:
                            cm.completed_actions[handle_id] = stopped
                        result = "Action stopped"
                    case "pause":
                        if handle_data:
                            handle_data["handle_actions"].append(
                                {
                                    "action_name": f"{operation}_{handle_id}",
                                    "query": param_value,
                                    "timestamp": prompt_now(),
                                },
                            )
                        await handle.pause()
                        result = "Action paused"
                    case "resume":
                        if handle_data:
                            handle_data["handle_actions"].append(
                                {
                                    "action_name": f"{operation}_{handle_id}",
                                    "query": param_value,
                                    "timestamp": prompt_now(),
                                },
                            )
                        await handle.resume()
                        result = "Action resumed"
                    case "answer_clarification":
                        if handle_data:
                            handle_data["handle_actions"].append(
                                {
                                    "action_name": f"{operation}_{handle_id}",
                                    "query": param_value,
                                    "timestamp": prompt_now(),
                                },
                            )
                        if call_id:
                            await handle.answer_clarification(call_id, param_value)
                            result = "Clarification answered"
                        else:
                            result = "No clarification call_id available"
                    case _:
                        if handle_data:
                            handle_data["handle_actions"].append(
                                {
                                    "action_name": f"{operation}_{handle_id}",
                                    "query": param_value,
                                    "timestamp": prompt_now(),
                                },
                            )
                        result = f"Unknown operation: {operation}"
            except Exception as e:
                result = f"Error: {e}"

            return {"status": "ok", "operation": operation, "result": result}

        # Copy signature + docstring from the handle's method. Parameters
        # starting with _ are automatically hidden by method_to_schema.
        if handle is not None and hasattr(handle, operation):
            DynamicToolFactory._adopt_signature_and_annotations(
                getattr(handle, operation),
                steering_tool,
            )

        # Append action context so the CM knows which action this tool steers.
        # Preserve the docstring set by _adopt_signature_and_annotations (or
        # fall back to the docstring passed in from SteeringOperation).
        base_doc = inspect.getdoc(steering_tool) or docstring
        steering_tool.__doc__ = f"{base_doc}\n\nFor action: {query}"

        return steering_tool
