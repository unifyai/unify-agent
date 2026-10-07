import asyncio
import re
from typing import TYPE_CHECKING, Any

from unify.common.hierarchical_logger import DEFAULT_ICON
from unify.legacy.conversation_manager.events import (
    ActionStopRequested,
    ActorClarificationRequest,
    ActorHandleResponse,
    ActorHandleStarted,
    ActorNotification,
    ActorResult,
    ActorSessionResponse,
    DirectMessageEvent,
    Error,
    Event,
    InitializationComplete,
    NotificationInjectedEvent,
    OpenSlowBrainTurn,
    UnifyMessageReceived,
    UnifyMessageSent,
)
from unify.legacy.conversation_manager.domains import managers_utils
from unify.logger import LOGGER

if TYPE_CHECKING:
    from unify.legacy.conversation_manager.conversation_manager import (
        ConversationManager,
    )


def _event_type_to_log_key(event_cls) -> str:
    """Convert an event class name to a log key for icon lookup."""
    name = event_cls.__name__

    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


class EventHandler:
    """Registry that maps event classes to their async handlers."""

    _registry: dict[type[Event], Any] = {}

    @classmethod
    def register(cls, event_cls: list[Event] | Event):
        """Register one handler for one or more event classes."""

        def wrapper(func):
            events_classes = (
                [event_cls] if not isinstance(event_cls, (list, tuple)) else event_cls
            )
            for e in events_classes:
                cls._registry[e] = func
            return func

        return wrapper

    @classmethod
    def handle_event(cls, event: Event, cm: "ConversationManager", *args, **kwargs):
        """Dispatch one event to its registered handler."""

        event_key = _event_type_to_log_key(event.__class__)
        if (
            hasattr(cm, "_session_logger")
            and not event.__class__.content_logged
            and event.__class__.loggable
        ):
            log_fn = (
                cm._session_logger.info
                if event.__class__.prominent
                else cm._session_logger.debug
            )
            log_fn(
                event_key,
                f"Event: {event.__class__.__name__}",
            )

        if event.__class__.loggable:
            asyncio.create_task(
                managers_utils.queue_operation(
                    managers_utils.publish_bus_events,
                    event,
                ),
            )

        f = cls._registry.get(event.__class__)
        if not f:
            return asyncio.sleep(0)
        return f(event, cm, *args, **kwargs)


@EventHandler.register(ActionStopRequested)
async def _(
    event: ActionStopRequested,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    stopped = await cm.stop_in_flight_action_by_calling_id(
        event.calling_id,
        reason=event.reason or "Stop requested by the user.",
    )
    if not stopped:
        LOGGER.info(
            "%s Action stop requested for unknown calling_id=%s (source=%s)",
            DEFAULT_ICON,
            event.calling_id,
            event.source or "chat",
        )


@EventHandler.register(ActorClarificationRequest)
async def _(
    event: ActorClarificationRequest,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    if event.handle_id in cm.in_flight_actions:
        from unify.common.prompt_helpers import now as prompt_now

        cm.in_flight_actions[event.handle_id]["handle_actions"].append(
            {
                "action_name": "clarification_request",
                "query": event.query,
                "call_id": event.call_id,
                "timestamp": prompt_now(),
            },
        )
        await cm.request_llm_run()


@EventHandler.register(ActorHandleResponse)
async def _(event: ActorHandleResponse, cm: "ConversationManager", *args, **kwargs):
    """Complete the pending steering query the response answers, then wake the brain.

    Both in-flight and completed actions are checked: a post-completion
    ``ask`` answers after ``ActorResult`` has already moved the action to
    ``completed_actions``.
    """
    handle_data = cm.in_flight_actions.get(
        event.handle_id,
    ) or cm.completed_actions.get(event.handle_id)
    if not handle_data:
        return
    handle_actions = handle_data.get("handle_actions", [])
    action_name = event.action_name or "ask"
    expected_action_name = f"{action_name}_{event.handle_id}"

    for action in reversed(handle_actions):
        if (
            action.get("action_name") == expected_action_name
            and action.get("status") == "pending"
        ):
            action["status"] = "completed"
            action["response"] = event.response
            break

    await cm.request_llm_run()


@EventHandler.register((UnifyMessageSent, UnifyMessageReceived))
async def _(event, cm: "ConversationManager", *args, **kwargs):
    match event:
        case UnifyMessageSent():
            notif_content = "Message sent"
            role = "assistant"
            cm._session_logger.info("unify_message_sent", f"Sent: {event.content}")
        case UnifyMessageReceived():
            notif_content = "Message received"
            role = "user"
            cm._session_logger.info(
                "unify_message_received",
                f"Received: {event.content}",
            )

    cm.chat_history.append(
        role=role,
        content=event.content,
        attachments=event.attachments,
        timestamp=event.timestamp,
    )
    cm.notifications_bar.push_notif("comms", notif_content, event.timestamp)

    if role == "user":
        # A question posed through the handle's ``ask`` owns the next user
        # turn: the reply answers it directly instead of waking the brain.
        ask_handle = cm.active_ask_handle
        if ask_handle is not None and not ask_handle.done():
            await ask_handle.interject(event.content)
            return
        await cm.request_llm_run(is_user_origin=True)
    elif not event.suppress_slow_brain_wake:
        await cm.request_llm_run()


@EventHandler.register(Error)
async def _(event: Error, cm: "ConversationManager", *args, **kwargs):
    """Surface errors to the brain via the notification bar.

    When an outbound tool such as ``send_unify_message`` fails, it publishes
    an Error event. Without a handler, the error is silently dropped and the
    brain never learns about the failure.

    The detailed error context is already pushed into the conversation thread
    by the tool itself. This handler adds a lightweight notification and
    triggers a follow-up brain turn so the brain can see the failure and
    decide how to recover.
    """
    cm.notifications_bar.push_notif("Error", event.message, event.timestamp)
    await cm.request_llm_run(delay=0)


@EventHandler.register(ActorHandleStarted)
async def _(event: ActorHandleStarted, cm: "ConversationManager", *args, **kwargs):
    pass


@EventHandler.register(NotificationInjectedEvent)
async def _(
    event: NotificationInjectedEvent,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    cm._session_logger.info(
        "notification_injected",
        f"Notification: {event.content[:50]}...",
    )

    cm.notifications_bar.push_notif(
        event.source,
        event.content,
        event.timestamp,
        pinned=event.pinned,
    )

    await cm.request_llm_run(delay=0)


@EventHandler.register(ActorResult)
async def _(event: ActorResult, cm: "ConversationManager", *args, **kwargs):
    from unify.common.prompt_helpers import now as prompt_now

    action_data = cm.in_flight_actions.get(event.handle_id, {})
    action_type = event.action_type or action_data.get("action_type") or "act"
    completion_entry = {
        "action_name": "act_completed" if event.success else "act_failed",
        "query": event.result if event.success else (event.error or event.result),
        "timestamp": prompt_now(),
        "success": bool(event.success),
        "action_type": action_type,
    }
    if event.result is not None:
        completion_entry["result"] = event.result
    if event.error:
        completion_entry["error"] = event.error

    # Log completion in handle_actions before moving to completed_actions.
    if action_data and "handle_actions" in action_data:
        action_data["handle_actions"].append(completion_entry)

    # Move to completed_actions (preserves handle for post-completion ask queries)
    completed = cm.in_flight_actions.pop(event.handle_id, None)
    if completed:
        cm.completed_actions[event.handle_id] = completed
    await cm.request_llm_run()


@EventHandler.register(ActorSessionResponse)
async def _(event: ActorSessionResponse, cm: "ConversationManager", *args, **kwargs):
    """A persistent session completed a turn and is awaiting input.

    This is semantically distinct from ``ActorNotification`` (progress update):
    a response means the actor is *done with this turn* and will not proceed
    until the brain interjects with the next instruction.
    """
    action_data = cm.in_flight_actions.get(event.handle_id, {})

    from unify.common.prompt_helpers import now as prompt_now

    if action_data and "handle_actions" in action_data:
        action_data["handle_actions"].append(
            {
                "action_name": "response",
                "query": event.content,
                "status": "awaiting_input",
                "timestamp": prompt_now(),
            },
        )
    await cm.request_llm_run()


@EventHandler.register(ActorNotification)
async def _(event: ActorNotification, cm: "ConversationManager", *args, **kwargs):
    """A progress notification from an in-flight action.

    Unlike ``ActorSessionResponse``, notifications arrive while the actor is
    still working.  Progress is recorded in the action's history so the slow brain
    sees accumulated progress when it next runs on a legitimate event.

    A notification can also arrive *late* -- after the handle has already
    moved from ``in_flight_actions`` to ``completed_actions`` (e.g. the
    StorageCheck phase finishing well after ``ActorResult`` resolved the
    action). Record those too instead of silently dropping them, so the
    handle's history reflects everything that actually happened.
    """
    from unify.common.prompt_helpers import now as prompt_now

    entry = {
        "action_name": "progress",
        "query": event.response,
        "timestamp": prompt_now(),
    }
    action_data = cm.in_flight_actions.get(event.handle_id) or cm.completed_actions.get(
        event.handle_id,
    )
    if action_data and "handle_actions" in action_data:
        action_data["handle_actions"].append(entry)


OPEN_SLOW_BRAIN_TURN_NOTIFICATION = (
    "Open slow-brain turn — your previous turn finished without calling "
    "`wait`. You have another thinking turn now. Continue any outstanding "
    "work (reply to the user, follow up on in-flight actions). Recurring "
    "turns keep opening until you explicitly "
    "call `wait()` or `wait(delay=…)`. Do not call `wait` while the user is "
    "still waiting on you in chat."
)


@EventHandler.register((OpenSlowBrainTurn,))
async def _(
    event: OpenSlowBrainTurn,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    cm.notifications_bar.push_notif(
        "System",
        OPEN_SLOW_BRAIN_TURN_NOTIFICATION,
        event.timestamp,
    )
    await cm.request_llm_run(delay=0)


# Notification text shown to the slow brain when initialization completes.
#
# Wording is deliberately strong about preferring `wait` over a follow-up
# message: in cold-start flows where the brain already replied to a user
# message during pre-init, a looser "review … and follow up if needed
# (correct, elaborate, or confirm)" wording gets interpreted as permission
# to send a rephrased duplicate. The brain must still be allowed to follow
# up legitimately when it deferred work, gave an incomplete/incorrect answer
# due to missing context, or has new history from hydration — so the
# directive enumerates those cases explicitly and tells the brain to call
# `wait` otherwise.
INITIALIZATION_COMPLETE_NOTIFICATION = (
    "Initialization complete — all actions are now available and the "
    "conversation history has been loaded. If your previous reply during "
    "initialization (a) deferred work you can now perform, (b) was "
    "incorrect or incomplete because of missing context now revealed by "
    "the loaded history, or (c) needs a concrete update, follow up now "
    "(act, correct, or elaborate). Otherwise call wait — do NOT send a "
    "message that simply rephrases, restates, or confirms a reply you "
    "already gave."
)


# The truthful variant for a boot whose hydration restored nothing because
# there is no prior conversation. Claiming "history has been loaded" on such
# a boot sends the brain hunting for context that is not there: told history
# was loaded while the conversation render was empty, an assistant went
# searching elsewhere for a spec that only ever existed in the unrendered
# conversation.
INITIALIZATION_COMPLETE_NO_HISTORY_NOTIFICATION = (
    "Initialization complete — all actions are now available. No prior "
    "conversation history was found to load, so the rendered conversation is "
    "the whole of what is known here. If your previous reply during "
    "initialization deferred work you can now perform or needs a concrete "
    "update, follow up now. Otherwise call wait — do NOT send a message "
    "that simply rephrases, restates, or confirms a reply you already "
    "gave."
)


@EventHandler.register((InitializationComplete,))
async def _(
    event: "InitializationComplete",
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    hydrated_count = int(getattr(cm, "_hydrated_history_count", 0) or 0)
    cm.notifications_bar.push_notif(
        "System",
        (
            INITIALIZATION_COMPLETE_NOTIFICATION
            if hydrated_count
            else INITIALIZATION_COMPLETE_NO_HISTORY_NOTIFICATION
        ),
        event.timestamp,
    )
    cm._session_logger.debug("initialization", "Initialization complete")
    await cm.request_llm_run(delay=0)


@EventHandler.register(DirectMessageEvent)
async def _(event: DirectMessageEvent, cm: "ConversationManager", *args, **kwargs):
    """A message delivered to the user verbatim is still part of the chat."""
    cm._session_logger.info(
        "direct_message",
        f"Direct message: {event.content[:50]}...",
    )

    cm.chat_history.append(
        role="assistant",
        content=event.content,
        timestamp=event.timestamp,
    )
