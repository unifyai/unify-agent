import asyncio
from time import perf_counter
from typing import TYPE_CHECKING

import unify

from unify.logger import LOGGER
from unify.common.startup_timing import log_startup_timing
from unify.common.hierarchical_logger import DEFAULT_ICON, ICONS
from unify.settings import SETTINGS
from unify.session_details import SESSION_DETAILS
from unify.common.async_tool_loop import SteerableToolHandle
from unify.conversation_manager.event_broker import get_event_broker
from unify.conversation_manager.events import (
    ActorClarificationRequest,
    ActorNotification,
    ActorResult,
    ActorSessionResponse,
    InitializationComplete,
)
from unify.events.event_bus import EVENT_BUS
from unify.manager_registry import ManagerRegistry

if TYPE_CHECKING:
    from unify.actor.base import BaseActor
    from unify.conversation_manager.conversation_manager import ConversationManager

event_broker = get_event_broker()


# CHAT HISTORY


async def run_boot_hydration(cm: "ConversationManager") -> int:
    """Load the stored chat history, then reopen the slow-brain render gate.

    The gate must reopen on every outcome — restored history, an empty
    table, or a failed read — because a turn held at the gate degrades to
    the pre-hydration view after its bounded wait anyway; keeping the gate
    closed past hydration buys nothing but latency.
    """
    try:
        return await hydrate_chat_history(cm)
    finally:
        cm._hydration_gate.set()


async def hydrate_chat_history(cm: "ConversationManager") -> int:
    """Prepend the messages earlier sessions stored in the chat table.

    Messages that arrived during initialization keep their chronological
    position at the end. Returns the number of messages restored — zero when
    there is no prior conversation. The caller records it so the
    initialization-complete notification can tell the brain the truth about
    what was loaded.
    """
    _t0 = perf_counter()
    restored = await asyncio.to_thread(cm.chat_history.load)
    log_startup_timing(
        LOGGER,
        "⏱️ [StartupTiming] managers.hydrate_chat_history duration=%.2fs restored=%d",
        perf_counter() - _t0,
        restored,
    )
    LOGGER.info(
        f"{ICONS['managers_worker']} [Hydration] Restored {restored} chat messages",
    )
    return restored


async def publish_bus_events(event):
    try:
        event_name = event.__class__.__name__
        bus_event = event.to_bus_event()
        LOGGER.debug(f"{DEFAULT_ICON} Publishing bus event {event_name}")
        await EVENT_BUS.publish(bus_event)
    except Exception as e:
        LOGGER.error(
            f"{ICONS['managers_worker']} [ManagersWorker] Error publishing bus event: {e}",
        )


# ACTOR
async def actor_watch_result(
    handle_id: int,
    handle: SteerableToolHandle,
    *,
    action_type: str = "",
) -> None:
    """Await final result and publish completion (or failure), then cleanup."""
    resolved_action_type = action_type or "act"
    try:
        result = await handle.result()
    except Exception as exc:
        error_text = f"Error getting actor result: {exc}"
        LOGGER.error(f"{ICONS['managers_worker']} [ManagersWorker] {error_text}")
        await event_broker.publish(
            "app:actor:result",
            ActorResult(
                handle_id=handle_id,
                success=False,
                result=None,
                error=error_text,
                action_type=resolved_action_type,
            ).to_json(),
        )
        return

    success = True
    error_text: str | None = None
    if isinstance(result, dict) and "error_kind" in result:
        success = False
        error_text = str(result.get("message") or result.get("error_kind"))
    elif isinstance(result, str):
        stripped_result = result.lstrip()
        if stripped_result[:5].lower() == "error":
            success = False
            error_text = result

    await event_broker.publish(
        "app:actor:result",
        ActorResult(
            handle_id=handle_id,
            success=success,
            result=result,
            error=error_text,
            action_type=resolved_action_type,
        ).to_json(),
    )


async def actor_watch_notifications(
    handle_id: int,
    handle: SteerableToolHandle,
) -> None:
    """Forward notifications and responses from the handle until it completes.

    The handle's notification queue carries two kinds of messages:

    - **``type="notification"``** — progress updates emitted by the
      ``send_notification`` tool (or the active-work heartbeat) while the
      actor is still working.
    - **``type="response"``** — turn-complete signals emitted when a
      persistent session enters its wait state. These mean the actor has
      finished the current turn and is awaiting the next ``interject``.

    Each type is published as a distinct CM event so the brain can tell
    them apart.

    Drain guarantee: a notification enqueued in the instant before
    ``handle.done()`` flips to ``True`` must still be observed -- the
    producer side always enqueues before signalling completion, but the
    ``while not handle.done()`` check races that signal. Once ``done()``
    is observed ``True``, keep polling with a short timeout (instead of
    exiting immediately) until the queue is actually empty.
    """
    _DRAIN_TIMEOUT = 0.1

    while True:
        already_done = handle.done()
        try:
            notif = await asyncio.wait_for(
                handle.next_notification(),
                timeout=_DRAIN_TIMEOUT if already_done else 30,
            )
        except asyncio.TimeoutError:
            if already_done:
                break
            continue

        # An empty payload means "nothing to report", never a real
        # notification — treat it as end-of-stream once the handle is done.
        # The explicit yield guards against a handle whose
        # ``next_notification`` completes synchronously: ``wait_for`` awaits
        # the coroutine inline, so without it this loop would never suspend
        # and would starve the entire event loop.
        if not notif:
            if already_done:
                break
            await asyncio.sleep(0)
            continue

        # Determine whether this is a turn-complete response or a progress
        # notification. The loop emits responses with {"type": "response", ...}.
        is_response = isinstance(notif, dict) and notif.get("type") == "response"

        if is_response:
            content = str(notif.get("content", ""))
            await event_broker.publish(
                "app:actor:session_response",
                ActorSessionResponse(
                    handle_id=handle_id,
                    content=content,
                ).to_json(),
            )
        else:
            # Extract a human-friendly message.
            #
            # Contract:
            # - Notifications may be plain strings (already display-ready), OR
            # - Structured dict payloads (recommended: include both "type" and "message").
            #
            # Fallback chain: "message" → "result_summary" → "type" → JSON dump.
            # "result_summary" is checked before "type" because step_complete
            # payloads carry their useful content in that field, not "message".
            msg: str
            if isinstance(notif, dict):
                if notif.get("message") is not None:
                    msg = str(notif.get("message"))
                elif notif.get("result_summary") is not None:
                    msg = str(notif.get("result_summary"))
                elif notif.get("type") is not None:
                    msg = str(notif.get("type"))
                else:
                    try:
                        import json as _json

                        msg = _json.dumps(notif, ensure_ascii=False, default=str)
                    except Exception:
                        msg = str(notif)
            else:
                msg = str(notif)

            completed = (
                bool(notif.get("completed", False))
                if isinstance(notif, dict)
                else False
            )
            kind = notif.get("type", "") if isinstance(notif, dict) else ""
            await event_broker.publish(
                "app:actor:notification",
                ActorNotification(
                    handle_id=handle_id,
                    response=msg,
                    completed=completed,
                    kind=kind,
                ).to_json(),
            )


async def actor_watch_clarifications(
    handle_id: int,
    handle: SteerableToolHandle,
) -> None:
    """Forward clarifications to CM until handle completes."""
    while not handle.done():
        # await clarification request
        try:
            clar = await asyncio.wait_for(handle.next_clarification(), timeout=30)
        except asyncio.TimeoutError:
            continue

        # get question and call id
        q = clar.get("question") if isinstance(clar, dict) else str(clar)
        call_id = clar.get("call_id") if isinstance(clar, dict) else None

        # publish clarification request
        await event_broker.publish(
            "app:actor:clarification_request",
            ActorClarificationRequest(
                handle_id=handle_id,
                query=q,
                call_id=call_id,
            ).to_json(),
        )


# Queueing operations that need managers

_operations_queue = asyncio.Queue()
_module_loop: asyncio.AbstractEventLoop | None = None


def _adopt_running_loop() -> None:
    """Recreate module-level asyncio primitives when their owning loop died.

    The queue and init lock are process-global, but they bind to the event
    loop that first awaits them. An in-process reboot — a fresh
    ConversationManager on a new loop over the same durable world — must not
    inherit them: a bound queue makes the successor's operations listener die
    on its first ``get`` (RuntimeError: bound to a different event loop),
    after which every queued operation — bus publishes among them —
    silently accumulates unprocessed; a lock held by a task frozen on the
    dead loop blocks the successor's init outright. The predecessor's queued
    operations die with it, exactly as they would with its process. A live
    owning loop is never preempted.
    """
    global _operations_queue, _init_lock, _module_loop
    loop = asyncio.get_running_loop()
    owner = _module_loop
    if owner is loop:
        return
    if owner is not None and owner.is_running() and not owner.is_closed():
        return
    if owner is not None:
        _operations_queue = asyncio.Queue()
        _init_lock = asyncio.Lock()
    _module_loop = loop


# A backlog this deep means nothing is consuming the queue: the listener
# normally drains an operation in milliseconds, and producers enqueue one
# per handled event. Growth past this line is a missing/dead
# listen_to_operations, not load.
_OPERATIONS_QUEUE_BACKLOG_WARN_AT = 50
_operations_backlog_warned = False


async def queue_operation(async_func: callable, *args, **kwargs) -> None:
    """
    Queue an async operation to be executed when managers are initialized.
    The operation will be processed by listen_to_operations().
    """
    global _operations_backlog_warned
    _adopt_running_loop()
    await _operations_queue.put((async_func, args, kwargs))
    # An unbounded queue with no consumer fails silently: every enqueued
    # operation — bus publishes among them — simply never happens,
    # and nothing anywhere says so. Embedders that boot the CM without
    # spawning listen_to_operations have lost whole conversation streams
    # this way. One warning per backlog episode turns that black hole into
    # a grep-able line.
    depth = _operations_queue.qsize()
    if depth >= _OPERATIONS_QUEUE_BACKLOG_WARN_AT:
        if not _operations_backlog_warned:
            _operations_backlog_warned = True
            LOGGER.warning(
                f"{ICONS['managers_worker']} [ManagersWorker] Operations "
                f"queue backlog reached {depth} with nothing draining it — "
                "is listen_to_operations running? Queued work (bus "
                "publishes among it) is not being executed.",
            )
    elif depth < _OPERATIONS_QUEUE_BACKLOG_WARN_AT // 2:
        _operations_backlog_warned = False


async def wait_for_initialization(
    cm: "ConversationManager",
) -> None:
    """
    Wait for initialization to complete.

    Polls cm.initialized with no timeout. Initialization failures are
    surfaced by init_conv_manager itself (logged errors). A timeout here
    would silently kill the operations queue processor on slow cold starts,
    causing queued work to be orphaned.
    """
    while not cm.initialized:
        await asyncio.sleep(0.1)


async def listen_to_operations(cm: "ConversationManager") -> None:
    """
    Worker loop that processes queued operations once initialized.
    Should be started as a background task alongside init_conv_manager.
    """
    _adopt_running_loop()
    # Wait for initialization to complete
    await wait_for_initialization(cm)

    LOGGER.info(
        f"{ICONS['managers_worker']} [ManagersWorker] Operations listener started, processing queue...",
    )

    # Process operations as they come in
    while True:
        # Wait for next operation (with timeout to allow checking for shutdown)
        try:
            async_func, args, kwargs = await asyncio.wait_for(
                _operations_queue.get(),
                timeout=1.0,
            )
        except asyncio.TimeoutError:
            continue

        # Execute the operation
        func_name = getattr(async_func, "__name__", str(async_func))
        try:
            await async_func(*args, **kwargs)
        except Exception as e:
            LOGGER.error(
                f"{ICONS['managers_worker']} [ManagersWorker] Error executing {func_name}: {e}",
            )
        finally:
            _operations_queue.task_done()


# Initialization

_init_lock = asyncio.Lock()


def _init_managers(
    cm: "ConversationManager",
    loop: asyncio.AbstractEventLoop,
    actor: "BaseActor | None" = None,
) -> None:
    """
    Initialize the runtime, the chat table, the handle and the actor in a
    separate thread. The main event loop is passed for work that must be
    scheduled back onto it.

    Args:
        cm: The ConversationManager instance to initialize.
        loop: The main event loop for scheduling async tasks.
        actor: Optional pre-instantiated Actor. If provided, used directly instead
            of creating one via ManagerRegistry. Useful for testing with specific
            Actor implementations.
    """
    start_time = perf_counter()

    # 0. Initialise the runtime (idempotent).
    LOGGER.debug(f"{ICONS['managers_worker']} [ManagersWorker] Initializing unify...")
    local_start_time = perf_counter()
    unify.init()
    LOGGER.info(
        f"{ICONS['managers_worker']} [ManagersWorker] Runtime initialized in "
        f"{perf_counter() - local_start_time:.2f} seconds",
    )

    # 1. Bind the chat table so every message from here on is written
    #    through, then hydrate the stored history concurrently on the main
    #    loop while the rest of init continues in this thread. Completion
    #    reopens the slow-brain render gate ``init_conv_manager`` closed.
    local_start_time = perf_counter()
    cm.chat_history.bind()
    LOGGER.info(
        f"{ICONS['managers_worker']} [ManagersWorker] Chat table bound in "
        f"{perf_counter() - local_start_time:.2f} seconds",
    )
    cm._hydration_future = asyncio.run_coroutine_threadsafe(
        run_boot_hydration(cm),
        loop,
    )

    # 2. Initialize ConversationManagerHandle (respects SETTINGS.conversation.IMPL)
    LOGGER.info(
        f"{ICONS['managers_worker']} [ManagersWorker] Initializing ConversationManagerHandle...",
    )
    local_start_time = perf_counter()
    if SETTINGS.conversation.IMPL == "simulated":
        cm._conversation_manager_handle = (
            ManagerRegistry.get_conversation_manager_handle(
                description="production deployment",
                assistant_id=SESSION_DETAILS.assistant.agent_id,
            )
        )
    else:
        cm._conversation_manager_handle = (
            ManagerRegistry.get_conversation_manager_handle(
                event_broker=cm.event_broker,
                conversation_manager=cm,
            )
        )
    LOGGER.info(
        f"{ICONS['managers_worker']} [ManagersWorker] ConversationManagerHandle ({type(cm._conversation_manager_handle).__name__}) initialized in "
        f"{perf_counter() - local_start_time:.2f} seconds",
    )

    # 3. Initialize Actor (use provided actor or create via ManagerRegistry)
    LOGGER.debug(f"{ICONS['managers_worker']} [ManagersWorker] Initializing Actor...")
    try:
        local_start_time = perf_counter()
        if actor is not None:
            # Use pre-instantiated actor (e.g., for testing)
            cm.actor = actor
        else:
            # Create via ManagerRegistry (respects SETTINGS.actor.IMPL)
            from unify.actor.environments.actor import top_level_environments

            cm.actor = ManagerRegistry.get_actor(
                description="production deployment",
                environments=top_level_environments(),
            )
        actor_cls = type(cm.actor).__name__
        LOGGER.info(
            f"{ICONS['managers_worker']} [ManagersWorker] Actor ({actor_cls}) initialized in "
            f"{perf_counter() - local_start_time:.2f} seconds",
        )
    except Exception as e:
        LOGGER.error(
            f"{ICONS['managers_worker']} [ManagersWorker] Error initializing Actor: {e}",
        )

    LOGGER.info(
        f"{ICONS['managers_worker']} [ManagersWorker] All managers initialized in "
        f"{perf_counter() - start_time:.2f} seconds",
    )


async def init_conv_manager(
    cm: "ConversationManager",
    *,
    actor: "BaseActor | None" = None,
) -> None:
    """
    Initialize all managers for the ConversationManager.
    All initialization runs in a separate thread (non-blocking).

    Args:
        cm: The ConversationManager instance to initialize.
        actor: Optional pre-instantiated Actor. If provided, used directly instead
            of creating one via ManagerRegistry. Useful for testing with specific
            Actor implementations (e.g., SimulatedActor).
    """
    LOGGER.debug(f"{ICONS['managers_worker']} [ManagersWorker] Processing startup")

    _adopt_running_loop()
    async with _init_lock:
        start_time = perf_counter()
        if cm.initialized:
            LOGGER.info(
                f"{ICONS['managers_worker']} [ManagersWorker] Already initialized, skipping",
            )
            return

        # Close the slow-brain render gate for the boot window. An inbound
        # that lands mid-boot still queues a turn, but that turn holds at
        # render time until hydration resolves (see
        # ``ConversationManager._run_llm``), so the first reply after a wake
        # never answers from an empty conversation view while the history
        # that would answer it is seconds from landing. ``run_boot_hydration``
        # reopens it; the ``finally`` below covers boots that die before the
        # hydration task ever gets created.
        cm._hydration_gate.clear()

        try:
            # Get the main event loop to pass to managers that need it
            loop = asyncio.get_running_loop()

            _t0 = perf_counter()
            await asyncio.to_thread(_init_managers, cm, loop, actor)
            log_startup_timing(
                LOGGER,
                "⏱️ [StartupTiming] managers.init_conv_manager.to_thread duration=%.2fs",
                perf_counter() - _t0,
            )

            cm.initialized = True

            # Await the concurrent hydration that was kicked off inside
            # _init_managers right after the chat table was bound. In
            # practice it finishes long before this point, so this is
            # effectively a no-op await.
            hydration_future = getattr(cm, "_hydration_future", None)
            if hydration_future is not None:
                try:
                    _t0 = perf_counter()
                    # The count feeds the initialization-complete notification:
                    # "history has been loaded" is only said when it is true.
                    cm._hydrated_history_count = int(
                        await asyncio.wrap_future(hydration_future) or 0,
                    )
                    log_startup_timing(
                        LOGGER,
                        "⏱️ [StartupTiming] managers.init_conv_manager.await_hydration duration=%.2fs",
                        perf_counter() - _t0,
                    )
                    LOGGER.info(
                        f"{ICONS['managers_worker']} [ManagersWorker] "
                        "Concurrent hydration completed",
                    )
                except Exception as e:
                    LOGGER.error(
                        f"{ICONS['managers_worker']} [ManagersWorker] "
                        f"Chat history hydration failed: {e}",
                    )
                    import traceback

                    traceback.print_exc()
                finally:
                    cm._hydration_future = None

            # Publish initialization complete event.  The registered
            # InitializationComplete handler pushes a notification and
            # triggers a brain turn so it can follow up on deferred requests.
            _t0 = perf_counter()
            await event_broker.publish(
                "app:comms:initialization_complete",
                InitializationComplete().to_json(),
            )
            log_startup_timing(
                LOGGER,
                "⏱️ [StartupTiming] managers.init_conv_manager.publish_initialization_complete duration=%.2fs",
                perf_counter() - _t0,
            )

            LOGGER.info(
                f"{ICONS['managers_worker']} [ManagersWorker] Initialization complete in "
                f"{perf_counter() - start_time:.2f} seconds",
            )

        except Exception as e:
            LOGGER.error(
                f"{ICONS['managers_worker']} [ManagersWorker] Error during initialization: {e}",
            )
            raise
        finally:
            # Idempotent: hydration completion normally reopened this long
            # ago. Guarantees no path out of a boot — success or failure —
            # leaves brain turns holding until their timeout.
            cm._hydration_gate.set()
