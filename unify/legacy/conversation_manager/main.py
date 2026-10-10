"""Boot a ConversationManager in the current process."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from unify.logger import LOGGER
from unify.common.hierarchical_logger import ICONS
from unify.session_details import SESSION_DETAILS
from unify.workspace import get_local_root
from unify.legacy.conversation_manager.event_broker import get_event_broker
from unify.legacy.conversation_manager.domains import managers_utils
from unify.legacy.conversation_manager.domains.utils import log_task_exc
from unify.legacy.conversation_manager.conversation_manager import ConversationManager
from unify.legacy.conversation_manager.in_memory_event_broker import InMemoryEventBroker


async def run_conversation_manager(
    *,
    event_broker: InMemoryEventBroker | None = None,
    stop_event: asyncio.Event | None = None,
) -> ConversationManager:
    """Start a ConversationManager and its background tasks; return it running.

    Args:
        event_broker: The broker to listen on. Defaults to the process singleton.
        stop_event: Set it to shut the manager down. Defaults to a fresh event.

    Returns:
        The running ConversationManager. Call ``cm.stop.set()`` to trigger
        shutdown, or ``await cm.cleanup()`` when done.
    """
    SESSION_DETAILS.populate_from_env()

    # Set the process working directory to the workspace root so that relative
    # file paths in CodeActActor-generated code (e.g. "Attachments/abc123_report.pdf")
    # and on chat attachments resolve against the same root.  This must
    # happen after settings/env are loaded but before any concurrent tasks are
    # created, since os.chdir() is process-global.
    local_root = Path(get_local_root())
    local_root.mkdir(parents=True, exist_ok=True)
    os.chdir(local_root)

    if event_broker is None:
        event_broker = get_event_broker()
    if stop_event is None:
        stop_event = asyncio.Event()

    cm = ConversationManager(event_broker, stop_event)

    asyncio.create_task(cm.wait_for_events()).add_done_callback(log_task_exc)
    asyncio.create_task(managers_utils.init_conv_manager(cm))
    asyncio.create_task(managers_utils.listen_to_operations(cm))

    LOGGER.debug(f"{ICONS['lifecycle']} ConversationManager is running...")
    return cm
