"""
ConversationManager lifecycle helpers.

``start_async`` runs one ConversationManager in the current process over the
in-memory event broker; ``stop_async`` retires it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

LOGGER = logging.getLogger("unify")

if TYPE_CHECKING:
    from unify.legacy.conversation_manager.conversation_manager import (
        ConversationManager,
    )

_conversation_manager: Optional["ConversationManager"] = None


async def start_async() -> "ConversationManager":
    """Start the ConversationManager in-process and return it.

    A second call while one is running returns the running instance.
    """
    global _conversation_manager

    if _conversation_manager is not None:
        from unify.common.hierarchical_logger import ICONS

        LOGGER.debug(f"{ICONS['lifecycle']} ConversationManager is already running")
        return _conversation_manager

    # Import here to avoid circular imports
    from unify.legacy.conversation_manager.main import run_conversation_manager

    _conversation_manager = await run_conversation_manager()

    return _conversation_manager


async def stop_async(reason: str = "manual_stop") -> None:
    """
    Stop the ConversationManager.

    An explicit stop is a retirement: it goes through the ``_request_shutdown``
    sequence (record the reason, log ``session_end``, set ``stop``, close the
    event broker), then runs ``cleanup()`` — which discards in-flight actions
    rather than waiting on them. The whole sequence completes in seconds so
    an in-process successor can boot over the same durable world immediately.

    Args:
        reason: Reason for stopping (recorded as the shutdown reason)
    """
    global _conversation_manager

    if _conversation_manager is None:
        return

    from unify.common.hierarchical_logger import ICONS

    LOGGER.debug(
        f"{ICONS['lifecycle']} Stopping ConversationManager (reason: {reason})...",
    )

    try:
        if _conversation_manager.shutdown_reason is None:
            await _conversation_manager._request_shutdown(
                reason,
                f"Explicit shutdown requested ({reason})",
            )
        else:
            # An internal exit already ran the retirement sequence; don't
            # overwrite its recorded reason.
            _conversation_manager.stop.set()

        await _conversation_manager.cleanup()

        LOGGER.debug(f"{ICONS['lifecycle']} ConversationManager stopped")
    except Exception as e:
        LOGGER.error(f"{ICONS['lifecycle']} Error stopping ConversationManager: {e}")
    finally:
        # A successor must not inherit the retired session's machinery: the
        # ConversationManager is a registry singleton (handing it back gives
        # the next boot a session whose stop event is already set), and the
        # broker singleton was just closed by the retirement sequence (a next
        # boot that received it would publish into a void).
        from unify.legacy.conversation_manager.event_broker import reset_event_broker
        from unify.manager_registry import ManagerRegistry

        ManagerRegistry.deregister_instance(type(_conversation_manager))
        reset_event_broker()
        _conversation_manager = None


def get_conversation_manager() -> Optional["ConversationManager"]:
    """The running ConversationManager, or None."""
    return _conversation_manager


def reset() -> None:
    """Forget the running instance without stopping it (for tests)."""
    global _conversation_manager
    _conversation_manager = None
