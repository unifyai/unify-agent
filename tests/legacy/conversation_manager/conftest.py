"""
tests/conversation_manager/conftest.py
==============================================

Fixtures for conversation manager integration tests.

Uses **direct handler testing** pattern:
- No event-driven initialization (no background task dependencies)
- Direct calls to event handlers via CMStepDriver
- Direct state inspection
- Works reliably with pytest-asyncio

The actor is simulated so the only real LLM calls are the slow brain's own.
"""

from __future__ import annotations

import os

# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL: environment-variable setup MUST happen at module top, *before*
# any `tests.helpers` import (which transitively imports unify modules that
# instantiate `SETTINGS = ProductionSettings()` at import time).
# Pydantic's BaseSettings reads env vars once at instantiation; if SETTINGS
# is already constructed when pytest_configure() later sets these vars, the
# overrides are silently ignored and prompts/feature flags fall back to
# production defaults.
# ─────────────────────────────────────────────────────────────────────────────
from tests.legacy.conversation_manager.assistant_identity_env import (
    ensure_test_assistant_identity_env,
)

ensure_test_assistant_identity_env()

import pytest
import pytest_asyncio

from tests.helpers import scenario_file_lock
from .cm_test_driver import CMStepDriver

# =============================================================================
# Module-level setup: Configure environment for in-process mode
# =============================================================================


def pytest_configure(config):
    """Configure environment variables before any tests run."""
    os.environ["UNIFY_ACTOR_IMPL"] = "simulated"

    # Enable incrementing timestamps for **NEW** marker comparisons
    os.environ["UNIFY_INCREMENTING_TIMESTAMPS"] = "true"

    ensure_test_assistant_identity_env()


# =============================================================================
# ConversationManager Fixtures (Direct Handler Testing)
# =============================================================================


@pytest_asyncio.fixture(scope="module")
async def conversation_manager(request) -> CMStepDriver:
    """
    Start and initialize ConversationManager in-process for the test module.

    Uses DIRECT initialization (not event-driven) to avoid background task
    issues with pytest-asyncio.

    Uses SimulatedActor explicitly for fast, deterministic testing.

    Returns a CMStepDriver that wraps the CM and provides step() and
    step_until_wait() methods for deterministic testing.
    """
    from unify.actor.simulated import SimulatedActor
    from unify.legacy.conversation_manager.event_broker import reset_event_broker
    from unify.legacy.conversation_manager import start_async, stop_async
    from unify.legacy.conversation_manager.domains import managers_utils

    # Reset any existing event broker state
    reset_event_broker()

    print("\n✓ Starting ConversationManager in-process...")
    cm = await start_async()
    print("✓ ConversationManager started (in-process mode)")
    print("  Using SimulatedActor for deterministic testing")

    # Create SimulatedActor for fast, deterministic testing.
    #
    # Uses steps=None, duration=None so actions run indefinitely until explicitly
    # completed via trigger_completion() in test cleanup. This makes tests fully
    # deterministic with no timing dependencies. Tests that verify steering (pause,
    # resume, stop, interject) just check that steering tools were called - they
    # don't need actions to auto-complete based on step counts.
    actor = SimulatedActor(
        steps=None,
        duration=None,
        log_mode="log",
        emit_notifications=False,
    )

    with scenario_file_lock("cm_conversation_manager"):
        # Initialize managers DIRECTLY (not via event handler)
        # This avoids the background task / event loop interleaving issues
        print("⏳ Initializing managers directly...")
        await managers_utils.init_conv_manager(cm, actor=actor)
        if not cm.initialized:
            raise RuntimeError(
                "ConversationManager managers failed to initialize - check logs for errors",
            )
        print("✅ Managers initialized")

    # Wrap in CMStepDriver for deterministic testing
    driver = CMStepDriver(cm)

    yield driver

    # Cleanup
    print("\n✓ Stopping ConversationManager...")
    await stop_async()
    reset_event_broker()


def _complete_in_flight_actions(cm: "CMStepDriver") -> None:
    """
    Complete all in-flight actions to unblock watcher threads.

    With steps=None, actions run indefinitely. The watcher tasks (actor_watch_result,
    actor_watch_notifications, actor_watch_clarifications) block on _done_event.wait()
    in thread pool threads. If we don't call trigger_completion(), these threads
    never terminate and cause "executor did not finish joining" warnings at shutdown.
    """
    for handle_data in list(cm.cm.in_flight_actions.values()):
        handle = handle_data.get("handle")
        if handle and hasattr(handle, "trigger_completion"):
            handle.trigger_completion()
    cm.cm.in_flight_actions.clear()
    cm.cm.completed_actions.clear()


@pytest.fixture
def initialized_cm(
    conversation_manager: CMStepDriver,
):
    """
    Per-test fixture that provides a clean ConversationManager.

    Clears conversation state between tests for isolation while reusing
    the expensive module-scoped CM instance. Also ensures in-flight actions
    are completed after each test to prevent thread leaks.
    """
    # Complete and clear in-flight actions from previous tests
    _complete_in_flight_actions(conversation_manager)

    # Clear the conversation from previous tests (in memory only; the
    # per-module store keeps its rows, which nothing reads back mid-test).
    conversation_manager.cm.chat_history.clear()

    # Reset handle_id counter to ensure deterministic action ids for caching.
    # Without this, handle_ids increment across tests, changing the rendered
    # ``<action id='N'>`` panes and breaking LLM cache hits.
    import unify.legacy.conversation_manager.domains.brain_action_tools as bat

    bat._next_handle_id = 0

    # Clear the brain's own LLM message list
    conversation_manager.cm.brain_messages.clear()

    # The recent-tool-executions pane would otherwise show the previous test's
    # actions as work already in progress.
    conversation_manager.cm._recent_tool_executions.clear()

    # Clear tool call tracking from previous tests
    conversation_manager.all_tool_calls.clear()

    # Reset last_snapshot to use the patched prompt_now.
    # The module-scoped conversation_manager fixture is created BEFORE the
    # function-scoped stub_external_deps fixture patches prompt_now, so
    # cm.last_snapshot gets set to real time while message timestamps use the
    # patched fixed time. This breaks the **NEW** marker comparison
    # (last_snapshot < message.timestamp). Re-initializing here ensures
    # last_snapshot uses the patched timestamp.
    from unify.common.prompt_helpers import now as prompt_now

    conversation_manager.cm.last_snapshot = prompt_now(as_string=False)

    yield conversation_manager

    # Cleanup after test: complete any in-flight actions created during this test
    _complete_in_flight_actions(conversation_manager)
