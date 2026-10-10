"""
tests/conversation_manager/core/test_init_resilience.py
=======================================================

Symbolic tests verifying that failures in non-essential initialization steps
degrade gracefully instead of preventing the session from becoming operational.

Each test mocks a single degradable subsystem to raise during init, then
asserts that ``cm.initialized`` still becomes ``True`` — proving the session
would survive and serve requests (with reduced capability) rather than
becoming a zombie.
"""

import pytest
import pytest_asyncio
from unittest.mock import patch

from tests.helpers import scenario_file_lock

# ---------------------------------------------------------------------------
# Fixture: lightweight CM factory for resilience tests
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def resilience_cm():
    """Create a ConversationManager, run init_conv_manager, and tear down.

    Yields the CM *before* init so individual tests can apply mocks around
    the ``init_conv_manager`` call.
    """
    from unify.legacy.conversation_manager.event_broker import reset_event_broker
    from unify.legacy.conversation_manager import start_async, stop_async

    reset_event_broker()

    cm = await start_async()

    yield cm

    await stop_async()
    reset_event_broker()


def _simulated_actor():
    from unify.actor.simulated import SimulatedActor

    return SimulatedActor(
        steps=None,
        duration=None,
        log_mode="log",
        emit_notifications=False,
    )


async def _init(cm, lock_name="init_resilience", actor=None):
    """Helper: run manager init with a SimulatedActor under file lock."""
    from unify.legacy.conversation_manager.domains import managers_utils

    if actor is None:
        actor = _simulated_actor()
    with scenario_file_lock(lock_name):
        await managers_utils.init_conv_manager(cm, actor=actor)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDegradableStepResilience:
    """Failures in optional init steps must not prevent initialization."""

    @pytest.mark.asyncio
    async def test_guidance_manager_init_failure(self, resilience_cm):
        cm = resilience_cm
        # The simulated actor resolves its own guidance manager, so it is
        # built before the registry is made to fail.
        actor = _simulated_actor()
        with patch(
            "unify.legacy.conversation_manager.domains.managers_utils.ManagerRegistry.get_guidance_manager",
            side_effect=ConnectionError("store unreachable"),
        ):
            await _init(cm, "resilience_guidance", actor=actor)

        assert cm.initialized is True
