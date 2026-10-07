"""Symbolic: the legacy conversation manager under the test configuration
(moved from tests/test_test_configuration.py): its steering pause tests always
call the model, and its prompts show the frozen clock.
"""

from __future__ import annotations

import pytest

# Imported here, before any test's fixtures run, so these modules bind the
# real prompt clock by name the way a test session's first import does.
import unify.legacy.conversation_manager.conversation_manager  # noqa: F401
import unify.legacy.conversation_manager.domains.brain_action_tools as brain_action_tools
from unify.legacy.conversation_manager.domains.renderer import Renderer

FROZEN_PROMPT_TIME = "Friday, June 13, 2025 at 12:00 PM UTC"


@pytest.mark.parametrize(
    "name",
    [
        "test_pause_resume_inflight_handle",
        "test_two_concurrent_handles_pause_one_other_completes",
    ],
)
def test_the_steering_pause_tests_always_call_the_model(name):
    from tests.legacy.conversation_manager.actions.integration import (
        test_steerability,
    )

    marks = {m.name for m in getattr(test_steerability, name).pytestmark}
    assert "fresh_llm_calls" in marks


def test_the_brain_prompt_clock_is_frozen():
    assert brain_action_tools.prompt_now() == FROZEN_PROMPT_TIME


def test_an_action_history_event_shows_the_frozen_time():
    history = Renderer._render_action_history(
        [
            {
                "action_name": "act_started",
                "query": "find the report",
                "timestamp": brain_action_tools.prompt_now(),
            },
        ],
        short_name="act",
        handle_id=0,
        max_history=5,
    )
    assert f"<event type='act_started' timestamp='{FROZEN_PROMPT_TIME}'>" in history
