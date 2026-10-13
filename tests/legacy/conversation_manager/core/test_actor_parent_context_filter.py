"""Actor parent-context filtering for CM-only tool surfaces."""

from __future__ import annotations

from unify.legacy.conversation_manager.domains.brain_action_tools import (
    _filter_cm_state_for_actor,
)


def test_filter_cm_state_strips_completed_action_steering_tools() -> None:
    snapshot = {
        "content": (
            "<completed_actions>\n"
            "<action id='3' short_name='search_web' status='completed' type='act'>\n"
            "<original_request>Search the web for X</original_request>\n"
            "<result>Found three sources.</result>\n"
            "<history>\n"
            "  - act_completed: Found three sources.\n"
            "</history>\n"
            "<steering_tools>\n"
            "  - ask_action(handle_id=3, ...): Ask about this completed action\n"
            "</steering_tools>\n"
            "</action>\n"
            "</completed_actions>\n"
            "<active_conversations>\n"
            "User: please continue\n"
            "</active_conversations>"
        ),
    }
    filtered = _filter_cm_state_for_actor(snapshot)
    content = filtered["content"]
    assert "<steering_tools>" not in content
    assert "ask_action(handle_id=3" not in content
    assert "<result>Found three sources.</result>" in content
    assert "<original_request>Search the web for X</original_request>" in content
    assert "User: please continue" in content


def test_filter_cm_state_strips_in_flight_actions_pane() -> None:
    snapshot = {
        "role": "user",
        "content": (
            "<notifications>\n</notifications>\n\n"
            "<in_flight_actions>\n"
            "<action id='1' short_name='search_web' status='executing' type='act'>\n"
            "<original_request>Search the web for X</original_request>\n"
            "<steering_tools>\n"
            "  - stop_action(handle_id=1, ...): Stop this task\n"
            "</steering_tools>\n"
            "</action>\n"
            "</in_flight_actions>\n\n"
            "<active_conversations>\n"
            "User: please continue\n"
            "</active_conversations>"
        ),
        "_cm_state_snapshot": True,
    }
    filtered = _filter_cm_state_for_actor(snapshot)
    content = filtered["content"]
    assert "<in_flight_actions>" not in content
    assert "stop_action(handle_id=1" not in content
    assert "<notifications>" in content
    assert "User: please continue" in content
    assert filtered["_cm_state_snapshot"] is True
    assert snapshot["content"].startswith("<notifications>")


def test_filter_cm_state_passes_empty_snapshot_through() -> None:
    assert _filter_cm_state_for_actor({}) == {}
    assert _filter_cm_state_for_actor({"content": ""}) == {"content": ""}
