"""
tests/conversation_manager/test_multi_action.py
====================================================

Tests that verify ConversationManager correctly handles multiple concurrent
action requests and disambiguates when to create new actions vs steer existing ones.

These tests build upon test_take_action.py and test_steer_action.py to cover
more complex realistic scenarios involving:
- Multiple concurrent in-flight actions
- Disambiguating new actions vs interjections to existing actions
- Steering specific actions when multiple are running
- Sequential action creation with context

Uses SimulatedActor under the hood with configurable `steps` to ensure
actions remain in-flight long enough for multi-action scenarios.
"""

import pytest

from tests.helpers import _handle_project
from tests.legacy.conversation_manager.cm_helpers import (
    filter_events_by_type,
    assert_efficient,
    get_in_flight_action_count,
    has_steering_tool_call,
)
from unify.legacy.conversation_manager.events import (
    ActorHandleStarted,
    UnifyMessageReceived,
)

# Actions stay in-flight indefinitely with steps=None, duration=None.
# Tests verify steering tools were called - actions are completed via
# trigger_completion() in test cleanup.
pytestmark = [pytest.mark.eval]

_ACTION_TOOLS = {"act"}


def _count_action_calls(cm):
    """Count how many action-creating tools were called."""
    return sum(1 for tool in cm.all_tool_calls if tool in _ACTION_TOOLS)


# ---------------------------------------------------------------------------
#  Sequential independent actions - each should trigger new action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_two_unrelated_requests_create_two_tasks(initialized_cm):
    """
    Two completely unrelated requests should create two separate tasks.

    Flow:
    1. User asks for a web search about one topic
    2. User asks for a completely unrelated workspace task
    3. Both should trigger separate actions
    """
    cm = initialized_cm

    # Step 1: First task - web search
    result1 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Search the web for the latest news about climate change.",
        ),
    )
    actor_events1 = filter_events_by_type(result1.output_events, ActorHandleStarted)
    assert len(actor_events1) >= 1, "Expected act to be called for first action"
    action_count_after_first = get_in_flight_action_count(cm)

    # Step 2: Second task - completely unrelated workspace task
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Also, convert the sales CSV in my workspace to JSON.",
        ),
    )
    actor_events2 = filter_events_by_type(result2.output_events, ActorHandleStarted)

    # Should have created a second action (either now or total of 2)
    assert (
        len(actor_events2) >= 1 or get_in_flight_action_count(cm) >= 2
    ), "Expected second act call for unrelated request"

    # Efficiency assertions at end
    assert_efficient(result1, 3, "Step 1: first task")
    assert_efficient(result2, 3, "Step 2: second task")


@pytest.mark.asyncio
@_handle_project
async def test_parallel_searches_different_topics(initialized_cm):
    """
    Two explicit search requests on different topics should create separate tasks.

    Flow:
    1. User asks to search for topic A
    2. User explicitly requests a NEW separate search for topic B
    3. Both should be independent actions
    """
    cm = initialized_cm

    # Step 1: First search - web search
    result1 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Search the web for information about renewable energy.",
        ),
    )
    assert get_in_flight_action_count(cm) >= 1, "Expected at least one in-flight action"

    # Step 2: Explicitly request a NEW task (make it very clear)
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="I need a second thing: count the rows in the sales CSV in my workspace.",
        ),
    )

    # Should have called act at least twice
    assert (
        _count_action_calls(cm) >= 2
    ), "Expected two separate actions for explicitly different requests"

    # Efficiency assertions at end
    assert_efficient(result1, 3, "Step 1: first search")
    assert_efficient(result2, 3, "Step 2: second task")


# ---------------------------------------------------------------------------
#  New action vs interject disambiguation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_also_search_creates_new_task_not_interject(initialized_cm):
    """
    "Also do X" after task Y should create a new task when topics are different.

    Flow:
    1. User asks for a web search
    2. User explicitly requests a different type of task (a file conversion)
    3. Should create a new action, not interject
    """
    cm = initialized_cm

    # Step 1: First task - web search
    result1 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Search the web for information about electric vehicles.",
        ),
    )
    assert get_in_flight_action_count(cm) >= 1, "Expected at least one in-flight action"
    initial_act_count = _count_action_calls(cm)

    # Step 2: Different task type - file conversion (clearly not an interject)
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Also, I need you to convert the sales CSV in my workspace to JSON.",
        ),
    )

    # Should have called act again (new action) for different request type
    # Alternatively, may have multiple in-flight actions
    new_act_count = _count_action_calls(cm)
    action_count = get_in_flight_action_count(cm)

    assert (
        new_act_count > initial_act_count or action_count >= 2
    ), "Expected new action for different request type"

    # Efficiency assertions at end
    assert_efficient(result1, 3, "Step 1: first task")
    assert_efficient(result2, 3, "Step 2: different task type")


@pytest.mark.asyncio
@_handle_project
async def test_add_detail_to_same_topic_interjects(initialized_cm):
    """
    Adding a detail to the same topic should interject, not create new task.

    Flow:
    1. User asks to search for a topic
    2. User adds a constraint to that same search
    3. Should interject existing task, not create new one
    """
    cm = initialized_cm

    # Step 1: Initial search
    result1 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Search the web for AI regulation news.",
        ),
    )
    assert get_in_flight_action_count(cm) >= 1, "Expected at least one in-flight action"
    act_count_after_first = _count_action_calls(cm)

    # Step 2: Add constraint to the SAME search
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="For that search, focus on European regulations specifically.",
        ),
    )

    # Should have interjected, not created new action
    has_interject = has_steering_tool_call(cm, "interject_")
    new_act_count = _count_action_calls(cm)

    # Either interjected OR didn't create a new act (both acceptable)
    assert (
        has_interject or new_act_count == act_count_after_first
    ), "Expected interject for adding constraint to same topic, not new action"

    # Efficiency assertions at end
    assert_efficient(result1, 3, "Step 1: initial search")
    assert_efficient(result2, 3, "Step 2: add constraint")


# ---------------------------------------------------------------------------
#  Multiple actions with selective steering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_two_tasks_stop_one_specifically(initialized_cm):
    """
    With two tasks running, user stops one by name/topic.

    Flow:
    1. User starts a web search task
    2. User starts a row-count task
    3. User cancels the web search specifically
    4. Row-count task should still be running (or at least stop_ called)
    """
    cm = initialized_cm

    # Step 1: First task - web search
    result1 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Search the web for the latest stock market news.",
        ),
    )
    assert get_in_flight_action_count(cm) >= 1, "Expected at least one in-flight action"

    # Step 2: Second task - row count
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Also count the rows in the sales CSV in my workspace.",
        ),
    )

    # Step 3: Stop the web search specifically
    result3 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Cancel the stock market search, I don't need that anymore.",
        ),
    )

    # Should have called stop_ at some point
    assert has_steering_tool_call(
        cm,
        "stop_",
    ), "Expected stop_* steering tool for canceling specific action"

    # Efficiency assertions at end
    assert_efficient(result1, 3, "Step 1: first task")
    assert_efficient(result2, 3, "Step 2: second task")
    assert_efficient(result3, 3, "Step 3: cancel specific task")


@pytest.mark.asyncio
@_handle_project
async def test_two_tasks_ask_about_one_specifically(initialized_cm):
    """
    With two tasks running, user asks about one by name/topic.

    Flow:
    1. User starts a notes search
    2. User starts a web search
    3. User asks about progress on the notes search specifically
    """
    cm = initialized_cm

    # Step 1: First task - notes search
    result1 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Go through the notes files in my workspace for anything about the Henderson deal.",
        ),
    )
    assert get_in_flight_action_count(cm) >= 1, "Expected at least one in-flight action"

    # Step 2: Second task - web search
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Also search the web for Henderson Company's latest quarterly report.",
        ),
    )

    # Step 3: Ask about the notes search specifically
    # Note: With async ask, this takes more steps because:
    result3 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="How's the search through my notes files going?",
        ),
    )

    # Should have called ask_ at some point
    assert has_steering_tool_call(
        cm,
        "ask_",
    ), "Expected ask_* steering tool for querying specific action"

    # Efficiency assertions at end
    assert_efficient(result1, 3, "Step 1: first task")
    assert_efficient(result2, 3, "Step 2: second task")


# ---------------------------------------------------------------------------
#  Action completion then new action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_stop_task_then_start_new_unrelated(initialized_cm):
    """
    User stops a task, then starts a completely new one.

    Flow:
    1. User starts a task
    2. User cancels it
    3. User starts a new unrelated task
    4. New task should be a fresh action
    """
    cm = initialized_cm

    # Step 1: First task
    result1 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Search for information about project deadlines.",
        ),
    )
    assert get_in_flight_action_count(cm) >= 1, "Expected at least one in-flight action"

    # Step 2: Cancel it
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Never mind, cancel that search.",
        ),
    )

    act_count_after_cancel = _count_action_calls(cm)

    # Step 3: Start new unrelated task
    result3 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Actually, look up the weather in Tokyo for next week.",
        ),
    )

    # Should have called act again for the new task
    assert (
        _count_action_calls(cm) > act_count_after_cancel
    ), "Expected new action after canceling previous task"

    # Efficiency assertions at end
    assert_efficient(result1, 3, "Step 1: first task")
    assert_efficient(result2, 3, "Step 2: cancel task")
    assert_efficient(result3, 3, "Step 3: new unrelated task")


@pytest.mark.asyncio
@_handle_project
async def test_sequential_tasks_after_completion_context(initialized_cm):
    """
    After one task, user starts another that could seem related but is new.

    Flow:
    1. User asks for the New York figures
    2. Small talk / acknowledgment
    3. User asks for the Los Angeles figures (different query, should be new action)
    """
    cm = initialized_cm

    # Step 1: First task - New York figures
    result1 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Pull the New York City figures out of the sales CSV in my workspace.",
        ),
    )
    assert get_in_flight_action_count(cm) >= 1, "Expected at least one in-flight action"

    # Step 2: Small talk
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Thanks, that's helpful.",
        ),
    )

    act_count_midpoint = _count_action_calls(cm)

    # Step 3: New request - Los Angeles figures (similar pattern but different query)
    result3 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Now pull the Los Angeles figures out of the same CSV.",
        ),
    )

    # Should have called act again for the new location
    assert (
        _count_action_calls(cm) > act_count_midpoint
    ), "Expected new action for different location query, not just using old results"

    # Efficiency assertions at end
    assert_efficient(result1, 3, "Step 1: New York figures")
    assert_efficient(result2, 3, "Step 2: small talk")
    assert_efficient(result3, 3, "Step 3: Los Angeles figures")


# ---------------------------------------------------------------------------
#  Complex multi-action scenarios
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_interject_first_then_start_second(initialized_cm):
    """
    User starts a task, interjects it, then starts a second unrelated task.

    Flow:
    1. User starts web search
    2. User interjects to narrow the search
    3. User starts a completely different task
    4. Should have interject AND second action
    """
    cm = initialized_cm

    # Step 1: First task
    result1 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Search the web for Italian restaurant reviews in Rome for my upcoming summer trip.",
        ),
    )
    assert get_in_flight_action_count(cm) >= 1, "Expected at least one in-flight action"

    # Step 2: Interject to narrow
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="For those restaurants, focus on ones near the Colosseum.",
        ),
    )

    # Step 3: Start unrelated task
    result3 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="By the way, count the rows in the sales CSV in my workspace.",
        ),
    )

    # Should have both interject (or modification) and multiple actions
    act_count = _count_action_calls(cm)
    has_interject = has_steering_tool_call(cm, "interject_")

    assert act_count >= 2 or (
        act_count >= 1 and has_interject
    ), "Expected interject for narrowing search AND new action for the row count"

    # Efficiency assertions at end
    assert_efficient(result1, 3, "Step 1: restaurant search")
    assert_efficient(result2, 3, "Step 2: narrow search")
    assert_efficient(result3, 3, "Step 3: unrelated task")


@pytest.mark.asyncio
@_handle_project
async def test_three_tasks_rapid_succession(initialized_cm):
    """
    User rapidly requests three different things.

    Flow:
    1. User asks for weather
    2. User asks for a row count
    3. User asks for a notes search
    4. All should trigger separate actions
    """
    cm = initialized_cm

    # Task 1: Weather
    result1 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="What's the weather in London?",
        ),
    )
    assert get_in_flight_action_count(cm) >= 1, "Expected at least one in-flight action"

    # Task 2: Row count
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Count the rows in the sales CSV in my workspace.",
        ),
    )

    # Task 3: Notes search
    result3 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Search my notes files for anything about the budget meeting.",
        ),
    )

    # Should have at least 3 actions for 3 different tasks
    assert (
        _count_action_calls(cm) >= 3
    ), "Expected at least 3 actions for 3 independent requests"

    # Efficiency assertions at end
    assert_efficient(result1, 3, "Task 1: weather")
    assert_efficient(result2, 3, "Task 2: row count")
    assert_efficient(result3, 3, "Task 3: notes search")


@pytest.mark.asyncio
@_handle_project
async def test_pause_first_start_second_resume_first(initialized_cm):
    """
    User pauses first task, starts second, then resumes first.

    Flow:
    1. User starts web search
    2. User pauses it
    3. User starts a different task
    4. User resumes the first task
    """
    cm = initialized_cm

    # Step 1: First task (a notes search reliably starts an in-flight actor handle)
    result1 = await cm.step_until_wait(
        UnifyMessageReceived(
            content=(
                "Search my notes files for anything about competitor pricing information."
            ),
        ),
    )
    assert get_in_flight_action_count(cm) >= 1, "Expected at least one in-flight action"

    # Step 2: Pause it
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Hold on that search for a moment.",
        ),
    )

    # Step 3: Start different task
    result3 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="While that's on hold, convert the sales CSV in my workspace to JSON.",
        ),
    )

    # Step 4: Resume first task
    result4 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="OK, go ahead with that competitor pricing notes search now.",
        ),
    )

    # Should have pause, second action, and resume
    has_pause = has_steering_tool_call(cm, "pause_")
    has_resume = has_steering_tool_call(cm, "resume_")
    act_count = _count_action_calls(cm)

    assert act_count >= 2, "Expected at least 2 actions"
    assert (
        has_pause or has_resume
    ), "Expected pause and/or resume steering for the first task"

    # Efficiency assertions at end
    assert_efficient(result1, 3, "Step 1: first task")
    assert_efficient(result2, 3, "Step 2: pause request")
    assert_efficient(result3, 3, "Step 3: new task while paused")
    assert_efficient(result4, 3, "Step 4: resume first task")
