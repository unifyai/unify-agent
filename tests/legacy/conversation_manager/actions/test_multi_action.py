"""
tests/conversation_manager/test_multi_action.py
====================================================

Tests that verify ConversationManager keeps multiple concurrent actions
apart: each ``act`` starts its own action, and a steer reaches only the
action it names.

These tests build upon test_take_action.py and test_steer_action.py to cover
more complex realistic scenarios involving:
- Multiple concurrent in-flight actions
- New actions next to interjections into existing ones
- Steering one specific action while others run
- Sequential action creation with context

The brain's model is scripted (``scripted_brain``): whether a live model
starts a new action or steers an existing one for a phrasing is a question
about the model. The actor is the SimulatedActor; its ``ask`` and
``interject`` make a model call of their own (``query_llm``), scripted too.
"""

import pytest

from tests.helpers import _handle_project
from tests.legacy.conversation_manager.actions.integration.helpers import (
    inject_actor_result,
)
from tests.legacy.conversation_manager.actions.scripted_brain import (
    act,
    prompt_of,
    say,
    state_of,
    steer,
    turn,
    user_texts,
    wait,
)
from tests.legacy.conversation_manager.cm_helpers import (
    filter_events_by_type,
    assert_efficient,
    get_in_flight_action_count,
    has_steering_tool_call,
)
from tests.scripted_model import reply
from unify.legacy.conversation_manager.events import (
    ActorHandleStarted,
    UnifyMessageReceived,
    UnifyMessageSent,
)
from unify.legacy.handle_state import get_handle_paused_state

# Actions stay in-flight indefinitely with steps=None, duration=None.
# Actions are completed via trigger_completion() in test cleanup.

_ACTION_TOOLS = {"act"}


def _count_action_calls(cm):
    """Count how many action-creating tools were called."""
    return sum(1 for tool in cm.all_tool_calls if tool in _ACTION_TOOLS)


async def _start(cm, brain, request: str, query: str, ack: str, handle_id: int):
    """The user's *request* makes the brain start action *handle_id* running
    *query*; returns the step's result."""
    brain.next(turn(act(query), say(ack), wait()))
    result = await cm.step_until_wait(UnifyMessageReceived(content=request))
    (started,) = filter_events_by_type(result.output_events, ActorHandleStarted)
    assert (started.handle_id, started.query) == (handle_id, query)
    assert cm.cm.in_flight_actions[handle_id]["handle"]._request == query
    return result


def _action(cm, handle_id: int) -> dict:
    data = cm.cm.in_flight_actions.get(handle_id) or cm.cm.completed_actions.get(
        handle_id,
    )
    assert data is not None, f"no action {handle_id}"
    return data


def _handle(cm, handle_id: int):
    return _action(cm, handle_id)["handle"]


def _history(cm, handle_id: int) -> list[str]:
    return [a["action_name"] for a in _action(cm, handle_id)["handle_actions"]]


# ---------------------------------------------------------------------------
#  Sequential independent actions - each should trigger new action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_two_unrelated_requests_create_two_tasks(initialized_cm, scripted_brain):
    """
    Two unrelated requests create two separate actions.

    Flow:
    1. User asks for a web search about one topic
    2. User asks for a completely unrelated workspace task
    3. Both run as separate actions, each with its own query
    """
    cm = initialized_cm
    brain = scripted_brain()
    news = "Search the web for the latest news about climate change."
    convert = "Convert the sales CSV in the user's workspace to JSON."

    # Step 1: First task - web search
    result1 = await _start(
        cm,
        brain,
        "Search the web for the latest news about climate change.",
        news,
        "Searching for climate change news.",
        0,
    )

    # Step 2: Second task - completely unrelated workspace task
    result2 = await _start(
        cm,
        brain,
        "Also, convert the sales CSV in my workspace to JSON.",
        convert,
        "I'll convert the sales CSV too.",
        1,
    )

    # The brain was shown the first action when the second request came in.
    state = brain.last_state()
    assert "<action id='0'" in state and f"<original_request>{news}" in state

    # Two separate actions, on separate handles.
    assert set(cm.cm.in_flight_actions) == {0, 1}
    assert _handle(cm, 0) is not _handle(cm, 1)
    assert (_handle(cm, 0)._request, _handle(cm, 1)._request) == (news, convert)
    assert _history(cm, 0) == _history(cm, 1) == ["act_started"]

    # Efficiency assertions at end
    assert_efficient(result1, 3, "Step 1: first task")
    assert_efficient(result2, 3, "Step 2: second task")


@pytest.mark.asyncio
@_handle_project
async def test_parallel_searches_different_topics(initialized_cm, scripted_brain):
    """
    Two actions on different topics run independently.

    Flow:
    1. User asks to search for topic A
    2. User explicitly requests a NEW separate task B
    3. Both run; B finishing leaves A running
    """
    cm = initialized_cm
    brain = scripted_brain()
    rows = "Count the rows in the sales CSV in the user's workspace."

    # Step 1: First search - web search
    result1 = await _start(
        cm,
        brain,
        "Search the web for information about renewable energy.",
        "Search the web for information about renewable energy.",
        "Searching for renewable energy information.",
        0,
    )
    assert get_in_flight_action_count(cm) >= 1, "Expected at least one in-flight action"

    # Step 2: Explicitly request a NEW task (make it very clear)
    result2 = await _start(
        cm,
        brain,
        "I need a second thing: count the rows in the sales CSV in my workspace.",
        rows,
        "Counting the rows in the sales CSV as well.",
        1,
    )

    assert (
        _count_action_calls(cm) == 2
    ), "Expected two separate actions for explicitly different requests"
    for handle_id in (0, 1):
        handle = _handle(cm, handle_id)
        assert not handle.done() and get_handle_paused_state(handle) is False

    # B finishing does not touch A.
    second = _handle(cm, 1)
    second.trigger_completion()
    assert (await second.result()).startswith(f"Completed '{rows}'")
    assert not _handle(cm, 0).done()
    assert get_handle_paused_state(_handle(cm, 0)) is False

    # Efficiency assertions at end
    assert_efficient(result1, 3, "Step 1: first search")
    assert_efficient(result2, 3, "Step 2: second task")


# ---------------------------------------------------------------------------
#  New action vs interject disambiguation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_also_search_creates_new_task_not_interject(initialized_cm, scripted_brain):
    """
    "Also do X" after task Y, answered with a new action, leaves Y alone.

    Flow:
    1. User asks for a web search
    2. User requests a different type of task (a file conversion)
    3. A new action starts; nothing is interjected into the first one
    """
    cm = initialized_cm
    brain = scripted_brain()

    # Step 1: First task - web search
    result1 = await _start(
        cm,
        brain,
        "Search the web for information about electric vehicles.",
        "Search the web for information about electric vehicles.",
        "Searching for electric vehicle information.",
        0,
    )
    initial_act_count = _count_action_calls(cm)

    # Step 2: Different task type - file conversion (not an interject)
    result2 = await _start(
        cm,
        brain,
        "Also, I need you to convert the sales CSV in my workspace to JSON.",
        "Convert the sales CSV in the user's workspace to JSON.",
        "Sure, converting the sales CSV to JSON.",
        1,
    )

    assert _count_action_calls(cm) == initial_act_count + 1
    assert get_in_flight_action_count(cm) == 2
    # The first action received nothing: no interjection, no question.
    assert not has_steering_tool_call(cm, "interject_")
    assert _handle(cm, 0)._guidance == []
    assert _history(cm, 0) == ["act_started"]
    assert brain.of("query_llm") == []

    # Efficiency assertions at end
    assert_efficient(result1, 3, "Step 1: first task")
    assert_efficient(result2, 3, "Step 2: different task type")


@pytest.mark.asyncio
@_handle_project
async def test_add_detail_to_same_topic_interjects(initialized_cm, scripted_brain):
    """
    A detail for the same topic, interjected, reaches the running action and
    starts no new one.

    Flow:
    1. User asks to search for a topic
    2. User adds a constraint to that same search
    3. The constraint reaches the existing action
    """
    cm = initialized_cm
    detail = "Focus on European regulations specifically."
    brain = scripted_brain(query_llm=[reply("Narrowing to European regulations.")])

    # Step 1: Initial search
    result1 = await _start(
        cm,
        brain,
        "Search the web for AI regulation news.",
        "Search the web for AI regulation news.",
        "Searching for AI regulation news.",
        0,
    )
    act_count_after_first = _count_action_calls(cm)

    # Step 2: Add constraint to the SAME search
    brain.next(
        turn(steer("interject", 0, message=detail), say("Got it, Europe only."), wait()),
    )
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="For that search, focus on European regulations specifically.",
        ),
    )

    assert has_steering_tool_call(cm, "interject_")
    assert _count_action_calls(cm) == act_count_after_first
    assert set(cm.cm.in_flight_actions) == {0}
    # The detail reached the action and the actor's own model.
    assert _handle(cm, 0)._guidance == [detail]
    (interjected,) = brain.of("query_llm")
    assert user_texts(interjected)[-2] == detail
    assert _history(cm, 0) == ["act_started", "interject_0"]

    # Efficiency assertions at end
    assert_efficient(result1, 3, "Step 1: initial search")
    assert_efficient(result2, 3, "Step 2: add constraint")


# ---------------------------------------------------------------------------
#  Multiple actions with selective steering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_two_tasks_stop_one_specifically(initialized_cm, scripted_brain):
    """
    With two tasks running, user stops one by topic.

    Flow:
    1. User starts a web search task
    2. User starts a row-count task
    3. User cancels the web search specifically
    4. Only the web search stops; the row count keeps running
    """
    cm = initialized_cm
    brain = scripted_brain()
    search = "Search the web for the latest stock market news."
    rows = "Count the rows in the sales CSV in the user's workspace."

    # Step 1: First task - web search
    result1 = await _start(
        cm,
        brain,
        "Search the web for the latest stock market news.",
        search,
        "Searching for stock market news.",
        0,
    )

    # Step 2: Second task - row count
    result2 = await _start(
        cm,
        brain,
        "Also count the rows in the sales CSV in my workspace.",
        rows,
        "Counting the sales CSV rows too.",
        1,
    )

    # Step 3: Stop the web search specifically
    turns_before = len(brain.of("brain"))
    brain.next(
        turn(
            steer("stop", 0, reason="The user no longer needs the stock market search."),
            say("Cancelled the stock market search; the row count continues."),
            wait(),
        ),
    )
    result3 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Cancel the stock market search, I don't need that anymore.",
        ),
    )

    assert has_steering_tool_call(
        cm,
        "stop_",
    ), "Expected stop_* steering tool for canceling specific action"
    # The brain was shown both actions by request when the user cancelled one.
    shown = state_of(brain.of("brain")[turns_before])
    assert f"<original_request>{search}</original_request>" in shown
    assert f"<original_request>{rows}</original_request>" in shown
    # Only the named action stopped.
    assert _handle(cm, 0).done() and not _handle(cm, 1).done()
    assert set(cm.cm.in_flight_actions) == {1}
    assert 0 in cm.cm.completed_actions
    assert _history(cm, 1) == ["act_started"]

    # Efficiency assertions at end
    assert_efficient(result1, 3, "Step 1: first task")
    assert_efficient(result2, 3, "Step 2: second task")
    assert_efficient(result3, 3, "Step 3: cancel specific task")


@pytest.mark.asyncio
@_handle_project
async def test_two_tasks_ask_about_one_specifically(initialized_cm, scripted_brain):
    """
    With two tasks running, user asks about one by topic.

    Flow:
    1. User starts a notes search
    2. User starts a web search
    3. User asks about progress on the notes search specifically; only that
       action is asked, and its answer is recorded on it
    """
    cm = initialized_cm
    notes = "Search the user's notes files for anything about the Henderson deal."
    web = "Search the web for Henderson Company's latest quarterly report."
    answer = "Two notes mention the Henderson deal; one has the draft terms."
    brain = scripted_brain(query_llm=[reply(answer)])

    # Step 1: First task - notes search
    result1 = await _start(
        cm,
        brain,
        "Go through the notes files in my workspace for anything about the Henderson deal.",
        notes,
        "Going through your notes for the Henderson deal.",
        0,
    )

    # Step 2: Second task - web search
    result2 = await _start(
        cm,
        brain,
        "Also search the web for Henderson Company's latest quarterly report.",
        web,
        "Also searching for Henderson's quarterly report.",
        1,
    )

    # Step 3: Ask about the notes search specifically
    brain.next(
        turn(steer("ask", 0, question="What have you found in the notes so far?"), wait()),
        turn(say(f"From your notes: {answer}"), wait()),
    )
    result3 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="How's the search through my notes files going?",
        ),
    )

    assert has_steering_tool_call(
        cm,
        "ask_",
    ), "Expected ask_* steering tool for querying specific action"
    # Only the notes search was asked, about its own request.
    (asked,) = brain.of("query_llm")
    prompt = prompt_of(asked)
    assert notes in prompt and web not in prompt
    (ask_event,) = [
        a for a in _action(cm, 0)["handle_actions"] if a["action_name"] == "ask_0"
    ]
    assert (ask_event["status"], ask_event["response"]) == ("completed", answer)
    assert _history(cm, 1) == ["act_started"]
    sent = filter_events_by_type(result3.output_events, UnifyMessageSent)
    assert [e.content for e in sent] == [f"From your notes: {answer}"]

    # Efficiency assertions at end
    assert_efficient(result1, 3, "Step 1: first task")
    assert_efficient(result2, 3, "Step 2: second task")


# ---------------------------------------------------------------------------
#  Action completion then new action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_stop_task_then_start_new_unrelated(initialized_cm, scripted_brain):
    """
    User stops a task, then starts a completely new one.

    Flow:
    1. User starts a task
    2. User cancels it
    3. User starts a new unrelated task
    4. The new task is a fresh action with its own id; the stopped one is
       shown to the brain as completed, not in flight
    """
    cm = initialized_cm
    brain = scripted_brain()

    # Step 1: First task
    result1 = await _start(
        cm,
        brain,
        "Search for information about project deadlines.",
        "Search the workspace for information about project deadlines.",
        "Searching for project deadline information.",
        0,
    )

    # Step 2: Cancel it
    brain.next(
        turn(steer("stop", 0, reason="The user cancelled."), say("Cancelled."), wait()),
    )
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Never mind, cancel that search.",
        ),
    )
    assert _handle(cm, 0).done()
    act_count_after_cancel = _count_action_calls(cm)

    # Step 3: Start new unrelated task
    turns_before = len(brain.of("brain"))
    result3 = await _start(
        cm,
        brain,
        "Actually, look up the weather in Tokyo for next week.",
        "Look up the weather forecast in Tokyo for next week.",
        "Checking next week's Tokyo forecast.",
        1,
    )

    assert (
        _count_action_calls(cm) > act_count_after_cancel
    ), "Expected new action after canceling previous task"
    shown = state_of(brain.of("brain")[turns_before])
    in_flight, _, rest = shown.partition("</in_flight_actions>")
    assert "No actions currently executing." in in_flight
    assert "<completed_actions>" in rest and "<action id='0'" in rest
    assert set(cm.cm.in_flight_actions) == {1}

    # Efficiency assertions at end
    assert_efficient(result1, 3, "Step 1: first task")
    assert_efficient(result2, 3, "Step 2: cancel task")
    assert_efficient(result3, 3, "Step 3: new unrelated task")


@pytest.mark.asyncio
@_handle_project
async def test_sequential_tasks_after_completion_context(initialized_cm, scripted_brain):
    """
    After one task completes, user starts another that could seem related.

    Flow:
    1. User asks for the New York figures; the action completes
    2. Small talk: the brain is shown the New York result
    3. User asks for the Los Angeles figures: a new action, the completed
       one untouched
    """
    cm = initialized_cm
    brain = scripted_brain()
    ny_result = "New York City: 1,240 units sold, 182,000 USD revenue."

    # Step 1: First task - New York figures
    result1 = await _start(
        cm,
        brain,
        "Pull the New York City figures out of the sales CSV in my workspace.",
        "Extract the New York City figures from the sales CSV in the workspace.",
        "Pulling the New York City figures.",
        0,
    )
    first = _handle(cm, 0)
    first.trigger_completion(ny_result)
    await inject_actor_result(cm, handle_id=0, result=await first.result())
    assert 0 in cm.cm.completed_actions and get_in_flight_action_count(cm) == 0

    # Step 2: Small talk
    brain.next(turn(say("You're welcome!"), wait()))
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Thanks, that's helpful.",
        ),
    )
    assert f"<result>{ny_result}</result>" in brain.last_state()

    act_count_midpoint = _count_action_calls(cm)

    # Step 3: New request - Los Angeles figures (similar pattern but different query)
    result3 = await _start(
        cm,
        brain,
        "Now pull the Los Angeles figures out of the same CSV.",
        "Extract the Los Angeles figures from the sales CSV in the workspace.",
        "Pulling the Los Angeles figures.",
        1,
    )

    assert (
        _count_action_calls(cm) > act_count_midpoint
    ), "Expected new action for different location query, not just using old results"
    assert set(cm.cm.in_flight_actions) == {1}
    assert _history(cm, 0) == ["act_started", "act_completed"]

    # Efficiency assertions at end
    assert_efficient(result1, 3, "Step 1: New York figures")
    assert_efficient(result2, 3, "Step 2: small talk")
    assert_efficient(result3, 3, "Step 3: Los Angeles figures")


# ---------------------------------------------------------------------------
#  Complex multi-action scenarios
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_interject_first_then_start_second(initialized_cm, scripted_brain):
    """
    User starts a task, interjects it, then starts a second unrelated task.

    Flow:
    1. User starts web search
    2. User interjects to narrow the search
    3. User starts a completely different task
    4. The interjection is on the first action only; the second is new
    """
    cm = initialized_cm
    narrow = "Only restaurants near the Colosseum."
    brain = scripted_brain(query_llm=[reply("Narrowing to the Colosseum area.")])

    # Step 1: First task
    result1 = await _start(
        cm,
        brain,
        "Search the web for Italian restaurant reviews in Rome for my upcoming summer trip.",
        "Search the web for reviews of Italian restaurants in Rome.",
        "Searching for Rome restaurant reviews.",
        0,
    )

    # Step 2: Interject to narrow
    brain.next(
        turn(steer("interject", 0, message=narrow), say("Will do, near the Colosseum."), wait()),
    )
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="For those restaurants, focus on ones near the Colosseum.",
        ),
    )

    # Step 3: Start unrelated task
    result3 = await _start(
        cm,
        brain,
        "By the way, count the rows in the sales CSV in my workspace.",
        "Count the rows in the sales CSV in the user's workspace.",
        "Counting the sales CSV rows.",
        1,
    )

    act_count = _count_action_calls(cm)
    has_interject = has_steering_tool_call(cm, "interject_")
    assert (
        act_count == 2 and has_interject
    ), "Expected interject for narrowing search AND new action for the row count"
    assert _handle(cm, 0)._guidance == [narrow]
    assert _handle(cm, 1)._guidance == []
    assert _history(cm, 0) == ["act_started", "interject_0"]
    assert _history(cm, 1) == ["act_started"]

    # Efficiency assertions at end
    assert_efficient(result1, 3, "Step 1: restaurant search")
    assert_efficient(result2, 3, "Step 2: narrow search")
    assert_efficient(result3, 3, "Step 3: unrelated task")


@pytest.mark.asyncio
@_handle_project
async def test_three_tasks_rapid_succession(initialized_cm, scripted_brain):
    """
    User rapidly requests three different things.

    Flow:
    1. User asks for weather
    2. User asks for a row count
    3. User asks for a notes search
    4. Three separate actions run; the brain is shown the earlier ones each time
    """
    cm = initialized_cm
    brain = scripted_brain()
    queries = [
        "Look up the current weather in London.",
        "Count the rows in the sales CSV in the user's workspace.",
        "Search the user's notes files for anything about the budget meeting.",
    ]
    requests = [
        "What's the weather in London?",
        "Count the rows in the sales CSV in my workspace.",
        "Search my notes files for anything about the budget meeting.",
    ]

    results = []
    for handle_id, (request, query) in enumerate(zip(requests, queries)):
        results.append(
            await _start(cm, brain, request, query, "On it.", handle_id),
        )
        state = brain.last_state()
        for earlier in range(handle_id):
            assert f"<action id='{earlier}'" in state
            assert f"<original_request>{queries[earlier]}</original_request>" in state

    assert (
        _count_action_calls(cm) == 3
    ), "Expected 3 actions for 3 independent requests"
    assert set(cm.cm.in_flight_actions) == {0, 1, 2}
    assert [_handle(cm, i)._request for i in range(3)] == queries

    # Efficiency assertions at end
    assert_efficient(results[0], 3, "Task 1: weather")
    assert_efficient(results[1], 3, "Task 2: row count")
    assert_efficient(results[2], 3, "Task 3: notes search")


@pytest.mark.asyncio
@_handle_project
async def test_pause_first_start_second_resume_first(initialized_cm, scripted_brain):
    """
    User pauses first task, starts second, then resumes first.

    Flow:
    1. User starts a notes search
    2. User pauses it
    3. User starts a different task: it runs while the first stays paused
    4. User resumes the first task; both run
    """
    cm = initialized_cm
    brain = scripted_brain()

    # Step 1: First task
    result1 = await _start(
        cm,
        brain,
        "Search my notes files for anything about competitor pricing information.",
        "Search the user's notes files for competitor pricing information.",
        "Searching your notes for competitor pricing.",
        0,
    )

    # Step 2: Pause it
    brain.next(turn(steer("pause", 0), say("Paused the notes search."), wait()))
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Hold on that search for a moment.",
        ),
    )

    # Step 3: Start different task
    result3 = await _start(
        cm,
        brain,
        "While that's on hold, convert the sales CSV in my workspace to JSON.",
        "Convert the sales CSV in the user's workspace to JSON.",
        "Converting the sales CSV while the search waits.",
        1,
    )
    assert get_handle_paused_state(_handle(cm, 0)) is True
    assert get_handle_paused_state(_handle(cm, 1)) is False

    # Step 4: Resume first task
    brain.next(turn(steer("resume", 0), say("Resumed the notes search."), wait()))
    result4 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="OK, go ahead with that competitor pricing notes search now.",
        ),
    )

    has_pause = has_steering_tool_call(cm, "pause_")
    has_resume = has_steering_tool_call(cm, "resume_")
    act_count = _count_action_calls(cm)

    assert act_count == 2, "Expected 2 actions"
    assert has_pause and has_resume, "Expected pause and resume steering for the first task"
    assert get_handle_paused_state(_handle(cm, 0)) is False
    assert get_handle_paused_state(_handle(cm, 1)) is False
    assert _history(cm, 0) == ["act_started", "pause_0", "resume_0"]
    assert _history(cm, 1) == ["act_started"]

    # Efficiency assertions at end
    assert_efficient(result1, 3, "Step 1: first task")
    assert_efficient(result2, 3, "Step 2: pause request")
    assert_efficient(result3, 3, "Step 3: new task while paused")
    assert_efficient(result4, 3, "Step 4: resume first task")
