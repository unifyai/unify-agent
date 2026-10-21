"""
tests/conversation_manager/test_managers.py
=================================================

Tests for task-related conversation flows through the ConversationManager.

Uses the ConversationManager "single-step" API:
- Construct an input Event (typically UnifyMessageReceived)
- Call `await cm.step(event)`
- Assert on state changes and output events

The slow brain's model is scripted (``tests/scripted_model.py``): each user
message gets the decision a model makes for it (start an action, steer the
running one, ask for clarification, report status). The tests check what
the ConversationManager does with those decisions: the action started with
its query, the steering reaching the simulated actor's handle, the replies
reaching the chat, and what each turn is shown of the conversation and the
action in flight.
"""

import pytest

from tests.helpers import _handle_project
from tests.legacy.conversation_manager.cm_helpers import filter_events_by_type
from tests.legacy.conversation_manager.scripted_brain import (
    WAIT,
    act,
    brain_requests,
    brain_turn,
    conversation,
    message_text,
    rendered_state,
    say,
    steer,
)
from tests.scripted_model import ScriptedModel, reply, scripted
from unify.legacy.conversation_manager.events import (
    ActorHandleStarted,
    UnifyMessageReceived,
    UnifyMessageSent,
)

CSV_REQUEST = "Go through every CSV in my workspace and list their column names."
CSV_QUERY = "List the column names of every CSV file in the workspace."


def _get_chat_response(events):
    """Get the first outbound chat message, if any."""
    sent = filter_events_by_type(events, UnifyMessageSent)
    return sent[0] if sent else None


def _start_csv_task() -> list:
    """The brain's turn for ``CSV_REQUEST``: start the action and say so."""
    return [
        brain_turn(act(CSV_QUERY), say("On it: I'll list each CSV's columns."), WAIT),
    ]


async def _started_csv_task(cm) -> dict:
    """Step ``CSV_REQUEST`` and return the in-flight action it started."""
    result = await cm.step(UnifyMessageReceived(content=CSV_REQUEST))
    (started,) = filter_events_by_type(result.output_events, ActorHandleStarted)
    assert (started.handle_id, started.query) == (0, CSV_QUERY)
    return cm.cm.in_flight_actions[0]


def _shows_action_in_flight(call) -> bool:
    """Whether a brain request shows action 0 in flight with its query."""
    state = rendered_state(call)
    return (
        "<action id='0'" in state
        and f"<original_request>{CSV_QUERY}</original_request>" in state
    )


# =============================================================================
# LLM Processing Tests
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_task_request_processed(initialized_cm):
    """
    A task request starts an action with the model's query.
    """
    cm = initialized_cm
    model = ScriptedModel(actor=_start_csv_task())
    with scripted(model):
        result = await cm.step(
            UnifyMessageReceived(
                content=CSV_REQUEST,
            ),
        )

    # LLM should have processed the request
    assert result.llm_ran, "Expected LLM to process task request"

    (started,) = filter_events_by_type(result.output_events, ActorHandleStarted)
    assert (started.handle_id, started.action_name) == (0, "act")
    assert started.query == CSV_QUERY
    action = cm.cm.in_flight_actions[0]
    assert action["query"] == CSV_QUERY
    assert not action["handle"].done(), "the action is running"
    assert _get_chat_response(result.output_events).content.startswith("On it")

    (request,) = brain_requests(model)
    assert conversation(request) == [CSV_REQUEST]
    assert "act" in request.tool_names
    model.assert_used_up()


@pytest.mark.asyncio
@_handle_project
async def test_stop_request_processed(initialized_cm):
    """
    A stop request stops the running action.
    """
    cm = initialized_cm
    model = ScriptedModel(
        actor=[
            *_start_csv_task(),
            brain_turn(
                steer("stop_action", 0, reason="The user no longer needs it."),
                say("Stopped. I won't go through the CSVs."),
                WAIT,
            ),
        ],
    )
    with scripted(model):
        # Start a task first
        action = await _started_csv_task(cm)

        # Then send stop request
        stop_result = await cm.step(
            UnifyMessageReceived(
                content="Stop that task, I don't need it anymore",
            ),
        )

    assert stop_result.llm_ran, "Expected LLM to process stop request"

    # The action is stopped and filed as completed.
    assert 0 not in cm.cm.in_flight_actions
    assert cm.cm.completed_actions[0] is action
    handle = action["handle"]
    assert handle.done()
    assert "The user no longer needs it." in await handle.result()
    assert "stop_action" in cm.all_tool_calls

    _, stop_turn = brain_requests(model)
    assert _shows_action_in_flight(stop_turn)
    model.assert_used_up()


@pytest.mark.asyncio
@_handle_project
async def test_status_query_processed(initialized_cm):
    """
    A status query is answered from the action the brain is shown in flight.
    """
    cm = initialized_cm
    status = "Still working through the CSVs; I'll share the columns when done."
    model = ScriptedModel(
        actor=[*_start_csv_task(), brain_turn(say(status), WAIT)],
    )
    with scripted(model):
        # Start a task
        action = await _started_csv_task(cm)

        # Ask about status
        status_result = await cm.step(
            UnifyMessageReceived(
                content="What's the status of that task you're working on?",
            ),
        )

    assert status_result.llm_ran, "Expected LLM to process status query"
    assert _get_chat_response(status_result.output_events).content == status
    assert cm.cm.in_flight_actions[0] is action, "a status query leaves it running"

    _, status_turn = brain_requests(model)
    assert _shows_action_in_flight(status_turn)
    model.assert_used_up()


@pytest.mark.asyncio
@_handle_project
async def test_modification_request_processed(initialized_cm):
    """
    A modification request reaches the running action as an interjection.
    """
    cm = initialized_cm
    guidance = "Skip any CSV file larger than 1 MB."
    model = ScriptedModel(
        actor=[
            *_start_csv_task(),
            brain_turn(
                steer("interject_action", 0, message=guidance),
                say("Will do: I'll skip files over 1 MB."),
                WAIT,
            ),
        ],
        # The simulated actor takes in an interjection with a model call.
        query_llm=[reply("Understood: skipping CSV files over 1 MB.")],
    )
    with scripted(model):
        # Start a task
        action = await _started_csv_task(cm)

        # Modify the task
        modify_result = await cm.step(
            UnifyMessageReceived(
                content="Actually, for that task, please skip any file over 1 MB",
            ),
        )

    assert modify_result.llm_ran, "Expected LLM to process modification request"

    # The guidance reached the running action, which keeps running.
    handle = action["handle"]
    assert handle._guidance == [guidance]
    assert not handle.done()
    assert cm.cm.in_flight_actions[0] is action
    (interjection,) = model.of("query_llm")
    assert any(guidance in message_text(m) for m in interjection.messages)

    _, modify_turn = brain_requests(model)
    assert _shows_action_in_flight(modify_turn)
    model.assert_used_up()


@pytest.mark.asyncio
@_handle_project
async def test_pause_request_processed(initialized_cm):
    """
    A pause request pauses the running action.
    """
    cm = initialized_cm
    model = ScriptedModel(
        actor=[
            *_start_csv_task(),
            brain_turn(steer("pause_action", 0), say("Paused for now."), WAIT),
        ],
    )
    with scripted(model):
        # Start a task
        action = await _started_csv_task(cm)

        # Pause request
        pause_result = await cm.step(
            UnifyMessageReceived(
                content="Pause that task for now",
            ),
        )

    assert pause_result.llm_ran, "Expected LLM to process pause request"
    handle = action["handle"]
    assert handle._paused is True
    assert not handle.done(), "a paused action is still in flight"
    assert cm.cm.in_flight_actions[0] is action

    _, pause_turn = brain_requests(model)
    assert _shows_action_in_flight(pause_turn)
    model.assert_used_up()


@pytest.mark.asyncio
@_handle_project
async def test_resume_request_processed(initialized_cm):
    """
    A resume request resumes the paused action.

    The task is paused first, so the resume has something to resume.
    """
    cm = initialized_cm
    model = ScriptedModel(
        actor=[
            *_start_csv_task(),
            brain_turn(steer("pause_action", 0), say("Paused for now."), WAIT),
            brain_turn(steer("resume_action", 0), say("Resumed."), WAIT),
        ],
    )
    with scripted(model):
        # Start a task, then pause it
        action = await _started_csv_task(cm)
        await cm.step(UnifyMessageReceived(content="Pause that task for now"))
        assert action["handle"]._paused is True

        # Resume request
        resume_result = await cm.step(
            UnifyMessageReceived(
                content="Resume that task please",
            ),
        )

    assert resume_result.llm_ran, "Expected LLM to process resume request"
    handle = action["handle"]
    assert handle._paused is False
    assert not handle.done()
    assert cm.all_tool_calls.count("resume_action") == 1
    model.assert_used_up()


# =============================================================================
# Clarification Tests
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_llm_asks_clarification_for_ambiguous_request(initialized_cm):
    """
    The brain's clarifying question for an ambiguous request reaches the
    chat at once, and nothing is started.
    """
    cm = initialized_cm
    question = "Happy to help. Which file is it, and what should I do with it?"
    model = ScriptedModel(actor=[brain_turn(say(question), WAIT)])
    with scripted(model):
        # Send an ambiguous request
        result = await cm.step(
            UnifyMessageReceived(
                content="I need help with a file",  # Ambiguous: which? what help?
            ),
        )

    # Should produce a chat response asking for clarification
    reply_event = _get_chat_response(result.output_events)
    assert reply_event is not None, "Expected a chat response"
    assert reply_event.content == question
    assert not filter_events_by_type(result.output_events, ActorHandleStarted)
    assert not cm.cm.in_flight_actions

    last = cm.cm.chat_history.recent()[-1]
    assert (last.role, last.content) == ("assistant", question)
    model.assert_used_up()


@pytest.mark.asyncio
@_handle_project
async def test_llm_processes_clarification_response(initialized_cm):
    """
    After asking for clarification, the user's details reach the brain with
    the earlier exchange, and the clarified request starts an action.
    """
    cm = initialized_cm
    question = "Which file is it, and what should I do with it?"
    clarified = "Only the CSV files, and list their row counts."
    query = "Count the rows in each CSV file in the workspace."
    model = ScriptedModel(
        actor=[
            brain_turn(say(question), WAIT),
            brain_turn(act(query), say("On it: counting the rows."), WAIT),
        ],
    )
    with scripted(model):
        # Send an ambiguous request
        await cm.step(
            UnifyMessageReceived(
                content="I need help with a file",
            ),
        )

        # Provide clarification
        result = await cm.step(
            UnifyMessageReceived(
                content=clarified,
            ),
        )

    # LLM should process the clarified request
    assert result.llm_ran, "Expected LLM to process clarification"
    (started,) = filter_events_by_type(result.output_events, ActorHandleStarted)
    assert started.query == query
    assert cm.cm.in_flight_actions[0]["query"] == query

    _, clarified_turn = brain_requests(model)
    assert conversation(clarified_turn) == [
        "I need help with a file",
        question,
        clarified,
    ]
    model.assert_used_up()


# =============================================================================
# Multi-turn Conversation Tests
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_multi_turn_task_conversation(initialized_cm):
    """
    Test a multi-turn conversation involving task management: start an
    action, report on it, then stop it.
    """
    cm = initialized_cm
    query = "List the files in the workspace."
    model = ScriptedModel(
        actor=[
            brain_turn(act(query), say("Listing your files now."), WAIT),
            brain_turn(say("Still listing; I'll send them when done."), WAIT),
            brain_turn(steer("stop_action", 0), say("Stopped."), WAIT),
        ],
    )
    with scripted(model):
        # Turn 1: Request a task
        result1 = await cm.step(
            UnifyMessageReceived(
                content="List the files in my workspace",
            ),
        )
        assert result1.llm_ran, "Expected LLM to run for task request"
        action = cm.cm.in_flight_actions[0]
        assert action["query"] == query

        # Turn 2: Ask about status
        result2 = await cm.step(
            UnifyMessageReceived(
                content="How's that task going?",
            ),
        )
        assert result2.llm_ran, "Expected LLM to run for status query"
        assert cm.cm.in_flight_actions[0] is action

        # Turn 3: Stop the task
        result3 = await cm.step(
            UnifyMessageReceived(
                content="Stop that task please",
            ),
        )
        assert result3.llm_ran, "Expected LLM to run for stop request"

    assert not cm.cm.in_flight_actions
    assert cm.cm.completed_actions[0] is action
    assert action["handle"].done()

    # Each turn is shown the whole conversation so far.
    third = brain_requests(model)[2]
    assert conversation(third) == [
        "List the files in my workspace",
        "Listing your files now.",
        "How's that task going?",
        "Still listing; I'll send them when done.",
        "Stop that task please",
    ]
    model.assert_used_up()
